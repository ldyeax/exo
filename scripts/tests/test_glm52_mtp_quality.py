from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from scripts import glm52_mtp_quality as quality


def _contract(*, mtp_enabled: bool) -> quality.RunContract:
    return quality.RunContract(
        common={"model": "glm52", "tensor_parallel_size": 2},
        artifact={"hybrid_content_id": "a" * 64},
        runtime_controls=quality.MarlinRuntimeControls(
            configured_backend="marlin",
            environment_backend="marlin",
            return_original_logprob_environment="1",
        ),
        mtp=quality.MtpConfiguration(
            enabled=mtp_enabled,
            speculative_algorithm="EAGLE" if mtp_enabled else None,
            speculative_num_steps=1 if mtp_enabled else None,
            speculative_eagle_topk=1 if mtp_enabled else None,
            speculative_num_draft_tokens=2 if mtp_enabled else None,
        ),
    )


def _census(*, mtp_enabled: bool) -> quality.MarlinCensus:
    module_count = 79 if mtp_enabled else 78
    return quality.MarlinCensus(
        mtp_enabled=mtp_enabled,
        requested_backend="marlin",
        expected_runtime_backend="marlin",
        passed=True,
        observations=(
            quality.MarlinRankObservation(
                tensor_parallel_rank=0,
                backend="marlin",
                module_count=module_count,
                local_heads_per_module=32,
            ),
            quality.MarlinRankObservation(
                tensor_parallel_rank=1,
                backend="marlin",
                module_count=module_count,
                local_heads_per_module=32,
            ),
        ),
    )


def _generate(
    request: quality.JsonObject,
    *,
    mtp_enabled: bool = False,
) -> quality.JsonObject:
    input_ids = request["input_ids"]
    assert isinstance(input_ids, list)
    sampling = request["sampling_params"]
    assert isinstance(sampling, dict)
    max_new_tokens = sampling["max_new_tokens"]
    top_k = request.get("top_logprobs_num")
    common_meta: quality.JsonObject = {
        "prompt_tokens": len(input_ids),
        "cached_tokens": 0,
    }
    if request["return_logprob"] is True and top_k == 0:
        common_meta.update(
            {
                "completion_tokens": 1,
                "finish_reason": {"type": "length", "length": 1},
                "input_token_logprobs": [
                    [None, input_ids[0], None],
                    [-0.2, input_ids[1], None],
                    [-0.3, input_ids[2], None],
                ],
                "output_token_logprobs": [[-0.1, 4, None]],
            }
        )
        return {"text": "x", "output_ids": [4], "meta_info": common_meta}
    if request["return_logprob"] is True:
        common_meta.update(
            {
                "completion_tokens": 1,
                "finish_reason": {"type": "length", "length": 1},
                "output_token_logprobs": [[-0.1, 4, None]],
                "output_top_logprobs": [[[-0.1, 4, None], [-0.2, 5, None]]],
            }
        )
        return {"text": "x", "output_ids": [4], "meta_info": common_meta}
    assert max_new_tokens == 2
    common_meta.update(
        {
            "completion_tokens": 2,
            "finish_reason": {"type": "length", "length": 2},
        }
    )
    if mtp_enabled:
        common_meta.update(
            {
                "spec_accept_rate": 1.0,
                "spec_accept_length": 2.0,
                "spec_accept_token_num": 1,
                "spec_draft_token_num": 1,
                "spec_verify_ct": 1,
                "spec_accept_histogram": [0, 1],
            }
        )
    return {"text": "xy", "output_ids": [4, 5], "meta_info": common_meta}


def test_capture_and_compare_matched_marlin_quality_runs() -> None:
    profile = quality.QualityProfile(
        name="smoke",
        representative_prompt_ids=("coding-python-race",),
        humaneval_ids=(),
        generation_max_new_tokens=2,
        top_k=2,
    )
    cases = (
        quality.TokenizedQualityCase(
            case_id="coding-python-race",
            category="coding",
            input_ids=(1, 2, 3),
        ),
    )
    mtp_off = quality.capture_quality_run(
        _generate,
        cases,
        run_id="off",
        contract=_contract(mtp_enabled=False),
        backend_census=_census(mtp_enabled=False),
        profile=profile,
    )
    mtp_on = quality.capture_quality_run(
        lambda request: _generate(request, mtp_enabled=True),
        cases,
        run_id="on",
        contract=_contract(mtp_enabled=True),
        backend_census=_census(mtp_enabled=True),
        profile=profile,
    )

    comparison = quality.compare_matched_runs(mtp_off, mtp_on)
    assert comparison.status == "passed"
    assert comparison.metrics.teacher_forced_token_count == 2
    assert mtp_off.speculative_decoding_evidence.mtp_enabled is False
    assert mtp_off.speculative_decoding_evidence.draft_token_count == 0
    assert mtp_on.speculative_decoding_evidence == (
        quality.SpeculativeDecodingExerciseEvidence(
            source="native_non_stream_generate_meta_info",
            mtp_enabled=True,
            generation_request_count=1,
            generation_requests_with_metrics=1,
            completion_token_count=2,
            accepted_draft_token_count=1,
            draft_token_count=1,
            rejected_draft_token_count=0,
            verification_pass_count=1,
            acceptance_rate=1.0,
            average_tokens_per_verification=2.0,
        )
    )
    mtp_on_receipt = asdict(mtp_on)
    assert mtp_on_receipt["speculative_decoding_evidence"] == {
        "source": "native_non_stream_generate_meta_info",
        "mtp_enabled": True,
        "generation_request_count": 1,
        "generation_requests_with_metrics": 1,
        "completion_token_count": 2,
        "accepted_draft_token_count": 1,
        "draft_token_count": 1,
        "rejected_draft_token_count": 0,
        "verification_pass_count": 1,
        "acceptance_rate": 1.0,
        "average_tokens_per_verification": 2.0,
    }
    assert quality.HUMANEVAL_REFERENCE_CONTENT_SHA256 == (
        "9a45e598c1ba1a38e0ec969769438b406b477be1f1c37cc958d9c852bb668b70"
    )

    mismatched_generation = replace(
        mtp_on.generations[0],
        output_ids=(4, 6),
    )
    mismatched_run = replace(mtp_on, generations=(mismatched_generation,))
    with pytest.raises(
        quality.Glm52MtpQualityError,
        match="greedy token parity failed",
    ):
        quality.compare_matched_runs(mtp_off, mismatched_run)


def test_teacher_forced_wire_schema_requires_sglang_null_sentinel() -> None:
    request = quality.TeacherForcedRequest(
        case_id="coding-python-race",
        input_ids=(1, 2, 3),
        score_start_index=1,
    )
    assert request.json_object()["logprob_start_len"] == 0

    malformed = _generate(request.json_object())
    meta_info = malformed["meta_info"]
    assert isinstance(meta_info, dict)
    meta_info["input_token_logprobs"] = [["not-a-logprob", 1, None]]
    with pytest.raises(quality.Glm52MtpQualityError) as raised:
        quality.parse_teacher_forced_response(request, malformed)
    message = str(raised.value)
    assert "teacher-forced response is invalid:" in message
    assert '"error_count":1' in message
    assert "meta_info" in message
    assert len(message.encode()) <= quality.MAXIMUM_VALIDATION_ERROR_BYTES


def test_mtp_on_quality_run_requires_complete_speculative_generation_metrics() -> None:
    profile = quality.QualityProfile(
        name="smoke",
        representative_prompt_ids=("coding-python-race",),
        humaneval_ids=(),
        generation_max_new_tokens=2,
        top_k=2,
    )
    cases = (
        quality.TokenizedQualityCase(
            case_id="coding-python-race",
            category="coding",
            input_ids=(1, 2, 3),
        ),
    )

    with pytest.raises(
        ValueError,
        match="lacks speculative metrics for every generation request",
    ):
        quality.capture_quality_run(
            _generate,
            cases,
            run_id="on-without-evidence",
            contract=_contract(mtp_enabled=True),
            backend_census=_census(mtp_enabled=True),
            profile=profile,
        )


def test_generation_rejects_internally_inconsistent_speculative_metrics() -> None:
    request = quality.DeterministicGenerationRequest(
        case_id="coding-python-race",
        input_ids=(1, 2, 3),
        max_new_tokens=2,
    )
    payload = _generate(request.json_object(), mtp_enabled=True)
    meta_info = payload["meta_info"]
    assert isinstance(meta_info, dict)
    meta_info["spec_accept_token_num"] = 0

    with pytest.raises(
        quality.Glm52MtpQualityError,
        match="speculative metrics are inconsistent",
    ):
        quality.parse_generation_response(request, payload)
