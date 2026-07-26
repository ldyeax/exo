from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

ATTEMPT_DIRECTORY = Path(
    "/var/lib/exo/benchmarks/"
    "glm52-fwuff-remote-eagle-k4-batched-prefill-20260726/"
    "attempt-02-boundary-decode-240x128"
)
STAGING_PATH = Path("/tmp/fwuff-component-comparison.json")
SERIAL_PATH = Path(
    "/var/lib/exo/benchmarks/"
    "glm52-fwuff-remote-eagle-k4-paired-smoke-20260726/"
    "attempt-06-native-page-bookkeeping/client-240x128.json"
)
PURE_BATCH_PATH = Path(
    "/var/lib/exo/benchmarks/"
    "glm52-fwuff-remote-eagle-k4-batched-prefill-20260726/"
    "attempt-01-240x128/client-240x128.json"
)
TARGET_PATH = ATTEMPT_DIRECTORY / "client-240x128.json"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified(path: Path, expected_file_sha256: str) -> dict[str, object]:
    assert file_sha256(path) == expected_file_sha256
    value = json.loads(path.read_text())
    recorded = value["receipt_content_sha256"]
    unsigned = dict(value)
    del unsigned["receipt_content_sha256"]
    assert canonical_sha256(unsigned) == recorded
    return value


def metrics(receipt: dict[str, object]) -> dict[str, object]:
    observation = receipt["observation"]
    assert isinstance(observation, dict)
    request = observation["requests"][0]["observation"]
    speculative = receipt["speculative_metrics"]["requests"][0]["metrics"]
    return {
        "aggregate_output_tokens_per_second": observation[
            "aggregate_output_tokens_per_second"
        ],
        "case_wall_seconds": observation["case_wall_seconds"],
        "end_to_end_output_tokens_per_second": request[
            "end_to_end_output_tokens_per_second"
        ],
        "end_to_end_seconds": request["end_to_end_seconds"],
        "generation_tokens_per_second": request["generation_tokens_per_second"],
        "generation_window_seconds": request["generation_window_seconds"],
        "spec_accept_histogram": speculative["spec_accept_histogram"],
        "spec_accept_length": speculative["spec_accept_length"],
        "spec_accept_rate": speculative["spec_accept_rate"],
        "spec_accept_token_num": speculative["spec_accept_token_num"],
        "spec_draft_token_num": speculative["spec_draft_token_num"],
        "spec_verify_ct": speculative["spec_verify_ct"],
        "stream_event_count": request["stream_event_count"],
        "ttft_seconds": request["ttft_seconds"],
    }


getcontext().prec = 50


def scalar_delta(candidate: int | float, reference: int | float) -> dict[str, object]:
    candidate_decimal = Decimal(str(candidate))
    reference_decimal = Decimal(str(reference))
    delta = candidate_decimal - reference_decimal
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_minus_reference_decimal": format(delta, "f"),
        "relative_percent_decimal_15_places": format(
            delta * Decimal(100) / reference_decimal,
            ".15f",
        ),
    }


def comparison(
    candidate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in candidate:
        if key == "spec_accept_histogram":
            candidate_histogram = candidate[key]
            reference_histogram = reference[key]
            assert isinstance(candidate_histogram, list)
            assert isinstance(reference_histogram, list)
            result[key] = {
                "candidate": candidate_histogram,
                "reference": reference_histogram,
                "candidate_minus_reference": [
                    left - right
                    for left, right in zip(
                        candidate_histogram, reference_histogram, strict=True
                    )
                ],
            }
            continue
        candidate_value = candidate[key]
        reference_value = reference[key]
        assert isinstance(candidate_value, (int, float))
        assert isinstance(reference_value, (int, float))
        result[key] = scalar_delta(candidate_value, reference_value)
    return result


serial = load_verified(
    SERIAL_PATH,
    "8982a2e7aa4562b6b53a73d3df18092c78f2c8b6eb46c55f847389ca176568bd",
)
pure_batch = load_verified(
    PURE_BATCH_PATH,
    "7a91f0fe6fc3ac234ab61db14504655a9b381c7cd3831e86a42c9bf1cde4f287",
)
target = load_verified(
    TARGET_PATH,
    "0bbb22d2bf3a285480021bc8e7a19262565ce4cdc93b400f1e0d5b7355dcd79e",
)
candidate_metrics = metrics(target)
serial_metrics = metrics(serial)
pure_batch_metrics = metrics(pure_batch)

candidate_ids = target["observation"]["requests"][0]["observation"]["output_ids"]
serial_ids = serial["observation"]["requests"][0]["observation"]["output_ids"]
pure_batch_ids = pure_batch["observation"]["requests"][0]["observation"]["output_ids"]


def common_prefix(left: list[int], right: list[int]) -> int:
    return next(
        (
            index
            for index, (left_token, right_token) in enumerate(
                zip(left, right, strict=True)
            )
            if left_token != right_token
        ),
        len(left),
    )


serial_prefix = common_prefix(candidate_ids, serial_ids)
pure_batch_prefix = common_prefix(candidate_ids, pure_batch_ids)
assert serial_prefix == pure_batch_prefix == 27

prompt_timings = [
    {
        "committed_forward_count": 2,
        "committed_rows": 128,
        "committed_total_milliseconds": 53.219576002447866,
        "proposal_ids": [99752, 18, 25, 2411],
        "response_timing_total_milliseconds": 68.27299202268478,
        "round_id": 0,
        "sequence_length": 128,
        "tentative_forward_count": 3,
        "tentative_total_milliseconds": 15.05341602023691,
        "timings": [
            {
                "execution_role": "batched_prefix_extend",
                "h2d_milliseconds": 0.2906560003757477,
                "model_milliseconds": 46.70451354980469,
                "row_count": 127,
                "sequence_length": 127,
                "total_milliseconds": 47.69663199840579,
            },
            {
                "execution_role": "boundary_decode",
                "h2d_milliseconds": 0.061055999249219894,
                "model_milliseconds": 4.625696182250977,
                "row_count": 1,
                "sequence_length": 128,
                "total_milliseconds": 5.522944004042074,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.015104000456631184,
                "model_milliseconds": 4.281760215759277,
                "row_count": 1,
                "sequence_length": 129,
                "total_milliseconds": 5.029420004575513,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.015104000456631184,
                "model_milliseconds": 4.163519859313965,
                "row_count": 1,
                "sequence_length": 130,
                "total_milliseconds": 4.909495008178055,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.014816000126302242,
                "model_milliseconds": 4.173312187194824,
                "row_count": 1,
                "sequence_length": 131,
                "total_milliseconds": 5.114501007483341,
            },
        ],
    },
    {
        "committed_forward_count": 2,
        "committed_rows": 112,
        "committed_total_milliseconds": 34.159112998167984,
        "proposal_ids": [44982, 5675, 1917, 34095],
        "response_timing_total_milliseconds": 48.13193899462931,
        "round_id": 1,
        "sequence_length": 240,
        "tentative_forward_count": 3,
        "tentative_total_milliseconds": 13.972825996461324,
        "timings": [
            {
                "execution_role": "batched_prefix_extend",
                "h2d_milliseconds": 0.4687359929084778,
                "model_milliseconds": 27.942079544067383,
                "row_count": 111,
                "sequence_length": 239,
                "total_milliseconds": 29.21824299846776,
            },
            {
                "execution_role": "boundary_decode",
                "h2d_milliseconds": 0.04364800080657005,
                "model_milliseconds": 4.171487808227539,
                "row_count": 1,
                "sequence_length": 240,
                "total_milliseconds": 4.940869999700226,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.016063999384641647,
                "model_milliseconds": 4.003456115722656,
                "row_count": 1,
                "sequence_length": 241,
                "total_milliseconds": 4.687501001171768,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.014368000440299511,
                "model_milliseconds": 3.955519914627075,
                "row_count": 1,
                "sequence_length": 242,
                "total_milliseconds": 4.632689000573009,
            },
            {
                "execution_role": "tentative_decode",
                "h2d_milliseconds": 0.015231999568641186,
                "model_milliseconds": 3.966559886932373,
                "row_count": 1,
                "sequence_length": 243,
                "total_milliseconds": 4.652635994716547,
            },
        ],
    },
]

receipt: dict[str, object] = {
    "schema_version": 1,
    "kind": "glm52_fwuff_boundary_prefill_component_comparison",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "request_id": "d859441a8b8948f2bcc41c3c7677f6a1",
    "evidence": {
        "target_receipt": {
            "path": str(TARGET_PATH),
            "file_sha256": file_sha256(TARGET_PATH),
            "receipt_content_sha256": target["receipt_content_sha256"],
        },
        "target_rank_log": {
            "path": str(ATTEMPT_DIRECTORY / "rank-0.log"),
            "file_sha256": file_sha256(ATTEMPT_DIRECTORY / "rank-0.log"),
        },
        "fwuff_server_log": {
            "host": "fwuff",
            "path": (
                "/var/lib/exo/benchmarks/"
                "glm52-fwuff-isolated-draft-batched-20260726T015022Z/"
                "attempt-03-prefill-boundary-decode/server.log"
            ),
            "request_line_first": 138,
            "request_line_last": 170,
            "request_slice_sha256": (
                "af6a78e633d35c5c17ba1b30dd9a2b85bfd545e93b61ae1c8a7812cd4062e3c2"
            ),
        },
    },
    "runtime_identity": {
        "endpoint": target["endpoint"],
        "fwuff_service_pid": target["fwuff_service_pid"],
        "fwuff_service_source": {
            "path": str(ATTEMPT_DIRECTORY / "glm52_fwuff_remote_draft.py"),
            "sha256": target["fwuff_service_source_sha256"],
        },
        "target_parent_pid": target["target_parent_pid"],
        "target_bridge_source": {
            "path": str(ATTEMPT_DIRECTORY / "glm52_remote_eagle_target.py"),
            "sha256": target["target_bridge_sha256"],
        },
    },
    "fwuff_lifecycle": {
        "all_statuses_ok": True,
        "response_count": 33,
        "round_ids_contiguous": [0, 32],
        "prompt": {
            "chunks": prompt_timings,
            "committed_forward_count": 4,
            "committed_rows": 240,
            "committed_total_milliseconds": 87.37868900061585,
            "tentative_forward_count": 6,
            "tentative_total_milliseconds": 29.026242016698234,
            "response_timing_total_milliseconds": 116.40493101731409,
        },
        "decode": {
            "response_count": 30,
            "round_ids": [2, 31],
            "committed_forward_count": 125,
            "committed_forward_count_histogram": {
                "1": 3,
                "2": 1,
                "3": 3,
                "4": 4,
                "5": 19,
            },
            "tentative_forward_count": 90,
            "timing_record_count": 215,
            "timing_total_milliseconds": 970.8529981144238,
            "first_sequence_length": 244,
            "last_sequence_length": 365,
        },
        "finish": {
            "round_id": 32,
            "sequence_length": 365,
            "status": "ok",
            "proposal_ids": [],
            "committed_forward_count": 0,
            "tentative_forward_count": 0,
            "timing_record_count": 0,
        },
    },
    "candidate_metrics": candidate_metrics,
    "comparisons": {
        "delta_convention": (
            "candidate minus reference; relative percent uses the reference "
            "as denominator"
        ),
        "serial_attempt06": {
            "receipt": {
                "path": str(SERIAL_PATH),
                "file_sha256": file_sha256(SERIAL_PATH),
                "receipt_content_sha256": serial["receipt_content_sha256"],
            },
            "reference_metrics": serial_metrics,
            "deltas": comparison(candidate_metrics, serial_metrics),
        },
        "pure_batch_attempt01": {
            "receipt": {
                "path": str(PURE_BATCH_PATH),
                "file_sha256": file_sha256(PURE_BATCH_PATH),
                "receipt_content_sha256": pure_batch["receipt_content_sha256"],
            },
            "reference_metrics": pure_batch_metrics,
            "deltas": comparison(candidate_metrics, pure_batch_metrics),
        },
    },
    "output_comparison": {
        "candidate_output_ids_sha256": target["observation"]["requests"][0][
            "observation"
        ]["output_ids_sha256"],
        "first_16_exact_to_serial_quality_gate": target["quality_gate"][
            "first_16_exact"
        ],
        "full_output_exact_to_serial": target["quality_gate"]["full_output_exact"],
        "common_prefix_token_count_vs_serial_attempt06": serial_prefix,
        "common_prefix_token_count_vs_pure_batch_attempt01": pure_batch_prefix,
        "candidate_first_difference_ids": candidate_ids[
            serial_prefix : serial_prefix + 4
        ],
        "serial_first_difference_ids": serial_ids[serial_prefix : serial_prefix + 4],
        "pure_batch_first_difference_ids": pure_batch_ids[
            pure_batch_prefix : pure_batch_prefix + 4
        ],
        "semantic_note": (
            "All three outputs share a 27-token prefix containing the complete "
            "requested answer: the operator retained the resident model, "
            "checked request ordering, observed pipeline coherence, and did "
            "not restart the model. The boundary run diverges only in the "
            "post-answer continuation, where it emits relevant timeline "
            "records; the full 128-token output is therefore semantically "
            "correct but not bit-exact to attempt06."
        ),
    },
}
assert receipt["evidence"]["target_rank_log"]["file_sha256"] == (
    "b2742a6626d93df096e8e0aedbc2c5f1f434dc2f2888c140eeb9e79e6df09933"
)
receipt["receipt_content_sha256"] = canonical_sha256(receipt)
STAGING_PATH.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(STAGING_PATH)
print(file_sha256(STAGING_PATH))
print(receipt["receipt_content_sha256"])
