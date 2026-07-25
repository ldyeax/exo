from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from scripts import glm52_routing_profiles as profiles


def _valid_counts(*, routed_count: int = 1) -> list[list[int]]:
    rows = [
        [0 for _ in range(profiles.EXPERT_COUNT)]
        for _ in range(profiles.MODEL_LAYER_COUNT)
    ]
    for layer in range(
        profiles.FIRST_ROUTED_LAYER,
        profiles.MODEL_LAYER_COUNT,
    ):
        for expert in range(profiles.EXPERTS_PER_TOKEN):
            rows[layer][expert] = routed_count
    return rows


def _inventory() -> profiles.ExpertTensorInventory:
    return profiles.ExpertTensorInventory(
        model_config_sha256="1" * 64,
        safetensors_index_sha256="2" * 64,
        tensor_inventory_sha256="3" * 64,
        hidden_size=6_144,
        moe_intermediate_size=2_048,
        dtype="BF16",
        bytes_per_element=2,
        full_expert_bytes=72 * 1024**2,
        per_tp_rank_expert_bytes=36 * 1024**2,
        routed_layer_count=75,
        tensor_count=75 * 256 * 3,
    )


def _baseline(tmp_path: Path) -> profiles.BaselineCapacity:
    return profiles.BaselineCapacity(
        receipt_path=str(tmp_path / "baseline.json"),
        receipt_sha256="4" * 64,
        minimum_observed_free_vram_mib=597,
        minimum_required_free_vram_mib=512,
        token_capacity=16_000,
        kvcache_gib_per_rank=1.5,
        conservative_kvcache_bytes_per_token_per_rank=100_327,
    )


def _write_placement_receipt(
    tmp_path: Path,
) -> tuple[Path, str, Path, str]:
    artifact = tmp_path / "frequency.pt"
    artifact.write_bytes(b"validated-placement")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "materialized",
        "kind": "glm52_frequency_placement_input_v1",
        "artifact": {
            "path": str(artifact),
            "sha256": artifact_sha256,
            "logical_count_shape": [1, 78, 256],
        },
        "model_contract": {
            "layer_count": 78,
            "expert_count": 256,
            "first_routed_layer": 3,
            "last_routed_layer": 77,
        },
    }
    receipt_path = tmp_path / "frequency.json"
    receipt_path.write_text(json.dumps(receipt))
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return artifact, artifact_sha256, receipt_path, receipt_sha256


def test_prompt_manifest_is_fixed_coding_and_agent_corpus() -> None:
    first = profiles.representative_prompt_manifest()
    second = profiles.representative_prompt_manifest()

    assert first == second
    assert first["prompt_count"] == 8
    prompt_rows = cast(list[dict[str, object]], first["prompts"])
    assert {row["category"] for row in prompt_rows} == {"coding", "agent"}
    assert cast(dict[str, object], first["sampling"])["max_new_tokens"] == 16
    manifest_hash = cast(str, first.pop("manifest_content_sha256"))
    assert manifest_hash == profiles._canonical_sha256(first)


def test_routing_matrix_requires_exact_glm52_shape_and_layers() -> None:
    counts = _valid_counts(routed_count=3)
    validated = profiles._validate_count_matrix(counts)
    assert len(validated) == 78
    assert sum(validated[3]) == 24
    assert sum(validated[77]) == 24

    counts[0][0] = 1
    with pytest.raises(
        profiles.Glm52RoutingProfileError,
        match="dense layer 0",
    ):
        profiles._validate_count_matrix(counts)

    counts = _valid_counts()
    counts[42][0] += 8
    with pytest.raises(
        profiles.Glm52RoutingProfileError,
        match="same number",
    ):
        profiles._validate_count_matrix(counts)

    with pytest.raises(
        profiles.Glm52RoutingProfileError,
        match="77 layers",
    ):
        profiles._validate_count_matrix(_valid_counts()[:-1])


def test_frequency_scores_preserve_counts_and_break_ties() -> None:
    counts = _valid_counts()
    counts[3][7] = 4
    counts[3][0] = 0
    counts[3][1] = 0
    counts[3][2] = 0
    # Preserve the per-layer total required by an exact route capture.
    scores = profiles._deterministic_placement_scores(counts)
    ranked = profiles._ranked_routed_experts(counts)

    assert scores[0] == tuple(0 for _ in range(256))
    assert scores[3][7] > scores[3][3]
    # Equal counts use ascending layer and expert, independent of torch.topk
    # tie behavior in a future runtime.
    assert scores[3][3] > scores[3][4] > scores[4][3]
    assert ranked[0] == (3, 7, 4)


def test_exact_ratio_encodes_requested_global_budget() -> None:
    for budget in (1, 14, 75, profiles.TOTAL_ROUTED_EXPERT_POSITIONS):
        ratio = float(profiles._ratio_for_exact_total_budget(budget))
        assert int(ratio * profiles.TOTAL_ROUTED_EXPERT_POSITIONS) == budget


def test_budget_rejects_16k_and_admits_sparse_short_lane(
    tmp_path: Path,
) -> None:
    baseline = _baseline(tmp_path)
    inventory = _inventory()
    canonical = profiles.calculate_resident_budget(
        baseline,
        inventory,
        target_token_capacity=16_000,
        allocator_guard_mib=256,
    )
    short = profiles.calculate_resident_budget(
        baseline,
        inventory,
        target_token_capacity=8_448,
        allocator_guard_mib=256,
    )

    assert canonical.maximum_safe_total_resident_experts == 0
    assert canonical.rejection_reason is not None
    assert short.maximum_safe_total_resident_experts == 15
    assert short.minimum_per_layer_profile_admitted is False
    assert short.allocatable_resident_bytes_per_rank >= 15 * 36 * 1024**2
    assert short.allocatable_resident_bytes_per_rank < 16 * 36 * 1024**2


def test_plan_emits_matched_chunk_placement_profiles_and_rejections(
    tmp_path: Path,
) -> None:
    artifact, artifact_hash, receipt, receipt_hash = _write_placement_receipt(
        tmp_path
    )
    plan = profiles.build_profile_plan(
        baseline=_baseline(tmp_path),
        inventory=_inventory(),
        placement_artifact=artifact,
        placement_sha256=artifact_hash,
        placement_receipt_path=receipt,
        placement_receipt_sha256=receipt_hash,
        result_root=tmp_path / "results",
        allocator_guard_mib=256,
    )

    emitted = cast(list[dict[str, object]], plan["profiles"])
    assert len(emitted) == 12
    assert {item["chunked_prefill_size"] for item in emitted} == {
        2_048,
        4_096,
        8_192,
    }
    matched = [
        item for item in emitted if item["lane"] == "matched_short_resident_ab"
    ]
    assert len(matched) == 9
    assert {item["placement_strategy"] for item in matched} == {
        "uniform",
        "frequency",
    }
    assert {item["total_resident_gpu_experts"] for item in matched} == {0, 15}
    assert {
        item["resident_bytes_per_tp_rank"]
        for item in matched
        if item["total_resident_gpu_experts"] == 15
    } == {15 * 36 * 1024**2}
    for item in matched:
        arguments = cast(list[object], item["launch_arguments"])
        assert "--kt-max-deferred-experts-per-token" not in arguments
        if item["total_resident_gpu_experts"] == 15:
            ratio_index = arguments.index("--kt-gpu-experts-ratio")
            assert (
                int(
                    float(cast(str, arguments[ratio_index + 1]))
                    * profiles.TOTAL_ROUTED_EXPERT_POSITIONS
                )
                == 15
            )
    rejected = cast(list[dict[str, object]], plan["rejected_profiles"])
    assert len(rejected) == 2
    assert all(item["admitted"] is False for item in rejected)
    assert all(item["maximum_total_tokens"] == 16_000 for item in rejected)


def test_plan_rejects_frequency_receipt_that_does_not_bind_artifact(
    tmp_path: Path,
) -> None:
    artifact, artifact_hash, receipt, _ = _write_placement_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["artifact"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(
        profiles.Glm52RoutingProfileError,
        match="does not bind",
    ):
        profiles.build_profile_plan(
            baseline=_baseline(tmp_path),
            inventory=_inventory(),
            placement_artifact=artifact,
            placement_sha256=artifact_hash,
            placement_receipt_path=receipt,
            placement_receipt_sha256=receipt_hash,
            result_root=tmp_path / "results",
            allocator_guard_mib=256,
        )


def test_baseline_reader_requires_exact_passed_zero_resident_contract(
    tmp_path: Path,
) -> None:
    receipt = {
        "status": "passed",
        "topology": {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 2,
            "parent_process_count": 1,
        },
        "process_spec": {
            "experts": {
                "resident_gpu_experts": 0,
                "max_deferred_experts_per_token": 0,
            },
        },
        "capacity_and_vram": {
            "postreadiness_snapshot": {
                "devices": [{"free_mib": 597}, {"free_mib": 683}]
            },
            "postreadiness_gate": {"minimum_free_vram_mib": 512},
        },
        "server_info": {
            "max_total_num_tokens": 16_000,
            "internal_states": [
                {
                    "memory_usage": {
                        "kvcache": 1.5,
                        "token_capacity": 16_000,
                    }
                }
            ],
        },
    }
    receipt_path = tmp_path / "baseline.json"
    receipt_path.write_text(json.dumps(receipt))
    capacity = profiles.read_baseline_capacity(
        receipt_path,
        expected_vram_floor_mib=512,
    )
    assert capacity.minimum_observed_free_vram_mib == 597
    assert capacity.conservative_kvcache_bytes_per_token_per_rank == 100_327

    receipt["status"] = "failed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(
        profiles.Glm52RoutingProfileError,
        match="not passed",
    ):
        profiles.read_baseline_capacity(
            receipt_path,
            expected_vram_floor_mib=512,
        )


def test_capture_server_contract_requires_unbounded_stat_recorder() -> None:
    valid = {
        "pp_size": 1,
        "tp_size": 2,
        "kt_method": "AMXINT4",
        "kt_max_deferred_experts_per_token": 0,
        "kt_enable_dynamic_expert_update": False,
        "expert_distribution_recorder_mode": "stat",
        "expert_distribution_recorder_buffer_size": -1,
    }
    assert profiles._validate_capture_server_info(valid) == valid

    for field, changed in (
        ("expert_distribution_recorder_mode", None),
        ("expert_distribution_recorder_buffer_size", 1_000),
        ("kt_max_deferred_experts_per_token", 1),
        ("kt_enable_dynamic_expert_update", True),
    ):
        invalid = {**valid, field: changed}
        with pytest.raises(
            profiles.Glm52RoutingProfileError,
            match=field,
        ):
            profiles._validate_capture_server_info(invalid)


def test_server_url_rejects_credentials_paths_and_non_http() -> None:
    assert (
        profiles._validate_server_url("http://192.168.40.24:62710/")
        == "http://192.168.40.24:62710"
    )
    for value in (
        "ssh://dwagon:62710",
        "http://user@dwagon:62710",
        "http://dwagon:62710/generate",
        "http://dwagon:62710?x=1",
    ):
        with pytest.raises(profiles.Glm52RoutingProfileError):
            profiles._validate_server_url(value)
