from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts import benchmark_dsv4_flash_hotspot as hotspot


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class FramingEncoder:
    def __call__(self, messages: list[dict[str, str]], *, thinking_mode: str) -> str:
        assert thinking_mode == "chat"
        return "<user>" + messages[0]["content"] + "<assistant>"


def test_http_only_disables_trace_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_dsv4_flash_hotspot", "--http-only"],
    )

    arguments = hotspot.parse_args()

    assert arguments.require_trace is False


def test_repeat_diagnostic_cli_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_dsv4_flash_hotspot", "--deterministic-repeat-count", "6"],
    )

    with pytest.raises(SystemExit):
        hotspot.parse_args()


def _diagnostic_arguments(tmp_path: Path) -> hotspot.HotspotArguments:
    return hotspot.HotspotArguments(
        generate_url="http://127.0.0.1:30010/generate",
        server_info_url="http://127.0.0.1:30010/server_info",
        flush_url="http://127.0.0.1:30010/flush_cache?timeout=30",
        control_url="http://127.0.0.1:30010/set_internal_state",
        hotspot_url="http://127.0.0.1:30010/kt_expert_hotspot",
        model_path=tmp_path,
        input_tokens=2_694,
        output_tokens=512,
        timeout_seconds=30.0,
        flush_timeout_seconds=10.0,
        progress_every=0,
        ignore_eos=False,
        near_prefix_ratio=0.70,
        verify_policy="4",
        hotspot_plan=None,
        hotspot_generation=None,
        hotspot_commit=False,
        expert_recorder_directory=None,
        expected_expert_plan=None,
        require_nvlink_traffic=False,
        require_trace=False,
        output_file=None,
        require_oscar_split_history=False,
        deterministic_repeat_count=3,
        deterministic_repeat_output_tokens=64,
    )


def test_fixed_length_repeats_flush_every_run_and_report_token_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flushes: list[str] = []
    requests: list[dict[str, object]] = []

    def fake_flush(url: str, timeout_seconds: float) -> dict[str, object]:
        assert timeout_seconds == 10.0
        flushes.append(url)
        return {"flushed": True}

    def fake_benchmark(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        requests.append(dict(kwargs))
        index = len(requests) - 1
        return {
            "diagnostic_only": True,
            "performance_claim_eligible": False,
            "ignore_eos": True,
            "exact_requested_token_shape": True,
            "completion_tokens": 64,
            "output_sha256": "a" * 64,
            "output_token_ids_sha256": ("b" if index < 2 else "c") * 64,
        }

    monkeypatch.setattr(hotspot, "flush_cache", fake_flush)
    monkeypatch.setattr(hotspot.baseline, "run_benchmark", fake_benchmark)

    diagnostic = hotspot.run_deterministic_repeat_diagnostic(
        _diagnostic_arguments(tmp_path),
        input_ids=[1, 2, 3],
        decode_output_ids=lambda _ids: "redacted",
        terminal_token_ids=(0,),
    )

    assert len(flushes) == 3
    assert all(request["ignore_eos"] is True for request in requests)
    assert all(request["diagnostic_only"] is True for request in requests)
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["performance_claim_eligible"] is False
    assert diagnostic["coherency_claim_eligible"] is False
    assert diagnostic["cache_flush_before_every_request"] is True
    assert diagnostic["comparison_basis"] == "output_token_ids_sha256"
    assert diagnostic["fixed_work_shape_observed"] is True
    assert diagnostic["repeat_trajectory_equal"] is False


def test_fixed_length_repeat_falls_back_to_redacted_text_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        hotspot,
        "flush_cache",
        lambda *_args, **_kwargs: {"flushed": True},
    )
    monkeypatch.setattr(
        hotspot.baseline,
        "run_benchmark",
        lambda *_args, **_kwargs: {
            "diagnostic_only": True,
            "performance_claim_eligible": False,
            "ignore_eos": True,
            "exact_requested_token_shape": True,
            "completion_tokens": 64,
            "output_sha256": "a" * 64,
            "output_token_ids_sha256": None,
        },
    )

    diagnostic = hotspot.run_deterministic_repeat_diagnostic(
        _diagnostic_arguments(tmp_path),
        input_ids=[1, 2, 3],
        decode_output_ids=lambda _ids: "redacted",
        terminal_token_ids=(),
    )

    assert diagnostic["comparison_basis"] == (
        "decoded_output_sha256_and_completion_tokens"
    )
    assert diagnostic["all_output_token_hashes_available"] is False
    assert diagnostic["repeat_trajectory_equal"] is True


def _nvlink_counter_output(base: int, *, include_extra_device: bool = False) -> str:
    lines = ["diagnostic text that must not enter the receipt: SECRET_PROMPT"]
    devices = (0, 1, 2) if include_extra_device else (0, 1)
    for device in devices:
        lines.append(f"GPU {device}: NVIDIA GeForce RTX 3090 (UUID: redacted)")
        for link in range(4):
            offset = device * 100 + link * 10
            lines.append(f"     Link {link}: Data Tx: {base + offset + 1} KiB")
            lines.append(f"     Link {link}: Data Rx: {base + offset + 2} KiB")
    return "\n".join(lines)


def test_prompt_pair_is_equal_shape_coherent_and_partially_shared() -> None:
    pair = hotspot.build_prompt_pair(
        CharacterTokenizer(),
        FramingEncoder(),
        3_000,
        desired_prefix_ratio=0.65,
    )

    assert len(pair.exact_ids) == 3_000
    assert len(pair.near_ids) == 3_000
    assert pair.exact_ids != pair.near_ids
    assert 600 <= pair.common_prefix_tokens < 3_000
    assert pair.common_suffix_tokens >= 16


def test_trace_source_selects_rank_with_complete_request_records() -> None:
    server_info: dict[str, object] = {
        "internal_states": [
            {
                "dspark_info_record": {
                    "components": ["core", "reqs"],
                    "records": [{"forward_ct": 1, "reqs": [{"rid": "other"}]}],
                }
            },
            {
                "dspark_info_record": {
                    "components": ["core", "reqs", "verify_logits"],
                    "records": [
                        {"forward_ct": 2, "reqs": [{"rid": "wanted"}]},
                        {"forward_ct": 3, "reqs": [{"rid": "wanted"}]},
                    ],
                }
            },
        ]
    }

    records, source, counts, components = hotspot._trace_sources(server_info, "wanted")

    assert [record["forward_ct"] for record in records] == [2, 3]
    assert source == 1
    assert counts == [0, 2]
    assert "verify_logits" in components


def test_fixed_startup_tier_witnesses_compact_mode_without_trace() -> None:
    assert (
        hotspot._ragged_verify_mode(
            {"speculative_dspark_fixed_verify_len": 4, "internal_states": [{}]}
        )
        == "compact"
    )


def test_trace_summary_reports_cycles_tiers_and_stepwise_logit_evidence() -> None:
    records = [
        {
            "forward_ct": 10,
            "step_cpu_ms": 82.0,
            "step_gpu_ms": 81.0,
            "draft_gpu_ms": 6.0,
            "target_verify_gpu_ms": 74.0,
            "internal_gpu_ms": {
                "target.attention_indexer.layer_02.c4_indexer": 1.25,
                "target.attention_indexer.layer_03.c4_indexer": 0.75,
                "target.routed_moe.layer_02.target": 4.0,
            },
            "verify_tokens_graph_key": 6,
            "reqs": [
                {
                    "rid": "wanted",
                    "acc_len": 2,
                    "verify_len": 6,
                    "greedy_step_matches": [True, False, True, False, False],
                    "greedy_target_logit_margins": [0.0, 1.5, 0.0, 2.0, 3.0],
                }
            ],
        },
        {
            "forward_ct": 11,
            "step_cpu_ms": 84.0,
            "step_gpu_ms": 83.0,
            "draft_gpu_ms": 7.0,
            "target_verify_gpu_ms": 76.0,
            "internal_gpu_ms": {
                "target.attention_indexer.layer_02.c4_indexer": 1.5,
                "target.attention_indexer.layer_03.c4_indexer": 0.5,
                "target.routed_moe.layer_02.target": 5.0,
            },
            "verify_tokens_graph_key": 6,
            "reqs": [
                {
                    "rid": "wanted",
                    "acc_len": 6,
                    "verify_len": 6,
                    "greedy_step_matches": [True, True, True, True, True],
                    "greedy_target_logit_margins": [0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ],
        },
    ]

    summary = hotspot.summarize_trace(records, "wanted")

    assert summary["cycle_count"] == 2
    assert summary["committed_tokens"] == 8
    assert summary["mean_committed_tokens_per_cycle"] == 4.0
    assert summary["committed_tokens_per_whole_gpu_cycle_second"] == 48.780488
    assert summary["committed_tokens_per_whole_cpu_cycle_second"] == 48.192771
    assert summary["whole_gpu_cycle_ms_total"] == 164.0
    assert summary["whole_cpu_cycle_ms_total"] == 166.0
    assert summary["acceptance_distribution"] == {"2": 1, "6": 1}
    assert summary["verify_graph_key_distribution"] == {"6": 2}
    assert summary["target_verify_gpu_ms"]["mean"] == 75.0
    assert summary["internal_gpu_ms_total"]["mean"] == 6.5
    assert (
        summary["internal_gpu_ms_by_category"]["target.attention_indexer"]["mean"]
        == 2.0
    )
    assert summary["internal_gpu_ms_by_category"]["target.routed_moe"]["mean"] == 4.5
    assert (
        summary["internal_gpu_ms_by_range"][
            "target.attention_indexer.layer_02.c4_indexer"
        ]["mean"]
        == 1.375
    )
    assert summary["stepwise_greedy_comparison"][1]["match_rate"] == 0.5
    assert summary["full_block_counterfactual_tiers"]["2"]["committed_tokens"] == 4
    assert summary["full_block_counterfactual_tiers"]["6"]["committed_tokens"] == 8

    aggregate = hotspot.summarize_whole_cycle_objective(
        {
            "first": {"trace": summary},
            "second": {
                "trace": {
                    "committed_tokens": 4,
                    "cycle_count": 1,
                    "whole_gpu_cycle_ms_total": 80.0,
                    "whole_cpu_cycle_ms_total": 82.0,
                }
            },
        }
    )
    assert aggregate["committed_tokens"] == 12
    assert aggregate["cycle_count"] == 3
    assert aggregate["committed_tokens_per_whole_gpu_cycle_second"] == 49.180328
    assert aggregate["committed_tokens_per_whole_cpu_cycle_second"] == 48.387097


def test_attribution_keeps_prefill_compile_confound_explicit() -> None:
    def phase(ttft: float, target_ms: float) -> dict[str, object]:
        return {
            "benchmark": {
                "time_to_first_token_seconds": ttft,
                "output_sha256": "a" * 64,
                "completion_tokens": 128,
            },
            "trace": {"target_verify_gpu_ms": {"mean": target_ms}},
        }

    phases = {
        "cold_first_exact": phase(4.0, 80.0),
        "radix_hot_exact": phase(0.5, 75.0),
        "radix_hot_near": phase(1.0, 76.0),
        "warm_no_radix_near": phase(3.0, 76.0),
        "warm_no_radix_exact": phase(3.5, 74.0),
    }
    attribution = hotspot.build_attribution(
        phases,
        {
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
            "disable_prefill_cuda_graph": True,
        },
    )

    assert (
        attribution["radix_kv_reuse"][
            "exact_ttft_seconds_saved_vs_warm_flushed_control"
        ]
        == 3.0
    )
    assert attribution["cpu_weight_page_cache_locality"]["eligible"] is True
    assert "confounded" in attribution["compilation"]["ttft_interpretation"]


def test_endpoint_derivation_keeps_authority_and_replaces_path() -> None:
    assert (
        hotspot._derive_endpoint(
            "http://127.0.0.1:30010/generate?old=1", "/flush_cache", "timeout=30"
        )
        == "http://127.0.0.1:30010/flush_cache?timeout=30"
    )


def test_nvlink_snapshot_uses_injected_command_and_projects_exact_tp2_links() -> None:
    observed_commands: list[tuple[str, ...]] = []

    def read_command(arguments: Sequence[str]) -> str:
        observed_commands.append(tuple(arguments))
        return _nvlink_counter_output(100, include_extra_device=True)

    counters = hotspot.snapshot_nvlink_counters(read_command)

    assert observed_commands == [("nvidia-smi", "nvlink", "-gt", "d")]
    assert frozenset(counters) == hotspot.expected_nvlink_counter_keys()
    assert len(counters) == 2 * 4 * 2
    assert not any(key.startswith("gpu2.") for key in counters)


def test_nvlink_receipt_requires_positive_traffic_on_every_direction() -> None:
    before = hotspot.snapshot_nvlink_counters(
        lambda _arguments: _nvlink_counter_output(100)
    )
    after = {
        key: value + index + 1 for index, (key, value) in enumerate(before.items())
    }

    receipt = hotspot.build_nvlink_traffic_receipt(before, after)

    assert set(receipt) == {
        "counters_before",
        "counters_after",
        "counter_deltas",
    }
    assert len(receipt["counter_deltas"]) == 16
    assert all(value > 0 for value in receipt["counter_deltas"].values())
    assert "SECRET_PROMPT" not in json.dumps(receipt)

    idle_after = dict(after)
    idle_key = next(iter(before))
    idle_after[idle_key] = before[idle_key]
    with pytest.raises(
        hotspot.HotspotBenchmarkError,
        match="did not move every expected NVLink payload counter",
    ):
        hotspot.build_nvlink_traffic_receipt(before, idle_after)


def test_nvlink_attribution_fails_closed_on_missing_or_decreased_counter() -> None:
    before = hotspot.snapshot_nvlink_counters(
        lambda _arguments: _nvlink_counter_output(100)
    )
    after = {key: value + 1 for key, value in before.items()}

    missing = dict(after)
    missing.pop(next(iter(missing)))
    with pytest.raises(hotspot.HotspotBenchmarkError, match="wrong counter set"):
        hotspot.build_nvlink_traffic_receipt(before, missing)

    decreased = dict(after)
    decreased_key = next(iter(decreased))
    decreased[decreased_key] = before[decreased_key] - 1
    with pytest.raises(hotspot.HotspotBenchmarkError, match="counter decreased"):
        hotspot.build_nvlink_traffic_receipt(before, decreased)


def test_nvlink_snapshot_fails_before_requests_when_a_link_is_not_visible() -> None:
    incomplete = _nvlink_counter_output(100).replace(
        "     Link 3: Data Rx: 132 KiB", ""
    )

    with pytest.raises(
        hotspot.HotspotBenchmarkError, match="missing expected counters"
    ):
        hotspot.snapshot_nvlink_counters(lambda _arguments: incomplete)


def test_hotspot_plan_request_preserves_dry_run_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_request(
        url: str,
        *,
        timeout_seconds: float,
        payload: dict[str, object] | None = None,
    ) -> tuple[int, bytes, object]:
        observed.update(
            {"url": url, "timeout_seconds": timeout_seconds, "payload": payload}
        )
        rank_receipts = [
            {
                "generation": 7,
                "dry_run": True,
                "plan_path": "/tmp/plan.json",
                "ep_rank": ep_rank,
                "ep_size": 2,
            }
            for ep_rank in range(2)
        ]
        return (
            200,
            b'{"success":true}',
            {
                "success": True,
                "receipts": [
                    {
                        **rank_receipts[0],
                        "rank_receipts": rank_receipts,
                    }
                ],
            },
        )

    monkeypatch.setattr(hotspot, "_request_json", fake_request)

    receipt = hotspot.request_hotspot_plan(
        "http://127.0.0.1:30010/kt_expert_hotspot",
        30.0,
        plan_path=Path("/tmp/plan.json"),
        generation=7,
        dry_run=True,
    )

    assert receipt["success"] is True
    assert observed["payload"] == {
        "plan_path": "/tmp/plan.json",
        "generation": 7,
        "dry_run": True,
    }


def test_hotspot_plan_request_rejects_partial_commit_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_receipts = [
        {
            "generation": 8,
            "dry_run": False,
            "plan_path": "/tmp/plan.json",
            "ep_rank": ep_rank,
            "ep_size": 2,
            "last_committed_generation": 8 if ep_rank == 0 else 7,
        }
        for ep_rank in range(2)
    ]
    monkeypatch.setattr(
        hotspot,
        "_request_json",
        lambda *args, **kwargs: (
            200,
            b'{"success":true}',
            {
                "success": True,
                "receipts": [
                    {
                        **rank_receipts[0],
                        "rank_receipts": rank_receipts,
                    }
                ],
            },
        ),
    )

    with pytest.raises(
        hotspot.HotspotBenchmarkError, match="did not commit every rank"
    ):
        hotspot.request_hotspot_plan(
            "http://127.0.0.1:30010/kt_expert_hotspot",
            30.0,
            plan_path=Path("/tmp/plan.json"),
            generation=8,
            dry_run=False,
        )


def test_dspark_control_accepts_per_rank_boolean_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hotspot,
        "_request_json",
        lambda *args, **kwargs: (200, b"[true,true]", [True, True]),
    )

    receipt = hotspot.set_dspark_control(
        "http://127.0.0.1:30010/set_internal_state",
        30.0,
        dspark_force_verify_len=4,
    )

    assert receipt == {"updated": True, "rank_results": [True, True]}


def test_server_contract_requires_graphs_524k_two_gpus_and_cpu_offload() -> None:
    valid = {
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "kv_cache_dtype": "fp8_e4m3",
        "dsv4_oscar_int2_kv_storage": True,
        "dsv4_oscar_algorithm": "oscar-int2-asym-g64-v1",
        "dsv4_oscar_artifact_sha256": "a" * 64,
        "dsv4_oscar_model_config_sha256": "b" * 64,
        "dsv4_oscar_artifact_provenance_sha256": "c" * 64,
        "dsv4_oscar_checkpoint_sha256": "d" * 64,
        "dsv4_oscar_checkpoint_fingerprint_sha256": "e" * 64,
        "dsv4_oscar_admission_sha256": "f" * 64,
        "dsv4_oscar_admission_receipt_sha256": "1" * 64,
        "dsv4_oscar_model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "dsv4_kv_storage_mode": "oscar_int2_asymmetric+protected_swa_bfloat16",
        "dsv4_swa_kv_bytes_per_token": 1_024,
        "dsv4_c4_kv_bytes_per_token": 272,
        "dsv4_c128_kv_bytes_per_token": 272,
        "dsv4_oscar_c4_scorer": True,
        "dsv4_oscar_c4_scorer_algorithm": ("oscar-int2-c4-asym-c128-fp32-adjacent4-v1"),
        "dsv4_c4_indexer_bytes_per_token": 40,
        "dsv4_int4_kv_storage": False,
        "dsv4_int4_c4_indexer_storage": False,
        "dsv4_sm86_c128_bf16_storage": False,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "disable_prefill_cuda_graph": False,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "enable_p2p_check": True,
        "pre_warm_nccl": True,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 2,
        "kt_cpuinfer": 56,
        "speculative_dspark_block_size": 5,
        "speculative_num_draft_tokens": 6,
    }
    hotspot.validate_server_contract(valid)

    candidate_valid = {
        **valid,
        "dsv4_fused_t5_moe_configured": True,
        "dsv4_fused_t5_moe_all_workers_active": True,
        "dsv4_fused_t5_moe_active_worker_count": 2,
        "dsv4_oscar_fused_c4_pipeline_configured": True,
        "dsv4_sm86_small_batch_gemm_worker_telemetry": [
            {
                "tp_rank": 0,
                "fused_t5_moe_configured": True,
                "fused_t5_moe_conversion_count": 14,
                "fused_t5_moe_apply_count": 28,
                "fused_t5_kt_routing_apply_count": 20,
            },
            {
                "tp_rank": 1,
                "fused_t5_moe_configured": True,
                "fused_t5_moe_conversion_count": 14,
                "fused_t5_moe_apply_count": 28,
                "fused_t5_kt_routing_apply_count": 20,
            },
        ],
    }
    hotspot.validate_server_contract(
        candidate_valid,
        require_fused_t5_moe=True,
        require_oscar_fused_c4_pipeline=True,
    )

    candidate_false_counter = {
        **candidate_valid,
        "dsv4_sm86_small_batch_gemm_worker_telemetry": [
            {
                **candidate_valid["dsv4_sm86_small_batch_gemm_worker_telemetry"][0],
                "fused_t5_moe_apply_count": False,
            },
            candidate_valid["dsv4_sm86_small_batch_gemm_worker_telemetry"][1],
        ],
    }
    with pytest.raises(
        hotspot.HotspotBenchmarkError,
        match="fused_t5_moe_worker_proof_incomplete",
    ):
        hotspot.validate_server_contract(
            candidate_false_counter,
            require_fused_t5_moe=True,
        )

    split_valid = {
        **valid,
        "dsv4_oscar_int2_split_history": True,
        "dsv4_oscar_int2_split_history_execution": (
            hotspot.OSCAR_SPLIT_HISTORY_EXECUTION
        ),
        "dsv4_oscar_int2_split_history_split_map": dict(
            hotspot.OSCAR_SPLIT_HISTORY_SPLIT_MAP
        ),
        "dsv4_oscar_int2_split_history_workspace_bytes": (
            hotspot.OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
        ),
        "dsv4_oscar_int2_split_history_max_partial_rows": 32,
        "dsv4_oscar_int2_split_history_sink_owner": "stage2-exactly-once",
        "dsv4_oscar_int2_split_history_prefill_enabled": False,
        "dsv4_oscar_int2_split_history_fixed_address": True,
        "dsv4_oscar_int2_split_history_workspace_address": 123_456,
        "internal_states": [
            {
                "dsv4_oscar_worker_telemetry_workers": [
                    {
                        "pid": 1001,
                        "gpu_id": 0,
                        "tp_rank": 0,
                        "pp_rank": 0,
                        "dp_rank": 0,
                        **hotspot.OSCAR_SPLIT_HISTORY_WORKER_FIELDS,
                        "dsv4_oscar_int2_split_history_workspace_address": 123_456,
                    },
                    {
                        "pid": 1002,
                        "gpu_id": 1,
                        "tp_rank": 1,
                        "pp_rank": 0,
                        "dp_rank": 0,
                        **hotspot.OSCAR_SPLIT_HISTORY_WORKER_FIELDS,
                        "dsv4_oscar_int2_split_history_workspace_address": 234_567,
                    },
                ]
            }
        ],
    }
    hotspot.validate_server_contract(
        split_valid,
        require_oscar_split_history=True,
    )
    assert hotspot.validate_oscar_split_history_workers(split_valid) == {
        "worker_count": 2,
        "worker_pids": [1001, 1002],
        "tp_pp_gpu_ranks": [[0, 0, 0], [1, 0, 1]],
        "workspace_bytes_per_worker": hotspot.OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES,
        "workspace_addresses": [123_456, 234_567],
        "fixed_address": True,
    }

    split_invalid = dict(split_valid)
    split_invalid["dsv4_oscar_int2_split_history_prefill_enabled"] = True
    with pytest.raises(
        hotspot.HotspotBenchmarkError,
        match="oscar_split_history_prefill_not_monolithic",
    ):
        hotspot.validate_server_contract(
            split_invalid,
            require_oscar_split_history=True,
        )

    split_missing_worker = dict(split_valid)
    split_missing_worker["internal_states"] = []
    with pytest.raises(
        hotspot.HotspotBenchmarkError,
        match="exactly two workers",
    ):
        hotspot.validate_server_contract(
            split_missing_worker,
            require_oscar_split_history=True,
        )

    invalid = dict(valid)
    invalid["disable_decode_cuda_graph"] = True
    invalid["context_length"] = 131_072
    with pytest.raises(hotspot.HotspotBenchmarkError, match="server contract mismatch"):
        hotspot.validate_server_contract(invalid)


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    (
        ("max_total_tokens", 524_287, "max_total_tokens_below_524288"),
        ("tp_size", 3, "tp_size_not_2"),
        ("pp_size", 2, "pp_size_not_1"),
        ("ep_size", 3, "ep_size_not_2"),
        ("disable_prefill_cuda_graph", True, "prefill_cuda_graph_disabled"),
        (
            "dsv4_oscar_int2_kv_storage",
            False,
            "oscar_int2_kv_not_admitted",
        ),
        ("dsv4_oscar_algorithm", "int2", "oscar_algorithm_mismatch"),
        (
            "dsv4_c4_indexer_bytes_per_token",
            64,
            "oscar_c4_scorer_row_not_40_bytes",
        ),
        ("dsv4_int4_kv_storage", True, "generic_int4_kv_enabled"),
        (
            "cuda_graph_backend_decode",
            "breakable",
            "decode_cuda_graph_backend_not_full",
        ),
        (
            "cuda_graph_backend_prefill",
            "disabled",
            "prefill_cuda_graph_backend_not_breakable",
        ),
    ),
)
def test_server_contract_rejects_stale_or_wrong_tp2_graph_shape(
    field: str,
    value: object,
    issue_code: str,
) -> None:
    server_info = {
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "kv_cache_dtype": "fp8_e4m3",
        "dsv4_oscar_int2_kv_storage": True,
        "dsv4_oscar_algorithm": "oscar-int2-asym-g64-v1",
        "dsv4_oscar_artifact_sha256": "a" * 64,
        "dsv4_oscar_model_config_sha256": "b" * 64,
        "dsv4_oscar_artifact_provenance_sha256": "c" * 64,
        "dsv4_oscar_checkpoint_sha256": "d" * 64,
        "dsv4_oscar_checkpoint_fingerprint_sha256": "e" * 64,
        "dsv4_oscar_admission_sha256": "f" * 64,
        "dsv4_oscar_admission_receipt_sha256": "1" * 64,
        "dsv4_oscar_model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "dsv4_kv_storage_mode": "oscar_int2_asymmetric+protected_swa_bfloat16",
        "dsv4_swa_kv_bytes_per_token": 1_024,
        "dsv4_c4_kv_bytes_per_token": 272,
        "dsv4_c128_kv_bytes_per_token": 272,
        "dsv4_oscar_c4_scorer": True,
        "dsv4_oscar_c4_scorer_algorithm": ("oscar-int2-c4-asym-c128-fp32-adjacent4-v1"),
        "dsv4_c4_indexer_bytes_per_token": 40,
        "dsv4_int4_kv_storage": False,
        "dsv4_int4_c4_indexer_storage": False,
        "dsv4_sm86_c128_bf16_storage": False,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "disable_prefill_cuda_graph": False,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "enable_p2p_check": True,
        "pre_warm_nccl": True,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 2,
        "kt_cpuinfer": 56,
        "speculative_dspark_block_size": 5,
        "speculative_num_draft_tokens": 6,
    }
    server_info[field] = value

    with pytest.raises(hotspot.HotspotBenchmarkError, match=issue_code):
        hotspot.validate_server_contract(server_info)


def test_adaptive_policy_rejects_a_startup_fixed_tier() -> None:
    with pytest.raises(
        hotspot.HotspotBenchmarkError,
        match="explicitly empty DSV4_DSPARK_FIXED_VERIFY_LEN",
    ):
        hotspot.validate_verify_policy(
            {
                "speculative_dspark_sps_table_path": "/tmp/sps.json",
                "speculative_dspark_fixed_verify_len": 4,
            },
            verify_policy="adaptive",
            ragged_mode="compact",
        )

    hotspot.validate_verify_policy(
        {
            "speculative_dspark_sps_table_path": "/tmp/sps.json",
            "speculative_dspark_fixed_verify_len": None,
        },
        verify_policy="adaptive",
        ragged_mode="compact",
    )


def test_loaded_expert_plan_is_bound_to_loader_validated_server_hash(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "g14-plan.pt"
    plan.write_bytes(b"loader-admitted-plan")
    expected_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()

    receipt = hotspot.validate_loaded_expert_plan(
        {"kt_hybrid_expert_plan_sha256": expected_sha256}, plan
    )

    assert receipt["expected_plan_sha256"] == expected_sha256
    assert receipt["binding"] == "launcher-hash-and-kt-loader-validated"
    with pytest.raises(hotspot.HotspotBenchmarkError, match="does not match"):
        hotspot.validate_loaded_expert_plan(
            {"kt_hybrid_expert_plan_sha256": "0" * 64}, plan
        )


def test_cache_contract_requires_measured_hot_reuse_and_cold_zero() -> None:
    pair = hotspot.PromptPair(
        exact_ids=list(range(100)),
        near_ids=list(range(100)),
        mutation_record_index=1,
        common_prefix_tokens=60,
        common_suffix_tokens=10,
    )
    cache_counts = {
        "cold_first_exact": 0,
        "radix_hot_exact": 96,
        "radix_hot_near": 48,
        "warm_no_radix_near": 0,
        "warm_no_radix_exact": 0,
    }
    phases = {
        name: {"benchmark": {"server_cached_tokens": count}}
        for name, count in cache_counts.items()
    }
    hotspot.validate_cache_contract(phases, pair)

    phases["radix_hot_near"]["benchmark"]["server_cached_tokens"] = 0
    hotspot.validate_cache_contract(phases, pair)

    phases["radix_hot_near"]["benchmark"]["server_cached_tokens"] = 61
    with pytest.raises(hotspot.HotspotBenchmarkError, match="common prefix"):
        hotspot.validate_cache_contract(phases, pair)


def test_repeated_prompts_report_greedy_output_trajectory_drift() -> None:
    prompt_kind = {
        "cold_first_exact": "exact",
        "radix_hot_exact": "exact",
        "radix_hot_near": "near",
        "warm_no_radix_near": "near",
        "warm_no_radix_exact": "exact",
    }
    phases = {
        name: {
            "benchmark": {
                "output_sha256": "exact-hash" if kind == "exact" else "near-hash",
                "completion_tokens": 128,
            }
        }
        for name, kind in prompt_kind.items()
    }

    stable = hotspot.summarize_repeat_trajectories(phases)
    assert stable["all_paired_trajectories_comparable"] is True

    phases["warm_no_radix_exact"]["benchmark"]["output_sha256"] = "drifted"
    drifted = hotspot.summarize_repeat_trajectories(phases)
    assert drifted["all_paired_trajectories_comparable"] is False
    assert drifted["groups"]["exact"]["paired_trajectory_comparable"] is False


def test_recorder_receipts_keep_only_identity_not_tensor_contents(
    tmp_path: Path,
) -> None:
    (tmp_path / "expert_distribution_recorder_old.pt").write_bytes(b"old")
    before = hotspot._recorder_names(tmp_path)
    first = tmp_path / "expert_distribution_recorder_1.pt"
    first.write_bytes(b"tp-ep-reduced-stat-tensor-placeholder")

    receipts = hotspot.wait_for_recorder_receipts(tmp_path, before, 1.0)

    assert [Path(receipt["path"]).name for receipt in receipts] == [first.name]
    assert all(set(receipt) == {"path", "size_bytes", "sha256"} for receipt in receipts)
    assert all(len(receipt["sha256"]) == 64 for receipt in receipts)
