from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import run_dsv4_flash_candidate_campaign as campaign

_PHASES = (
    ("cold_first_exact", True),
    ("radix_hot_exact", False),
    ("radix_hot_near", False),
    ("warm_no_radix_near", True),
    ("warm_no_radix_exact", True),
)


def _small_batch_worker(tp_rank: int) -> dict[str, object]:
    signatures = [
        {
            "m": 8,
            "logical_rows": 1,
            "n": 4096,
            "k": k,
            "local_experts": 14 + tp_rank,
            "selection_count": 2,
        }
        for k in (2048, 4096)
    ]
    return {
        "configured": True,
        "patch_state": "installed",
        "patch_installed": True,
        "patch_error": None,
        "selection_count": 4,
        "observed_signatures": signatures,
        "selected_config": {
            "block_n": 128,
            "split_k": 2,
            "num_stages": 4,
            "num_warps": 4,
        },
        "expected_call_parameters": list(
            campaign.EXPECTED_SM86_SMALL_BATCH_CALL_PARAMETERS
        ),
        "pid": 1000 + tp_rank,
        "gpu_id": tp_rank,
        "tp_rank": tp_rank,
        "pp_rank": 0,
        "dp_rank": 0,
    }


def _small_batch_server_info() -> dict[str, object]:
    return {
        "dsv4_sm86_small_batch_gemm_expected_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_reporting_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_active_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_all_workers_active": True,
        "dsv4_sm86_small_batch_gemm_worker_telemetry": [
            _small_batch_worker(0),
            _small_batch_worker(1),
        ],
    }


def _inline_worker(tp_rank: int) -> dict[str, object]:
    return {
        "pid": 2000 + tp_rank,
        "gpu_id": tp_rank,
        "tp_rank": tp_rank,
        "pp_rank": 0,
        "dp_rank": None,
        "moe_ep_rank": tp_rank,
        "moe_dp_rank": 0,
        "telemetry": {
            "environment_enabled": True,
            "expected_numa_id": tp_rank,
            "required_worker_count": 56,
            "singleton_instance_count": 1,
            "registered_configuration_count": 1,
            "dispatch_count_before_startup_probe": 0,
            "dispatch_count_after_startup_probe": 1,
            "startup_probe_advanced_dispatch_count": True,
            "task_queue_affinity": {},
            "single_numa_inline_dispatch": {},
            "worker_pool_affinity": {},
            "startup_probe": {},
            "task_queue_live_cpu_affinity": [],
            "worker_live_cpu_affinities": [],
            "all_live_worker_affinities_exact": True,
            "all_worker_cpus_in_expected_numa": True,
        },
        "validation_error": None,
    }


def _inline_server_info() -> dict[str, object]:
    return {
        "kt_single_numa_inline_dispatch_configured": True,
        "kt_single_numa_inline_dispatch_expected_worker_count": 2,
        "kt_single_numa_inline_dispatch_reporting_worker_count": 2,
        "kt_single_numa_inline_dispatch_active_worker_count": 2,
        "kt_single_numa_inline_dispatch_invalid_worker_count": 0,
        "kt_single_numa_inline_dispatch_duplicate_worker_count": 0,
        "kt_single_numa_inline_dispatch_rank_coverage_valid": True,
        "kt_single_numa_inline_dispatch_ep2_topology_valid": True,
        "kt_single_numa_inline_dispatch_all_workers_active": True,
        "kt_single_numa_inline_dispatch_worker_telemetry": [
            _inline_worker(0),
            _inline_worker(1),
        ],
    }


def _scale_fold_telemetry() -> dict[str, object]:
    return {
        "schema_version": 1,
        "requested_mode": "lut-v1",
        "configuration_valid": True,
        "architecture_supported": True,
        "execution_mode": "not-executed",
        "n_block": 128,
        "fold_safe_minimum": 2,
        "fold_safe_maximum": 252,
        "lut_identity": "mxfp4-e2m1-bf16-ue8m0-lut-v1",
        "lut_hash_algorithm": "fnv1a64-le",
        "lut_hash": "06d1a83dbf20f545",
        "lut_bytes": 16_384,
        "buffers_constructed": 258,
        "buffers_finalized": 258,
        "buffers_admitted": 258,
        "buffers_rejected": 0,
        "whole_buffer_domain_finalized": True,
        "whole_buffer_domain_admitted": True,
        "scale_bytes_audited": 1_000_000,
        "unsafe_scale_bytes": 0,
        "nan_scale_bytes": 0,
        "invalid_mode_requests": 0,
        "observed_scale_minimum": 118,
        "observed_scale_maximum": 126,
        "decode_dispatch_count": 0,
        "prefill_dispatch_count": 0,
        "real_dispatch_count": 0,
        "scale_fold_dispatch_count": 0,
        "lut_decode_dispatch_count": 0,
        "lut_prefill_dispatch_count": 0,
        "exponent_decode_dispatch_count": 0,
        "exponent_prefill_dispatch_count": 0,
        "fallback_dispatch_count": 0,
        "fallback_decode_dispatch_count": 0,
        "fallback_prefill_dispatch_count": 0,
        "zero_invalid_or_fallback_counts": True,
    }


def _scale_fold_worker(tp_rank: int) -> dict[str, object]:
    return {
        "pid": 2000 + tp_rank,
        "gpu_id": tp_rank,
        "tp_rank": tp_rank,
        "pp_rank": 0,
        "dp_rank": None,
        "moe_ep_rank": tp_rank,
        "moe_dp_rank": 0,
        "telemetry": _scale_fold_telemetry(),
        "validation_error": None,
    }


def _scale_fold_server_info() -> dict[str, object]:
    return {
        "kt_mxfp4_avx_scale_fold_configured": True,
        "kt_mxfp4_avx_scale_fold_requested_mode": "lut-v1",
        "kt_mxfp4_avx_scale_fold_expected_n_block": 128,
        "kt_mxfp4_avx_scale_fold_expected_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_reporting_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_active_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_invalid_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_duplicate_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_rank_coverage_valid": True,
        "kt_mxfp4_avx_scale_fold_ep2_topology_valid": True,
        "kt_mxfp4_avx_scale_fold_all_workers_active": True,
        "kt_mxfp4_avx_scale_fold_worker_telemetry": [
            _scale_fold_worker(0),
            _scale_fold_worker(1),
        ],
    }


def _oscar_server_info() -> dict[str, object]:
    return {
        **campaign.OSCAR_STATIC_SERVER_INFO,
        "dsv4_oscar_artifact_sha256": "1" * 64,
        "dsv4_oscar_model_config_sha256": "2" * 64,
        "dsv4_oscar_artifact_provenance_sha256": "3" * 64,
        "dsv4_oscar_checkpoint_sha256": "4" * 64,
        "dsv4_oscar_checkpoint_fingerprint_sha256": "5" * 64,
        "dsv4_oscar_admission_sha256": "6" * 64,
        "dsv4_oscar_admission_receipt_sha256": "7" * 64,
        "dsv4_oscar_wo_a_absorption_state": {
            "enabled": True,
            "consumer_role": "target_compressed",
            "target_only": True,
            "applied": True,
            "apply_count": 1,
            "artifact_sha256": "1" * 64,
            "admission_sha256": "6" * 64,
            "expected_local_compressed_layer_ids": list(range(2, 43)),
            "absorbed_local_layer_ids": list(range(2, 43)),
            "runtime_restore_skipped_layer_ids": list(range(2, 43)),
            "all_local_target_compressed_layers_absorbed": True,
            "all_local_target_compressed_layers_skip_runtime_restore": True,
            "weight_dtype": "bfloat16",
            "head_layout": "per-head-nope448-rope64",
            "fold_orientation": "wo_a_nope@rotation",
            "rope_columns_unchanged": True,
        },
    }


def _hotspot_receipt(
    decode_rate: float,
    *,
    ttft_seconds: float = 4.0,
    performance_claim_eligible: bool = True,
    exact_completion_tokens: tuple[int, int, int] = (128, 128, 128),
    near_completion_tokens: tuple[int, int] = (128, 128),
) -> dict[str, object]:
    output_hashes = {
        "cold_first_exact": "a" * 64,
        "radix_hot_exact": "a" * 64,
        "radix_hot_near": "b" * 64,
        "warm_no_radix_near": "b" * 64,
        "warm_no_radix_exact": "a" * 64,
    }
    if not performance_claim_eligible:
        output_hashes["radix_hot_exact"] = "c" * 64
        if near_completion_tokens == (128, 128):
            near_completion_tokens = (128, 129)

    completion_tokens = {
        "cold_first_exact": exact_completion_tokens[0],
        "radix_hot_exact": exact_completion_tokens[1],
        "warm_no_radix_exact": exact_completion_tokens[2],
        "radix_hot_near": near_completion_tokens[0],
        "warm_no_radix_near": near_completion_tokens[1],
    }

    exact_hashes = [
        output_hashes[name]
        for name in ("cold_first_exact", "radix_hot_exact", "warm_no_radix_exact")
    ]
    exact_counts = [
        completion_tokens[name]
        for name in ("cold_first_exact", "radix_hot_exact", "warm_no_radix_exact")
    ]
    near_hashes = [
        output_hashes[name] for name in ("radix_hot_near", "warm_no_radix_near")
    ]
    near_counts = [
        completion_tokens[name] for name in ("radix_hot_near", "warm_no_radix_near")
    ]
    expected_nvlink_counters = campaign.hotspot.expected_nvlink_counter_keys()
    exact_comparable = len(set(exact_hashes)) == 1 and len(set(exact_counts)) == 1
    near_comparable = len(set(near_hashes)) == 1 and len(set(near_counts)) == 1

    phases: dict[str, dict[str, object]] = {}
    for name, flush_before in _PHASES:
        phase_completion_tokens = completion_tokens[name]
        phase_committed_tokens = phase_completion_tokens + 1
        cycles = max(2, (phase_committed_tokens + 2) // 3)
        base_acceptance = phase_committed_tokens // cycles
        higher_acceptance_count = phase_committed_tokens % cycles
        acceptance_distribution = {
            str(base_acceptance): cycles - higher_acceptance_count
        }
        if higher_acceptance_count:
            acceptance_distribution[str(base_acceptance + 1)] = higher_acceptance_count
        prompt_kind = "exact" if name.endswith("exact") else "near"
        input_sha256 = "e" * 64 if prompt_kind == "exact" else "f" * 64
        cached_tokens = 2560 if name == "radix_hot_exact" else 0
        decode_seconds = (phase_completion_tokens - 1) / decode_rate
        elapsed_seconds = ttft_seconds + decode_seconds
        phases[name] = {
            "prompt_kind": prompt_kind,
            "radix_flush_before": flush_before,
            "benchmark": {
                "receipt_version": campaign.hotspot.baseline.RECEIPT_VERSION,
                "accepted": True,
                "performance_claim_eligible": True,
                "receipt_safety": "validated_natural_stop",
                "ignore_eos": False,
                "finish_reason": "stop",
                "exact_requested_token_shape": False,
                "requested_max_completion_tokens": 512,
                "terminal_token_scan": "verified",
                "semantic_validation": {"passed": True, "issue_codes": []},
                "output_sha256": output_hashes[name],
                "input_sha256": input_sha256,
                "input_tokens": 2_694,
                "server_prompt_tokens": 2_694,
                "http_status": 200,
                "saw_done": True,
                "request_id": f"dsv4-hotspot-test-{name}",
                "completion_tokens": phase_completion_tokens,
                "server_cached_tokens": cached_tokens,
                "elapsed_seconds": elapsed_seconds,
                "decode_seconds": decode_seconds,
                "decode_tokens_per_second": decode_rate,
                "time_to_first_token_seconds": ttft_seconds,
                "prefill_tokens_per_second": 2_694 / ttft_seconds,
                "total_tokens_per_second": (2_694 + phase_completion_tokens)
                / elapsed_seconds,
                "first_stream_completion_tokens": 1,
                "stream_events": cycles,
                "event_count": cycles,
            },
            "trace": {
                "committed_tokens": phase_committed_tokens,
                "cycle_count": cycles,
                "request_observation_count": cycles,
                "mean_committed_tokens_per_cycle": round(
                    phase_committed_tokens / cycles, 6
                ),
                "acceptance_distribution": acceptance_distribution,
                "verify_len_distribution": {"4": cycles},
                "verify_graph_key_distribution": {"4": cycles},
                "target_verify_gpu_ms": {
                    "mean": 60.0,
                    "p50": 60.0,
                    "p95": 60.0,
                    "count": cycles,
                },
                "draft_gpu_ms": {
                    "mean": 6.0,
                    "p50": 6.0,
                    "p95": 6.0,
                    "count": cycles,
                },
                "step_gpu_ms": {
                    "mean": 67.0,
                    "p50": 67.0,
                    "p95": 67.0,
                    "count": cycles,
                },
                "step_cpu_ms": {
                    "mean": 66.0,
                    "p50": 66.0,
                    "p95": 66.0,
                    "count": cycles - 1,
                },
            },
            "trace_components": sorted(campaign.hotspot.REQUIRED_TRACE_COMPONENTS),
            "trace_source_index": 0,
            "trace_records_by_source": [cycles],
            "expert_recorder": {"enabled": False},
        }

    return {
        "receipt_version": campaign.hotspot.RECEIPT_VERSION,
        "accepted": True,
        "performance_claim_eligible": performance_claim_eligible,
        "measurement_mode": "trace",
        "verify_policy": "4",
        "verify_logits_diagnostic": False,
        "expert_plan_provenance": {
            "expected_plan_path": "/tmp/plan.pt",
            "expected_plan_sha256": "d" * 64,
            "binding": "launcher-hash-and-kt-loader-validated",
        },
        "nvlink_traffic": {
            "counters_before": {key: 100 for key in expected_nvlink_counters},
            "counters_after": {key: 110 for key in expected_nvlink_counters},
            "counter_deltas": {key: 10 for key in expected_nvlink_counters},
        },
        "prompt_pair": {
            "input_tokens": 2_694,
            "exact_sha256": "e" * 64,
            "near_sha256": "f" * 64,
            "common_prefix_tokens": 1_877,
            "common_suffix_tokens": 137,
            "mutation_record_index": 34,
        },
        "phases": phases,
        "repeat_trajectories": {
            "all_paired_trajectories_comparable": (
                exact_comparable and near_comparable
            ),
            "groups": {
                "exact": {
                    "phase_names": [
                        "cold_first_exact",
                        "radix_hot_exact",
                        "warm_no_radix_exact",
                    ],
                    "output_sha256": exact_hashes,
                    "completion_tokens": exact_counts,
                    "hashes_equal": len(set(exact_hashes)) == 1,
                    "token_counts_equal": len(set(exact_counts)) == 1,
                    "paired_trajectory_comparable": exact_comparable,
                },
                "near": {
                    "phase_names": ["radix_hot_near", "warm_no_radix_near"],
                    "output_sha256": near_hashes,
                    "completion_tokens": near_counts,
                    "hashes_equal": len(set(near_hashes)) == 1,
                    "token_counts_equal": len(set(near_counts)) == 1,
                    "paired_trajectory_comparable": near_comparable,
                },
            },
        },
        "attribution": {
            "cpu_weight_page_cache_locality": {
                "exact_output_trajectory_comparable": exact_comparable,
                "eligible": exact_comparable,
            }
        },
    }


def _candidate(
    temporary_path: Path,
    *,
    identifier: str = "candidate",
    prerequisites: tuple[str, ...] = (),
) -> campaign.Candidate:
    return campaign.Candidate(
        identifier=identifier,
        priority=1,
        description="test candidate",
        prerequisites=prerequisites,
        environment={},
        expected_server_info={},
        expected_plan=temporary_path / "plan.pt",
        expected_plan_sha256="0" * 64,
        plan_materialized_at_launch=False,
        artifacts=(),
        policy="speed",
        predicted={},
    )


def _gates(temporary_path: Path) -> campaign.Gates:
    return campaign.Gates(
        model_path=temporary_path,
        agents_path=temporary_path / "AGENTS.md",
        server_url="http://127.0.0.1:30010/server_info",
        chat_url="http://127.0.0.1:30010/v1/chat/completions",
        input_tokens=2_694,
        output_tokens=512,
        baseline_repetitions=3,
        screen_repetitions=1,
        confirmation_repetitions=2,
        coherency_repetitions=2,
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        shutdown_timeout_seconds=1.0,
        maximum_median_ttft_seconds=7.0,
        minimum_screen_decode_ratio=0.96,
        minimum_acceptance_ratio=0.97,
        maximum_target_verify_ratio=1.05,
        minimum_confirm_ci_ratio=0.94,
        minimum_headroom_mib=1_536,
        bootstrap_samples=1_000,
    )


def _manifest_with_candidates(
    temporary_path: Path, candidates: tuple[campaign.Candidate, ...]
) -> campaign.CampaignManifest:
    baseline = campaign.Candidate(
        identifier="baseline",
        priority=0,
        description="baseline",
        prerequisites=(),
        environment={},
        expected_server_info={},
        expected_plan=temporary_path / "baseline.pt",
        expected_plan_sha256=None,
        plan_materialized_at_launch=True,
        artifacts=(),
        policy="baseline",
        predicted={},
    )
    manifest_path = temporary_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return campaign.CampaignManifest(
        path=manifest_path,
        launcher=temporary_path / "launcher.sh",
        launcher_sha256="0" * 64,
        fixed_environment={},
        controlled_environment_keys=frozenset(),
        oscar_contract=campaign.OscarContract(
            calibration_artifact=campaign.Artifact(
                temporary_path / "oscar.pt", "0" * 64
            ),
            checkpoint_fingerprint=campaign.Artifact(
                temporary_path / "fingerprint.json", "0" * 64
            ),
            admission_receipt=campaign.Artifact(
                temporary_path / "admission.json", "0" * 64
            ),
            model_id=campaign.OSCAR_MODEL_ID,
        ),
        source_artifacts=(),
        gates=_gates(temporary_path),
        baseline=baseline,
        candidates=candidates,
    )


def _write_result(
    manifest: campaign.CampaignManifest,
    work_directory: Path,
    candidate_id: str,
    stage: str,
    *,
    qualified: bool = True,
    decode_rate: float = 36.0,
) -> None:
    result_path = work_directory / candidate_id / f"{stage}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "format": campaign.RESULT_FORMAT,
                "manifest_sha256": campaign.sha256_file(manifest.path),
                "qualified": qualified,
                "hotspot_receipts": [_hotspot_receipt(decode_rate)],
            }
        ),
        encoding="utf-8",
    )


def test_hotspot_summary_uses_whole_receipts_as_resampling_units() -> None:
    summary = campaign.summarize_hotspot_receipts(
        [_hotspot_receipt(30.0), _hotspot_receipt(42.0)],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )

    assert summary["phase_observation_count"] == 10
    assert summary["receipt_decode_means"] == [30.0, 42.0]
    assert summary["mean_decode_tokens_per_second"] == 36.0
    assert summary["mean_committed_tokens_per_cycle"] == 3.0
    assert summary["median_flushed_ttft_seconds"] == 4.0
    assert summary["performance_claim_methodology"] == "strict_paired"
    assert summary["performance_claim_scope"] == "paired_phase_and_whole_receipt"


@pytest.mark.parametrize(
    ("decode_rate", "target_met", "stretch_met"),
    ((79.999, False, False), (80.0, True, False), (90.0, True, True)),
)
def test_absolute_decode_goals_have_explicit_boundaries_without_rejecting_screens(
    tmp_path: Path,
    decode_rate: float,
    target_met: bool,
    stretch_met: bool,
) -> None:
    baseline_summary = campaign.summarize_hotspot_receipts(
        [_hotspot_receipt(decode_rate)],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )
    goals = campaign.absolute_goal_state(baseline_summary, _gates(tmp_path))
    decision = campaign.candidate_decision(
        _candidate(tmp_path),
        baseline_summary,
        baseline_summary,
        _gates(tmp_path),
        confirmation=False,
    )

    assert goals["target_decode_tokens_per_second"] == 80.0
    assert goals["stretch_decode_tokens_per_second"] == 90.0
    assert goals["maximum_ttft_seconds"] == 7.0
    assert goals["target_decode_met"] is target_met
    assert goals["stretch_decode_met"] is stretch_met
    assert decision["promoted"] is True
    assert decision["absolute_goal_state"] == goals
    assert decision["advisories"] == goals["advisories"]


def test_absolute_ttft_goal_uses_slowest_flushed_observation(tmp_path: Path) -> None:
    summary = campaign.summarize_hotspot_receipts(
        [_hotspot_receipt(90.0)],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )
    summary["median_flushed_ttft_seconds"] = 6.0
    summary["maximum_flushed_ttft_seconds"] = 7.001

    goals = campaign.absolute_goal_state(summary, _gates(tmp_path))

    assert goals["target_decode_met"] is True
    assert goals["stretch_decode_met"] is True
    assert goals["maximum_ttft_met"] is False
    assert goals["target_configuration_goal_met"] is False
    assert "absolute_7s_maximum_ttft_goal_not_met" in goals["advisories"]


def test_hotspot_summary_accepts_validated_natural_stop_trajectory_drift() -> None:
    summary = campaign.summarize_hotspot_receipts(
        [_hotspot_receipt(40.0, performance_claim_eligible=False)],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )

    assert summary["trajectory_divergent_receipt_count"] == 1
    assert summary["performance_claim_methodology"] == "unpaired_natural_stop_ensemble"


def test_hotspot_summary_rejects_ineligible_receipt_without_nvlink_evidence() -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    del receipt["nvlink_traffic"]

    with pytest.raises(campaign.CampaignError, match="NVLink traffic evidence"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


@pytest.mark.parametrize(
    ("exact_counts", "accepted"),
    [
        ((200, 250, 200), True),
        ((200, 251, 200), False),
    ],
)
def test_natural_stop_completion_spread_has_exact_twenty_five_percent_boundary(
    exact_counts: tuple[int, int, int], accepted: bool
) -> None:
    receipt = _hotspot_receipt(
        40.0,
        performance_claim_eligible=False,
        exact_completion_tokens=exact_counts,
        near_completion_tokens=(200, 200),
    )

    if accepted:
        summary = campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )
        assert summary["performance_claim_methodology"] == (
            "unpaired_natural_stop_ensemble"
        )
    else:
        with pytest.raises(campaign.CampaignError, match="spread exceeds 25%"):
            campaign.summarize_hotspot_receipts(
                [receipt], expected_input_tokens=2_694, expected_output_tokens=512
            )


def test_natural_stop_spread_must_pass_for_both_prompt_groups() -> None:
    receipt = _hotspot_receipt(
        40.0,
        performance_claim_eligible=False,
        exact_completion_tokens=(200, 250, 200),
        near_completion_tokens=(200, 251),
    )

    with pytest.raises(campaign.CampaignError, match="near.*spread exceeds 25%"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


def test_current_baseline_natural_stop_count_shapes_are_accepted() -> None:
    receipts = [
        _hotspot_receipt(
            35.0,
            performance_claim_eligible=False,
            exact_completion_tokens=(232, 240, 250),
            near_completion_tokens=(229, 231),
        ),
        _hotspot_receipt(
            35.0,
            performance_claim_eligible=False,
            exact_completion_tokens=(238, 253, 232),
            near_completion_tokens=(270, 255),
        ),
        _hotspot_receipt(
            35.0,
            performance_claim_eligible=False,
            exact_completion_tokens=(234, 232, 242),
            near_completion_tokens=(231, 216),
        ),
    ]

    summary = campaign.summarize_hotspot_receipts(
        receipts, expected_input_tokens=2_694, expected_output_tokens=512
    )

    assert summary["receipt_count"] == 3
    assert summary["phase_observation_count"] == 15
    assert summary["trajectory_divergent_receipt_count"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted", False),
        ("performance_claim_eligible", False),
        ("receipt_safety", "validated_forced_length"),
        ("ignore_eos", True),
        ("finish_reason", "length"),
        ("exact_requested_token_shape", True),
        ("requested_max_completion_tokens", 128),
        ("terminal_token_scan", "unused"),
        ("completion_tokens", 512),
        ("completion_tokens", True),
        ("output_sha256", "malformed"),
    ],
)
def test_natural_stop_fallback_rejects_invalid_phase_metadata(
    field: str, value: object
) -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    benchmark = phases["cold_first_exact"]["benchmark"]
    assert isinstance(benchmark, dict)
    benchmark[field] = value

    with pytest.raises(campaign.CampaignError):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


def test_natural_stop_fallback_rejects_phase_repeat_mismatch() -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    repeat = receipt["repeat_trajectories"]
    assert isinstance(repeat, dict)
    groups = repeat["groups"]
    assert isinstance(groups, dict)
    exact = groups["exact"]
    assert isinstance(exact, dict)
    exact["completion_tokens"] = [128, 128, 127]

    with pytest.raises(campaign.CampaignError, match="counts do not match"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("wrong_prompt_kind", "phase cold_first_exact contract"),
        ("missing_component", "required trace components"),
        ("zero_target_mean", "target_verify_gpu_ms.mean"),
        ("wrong_target_count", "target_verify_gpu_ms count"),
        ("wrong_graph_key", "graph-key distribution"),
        ("wrong_trace_total", "trace/output token totals"),
        ("wrong_decode_rate", "decode rate does not reconcile"),
        ("ttft_exceeds_elapsed", "timing intervals do not reconcile"),
        ("negative_first_stream", "stream accounting"),
        ("stream_trace_mismatch", "trace cycles do not match stream events"),
        ("page_locality_eligible", "page-locality eligibility"),
    ],
)
def test_natural_stop_fallback_rejects_invalid_phase_or_trace_proof(
    mutation: str, error_match: str
) -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    phase = phases["cold_first_exact"]
    assert isinstance(phase, dict)
    trace = phase["trace"]
    assert isinstance(trace, dict)
    if mutation == "wrong_prompt_kind":
        phase["prompt_kind"] = "near"
    elif mutation == "missing_component":
        components = phase["trace_components"]
        assert isinstance(components, list)
        components.remove("target_verify_gpu_time")
    elif mutation == "zero_target_mean":
        target = trace["target_verify_gpu_ms"]
        assert isinstance(target, dict)
        target["mean"] = 0.0
    elif mutation == "wrong_target_count":
        target = trace["target_verify_gpu_ms"]
        assert isinstance(target, dict)
        target["count"] = int(trace["cycle_count"]) - 1
    elif mutation == "wrong_graph_key":
        trace["verify_graph_key_distribution"] = {"5": trace["cycle_count"]}
    elif mutation == "wrong_trace_total":
        trace["committed_tokens"] = int(trace["committed_tokens"]) + 2
    elif mutation == "wrong_decode_rate":
        benchmark = phase["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["decode_tokens_per_second"] = 1_000_000.0
    elif mutation == "ttft_exceeds_elapsed":
        benchmark = phase["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["time_to_first_token_seconds"] = 999.0
    elif mutation == "negative_first_stream":
        benchmark = phase["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["first_stream_completion_tokens"] = -500
    elif mutation == "stream_trace_mismatch":
        benchmark = phase["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["stream_events"] = int(trace["cycle_count"]) + 1
        benchmark["event_count"] = int(trace["cycle_count"]) + 1
    else:
        attribution = receipt["attribution"]
        assert isinstance(attribution, dict)
        page_locality = attribution["cpu_weight_page_cache_locality"]
        assert isinstance(page_locality, dict)
        page_locality["eligible"] = True

    with pytest.raises(campaign.CampaignError, match=error_match):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


def test_false_top_level_eligibility_requires_actual_trajectory_drift() -> None:
    receipt = _hotspot_receipt(40.0)
    receipt["performance_claim_eligible"] = False

    with pytest.raises(campaign.CampaignError, match="non-trajectory reason"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


@pytest.mark.parametrize("semantic_mutation", ["failed", "issues"])
def test_natural_stop_fallback_rejects_invalid_semantic_proof(
    semantic_mutation: str,
) -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    benchmark = phases["cold_first_exact"]["benchmark"]
    assert isinstance(benchmark, dict)
    semantic = benchmark["semantic_validation"]
    assert isinstance(semantic, dict)
    if semantic_mutation == "failed":
        semantic["passed"] = False
    else:
        semantic["issue_codes"] = ["repetition"]

    with pytest.raises(campaign.CampaignError, match="natural-stop semantics"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


def test_natural_stop_fallback_rejects_prompt_fingerprint_mismatch() -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    benchmark = phases["cold_first_exact"]["benchmark"]
    assert isinstance(benchmark, dict)
    benchmark["input_sha256"] = "0" * 64

    with pytest.raises(campaign.CampaignError, match="I/O proof"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


@pytest.mark.parametrize("phase_mutation", ["missing", "extra"])
def test_natural_stop_fallback_requires_exactly_five_phases(
    phase_mutation: str,
) -> None:
    receipt = _hotspot_receipt(40.0, performance_claim_eligible=False)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    if phase_mutation == "missing":
        del phases["cold_first_exact"]
    else:
        phases["unexpected"] = phases["cold_first_exact"]

    with pytest.raises(campaign.CampaignError, match="five fixed phases"):
        campaign.summarize_hotspot_receipts(
            [receipt], expected_input_tokens=2_694, expected_output_tokens=512
        )


def test_candidate_decision_rejects_shorter_completion_distribution(
    tmp_path: Path,
) -> None:
    baseline_summary = campaign.summarize_hotspot_receipts(
        [_hotspot_receipt(35.0)],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )
    candidate_summary = campaign.summarize_hotspot_receipts(
        [
            _hotspot_receipt(
                40.0,
                performance_claim_eligible=False,
                exact_completion_tokens=(110, 110, 110),
                near_completion_tokens=(110, 110),
            )
        ],
        expected_input_tokens=2_694,
        expected_output_tokens=512,
    )

    decision = campaign.candidate_decision(
        _candidate(tmp_path),
        candidate_summary,
        baseline_summary,
        _gates(tmp_path),
        confirmation=False,
    )

    assert "completion_length_ratio_outside_fairness_band" in decision["reasons"]


def test_hybrid_plan_inspection_binds_hash_and_variable_counts(
    tmp_path: Path,
) -> None:
    masks = torch.zeros((2, 43, 32), dtype=torch.bool)
    masks[0, :, :14] = True
    masks[1, :, 14:28] = True
    masks[0, 9, 28:32] = True
    plan_path = tmp_path / "variable.pt"
    torch.save(
        {
            "format": "sglang_kt_hybrid_expert_shard_v2_variable",
            "gpu_experts_mask_by_rank": masks,
            "placement_semantics_sha256": "a" * 64,
        },
        plan_path,
    )

    inspected = campaign.inspect_hybrid_plan(plan_path)

    assert inspected["sha256"] == campaign.sha256_file(plan_path)
    assert inspected["minimum_gpu_width"] == 14
    assert inspected["maximum_gpu_width"] == 18
    assert inspected["gpu_rank_counts_by_layer"][0][9] == 18


def test_server_info_must_publish_hash_bound_variable_counts(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    counts = [[14] * 43, [14] * 43]
    plan = {"sha256": "b" * 64, "gpu_rank_counts_by_layer": counts}
    info: dict[str, object] = {
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "kv_cache_dtype": "fp8_e4m3",
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
        "speculative_dspark_fixed_verify_len": 4,
        "speculative_num_draft_tokens": 6,
        "dsv4_sm86_small_batch_gemm_configured": True,
        "kt_draft_hybrid_expert_plan_sha256": "c" * 64,
        "kt_draft_hybrid_expert_plan_format": ("sglang_kt_hybrid_expert_shard_v1"),
        "kt_draft_hybrid_gpu_rank_counts_by_layer": [[14] * 3, [14] * 3],
        "kt_draft_hybrid_min_gpu_experts_per_rank_per_layer": 14,
        "kt_draft_hybrid_max_gpu_experts_per_rank_per_layer": 14,
        "kt_draft_hybrid_total_gpu_expert_layers_by_rank": [42, 42],
        "kt_draft_hybrid_source_profile_sha256": (
            "9100d1ef47685c50bc2eb3e47c34a62e39070103626479eeb398c2f5e43e4425"
        ),
        "kt_draft_hybrid_source_ordering_sha256": (
            "3f893065bf9cc3686a6a4926bf210efc4e7cc2b2035495262db4911f92f22432"
        ),
        "kt_draft_hybrid_gpu_selection_strategy": "profile-hot",
        "kt_hybrid_expert_plan_sha256": "b" * 64,
        "kt_hybrid_gpu_rank_counts_by_layer": counts,
        **_oscar_server_info(),
        **_small_batch_server_info(),
    }

    campaign.validate_server_info(info, candidate, plan, _oscar_server_info())
    info["dsv4_oscar_artifact_sha256"] = "8" * 64
    with pytest.raises(campaign.CampaignError, match="server-info contract mismatch"):
        campaign.validate_server_info(info, candidate, plan, _oscar_server_info())
    info["dsv4_oscar_artifact_sha256"] = "1" * 64
    info["kt_hybrid_gpu_rank_counts_by_layer"] = [[13] * 43, [14] * 43]
    with pytest.raises(campaign.CampaignError, match="counts"):
        campaign.validate_server_info(info, candidate, plan, _oscar_server_info())


def test_oscar_contract_derives_exact_model_bound_server_telemetry(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "oscar.pt"
    fingerprint = tmp_path / "checkpoint-fingerprint.json"
    admission = tmp_path / "admission.json"
    calibration.write_bytes(b"calibrated-oscar")
    fingerprint.write_text('{"checkpoint":"bound"}\n', encoding="utf-8")
    calibration_sha256 = campaign.sha256_file(calibration)
    fingerprint_sha256 = campaign.sha256_file(fingerprint)
    receipt: dict[str, object] = {
        "format": "dsv4-oscar-int2-admission",
        "format_version": 1,
        "admitted": True,
        "model_id": campaign.OSCAR_MODEL_ID,
        "artifact_path": str(calibration.resolve()),
        "artifact_file_sha256": calibration_sha256,
        "artifact_provenance_sha256": "1" * 64,
        "checkpoint_path": str(tmp_path.resolve()),
        "checkpoint_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "checkpoint_fingerprint_path": str(fingerprint.resolve()),
        "checkpoint_fingerprint_sha256": fingerprint_sha256,
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
    }
    receipt["admission_sha256"] = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    admission.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = campaign.OscarContract(
        calibration_artifact=campaign.Artifact(calibration, calibration_sha256),
        checkpoint_fingerprint=campaign.Artifact(fingerprint, fingerprint_sha256),
        admission_receipt=campaign.Artifact(admission, campaign.sha256_file(admission)),
        model_id=campaign.OSCAR_MODEL_ID,
    )

    proof = campaign.inspect_oscar_contract(contract)
    expected = proof["expected_server_info"]
    assert isinstance(expected, dict)
    assert expected["dsv4_oscar_artifact_sha256"] == calibration_sha256
    assert expected["dsv4_oscar_checkpoint_fingerprint_sha256"] == (fingerprint_sha256)
    assert expected["dsv4_oscar_admission_receipt_sha256"] == (
        campaign.sha256_file(admission)
    )
    assert expected["dsv4_int4_kv_storage"] is False
    assert expected["dsv4_c4_indexer_bytes_per_token"] == 40
    assert expected["dsv4_oscar_masked_writer_execution"] == (
        "device-uniform-live-mask-row-v1"
    )
    absorption = expected["dsv4_oscar_wo_a_absorption_state"]
    assert isinstance(absorption, dict)
    assert absorption["artifact_sha256"] == calibration_sha256
    assert absorption["admission_sha256"] == receipt["admission_sha256"]
    assert absorption["expected_local_compressed_layer_ids"] == list(range(2, 43))


def test_current_manifest_contains_only_oscar_compatible_candidates() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    manifest = campaign.load_manifest(manifest_path)

    assert manifest.fixed_environment["SGLANG_DSV4_OSCAR_INT2_KV_STORAGE"] == "1"
    assert manifest.fixed_environment["SGLANG_DSV4_INT4_KV_STORAGE"] == "0"
    assert manifest.fixed_environment["SGLANG_DSV4_INT4_C4_INDEXER_STORAGE"] == "0"
    assert manifest.fixed_environment["SGLANG_DSV4_SM86_C128_BF16_STORAGE"] == "0"
    assert manifest.fixed_environment["DSV4_CONTEXT_LENGTH"] == "524288"
    assert manifest.fixed_environment["DSV4_MAX_TOTAL_TOKENS"] == "524288"
    assert manifest.fixed_environment["DSV4_PREFILL_GRAPH_BACKEND"] == "breakable"
    assert manifest.fixed_environment["DSV4_DECODE_GRAPH_BACKEND"] == "full"
    assert manifest.fixed_environment["DSV4_DISABLE_SPECULATIVE"] == "0"
    assert manifest.fixed_environment["DSV4_TARGET_VERIFY_EAGER"] == "0"
    assert manifest.gates.baseline_repetitions == 3
    assert manifest.gates.screen_repetitions == 1
    assert manifest.gates.confirmation_repetitions == 3
    all_candidates = (manifest.baseline, *manifest.candidates)
    assert all("oscar-int2" in candidate.identifier for candidate in all_candidates)
    assert all(
        not (
            set(candidate.expected_server_info) & campaign.OSCAR_OWNED_SERVER_INFO_KEYS
        )
        for candidate in all_candidates
    )
    source_artifact_kinds = {
        artifact.path: artifact.kind for artifact in manifest.source_artifacts
    }
    repository_root = Path(__file__).resolve().parents[2]
    assert all(
        source_artifact_kinds[(repository_root / path).resolve()] == "file"
        for path in campaign.REQUIRED_LOCAL_LAUNCH_CHAIN
    )
    optimized = next(
        candidate
        for candidate in manifest.candidates
        if candidate.identifier == campaign.KT_CPU_OPTIMIZED_CANDIDATE_ID
    )
    assert optimized.priority == 10
    assert optimized.prerequisites == (manifest.baseline.identifier,)
    assert optimized.expected_plan == manifest.baseline.expected_plan
    assert optimized.expected_plan_sha256 == manifest.baseline.expected_plan_sha256
    assert {
        key: optimized.environment[key] for key in campaign.KT_CPU_OPTIMIZED_ENVIRONMENT
    } == campaign.KT_CPU_OPTIMIZED_ENVIRONMENT


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DSV4_STAGE_KT_AVX_TAIL_OVERLAY", "1"),
        ("DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY", "0"),
        ("DSV4_KT_CPU_OPTIMIZED_CANDIDATE", "/tmp/unbound.so"),
        ("DSV4_KT_CPU_OPTIMIZED_CACHE_ROOT", "/tmp/unbound-cache"),
        ("KT_MXFP4_AVX_SCALE_FOLD_MODE", "exponent-v1"),
        ("KT_SINGLE_NUMA_INLINE_DISPATCH", "0"),
        ("KT_TASK_QUEUE_PIN_FIRST_CORE", "0"),
    ],
)
def test_manifest_rejects_partial_cpu_optimized_tuple(
    tmp_path: Path, key: str, value: str
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    optimized = next(
        candidate
        for candidate in raw["candidates"]
        if candidate["id"] == campaign.KT_CPU_OPTIMIZED_CANDIDATE_ID
    )
    optimized["environment"][key] = value
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="CPU-optimized|scale-fold"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_runtime_n_block_override(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["controlled_environment_keys"].append("KT_MXFP4_N_BLOCK")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="compile-time provenance"):
        campaign.load_manifest(manifest_path)


def test_other_candidates_cannot_borrow_cpu_optimized_overrides(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    other = next(
        candidate
        for candidate in raw["candidates"]
        if candidate["id"] != campaign.KT_CPU_OPTIMIZED_CANDIDATE_ID
    )
    other["environment"]["DSV4_STAGE_KT_AVX_TAIL_OVERLAY"] = "0"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="only the exact g14"):
        campaign.load_manifest(manifest_path)


def test_current_manifest_schedules_optimized_g14_before_residency_and_blocks_verify(
    tmp_path: Path,
) -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    manifest = campaign.load_manifest(manifest_path)
    work_directory = tmp_path / "work"
    baseline_path = work_directory / manifest.baseline.identifier / "baseline.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(
            {
                "format": campaign.RESULT_FORMAT,
                "manifest_sha256": campaign.sha256_file(manifest.path),
                "qualified": True,
                "hotspot_receipts": [
                    _hotspot_receipt(34.42)
                    for _ in range(manifest.gates.baseline_repetitions)
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = campaign.summarize_campaign(manifest, work_directory)

    assert summary["next_action"] == {
        "candidate": campaign.KT_CPU_OPTIMIZED_CANDIDATE_ID,
        "stage": "screen",
    }
    rows = {row["id"]: row for row in summary["candidates"] if isinstance(row, dict)}
    assert rows["residency-k172-oscar-int2-verify5"]["status"] == (
        "waiting_prerequisites"
    )
    assert rows["residency-k172-oscar-int2-verify6"]["status"] == (
        "waiting_prerequisites"
    )


@pytest.mark.parametrize("missing_path", campaign.REQUIRED_LOCAL_LAUNCH_CHAIN)
def test_manifest_rejects_unbound_local_launch_chain(
    tmp_path: Path, missing_path: str
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["source_artifacts"] = [
        artifact
        for artifact in raw["source_artifacts"]
        if artifact["path"] != missing_path
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="complete local launch chain"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_alternate_launcher(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["launcher"] = "scripts/dsv4_flash_fwuff_parity.sh"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="OSCAR-only local OpenCode"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_candidate_owned_generic_cache_telemetry(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["candidates"][0]["expected_server_info"]["dsv4_kv_storage_mode"] = "fp8_e4m3"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="OSCAR-owned server telemetry"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_disabling_oscar_for_generic_int4(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["fixed_environment"]["SGLANG_DSV4_OSCAR_INT2_KV_STORAGE"] = "0"
    raw["fixed_environment"]["SGLANG_DSV4_INT4_KV_STORAGE"] = "1"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="serving contract"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_offline_rejected_72_thread_cpuinfer_candidate(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["candidates"][0]["environment"]["DSV4_CPUINFER_THREADS"] = "72"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="CPUInfer thread count"):
        campaign.load_manifest(manifest_path)


@pytest.mark.parametrize("thread_count", ["1", "55", "72", "73", "112"])
def test_manifest_rejects_unqualified_cpuinfer_candidate_thread_count(
    tmp_path: Path, thread_count: str
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["candidates"][0]["environment"]["DSV4_CPUINFER_THREADS"] = thread_count
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="CPUInfer thread count"):
        campaign.load_manifest(manifest_path)


def test_manifest_rejects_72_thread_baseline(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["baseline"]["environment"]["DSV4_CPUINFER_THREADS"] = "72"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="CPUInfer thread count"):
        campaign.load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_worker", "summary mismatch"),
        ("fallback", "worker contract mismatch"),
        ("wrong_rank", "distinct TP0/TP1"),
        ("wrong_shape", "unqualified signature"),
        ("missing_w2", "both W13 and W2"),
        ("count_drift", "do not reconcile"),
    ],
)
def test_small_batch_server_telemetry_fails_closed(mutation: str, match: str) -> None:
    info = _small_batch_server_info()
    workers = info["dsv4_sm86_small_batch_gemm_worker_telemetry"]
    assert isinstance(workers, list)
    assert all(isinstance(worker, dict) for worker in workers)
    if mutation == "missing_worker":
        info["dsv4_sm86_small_batch_gemm_reporting_worker_count"] = 1
        workers.pop()
    elif mutation == "fallback":
        workers[1]["patch_installed"] = False
    elif mutation == "wrong_rank":
        workers[1]["tp_rank"] = 0
    elif mutation == "wrong_shape":
        signatures = workers[1]["observed_signatures"]
        assert isinstance(signatures, list)
        assert isinstance(signatures[0], dict)
        signatures[0]["n"] = 8192
    elif mutation == "missing_w2":
        workers[1]["observed_signatures"] = [workers[1]["observed_signatures"][0]]
        workers[1]["selection_count"] = 2
    else:
        workers[1]["selection_count"] = 5

    with pytest.raises(campaign.CampaignError, match=match):
        campaign.validate_sm86_small_batch_server_telemetry(info)


def test_scale_fold_static_proof_matches_exact_inline_worker_pids() -> None:
    inline_info = _inline_server_info()
    inline_proof = campaign.validate_kt_single_numa_inline_dispatch_server_telemetry(
        inline_info
    )
    worker_pids = inline_proof["worker_pids"]
    assert isinstance(worker_pids, list)

    scale_proof = campaign.validate_mxfp4_avx_scale_fold_server_telemetry(
        _scale_fold_server_info(),
        inline_worker_pids=worker_pids,
    )

    assert scale_proof["worker_pids"] == [2000, 2001]
    assert scale_proof["kt_mxfp4_avx_scale_fold_requested_mode"] == "lut-v1"
    assert scale_proof["kt_mxfp4_avx_scale_fold_expected_n_block"] == 128


@pytest.mark.parametrize(
    "mutation",
    [
        "summary_mode",
        "n_block",
        "buffer_coverage",
        "unsafe_scale",
        "observed_range",
        "fallback",
        "wrong_pid",
        "duplicate_rank",
    ],
)
def test_scale_fold_static_proof_fails_closed(mutation: str) -> None:
    info = _scale_fold_server_info()
    workers = info["kt_mxfp4_avx_scale_fold_worker_telemetry"]
    assert isinstance(workers, list)
    assert all(isinstance(worker, dict) for worker in workers)
    first = workers[0]
    telemetry = first["telemetry"]
    assert isinstance(telemetry, dict)
    if mutation == "summary_mode":
        info["kt_mxfp4_avx_scale_fold_requested_mode"] = "exponent-v1"
    elif mutation == "n_block":
        telemetry["n_block"] = 64
    elif mutation == "buffer_coverage":
        telemetry["buffers_admitted"] = 257
    elif mutation == "unsafe_scale":
        telemetry["unsafe_scale_bytes"] = 1
    elif mutation == "observed_range":
        telemetry["observed_scale_maximum"] = 127
    elif mutation == "fallback":
        telemetry["fallback_dispatch_count"] = 1
    elif mutation == "wrong_pid":
        first["pid"] = 9999
    else:
        workers[1]["tp_rank"] = 0

    with pytest.raises(campaign.CampaignError, match="MXFP4 AVX scale-fold"):
        campaign.validate_mxfp4_avx_scale_fold_server_telemetry(
            info,
            inline_worker_pids=[2000, 2001],
        )


def test_scale_fold_dispatch_log_correlates_exact_two_worker_pids(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "[KT][MXFP4] first AVX-512 decode dispatch "
        "(pid=2001 m=1 n=2048 k=4096 group=32 thread=1/56 "
        "scale_fold=lut-v1 domain_finalized=1 n_block=128)\n"
        "[KT][MXFP4] first AVX-512 decode dispatch "
        "(pid=2000 m=1 n=4096 k=2048 group=32 thread=0/56 "
        "scale_fold=lut-v1 domain_finalized=1 n_block=128)\n",
        encoding="utf-8",
    )

    proof = campaign.validate_mxfp4_avx_scale_fold_dispatch_log(
        log_path,
        expected_worker_pids=[2000, 2001],
    )

    assert proof["record_count"] == 2
    assert proof["observed_worker_pids"] == [2000, 2001]


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "wrong_pid", "wrong_mode", "unfinalized", "n64"],
)
def test_scale_fold_dispatch_log_rejects_incomplete_or_wrong_proof(
    tmp_path: Path, mutation: str
) -> None:
    rank_zero = (
        "[KT][MXFP4] first AVX-512 decode dispatch "
        "(pid=2000 m=1 n=4096 k=2048 group=32 thread=0/56 "
        "scale_fold=lut-v1 domain_finalized=1 n_block=128)\n"
    )
    rank_one = rank_zero.replace("pid=2000", "pid=2001").replace(
        "thread=0/56", "thread=1/56"
    )
    if mutation == "missing":
        contents = rank_zero
    elif mutation == "duplicate":
        contents = rank_zero + rank_zero
    elif mutation == "wrong_pid":
        contents = rank_zero + rank_one.replace("pid=2001", "pid=2999")
    elif mutation == "wrong_mode":
        contents = rank_zero + rank_one.replace("lut-v1", "exponent-v1")
    elif mutation == "unfinalized":
        contents = rank_zero + rank_one.replace(
            "domain_finalized=1", "domain_finalized=0"
        )
    else:
        contents = rank_zero + rank_one.replace("n_block=128", "n_block=64")
    log_path = tmp_path / "server.log"
    log_path.write_text(contents, encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="MXFP4 AVX scale-fold"):
        campaign.validate_mxfp4_avx_scale_fold_dispatch_log(
            log_path,
            expected_worker_pids=[2000, 2001],
        )


def test_qualified_baseline_satisfies_candidate_prerequisite(tmp_path: Path) -> None:
    baseline = campaign.Candidate(
        identifier="baseline",
        priority=0,
        description="baseline",
        prerequisites=(),
        environment={},
        expected_server_info={},
        expected_plan=tmp_path / "baseline.pt",
        expected_plan_sha256=None,
        plan_materialized_at_launch=True,
        artifacts=(),
        policy="baseline",
        predicted={},
    )
    candidate = _candidate(tmp_path, prerequisites=("baseline",))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = campaign.CampaignManifest(
        path=manifest_path,
        launcher=tmp_path / "launcher.sh",
        launcher_sha256="0" * 64,
        fixed_environment={},
        controlled_environment_keys=frozenset(),
        oscar_contract=campaign.OscarContract(
            calibration_artifact=campaign.Artifact(tmp_path / "oscar.pt", "0" * 64),
            checkpoint_fingerprint=campaign.Artifact(
                tmp_path / "fingerprint.json", "0" * 64
            ),
            admission_receipt=campaign.Artifact(tmp_path / "admission.json", "0" * 64),
            model_id=campaign.OSCAR_MODEL_ID,
        ),
        source_artifacts=(),
        gates=_gates(tmp_path),
        baseline=baseline,
        candidates=(candidate,),
    )
    result_path = tmp_path / "work" / "baseline" / "baseline.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "format": campaign.RESULT_FORMAT,
                "manifest_sha256": campaign.sha256_file(manifest_path),
                "qualified": True,
                "hotspot_receipts": [_hotspot_receipt(36.0)],
            }
        ),
        encoding="utf-8",
    )

    summary = campaign.summarize_campaign(manifest, tmp_path / "work")

    assert summary["absolute_goal_contract"] == {
        "target_decode_tokens_per_second": 80.0,
        "stretch_decode_tokens_per_second": 90.0,
        "maximum_ttft_seconds": 7.0,
        "decode_metric": "five_phase_mean_decode_tokens_per_second",
        "ttft_metric": "maximum_flushed_ttft_seconds",
        "diagnostic_qualification_independent": True,
    }
    baseline_goals = summary["baseline_absolute_goal_state"]
    assert isinstance(baseline_goals, dict)
    assert baseline_goals["target_configuration_goal_met"] is False
    assert summary["candidates"][0]["status"] == "pending_screen"
    assert summary["next_action"] == {"candidate": "candidate", "stage": "screen"}


def test_endpoint_derivation_rejects_remote_host() -> None:
    with pytest.raises(campaign.CampaignError, match="local"):
        campaign.derive_endpoint("https://example.com/server_info", "/generate")


class _FakeProcessStatus:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


def test_server_wait_requires_post_warmup_health_before_server_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter((503, 503, 200))
    events: list[str] = []

    def fake_status(url: str, timeout_seconds: float) -> int:
        assert url == "http://127.0.0.1:30010/health"
        assert timeout_seconds > 0
        events.append("health")
        return next(statuses)

    def fake_server_info(url: str, timeout_seconds: float) -> dict[str, object]:
        assert url == "http://127.0.0.1:30010/server_info"
        assert timeout_seconds > 0
        events.append("server_info")
        return {"ready": True}

    monkeypatch.setattr(campaign, "_http_status", fake_status)
    monkeypatch.setattr(campaign.hotspot, "get_server_info", fake_server_info)
    monkeypatch.setattr(campaign.time, "sleep", lambda _seconds: None)

    info = campaign._wait_for_server(
        "http://127.0.0.1:30010/server_info",
        1.0,
        process=_FakeProcessStatus(),
    )

    assert info == {"ready": True}
    assert events == ["health", "health", "health", "server_info"]


def test_server_wait_fails_immediately_when_launcher_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(
        campaign,
        "_http_status",
        lambda *_args, **_kwargs: probes.append("health") or 200,
    )

    with pytest.raises(campaign.CampaignError, match="launcher exited before"):
        campaign._wait_for_server(
            "http://127.0.0.1:30010/server_info",
            1.0,
            process=_FakeProcessStatus(17),
        )

    assert probes == []


def test_small_batch_log_proves_both_fixed_tp_ep_ranks(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "[2026-08-03 TP1 EP1] V4 MXFP4 SM86 small-batch GEMM enabled: "
        "block_n=128 split_k=2 stages=4\n"
        "[2026-08-03 TP0 EP0] V4 MXFP4 SM86 small-batch GEMM enabled: "
        "block_n=128 split_k=2 stages=4\n",
        encoding="utf-8",
    )

    proof = campaign.validate_sm86_small_batch_log(log_path)

    assert proof["expected_tp_ep_ranks"] == [[0, 0], [1, 1]]


@pytest.mark.parametrize("mutation", ["missing_rank", "wrong_config", "duplicate"])
def test_small_batch_log_rejects_incomplete_or_ambiguous_proof(
    tmp_path: Path, mutation: str
) -> None:
    rank_zero = (
        "[2026-08-03 TP0 EP0] V4 MXFP4 SM86 small-batch GEMM enabled: "
        "block_n=128 split_k=2 stages=4\n"
    )
    rank_one = (
        "[2026-08-03 TP1 EP1] V4 MXFP4 SM86 small-batch GEMM enabled: "
        "block_n=128 split_k=2 stages=4\n"
    )
    if mutation == "missing_rank":
        contents = rank_zero
    elif mutation == "wrong_config":
        contents = rank_zero + rank_one.replace("split_k=2", "split_k=3")
    else:
        contents = rank_zero + rank_zero + rank_one
    log_path = tmp_path / "server.log"
    log_path.write_text(contents, encoding="utf-8")

    with pytest.raises(campaign.CampaignError, match="SM86 small-batch"):
        campaign.validate_sm86_small_batch_log(log_path)


def test_explicit_screen_can_select_later_independent_candidate(
    tmp_path: Path,
) -> None:
    first = _candidate(tmp_path, identifier="first", prerequisites=("baseline",))
    second = _candidate(tmp_path, identifier="second", prerequisites=("baseline",))
    manifest = _manifest_with_candidates(tmp_path, (first, second))
    work_directory = tmp_path / "work"
    _write_result(manifest, work_directory, "baseline", "baseline")

    selected = campaign.select_explicit_candidate_stage(
        manifest, work_directory, "second", "screen"
    )

    assert selected.identifier == "second"


def test_explicit_screen_refuses_to_overwrite_receipt(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, prerequisites=("baseline",))
    manifest = _manifest_with_candidates(tmp_path, (candidate,))
    work_directory = tmp_path / "work"
    _write_result(manifest, work_directory, "baseline", "baseline")
    _write_result(manifest, work_directory, candidate.identifier, "screen")

    with pytest.raises(campaign.CampaignError, match="refusing to overwrite"):
        campaign.select_explicit_candidate_stage(
            manifest, work_directory, candidate.identifier, "screen"
        )


def test_explicit_confirmation_requires_qualified_screen(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, prerequisites=("baseline",))
    manifest = _manifest_with_candidates(tmp_path, (candidate,))
    work_directory = tmp_path / "work"
    _write_result(manifest, work_directory, "baseline", "baseline")

    with pytest.raises(campaign.CampaignError, match="qualified screen"):
        campaign.select_explicit_candidate_stage(
            manifest, work_directory, candidate.identifier, "confirm"
        )


def test_explicit_screen_requires_confirmed_nonbaseline_prerequisite(
    tmp_path: Path,
) -> None:
    prerequisite = _candidate(
        tmp_path, identifier="prerequisite", prerequisites=("baseline",)
    )
    candidate = _candidate(
        tmp_path, identifier="dependent", prerequisites=("prerequisite",)
    )
    manifest = _manifest_with_candidates(tmp_path, (prerequisite, candidate))
    work_directory = tmp_path / "work"
    _write_result(manifest, work_directory, "baseline", "baseline")
    _write_result(manifest, work_directory, prerequisite.identifier, "screen")

    with pytest.raises(campaign.CampaignError, match="not confirmed"):
        campaign.select_explicit_candidate_stage(
            manifest, work_directory, candidate.identifier, "screen"
        )
