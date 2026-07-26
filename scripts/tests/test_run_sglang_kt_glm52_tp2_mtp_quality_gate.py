from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from scripts import run_sglang_kt_glm52_tp2_local_benchmark as benchmark
from scripts import run_sglang_kt_glm52_tp2_mtp_quality_gate as gate


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: Literal[False],
        add_generation_prompt: Literal[True],
        enable_thinking: Literal[False],
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return f"USER:{conversation[0]['content']}\nASSISTANT:"

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: Literal[False],
    ) -> list[int]:
        del add_special_tokens
        raw = text.encode()
        return [
            100 + sum(raw[offset : offset + 4]) % 100_000
            for offset in range(0, len(raw), 4)
        ]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: Literal[True],
        clean_up_tokenization_spaces: Literal[False],
    ) -> str:
        del token_ids, skip_special_tokens, clean_up_tokenization_spaces
        return ""


def make_gate_config(tmp_path: Path) -> gate.QualityGateConfig:
    arguments = gate._parser().parse_args(
        (
            "--run-id",
            "matched-quality-test",
            "--result-directory",
            str(tmp_path / "quality"),
        )
    )
    return gate._config_from_arguments(arguments)


def test_default_gate_is_exact_known_marlin_c1_profile(tmp_path: Path) -> None:
    config = make_gate_config(tmp_path)
    launch = config.benchmark_config

    assert launch.runtime_python == gate.DEFAULT_RUNTIME_PYTHON
    assert launch.runtime_install_receipt_sha256 == (
        gate.DEFAULT_RUNTIME_INSTALL_RECEIPT_SHA256
    )
    assert launch.local_source_directory == "/tmp/kvb-marlin-src"
    assert launch.model_path == "/mnt/sanic/glm52-AMXINT4-W8A16-hybrid"
    assert launch.ktransformers_weight_path == "/mnt/sanic/glm52-AMXINT4"
    assert launch.benchmark_concurrencies == (1,)
    assert launch.maximum_total_tokens == 896
    assert launch.kv_cache_dtype == "bfloat16"
    assert launch.mla_kv_b_w8_backend == "marlin"
    assert launch.enable_shared_host_weights is True
    assert launch.enable_two_batch_overlap is False
    assert launch.enable_stream_prefill is False
    assert launch.enable_mtp is False
    assert launch.resident_gpu_expert_budget_total == 0
    assert config.profile == gate.quality.THIN_FIRST_RUN_PROFILE


def test_quality_mode_specs_use_disjoint_ports_and_attest_original_logprobs(
    tmp_path: Path,
) -> None:
    config = make_gate_config(tmp_path)
    off_config = gate._mode_config(config, enable_mtp=False)
    on_config = gate._mode_config(config, enable_mtp=True)
    off = gate.build_quality_process_spec(
        off_config,
        enable_mtp=False,
    )
    on = gate.build_quality_process_spec(
        on_config,
        enable_mtp=True,
    )

    assert (off_config.distributed_port, off_config.service_port) == (62700, 62710)
    assert (on_config.distributed_port, on_config.service_port) == (62720, 62730)
    assert {
        off.distributed_coordinator.port,
        off.service_endpoint.port,
    }.isdisjoint(
        {
            on.distributed_coordinator.port,
            on.service_endpoint.port,
        }
    )
    assert dict(off.environment)["SGLANG_MLA_KV_B_W8_BACKEND"] == "marlin"
    assert dict(off.environment)["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "1"
    assert off.maximum_running_requests == 1
    assert off.kv_cache_dtype == "bfloat16"
    assert "--enable-two-batch-overlap" not in off.command
    assert off.command[off.command.index("--load-format") + 1] == "safetensors"
    assert off.command[off.command.index("--random-seed") + 1] == str(
        gate.quality.DEFAULT_SAMPLING_SEED
    )
    assert "--disable-overlap-schedule" in off.command
    assert "--kt-stream-prefill" not in off.command
    assert "--speculative-algorithm" not in off.command
    assert on.command[on.command.index("--load-format") + 1] == "safetensors"
    assert on.command[on.command.index("--random-seed") + 1] == str(
        gate.quality.DEFAULT_SAMPLING_SEED
    )
    assert "--disable-overlap-schedule" in on.command
    assert on.command[on.command.index("--speculative-algorithm") + 1] == "NEXTN"
    assert on.command[on.command.index("--speculative-num-steps") + 1] == "1"
    assert on.command[on.command.index("--speculative-eagle-topk") + 1] == "1"
    assert on.command[on.command.index("--speculative-num-draft-tokens") + 1] == "2"

    receipt = on.receipt()
    assert receipt["service_endpoint"] == {
        "ip": on.service_endpoint.ip,
        "port": 62730,
    }
    assert receipt["distributed_coordinator"] == {
        "ip": on.distributed_coordinator.ip,
        "port": 62720,
    }
    assert receipt["quality_logprob_contract"] == {
        "return_original_logprob": True,
        "environment_variable": "SGLANG_RETURN_ORIGINAL_LOGPROB",
        "environment_value": "1",
    }
    assert receipt["quality_execution_contract"] == {
        "server_random_seed": gate.quality.DEFAULT_SAMPLING_SEED,
        "disable_overlap_schedule": True,
        "load_format": "safetensors",
    }


def test_mode_contract_derives_only_declared_mtp_launch_intent() -> None:
    verified: benchmark.JsonObject = {
        "compact_mla_kv_b_w8": True,
        "hybrid_checkpoint_manifest": {
            "content_id": "a" * 64,
        },
        "runtime": {"install_receipt_sha256": "b" * 64},
        "mtp_policy": {
            "causal_layer_range": [0, benchmark.MODEL_LAYER_COUNT],
            "layer_78_is_nextn_predict": True,
            "layer_78_loaded": True,
            "layer_78_experts": "persistent_amxint4",
            "speculative_decoding_enabled": True,
        },
    }
    original = deepcopy(verified)

    off = gate._mode_runtime_contract(verified, enable_mtp=False)
    on = gate._mode_runtime_contract(verified, enable_mtp=True)

    assert verified == original
    off_policy = cast(benchmark.JsonObject, off["mtp_policy"])
    on_policy = cast(benchmark.JsonObject, on["mtp_policy"])
    assert off_policy["layer_78_loaded"] is False
    assert off_policy["layer_78_experts"] == "not_loaded"
    assert off_policy["speculative_decoding_enabled"] is False
    assert on_policy == original["mtp_policy"]
    for key in ("runtime", "hybrid_checkpoint_manifest"):
        assert off[key] == verified[key]
        assert on[key] == verified[key]


def test_gate_rejects_non_marlin_or_tbo_profile(tmp_path: Path) -> None:
    base = make_gate_config(tmp_path).benchmark_config
    invalid = replace(base, enable_two_batch_overlap=True)

    with pytest.raises(ValueError, match="quality gate requires"):
        gate.QualityGateConfig(
            benchmark_config=invalid,
            profile=gate.quality.THIN_FIRST_RUN_PROFILE,
            thresholds=gate.quality.QualityThresholds(),
        )


@pytest.mark.parametrize(
    ("distributed_port", "service_port"),
    (
        (62_700, 62_720),
        (65_520, 65_000),
    ),
)
def test_gate_rejects_colliding_or_out_of_range_mode_endpoint_plan(
    tmp_path: Path,
    distributed_port: int,
    service_port: int,
) -> None:
    valid = make_gate_config(tmp_path)
    invalid = replace(
        valid.benchmark_config,
        distributed_port=distributed_port,
        service_port=service_port,
    )

    with pytest.raises(ValueError, match="distinct, disjoint, in-range"):
        gate.QualityGateConfig(
            benchmark_config=invalid,
            profile=valid.profile,
            thresholds=valid.thresholds,
        )


def test_core_adapter_tokenizes_embedded_thin_workload_and_marlin_census(
    tmp_path: Path,
) -> None:
    config = make_gate_config(tmp_path)
    cases, workload = gate._tokenize_quality_cases(FakeTokenizer(), config)
    runtime_contract: benchmark.JsonObject = {
        "runtime": {
            "install_receipt_sha256": "a" * 64,
        },
        "checkpoints": [{"role": "persistent_w8a16_gpu_model"}],
        "hybrid_checkpoint_manifest": {
            "content_id": "b" * 64,
            "mla_kv_b_w8_backend": "marlin",
        },
        "shared_host_weights": {
            "enabled": True,
            "content_id": "c" * 64,
        },
    }
    run_contract = gate._quality_run_contract(
        config,
        runtime_contract,
        enable_mtp=False,
        workload_receipt=workload,
    )
    attestation: benchmark.JsonObject = {
        "passed": True,
        "requested_backend": "marlin",
        "expected_runtime_backend": "marlin",
        "observations": [
            {
                "tensor_parallel_rank": rank,
                "backend": "marlin",
                "module_count": 78,
                "local_heads_per_module": 32,
            }
            for rank in range(2)
        ],
    }
    census = gate._marlin_census(attestation, enable_mtp=False)

    assert len(cases) == 10
    assert tuple(item.case_id for item in cases) == config.profile.case_ids
    assert cases[-1].teacher_forced_target_kind == "reference_completion"
    assert workload["humaneval_reference_content_sha256"] == (
        gate.quality.HUMANEVAL_REFERENCE_CONTENT_SHA256
    )
    assert run_contract.mtp.enabled is False
    assert run_contract.runtime_controls.configured_backend == "marlin"
    assert census.mtp_enabled is False
    assert tuple(item.module_count for item in census.observations) == (78, 78)


def _gpu_snapshot(
    observation: int,
    processes: tuple[benchmark.GpuComputeProcess, ...],
) -> benchmark.GpuCapacitySnapshot:
    devices = tuple(
        benchmark.GpuMemoryDevice(
            uuid=gpu_uuid,
            index=index,
            pci_bus_id=f"00000000:{index:02x}:00.0",
            compute_capability="8.6",
            total_mib=24_576,
            free_mib=24_000,
            used_mib=124,
        )
        for index, gpu_uuid in enumerate(benchmark.ORDERED_GPU_UUIDS)
    )
    return benchmark.GpuCapacitySnapshot(
        observed_at_utc=f"2026-07-25T22:31:{observation:02d}+00:00",
        inventory_command=("nvidia-smi",),
        inventory_stdout_sha256=f"{observation + 1:064x}",
        compute_process_command=("nvidia-smi",),
        compute_process_stdout_sha256=f"{observation + 101:064x}",
        devices=devices,
        compute_processes=processes,
    )


def test_inter_mode_gpu_owner_wait_records_transient_owner_then_quiescence() -> None:
    owner = benchmark.GpuComputeProcess(
        gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
        pid=3_616_928,
        used_memory_mib=12_082,
    )
    snapshots = iter(
        (
            _gpu_snapshot(0, (owner,)),
            _gpu_snapshot(1, ()),
            _gpu_snapshot(2, ()),
        )
    )
    now = [0.0]

    receipt = gate.wait_for_pinned_gpu_owner_quiescence(
        benchmark.ORDERED_GPU_UUIDS,
        timeout_seconds=1.0,
        poll_seconds=0.25,
        snapshot_collector=lambda: next(snapshots),
        monotonic_clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert receipt["status"] == "passed"
    assert receipt["observation_count"] == 3
    observations = cast(list[benchmark.JsonValue], receipt["observations"])
    first = cast(benchmark.JsonObject, observations[0])
    final = cast(benchmark.JsonObject, observations[2])
    assert cast(list[benchmark.JsonValue], first["pinned_compute_owners"])[0] == {
        "gpu_uuid": benchmark.ORDERED_GPU_UUIDS[0],
        "pid": 3_616_928,
        "used_memory_mib": 12_082,
    }
    assert final["pinned_compute_owners"] == []
    assert receipt["elapsed_seconds"] == 0.5


def test_inter_mode_gpu_owner_wait_times_out_without_ignoring_owner() -> None:
    owner = benchmark.GpuComputeProcess(
        gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
        pid=1234,
        used_memory_mib=64,
    )
    now = [0.0]

    receipt = gate.wait_for_pinned_gpu_owner_quiescence(
        benchmark.ORDERED_GPU_UUIDS,
        timeout_seconds=0.5,
        poll_seconds=0.25,
        snapshot_collector=lambda: _gpu_snapshot(0, (owner,)),
        monotonic_clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert receipt["status"] == "failed"
    assert receipt["failure_reason"] == "timeout_with_pinned_compute_owners"
    assert receipt["observation_count"] == 3
    assert receipt["elapsed_seconds"] == 0.5
