from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from scripts import tune_sglang_kt_glm47_moe as tuner


def _spec(**overrides: object) -> tuner.Glm47MoeTuningSpec:
    values: dict[str, object] = {
        "hidden_size": 2_048,
        "intermediate_size": 1_536,
        "resident_experts": 4,
        "global_experts": 64,
        "top_k": 4,
        "batch_sizes": (1, 8),
        "seed": 20_260_719,
        "warmup_iterations": 2,
        "measurement_iterations": 4,
        "independent_samples": 3,
        "search_profile": "quick",
        "relative_l1_tolerance": 0.02,
        "max_absolute_tolerance": 0.02,
    }
    values.update(overrides)
    return tuner.Glm47MoeTuningSpec(**values)  # type: ignore[arg-type]


def _config(
    *,
    block_size_m: int,
    block_size_n: int = 64,
    block_size_k: int = 128,
) -> tuner.KernelConfig:
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def _scenario_evidence() -> tuple[tuner.ScenarioNumericalEvidence, ...]:
    evidence = tuner.NumericalEvidence(
        relative_l1=0.001,
        max_absolute=0.002,
        repeat_exact=True,
    )
    return tuple(
        tuner.ScenarioNumericalEvidence(
            scenario=cast(tuner.RouteScenarioName, scenario),
            evidence=evidence,
        )
        for scenario in ("uniform", "zero_resident", "mixed", "resident_skew")
    )


def _measurement(
    *,
    stage: tuner.KernelStage,
    config: tuner.KernelConfig,
    samples: tuple[float, ...] = (90.0, 90.0, 90.0),
    fallback_before: tuple[float, ...] = (100.0, 100.0, 100.0),
    fallback_after: tuple[float, ...] = (100.0, 100.0, 100.0),
) -> tuner.StageMeasurement:
    timing_strata = tuple(
        tuner.StratumTimingEvidence(
            name=f"resident_count_{resident_count}",
            resident_route_count=resident_count,
            resident_routes_per_token=resident_count,
            probability_weight=probability_weight,
            sample_microseconds=samples,
            fallback_before_microseconds=fallback_before,
            fallback_after_microseconds=fallback_after,
            numerical_evidence=tuner.NumericalEvidence(
                relative_l1=0.001,
                max_absolute=0.002,
                repeat_exact=True,
            ),
        )
        for resident_count, probability_weight in ((0, 0.75), (1, 0.25))
    )
    return tuner.StageMeasurement(
        stage=stage,
        config=config,
        sample_microseconds=samples,
        fallback_before_microseconds=fallback_before,
        fallback_after_microseconds=fallback_after,
        numerical_evidence=_scenario_evidence(),
        timing_strata=timing_strata,
    )


def _fallback_measurement(stage: tuner.KernelStage) -> tuner.StageMeasurement:
    return _measurement(
        stage=stage,
        config=cast(tuner.KernelConfig, dict(tuner._FALLBACK_CONFIG)),
        samples=(100.0, 100.0, 100.0),
    )


def _result(
    spec: tuner.Glm47MoeTuningSpec, batch_size: int
) -> tuner.AnchorTuningResult:
    gate_up = _fallback_measurement("gate_up")
    down = _fallback_measurement("down")
    fallback = cast(tuner.KernelConfig, dict(tuner._FALLBACK_CONFIG))
    return tuner.AnchorTuningResult(
        batch_size=batch_size,
        route_scenarios=tuner.build_route_scenarios(spec, batch_size),
        timing_route_strata=tuner.build_timing_route_strata(spec, batch_size),
        selected_pair=tuner.CandidatePair(
            block_size_m=fallback["BLOCK_SIZE_M"],
            gate_up=gate_up,
            down=down,
        ),
        candidate_records=(
            tuner.CandidateRecord(
                config=fallback,
                measurements=(gate_up, down),
                rejections=(),
            ),
        ),
    )


def _runtime_contract() -> dict[str, object]:
    sha256 = "e" * 64
    return {
        "schema_version": 1,
        "install_id": "1" * 64,
        "install_root": "/immutable/runtime",
        "runtime_python": {
            "path": "/immutable/runtime/bin/python",
            "sha256": sha256,
            "symlink_chain": [],
        },
        "base_runtime_python_sha256": sha256,
        "base_runtime_pip_freeze_sha256": sha256,
        "installed_distribution_record_sha256": {
            "kt-kernel": sha256,
            "ktransformers": sha256,
            "sglang-kt": sha256,
        },
        "installed_distribution_file_counts": {
            "kt-kernel": 10,
            "ktransformers": 10,
            "sglang-kt": 10,
        },
        "build_id": "2" * 64,
        "sglang_revision": "a" * 40,
        "ktransformers_revision": "b" * 40,
        "package_version": "0.0.test",
        "runtime_wheel_sha256": {
            "kt-kernel": sha256,
            "ktransformers": sha256,
            "sglang-kt": sha256,
        },
        "fused_moe_distribution_sha256": sha256,
        "kt_extension_sha256": sha256,
        "embedded_provenance_sha256": {
            "kt-kernel": sha256,
            "sglang-kt": sha256,
        },
        "torch_version": "2.9.1+cu128",
        "triton_version": "3.5.1",
        "torch_cuda_version": "12.8",
        "gpu_uuid": "GPU-test",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "compute_capability": [8, 6],
        "capabilities": ["kt_bf16_amx_executed_v1"],
        "receipt_bindings": {
            "runtime_install": {
                "path": "/receipts/install.json",
                "sha256": sha256,
            },
            "runtime_build": {
                "path": "/receipts/build.json",
                "sha256": sha256,
            },
            "kernel_validation": {
                "path": "/receipts/kernel.json",
                "sha256": sha256,
            },
        },
    }


def _authorization(
    tuner_sha256: str, output_directory_descriptor: int
) -> dict[str, object]:
    output_status = os.fstat(output_directory_descriptor)
    return {
        "schema_version": 1,
        "authorization_sha256": "a" * 64,
        "lease_id": "lease-test",
        "run_id": "run-test",
        "namespace": "exo-test",
        "receipt_sha256": {
            "runtime_python": "e" * 64,
            "tuner_script": tuner_sha256,
            "source_identity": "c" * 64,
            "runtime_install": "e" * 64,
            "runtime_build": "e" * 64,
            "kernel_validation": "e" * 64,
        },
        "runtime_contract": _runtime_contract(),
        "output_directory_descriptor": output_directory_descriptor,
        "output_directory_identity": {
            "device": output_status.st_dev,
            "inode": output_status.st_ino,
        },
        "output_empty_at_authorization": True,
    }


def _write_bundle(
    output_directory: Path,
    spec: tuner.Glm47MoeTuningSpec,
    results: Sequence[tuner.AnchorTuningResult],
) -> tuner.Mapping[str, object]:
    output_directory.mkdir()
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        return _write_bundle_to_descriptor(
            output_directory,
            output_descriptor,
            spec,
            results,
        )
    finally:
        os.close(output_descriptor)


def _write_bundle_to_descriptor(
    output_directory: Path,
    output_directory_descriptor: int,
    spec: tuner.Glm47MoeTuningSpec,
    results: Sequence[tuner.AnchorTuningResult],
    *,
    authorization_evidence: dict[str, object] | None = None,
) -> tuner.Mapping[str, object]:
    tuner_path = Path(tuner.__file__).resolve()
    tuner_sha256 = hashlib.sha256(tuner_path.read_bytes()).hexdigest()
    return tuner.write_tuning_bundle(
        output_directory=output_directory,
        output_directory_descriptor=output_directory_descriptor,
        spec=spec,
        triton_version="3.5.1",
        torch_version="2.9.1+cu128",
        cuda_version="12.8",
        sglang_revision="a" * 40,
        ktransformers_revision="b" * 40,
        device_name="NVIDIA GeForce RTX 3090",
        gpu_uuid="GPU-test",
        results=results,
        authorization_evidence=(
            authorization_evidence
            or _authorization(tuner_sha256, output_directory_descriptor)
        ),
        tuner_path=tuner_path,
        tuner_sha256=tuner_sha256,
    )


def _live_arguments(output_directory: Path) -> dict[str, object]:
    return {
        "spec": _spec(),
        "output_directory": output_directory,
    }


def test_import_does_not_require_torch_triton_or_sglang() -> None:
    source = Path(tuner.__file__).resolve()
    repository = source.parents[1]
    program = f"""
import importlib.abc
import importlib.util
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {{"torch", "triton", "sglang"}} or fullname.startswith(("torch.", "triton.", "sglang.")):
            raise ModuleNotFoundError(f"{{fullname}} is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, {str(repository)!r})
spec = importlib.util.spec_from_file_location("isolated_glm47_tuner", {str(source)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert "triton" not in sys.modules
assert "sglang" not in sys.modules
"""
    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_fused_moe_signature_validation_binds_full_parameter_contract() -> None:
    def kernel(**_kwargs: object) -> None:
        return None

    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=tuner._FUSED_MOE_OPTIONAL_DEFAULTS.get(
                name, inspect.Parameter.empty
            ),
        )
        for name in tuner._FUSED_MOE_PARAMETER_NAMES
    ]
    kernel.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    tuner._validate_fused_moe_kernel_signature(kernel)

    drifted = list(parameters)
    drifted[-1] = drifted[-1].replace(default=False)
    kernel.__signature__ = inspect.Signature(drifted)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="changed its default"):
        tuner._validate_fused_moe_kernel_signature(kernel)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], name: str) -> None:
        self.shape = shape
        self.name = name
        self.copied_from: FakeTensor | None = None
        self.last_slice: object = None

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def __getitem__(self, key: object) -> FakeTensor:
        self.last_slice = key
        return self

    def copy_(self, source: FakeTensor) -> FakeTensor:
        self.copied_from = source
        return self


class FakeTorch:
    bfloat16 = object()

    def __init__(self) -> None:
        self.created: list[FakeTensor] = []

    def zeros(
        self, shape: tuple[int, ...], *, device: str, dtype: object
    ) -> FakeTensor:
        assert device == "cuda"
        assert dtype is self.bfloat16
        tensor = FakeTensor(shape, f"zeros-{len(self.created)}")
        self.created.append(tensor)
        return tensor


def test_prepared_gate_and_down_calls_bind_full_fused_moe_kwargs() -> None:
    fake_torch = FakeTorch()
    calls: list[dict[str, object]] = []
    align_calls: list[tuple[object, int, int]] = []
    sorted_token_ids = FakeTensor((16,), "sorted")
    expert_ids = FakeTensor((4,), "experts")
    padded_count = FakeTensor((1,), "padded-count")

    def invoke(**kwargs: object) -> None:
        calls.append(kwargs)

    def align(
        route_ids: object, block_size_m: int, resident_experts: int
    ) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
        align_calls.append((route_ids, block_size_m, resident_experts))
        return sorted_token_ids, expert_ids, padded_count

    runtime = tuner.RuntimeBindings(
        torch=fake_torch,
        triton=SimpleNamespace(),
        triton_language=SimpleNamespace(bfloat16="tl.bfloat16"),
        invoke_fused_moe_kernel=invoke,
        moe_align_block_size=align,
        sglang_revision="a" * 40,
        ktransformers_revision="b" * 40,
    )
    spec = _spec(hidden_size=8, intermediate_size=4, batch_sizes=(2,))
    hidden_states = FakeTensor((2, 8), "hidden")
    gate_up_weights = FakeTensor((4, 8, 8), "gate-up-weights")
    down_weights = FakeTensor((4, 8, 4), "down-weights")
    route_weights = FakeTensor((2, 4), "route-weights")
    masked_route_ids = FakeTensor((2, 4), "masked-route-ids")
    gate_reference = FakeTensor((8, 8), "gate-reference")
    down_input = FakeTensor((8, 4), "down-input")
    down_reference = FakeTensor((8, 8), "down-reference")
    workload = tuner.RuntimeWorkload(
        scenario="uniform",
        hidden_states=hidden_states,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        route_weights=route_weights,
        global_route_ids=FakeTensor((2, 4), "global-route-ids"),
        masked_route_ids=masked_route_ids,
        gate_up_reference=gate_reference,
        down_input=down_input,
        down_reference=down_reference,
    )
    config = _config(block_size_m=16)

    gate = tuner._prepare_stage_operation(
        runtime, spec=spec, workload=workload, config=config, stage="gate_up"
    )
    gate.operation()
    down = tuner._prepare_stage_operation(
        runtime, spec=spec, workload=workload, config=config, stage="down"
    )
    down.operation()

    assert align_calls == [
        (masked_route_ids, 16, 4),
        (masked_route_ids, 16, 4),
    ]
    assert gate.output.shape == (16, 8)
    assert gate.expected is gate_reference
    assert gate.route_count == 8
    assert down.output.shape == (2, 4, 8)
    assert down.expected is down_reference
    assert down.route_count == 8
    padded_down_input = fake_torch.created[1]
    assert padded_down_input.shape == (16, 4)
    assert padded_down_input.copied_from is down_input

    assert len(calls) == 2
    gate_call, down_call = calls
    assert tuple(gate_call) == tuner._FUSED_MOE_PARAMETER_NAMES
    assert tuple(down_call) == tuner._FUSED_MOE_PARAMETER_NAMES
    shared = {
        "bias": None,
        "A_scale": None,
        "B_scale": None,
        "B_zp": None,
        "topk_weights": route_weights,
        "topk_ids": masked_route_ids,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": padded_count,
        "config": config,
        "compute_type": "tl.bfloat16",
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_channel_quant": False,
        "block_shape": None,
        "no_combine": False,
        "a_use_tma": False,
        "b_use_tma": False,
        "c_sorted": False,
        "filter_expert": True,
    }
    assert gate_call == {
        "A": hidden_states,
        "B": gate_up_weights,
        "C": gate.output,
        "mul_routed_weight": False,
        "top_k": 4,
        **shared,
    }
    assert down_call == {
        "A": padded_down_input,
        "B": down_weights,
        "C": down.output,
        "mul_routed_weight": True,
        "top_k": 1,
        **shared,
    }


@pytest.mark.parametrize(
    ("resident_experts", "expected_mean", "tolerance"),
    [(4, 0.25, 0.02), (1, 0.0625, 0.01)],
)
def test_uniform_routes_match_global_top_k_distribution_at_4096(
    resident_experts: int,
    expected_mean: float,
    tolerance: float,
) -> None:
    spec = _spec(resident_experts=resident_experts, batch_sizes=(4_096,))

    uniform = tuner.build_route_scenarios(spec, 4_096)[0]

    assert uniform.name == "uniform"
    assert uniform.resident_routes_per_token == pytest.approx(
        expected_mean, abs=tolerance
    )
    assert any(
        all(expert_id == -1 for expert_id in row) for row in uniform.masked_routes
    )
    assert all(len(set(row)) == spec.top_k for row in uniform.global_routes)
    assert uniform.global_routes == tuner.build_global_routes(spec, 4_096)


@pytest.mark.parametrize("resident_experts", [1, 4])
def test_zero_mixed_and_skewed_route_scenarios(resident_experts: int) -> None:
    spec = _spec(resident_experts=resident_experts)
    scenarios = {
        scenario.name: scenario for scenario in tuner.build_route_scenarios(spec, 8)
    }

    assert scenarios["zero_resident"].resident_route_count == 0
    assert scenarios["zero_resident"].masked_cpu_route_count == 8 * spec.top_k
    assert scenarios["mixed"].resident_route_count == 8
    assert scenarios["mixed"].masked_cpu_route_count == 8 * (spec.top_k - 1)
    expected_skew = 8 * min(resident_experts, spec.top_k)
    assert scenarios["resident_skew"].resident_route_count == expected_skew
    assert scenarios["resident_skew"].masked_cpu_route_count == (
        8 * spec.top_k - expected_skew
    )


@pytest.mark.parametrize(
    ("resident_experts", "expected_mean"),
    [(1, 0.0625), (4, 0.25)],
)
def test_batch_one_timing_strata_include_weighted_resident_gemm(
    resident_experts: int,
    expected_mean: float,
) -> None:
    spec = _spec(resident_experts=resident_experts, batch_sizes=(1,))

    strata = tuner.build_timing_route_strata(spec, 1)

    assert len(strata) == 2
    assert strata[0].resident_route_count == 0
    assert any(stratum.resident_route_count > 0 for stratum in strata)
    assert sum(stratum.probability_weight for stratum in strata) == pytest.approx(1)
    assert sum(
        stratum.resident_routes_per_token * stratum.probability_weight
        for stratum in strata
    ) == pytest.approx(expected_mean)
    for stratum in strata:
        assert len(stratum.global_routes) == 1
        assert (
            sum(expert_id >= 0 for expert_id in stratum.masked_routes[0])
            == stratum.resident_route_count
        )


@pytest.mark.parametrize("resident_experts", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1_024, 4_096])
def test_timing_strata_preserve_expected_total_resident_routes(
    resident_experts: int,
    batch_size: int,
) -> None:
    spec = _spec(resident_experts=resident_experts, batch_sizes=(batch_size,))

    strata = tuner.build_timing_route_strata(spec, batch_size)

    expected_total = (
        batch_size * spec.top_k * spec.resident_experts / spec.global_experts
    )
    assert len(strata) <= 2
    assert sum(
        stratum.resident_route_count * stratum.probability_weight for stratum in strata
    ) == pytest.approx(expected_total)
    for stratum in strata:
        masked_count = sum(
            expert_id >= 0
            for token_routes in stratum.masked_routes
            for expert_id in token_routes
        )
        assert masked_count == stratum.resident_route_count
        assert stratum.resident_routes_per_token == pytest.approx(
            stratum.resident_route_count / batch_size
        )


def test_route_hashes_are_deterministic_and_bind_masking() -> None:
    spec = _spec()
    routes = tuner.build_global_routes(spec, 8)
    masked = tuner.mask_global_routes(
        routes,
        resident_experts=spec.resident_experts,
        global_experts=spec.global_experts,
    )

    assert routes == tuner.build_global_routes(spec, 8)
    assert tuner.route_sha256(routes) == tuner.route_sha256(routes)
    assert tuner.route_sha256(routes) != tuner.route_sha256(masked)


def test_mask_global_routes_rejects_invalid_and_ragged_routes() -> None:
    with pytest.raises(ValueError, match="out of range"):
        tuner.mask_global_routes(((0, 64),), resident_experts=4, global_experts=64)
    with pytest.raises(ValueError, match="rectangular"):
        tuner.mask_global_routes(((0, 4), (1,)), resident_experts=4, global_experts=64)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resident_experts": 64}, "smaller than global_experts"),
        ({"top_k": 65}, "top_k cannot exceed"),
        ({"resident_experts": 63}, "enough non-resident experts"),
        ({"batch_sizes": (8, 1)}, "strictly increasing"),
        ({"independent_samples": 2}, "at least three"),
    ],
)
def test_spec_rejects_unsafe_workloads(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**overrides)


def test_default_anchor_contract_includes_4096() -> None:
    assert tuner.GLM47_BATCH_ANCHORS == (1, 8, 32, 128, 512, 1_024, 4_096)


def test_search_space_is_bounded_canonical_and_contains_runtime_fallback() -> None:
    quick = tuner.build_rtx3090_search_space("quick")
    balanced = tuner.build_rtx3090_search_space("balanced")

    assert len(quick) == 64
    assert len(balanced) == 192
    assert len({tuple(config.values()) for config in quick}) == len(quick)
    assert tuner._FALLBACK_CONFIG in quick
    assert set(map(tuple, (config.values() for config in quick))) < set(
        map(tuple, (config.values() for config in balanced))
    )
    assert tuple(map(tuner._kernel_config_key, quick)) == tuple(
        sorted(map(tuner._kernel_config_key, quick))
    )


def test_stable_pair_is_selected_with_shared_block_size_m() -> None:
    fallback_gate = _fallback_measurement("gate_up")
    fallback_down = _fallback_measurement("down")
    candidate_gate = _measurement(stage="gate_up", config=_config(block_size_m=32))
    candidate_down = _measurement(
        stage="down", config=_config(block_size_m=32, block_size_n=128)
    )

    selected = tuner.select_admitted_pair(
        (fallback_gate, candidate_gate),
        (fallback_down, candidate_down),
    )

    assert selected.block_size_m == 32
    assert selected.gate_up is candidate_gate
    assert selected.down is candidate_down
    assert candidate_gate.relative_improvement_samples == (0.1, 0.1, 0.1)


def test_fallback_is_retained_when_one_candidate_sample_misses_five_percent() -> None:
    fallback_gate = _fallback_measurement("gate_up")
    fallback_down = _fallback_measurement("down")
    unstable_gate = _measurement(
        stage="gate_up",
        config=_config(block_size_m=32),
        samples=(90.0, 96.0, 90.0),
    )
    stable_down = _measurement(
        stage="down", config=_config(block_size_m=32, block_size_n=128)
    )

    selected = tuner.select_admitted_pair(
        (fallback_gate, unstable_gate),
        (fallback_down, stable_down),
    )

    assert selected.gate_up.config == tuner._FALLBACK_CONFIG
    assert selected.down.config == tuner._FALLBACK_CONFIG


def test_nearest_kernel_lookup_covers_ties_and_4096() -> None:
    configs = {
        1: _config(block_size_m=16),
        128: _config(block_size_m=32),
        4_096: _config(block_size_m=64),
    }

    assert tuner.nearest_kernel_config(configs, 64) is configs[1]
    assert tuner.nearest_kernel_config(configs, 2_112) is configs[128]
    assert tuner.nearest_kernel_config(configs, 4_096) is configs[4_096]
    assert tuner.nearest_kernel_config(configs, 8_192) is configs[4_096]


def test_config_file_names_match_pinned_sglang_bf16_lookup() -> None:
    common = {
        "resident_experts": 4,
        "intermediate_size": 1_536,
        "device_name": "NVIDIA GeForce RTX 3090",
    }

    assert tuner.config_file_name(**common, down=False) == (
        "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090.json"
    )
    assert tuner.config_file_name(**common, down=True) == (
        "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090_down.json"
    )


def test_write_tuning_bundle_creates_stage_only_candidate_receipt(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "glm47-e4"
    spec = _spec()
    results = tuple(_result(spec, batch_size) for batch_size in spec.batch_sizes)

    manifest = _write_bundle(output_directory, spec, results)

    config_directory = output_directory / "configs" / "triton_3_5_1"
    gate_path = config_directory / (
        "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090.json"
    )
    down_path = config_directory / (
        "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090_down.json"
    )
    gate_contents = gate_path.read_bytes()
    down_contents = down_path.read_bytes()
    disk_manifest = json.loads((output_directory / "manifest.json").read_text())

    assert json.loads(gate_contents) == {
        "1": tuner._FALLBACK_CONFIG,
        "8": tuner._FALLBACK_CONFIG,
    }
    assert json.loads(down_contents) == {
        "1": tuner._FALLBACK_CONFIG,
        "8": tuner._FALLBACK_CONFIG,
    }
    assert disk_manifest == manifest
    assert disk_manifest["candidate"] is True
    assert disk_manifest["deployment_admitted"] is False
    assert disk_manifest["production_path_reproduced"] is False
    assert disk_manifest["limitations"] == [
        "alignment is prepared outside timed regions",
        "the pinned filtered activation and final reduction are not timed",
        "concurrent CPU AMX expert execution and resource contention are not reproduced",
        "resident-route strata preserve the uniform expected count but are synthetic",
        "stratum weights assume uniform global top-k routing, not a captured trace",
        "the untuned serving baseline profile does not consume this bundle",
        "the current serving_baseline profile strips SGLANG_* variables",
    ]
    assert disk_manifest["serving_admission_required"] == {
        "baseline_profile_consumes_this_bundle": False,
        "baseline_profile_is_untuned": True,
        "baseline_target_profile": ("GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE"),
        "consumer_chain": "KTEP gpu_method.apply -> SGLang fused_moe config loader",
        "current_serving_baseline_strips_sglang_environment": True,
        "metric": "matched end-to-end serving performance",
        "minimum_improvement": 0.03,
        "profile": "future tuned profile with exact config SHA-256 binding",
        "required_environment_binding": (
            "SGLANG_MOE_CONFIG_DIR with exact config file SHA-256"
        ),
    }
    assert disk_manifest["workload"] == {
        "resident_experts": 4,
        "batch_sizes": [1, 8],
        "global_experts": 64,
        "top_k": 4,
    }
    first_anchor = disk_manifest["anchors"][0]
    assert len(first_anchor["route_scenarios"]) == 4
    assert len(first_anchor["timing_route_strata"]) == 2
    assert first_anchor["timing_route_strata"][0]["resident_route_count"] == 0
    assert first_anchor["timing_route_strata"][0]["resident_routes_per_token"] == 0
    assert len(first_anchor["candidate_records"]) == 1
    assert len(first_anchor["candidate_records"][0]["measurements"]) == 2
    assert (
        len(
            first_anchor["candidate_records"][0]["measurements"][0][
                "numerical_evidence"
            ]
        )
        == 4
    )
    assert (
        len(first_anchor["candidate_records"][0]["measurements"][0]["timing_strata"])
        == 2
    )
    assert disk_manifest["authorization"]["authorization_sha256"] == "a" * 64
    assert disk_manifest["runtime_contract"] == _runtime_contract()
    assert disk_manifest["config_files"]["gate_up"]["shape"] == {
        "E": 4,
        "N": 1_536,
    }
    assert disk_manifest["config_files"]["gate_up"]["batch_keys"] == [1, 8]
    assert disk_manifest["config_files"]["gate_up"]["sha256"] == (
        hashlib.sha256(gate_contents).hexdigest()
    )
    assert disk_manifest["config_files"]["down"]["sha256"] == (
        hashlib.sha256(down_contents).hexdigest()
    )


def test_write_tuning_bundle_rejects_duplicate_batches_before_bundle_write(
    tmp_path: Path,
) -> None:
    spec = _spec()
    duplicate = _result(spec, 1)
    output_directory = tmp_path / "duplicate"

    with pytest.raises(ValueError, match="duplicate batch sizes"):
        _write_bundle(output_directory, spec, (duplicate, duplicate))

    assert output_directory.is_dir()
    assert list(output_directory.iterdir()) == []


def test_write_tuning_bundle_requires_exact_anchor_coverage(tmp_path: Path) -> None:
    spec = _spec()
    output_directory = tmp_path / "missing"

    with pytest.raises(ValueError, match="exactly cover"):
        _write_bundle(output_directory, spec, (_result(spec, 1),))

    assert output_directory.is_dir()
    assert list(output_directory.iterdir()) == []


def test_write_tuning_bundle_rejects_nonempty_authorized_descriptor(
    tmp_path: Path,
) -> None:
    spec = _spec()
    output_directory = tmp_path / "nonempty"
    output_directory.mkdir()
    (output_directory / "unexpected").write_text("occupied", encoding="ascii")
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )

    try:
        with pytest.raises(FileExistsError, match="empty directory"):
            _write_bundle_to_descriptor(
                output_directory,
                output_descriptor,
                spec,
                tuple(_result(spec, size) for size in spec.batch_sizes),
            )
    finally:
        os.close(output_descriptor)

    assert {path.name for path in output_directory.iterdir()} == {"unexpected"}


def test_descriptor_writer_is_contained_after_output_path_replacement(
    tmp_path: Path,
) -> None:
    spec = _spec()
    output_directory = tmp_path / "authorized"
    moved_directory = tmp_path / "authorized-moved"
    output_directory.mkdir()
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    authorization = _authorization(
        hashlib.sha256(Path(tuner.__file__).read_bytes()).hexdigest(),
        output_descriptor,
    )
    output_directory.rename(moved_directory)
    output_directory.mkdir()

    try:
        _write_bundle_to_descriptor(
            output_directory,
            output_descriptor,
            spec,
            tuple(_result(spec, size) for size in spec.batch_sizes),
            authorization_evidence=authorization,
        )
    finally:
        os.close(output_descriptor)

    assert list(output_directory.iterdir()) == []
    assert (moved_directory / "manifest.json").is_file()
    assert (moved_directory / "configs" / "triton_3_5_1").is_dir()


def test_live_tuning_rejects_missing_authorization_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(tuner.__file__).resolve().parent))
    monkeypatch.delenv("EXO_GLM47_TUNER_AUTHORIZATION_FD", raising=False)
    monkeypatch.delenv("EXO_GLM47_TUNER_RESULT_DIRECTORY_FD", raising=False)

    def forbidden_runtime() -> NoReturn:
        raise AssertionError("CUDA runtime must not load")

    monkeypatch.setattr(tuner, "load_runtime_bindings", forbidden_runtime)
    with pytest.raises(RuntimeError, match="was not inherited"):
        tuner.run_live_tuning(**_live_arguments(tmp_path / "unauthorized"))  # type: ignore[arg-type]


def test_live_tuning_rejects_authorized_digest_mismatch_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_harness = types.ModuleType("run_sglang_kt_glm47_moe_tuning")
    output_directory = tmp_path / "mismatch"
    output_directory.mkdir()
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )

    def authorize(*, output_directory: Path) -> dict[str, object]:
        assert output_directory == tmp_path / "mismatch"
        return _authorization("0" * 64, output_descriptor)

    fake_harness.validate_tuner_authorization = authorize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "run_sglang_kt_glm47_moe_tuning", fake_harness)

    def forbidden_runtime() -> NoReturn:
        raise AssertionError("CUDA runtime must not load")

    monkeypatch.setattr(tuner, "load_runtime_bindings", forbidden_runtime)
    try:
        with pytest.raises(RuntimeError, match="authorized tuner digest differs"):
            tuner.run_live_tuning(**_live_arguments(output_directory))  # type: ignore[arg-type]
    finally:
        os.close(output_descriptor)


def test_live_tuning_rejects_incomplete_runtime_provenance_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_harness = types.ModuleType("run_sglang_kt_glm47_moe_tuning")
    output_directory = tmp_path / "bad-provenance"
    output_directory.mkdir()
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    tuner_sha256 = hashlib.sha256(Path(tuner.__file__).read_bytes()).hexdigest()
    authorization = _authorization(tuner_sha256, output_descriptor)
    receipt_sha256 = cast(dict[str, object], authorization["receipt_sha256"])
    del receipt_sha256["kernel_validation"]

    def authorize(*, output_directory: Path) -> dict[str, object]:
        assert output_directory == tmp_path / "bad-provenance"
        return authorization

    fake_harness.validate_tuner_authorization = authorize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "run_sglang_kt_glm47_moe_tuning", fake_harness)

    def forbidden_runtime() -> NoReturn:
        raise AssertionError("CUDA runtime must not load")

    monkeypatch.setattr(tuner, "load_runtime_bindings", forbidden_runtime)
    try:
        with pytest.raises(RuntimeError, match="kernel_validation receipt"):
            tuner.run_live_tuning(**_live_arguments(output_directory))  # type: ignore[arg-type]
    finally:
        os.close(output_descriptor)


def test_runtime_contract_rejects_missing_authenticated_field() -> None:
    authorization = _authorization("0" * 64, 0)
    contract = cast(dict[str, object], authorization["runtime_contract"])
    del contract["installed_distribution_file_counts"]

    with pytest.raises(RuntimeError, match="runtime contract keys differ"):
        tuner._validated_runtime_contract(authorization)


def test_runtime_contract_rejects_unexpected_authenticated_field() -> None:
    authorization = _authorization("0" * 64, 0)
    contract = cast(dict[str, object], authorization["runtime_contract"])
    contract["mutable_override"] = "not authenticated by the schema"

    with pytest.raises(RuntimeError, match="runtime contract keys differ"):
        tuner._validated_runtime_contract(authorization)


def test_live_tuning_rejects_output_descriptor_identity_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_harness = types.ModuleType("run_sglang_kt_glm47_moe_tuning")
    output_directory = tmp_path / "bad-identity"
    output_directory.mkdir()
    output_descriptor = os.open(
        output_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    tuner_sha256 = hashlib.sha256(Path(tuner.__file__).read_bytes()).hexdigest()
    authorization = _authorization(tuner_sha256, output_descriptor)
    identity = cast(dict[str, object], authorization["output_directory_identity"])
    identity["inode"] = cast(int, identity["inode"]) + 1

    def authorize(*, output_directory: Path) -> dict[str, object]:
        assert output_directory == tmp_path / "bad-identity"
        return authorization

    fake_harness.validate_tuner_authorization = authorize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "run_sglang_kt_glm47_moe_tuning", fake_harness)

    def forbidden_runtime() -> NoReturn:
        raise AssertionError("CUDA runtime must not load")

    monkeypatch.setattr(tuner, "load_runtime_bindings", forbidden_runtime)
    try:
        with pytest.raises(RuntimeError, match="descriptor identity changed"):
            tuner.run_live_tuning(**_live_arguments(output_directory))  # type: ignore[arg-type]
    finally:
        os.close(output_descriptor)


def test_tune_anchor_does_not_swallow_arbitrary_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OutOfResourcesError(RuntimeError):
        pass

    runtime = SimpleNamespace(
        triton=SimpleNamespace(
            runtime=SimpleNamespace(
                autotuner=SimpleNamespace(OutOfResources=OutOfResourcesError)
            )
        )
    )
    monkeypatch.setattr(
        tuner,
        "_create_runtime_workload",
        lambda runtime, spec, batch_size, scenario: SimpleNamespace(
            scenario=scenario.name
        ),
    )

    def fail_stage(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("unexpected CUDA launch failure")

    monkeypatch.setattr(tuner, "_validate_and_measure_stage", fail_stage)
    with pytest.raises(RuntimeError, match="unexpected CUDA launch failure"):
        tuner.tune_anchor(
            runtime=cast(tuner.RuntimeBindings, runtime),
            spec=_spec(batch_sizes=(1,)),
            batch_size=1,
            search_space=(cast(tuner.KernelConfig, dict(tuner._FALLBACK_CONFIG)),),
        )


def test_dry_run_is_offline_and_does_not_create_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_directory = tmp_path / "unused"

    exit_code = tuner.main(
        (
            "--output-dir",
            str(output_directory),
            "--resident-experts",
            "4",
            "--dry-run",
        )
    )

    preview = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert preview["spec"]["batch_sizes"][-1] == 4_096
    assert preview["spec"]["uniform_expected_resident_routes_per_token"] == 0.25
    assert preview["spec"]["route_scenarios"] == [
        "uniform",
        "zero_resident",
        "mixed",
        "resident_skew",
    ]
    assert not output_directory.exists()
