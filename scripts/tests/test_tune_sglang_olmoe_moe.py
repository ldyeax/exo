from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import tune_sglang_olmoe_moe as tuner


def _config(
    block_m: int,
    block_n: int = 64,
    block_k: int = 128,
    group_m: int = 1,
    warps: int = 4,
    stages: int = 2,
) -> tuner.KernelConfig:
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": warps,
        "num_stages": stages,
    }


def _timing(
    candidate: float,
    *,
    before: float = 100.0,
    after: float = 100.0,
) -> tuple[tuner.BracketTiming, ...]:
    return tuple(
        tuner.BracketTiming(
            route_seed=seed,
            candidate_microseconds=(candidate,) * tuner.DEFAULT_INDEPENDENT_SAMPLES,
            fallback_before_microseconds=(before,) * tuner.DEFAULT_INDEPENDENT_SAMPLES,
            fallback_after_microseconds=(after,) * tuner.DEFAULT_INDEPENDENT_SAMPLES,
        )
        for seed in tuner.OLMOE_ROUTE_SEEDS
    )


def _stage(
    stage: tuner.KernelStage,
    config: tuner.KernelConfig,
    rank_times: tuple[float, ...],
) -> tuner.StageMeasurement:
    include_ep2_cases = len(rank_times) == 2
    ranks = tuple(
        tuner.RankStageMeasurement(
            rank=rank,
            numerical_evidence=tuple(
                tuner.NumericalEvidence(
                    scenario=scenario,
                    rank=rank,
                    relative_l1=0.001,
                    max_absolute=0.002,
                    repeat_exact=True,
                    global_route_sha256="a" * 64,
                    local_route_sha256="b" * 64,
                )
                for scenario in (
                    tuple(
                        f"softmax_topk_seed_{seed}" for seed in tuner.OLMOE_ROUTE_SEEDS
                    )
                    + (("all_remote", "boundary_31_32") if include_ep2_cases else ())
                )
            ),
            route_timings=_timing(value),
        )
        for rank, value in enumerate(rank_times)
    )
    return tuner.StageMeasurement(
        stage=stage,
        config=config,
        ranks=ranks,
        minimum_stable_improvement=0.05,
    )


def _candidate(
    config: tuner.KernelConfig,
    *,
    gate_times: tuple[float, ...],
    down_times: tuple[float, ...],
) -> tuner.CandidateMeasurement:
    return tuner.CandidateMeasurement(
        config=config,
        gate_up=_stage("gate_up", config, gate_times),
        down=_stage("down", config, down_times),
    )


def _fallback_candidate(rank_count: int) -> tuner.CandidateMeasurement:
    fallback = cast(tuner.KernelConfig, dict(tuner._FALLBACK_CONFIG))
    return _candidate(
        fallback,
        gate_times=(100.0,) * rank_count,
        down_times=(100.0,) * rank_count,
    )


def _anchor_result(ep_size: int, batch_size: int) -> tuner.AnchorResult:
    rank_count = 1 if ep_size == 1 else 2
    fallback = _fallback_candidate(rank_count)
    assert fallback.gate_up is not None
    assert fallback.down is not None
    candidates = (fallback,) + tuple(
        _candidate(
            config,
            gate_times=(100.0,) * rank_count,
            down_times=(100.0,) * rank_count,
        )
        for config in tuner.build_rtx3090_search_space()[1:]
    )
    return tuner.AnchorResult(
        ep_size=cast(Any, ep_size),
        batch_size=batch_size,
        candidates=candidates,
        selected=tuner.SelectedPair(
            gate_up=fallback.gate_up,
            down=fallback.down,
            admitted_stable_win=False,
        ),
    )


def _all_results() -> tuple[tuner.AnchorResult, ...]:
    return tuple(
        _anchor_result(topology.ep_size, batch_size)
        for topology in tuner.OLMOE_TOPOLOGIES
        for batch_size in tuner.OLMOE_BATCH_ANCHORS
    )


def test_import_and_dry_run_do_not_require_cuda_runtime(tmp_path: Path) -> None:
    source = Path(tuner.__file__).resolve()
    program = f"""
import importlib.abc
import importlib.util
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {{"torch", "triton", "sglang"}} or fullname.startswith(("torch.", "triton.", "sglang.")):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockRuntime())
spec = importlib.util.spec_from_file_location("isolated_olmoe_tuner", {str(source)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
raise SystemExit(module.main(["--output-dir", {str(tmp_path / "unused")!r}, "--dry-run"]))
"""
    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["batch_anchors"] == [1, 8, 128, 1024]
    assert preview["model"]["normalize_top_k_weights"] is False
    assert len(preview["candidate_configs"]) == 6
    assert "torch" not in result.stderr


def test_exact_topology_shapes_and_config_names() -> None:
    ep1 = tuner.topology_for_ep_size(1)
    ep2 = tuner.topology_for_ep_size(2)

    assert ep1.gate_up_weight_shape == (64, 1024, 2048)
    assert ep1.down_weight_shape == (64, 2048, 512)
    assert ep2.gate_up_weight_shape == (32, 2048, 2048)
    assert ep2.down_weight_shape == (32, 2048, 1024)
    assert tuner.config_file_name(ep1, down=False) == (
        "E=64,N=512,device_name=NVIDIA_GeForce_RTX_3090.json"
    )
    assert tuner.config_file_name(ep2, down=True) == (
        "E=32,N=1024,device_name=NVIDIA_GeForce_RTX_3090_down.json"
    )

    with pytest.raises(ValueError, match="exact OLMoE shape"):
        tuner.OlmoeTopology(
            ep_size=2, local_experts=64, intermediate_size=512, rank_count=2
        )


def test_search_space_is_exact_bounded_and_keeps_fallback() -> None:
    observed = tuple(map(tuner._kernel_config_key, tuner.build_rtx3090_search_space()))

    assert observed == tuner._RTX3090_CANDIDATE_TUPLES
    assert len(set(observed)) == 6
    assert tuner.build_rtx3090_search_space()[0] == tuner._FALLBACK_CONFIG
    assert {
        config["BLOCK_SIZE_M"] for config in tuner.build_rtx3090_search_space()
    } == {
        16,
        64,
    }


def test_ep1_routes_are_unchanged_and_ep2_routes_are_rank_local() -> None:
    routes = ((0, 1, 30, 31, 32, 33, 62, 63),)

    assert (
        tuner.map_global_route_rows(
            routes, topology=tuner.topology_for_ep_size(1), rank=0
        )
        == routes
    )
    assert tuner.map_global_route_rows(
        routes, topology=tuner.topology_for_ep_size(2), rank=0
    ) == ((0, 1, 30, 31, -1, -1, -1, -1),)
    assert tuner.map_global_route_rows(
        routes, topology=tuner.topology_for_ep_size(2), rank=1
    ) == ((-1, -1, -1, -1, 0, 1, 30, 31),)


@pytest.mark.parametrize("rank", [0, 1])
def test_ep2_correctness_cases_cover_remote_and_31_32_boundary(rank: int) -> None:
    cases = tuner.build_ep2_correctness_cases(
        topology=tuner.topology_for_ep_size(2), rank=rank, batch_size=3
    )

    assert tuple(case.name for case in cases) == ("all_remote", "boundary_31_32")
    assert all(expert == -1 for row in cases[0].local_routes for expert in row)
    assert {31, 32} <= set(cases[1].global_routes[0])
    assert any(expert == -1 for expert in cases[1].local_routes[0])
    assert any(expert >= 0 for expert in cases[1].local_routes[0])
    assert all(sum(row) < 1 for case in cases for row in case.route_weights)


def test_stable_win_requires_five_percent_against_both_brackets() -> None:
    stable = tuner.BracketTiming(
        route_seed=1,
        candidate_microseconds=(90.0, 90.0, 90.0),
        fallback_before_microseconds=(100.0, 100.0, 100.0),
        fallback_after_microseconds=(96.0, 96.0, 96.0),
    )
    misses_after = tuner.BracketTiming(
        route_seed=1,
        candidate_microseconds=(92.0, 92.0, 92.0),
        fallback_before_microseconds=(100.0, 100.0, 100.0),
        fallback_after_microseconds=(96.0, 96.0, 96.0),
    )

    assert stable.is_stable_win(0.05)
    assert not misses_after.is_stable_win(0.05)


def test_rank_stage_rejects_missing_numerical_evidence() -> None:
    with pytest.raises(ValueError, match="numerical evidence"):
        tuner.RankStageMeasurement(
            rank=0,
            numerical_evidence=(),
            route_timings=_timing(90.0),
        )


def test_pair_selection_uses_cross_stage_configs_and_worst_ep2_rank() -> None:
    fallback = _fallback_candidate(2)
    gate_fast = _candidate(
        _config(16, 32, 64), gate_times=(60.0, 70.0), down_times=(88.0, 90.0)
    )
    down_fast = _candidate(
        _config(16, 64, 128), gate_times=(85.0, 86.0), down_times=(50.0, 65.0)
    )

    selected = tuner.select_admitted_pair((fallback, gate_fast, down_fast))

    assert selected.admitted_stable_win
    assert selected.gate_up.config == gate_fast.config
    assert selected.down.config == down_fast.config
    assert selected.gate_up.worst_rank_median_microseconds == 70.0
    assert selected.down.worst_rank_median_microseconds == 65.0


def test_pair_selection_retains_both_fallbacks_when_one_stage_lacks_win() -> None:
    fallback = _fallback_candidate(1)
    config = _config(16, 32, 64)
    gate = _stage("gate_up", config, (80.0,))
    down_rank = tuner.RankStageMeasurement(
        rank=0,
        numerical_evidence=_stage("down", config, (94.0,)).ranks[0].numerical_evidence,
        route_timings=tuple(
            tuner.BracketTiming(
                route_seed=seed,
                candidate_microseconds=(94.0, 94.0, 94.0),
                fallback_before_microseconds=(100.0, 100.0, 100.0),
                fallback_after_microseconds=(98.0, 98.0, 98.0),
            )
            for seed in tuner.OLMOE_ROUTE_SEEDS
        ),
    )
    candidate = tuner.CandidateMeasurement(
        config=config,
        gate_up=gate,
        down=tuner.StageMeasurement(
            stage="down",
            config=config,
            ranks=(down_rank,),
            minimum_stable_improvement=0.05,
        ),
    )

    selected = tuner.select_admitted_pair((fallback, candidate))

    assert not selected.admitted_stable_win
    assert selected.gate_up.config == tuner._FALLBACK_CONFIG
    assert selected.down.config == tuner._FALLBACK_CONFIG


def test_fused_kernel_signature_and_direct_call_contract() -> None:
    calls: list[dict[str, object]] = []

    def invoke(**kwargs: object) -> None:
        calls.append(kwargs)

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
    invoke.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    tuner._validate_fused_moe_kernel_signature(invoke)

    runtime = tuner.RuntimeBindings(
        torch=SimpleNamespace(),
        triton=SimpleNamespace(),
        triton_language=SimpleNamespace(bfloat16="tl.bfloat16"),
        invoke_fused_moe_kernel=invoke,
        moe_align_block_size=lambda *_args: (None, None, None),
        sglang_revision="a" * 40,
        ktransformers_revision="b" * 40,
        source_files={},
        source_modules={},
    )
    inputs = tuner.RuntimeInputs("hidden", "gate-weights", "down-weights")
    workload = tuner.RuntimeWorkload(
        scenario="test",
        route_seed=1,
        inputs=inputs,
        route_weights="route-weights",
        global_route_ids="global-ids",
        local_route_ids="local-ids",
        gate_up_reference="gate-reference",
        down_input="down-input",
        down_reference="down-reference",
        global_route_sha256="a" * 64,
        local_route_sha256="b" * 64,
    )
    config = _config(16)
    tuner._invoke_kernel(
        runtime,
        workload=workload,
        config=config,
        aligned_routes=("sorted", "experts", "padded"),
        activation="activation",
        weights="weights",
        output="output",
        mul_routed_weight=False,
        top_k=8,
    )

    assert tuple(calls[0]) == tuner._FUSED_MOE_PARAMETER_NAMES
    assert calls[0]["topk_ids"] == "local-ids"
    assert calls[0]["filter_expert"] is True
    assert calls[0]["mul_routed_weight"] is False
    assert calls[0]["top_k"] == 8
    assert calls[0]["compute_type"] == "tl.bfloat16"


def test_bundle_contains_four_canonical_configs_receipt_and_hash_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "olmoe-tuning"
    runtime_provenance = {
        "triton_version": "3.5.1",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_uuid": "GPU-test",
        "runtime_source_sha256": {"kernel": "f" * 64},
    }

    try:
        manifest = tuner.write_tuning_bundle(
            output_directory=output,
            spec=tuner.TuningSpec(),
            results=_all_results(),
            runtime_provenance=runtime_provenance,
            input_bindings={
                "runtime_install_receipt": {"sha256": "a" * 64},
                "model_config": {"sha256": "b" * 64},
            },
            tuner_path=Path(tuner.__file__),
            expected_tuner_sha256=hashlib.sha256(
                Path(tuner.__file__).read_bytes()
            ).hexdigest(),
        )

        version_directory = output / "configs" / "triton_3_5_1"
        config_names = sorted(path.name for path in version_directory.iterdir())
        assert config_names == [
            "E=32,N=1024,device_name=NVIDIA_GeForce_RTX_3090.json",
            "E=32,N=1024,device_name=NVIDIA_GeForce_RTX_3090_down.json",
            "E=64,N=512,device_name=NVIDIA_GeForce_RTX_3090.json",
            "E=64,N=512,device_name=NVIDIA_GeForce_RTX_3090_down.json",
        ]
        for path in version_directory.iterdir():
            parsed_config = json.loads(path.read_text())
            assert list(parsed_config) == ["1", "8", "128", "1024"]
            assert parsed_config == {
                str(anchor): tuner._FALLBACK_CONFIG
                for anchor in tuner.OLMOE_BATCH_ANCHORS
            }
            assert path.stat().st_mode & 0o222 == 0

        receipt = json.loads((output / "tuning-receipt.json").read_text())
        assert receipt["candidate"] is True
        assert receipt["deployment_admitted"] is False
        assert receipt["model"]["normalize_top_k_weights"] is False
        assert receipt["runtime_provenance"] == runtime_provenance
        assert len(receipt["anchors"]) == 8
        assert receipt["topologies"][0]["gate_up_weight_shape"] == [64, 1024, 2048]
        assert receipt["topologies"][1]["down_weight_shape"] == [32, 2048, 1024]

        disk_manifest = json.loads((output / "manifest.json").read_text())
        assert disk_manifest == manifest
        assert len(disk_manifest["files"]) == 5
        for relative_path, evidence in disk_manifest["files"].items():
            contents = (output / relative_path).read_bytes()
            assert evidence["sha256"] == hashlib.sha256(contents).hexdigest()
            assert evidence["size_bytes"] == len(contents)
        assert output.stat().st_mode & 0o222 == 0
    finally:
        if output.exists():
            os.chmod(output, 0o755)
            configs = output / "configs"
            version = configs / "triton_3_5_1"
            if configs.exists():
                os.chmod(configs, 0o755)
            if version.exists():
                os.chmod(version, 0o755)


def test_bundle_refuses_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        tuner.write_tuning_bundle(
            output_directory=output,
            spec=tuner.TuningSpec(),
            results=_all_results(),
            runtime_provenance={},
            input_bindings={},
            tuner_path=Path(tuner.__file__),
            expected_tuner_sha256=hashlib.sha256(
                Path(tuner.__file__).read_bytes()
            ).hexdigest(),
        )


def test_bundle_rejects_timing_sample_count_outside_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample count"):
        tuner.write_tuning_bundle(
            output_directory=tmp_path / "unused",
            spec=tuner.TuningSpec(independent_samples=7),
            results=_all_results(),
            runtime_provenance={},
            input_bindings={
                "runtime_install_receipt": {"sha256": "a" * 64},
                "model_config": {"sha256": "b" * 64},
            },
            tuner_path=Path(tuner.__file__),
            expected_tuner_sha256=hashlib.sha256(
                Path(tuner.__file__).read_bytes()
            ).hexdigest(),
        )


def test_live_input_bindings_authenticate_receipt_and_exact_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_id = "1" * 64
    install_root = tmp_path / install_id
    install_root.mkdir()
    runtime_python = install_root / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"test-python")
    site_packages = install_root / "venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(tuner.sys, "executable", str(runtime_python))
    runtime_receipt = install_root / "install-receipt.json"
    runtime_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "install_complete",
                "install_id": install_id,
                "layout": {
                    "install_root": str(install_root),
                    "python": str(runtime_python),
                    "receipt": str(runtime_receipt),
                    "site_packages": str(site_packages),
                },
                "build": {
                    "sglang_revision": "a" * 40,
                    "ktransformers_revision": "b" * 40,
                },
            }
        )
    )
    model_config = tmp_path / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "architectures": ["OlmoeForCausalLM"],
                "model_type": "olmoe",
                "hidden_size": 2048,
                "intermediate_size": 1024,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "norm_topk_prob": False,
                "torch_dtype": "bfloat16",
            }
        )
    )

    bindings = tuner.validate_live_input_bindings(
        runtime_install_receipt=runtime_receipt,
        runtime_install_receipt_sha256=hashlib.sha256(
            runtime_receipt.read_bytes()
        ).hexdigest(),
        model_config=model_config,
        model_config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
    )

    assert bindings["runtime_install_receipt"]["schema_version"] == 1  # type: ignore[index]
    assert bindings["model_config"]["validated_fields"] == {  # type: ignore[index]
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "norm_topk_prob": False,
        "torch_dtype": "bfloat16",
    }

    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        tuner.validate_live_input_bindings(
            runtime_install_receipt=runtime_receipt,
            runtime_install_receipt_sha256="0" * 64,
            model_config=model_config,
            model_config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
        )

    model_symlink = tmp_path / "config-link.json"
    model_symlink.symlink_to(model_config)
    with pytest.raises(RuntimeError, match="cannot open bound OLMoE model config"):
        tuner.validate_live_input_bindings(
            runtime_install_receipt=runtime_receipt,
            runtime_install_receipt_sha256=hashlib.sha256(
                runtime_receipt.read_bytes()
            ).hexdigest(),
            model_config=model_symlink,
            model_config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
        )


def test_live_mode_requires_all_provenance_bindings_before_cuda(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        tuner.main(["--output-dir", str(tmp_path / "unused")])

    assert raised.value.code == 2


def test_default_numeric_tolerances_cover_one_bfloat16_step() -> None:
    spec = tuner.TuningSpec()

    assert spec.relative_l1_tolerance == 0.02
    assert spec.max_absolute_tolerance == 0.04


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_anchors": (1, 8)}, "batch anchors"),
        ({"route_seeds": (1, 2)}, "exactly three"),
        ({"independent_samples": 2}, "at least three"),
        ({"minimum_stable_improvement": 0.0}, "in \\(0, 1\\)"),
    ],
)
def test_spec_rejects_noncanonical_or_unstable_contracts(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tuner.TuningSpec(**overrides)  # type: ignore[arg-type]
