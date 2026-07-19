import hashlib
import importlib
import importlib.machinery
import json
import pathlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, cast

import pytest
from pydantic import ValidationError

from exo.shared.types.common import Host, ModelId
from exo.shared.types.compute_resources import NvidiaGpuComputeResource
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    NetworkPort,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_process_launch_specs,
    build_glm_5_2_fp8_process_launch_specs,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtHostPreflightObservation,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtPreflightPassed,
    SglangKtPythonVersionObservation,
    SglangKtRuntimeObservation,
    SglangKtRuntimeValidationReceiptObservation,
    evaluate_sglang_kt_preflight,
)
from exo.worker.sglang_kt.preflight_collector import (
    SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE,
    ExternalPythonSglangKtRuntimeProbe,
    LinuxSglangKtHostInventoryProbe,
    LocalSglangKtFilesystemProbe,
    SglangKtLocalHostInventory,
    SglangKtModelSnapshotCompatibility,
    SglangKtRuntimeCommandResult,
    SocketSglangKtPortProbe,
    calculate_sglang_kt_artifact_build_id,
    collect_sglang_kt_local_host_preflight_observation,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_plan,
    make_plan,
)

CONFIG_SHA256 = "a" * 64
FULL_INDEXER_LAYER_STARTS = (0, 1, 2, *range(6, 78, 4))
TORCH_VERSION = "2.10.0+cu130"
CUDA_VERSION = "13.0"
SGL_KERNEL_BUILD_ID = "sgl-kernel-test-build"
DEEP_GEMM_BUILD_ID = "deep-gemm-test-build"
KT_KERNEL_BUILD_ID = "kt-kernel-test-build"


def make_specs() -> tuple[SglangKtProcessLaunchSpec, ...]:
    return build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)


def make_glm_5_2_fp8_config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
        "index_topk_freq": 4,
        "index_topk_pattern": None,
        "index_skip_topk_offset": 3,
        "index_share_for_mtp_iteration": True,
        "indexer_types": tuple(
            "full" if index in FULL_INDEXER_LAYER_STARTS else "shared"
            for index in range(78)
        ),
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.embed_tokens"],
        },
        # Snapshot metadata is intentionally not part of the runtime pin check.
        "transformers_version": "999.0.0",
        "unrelated_checkpoint_metadata": {"ignored": True},
    }


def make_glm_4_7_flash_bf16_config() -> dict[str, object]:
    return {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "pad_token_id": 154820,
        "eos_token_id": [154820, 154827, 154829],
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 10240,
        "max_position_embeddings": 202752,
        "model_type": "glm4_moe_lite",
        "moe_intermediate_size": 1536,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "num_attention_heads": 20,
        "n_group": 1,
        "topk_group": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.8,
        "num_experts_per_tok": 4,
        "first_k_dense_replace": 1,
        "num_hidden_layers": 47,
        "num_key_value_heads": 20,
        "num_nextn_predict_layers": 1,
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "transformers_version": "5.0.0rc0",
        "q_lora_rank": 768,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "vocab_size": 154880,
    }


def write_model_config(path: Path, config: dict[str, object]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def write_exact_glm_4_7_flash_config(path: Path) -> None:
    path.mkdir()
    config_json = json.dumps(make_glm_4_7_flash_bf16_config(), indent=2) + "\n"
    (path / "config.json").write_text(config_json, encoding="utf-8")


def observe_default_glm_4_7_flash_bf16_compatibility(
    path: Path,
    *,
    revision: GitRevision = GLM_4_7_FLASH_BF16_MODEL_REVISION,
) -> SglangKtModelSnapshotReceiptObservation | None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_completeness_checker=(lambda _path, _model_id, _revision: True)
    )
    return probe.observe_model_snapshot(
        str(path),
        spec.model_id,
        revision,
    )


def observe_default_glm_5_2_fp8_compatibility(
    path: Path,
) -> SglangKtModelSnapshotReceiptObservation | None:
    spec = make_specs()[0]
    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_completeness_checker=(lambda _path, _model_id, _revision: True)
    )
    return probe.observe_model_snapshot(
        str(path),
        spec.model_id,
        spec.expected_model_revision,
    )


def make_runtime(spec: SglangKtProcessLaunchSpec) -> SglangKtRuntimeObservation:
    return SglangKtRuntimeObservation(
        executable=spec.executable,
        python_implementation="CPython",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=13,
            patch=7,
        ),
        sglang_revision=spec.expected_sglang_revision,
        ktransformers_revision=spec.expected_ktransformers_revision,
        transformers_distribution_version=spec.required_transformers_version,
        transformers_module_version=spec.required_transformers_version,
        torch_version=TORCH_VERSION,
        cuda_version=CUDA_VERSION,
        sgl_kernel_build_id=SGL_KERNEL_BUILD_ID,
        deep_gemm_build_id=DEEP_GEMM_BUILD_ID,
        kt_kernel_build_id=KT_KERNEL_BUILD_ID,
    )


def make_runtime_validation_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtRuntimeValidationReceiptObservation:
    return SglangKtRuntimeValidationReceiptObservation(
        target_profile=spec.target_profile,
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        cpu_cores=spec.cpu_cores,
        memory_nodes=spec.memory_nodes,
        executed_cpu_backend="AMX",
        model_id=spec.model_id,
        model_revision=spec.expected_model_revision,
        model_config_sha256=CONFIG_SHA256,
        sglang_revision=spec.expected_sglang_revision,
        ktransformers_revision=spec.expected_ktransformers_revision,
        transformers_distribution_version=spec.required_transformers_version,
        transformers_module_version=spec.required_transformers_version,
        torch_version=TORCH_VERSION,
        cuda_version=CUDA_VERSION,
        sgl_kernel_build_id=SGL_KERNEL_BUILD_ID,
        deep_gemm_build_id=DEEP_GEMM_BUILD_ID,
        kt_kernel_build_id=KT_KERNEL_BUILD_ID,
        ktransformers_method=spec.ktransformers_method,
        resident_gpu_experts=spec.stage.resident_gpu_experts,
        attention_backend=spec.attention_backend,
        kv_cache_dtype=spec.kv_cache_dtype,
        max_total_tokens=spec.plan.max_total_tokens,
        static_memory_fraction=spec.plan.static_memory_fraction,
        capabilities=(
            "kt_tp_group_local_broadcast_v1",
            "glm52_nsa_sm86_short_forward_v1",
            "kt_physical_numa_mapping_v1",
            "kt_process_cpu_affinity_v1",
            "kt_fp8_amx_executed_v1",
        ),
    )


def make_gpu_resource(
    spec: SglangKtProcessLaunchSpec,
) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=spec.gpu_uuid,
        pci_bus_id=f"0000:{spec.pipeline_rank + 1:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
        numa_node=spec.memory_nodes[0],
        cpu_affinity=spec.cpu_cores,
    )


@dataclass
class StaticRuntimeProbe:
    observation: SglangKtRuntimeObservation
    calls: list[AbsoluteRuntimePath] = field(default_factory=list)

    def observe_runtime(
        self, executable: AbsoluteRuntimePath
    ) -> SglangKtRuntimeObservation:
        self.calls.append(executable)
        return self.observation


@dataclass
class SuccessfulFilesystemProbe:
    readable_calls: list[AbsoluteRuntimePath] = field(default_factory=list)
    snapshot_calls: list[tuple[AbsoluteRuntimePath, ModelId, GitRevision]] = field(
        default_factory=list
    )

    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
        self.readable_calls.append(path)
        return True

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation:
        self.snapshot_calls.append((path, model_id, revision))
        return SglangKtModelSnapshotReceiptObservation(
            model_path=path,
            model_id=model_id,
            revision=revision,
            weight_format="safetensors",
            ktransformers_method="FP8",
            config_sha256=CONFIG_SHA256,
            full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
            receipt_verified=True,
            snapshot_complete=True,
        )


@dataclass
class IncompatibleFilesystemProbe:
    mode: Literal["missing", "wrong_method", "wrong_path"]

    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
        del path
        return True

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation | None:
        if self.mode == "missing":
            return None
        return SglangKtModelSnapshotReceiptObservation(
            model_path="/injected/wrong-path" if self.mode == "wrong_path" else path,
            model_id=model_id,
            revision=revision,
            weight_format="safetensors",
            ktransformers_method=("BF16" if self.mode == "wrong_method" else "FP8"),
            config_sha256=CONFIG_SHA256,
            full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
            receipt_verified=True,
            snapshot_complete=True,
        )


@dataclass
class StaticInventoryProbe:
    inventory: SglangKtLocalHostInventory
    calls: list[tuple[NvidiaGpuComputeResource, ...]] = field(default_factory=list)

    def observe_inventory(
        self,
        gpu_resources: tuple[NvidiaGpuComputeResource, ...],
    ) -> SglangKtLocalHostInventory:
        self.calls.append(gpu_resources)
        return self.inventory


@dataclass
class SuccessfulPortProbe:
    endpoint_calls: list[Host] = field(default_factory=list)

    def can_bind_endpoint(self, endpoint: Host) -> bool:
        self.endpoint_calls.append(endpoint)
        return True


def make_inventory(
    specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> SglangKtLocalHostInventory:
    return SglangKtLocalHostInventory(
        gpu_resources=tuple(make_gpu_resource(spec) for spec in specs),
        cpu_cores=tuple(core for spec in specs for core in spec.cpu_cores),
        memory_nodes=tuple(
            dict.fromkeys(node for spec in specs for node in spec.memory_nodes)
        ),
        hca_devices=tuple(
            dict.fromkeys(device for spec in specs for device in spec.hca_devices)
        ),
    )


def collect_successful_observation(
    specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[
    SglangKtHostPreflightObservation,
    SuccessfulFilesystemProbe,
    StaticInventoryProbe,
    SuccessfulPortProbe,
]:
    gpu_resources = tuple(make_gpu_resource(spec) for spec in specs)
    filesystem_probe = SuccessfulFilesystemProbe()
    inventory_probe = StaticInventoryProbe(make_inventory(specs))
    port_probe = SuccessfulPortProbe()
    observation = collect_sglang_kt_local_host_preflight_observation(
        specs,
        gpu_resources=gpu_resources,
        runtime_probe=StaticRuntimeProbe(make_runtime(specs[0])),
        filesystem_probe=filesystem_probe,
        inventory_probe=inventory_probe,
        port_probe=port_probe,
        runtime_validation_receipts=tuple(
            make_runtime_validation_receipt(spec) for spec in specs
        ),
    )
    return observation, filesystem_probe, inventory_probe, port_probe


def test_collects_exact_facts_for_multiple_local_process_specs() -> None:
    specs = make_specs()
    dwagon, filesystem_probe, inventory_probe, port_probe = (
        collect_successful_observation((specs[0], specs[1]))
    )
    fwuff, _filesystem, _inventory, _ports = collect_successful_observation((specs[2],))

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightPassed)
    assert dwagon.node_id == specs[0].node_id
    assert dwagon.gpu_uuids == (specs[0].gpu_uuid, specs[1].gpu_uuid)
    assert dwagon.cpu_cores == (*specs[0].cpu_cores, *specs[1].cpu_cores)
    assert dwagon.memory_nodes == (0, 1)
    assert dwagon.hca_devices == ("mlx4_0:1", "mlx4_0:2")
    assert dwagon.readable_directories == (specs[0].model_path,)
    assert tuple(receipt.model_path for receipt in dwagon.model_snapshot_receipts) == (
        specs[0].model_path,
    )
    assert filesystem_probe.readable_calls == [specs[0].model_path]
    assert len(filesystem_probe.snapshot_calls) == 1
    assert inventory_probe.calls == [
        (make_gpu_resource(specs[0]), make_gpu_resource(specs[1]))
    ]
    assert port_probe.endpoint_calls == [
        specs[0].service_endpoint,
        specs[1].service_endpoint,
        specs[0].distributed_coordinator,
    ]


@pytest.mark.parametrize("mode", ["missing", "wrong_method", "wrong_path"])
def test_collector_omits_unestablished_or_incompatible_snapshot_facts(
    mode: Literal["missing", "wrong_method", "wrong_path"],
) -> None:
    spec = make_specs()[0]
    gpu_resource = make_gpu_resource(spec)

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(gpu_resource,),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=IncompatibleFilesystemProbe(mode),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
    )

    assert observation.model_snapshot_receipts == ()
    assert observation.runtime_validation_receipts == ()


def test_collector_never_infers_runtime_validation_from_versions() -> None:
    spec = make_specs()[0]

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
    )

    assert observation.runtime == make_runtime(spec)
    assert observation.runtime_validation_receipts == ()


def test_collector_discards_validation_receipt_for_an_unobserved_gpu() -> None:
    spec = make_specs()[0]
    other_spec = make_specs()[1]

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        runtime_validation_receipts=(make_runtime_validation_receipt(other_spec),),
    )

    assert observation.runtime_validation_receipts == ()


def test_shared_model_and_ktransformers_path_is_probed_once() -> None:
    plan = make_plan()
    shared_path_plan = plan.model_copy(
        update={
            "stages": tuple(
                stage.model_copy(update={"ktransformers_weight_path": stage.model_path})
                for stage in plan.stages
            )
        }
    )
    specs = build_glm_5_2_fp8_process_launch_specs(
        shared_path_plan,
        PYTHON_EXECUTABLE,
    )

    dwagon, filesystem_probe, _inventory, _ports = collect_successful_observation(
        (specs[0], specs[1])
    )
    fwuff, _filesystem, _inventory, _ports = collect_successful_observation((specs[2],))

    assert isinstance(
        evaluate_sglang_kt_preflight(specs, (dwagon, fwuff)),
        SglangKtPreflightPassed,
    )
    assert filesystem_probe.snapshot_calls == [
        (specs[0].model_path, specs[0].model_id, specs[0].expected_model_revision)
    ]
    assert len(dwagon.model_snapshot_receipts) == 1


@dataclass
class RecordingRuntimeCommandRunner:
    result: SglangKtRuntimeCommandResult
    calls: list[tuple[tuple[str, ...], float]] = field(default_factory=list)

    def __call__(
        self,
        command: tuple[str, ...],
        timeout_seconds: float,
    ) -> SglangKtRuntimeCommandResult:
        self.calls.append((command, timeout_seconds))
        return self.result


def test_external_runtime_probe_parses_only_the_external_python_payload() -> None:
    spec = make_specs()[0]
    expected = make_runtime(spec)
    command_runner = RecordingRuntimeCommandRunner(
        SglangKtRuntimeCommandResult(
            return_code=0,
            stdout=expected.model_dump_json(),
            stderr="",
        )
    )

    observed = ExternalPythonSglangKtRuntimeProbe(command_runner).observe_runtime(
        spec.executable
    )

    assert observed == expected
    assert len(command_runner.calls) == 1
    command, timeout_seconds = command_runner.calls[0]
    assert command[:3] == (spec.executable, "-I", "-c")
    assert 'source_revision("sglang")' in command[3]
    assert 'source_revision("ktransformers")' in command[3]
    assert 'distribution_version("transformers-kt")' in command[3]
    assert 'module_version("transformers")' in command[3]
    assert "torch_runtime_versions()" in command[3]
    assert '"sgl_kernel", "common_ops"' in command[3]
    assert '"deep_gemm", "deep_gemm"' in command[3]
    assert '"kt_kernel", "kt_kernel_ext", ("kt_kernel_ext",)' in command[3]
    assert "exo-sglang-kt-artifact-v1" in command[3]
    assert '"status"' in command[3]
    assert '"--untracked-files=no"' in command[3]
    assert timeout_seconds == 30.0


def test_artifact_build_id_includes_discoverable_top_level_extension(
    tmp_path: Path,
) -> None:
    extension_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    package_name = "_exo_test_kernel_package"
    extension_module_name = "_exo_test_kernel_ext"
    package_path = tmp_path / package_name
    package_path.mkdir()
    (package_path / "__init__.py").write_text("BUILD = 1\n", encoding="utf-8")
    (package_path / f"kernel_ext{extension_suffix}").write_bytes(b"package-native")
    top_level_extension = tmp_path / f"{extension_module_name}{extension_suffix}"
    top_level_extension.write_bytes(b"top-level-native-v1")
    sys.path.insert(0, str(tmp_path))
    try:
        importlib.invalidate_caches()
        probe_namespace: dict[str, object] = {
            "hashlib": hashlib,
            "importlib": importlib,
            "pathlib": pathlib,
        }
        exec(SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE, probe_namespace)
        probe_calculator = cast(
            Callable[[str, str, tuple[str, ...]], str | None],
            probe_namespace["calculate_sglang_kt_artifact_build_id"],
        )

        first_build_id = calculate_sglang_kt_artifact_build_id(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )
        first_probe_build_id = probe_calculator(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )
        top_level_extension.write_bytes(b"top-level-native-v2")
        second_build_id = calculate_sglang_kt_artifact_build_id(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )
        second_probe_build_id = probe_calculator(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )

        assert first_build_id is not None
        assert len(first_build_id) == 64
        assert first_probe_build_id == first_build_id
        assert second_build_id is not None
        assert second_probe_build_id == second_build_id
        assert first_build_id != second_build_id
        missing_build_id = calculate_sglang_kt_artifact_build_id(
            package_name, "missing-native"
        )
        assert missing_build_id is None
        assert probe_calculator(package_name, "missing-native", ()) == missing_build_id
    finally:
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()


def test_artifact_build_id_is_stable_after_internal_extension_alias_load(
    tmp_path: Path,
) -> None:
    extension_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    package_name = "_exo_test_internal_kernel_package"
    extension_module_name = "_exo_test_internal_kernel_ext"
    package_path = tmp_path / package_name
    package_path.mkdir()
    (package_path / "__init__.py").write_text("BUILD = 1\n", encoding="utf-8")
    internal_extension = package_path / f"kernel_ext{extension_suffix}"
    internal_extension.write_bytes(b"internal-native")
    sys.path.insert(0, str(tmp_path))
    try:
        importlib.invalidate_caches()
        before_alias = calculate_sglang_kt_artifact_build_id(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )
        extension_module = types.ModuleType(extension_module_name)
        extension_module.__spec__ = importlib.machinery.ModuleSpec(
            extension_module_name,
            loader=None,
            origin=str(internal_extension),
        )
        sys.modules[extension_module_name] = extension_module
        after_alias = calculate_sglang_kt_artifact_build_id(
            package_name,
            "kernel_ext",
            (extension_module_name,),
        )

        assert before_alias is not None
        assert after_alias == before_alias
    finally:
        sys.modules.pop(extension_module_name, None)
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()


@pytest.mark.parametrize(
    "command_result",
    [
        SglangKtRuntimeCommandResult(return_code=1, stdout="{}", stderr="failure"),
        SglangKtRuntimeCommandResult(return_code=0, stdout="not-json", stderr=""),
        SglangKtRuntimeCommandResult(
            return_code=0,
            stdout='{"executable":"/usr/bin/python","sglangRevision":"main"}',
            stderr="",
        ),
    ],
)
def test_external_runtime_probe_fails_closed_on_untrusted_results(
    command_result: SglangKtRuntimeCommandResult,
) -> None:
    observed = ExternalPythonSglangKtRuntimeProbe(
        RecordingRuntimeCommandRunner(command_result)
    ).observe_runtime(PYTHON_EXECUTABLE)

    assert observed == SglangKtRuntimeObservation()


def test_external_runtime_probe_leaves_no_git_source_revisions_unobserved() -> None:
    spec = make_specs()[0]
    no_git_runtime = SglangKtRuntimeObservation(
        executable=spec.executable,
        python_implementation="CPython",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=13,
            patch=7,
        ),
        sglang_revision=None,
        ktransformers_revision=None,
        transformers_distribution_version=spec.required_transformers_version,
        transformers_module_version=spec.required_transformers_version,
    )
    command_runner = RecordingRuntimeCommandRunner(
        SglangKtRuntimeCommandResult(
            return_code=0,
            stdout=no_git_runtime.model_dump_json(),
            stderr="",
        )
    )

    observed = ExternalPythonSglangKtRuntimeProbe(command_runner).observe_runtime(
        spec.executable
    )

    assert observed == no_git_runtime
    assert observed.sglang_revision is None
    assert observed.ktransformers_revision is None


def test_default_glm_5_2_fp8_compatibility_verifier_accepts_exact_config(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    write_model_config(model_path, make_glm_5_2_fp8_config())

    receipt = observe_default_glm_5_2_fp8_compatibility(model_path)

    assert receipt is not None
    assert receipt.weight_format == "safetensors"
    assert receipt.ktransformers_method == "FP8"
    assert receipt.full_indexer_layer_starts == FULL_INDEXER_LAYER_STARTS
    assert len(receipt.config_sha256) == 64
    assert receipt.receipt_verified
    assert receipt.snapshot_complete


def test_default_glm_4_7_flash_verifier_accepts_only_exact_official_bf16_config(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    write_exact_glm_4_7_flash_config(model_path)

    receipt = observe_default_glm_4_7_flash_bf16_compatibility(model_path)

    assert receipt is not None
    assert receipt.model_id == GLM_4_7_FLASH_BF16_MODEL_ID
    assert receipt.revision == GLM_4_7_FLASH_BF16_MODEL_REVISION
    assert receipt.weight_format == "safetensors"
    assert receipt.ktransformers_method == "BF16"
    assert receipt.config_sha256 == GLM_4_7_FLASH_BF16_CONFIG_SHA256
    assert receipt.full_indexer_layer_starts == (0,)
    assert receipt.receipt_verified
    assert receipt.snapshot_complete


def test_default_glm_4_7_flash_verifier_rejects_wrong_revision_or_config_bytes(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    write_exact_glm_4_7_flash_config(model_path)

    wrong_revision = observe_default_glm_4_7_flash_bf16_compatibility(
        model_path,
        revision="1" * 40,
    )
    config = make_glm_4_7_flash_bf16_config()
    config["dtype"] = "float16"
    (model_path / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    wrong_config = observe_default_glm_4_7_flash_bf16_compatibility(model_path)

    assert wrong_revision is None
    assert wrong_config is None


def test_default_snapshot_verifier_rejects_unadmitted_model_family(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    write_exact_glm_4_7_flash_config(model_path)
    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_completeness_checker=(lambda _path, _model_id, _revision: True)
    )

    receipt = probe.observe_model_snapshot(
        str(model_path), ModelId("untrusted/model"), "1" * 40
    )

    assert receipt is None


def test_default_glm_5_2_fp8_compatibility_verifier_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{not-json", encoding="utf-8")

    assert observe_default_glm_5_2_fp8_compatibility(model_path) is None


@pytest.mark.parametrize(
    "mismatch",
    (
        {"architectures": ["GlmForCausalLM"]},
        {"model_type": "glm"},
        {"num_hidden_layers": 77},
        {"n_routed_experts": 255},
        {"num_experts_per_tok": 7},
        {"n_shared_experts": 0},
        {"first_k_dense_replace": 2},
        {"num_nextn_predict_layers": 0},
        {"index_topk_freq": 1},
        {"index_topk_pattern": ["full"]},
        {"index_skip_topk_offset": 2},
        {"index_share_for_mtp_iteration": False},
        {"indexer_types": ("full",) * 78},
        {"indexer_types": ("full",) * 77},
        {
            "quantization_config": {
                "activation_scheme": "static",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        },
        {
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e5m2",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        },
        {
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "int8",
                "weight_block_size": [128, 128],
            }
        },
        {
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [64, 128],
            }
        },
        {"num_hidden_layers": "78"},
    ),
)
def test_default_glm_5_2_fp8_compatibility_verifier_rejects_mismatch(
    tmp_path: Path,
    mismatch: dict[str, object],
) -> None:
    model_path = tmp_path / "model"
    config = make_glm_5_2_fp8_config()
    config.update(mismatch)
    write_model_config(model_path, config)

    assert observe_default_glm_5_2_fp8_compatibility(model_path) is None


def test_default_glm_5_2_fp8_compatibility_verifier_rejects_read_error(
    tmp_path: Path,
) -> None:
    assert observe_default_glm_5_2_fp8_compatibility(tmp_path / "missing") is None


def test_local_filesystem_probe_preserves_receipt_and_completeness_distinction() -> (
    None
):
    spec = make_specs()[0]
    checked_directories: list[Path] = []
    checked_snapshots: list[tuple[Path, ModelId, GitRevision]] = []

    def check_directory(path: Path) -> bool:
        checked_directories.append(path)
        return True

    def check_snapshot(path: Path, model_id: ModelId, revision: GitRevision) -> bool:
        checked_snapshots.append((path, model_id, revision))
        return False

    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_compatibility_verifier=lambda _path, _model_id, _revision: (
            SglangKtModelSnapshotCompatibility(
                weight_format="safetensors",
                ktransformers_method="FP8",
                config_sha256=CONFIG_SHA256,
                full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
            )
        ),
        readable_directory_checker=check_directory,
        model_snapshot_completeness_checker=check_snapshot,
    )

    readable = probe.is_readable_directory(spec.model_path)
    receipt = probe.observe_model_snapshot(
        spec.model_path,
        spec.model_id,
        spec.expected_model_revision,
    )

    assert readable
    assert receipt is not None
    assert receipt.receipt_verified
    assert not receipt.snapshot_complete
    assert checked_directories == [Path(spec.model_path)]
    assert checked_snapshots == [
        (Path(spec.model_path), spec.model_id, spec.expected_model_revision)
    ]


def test_local_filesystem_probe_fails_closed_when_an_injected_check_raises() -> None:
    spec = make_specs()[0]

    def raise_error(*_arguments: object) -> bool:
        raise OSError("injected failure")

    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_compatibility_verifier=lambda _path, _model_id, _revision: (
            SglangKtModelSnapshotCompatibility(
                weight_format="safetensors",
                ktransformers_method="FP8",
                config_sha256=CONFIG_SHA256,
                full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
            )
        ),
        readable_directory_checker=raise_error,
        model_snapshot_completeness_checker=raise_error,
    )

    assert not probe.is_readable_directory(spec.model_path)
    receipt = probe.observe_model_snapshot(
        spec.model_path,
        spec.model_id,
        spec.expected_model_revision,
    )
    assert receipt is None


def test_local_filesystem_probe_omits_receipt_without_verified_compatibility() -> None:
    spec = make_specs()[0]
    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_compatibility_verifier=(
            lambda _path, _model_id, _revision: None
        ),
        readable_directory_checker=lambda _path: True,
        model_snapshot_completeness_checker=(lambda _path, _model_id, _revision: True),
    )

    assert (
        probe.observe_model_snapshot(
            spec.model_path,
            spec.model_id,
            spec.expected_model_revision,
        )
        is None
    )


def test_linux_inventory_uses_injected_cpu_numa_gpu_and_active_hca_facts() -> None:
    spec = make_specs()[0]
    numa_root = Path("/injected/sys/devices/system/node")
    infiniband_root = Path("/injected/sys/class/infiniband")
    directory_names = {
        infiniband_root: ("mlx5_0", "mlx4_0", "invalid:device"),
        infiniband_root / "mlx5_0" / "ports": ("2", "1"),
        infiniband_root / "mlx4_0" / "ports": ("1",),
    }
    text = {
        numa_root / "online": "0-2,4\n",
        infiniband_root / "mlx5_0" / "ports" / "1" / "state": "4: ACTIVE\n",
        infiniband_root / "mlx5_0" / "ports" / "2" / "state": "1: DOWN\n",
        infiniband_root / "mlx4_0" / "ports" / "1" / "state": "4: Active\n",
    }

    def read_directory_names(path: Path) -> tuple[str, ...]:
        return directory_names[path]

    def read_text(path: Path) -> str:
        return text[path]

    gpu_resource = make_gpu_resource(spec)
    probe = LinuxSglangKtHostInventoryProbe(
        cpu_affinity_reader=lambda: {3, 2, 1},
        directory_names_reader=read_directory_names,
        text_reader=read_text,
        numa_nodes_path=numa_root,
        infiniband_devices_path=infiniband_root,
    )

    inventory = probe.observe_inventory((gpu_resource,))

    assert inventory.gpu_resources == (gpu_resource,)
    assert inventory.gpu_resources[0].numa_node == spec.memory_nodes[0]
    assert inventory.gpu_resources[0].cpu_affinity == spec.cpu_cores
    assert inventory.cpu_cores == (1, 2, 3)
    assert inventory.memory_nodes == (0, 1, 2, 4)
    assert inventory.hca_devices == ("mlx4_0:1", "mlx5_0:1")


def test_linux_inventory_fails_each_unobserved_sysfs_fact_closed() -> None:
    def raise_error(*_arguments: object) -> tuple[str, ...]:
        raise OSError("injected failure")

    probe = LinuxSglangKtHostInventoryProbe(
        cpu_affinity_reader=lambda: (_ for _ in ()).throw(OSError("failure")),
        directory_names_reader=raise_error,
        text_reader=lambda _path: "malformed-range",
        numa_nodes_path=Path("/injected/numa"),
        infiniband_devices_path=Path("/injected/infiniband"),
    )

    inventory = probe.observe_inventory(())

    assert inventory == SglangKtLocalHostInventory()


def test_collector_rejects_reordered_supplied_gpu_inventory() -> None:
    specs = make_specs()[:2]
    supplied_gpu_resources = tuple(make_gpu_resource(spec) for spec in specs)
    inventory = make_inventory(specs)
    reordered_inventory = SglangKtLocalHostInventory(
        gpu_resources=tuple(reversed(inventory.gpu_resources)),
        cpu_cores=inventory.cpu_cores,
        memory_nodes=inventory.memory_nodes,
        hca_devices=inventory.hca_devices,
    )

    observation = collect_sglang_kt_local_host_preflight_observation(
        specs,
        gpu_resources=supplied_gpu_resources,
        runtime_probe=StaticRuntimeProbe(make_runtime(specs[0])),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(reordered_inventory),
        port_probe=SuccessfulPortProbe(),
    )

    assert observation.gpu_uuids == ()
    assert observation.cpu_cores == ()
    assert observation.memory_nodes == ()
    assert observation.hca_devices == ()


def test_socket_port_probe_uses_only_the_injected_bind_effect() -> None:
    calls: list[tuple[str, int]] = []

    def bind_probe(ip: str, port: NetworkPort) -> bool:
        calls.append((ip, port))
        return port != 31_001

    probe = SocketSglangKtPortProbe(bind_probe)

    assert probe.can_bind_endpoint(Host(ip="192.0.2.10", port=30_000))
    assert calls == [("192.0.2.10", 30_000)]


class ExplodingProbe:
    def observe_runtime(
        self, executable: AbsoluteRuntimePath
    ) -> SglangKtRuntimeObservation:
        del executable
        raise RuntimeError("injected runtime failure")

    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
        del path
        raise RuntimeError("injected filesystem failure")

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation:
        del path, model_id, revision
        raise RuntimeError("injected receipt failure")

    def observe_inventory(
        self,
        gpu_resources: tuple[NvidiaGpuComputeResource, ...],
    ) -> SglangKtLocalHostInventory:
        del gpu_resources
        raise RuntimeError("injected inventory failure")

    def can_bind_endpoint(self, endpoint: Host) -> bool:
        del endpoint
        raise RuntimeError("injected port failure")


def test_collector_omits_every_fact_whose_injected_probe_failed() -> None:
    spec = make_specs()[0]
    exploding_probe = ExplodingProbe()

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=exploding_probe,
        filesystem_probe=exploding_probe,
        inventory_probe=exploding_probe,
        port_probe=exploding_probe,
    )

    assert observation.runtime == SglangKtRuntimeObservation()
    assert observation.readable_directories == ()
    assert observation.gpu_uuids == ()
    assert observation.cpu_cores == ()
    assert observation.memory_nodes == ()
    assert observation.hca_devices == ()
    assert observation.available_bind_endpoints == ()
    assert observation.model_snapshot_receipts == ()


def test_collector_rejects_empty_mixed_node_or_duplicate_rank_inputs() -> None:
    specs = make_specs()
    probe = ExplodingProbe()

    def collect(local_specs: tuple[SglangKtProcessLaunchSpec, ...]) -> object:
        return collect_sglang_kt_local_host_preflight_observation(
            local_specs,
            gpu_resources=(),
            runtime_probe=probe,
            filesystem_probe=probe,
            inventory_probe=probe,
            port_probe=probe,
        )

    with pytest.raises(ValueError, match="requires process specs"):
        collect(())
    with pytest.raises(ValueError, match="target one node"):
        collect((specs[0], specs[2]))
    with pytest.raises(ValueError, match="ranks must be unique"):
        collect((specs[0], specs[0]))


def test_local_inventory_rejects_duplicate_gpu_identities() -> None:
    gpu_resource = make_gpu_resource(make_specs()[0])

    with pytest.raises(ValidationError, match="gpu_resources must be unique"):
        SglangKtLocalHostInventory(
            gpu_resources=(gpu_resource, gpu_resource),
        )
