import hashlib
import importlib
import importlib.machinery
import json
import pathlib
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, cast

import pytest
from pydantic import ValidationError

import exo.worker.sglang_kt.preflight_collector as preflight_collector
from exo.shared.types.common import Host, ModelId
from exo.shared.types.compute_resources import NvidiaGpuComputeResource
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    KTransformersMethod,
    NetworkPort,
)
from exo.worker.sglang_kt.artifact_identity import (
    SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE,
    calculate_sglang_kt_artifact_build_id,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_process_launch_specs,
    build_glm_5_2_fp8_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    SglangKtModelRuntimeValidationReceiptObservation,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtBoundModelRuntimeValidationReceipt,
    SglangKtHostPreflightObservation,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtPreflightPassed,
    SglangKtPythonVersionObservation,
    SglangKtRuntimeObservation,
    SglangKtRuntimeValidationReceiptObservation,
    evaluate_sglang_kt_preflight,
)
from exo.worker.sglang_kt.preflight_collector import (
    SGLANG_KT_RUNTIME_PROBE_SCRIPT,
    ExternalPythonSglangKtRuntimeProbe,
    LinuxSglangKtHostInventoryProbe,
    LocalSglangKtFilesystemProbe,
    LocalSglangKtKernelRuntimeValidationProbe,
    LocalSglangKtModelContractProbe,
    LocalSglangKtModelRuntimeValidationProbe,
    SglangKtKernelRuntimeValidationBinding,
    SglangKtLocalHostInventory,
    SglangKtModelContractBinding,
    SglangKtModelRuntimeValidationBinding,
    SglangKtModelSnapshotCompatibility,
    SglangKtRuntimeCommandResult,
    SocketSglangKtPortProbe,
    collect_sglang_kt_local_host_preflight_observation,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_plan,
    make_plan,
)
from exo.worker.tests.unittests.test_sglang_kt_preflight import (
    make_model_runtime_validation_receipt,
)

CONFIG_SHA256 = "a" * 64
FULL_INDEXER_LAYER_STARTS = (0, 1, 2, *range(6, 78, 4))
TORCH_VERSION = "2.10.0+cu130"
CUDA_VERSION = "13.0"
SGL_KERNEL_BUILD_ID = "1" * 64
DEEP_GEMM_BUILD_ID = "2" * 64
KT_KERNEL_BUILD_ID = "3" * 64


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
        transformers_distribution_version=(
            spec.required_transformers_distribution_version
        ),
        transformers_module_version=spec.required_transformers_module_version,
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
        transformers_distribution_version=(
            spec.required_transformers_distribution_version
        ),
        transformers_module_version=spec.required_transformers_module_version,
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


def make_kernel_runtime_validation_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtKernelRuntimeValidationReceiptObservation:
    return SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path="/receipts/kernel.json",
        receipt_size_bytes=1,
        receipt_sha256="4" * 64,
        schema_version=1,
        generated_at_utc="2026-07-19T00:00:00+00:00",
        capabilities=("kt_bf16_amx_executed_v1",),
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        gpu_pci_bus_id="0000:01:00.0",
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_total_memory_bytes=24 * 1024**3,
        driver_version="test-driver",
        hostname=str(spec.node_id),
        executable=spec.executable,
        cpu_cores=spec.cpu_cores,
        allowed_memory_nodes=spec.memory_nodes,
        memory_nodes=spec.memory_nodes,
        threads_per_subpool=(spec.stage.cpu_infer_threads,),
        build_receipt_path="/receipts/build.json",
        build_receipt_sha256="5" * 64,
        runtime_build_id="6" * 64,
        builder_sha256="7" * 64,
        kt_extension_sha256="8" * 64,
        host_profile=str(spec.node_id),
        package_version="0.6.3.post1",
        sglang_revision=spec.expected_sglang_revision,
        ktransformers_revision=spec.expected_ktransformers_revision,
        torch_version=TORCH_VERSION,
        cuda_version=CUDA_VERSION,
        transformers_distribution_version=(
            spec.required_transformers_distribution_version
        ),
        transformers_module_version=spec.required_transformers_module_version,
        sgl_kernel_build_id=SGL_KERNEL_BUILD_ID,
        deep_gemm_build_id=DEEP_GEMM_BUILD_ID,
        kt_kernel_build_id=KT_KERNEL_BUILD_ID,
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
class StaticRuntimeValidationProbe:
    observer: Callable[
        [SglangKtProcessLaunchSpec],
        SglangKtRuntimeValidationReceiptObservation | None,
    ]
    calls: list[SglangKtProcessLaunchSpec] = field(default_factory=list)

    def observe_runtime_validation(
        self, process_spec: SglangKtProcessLaunchSpec
    ) -> SglangKtRuntimeValidationReceiptObservation | None:
        self.calls.append(process_spec)
        return self.observer(process_spec)


@dataclass
class StaticKernelRuntimeValidationProbe:
    observer: Callable[
        [SglangKtProcessLaunchSpec],
        SglangKtKernelRuntimeValidationReceiptObservation | None,
    ]

    def observe_kernel_runtime_validation(
        self, process_spec: SglangKtProcessLaunchSpec
    ) -> SglangKtKernelRuntimeValidationReceiptObservation | None:
        return self.observer(process_spec)


@dataclass
class StaticModelRuntimeValidationProbe:
    observation: SglangKtBoundModelRuntimeValidationReceipt
    calls: list[
        tuple[
            SglangKtProcessLaunchSpec,
            SglangKtModelSnapshotReceiptObservation,
            SglangKtKernelRuntimeValidationReceiptObservation,
        ]
    ] = field(default_factory=list)

    def observe_model_runtime_validation(
        self,
        process_spec: SglangKtProcessLaunchSpec,
        model_snapshot_receipt: SglangKtModelSnapshotReceiptObservation,
        kernel_runtime_validation_receipt: (
            SglangKtKernelRuntimeValidationReceiptObservation
        ),
    ) -> SglangKtBoundModelRuntimeValidationReceipt:
        self.calls.append(
            (
                process_spec,
                model_snapshot_receipt,
                kernel_runtime_validation_receipt,
            )
        )
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
class StaticModelContractProbe:
    observation: SglangKtVerifiedModelSnapshot | None
    calls: list[
        tuple[AbsoluteRuntimePath, ModelId, GitRevision, KTransformersMethod]
    ] = field(default_factory=list)

    def verify_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
        ktransformers_method: KTransformersMethod,
    ) -> SglangKtVerifiedModelSnapshot | None:
        self.calls.append((path, model_id, revision, ktransformers_method))
        return self.observation


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
        runtime_validation_probe=StaticRuntimeValidationProbe(
            make_runtime_validation_receipt
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
        runtime_validation_probe=StaticRuntimeValidationProbe(
            lambda _process_spec: make_runtime_validation_receipt(other_spec)
        ),
    )

    assert observation.runtime_validation_receipts == ()


def test_collector_accepts_kernel_evidence_only_through_its_probe() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    kernel_receipt = make_kernel_runtime_validation_receipt(spec)

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        kernel_runtime_validation_probe=StaticKernelRuntimeValidationProbe(
            lambda _process_spec: kernel_receipt
        ),
    )

    assert observation.kernel_runtime_validation_receipts == (kernel_receipt,)


def test_collector_skips_legacy_runtime_validation_for_glm_4_7() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    legacy_probe = StaticRuntimeValidationProbe(make_runtime_validation_receipt)

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        runtime_validation_probe=legacy_probe,
    )

    assert observation.runtime_validation_receipts == ()
    assert legacy_probe.calls == []


def test_collector_discards_kernel_evidence_for_a_different_gpu() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    wrong_receipt = make_kernel_runtime_validation_receipt(spec).model_copy(
        update={"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000099"}
    )

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        kernel_runtime_validation_probe=StaticKernelRuntimeValidationProbe(
            lambda _process_spec: wrong_receipt
        ),
    )

    assert observation.kernel_runtime_validation_receipts == ()


def test_local_kernel_probe_requires_unique_explicit_gpu_bindings() -> None:
    binding = SglangKtKernelRuntimeValidationBinding(
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
        receipt_path="/receipts/kernel.json",
        receipt_sha256="4" * 64,
    )

    with pytest.raises(ValueError, match="unique GPUs"):
        LocalSglangKtKernelRuntimeValidationProbe((binding, binding))


def make_glm_4_7_snapshot_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtModelSnapshotReceiptObservation:
    return SglangKtModelSnapshotReceiptObservation(
        model_path=spec.model_path,
        model_id=spec.model_id,
        revision=spec.expected_model_revision,
        weight_format="safetensors",
        ktransformers_method="BF16",
        config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        full_indexer_layer_starts=(0,),
        receipt_verified=True,
        snapshot_complete=True,
        contract_path="/contracts/glm47.json",
        contract_receipt_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        index_sha256="91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791",
        weight_map_entries=9_703,
        shard_count=48,
        physical_weight_bytes=62_444_175_504,
    )


def bind_model_runtime_receipt(
    spec: SglangKtProcessLaunchSpec,
    receipt: SglangKtModelRuntimeValidationReceiptObservation,
) -> SglangKtBoundModelRuntimeValidationReceipt:
    return SglangKtBoundModelRuntimeValidationReceipt(
        binding=SglangKtModelRuntimeValidationBinding(
            process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(spec),
            validator_sha256=receipt.validator_sha256,
            receipt_path=receipt.receipt_path,
            receipt_sha256=receipt.receipt_sha256,
        ),
        receipt=receipt,
    )


def test_local_model_runtime_probe_binds_every_parent_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    process_spec_sha256 = calculate_sglang_kt_process_launch_spec_sha256(spec)
    snapshot_receipt = make_glm_4_7_snapshot_receipt(spec)
    kernel_receipt = make_kernel_runtime_validation_receipt(spec)
    model_receipt = make_model_runtime_validation_receipt(
        spec,
        snapshot_receipt,
        kernel_receipt,
    )
    bound_model_receipt = bind_model_runtime_receipt(spec, model_receipt)
    calls: list[tuple[Path, str, str, str, str, str | None]] = []

    def load_model_receipt(
        path: Path,
        *,
        expected_validator_sha256: str,
        expected_process_spec_sha256: str,
        expected_model_contract_receipt_sha256: str,
        expected_kernel_receipt_sha256: str,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtModelRuntimeValidationReceiptObservation:
        calls.append(
            (
                path,
                expected_validator_sha256,
                expected_process_spec_sha256,
                expected_model_contract_receipt_sha256,
                expected_kernel_receipt_sha256,
                expected_receipt_sha256,
            )
        )
        return model_receipt

    monkeypatch.setattr(
        preflight_collector,
        "load_sglang_kt_model_runtime_validation_receipt",
        load_model_receipt,
    )
    probe = LocalSglangKtModelRuntimeValidationProbe((bound_model_receipt.binding,))

    observed = probe.observe_model_runtime_validation(
        spec,
        snapshot_receipt,
        kernel_receipt,
    )

    assert observed == bound_model_receipt
    assert calls == [
        (
            Path(model_receipt.receipt_path),
            model_receipt.validator_sha256,
            process_spec_sha256,
            GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
            kernel_receipt.receipt_sha256,
            model_receipt.receipt_sha256,
        )
    ]


def test_local_model_runtime_probe_requires_unique_process_spec_bindings() -> None:
    binding = SglangKtModelRuntimeValidationBinding(
        process_spec_sha256="a" * 64,
        validator_sha256="c" * 64,
        receipt_path="/receipts/model.json",
        receipt_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="unique process specs"):
        LocalSglangKtModelRuntimeValidationProbe((binding, binding))


def test_collector_observes_model_runtime_only_after_snapshot_and_kernel() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    snapshot_receipt = make_glm_4_7_snapshot_receipt(spec)
    kernel_receipt = make_kernel_runtime_validation_receipt(spec)
    model_receipt = make_model_runtime_validation_receipt(
        spec,
        snapshot_receipt,
        kernel_receipt,
    )
    bound_model_receipt = bind_model_runtime_receipt(spec, model_receipt)
    model_probe = StaticModelRuntimeValidationProbe(bound_model_receipt)

    @dataclass
    class FlashFilesystemProbe:
        def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
            return path == spec.model_path

        def observe_model_snapshot(
            self,
            path: AbsoluteRuntimePath,
            model_id: ModelId,
            revision: GitRevision,
        ) -> SglangKtModelSnapshotReceiptObservation:
            assert path == spec.model_path
            assert model_id == spec.model_id
            assert revision == spec.expected_model_revision
            return snapshot_receipt

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=FlashFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        kernel_runtime_validation_probe=StaticKernelRuntimeValidationProbe(
            lambda _process_spec: kernel_receipt
        ),
        model_runtime_validation_probe=model_probe,
    )

    assert observation.bound_model_runtime_validation_receipts == (bound_model_receipt,)
    assert model_probe.calls == [(spec, snapshot_receipt, kernel_receipt)]


def test_collector_withholds_model_runtime_without_kernel_parent() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    snapshot_receipt = make_glm_4_7_snapshot_receipt(spec)
    kernel_receipt = make_kernel_runtime_validation_receipt(spec)
    model_receipt = make_model_runtime_validation_receipt(
        spec,
        snapshot_receipt,
        kernel_receipt,
    )
    model_probe = StaticModelRuntimeValidationProbe(
        bind_model_runtime_receipt(
            spec,
            model_receipt,
        )
    )

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(make_runtime(spec)),
        filesystem_probe=SuccessfulFilesystemProbe(),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        model_runtime_validation_probe=model_probe,
    )

    assert observation.bound_model_runtime_validation_receipts == ()
    assert model_probe.calls == []


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
    assert "embedded_source_revisions()" in command[3]
    assert 'for package_name in ("sglang", "kt_kernel")' in command[3]
    assert '"_exo_build_provenance.py"' in command[3]
    assert 'distribution_version("transformers-kt")' in command[3]
    assert 'module_version("transformers")' in command[3]
    assert "torch_runtime_versions()" in command[3]
    assert '"sgl_kernel", "common_ops"' in command[3]
    assert '"deep_gemm", "deep_gemm"' in command[3]
    assert '"kt_kernel", "kt_kernel_ext", ("kt_kernel_ext",)' in command[3]
    assert "exo-sglang-kt-artifact-v1" in command[3]
    assert '"git"' not in command[3]
    assert "subprocess" not in command[3]
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


def test_external_runtime_probe_leaves_missing_embedded_revisions_unobserved() -> None:
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
        transformers_distribution_version=(
            spec.required_transformers_distribution_version
        ),
        transformers_module_version=spec.required_transformers_module_version,
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


def _execute_runtime_probe_with_provenance(
    tmp_path: Path,
    *,
    kt_kernel_sglang_revision: str,
) -> SglangKtRuntimeObservation:
    ktransformers_revision = "1" * 40
    sglang_revision = "2" * 40
    for package_name, package_sglang_revision in (
        ("sglang", sglang_revision),
        ("kt_kernel", kt_kernel_sglang_revision),
    ):
        package = tmp_path / package_name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "_exo_build_provenance.py").write_text(
            "SCHEMA_VERSION = 1\n"
            f'KTRANSFORMERS_REVISION = "{ktransformers_revision}"\n'
            f'SGLANG_REVISION = "{package_sglang_revision}"\n',
            encoding="ascii",
        )
    wrapper = (
        "import sys;"
        f"sys.path.insert(0, {str(tmp_path)!r});"
        f"exec({SGLANG_KT_RUNTIME_PROBE_SCRIPT!r})"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-c", wrapper),
        check=True,
        capture_output=True,
        text=True,
    )
    return SglangKtRuntimeObservation.model_validate_json(result.stdout)


def test_runtime_probe_accepts_matching_embedded_wheel_provenance(
    tmp_path: Path,
) -> None:
    observed = _execute_runtime_probe_with_provenance(
        tmp_path,
        kt_kernel_sglang_revision="2" * 40,
    )

    assert observed.ktransformers_revision == "1" * 40
    assert observed.sglang_revision == "2" * 40


def test_runtime_probe_rejects_disagreeing_embedded_wheel_provenance(
    tmp_path: Path,
) -> None:
    observed = _execute_runtime_probe_with_provenance(
        tmp_path,
        kt_kernel_sglang_revision="3" * 40,
    )

    assert observed.ktransformers_revision is None
    assert observed.sglang_revision is None


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
    assert receipt.contract_sha256 is None


def test_filesystem_probe_carries_complete_exact_model_contract_evidence(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    write_exact_glm_4_7_flash_config(model_path)
    contract_path = tmp_path / "contract.json"
    verified = SglangKtVerifiedModelSnapshot(
        model_path=str(model_path),
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        weight_format="safetensors",
        ktransformers_method="BF16",
        full_indexer_layer_starts=(0,),
        contract_path=str(contract_path),
        contract_receipt_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        index_sha256="9" * 64,
        weight_map_entries=9_703,
        shard_count=48,
        physical_weight_bytes=62_444_175_504,
    )
    contract_probe = StaticModelContractProbe(verified)
    probe = LocalSglangKtFilesystemProbe(
        model_snapshot_completeness_checker=(lambda _path, _model_id, _revision: True),
        model_contract_probe=contract_probe,
    )

    receipt = probe.observe_model_snapshot(
        str(model_path),
        GLM_4_7_FLASH_BF16_MODEL_ID,
        GLM_4_7_FLASH_BF16_MODEL_REVISION,
    )

    assert receipt is not None
    assert receipt.contract_path == str(contract_path)
    assert receipt.contract_receipt_sha256 == GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    assert receipt.contract_sha256 == GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    assert receipt.index_sha256 == "9" * 64
    assert receipt.weight_map_entries == 9_703
    assert receipt.shard_count == 48
    assert receipt.physical_weight_bytes == 62_444_175_504
    assert contract_probe.calls == [
        (
            str(model_path),
            GLM_4_7_FLASH_BF16_MODEL_ID,
            GLM_4_7_FLASH_BF16_MODEL_REVISION,
            "BF16",
        )
    ]


def test_local_model_contract_probe_rejects_duplicate_identity_bindings() -> None:
    binding = SglangKtModelContractBinding(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        ktransformers_method="BF16",
        contract_path="/contracts/glm47.json",
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    )

    with pytest.raises(ValueError, match="unique identities"):
        LocalSglangKtModelContractProbe((binding, binding))


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
