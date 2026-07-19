import copy
import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast

import pytest

from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import (
    SglangKtLaunchPlan,
    SglangKtStageSpec,
    SglangKtTargetProfile,
)
from exo.worker.sglang_kt import model_contract as model_contract_module
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs,
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import (
    SglangKtVerifiedModelSnapshot,
    load_sglang_kt_model_contract,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION,
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
    build_glm_4_7_flash_layer_one_reference_tensor_keys,
    calculate_glm_4_7_flash_expert_mask_sha256,
    calculate_sglang_kt_model_runtime_validator_bundle_sha256,
    canonicalize_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.receipt_io import SglangKtBoundFile
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from scripts.sglang_kt_glm47_backend import (
    Glm47BackendEvidence,
    Glm47LayerOneExpertProbeEvidence,
    Glm47RuntimeEvidence,
    Glm47ShortForwardEvidence,
    Glm47ShortForwardInvocationEvidence,
    Glm47WrapperCoverageEvidence,
    Glm47WrapperForwardInvocationEvidence,
    Glm47WrapperLayerEvidence,
)
from scripts.sglang_kt_glm47_live import (
    ValidatorBundleIdentity,
    ValidatorSourceIdentity,
)
from scripts.sglang_kt_glm47_receipt import (
    Glm47ReceiptAssemblyError,
    build_glm47_model_runtime_validation_receipt_payload,
    require_glm47_model_runtime_validation_receipt_parent_bindings,
)
from scripts.sglang_kt_glm47_reference import (
    Glm47LayerOneHybridMergeEvidence,
    Glm47LayerOneOutputEvidence,
    Glm47NumericalEvidence,
)
from scripts.validate_sglang_kt_glm47_model import (
    BoundProcessSpec,
    Glm47ModelExecutionPreflight,
    select_resident_expert_route,
)

GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"
MODEL_PATH = "/var/lib/exo/models/glm-4.7-flash-bf16"
PYTHON_EXECUTABLE = "/var/lib/exo/runtimes/glm47/bin/python"
KERNEL_RECEIPT_PATH = Path("/var/lib/exo/receipts/kernel.json")
KERNEL_RECEIPT_SHA256 = "3" * 64


def _process_spec(resident_gpu_experts: int) -> SglangKtProcessLaunchSpec:
    target_profile: SglangKtTargetProfile = (
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        if resident_gpu_experts == 0
        else GLM_4_7_FLASH_TARGET_PROFILE
    )
    plan = SglangKtLaunchPlan(
        target_profile=target_profile,
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        total_layers=47,
        context_length=202_752,
        max_total_tokens=4_096,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(ip="192.168.40.248", port=29_510),
        rank_zero_endpoint=Host(ip="192.168.40.248", port=30_100),
        stages=(
            SglangKtStageSpec(
                pipeline_rank=0,
                start_layer=0,
                end_layer=47,
                node_id=NodeId("dwagon"),
                gpu_uuid=GPU_UUID,
                service_endpoint=Host(ip="192.168.40.248", port=30_100),
                model_path=MODEL_PATH,
                ktransformers_weight_path=MODEL_PATH,
                cpu_cores=(0, 1, 2, 3),
                memory_nodes=(0,),
                cpu_infer_threads=4,
                threadpool_count=1,
                ktransformers_method="BF16",
                resident_gpu_experts=resident_gpu_experts,
            ),
        ),
    )
    if resident_gpu_experts == 0:
        return build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
            plan,
            PYTHON_EXECUTABLE,
        )[0]
    return build_glm_4_7_flash_bf16_process_launch_specs(
        plan,
        PYTHON_EXECUTABLE,
    )[0]


def _validator_bundle() -> ValidatorBundleIdentity:
    sources = tuple(
        ValidatorSourceIdentity(
            path=f"/opt/exo/{relative_path}",
            size_bytes=index,
            sha256=hashlib.sha256(relative_path.encode()).hexdigest(),
        )
        for index, relative_path in enumerate(
            MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
            start=1,
        )
    )
    pairs = tuple((source.path, source.sha256) for source in sources)
    return ValidatorBundleIdentity(
        canonicalization=MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION,
        sources=sources,
        sha256=calculate_sglang_kt_model_runtime_validator_bundle_sha256(pairs),
    )


def _preflight(resident_gpu_experts: int) -> Glm47ModelExecutionPreflight:
    process_spec = _process_spec(resident_gpu_experts)
    contract_path = (
        Path(model_contract_module.__file__).parent
        / "manifests"
        / GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME
    )
    loaded_contract = load_sglang_kt_model_contract(
        contract_path,
        expected_contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    )
    process_spec_sha256 = calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    contract = loaded_contract.contract
    config_sha256 = next(
        file.sha256 for file in contract.files if file.role == "config"
    )
    index_sha256 = next(
        file.sha256 for file in contract.files if file.role == "safetensors_index"
    )
    snapshot = SglangKtVerifiedModelSnapshot(
        model_path=MODEL_PATH,
        model_id=contract.model_id,
        revision=contract.revision,
        weight_format=contract.weight_format,
        ktransformers_method=contract.ktransformers_method,
        full_indexer_layer_starts=contract.full_indexer_layer_starts,
        contract_path=loaded_contract.path,
        contract_receipt_sha256=loaded_contract.receipt_sha256,
        contract_sha256=loaded_contract.contract_sha256,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        weight_map_entries=contract.weight_map_entries,
        shard_count=sum(file.role == "weight_shard" for file in contract.files),
        physical_weight_bytes=contract.physical_weight_bytes,
    )
    return Glm47ModelExecutionPreflight(
        validator=_validator_bundle(),
        process=BoundProcessSpec(
            path=Path("/var/lib/exo/specs/glm47.json"),
            receipt_size_bytes=1,
            receipt_sha256="2" * 64,
            process_spec_sha256=process_spec_sha256,
            process_spec=process_spec,
        ),
        model_contract=loaded_contract,
        kernel_runtime_receipt=SglangKtBoundFile(
            path=KERNEL_RECEIPT_PATH,
            contents=b"{}",
            sha256=KERNEL_RECEIPT_SHA256,
        ),
        output=Path("/var/lib/exo/receipts/model.json"),
        route=select_resident_expert_route(resident_gpu_experts),
        model_snapshot=snapshot,
    )


def _kernel_runtime() -> SglangKtKernelRuntimeValidationReceiptObservation:
    return SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path=str(KERNEL_RECEIPT_PATH),
        receipt_size_bytes=1,
        receipt_sha256=KERNEL_RECEIPT_SHA256,
        schema_version=1,
        generated_at_utc="2026-07-19T11:00:00+00:00",
        capabilities=("kt_bf16_amx_executed_v1",),
        gpu_uuid=GPU_UUID,
        gpu_compute_capability=(8, 6),
        gpu_pci_bus_id="0000:01:00.0",
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_total_memory_bytes=24 * 1024**3,
        driver_version="570.00",
        hostname="dwagon",
        executable=PYTHON_EXECUTABLE,
        cpu_cores=(0, 1, 2, 3),
        allowed_memory_nodes=(0,),
        memory_nodes=(0,),
        threads_per_subpool=(4,),
        build_receipt_path="/var/lib/exo/receipts/build.json",
        build_receipt_sha256="4" * 64,
        runtime_build_id="5" * 64,
        builder_sha256="6" * 64,
        kt_extension_sha256="7" * 64,
        host_profile="dwagon-sm86-amx",
        package_version="0.0.1",
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        torch_version="2.9.1+cu128",
        cuda_version="12.8",
        transformers_distribution_version="5.6.0.post1",
        transformers_module_version="5.6.0",
        sgl_kernel_build_id="8" * 64,
        deep_gemm_build_id="9" * 64,
        kt_kernel_build_id="a" * 64,
    )


def _numerical() -> Glm47NumericalEvidence:
    return Glm47NumericalEvidence(
        shape=(1, 2_048),
        execution_dtype="torch.bfloat16",
        reference_dtype="torch.float32",
        output_finite=True,
        reference_finite=True,
        mean_absolute_error=0.001,
        maximum_absolute_error=0.01,
        reference_mean_absolute=0.5,
        relative_l1_error=0.002,
        relative_l1_tolerance=0.02,
    )


def _output(actual: str, reference: str) -> Glm47LayerOneOutputEvidence:
    return Glm47LayerOneOutputEvidence(
        actual_output_sha256=actual,
        repeat_actual_output_sha256=actual,
        reference_output_sha256=reference,
        numerical=_numerical(),
    )


def _backend(preflight: Glm47ModelExecutionPreflight) -> Glm47BackendEvidence:
    route = preflight.route
    resident_by_layer = tuple(route.resident_gpu_expert_ids for _ in range(46))
    mask_sha256 = calculate_glm_4_7_flash_expert_mask_sha256(resident_by_layer)
    cpu_control = not route.gpu_expert_ids
    combined = _output("a" * 64, "d" * 64)
    cpu = combined if cpu_control else _output("e" * 64, "f" * 64)
    gpu = None if cpu_control else _output("7" * 64, "8" * 64)
    hybrid = (
        None
        if cpu_control
        else Glm47LayerOneHybridMergeEvidence(
            combined_output_sha256="a" * 64,
            cpu_output_sha256="e" * 64,
            gpu_output_sha256="7" * 64,
            merged_backend_output_sha256="a" * 64,
            repeat_merged_backend_output_sha256="a" * 64,
            reference_merged_backend_output_sha256="6" * 64,
            numerical=_numerical(),
        )
    )
    return Glm47BackendEvidence(
        runtime=Glm47RuntimeEvidence(
            server_arguments=preflight.process.process_spec.arguments[2:],
            service_endpoint=str(preflight.process.process_spec.service_endpoint),
            distributed_coordinator=str(
                preflight.process.process_spec.distributed_coordinator
            ),
            model_runner_nccl_port=(
                preflight.process.process_spec.service_endpoint.port
            ),
            server_args_class="sglang.srt.server_args.ServerArgs",
            model_config_class="sglang.srt.configs.model_config.ModelConfig",
            model_runner_class="sglang.srt.model_executor.ModelRunner",
            tokenizer_class="transformers.PreTrainedTokenizerFast",
            tp_size=1,
            pp_size=1,
            expert_parallel_size=1,
        ),
        wrapper_coverage=Glm47WrapperCoverageEvidence(
            global_expert_mask_sha256=mask_sha256,
            layers=tuple(
                Glm47WrapperLayerEvidence(
                    layer_index=layer,
                    expert_module_name=f"model.layers.{layer}.mlp.experts",
                    quant_method_wrapper="kt_ep",
                    expert_count=64,
                    resident_gpu_expert_ids=route.resident_gpu_expert_ids,
                    cpu_backend_wrapper_class="NativeMoEWrapper",
                    cpu_kernel_class="AMXBF16_MOE",
                    global_expert_mask_sha256=mask_sha256,
                )
                for layer in range(1, 47)
            ),
        ),
        layer_one_expert_probe=Glm47LayerOneExpertProbeEvidence(
            layer_index=1,
            random_seed=20_260_719,
            probe_invocation_count=2,
            input_shape=(1, 2_048),
            input_dtype="torch.bfloat16",
            input_sha256="b" * 64,
            selected_expert_ids=cast(
                tuple[int, int, int, int],
                route.selected_expert_ids,
            ),
            repeat_selected_expert_ids=cast(
                tuple[int, int, int, int],
                route.selected_expert_ids,
            ),
            routing_weights=(0.4, 0.3, 0.2, 0.1),
            reference_tensor_keys=(
                build_glm_4_7_flash_layer_one_reference_tensor_keys(
                    route.selected_expert_ids
                )
            ),
            cpu_expert_ids=route.cpu_expert_ids,
            gpu_expert_ids=route.gpu_expert_ids,
            cpu_backend_wrapper_class="NativeMoEWrapper",
            cpu_kernel_class="AMXBF16_MOE",
            sglang_cpu_submit_count=2,
            sglang_cpu_sync_count=2,
            native_cpu_submit_count=2,
            native_cpu_sync_count=2,
            gpu_forward_count=0 if cpu_control else 2,
            output_merge_count=2,
            global_expert_mask_sha256=mask_sha256,
            combined_output=combined,
            cpu_output=cpu,
            gpu_output=gpu,
            hybrid_merge=hybrid,
        ),
        short_forward=Glm47ShortForwardEvidence(
            random_seed=20_260_719,
            extend=Glm47ShortForwardInvocationEvidence(
                forward_mode="extend",
                input_token_ids=tuple(range(1, 9)),
                positions=tuple(range(8)),
                kv_cache_length_before=0,
                kv_cache_length_after=8,
                model_forward_invocation_count=1,
                logits_shape=(1, 154_880),
                logits_dtype="torch.float32",
                logits_finite=True,
                logits_sha256="c" * 64,
                argmax_token_id=42,
                global_expert_mask_sha256_before=mask_sha256,
                global_expert_mask_sha256_after=mask_sha256,
            ),
            decode=Glm47ShortForwardInvocationEvidence(
                forward_mode="decode",
                input_token_ids=(42,),
                positions=(8,),
                kv_cache_length_before=8,
                kv_cache_length_after=9,
                model_forward_invocation_count=1,
                logits_shape=(1, 154_880),
                logits_dtype="torch.float32",
                logits_finite=True,
                logits_sha256="d" * 64,
                argmax_token_id=43,
                global_expert_mask_sha256_before=mask_sha256,
                global_expert_mask_sha256_after=mask_sha256,
            ),
            wrapper_invocations=tuple(
                Glm47WrapperForwardInvocationEvidence(
                    layer_index=layer,
                    extend_invocation_count=1,
                    decode_invocation_count=1,
                )
                for layer in range(1, 47)
            ),
        ),
        trace_events=(),
        trace_counters=(),
        cleanup_completed=True,
    )


@pytest.mark.parametrize("resident_gpu_experts", (0, 2))
def test_builds_exact_canonical_schema_v1_payload(
    resident_gpu_experts: int,
) -> None:
    preflight = _preflight(resident_gpu_experts)
    kernel = _kernel_runtime()
    backend = _backend(preflight)
    generated_at = datetime(2026, 7, 19, 12, 34, 56, tzinfo=UTC)

    first = build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=generated_at,
        preflight=preflight,
        kernel_runtime=kernel,
        backend=backend,
    )
    second = build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=generated_at,
        preflight=preflight,
        kernel_runtime=kernel,
        backend=backend,
    )

    assert canonicalize_sglang_kt_model_runtime_validation_receipt(first) == (
        canonicalize_sglang_kt_model_runtime_validation_receipt(second)
    )
    assert first["generated_at_utc"] == "2026-07-19T12:34:56+00:00"
    assert first["profiler"] == "none"
    assert first["failures"] == []
    assert first["wrapper_coverage"] == backend.wrapper_coverage.as_receipt_json()
    assert first["layer_one_expert_probe"] == (
        backend.layer_one_expert_probe.as_receipt_json()
    )
    assert first["short_forward"] == backend.short_forward.as_receipt_json()


def test_rejects_non_utc_time_and_incomplete_backend_cleanup() -> None:
    preflight = _preflight(2)
    kernel = _kernel_runtime()
    backend = _backend(preflight)

    with pytest.raises(Glm47ReceiptAssemblyError, match="must be UTC"):
        build_glm47_model_runtime_validation_receipt_payload(
            generated_at_utc=datetime(
                2026,
                7,
                19,
                tzinfo=timezone(timedelta(hours=1)),
            ),
            preflight=preflight,
            kernel_runtime=kernel,
            backend=backend,
        )

    incomplete = replace(
        backend,
        cleanup_completed=cast(Literal[True], cast(object, False)),
    )
    with pytest.raises(Glm47ReceiptAssemblyError, match="cleanup"):
        build_glm47_model_runtime_validation_receipt_payload(
            generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
            preflight=preflight,
            kernel_runtime=kernel,
            backend=incomplete,
        )


def test_rejects_kernel_evidence_from_a_different_parent() -> None:
    preflight = _preflight(2)
    mismatched = _kernel_runtime().model_copy(update={"receipt_sha256": "f" * 64})

    with pytest.raises(Glm47ReceiptAssemblyError, match="kernel runtime evidence"):
        build_glm47_model_runtime_validation_receipt_payload(
            generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
            preflight=preflight,
            kernel_runtime=mismatched,
            backend=_backend(preflight),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("service_endpoint", "127.0.0.1:1"),
        ("distributed_coordinator", "127.0.0.1:2"),
        ("model_runner_nccl_port", 1),
    ),
)
def test_rejects_backend_port_evidence_from_a_different_process_spec(
    field: str,
    value: object,
) -> None:
    preflight = _preflight(2)
    backend = _backend(preflight)
    mismatched = replace(
        backend,
        runtime=replace(backend.runtime, **{field: value}),
    )

    with pytest.raises(Glm47ReceiptAssemblyError, match="backend runtime evidence"):
        build_glm47_model_runtime_validation_receipt_payload(
            generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
            preflight=preflight,
            kernel_runtime=_kernel_runtime(),
            backend=mismatched,
        )


def test_assembly_performs_no_filesystem_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _preflight(2)
    kernel = _kernel_runtime()
    backend = _backend(preflight)

    def reject_io(*_arguments: object, **_keyword_arguments: object) -> None:
        raise AssertionError("receipt assembly attempted filesystem I/O")

    monkeypatch.setattr(Path, "open", reject_io)
    monkeypatch.setattr(Path, "read_bytes", reject_io)
    monkeypatch.setattr(Path, "read_text", reject_io)
    build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
        preflight=preflight,
        kernel_runtime=kernel,
        backend=backend,
    )


def test_parent_binding_rejects_schema_valid_child_route_substitution() -> None:
    preflight = _preflight(2)
    kernel = _kernel_runtime()
    payload = build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
        preflight=preflight,
        kernel_runtime=kernel,
        backend=_backend(preflight),
    )
    substituted = copy.deepcopy(payload)
    probe = cast(dict[str, object], substituted["layer_one_expert_probe"])
    substituted_ids = (0, 1, 4, 5)
    probe["selected_expert_ids"] = list(substituted_ids)
    probe["repeat_selected_expert_ids"] = list(substituted_ids)
    probe["cpu_expert_ids"] = [4, 5]
    probe["reference_tensor_keys"] = list(
        build_glm_4_7_flash_layer_one_reference_tensor_keys(substituted_ids)
    )

    canonicalize_sglang_kt_model_runtime_validation_receipt(substituted)
    with pytest.raises(Glm47ReceiptAssemblyError, match="expert route"):
        require_glm47_model_runtime_validation_receipt_parent_bindings(
            substituted,
            preflight=preflight,
            kernel_runtime=kernel,
        )


def test_parent_binding_rejects_schema_valid_process_parent_substitution() -> None:
    preflight = _preflight(2)
    kernel = _kernel_runtime()
    payload = build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
        preflight=preflight,
        kernel_runtime=kernel,
        backend=_backend(preflight),
    )
    substituted = copy.deepcopy(payload)
    parents = cast(dict[str, object], substituted["parents"])
    parents["process_spec_sha256"] = "f" * 64

    canonicalize_sglang_kt_model_runtime_validation_receipt(substituted)
    with pytest.raises(Glm47ReceiptAssemblyError, match="bound parents"):
        require_glm47_model_runtime_validation_receipt_parent_bindings(
            substituted,
            preflight=preflight,
            kernel_runtime=kernel,
        )
