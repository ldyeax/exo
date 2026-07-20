import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelId
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
    GLM_4_7_FLASH_TARGET_PROFILES,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs,
    build_glm_4_7_flash_bf16_process_launch_specs,
    build_glm_4_7_flash_bf16_serving_baseline_process_launch_specs,
    build_glm_5_2_fp8_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    SglangKtModelRuntimeValidationReceiptObservation,
)
from exo.worker.sglang_kt.preflight import (
    GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS,
    SglangKtBoundModelRuntimeValidationReceipt,
    SglangKtHostPreflightObservation,
    SglangKtModelRuntimeValidationBinding,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtPreflightFailed,
    SglangKtPreflightPassed,
    SglangKtPythonVersionObservation,
    SglangKtRuntimeObservation,
    SglangKtRuntimeValidationReceiptObservation,
    evaluate_sglang_kt_preflight,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_cpu_routed_experts_plan,
    make_glm_4_7_flash_bf16_plan,
    make_glm_4_7_flash_bf16_serving_baseline_plan,
    make_plan,
)

CONFIG_SHA256 = "a" * 64
FULL_INDEXER_LAYER_STARTS = (0, 1, 2, *range(6, 78, 4))
TORCH_VERSION = "2.10.0+cu130"
CUDA_VERSION = "13.0"
SGL_KERNEL_BUILD_ID = "1" * 64
DEEP_GEMM_BUILD_ID = "2" * 64
KT_KERNEL_BUILD_ID = "3" * 64
GLM_4_7_FLASH_INDEX_SHA256 = (
    "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
)


def make_specs() -> tuple[SglangKtProcessLaunchSpec, ...]:
    return build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)


def make_runtime(spec: SglangKtProcessLaunchSpec) -> SglangKtRuntimeObservation:
    return SglangKtRuntimeObservation(
        executable=spec.executable,
        python_implementation="CPython",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=13,
            patch=14,
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
    is_flash = spec.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
    is_cpu_routed_experts_control = (
        spec.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    )
    return SglangKtRuntimeValidationReceiptObservation(
        target_profile=spec.target_profile,
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        cpu_cores=spec.cpu_cores,
        memory_nodes=spec.memory_nodes,
        executed_cpu_backend="AMX_BF16" if is_flash else "AMX",
        model_id=spec.model_id,
        model_revision=spec.expected_model_revision,
        model_config_sha256=(
            GLM_4_7_FLASH_BF16_CONFIG_SHA256 if is_flash else CONFIG_SHA256
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
        ktransformers_method=spec.ktransformers_method,
        resident_gpu_experts=spec.stage.resident_gpu_experts,
        attention_backend=spec.attention_backend,
        kv_cache_dtype=spec.kv_cache_dtype,
        max_total_tokens=spec.plan.max_total_tokens,
        static_memory_fraction=spec.plan.static_memory_fraction,
        capabilities=(
            (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_kt_wrapper_layers_1_46_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
                "glm47_flash_bf16_cpu_routed_experts_executed_v1",
            )
            if is_cpu_routed_experts_control
            else (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_kt_wrapper_layers_1_46_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
                "kt_bf16_cpu_gpu_hybrid_executed_v1",
            )
            if is_flash
            else (
                "kt_tp_group_local_broadcast_v1",
                "glm52_nsa_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_fp8_amx_executed_v1",
            )
        ),
        ktransformers_wrapped_expert_layers=(
            GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS if is_flash else ()
        ),
    )


def make_kernel_runtime_validation_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtKernelRuntimeValidationReceiptObservation:
    return SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path=f"/receipts/kernel-rank-{spec.pipeline_rank}.json",
        receipt_size_bytes=1,
        receipt_sha256=f"{spec.pipeline_rank + 4:x}" * 64,
        schema_version=1,
        generated_at_utc="2026-07-19T00:00:00+00:00",
        capabilities=("kt_bf16_amx_executed_v1",),
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        gpu_pci_bus_id=f"0000:{spec.pipeline_rank + 1:02x}:00.0",
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_total_memory_bytes=24 * 1024**3,
        driver_version="test-driver",
        hostname=str(spec.node_id),
        executable=spec.executable,
        cpu_cores=spec.cpu_cores,
        allowed_memory_nodes=spec.memory_nodes,
        memory_nodes=spec.memory_nodes,
        threads_per_subpool=(spec.stage.cpu_infer_threads,),
        build_receipt_path=f"/receipts/build-rank-{spec.pipeline_rank}.json",
        build_receipt_sha256="6" * 64,
        runtime_build_id="7" * 64,
        builder_sha256="8" * 64,
        kt_extension_sha256="9" * 64,
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


def make_model_runtime_validation_receipt(
    spec: SglangKtProcessLaunchSpec,
    snapshot_receipt: SglangKtModelSnapshotReceiptObservation,
    kernel_receipt: SglangKtKernelRuntimeValidationReceiptObservation,
) -> SglangKtModelRuntimeValidationReceiptObservation:
    assert snapshot_receipt.contract_path is not None
    assert snapshot_receipt.contract_receipt_sha256 is not None
    assert snapshot_receipt.contract_sha256 is not None
    assert snapshot_receipt.index_sha256 is not None
    assert snapshot_receipt.weight_map_entries is not None
    assert snapshot_receipt.shard_count is not None
    assert snapshot_receipt.physical_weight_bytes is not None
    is_cpu_control = (
        spec.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    )
    return SglangKtModelRuntimeValidationReceiptObservation(
        receipt_path=f"/receipts/model-rank-{spec.pipeline_rank}.json",
        receipt_size_bytes=1,
        receipt_sha256=f"{spec.pipeline_rank + 10:x}" * 64,
        schema_version=1,
        generated_at_utc="2026-07-19T00:00:00+00:00",
        validator_sha256="b" * 64,
        process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(spec),
        model_contract_path=snapshot_receipt.contract_path,
        model_path=snapshot_receipt.model_path,
        model_contract_receipt_sha256=(snapshot_receipt.contract_receipt_sha256),
        model_contract_sha256=snapshot_receipt.contract_sha256,
        model_config_sha256=snapshot_receipt.config_sha256,
        model_index_sha256=snapshot_receipt.index_sha256,
        model_weight_map_entries=snapshot_receipt.weight_map_entries,
        model_shard_count=snapshot_receipt.shard_count,
        model_physical_weight_bytes=snapshot_receipt.physical_weight_bytes,
        kernel_runtime_validation_receipt_path=kernel_receipt.receipt_path,
        kernel_runtime_validation_receipt_sha256=kernel_receipt.receipt_sha256,
        target_profile=spec.target_profile,
        gpu_uuid=spec.gpu_uuid,
        gpu_compute_capability=(8, 6),
        cpu_cores=spec.cpu_cores,
        memory_nodes=spec.memory_nodes,
        executed_cpu_backend="AMX_BF16",
        model_id=spec.model_id,
        model_revision=spec.expected_model_revision,
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
        ktransformers_method="BF16",
        resident_gpu_experts=spec.stage.resident_gpu_experts,
        attention_backend="flashinfer",
        kv_cache_dtype="bfloat16",
        max_total_tokens=spec.plan.max_total_tokens,
        static_memory_fraction=spec.plan.static_memory_fraction,
        capabilities=(
            "glm47_flash_kt_wrapper_active_v1",
            "glm47_flash_kt_wrapper_layers_1_46_v1",
            "glm47_flash_bf16_sm86_short_forward_v1",
            "kt_physical_numa_mapping_v1",
            "kt_process_cpu_affinity_v1",
            "kt_bf16_amx_executed_v1",
            (
                "glm47_flash_bf16_cpu_routed_experts_executed_v1"
                if is_cpu_control
                else "kt_bf16_cpu_gpu_hybrid_executed_v1"
            ),
        ),
        ktransformers_wrapped_expert_layers=(GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS),
        global_expert_mask_sha256="c" * 64,
        layer_one_selected_expert_ids=(4, 5, 6, 7) if is_cpu_control else (0, 1, 4, 5),
        extend_logits_sha256="d" * 64,
        decode_logits_sha256="e" * 64,
    )


def make_host_observation(
    host_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> SglangKtHostPreflightObservation:
    first_spec = host_specs[0]
    model_paths = tuple(dict.fromkeys(spec.model_path for spec in host_specs))
    weight_paths = tuple(
        dict.fromkeys(spec.ktransformers_weight_path for spec in host_specs)
    )
    snapshot_paths = tuple(dict.fromkeys((*model_paths, *weight_paths)))
    rank_zero_spec = next(
        (spec for spec in host_specs if spec.pipeline_rank == 0),
        None,
    )
    is_flash = first_spec.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
    snapshot_receipts = tuple(
        SglangKtModelSnapshotReceiptObservation(
            model_path=snapshot_path,
            model_id=first_spec.model_id,
            revision=first_spec.expected_model_revision,
            weight_format="safetensors",
            ktransformers_method=first_spec.ktransformers_method,
            config_sha256=(
                GLM_4_7_FLASH_BF16_CONFIG_SHA256 if is_flash else CONFIG_SHA256
            ),
            full_indexer_layer_starts=((0,) if is_flash else FULL_INDEXER_LAYER_STARTS),
            receipt_verified=True,
            snapshot_complete=True,
            contract_path="/contracts/glm47-flash-bf16.json" if is_flash else None,
            contract_receipt_sha256=(
                GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256 if is_flash else None
            ),
            contract_sha256=(
                GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256 if is_flash else None
            ),
            index_sha256=GLM_4_7_FLASH_INDEX_SHA256 if is_flash else None,
            weight_map_entries=9_703 if is_flash else None,
            shard_count=48 if is_flash else None,
            physical_weight_bytes=62_444_175_504 if is_flash else None,
        )
        for snapshot_path in snapshot_paths
    )
    kernel_receipts = tuple(
        make_kernel_runtime_validation_receipt(spec) for spec in host_specs if is_flash
    )
    snapshots_by_path = {receipt.model_path: receipt for receipt in snapshot_receipts}
    kernels_by_gpu = {receipt.gpu_uuid: receipt for receipt in kernel_receipts}
    bound_model_runtime_receipts = tuple(
        SglangKtBoundModelRuntimeValidationReceipt(
            binding=SglangKtModelRuntimeValidationBinding(
                process_spec_sha256=(
                    calculate_sglang_kt_process_launch_spec_sha256(spec)
                ),
                validator_sha256=model_receipt.validator_sha256,
                receipt_path=model_receipt.receipt_path,
                receipt_sha256=model_receipt.receipt_sha256,
            ),
            receipt=model_receipt,
        )
        for spec in host_specs
        if is_flash
        for model_receipt in (
            make_model_runtime_validation_receipt(
                spec,
                snapshots_by_path[spec.model_path],
                kernels_by_gpu[spec.gpu_uuid],
            ),
        )
    )
    return SglangKtHostPreflightObservation(
        node_id=first_spec.node_id,
        runtime=make_runtime(first_spec),
        runtime_validation_receipts=tuple(
            make_runtime_validation_receipt(spec) for spec in host_specs if not is_flash
        ),
        bound_model_runtime_validation_receipts=bound_model_runtime_receipts,
        kernel_runtime_validation_receipts=kernel_receipts,
        readable_directories=tuple(dict.fromkeys((*model_paths, *weight_paths))),
        model_snapshot_receipts=snapshot_receipts,
        gpu_uuids=tuple(spec.gpu_uuid for spec in host_specs),
        cpu_cores=tuple(core for spec in host_specs for core in spec.cpu_cores),
        memory_nodes=tuple(
            dict.fromkeys(node for spec in host_specs for node in spec.memory_nodes)
        ),
        hca_devices=tuple(
            dict.fromkeys(device for spec in host_specs for device in spec.hca_devices)
        ),
        available_bind_endpoints=(
            *(spec.service_endpoint for spec in host_specs),
            *(
                (rank_zero_spec.distributed_coordinator,)
                if rank_zero_spec is not None
                else ()
            ),
        ),
    )


def make_observations(
    specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[SglangKtHostPreflightObservation, ...]:
    return (
        make_host_observation((specs[0], specs[1])),
        make_host_observation((specs[2],)),
    )


def checks_for_rank(result: SglangKtPreflightFailed, rank: int) -> tuple[str, ...]:
    return tuple(
        failure.check for failure in result.failures if failure.pipeline_rank == rank
    )


def model_runtime_receipts(
    observation: SglangKtHostPreflightObservation,
) -> tuple[SglangKtModelRuntimeValidationReceiptObservation, ...]:
    return tuple(
        bound.receipt for bound in observation.bound_model_runtime_validation_receipts
    )


def replace_model_runtime_receipts(
    observation: SglangKtHostPreflightObservation,
    receipts: tuple[SglangKtModelRuntimeValidationReceiptObservation, ...],
) -> SglangKtHostPreflightObservation:
    existing = observation.bound_model_runtime_validation_receipts
    if len(existing) != len(receipts):
        if not receipts:
            return observation.model_copy(
                update={"bound_model_runtime_validation_receipts": ()}
            )
        raise AssertionError("test replacement must preserve receipt cardinality")
    return observation.model_copy(
        update={
            "bound_model_runtime_validation_receipts": tuple(
                bound.model_copy(update={"receipt": receipt})
                for bound, receipt in zip(existing, receipts, strict=True)
            )
        }
    )


def test_valid_observations_release_the_complete_process_group() -> None:
    specs = make_specs()
    observations = make_observations(specs)

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightPassed)
    assert result.process_specs == specs
    assert tuple(
        binding.pipeline_rank for binding in result.admission_bindings
    ) == tuple(spec.pipeline_rank for spec in specs)
    assert tuple(
        binding.process_spec_sha256 for binding in result.admission_bindings
    ) == tuple(calculate_sglang_kt_process_launch_spec_sha256(spec) for spec in specs)
    assert (
        SglangKtPreflightPassed.model_validate_json(result.model_dump_json()) == result
    )


def test_flash_smoke_requires_exact_executed_hybrid_runtime_receipt() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))

    passed = evaluate_sglang_kt_preflight((spec,), (observation,))
    missing_receipt = evaluate_sglang_kt_preflight(
        (spec,),
        (replace_model_runtime_receipts(observation, ()),),
    )

    assert isinstance(passed, SglangKtPreflightPassed)
    (admission_binding,) = passed.admission_bindings
    assert admission_binding.model_snapshot_receipts == (
        observation.model_snapshot_receipts[0],
    )
    assert (
        admission_binding.bound_model_runtime_validation_receipt
        == observation.bound_model_runtime_validation_receipts[0]
    )
    assert (
        admission_binding.kernel_runtime_validation_receipt
        == observation.kernel_runtime_validation_receipts[0]
    )
    assert isinstance(missing_receipt, SglangKtPreflightFailed)
    assert checks_for_rank(missing_receipt, 0) == ("runtime_validation_receipt",)


@pytest.mark.parametrize(
    "binding_update",
    (
        {"process_spec_sha256": "0" * 64},
        {"validator_sha256": "1" * 64},
        {"receipt_path": "/different/model-runtime.json"},
        {"receipt_sha256": "2" * 64},
    ),
)
def test_bound_model_runtime_receipt_rejects_self_attested_identity(
    binding_update: dict[str, object],
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    bound_receipt = make_host_observation(
        (spec,)
    ).bound_model_runtime_validation_receipts[0]

    with pytest.raises(ValidationError, match="independent binding"):
        SglangKtBoundModelRuntimeValidationReceipt(
            binding=bound_receipt.binding.model_copy(update=binding_update),
            receipt=bound_receipt.receipt,
        )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"process_spec_sha256": "0" * 64},
        {"model_contract_path": "/different/contract.json"},
        {"model_contract_receipt_sha256": "1" * 64},
        {"model_contract_sha256": "2" * 64},
        {"model_index_sha256": "3" * 64},
        {"kernel_runtime_validation_receipt_path": "/different/kernel.json"},
        {"kernel_runtime_validation_receipt_sha256": "f" * 64},
    ),
)
def test_flash_smoke_requires_exact_model_receipt_parent_bindings(
    receipt_update: dict[str, object],
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    receipt = model_runtime_receipts(observation)[0].model_copy(update=receipt_update)

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (replace_model_runtime_receipts(observation, (receipt,)),),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_passed_preflight_rejects_missing_or_changed_admission_bindings() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    result = evaluate_sglang_kt_preflight(
        (spec,),
        (make_host_observation((spec,)),),
    )
    assert isinstance(result, SglangKtPreflightPassed)
    payload = result.model_dump()

    missing_binding_payload = dict(payload)
    missing_binding_payload["admission_bindings"] = ()
    with pytest.raises(ValidationError, match="one admission binding"):
        SglangKtPreflightPassed.model_validate(missing_binding_payload)

    changed_binding_payload = result.model_dump()
    changed_binding = result.admission_bindings[0].model_dump()
    changed_binding["process_spec_sha256"] = "0" * 64
    changed_binding_payload["admission_bindings"] = (changed_binding,)
    with pytest.raises(ValidationError, match="process spec digest"):
        SglangKtPreflightPassed.model_validate(changed_binding_payload)

    (binding,) = result.admission_bindings
    bound_model_runtime_receipt = binding.bound_model_runtime_validation_receipt
    assert bound_model_runtime_receipt is not None
    forged_runtime_binding = binding.model_copy(
        update={
            "bound_model_runtime_validation_receipt": (
                bound_model_runtime_receipt.model_copy(
                    update={
                        "receipt": bound_model_runtime_receipt.receipt.model_copy(
                            update={"torch_version": "forged"}
                        )
                    }
                )
            )
        }
    )
    with pytest.raises(ValidationError, match="model runtime evidence"):
        SglangKtPreflightPassed(
            process_specs=(spec,),
            admission_bindings=(forged_runtime_binding,),
        )


def test_flash_smoke_requires_independent_file_bound_kernel_evidence() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (observation.model_copy(update={"kernel_runtime_validation_receipts": ()}),),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_flash_smoke_rejects_kernel_evidence_from_another_executable() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    (kernel_receipt,) = observation.kernel_runtime_validation_receipts

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (
            observation.model_copy(
                update={
                    "kernel_runtime_validation_receipts": (
                        kernel_receipt.model_copy(
                            update={"executable": "/different/python"}
                        ),
                    )
                }
            ),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_runtime_validation_rejects_a_capability_superset() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    (receipt,) = model_runtime_receipts(observation)
    receipt_with_unrelated_capability = receipt.model_copy(
        update={
            "capabilities": (
                *receipt.capabilities,
                "kt_tp_group_local_broadcast_v1",
            )
        }
    )

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (
            replace_model_runtime_receipts(
                observation,
                (receipt_with_unrelated_capability,),
            ),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_flash_smoke_requires_the_exact_model_contract_evidence() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    (snapshot_receipt,) = observation.model_snapshot_receipts
    without_contract = snapshot_receipt.model_copy(
        update={
            "contract_path": None,
            "contract_receipt_sha256": None,
            "contract_sha256": None,
            "index_sha256": None,
            "weight_map_entries": None,
            "shard_count": None,
            "physical_weight_bytes": None,
        }
    )

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (
            observation.model_copy(
                update={"model_snapshot_receipts": (without_contract,)}
            ),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
        "runtime_validation_receipt",
    )


def test_snapshot_contract_evidence_must_be_complete_or_absent() -> None:
    with pytest.raises(ValidationError, match="contract evidence must be complete"):
        SglangKtModelSnapshotReceiptObservation(
            model_path="/model",
            model_id=ModelId("zai-org/GLM-4.7-Flash"),
            revision="7" * 40,
            weight_format="safetensors",
            ktransformers_method="BF16",
            config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
            full_indexer_layer_starts=(0,),
            receipt_verified=True,
            snapshot_complete=True,
            contract_path="/contract.json",
        )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"ktransformers_wrapped_expert_layers": tuple(range(1, 46))},
        {
            "capabilities": (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
                "kt_bf16_cpu_gpu_hybrid_executed_v1",
            )
        },
    ),
)
def test_flash_hybrid_rejects_incomplete_wrapped_layer_evidence(
    receipt_update: dict[str, object],
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    receipt = model_runtime_receipts(observation)[0].model_copy(update=receipt_update)

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (replace_model_runtime_receipts(observation, (receipt,)),),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_flash_cpu_routed_experts_control_requires_exact_execution_receipt() -> None:
    (spec,) = build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
        make_glm_4_7_flash_bf16_cpu_routed_experts_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    (receipt,) = model_runtime_receipts(observation)

    result = evaluate_sglang_kt_preflight((spec,), (observation,))

    assert isinstance(result, SglangKtPreflightPassed)
    assert receipt.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    assert receipt.gpu_compute_capability == (8, 6)
    assert receipt.executed_cpu_backend == "AMX_BF16"
    assert receipt.resident_gpu_experts == 0
    assert "kt_bf16_cpu_gpu_hybrid_executed_v1" not in receipt.capabilities
    assert (
        receipt.ktransformers_wrapped_expert_layers
        == GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS
        == tuple(range(1, 47))
    )


def test_flash_serving_baseline_requires_exact_hybrid_execution_receipt() -> None:
    (spec,) = build_glm_4_7_flash_bf16_serving_baseline_process_launch_specs(
        make_glm_4_7_flash_bf16_serving_baseline_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    (receipt,) = model_runtime_receipts(observation)

    result = evaluate_sglang_kt_preflight((spec,), (observation,))

    assert isinstance(result, SglangKtPreflightPassed)
    assert receipt.target_profile == GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE
    assert receipt.resident_gpu_experts == 4
    assert "kt_bf16_cpu_gpu_hybrid_executed_v1" in receipt.capabilities


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"executed_cpu_backend": "AMX"},
        {"resident_gpu_experts": 1},
        {"gpu_compute_capability": (9, 0)},
        {"ktransformers_wrapped_expert_layers": tuple(range(1, 46))},
        {
            "capabilities": (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
                "glm47_flash_bf16_cpu_routed_experts_executed_v1",
            )
        },
        {
            "capabilities": (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_kt_wrapper_layers_1_46_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "glm47_flash_bf16_cpu_routed_experts_executed_v1",
            )
        },
        {
            "capabilities": (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_kt_wrapper_layers_1_46_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
            )
        },
    ),
)
def test_flash_cpu_routed_experts_control_rejects_incomplete_evidence(
    receipt_update: dict[str, object],
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
        make_glm_4_7_flash_bf16_cpu_routed_experts_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    receipt = model_runtime_receipts(observation)[0].model_copy(update=receipt_update)

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (replace_model_runtime_receipts(observation, (receipt,)),),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_flash_cpu_routed_experts_control_still_requires_assigned_cuda_gpu() -> None:
    (spec,) = build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
        make_glm_4_7_flash_bf16_cpu_routed_experts_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,)).model_copy(update={"gpu_uuids": ()})

    result = evaluate_sglang_kt_preflight((spec,), (observation,))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("gpu_uuid",)


def test_flash_smoke_rejects_unpinned_config_even_when_receipts_agree() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    unpinned_config_sha256 = "b" * 64
    snapshot_receipts = tuple(
        receipt.model_copy(update={"config_sha256": unpinned_config_sha256})
        for receipt in observation.model_snapshot_receipts
    )
    runtime_receipt = model_runtime_receipts(observation)[0].model_copy(
        update={"model_config_sha256": unpinned_config_sha256}
    )

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (
            replace_model_runtime_receipts(
                observation.model_copy(
                    update={"model_snapshot_receipts": snapshot_receipts}
                ),
                (runtime_receipt,),
            ),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert all(
        failure.expected[4] == GLM_4_7_FLASH_BF16_CONFIG_SHA256
        for failure in result.failures
    )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"target_profile": "glm52_fp8_pp3_sm86_v1"},
        {"executed_cpu_backend": "AMX"},
        {"ktransformers_method": "FP8"},
        {"resident_gpu_experts": 0},
        {"attention_backend": "nsa"},
        {"kv_cache_dtype": "fp8_e4m3"},
        {
            "capabilities": (
                "glm47_flash_kt_wrapper_active_v1",
                "glm47_flash_bf16_sm86_short_forward_v1",
                "kt_physical_numa_mapping_v1",
                "kt_process_cpu_affinity_v1",
                "kt_bf16_amx_executed_v1",
            )
        },
    ),
)
def test_flash_smoke_rejects_stale_or_incomplete_execution_evidence(
    receipt_update: dict[str, object],
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )
    observation = make_host_observation((spec,))
    receipt = model_runtime_receipts(observation)[0].model_copy(update=receipt_update)

    result = evaluate_sglang_kt_preflight(
        (spec,),
        (replace_model_runtime_receipts(observation, (receipt,)),),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)


def test_version_only_runtime_facts_cannot_release_the_process_group() -> None:
    specs = make_specs()
    observations = tuple(
        observation.model_copy(update={"runtime_validation_receipts": ()})
        for observation in make_observations(specs)
    )

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightFailed)
    assert all(
        checks_for_rank(result, rank) == ("runtime_validation_receipt",)
        for rank in range(3)
    )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"gpu_compute_capability": (9, 0)},
        {"cpu_cores": (999,)},
        {"memory_nodes": (999,)},
        {"max_total_tokens": 8_192},
        {"model_config_sha256": "b" * 64},
        {"sglang_revision": "c" * 40},
        {"transformers_distribution_version": "5.6.0.post2"},
        {"transformers_module_version": "5.6.1"},
        {"ktransformers_method": "BF16"},
        {"resident_gpu_experts": 1},
        {"attention_backend": "flashinfer"},
        {"capabilities": ("kt_tp_group_local_broadcast_v1",)},
    ),
)
def test_stale_or_incomplete_runtime_validation_receipt_fails_closed(
    receipt_update: dict[str, object],
) -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    first_receipt = dwagon.runtime_validation_receipts[0].model_copy(
        update=receipt_update
    )
    dwagon = dwagon.model_copy(
        update={
            "runtime_validation_receipts": (
                first_receipt,
                dwagon.runtime_validation_receipts[1],
            )
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)
    assert checks_for_rank(result, 1) == ()
    assert checks_for_rank(result, 2) == ()


@pytest.mark.parametrize(
    ("runtime_field", "stale_value"),
    (
        ("torch_version", "2.11.0+cu130"),
        ("cuda_version", "13.1"),
        ("sgl_kernel_build_id", "stale-sgl-kernel-build"),
        ("deep_gemm_build_id", "stale-deep-gemm-build"),
        ("kt_kernel_build_id", "stale-kt-kernel-build"),
    ),
)
def test_execution_receipt_must_match_current_runtime_artifacts(
    runtime_field: str,
    stale_value: str,
) -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    current_runtime = dwagon.runtime.model_copy(update={runtime_field: stale_value})
    dwagon = dwagon.model_copy(update={"runtime": current_runtime})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("runtime_validation_receipt",)
    assert checks_for_rank(result, 1) == ("runtime_validation_receipt",)
    assert checks_for_rank(result, 2) == ()


def test_snapshot_receipt_boundaries_must_authorize_every_pipeline_start() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    incomplete_boundaries = tuple(
        start for start in FULL_INDEXER_LAYER_STARTS if start != 58
    )
    dwagon_receipt = dwagon.model_snapshot_receipts[0].model_copy(
        update={"full_indexer_layer_starts": incomplete_boundaries}
    )
    fwuff_receipt = fwuff.model_snapshot_receipts[0].model_copy(
        update={"full_indexer_layer_starts": incomplete_boundaries}
    )

    result = evaluate_sglang_kt_preflight(
        specs,
        (
            dwagon.model_copy(update={"model_snapshot_receipts": (dwagon_receipt,)}),
            fwuff.model_copy(update={"model_snapshot_receipts": (fwuff_receipt,)}),
        ),
    )

    assert isinstance(result, SglangKtPreflightFailed)
    for rank in range(3):
        assert checks_for_rank(result, rank) == (
            "model_revision_receipt",
            "ktransformers_weight_revision_receipt",
        )


def test_runtime_mismatches_are_aggregated_for_each_affected_stage() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_runtime = SglangKtRuntimeObservation(
        executable="/opt/wrong/bin/python",
        python_implementation="PyPy",
        python_version=SglangKtPythonVersionObservation(
            major=3,
            minor=10,
            patch=16,
        ),
        sglang_revision="5" * 40,
        ktransformers_revision="6" * 40,
        transformers_distribution_version="5.6.0.post2",
        transformers_module_version="5.6.0.post2",
        torch_version=TORCH_VERSION,
        cuda_version=CUDA_VERSION,
        sgl_kernel_build_id=SGL_KERNEL_BUILD_ID,
        deep_gemm_build_id=DEEP_GEMM_BUILD_ID,
        kt_kernel_build_id=KT_KERNEL_BUILD_ID,
    )
    dwagon = dwagon.model_copy(update={"runtime": bad_runtime})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    expected_checks = (
        "python_executable",
        "python_implementation",
        "python_version",
        "sglang_revision",
        "ktransformers_revision",
        "transformers_distribution_version",
        "transformers_module_version",
    )
    assert checks_for_rank(result, 0) == expected_checks
    assert checks_for_rank(result, 1) == expected_checks
    assert checks_for_rank(result, 2) == ()
    assert all(
        failure.message.startswith(
            f"pipeline rank {failure.pipeline_rank} on {failure.node_id}:"
        )
        for failure in result.failures
    )


def test_unobserved_runtime_facts_fail_closed() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    dwagon = dwagon.model_copy(update={"runtime": SglangKtRuntimeObservation()})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    expected_checks = (
        "python_executable",
        "python_implementation",
        "python_version",
        "sglang_revision",
        "ktransformers_revision",
        "transformers_distribution_version",
        "transformers_module_version",
        "runtime_validation_receipt",
    )
    assert checks_for_rank(result, 0) == expected_checks
    assert all(
        failure.observed == ("<unobserved>",)
        for failure in result.failures
        if failure.pipeline_rank == 0 and failure.check != "runtime_validation_receipt"
    )


def test_missing_stage_resources_and_ports_fail_as_one_group() -> None:
    specs = make_specs()
    _dwagon, fwuff = make_observations(specs)
    empty_dwagon = SglangKtHostPreflightObservation(
        node_id=specs[0].node_id,
        runtime=make_runtime(specs[0]),
    )

    result = evaluate_sglang_kt_preflight(specs, (empty_dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    common_checks = (
        "model_path",
        "ktransformers_weight_path",
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
        "runtime_validation_receipt",
        "gpu_uuid",
        "cpu_cores",
        "memory_nodes",
        "hca_devices",
        "service_endpoint",
    )
    assert checks_for_rank(result, 0) == (
        *common_checks,
        "distributed_coordinator",
    )
    assert checks_for_rank(result, 1) == common_checks
    assert checks_for_rank(result, 2) == ()


def test_model_receipt_must_match_path_model_revision_and_completeness() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_receipt = SglangKtModelSnapshotReceiptObservation(
        model_path=specs[0].model_path,
        model_id=ModelId("zai-org/GLM-5-FP8"),
        revision="9" * 40,
        weight_format="safetensors",
        ktransformers_method="FP8",
        config_sha256=CONFIG_SHA256,
        full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
        receipt_verified=False,
        snapshot_complete=False,
    )
    dwagon = dwagon.model_copy(update={"model_snapshot_receipts": (bad_receipt,)})

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert checks_for_rank(result, 1) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert result.failures[0].expected == (
        "zai-org/GLM-5.2-FP8",
        specs[0].expected_model_revision,
        "safetensors",
        "FP8",
        "config_sha256=<verified>",
        "pipeline starts on verified full indexers",
        "receipt_verified=True",
        "snapshot_complete=True",
    )
    assert result.failures[0].observed == (
        "zai-org/GLM-5-FP8",
        "9" * 40,
        "safetensors",
        "FP8",
        CONFIG_SHA256,
        "full_indexer_layer_starts="
        + ",".join(str(start) for start in FULL_INDEXER_LAYER_STARTS),
        "receipt_verified=False",
        "snapshot_complete=False",
    )


def test_ktransformers_weight_receipt_must_be_exact_and_compatible() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    bad_weight_receipt = SglangKtModelSnapshotReceiptObservation(
        model_path=specs[0].ktransformers_weight_path,
        model_id=specs[0].model_id,
        revision=specs[0].expected_model_revision,
        weight_format="safetensors",
        ktransformers_method="BF16",
        config_sha256=CONFIG_SHA256,
        full_indexer_layer_starts=FULL_INDEXER_LAYER_STARTS,
        receipt_verified=True,
        snapshot_complete=True,
    )
    dwagon = dwagon.model_copy(
        update={
            "model_snapshot_receipts": (bad_weight_receipt,),
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )
    assert checks_for_rank(result, 1) == (
        "model_revision_receipt",
        "ktransformers_weight_revision_receipt",
    )


def test_each_planned_endpoint_requires_an_injected_availability_fact() -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    dwagon = dwagon.model_copy(
        update={
            "available_bind_endpoints": (specs[0].service_endpoint,),
        }
    )

    result = evaluate_sglang_kt_preflight(specs, (dwagon, fwuff))

    assert isinstance(result, SglangKtPreflightFailed)
    assert checks_for_rank(result, 0) == ("distributed_coordinator",)
    assert checks_for_rank(result, 1) == ("service_endpoint",)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_ambiguous_host_observations_fail_closed(duplicate: bool) -> None:
    specs = make_specs()
    dwagon, fwuff = make_observations(specs)
    observations = (dwagon, dwagon, fwuff) if duplicate else (dwagon,)

    result = evaluate_sglang_kt_preflight(specs, observations)

    assert isinstance(result, SglangKtPreflightFailed)
    affected_ranks = (0, 1) if duplicate else (2,)
    for rank in affected_ranks:
        assert checks_for_rank(result, rank) == ("host_observation",)
    assert not isinstance(result, SglangKtPreflightPassed)


def test_host_observations_reject_ambiguous_duplicate_facts() -> None:
    specs = make_specs()

    with pytest.raises(ValidationError, match="gpu_uuids observations must be unique"):
        SglangKtHostPreflightObservation(
            node_id=specs[0].node_id,
            runtime=make_runtime(specs[0]),
            gpu_uuids=(specs[0].gpu_uuid, specs[0].gpu_uuid),
        )


def test_empty_or_duplicate_process_groups_are_rejected_before_release() -> None:
    specs = make_specs()

    with pytest.raises(ValueError, match="requires process specs"):
        evaluate_sglang_kt_preflight((), ())
    with pytest.raises(ValueError, match="ranks must be unique"):
        evaluate_sglang_kt_preflight((specs[0], specs[0]), ())
    with pytest.raises(ValueError, match="ranks must be contiguous"):
        evaluate_sglang_kt_preflight((specs[0], specs[2]), ())


def test_process_groups_must_share_one_canonical_launch_plan() -> None:
    specs = make_specs()
    other_plan = make_plan().model_copy(update={"model_revision": "7" * 40})
    other_specs = build_glm_5_2_fp8_process_launch_specs(other_plan, PYTHON_EXECUTABLE)

    with pytest.raises(ValueError, match="share one launch plan"):
        evaluate_sglang_kt_preflight(
            (specs[0], other_specs[1], specs[2]),
            (),
        )
