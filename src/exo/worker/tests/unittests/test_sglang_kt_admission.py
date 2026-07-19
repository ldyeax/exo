from pathlib import Path

import pytest
from pydantic import ValidationError

from exo.shared.types.common import ModelId
from exo.shared.types.worker.sglang_kt import KTransformersMethod
from exo.worker.sglang_kt import admission
from exo.worker.sglang_kt.admission import (
    SglangKtAdmissionEvidenceError,
    verify_sglang_kt_local_admission_bindings_sync,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    SglangKtModelRuntimeValidationReceiptObservation,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtBoundModelRuntimeValidationReceipt,
    SglangKtModelRuntimeValidationBinding,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtRankAdmissionBinding,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_plan,
)
from exo.worker.tests.unittests.test_sglang_kt_preflight import (
    CUDA_VERSION,
    TORCH_VERSION,
    make_model_runtime_validation_receipt,
)

INDEX_SHA256 = "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
SGL_KERNEL_BUILD_ID = "1" * 64
DEEP_GEMM_BUILD_ID = "2" * 64
KT_KERNEL_BUILD_ID = "3" * 64


def make_spec() -> SglangKtProcessLaunchSpec:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(),
        PYTHON_EXECUTABLE,
    )
    return spec


def make_snapshot_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtModelSnapshotReceiptObservation:
    return SglangKtModelSnapshotReceiptObservation(
        model_path=spec.model_path,
        model_id=spec.model_id,
        revision=spec.expected_model_revision,
        weight_format="safetensors",
        ktransformers_method=spec.ktransformers_method,
        config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        full_indexer_layer_starts=(0,),
        receipt_verified=True,
        snapshot_complete=True,
        contract_path="/contracts/glm47-flash-bf16.json",
        contract_receipt_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        index_sha256=INDEX_SHA256,
        weight_map_entries=9_703,
        shard_count=48,
        physical_weight_bytes=62_444_175_504,
    )


def make_verified_snapshot(
    receipt: SglangKtModelSnapshotReceiptObservation,
) -> SglangKtVerifiedModelSnapshot:
    assert receipt.contract_path is not None
    assert receipt.contract_receipt_sha256 is not None
    assert receipt.contract_sha256 is not None
    assert receipt.index_sha256 is not None
    assert receipt.weight_map_entries is not None
    assert receipt.shard_count is not None
    assert receipt.physical_weight_bytes is not None
    return SglangKtVerifiedModelSnapshot(
        model_path=receipt.model_path,
        model_id=receipt.model_id,
        revision=receipt.revision,
        weight_format=receipt.weight_format,
        ktransformers_method=receipt.ktransformers_method,
        full_indexer_layer_starts=receipt.full_indexer_layer_starts,
        contract_path=receipt.contract_path,
        contract_receipt_sha256=receipt.contract_receipt_sha256,
        contract_sha256=receipt.contract_sha256,
        config_sha256=receipt.config_sha256,
        index_sha256=receipt.index_sha256,
        weight_map_entries=receipt.weight_map_entries,
        shard_count=receipt.shard_count,
        physical_weight_bytes=receipt.physical_weight_bytes,
    )


def make_kernel_receipt(
    spec: SglangKtProcessLaunchSpec,
) -> SglangKtKernelRuntimeValidationReceiptObservation:
    return SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path=f"/receipts/kernel-rank-{spec.pipeline_rank}.json",
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


def make_binding(
    spec: SglangKtProcessLaunchSpec,
    snapshot_receipt: SglangKtModelSnapshotReceiptObservation,
    kernel_receipt: SglangKtKernelRuntimeValidationReceiptObservation,
) -> SglangKtRankAdmissionBinding:
    model_runtime_receipt = make_model_runtime_validation_receipt(
        spec,
        snapshot_receipt,
        kernel_receipt,
    )
    return SglangKtRankAdmissionBinding(
        pipeline_rank=spec.pipeline_rank,
        process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(spec),
        model_snapshot_receipts=(snapshot_receipt,),
        bound_model_runtime_validation_receipt=(
            SglangKtBoundModelRuntimeValidationReceipt(
                binding=SglangKtModelRuntimeValidationBinding(
                    process_spec_sha256=(
                        calculate_sglang_kt_process_launch_spec_sha256(spec)
                    ),
                    validator_sha256=model_runtime_receipt.validator_sha256,
                    receipt_path=model_runtime_receipt.receipt_path,
                    receipt_sha256=model_runtime_receipt.receipt_sha256,
                ),
                receipt=model_runtime_receipt,
            )
        ),
        kernel_runtime_validation_receipt=kernel_receipt,
    )


def binding_model_runtime_receipt(
    binding: SglangKtRankAdmissionBinding,
) -> SglangKtModelRuntimeValidationReceiptObservation:
    bound = binding.bound_model_runtime_validation_receipt
    assert bound is not None
    return bound.receipt


def install_unchanged_evidence_loaders(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_receipt: SglangKtModelSnapshotReceiptObservation,
    kernel_receipt: SglangKtKernelRuntimeValidationReceiptObservation,
    model_runtime_receipts: tuple[
        SglangKtModelRuntimeValidationReceiptObservation, ...
    ],
) -> tuple[list[Path], list[Path], list[Path]]:
    model_paths: list[Path] = []
    kernel_paths: list[Path] = []
    model_runtime_paths: list[Path] = []
    model_runtime_receipts_by_path = {
        Path(receipt.receipt_path): receipt for receipt in model_runtime_receipts
    }

    def verify_model_snapshot(
        snapshot_path: Path,
        contract_path: Path,
        *,
        expected_contract_sha256: str,
        expected_model_id: ModelId,
        expected_revision: str,
        expected_ktransformers_method: KTransformersMethod,
    ) -> SglangKtVerifiedModelSnapshot:
        assert contract_path == Path(snapshot_receipt.contract_path or "")
        assert expected_contract_sha256 == snapshot_receipt.contract_sha256
        assert expected_model_id == snapshot_receipt.model_id
        assert expected_revision == snapshot_receipt.revision
        assert expected_ktransformers_method == snapshot_receipt.ktransformers_method
        model_paths.append(snapshot_path)
        return make_verified_snapshot(snapshot_receipt)

    def load_kernel_receipt(
        path: Path,
        *,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtKernelRuntimeValidationReceiptObservation:
        assert expected_receipt_sha256 == kernel_receipt.receipt_sha256
        kernel_paths.append(path)
        return kernel_receipt

    def load_model_runtime_receipt(
        path: Path,
        *,
        expected_validator_sha256: str,
        expected_process_spec_sha256: str,
        expected_model_contract_receipt_sha256: str,
        expected_kernel_receipt_sha256: str,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtModelRuntimeValidationReceiptObservation:
        model_runtime_receipt = model_runtime_receipts_by_path[path]
        assert expected_validator_sha256 == model_runtime_receipt.validator_sha256
        assert expected_process_spec_sha256 == model_runtime_receipt.process_spec_sha256
        assert (
            expected_model_contract_receipt_sha256
            == model_runtime_receipt.model_contract_receipt_sha256
        )
        assert (
            expected_kernel_receipt_sha256
            == model_runtime_receipt.kernel_runtime_validation_receipt_sha256
        )
        assert expected_receipt_sha256 == model_runtime_receipt.receipt_sha256
        model_runtime_paths.append(path)
        return model_runtime_receipt

    monkeypatch.setattr(
        admission,
        "verify_sglang_kt_model_snapshot",
        verify_model_snapshot,
    )
    monkeypatch.setattr(
        admission,
        "load_sglang_kt_kernel_runtime_validation_receipt",
        load_kernel_receipt,
    )
    monkeypatch.setattr(
        admission,
        "load_sglang_kt_model_runtime_validation_receipt",
        load_model_runtime_receipt,
    )
    return model_paths, kernel_paths, model_runtime_paths


def test_revalidates_unchanged_file_bound_evidence_without_external_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt)
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    model_paths, kernel_paths, model_runtime_paths = install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )

    verify_sglang_kt_local_admission_bindings_sync((spec,), (binding,))

    assert model_paths == [Path(snapshot_receipt.model_path)]
    assert kernel_paths == [Path(kernel_receipt.receipt_path)]
    assert model_runtime_paths == [Path(model_runtime_receipt.receipt_path)]


def test_rejects_process_spec_digest_mutation_before_loading_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt)
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    model_paths, kernel_paths, model_runtime_paths = install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )
    changed_spec = spec.model_copy(update={"executable": "/different/python"})

    with pytest.raises(SglangKtAdmissionEvidenceError, match="no longer matches"):
        verify_sglang_kt_local_admission_bindings_sync(
            (changed_spec,),
            (binding,),
        )

    assert model_paths == []
    assert kernel_paths == []
    assert model_runtime_paths == []


def test_rejects_local_process_and_binding_rank_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt).model_copy(
        update={"pipeline_rank": 1}
    )
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    model_paths, kernel_paths, model_runtime_paths = install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )

    with pytest.raises(SglangKtAdmissionEvidenceError, match="one binding per"):
        verify_sglang_kt_local_admission_bindings_sync((spec,), (binding,))

    assert model_paths == []
    assert kernel_paths == []
    assert model_runtime_paths == []


def test_rejects_model_snapshot_evidence_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt)
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )

    def return_changed_snapshot(
        *_args: object,
        **_kwargs: object,
    ) -> SglangKtVerifiedModelSnapshot:
        return make_verified_snapshot(snapshot_receipt).model_copy(
            update={"config_sha256": "f" * 64}
        )

    monkeypatch.setattr(
        admission,
        "verify_sglang_kt_model_snapshot",
        return_changed_snapshot,
    )

    with pytest.raises(
        SglangKtAdmissionEvidenceError,
        match="model snapshot evidence changed",
    ):
        verify_sglang_kt_local_admission_bindings_sync((spec,), (binding,))


def test_rejects_kernel_receipt_reload_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt)
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )

    def return_changed_kernel_receipt(
        _path: Path,
        *,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtKernelRuntimeValidationReceiptObservation:
        assert expected_receipt_sha256 == kernel_receipt.receipt_sha256
        return kernel_receipt.model_copy(
            update={"receipt_size_bytes": kernel_receipt.receipt_size_bytes + 1}
        )

    monkeypatch.setattr(
        admission,
        "load_sglang_kt_kernel_runtime_validation_receipt",
        return_changed_kernel_receipt,
    )

    with pytest.raises(
        SglangKtAdmissionEvidenceError,
        match="kernel runtime validation evidence changed",
    ):
        verify_sglang_kt_local_admission_bindings_sync((spec,), (binding,))


def test_rejects_model_runtime_receipt_reload_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)
    kernel_receipt = make_kernel_receipt(spec)
    binding = make_binding(spec, snapshot_receipt, kernel_receipt)
    model_runtime_receipt = binding_model_runtime_receipt(binding)
    install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        (model_runtime_receipt,),
    )

    def return_changed_model_runtime_receipt(
        _path: Path,
        *,
        expected_validator_sha256: str,
        expected_process_spec_sha256: str,
        expected_model_contract_receipt_sha256: str,
        expected_kernel_receipt_sha256: str,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtModelRuntimeValidationReceiptObservation:
        del (
            expected_process_spec_sha256,
            expected_validator_sha256,
            expected_model_contract_receipt_sha256,
            expected_kernel_receipt_sha256,
            expected_receipt_sha256,
        )
        return model_runtime_receipt.model_copy(
            update={"receipt_size_bytes": model_runtime_receipt.receipt_size_bytes + 1}
        )

    monkeypatch.setattr(
        admission,
        "load_sglang_kt_model_runtime_validation_receipt",
        return_changed_model_runtime_receipt,
    )

    with pytest.raises(
        SglangKtAdmissionEvidenceError,
        match="model runtime validation evidence changed",
    ):
        verify_sglang_kt_local_admission_bindings_sync((spec,), (binding,))


def test_deduplicates_shared_parent_evidence_and_reloads_each_rank_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_spec = make_spec()
    stage_zero = base_spec.plan.stages[0]
    stage_one = stage_zero.model_copy(update={"pipeline_rank": 1})
    two_rank_plan = base_spec.plan.model_copy(
        update={"stages": (stage_zero, stage_one)}
    )
    spec_zero = base_spec.model_copy(update={"plan": two_rank_plan})
    spec_one = base_spec.model_copy(update={"plan": two_rank_plan, "pipeline_rank": 1})
    snapshot_receipt = make_snapshot_receipt(spec_zero)
    kernel_receipt = make_kernel_receipt(spec_zero)
    bindings = (
        make_binding(spec_zero, snapshot_receipt, kernel_receipt),
        make_binding(spec_one, snapshot_receipt, kernel_receipt),
    )
    model_runtime_receipts = tuple(
        receipt
        for binding in bindings
        for receipt in (binding_model_runtime_receipt(binding),)
    )
    assert len(model_runtime_receipts) == 2
    model_paths, kernel_paths, model_runtime_paths = install_unchanged_evidence_loaders(
        monkeypatch,
        snapshot_receipt,
        kernel_receipt,
        model_runtime_receipts,
    )

    verify_sglang_kt_local_admission_bindings_sync(
        (spec_zero, spec_one),
        bindings,
    )

    assert model_paths == [Path(snapshot_receipt.model_path)]
    assert kernel_paths == [Path(kernel_receipt.receipt_path)]
    assert model_runtime_paths == [
        Path(receipt.receipt_path) for receipt in model_runtime_receipts
    ]


def test_binding_rejects_duplicate_snapshot_paths() -> None:
    spec = make_spec()
    snapshot_receipt = make_snapshot_receipt(spec)

    with pytest.raises(ValidationError, match="snapshot paths must be unique"):
        SglangKtRankAdmissionBinding(
            pipeline_rank=spec.pipeline_rank,
            process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(spec),
            model_snapshot_receipts=(snapshot_receipt, snapshot_receipt),
            kernel_runtime_validation_receipt=make_kernel_receipt(spec),
        )
