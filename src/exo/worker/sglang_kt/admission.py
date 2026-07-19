from pathlib import Path

from anyio import to_process

from exo.shared.types.worker.sglang_kt import AbsoluteRuntimePath, GpuUuid
from exo.worker.sglang_kt.launch_spec import SglangKtProcessLaunchSpec
from exo.worker.sglang_kt.model_contract import (
    SglangKtModelContractError,
    verify_sglang_kt_model_snapshot,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    SglangKtModelRuntimeValidationReceiptError,
    load_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtBoundModelRuntimeValidationReceipt,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtRankAdmissionBinding,
    validate_sglang_kt_rank_admission_binding,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptError,
    SglangKtKernelRuntimeValidationReceiptObservation,
    load_sglang_kt_kernel_runtime_validation_receipt,
)


class SglangKtAdmissionEvidenceError(RuntimeError):
    """Raised when evidence changed between preflight and process launch."""


def _verify_model_snapshot_receipt(
    receipt: SglangKtModelSnapshotReceiptObservation,
) -> None:
    if (
        receipt.contract_path is None
        or receipt.contract_receipt_sha256 is None
        or receipt.contract_sha256 is None
        or receipt.index_sha256 is None
        or receipt.weight_map_entries is None
        or receipt.shard_count is None
        or receipt.physical_weight_bytes is None
    ):
        raise SglangKtAdmissionEvidenceError(
            "launch admission requires complete model contract evidence"
        )
    try:
        verified = verify_sglang_kt_model_snapshot(
            Path(receipt.model_path),
            Path(receipt.contract_path),
            expected_contract_sha256=receipt.contract_sha256,
            expected_model_id=receipt.model_id,
            expected_revision=receipt.revision,
            expected_ktransformers_method=receipt.ktransformers_method,
        )
    except SglangKtModelContractError as error:
        raise SglangKtAdmissionEvidenceError(
            f"model snapshot changed after preflight: {receipt.model_path}"
        ) from error

    if (
        verified.model_path != receipt.model_path
        or verified.model_id != receipt.model_id
        or verified.revision != receipt.revision
        or verified.weight_format != receipt.weight_format
        or verified.ktransformers_method != receipt.ktransformers_method
        or verified.full_indexer_layer_starts != receipt.full_indexer_layer_starts
        or verified.contract_path != receipt.contract_path
        or verified.contract_receipt_sha256 != receipt.contract_receipt_sha256
        or verified.contract_sha256 != receipt.contract_sha256
        or verified.config_sha256 != receipt.config_sha256
        or verified.index_sha256 != receipt.index_sha256
        or verified.weight_map_entries != receipt.weight_map_entries
        or verified.shard_count != receipt.shard_count
        or verified.physical_weight_bytes != receipt.physical_weight_bytes
    ):
        raise SglangKtAdmissionEvidenceError(
            f"model snapshot evidence changed after preflight: {receipt.model_path}"
        )


def _verify_kernel_runtime_receipt(
    receipt: SglangKtKernelRuntimeValidationReceiptObservation,
) -> None:
    try:
        reloaded = load_sglang_kt_kernel_runtime_validation_receipt(
            Path(receipt.receipt_path),
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    except SglangKtKernelRuntimeValidationReceiptError as error:
        raise SglangKtAdmissionEvidenceError(
            "kernel runtime validation receipt changed after preflight"
        ) from error
    if reloaded != receipt:
        raise SglangKtAdmissionEvidenceError(
            "kernel runtime validation evidence changed after preflight"
        )


def _verify_model_runtime_receipt(
    bound_receipt: SglangKtBoundModelRuntimeValidationReceipt,
) -> None:
    binding = bound_receipt.binding
    receipt = bound_receipt.receipt
    try:
        reloaded = load_sglang_kt_model_runtime_validation_receipt(
            Path(binding.receipt_path),
            expected_validator_sha256=binding.validator_sha256,
            expected_process_spec_sha256=binding.process_spec_sha256,
            expected_model_contract_receipt_sha256=(
                receipt.model_contract_receipt_sha256
            ),
            expected_kernel_receipt_sha256=(
                receipt.kernel_runtime_validation_receipt_sha256
            ),
            expected_receipt_sha256=binding.receipt_sha256,
        )
    except SglangKtModelRuntimeValidationReceiptError as error:
        raise SglangKtAdmissionEvidenceError(
            "model runtime validation receipt changed after preflight"
        ) from error
    if reloaded != receipt:
        raise SglangKtAdmissionEvidenceError(
            "model runtime validation evidence changed after preflight"
        )


def verify_sglang_kt_local_admission_bindings_sync(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
    admission_bindings: tuple[SglangKtRankAdmissionBinding, ...],
) -> None:
    """Revalidate every local binding without launching or probing readiness."""

    if not process_specs:
        raise SglangKtAdmissionEvidenceError(
            "local launch admission requires process specs"
        )
    specs_by_rank = {spec.pipeline_rank: spec for spec in process_specs}
    bindings_by_rank = {
        binding.pipeline_rank: binding for binding in admission_bindings
    }
    if (
        len(specs_by_rank) != len(process_specs)
        or len(bindings_by_rank) != len(admission_bindings)
        or set(specs_by_rank) != set(bindings_by_rank)
    ):
        raise SglangKtAdmissionEvidenceError(
            "local launch admission requires one binding per process rank"
        )

    snapshots_by_path: dict[
        AbsoluteRuntimePath, SglangKtModelSnapshotReceiptObservation
    ] = {}
    kernel_receipts_by_gpu: dict[
        GpuUuid, SglangKtKernelRuntimeValidationReceiptObservation
    ] = {}
    model_runtime_receipts_by_path: dict[
        AbsoluteRuntimePath, SglangKtBoundModelRuntimeValidationReceipt
    ] = {}
    for pipeline_rank in sorted(specs_by_rank):
        process_spec = specs_by_rank[pipeline_rank]
        binding = bindings_by_rank[pipeline_rank]
        try:
            validate_sglang_kt_rank_admission_binding(process_spec, binding)
        except ValueError as error:
            raise SglangKtAdmissionEvidenceError(
                f"rank {pipeline_rank} admission binding no longer matches its spec"
            ) from error
        for receipt in binding.model_snapshot_receipts:
            previous = snapshots_by_path.setdefault(receipt.model_path, receipt)
            if previous != receipt:
                raise SglangKtAdmissionEvidenceError(
                    "local ranks disagree about model snapshot evidence"
                )
        kernel_receipt = binding.kernel_runtime_validation_receipt
        if kernel_receipt is not None:
            previous_kernel = kernel_receipts_by_gpu.setdefault(
                kernel_receipt.gpu_uuid, kernel_receipt
            )
            if previous_kernel != kernel_receipt:
                raise SglangKtAdmissionEvidenceError(
                    "local ranks disagree about kernel runtime evidence"
                )
        bound_model_runtime_receipt = binding.bound_model_runtime_validation_receipt
        if bound_model_runtime_receipt is not None:
            previous_model_runtime = model_runtime_receipts_by_path.setdefault(
                bound_model_runtime_receipt.binding.receipt_path,
                bound_model_runtime_receipt,
            )
            if previous_model_runtime != bound_model_runtime_receipt:
                raise SglangKtAdmissionEvidenceError(
                    "local ranks disagree about model runtime evidence"
                )

    for receipt in snapshots_by_path.values():
        _verify_model_snapshot_receipt(receipt)
    for receipt in kernel_receipts_by_gpu.values():
        _verify_kernel_runtime_receipt(receipt)
    for receipt in model_runtime_receipts_by_path.values():
        _verify_model_runtime_receipt(receipt)


async def verify_sglang_kt_local_admission_bindings(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
    admission_bindings: tuple[SglangKtRankAdmissionBinding, ...],
) -> None:
    """Run potentially expensive artifact hashing outside the event loop."""

    await to_process.run_sync(
        verify_sglang_kt_local_admission_bindings_sync,
        process_specs,
        admission_bindings,
        cancellable=True,
    )
