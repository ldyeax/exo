import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    KTransformersMethod,
)
from exo.worker.sglang_kt import admission
from exo.worker.sglang_kt.admission import (
    verify_sglang_kt_local_admission_bindings_sync,
)
from exo.worker.sglang_kt.launch_spec import (
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.preflight import (
    SglangKtModelRuntimeValidationBinding,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtPreflightPassed,
    SglangKtPythonVersionObservation,
    SglangKtRuntimeObservation,
    evaluate_sglang_kt_preflight,
)
from exo.worker.sglang_kt.preflight_collector import (
    LocalSglangKtModelRuntimeValidationProbe,
    collect_sglang_kt_local_host_preflight_observation,
)
from exo.worker.sglang_kt.receipt_io import canonical_sglang_kt_json
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from exo.worker.tests.unittests.test_sglang_kt_admission import (
    make_verified_snapshot,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_plan,
)
from exo.worker.tests.unittests.test_sglang_kt_model_runtime_validation_receipt import (
    VALIDATOR_SHA256,
    make_receipt,
)
from exo.worker.tests.unittests.test_sglang_kt_preflight_collector import (
    StaticInventoryProbe,
    StaticKernelRuntimeValidationProbe,
    StaticRuntimeProbe,
    SuccessfulPortProbe,
    make_glm_4_7_snapshot_receipt,
    make_gpu_resource,
    make_inventory,
)

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]

TORCH_VERSION = "2.9.1+cu128"
CUDA_VERSION = "12.8"
SGL_KERNEL_BUILD_ID = "5" * 64
DEEP_GEMM_BUILD_ID = "6" * 64
KT_KERNEL_BUILD_ID = "7" * 64


def _object_field(parent: JsonObject, name: str) -> JsonObject:
    value = parent[name]
    if not isinstance(value, dict):
        raise AssertionError(f"{name} is not an object")
    return value


def _write_bound_model_receipt(
    path: Path,
    process_spec_sha256: str,
    snapshot_receipt: SglangKtModelSnapshotReceiptObservation,
    kernel_receipt: SglangKtKernelRuntimeValidationReceiptObservation,
    *,
    gpu_uuid: str,
    cpu_cores: tuple[int, ...],
    memory_nodes: tuple[int, ...],
    resident_gpu_experts: int,
) -> str:
    receipt = cast(JsonObject, make_receipt(cpu_control=False))
    parents = _object_field(receipt, "parents")
    parents["process_spec_sha256"] = process_spec_sha256

    assert snapshot_receipt.contract_path is not None
    assert snapshot_receipt.contract_receipt_sha256 is not None
    model_contract = _object_field(parents, "model_contract")
    model_contract["path"] = snapshot_receipt.contract_path
    model_contract["receipt_sha256"] = snapshot_receipt.contract_receipt_sha256

    kernel_parent = _object_field(parents, "kernel_runtime_validation")
    kernel_parent["receipt_path"] = kernel_receipt.receipt_path
    kernel_parent["receipt_sha256"] = kernel_receipt.receipt_sha256

    runtime = _object_field(receipt, "runtime")
    runtime["gpu_uuid"] = gpu_uuid
    runtime["cpu_cores"] = list(cpu_cores)
    runtime["memory_nodes"] = list(memory_nodes)
    runtime["resident_gpu_experts"] = resident_gpu_experts
    runtime["torch_version"] = TORCH_VERSION
    runtime["cuda_version"] = CUDA_VERSION
    runtime["sgl_kernel_build_id"] = SGL_KERNEL_BUILD_ID
    runtime["deep_gemm_build_id"] = DEEP_GEMM_BUILD_ID
    runtime["kt_kernel_build_id"] = KT_KERNEL_BUILD_ID

    contents = canonical_sglang_kt_json(receipt)
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def _make_runtime_observation(
    executable: AbsoluteRuntimePath,
    sglang_revision: GitRevision,
    ktransformers_revision: GitRevision,
    transformers_distribution_version: str,
    transformers_module_version: str,
) -> SglangKtRuntimeObservation:
    return SglangKtRuntimeObservation(
        executable=executable,
        python_implementation="CPython",
        python_version=SglangKtPythonVersionObservation(major=3, minor=13, patch=7),
        sglang_revision=sglang_revision,
        ktransformers_revision=ktransformers_revision,
        transformers_distribution_version=transformers_distribution_version,
        transformers_module_version=transformers_module_version,
        torch_version=TORCH_VERSION,
        cuda_version=CUDA_VERSION,
        sgl_kernel_build_id=SGL_KERNEL_BUILD_ID,
        deep_gemm_build_id=DEEP_GEMM_BUILD_ID,
        kt_kernel_build_id=KT_KERNEL_BUILD_ID,
    )


@dataclass(frozen=True)
class _SnapshotFilesystemProbe:
    receipt: SglangKtModelSnapshotReceiptObservation

    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
        return path == self.receipt.model_path

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation | None:
        if (
            path != self.receipt.model_path
            or model_id != self.receipt.model_id
            or revision != self.receipt.revision
        ):
            return None
        return self.receipt


def test_real_file_bound_model_receipt_survives_collection_and_admission_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(resident_gpu_experts=2),
        PYTHON_EXECUTABLE,
    )
    process_spec_sha256 = calculate_sglang_kt_process_launch_spec_sha256(spec)
    snapshot_receipt = make_glm_4_7_snapshot_receipt(spec)
    kernel_receipt = SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path="/receipts/kernel-glm47-rank-0.json",
        receipt_size_bytes=1,
        receipt_sha256="3" * 64,
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
        build_receipt_path="/receipts/build-glm47-rank-0.json",
        build_receipt_sha256="8" * 64,
        runtime_build_id="9" * 64,
        builder_sha256="a" * 64,
        kt_extension_sha256="b" * 64,
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
    model_receipt_path = tmp_path / "glm47-model-runtime-validation.json"
    model_receipt_sha256 = _write_bound_model_receipt(
        model_receipt_path,
        process_spec_sha256,
        snapshot_receipt,
        kernel_receipt,
        gpu_uuid=spec.gpu_uuid,
        cpu_cores=spec.cpu_cores,
        memory_nodes=spec.memory_nodes,
        resident_gpu_experts=spec.stage.resident_gpu_experts,
    )
    model_probe = LocalSglangKtModelRuntimeValidationProbe(
        (
            SglangKtModelRuntimeValidationBinding(
                process_spec_sha256=process_spec_sha256,
                validator_sha256=VALIDATOR_SHA256,
                receipt_path=str(model_receipt_path),
                receipt_sha256=model_receipt_sha256,
            ),
        )
    )
    runtime = _make_runtime_observation(
        spec.executable,
        spec.expected_sglang_revision,
        spec.expected_ktransformers_revision,
        spec.required_transformers_distribution_version,
        spec.required_transformers_module_version,
    )

    observation = collect_sglang_kt_local_host_preflight_observation(
        (spec,),
        gpu_resources=(make_gpu_resource(spec),),
        runtime_probe=StaticRuntimeProbe(runtime),
        filesystem_probe=_SnapshotFilesystemProbe(snapshot_receipt),
        inventory_probe=StaticInventoryProbe(make_inventory((spec,))),
        port_probe=SuccessfulPortProbe(),
        kernel_runtime_validation_probe=StaticKernelRuntimeValidationProbe(
            lambda _process_spec: kernel_receipt
        ),
        model_runtime_validation_probe=model_probe,
    )

    preflight = evaluate_sglang_kt_preflight((spec,), (observation,))

    assert isinstance(preflight, SglangKtPreflightPassed)
    assert len(observation.bound_model_runtime_validation_receipts) == 1
    observed_model_receipt = observation.bound_model_runtime_validation_receipts[
        0
    ].receipt
    assert observed_model_receipt.torch_version == TORCH_VERSION
    assert observed_model_receipt.cuda_version == CUDA_VERSION
    assert observed_model_receipt.receipt_sha256 == model_receipt_sha256

    def verify_snapshot(
        snapshot_path: Path,
        contract_path: Path,
        *,
        expected_contract_sha256: str,
        expected_model_id: ModelId,
        expected_revision: GitRevision,
        expected_ktransformers_method: KTransformersMethod,
    ) -> SglangKtVerifiedModelSnapshot:
        assert snapshot_path == Path(snapshot_receipt.model_path)
        assert contract_path == Path(snapshot_receipt.contract_path or "")
        assert expected_contract_sha256 == snapshot_receipt.contract_sha256
        assert expected_model_id == snapshot_receipt.model_id
        assert expected_revision == snapshot_receipt.revision
        assert expected_ktransformers_method == snapshot_receipt.ktransformers_method
        return make_verified_snapshot(snapshot_receipt)

    def load_kernel_receipt(
        receipt_path: Path,
        *,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtKernelRuntimeValidationReceiptObservation:
        assert receipt_path == Path(kernel_receipt.receipt_path)
        assert expected_receipt_sha256 == kernel_receipt.receipt_sha256
        return kernel_receipt

    monkeypatch.setattr(admission, "verify_sglang_kt_model_snapshot", verify_snapshot)
    monkeypatch.setattr(
        admission,
        "load_sglang_kt_kernel_runtime_validation_receipt",
        load_kernel_receipt,
    )

    verify_sglang_kt_local_admission_bindings_sync(
        preflight.process_specs,
        preflight.admission_bindings,
    )
