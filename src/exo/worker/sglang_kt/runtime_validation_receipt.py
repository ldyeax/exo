import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, final

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    ResourceIndex,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)

KERNEL_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
KERNEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES = 1024 * 1024

_EXPECTED_PACKAGE_VERSION = "0.6.3.post1"
_EXPECTED_TORCH_MODULE_VERSION = "2.9.1+cu128"
_EXPECTED_TORCH_CUDA_VERSION = "12.8"
_EXPECTED_TRANSFORMERS_DISTRIBUTION_VERSION = "5.6.0.post1"
_EXPECTED_TRANSFORMERS_MODULE_VERSION = "5.6.0"
_EXPECTED_COMPUTE_CAPABILITY = (8, 6)
_EXPECTED_DIRECT_QLENS = (1, 16)
_EXPECTED_AMX_KERNEL_CLASS = "AMXBF16_MOE"
_EXPECTED_AMX_RELATIVE_L1_TOLERANCE = 0.02
_EXPECTED_CUDA_RELATIVE_L1_TOLERANCE = 0.02
_EXPECTED_RANDOM_SEED = 20_260_719
_EXPECTED_HIDDEN_SIZE = 256
_EXPECTED_CUDA_INPUT_VALUE = 0.125
_EXPECTED_KERNEL_CAPABILITY = "kt_bf16_amx_executed_v1"
_EXPECTED_CPU_FEATURES = frozenset(
    {"amx_bf16", "amx_int8", "amx_tile", "avx512_bf16", "avx512f"}
)
_EXPECTED_BUILD_ENVIRONMENT = {
    "CPUINFER_BUILD_ALL_VARIANTS": "0",
    "CPUINFER_CPU_INSTRUCT": "NATIVE",
    "CPUINFER_CUDA_ARCHS": "86",
    "CPUINFER_CUDA_STATIC_RUNTIME": "ON",
    "CPUINFER_ENABLE_AMX": "ON",
    "CPUINFER_USE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.6",
}
_EXPECTED_DISTRIBUTION_VERSIONS = {
    "flashinfer-cubin": "0.6.3",
    "flashinfer-python": "0.6.3",
    "ktransformers": _EXPECTED_PACKAGE_VERSION,
    "kt-kernel": _EXPECTED_PACKAGE_VERSION,
    "sgl-kernel": "0.3.21",
    "sglang-kt": _EXPECTED_PACKAGE_VERSION,
    "torch": "2.9.1",
    "transformers-kt": _EXPECTED_TRANSFORMERS_DISTRIBUTION_VERSION,
}
_EXPECTED_WHEEL_DISTRIBUTIONS = frozenset({"ktransformers", "kt-kernel", "sglang-kt"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

NonemptyText = Annotated[str, StringConstraints(min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SglangKtKernelRuntimeCapability = Literal["kt_bf16_amx_executed_v1"]


class SglangKtKernelRuntimeValidationReceiptError(ValueError):
    """Raised when kernel validation evidence cannot be admitted exactly."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@final
class _WorkerPoolAssignment(_StrictModel):
    numa_nodes: tuple[ResourceIndex, ...]
    threads_per_subpool: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def validate_assignment(self) -> "_WorkerPoolAssignment":
        if (
            not self.numa_nodes
            or self.numa_nodes != tuple(sorted(set(self.numa_nodes)))
            or len(self.numa_nodes) != len(self.threads_per_subpool)
        ):
            raise ValueError("worker-pool NUMA and thread assignments are not exact")
        return self


@final
class _ValidationConfig(_StrictModel):
    gpu_uuid: GpuUuid
    build_receipt_path: AbsoluteRuntimePath
    worker_pool: _WorkerPoolAssignment


@final
class _WheelArtifactEvidence(_StrictModel):
    distribution: NonemptyText
    version: NonemptyText
    path: AbsoluteRuntimePath
    size_bytes: PositiveInt
    sha256: Sha256Digest


@final
class _EmbeddedProvenanceEvidence(_StrictModel):
    distribution: NonemptyText
    # Wheel provenance names an archive member while installed provenance uses
    # an absolute filesystem path. Both are evidence strings in schema v1.
    location: NonemptyText
    sha256: Sha256Digest
    schema_version: Literal[1]
    ktransformers_revision: GitRevision
    sglang_revision: GitRevision

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str) -> str:
        if "\\" in value or "\0" in value or value != os.path.normpath(value):
            raise ValueError("embedded provenance path must be normalized POSIX text")
        path = PurePosixPath(value)
        if not path.is_absolute() and (
            not path.parts
            or any(part in {".", ".."} for part in path.parts)
            or path.as_posix() != value
        ):
            raise ValueError("embedded provenance archive path is unsafe")
        return value


@final
class _BuildProvenanceEvidence(_StrictModel):
    receipt_path: AbsoluteRuntimePath
    receipt_sha256: Sha256Digest
    build_id: Sha256Digest
    builder_sha256: Sha256Digest
    host_profile: NonemptyText
    hostname: NonemptyText
    ktransformers_revision: GitRevision
    sglang_revision: GitRevision
    package_version: NonemptyText
    cuda_architectures: NonemptyText
    cpu_features: tuple[NonemptyText, ...]
    build_environment: tuple[tuple[NonemptyText, NonemptyText], ...]
    bootstrap_wheels: tuple[_WheelArtifactEvidence, ...]
    wheels: tuple[_WheelArtifactEvidence, ...]
    embedded_provenance: tuple[_EmbeddedProvenanceEvidence, ...]
    kt_extension_member: NonemptyText
    kt_extension_sha256: Sha256Digest
    verified: bool


@final
class _PackageEvidence(_StrictModel):
    distribution: NonemptyText
    expected_version: NonemptyText
    observed_version: NonemptyText
    location: AbsoluteRuntimePath
    installer: NonemptyText
    direct_url_sha256: Sha256Digest | None


@final
class _RuntimeIdentityEvidence(_StrictModel):
    torch_module_version: NonemptyText
    torch_cuda_version: NonemptyText
    transformers_distribution_version: NonemptyText
    transformers_module_version: NonemptyText
    sgl_kernel_build_id: Sha256Digest
    deep_gemm_build_id: Sha256Digest
    kt_kernel_build_id: Sha256Digest
    embedded_provenance: tuple[_EmbeddedProvenanceEvidence, ...]


@final
class _NumaNodeEvidence(_StrictModel):
    node: ResourceIndex
    cpu_ids: tuple[ResourceIndex, ...]

    @model_validator(mode="after")
    def validate_cpu_ids(self) -> "_NumaNodeEvidence":
        if not self.cpu_ids or self.cpu_ids != tuple(sorted(set(self.cpu_ids))):
            raise ValueError("NUMA CPU IDs must be nonempty, sorted, and unique")
        return self


@final
class _ProcessEvidence(_StrictModel):
    hostname: NonemptyText
    executable: AbsoluteRuntimePath
    python_implementation: NonemptyText
    python_version: tuple[PositiveInt, ResourceIndex, ResourceIndex]
    pid: PositiveInt
    parent_pid: ResourceIndex
    uid: ResourceIndex
    gid: ResourceIndex
    cwd: AbsoluteRuntimePath
    argv: tuple[NonemptyText, ...]
    affinity_cpu_ids: tuple[ResourceIndex, ...]
    allowed_memory_nodes: tuple[ResourceIndex, ...]
    environment: tuple[tuple[NonemptyText, str], ...]

    @model_validator(mode="after")
    def validate_process_assignment(self) -> "_ProcessEvidence":
        if not self.affinity_cpu_ids or self.affinity_cpu_ids != tuple(
            sorted(set(self.affinity_cpu_ids))
        ):
            raise ValueError(
                "process affinity CPUs must be nonempty, sorted, and unique"
            )
        if not self.allowed_memory_nodes or self.allowed_memory_nodes != tuple(
            sorted(set(self.allowed_memory_nodes))
        ):
            raise ValueError(
                "process memory-node allowance must be nonempty, sorted, and unique"
            )
        _unique_pair_mapping(self.environment, "process environment")
        return self


@final
class _HostEvidence(_StrictModel):
    process: _ProcessEvidence
    operating_system: NonemptyText
    machine: NonemptyText
    cpu_features: tuple[NonemptyText, ...]
    numa_nodes: tuple[_NumaNodeEvidence, ...]
    collection_method: Literal["direct_execution_only"]
    profiler: Literal["none"]


@final
class _NumericalEvidence(_StrictModel):
    shape: tuple[PositiveInt, ...]
    seed: int
    execution_dtype: NonemptyText
    reference_dtype: NonemptyText
    output_finite: bool
    reference_finite: bool
    mean_absolute_error: float
    maximum_absolute_error: float
    reference_mean_absolute: float
    relative_l1_error: float
    relative_l1_tolerance: float


@final
class _CudaExecutionEvidence(_StrictModel):
    visible_devices: NonemptyText
    logical_device_count: PositiveInt
    logical_device_index: ResourceIndex
    gpu_uuid: GpuUuid
    pci_bus_id: NonemptyText
    gpu_name: NonemptyText
    compute_capability: tuple[PositiveInt, ResourceIndex]
    total_memory_bytes: PositiveInt
    driver_version: NonemptyText
    torch_cuda_version: NonemptyText
    torch_device_uuid_raw: NonemptyText
    torch_device_uuid: GpuUuid
    numerical: _NumericalEvidence


@final
class _AmxExecutionEvidence(_StrictModel):
    route: Literal["direct", "cuda_stream"]
    qlen: PositiveInt
    kernel_class: NonemptyText
    cpu_variant: NonemptyText
    extension_path: AbsoluteRuntimePath
    extension_sha256: Sha256Digest
    worker_pool: _WorkerPoolAssignment
    load_submit_count: ResourceIndex
    load_sync_count: ResourceIndex
    direct_forward_submit_count: ResourceIndex
    direct_forward_sync_count: ResourceIndex
    cuda_stream_api_available: bool | None
    cuda_stream_submit_count: ResourceIndex
    cuda_stream_sync_count: ResourceIndex
    cuda_stream_id: ResourceIndex | None
    default_cuda_stream_id: ResourceIndex | None
    cuda_input_producer_count: ResourceIndex
    cuda_input_to_cpu_copy_count: ResourceIndex
    cuda_output_from_cpu_copy_count: ResourceIndex
    cuda_output_consumer_count: ResourceIndex
    cuda_input_produced_checksum: float | None
    cpu_input_observed_checksum: float | None
    cuda_output_consumed_l1: float | None
    cuda_output_reference_l1: float | None
    numerical: _NumericalEvidence


@final
class _PassedKernelRuntimeValidationReceiptV1(_StrictModel):
    schema_version: Literal[1]
    status: Literal["passed"]
    generated_at_utc: NonemptyText
    profiler: Literal["none"]
    config: _ValidationConfig
    provenance: _BuildProvenanceEvidence
    packages: tuple[_PackageEvidence, ...]
    runtime_identity: _RuntimeIdentityEvidence
    host: _HostEvidence
    cuda: _CudaExecutionEvidence
    amx: tuple[_AmxExecutionEvidence, ...]
    cuda_stream: _AmxExecutionEvidence
    capabilities: tuple[SglangKtKernelRuntimeCapability, ...]
    failures: tuple[NonemptyText, ...]

    @field_validator("generated_at_utc")
    @classmethod
    def validate_generated_at_utc(cls, value: str) -> str:
        try:
            generated_at = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                "kernel receipt generation time is not ISO-8601"
            ) from error
        if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
            raise ValueError("kernel receipt generation time must be UTC")
        return value

    @model_validator(mode="after")
    def validate_admission_evidence(
        self,
    ) -> "_PassedKernelRuntimeValidationReceiptV1":
        if self.capabilities != (_EXPECTED_KERNEL_CAPABILITY,):
            raise ValueError("kernel receipt capability set is not exact")
        if self.failures:
            raise ValueError("passed kernel receipt contains failures")
        _validate_build_and_runtime_identity(self)
        _validate_host_assignment(self)
        _validate_cuda_evidence(self)
        _validate_amx_evidence(self)
        return self


@final
class SglangKtKernelRuntimeValidationReceiptObservation(_StrictModel):
    """File-bound facts from one admitted kernel-only runtime validation."""

    receipt_path: AbsoluteRuntimePath
    receipt_size_bytes: PositiveInt
    receipt_sha256: Sha256Digest
    schema_version: Literal[1]
    generated_at_utc: NonemptyText
    capabilities: tuple[SglangKtKernelRuntimeCapability, ...]
    gpu_uuid: GpuUuid
    gpu_compute_capability: tuple[Literal[8], Literal[6]]
    gpu_pci_bus_id: NonemptyText
    gpu_name: NonemptyText
    gpu_total_memory_bytes: PositiveInt
    driver_version: NonemptyText
    hostname: NonemptyText
    executable: AbsoluteRuntimePath
    cpu_cores: tuple[ResourceIndex, ...]
    allowed_memory_nodes: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    threads_per_subpool: tuple[PositiveInt, ...]
    build_receipt_path: AbsoluteRuntimePath
    build_receipt_sha256: Sha256Digest
    runtime_build_id: Sha256Digest
    builder_sha256: Sha256Digest
    kt_extension_sha256: Sha256Digest
    host_profile: NonemptyText
    package_version: NonemptyText
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    torch_version: NonemptyText
    cuda_version: NonemptyText
    transformers_distribution_version: NonemptyText
    transformers_module_version: NonemptyText
    sgl_kernel_build_id: Sha256Digest
    deep_gemm_build_id: Sha256Digest
    kt_kernel_build_id: Sha256Digest


def _unique_pair_mapping(
    pairs: tuple[tuple[str, str], ...], description: str
) -> dict[str, str]:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError(f"{description} keys must be unique")
    return values


def _version_matches(observed: str, expected: str) -> bool:
    return observed.partition("+")[0] == expected


def _embedded_by_distribution(
    evidence: tuple[_EmbeddedProvenanceEvidence, ...],
    description: str,
) -> dict[str, _EmbeddedProvenanceEvidence]:
    values = {item.distribution: item for item in evidence}
    if len(values) != len(evidence):
        raise ValueError(f"{description} distributions must be unique")
    return values


def _validate_build_and_runtime_identity(
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    provenance = receipt.provenance
    runtime = receipt.runtime_identity
    if not provenance.verified:
        raise ValueError("kernel receipt build provenance is not verified")
    if provenance.receipt_path != receipt.config.build_receipt_path:
        raise ValueError("kernel receipt build path changed after validation")
    if (
        provenance.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or provenance.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    ):
        raise ValueError("kernel receipt source revisions are not pinned")
    if (
        provenance.package_version != _EXPECTED_PACKAGE_VERSION
        or provenance.cuda_architectures != "86"
    ):
        raise ValueError("kernel receipt build profile is not pinned")
    if not _EXPECTED_CPU_FEATURES.issubset(provenance.cpu_features):
        raise ValueError("kernel receipt build CPU features are incomplete")
    build_environment = _unique_pair_mapping(
        provenance.build_environment, "build environment"
    )
    if any(
        build_environment.get(name) != expected
        for name, expected in _EXPECTED_BUILD_ENVIRONMENT.items()
    ):
        raise ValueError("kernel receipt build environment is not pinned")

    wheels = {wheel.distribution: wheel for wheel in provenance.wheels}
    if (
        len(wheels) != len(provenance.wheels)
        or frozenset(wheels) != _EXPECTED_WHEEL_DISTRIBUTIONS
    ):
        raise ValueError("kernel receipt runtime wheel set is not exact")
    packages = {package.distribution: package for package in receipt.packages}
    if len(packages) != len(receipt.packages) or set(packages) != set(
        _EXPECTED_DISTRIBUTION_VERSIONS
    ):
        raise ValueError("kernel receipt package evidence set is not exact")
    for distribution, expected_version in _EXPECTED_DISTRIBUTION_VERSIONS.items():
        package = packages[distribution]
        if package.expected_version != expected_version or not _version_matches(
            package.observed_version, expected_version
        ):
            raise ValueError(
                f"kernel receipt package version is not pinned: {distribution}"
            )
        wheel = wheels.get(distribution)
        if wheel is not None and package.direct_url_sha256 != wheel.sha256:
            raise ValueError(
                f"kernel receipt package is not bound to its wheel: {distribution}"
            )

    if (
        runtime.torch_module_version != _EXPECTED_TORCH_MODULE_VERSION
        or runtime.torch_cuda_version != _EXPECTED_TORCH_CUDA_VERSION
        or runtime.transformers_distribution_version
        != _EXPECTED_TRANSFORMERS_DISTRIBUTION_VERSION
        or runtime.transformers_module_version != _EXPECTED_TRANSFORMERS_MODULE_VERSION
    ):
        raise ValueError("kernel receipt runtime versions are not pinned")
    wheel_embedded = _embedded_by_distribution(
        provenance.embedded_provenance, "wheel embedded provenance"
    )
    runtime_embedded = _embedded_by_distribution(
        runtime.embedded_provenance, "runtime embedded provenance"
    )
    if set(wheel_embedded) != {"kt-kernel", "sglang-kt"} or set(runtime_embedded) != {
        "kt-kernel",
        "sglang-kt",
    }:
        raise ValueError("kernel receipt embedded provenance set is not exact")
    for distribution, installed in runtime_embedded.items():
        built = wheel_embedded[distribution]
        if (
            installed.sha256 != built.sha256
            or installed.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
            or installed.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        ):
            raise ValueError("kernel receipt embedded provenance is not pinned")


def _validate_host_assignment(
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    host = receipt.host
    process = host.process
    if receipt.provenance.hostname != process.hostname:
        raise ValueError("kernel receipt build and execution hosts differ")
    if process.python_implementation != "CPython" or process.python_version[:2] != (
        3,
        12,
    ):
        raise ValueError("kernel receipt Python runtime is not pinned")
    if not _EXPECTED_CPU_FEATURES.issubset(host.cpu_features):
        raise ValueError("kernel receipt execution CPU features are incomplete")
    environment = _unique_pair_mapping(process.environment, "process environment")
    if environment.get("CUDA_VISIBLE_DEVICES") != receipt.config.gpu_uuid:
        raise ValueError("kernel receipt process GPU binding is not exact")

    nodes = {item.node: item for item in host.numa_nodes}
    if len(nodes) != len(host.numa_nodes) or tuple(nodes) != tuple(sorted(nodes)):
        raise ValueError("kernel receipt host NUMA nodes are not sorted and unique")
    affinity = set(process.affinity_cpu_ids)
    allowed_memory_nodes = set(process.allowed_memory_nodes)
    for node, threads in zip(
        receipt.config.worker_pool.numa_nodes,
        receipt.config.worker_pool.threads_per_subpool,
        strict=True,
    ):
        node_evidence = nodes.get(node)
        if node_evidence is None or node not in allowed_memory_nodes:
            raise ValueError("kernel receipt worker NUMA node is unavailable")
        if len(affinity.intersection(node_evidence.cpu_ids)) < threads:
            raise ValueError("kernel receipt worker pool exceeds process CPU affinity")


def _validate_numerical_evidence(
    evidence: _NumericalEvidence,
    *,
    expected_shape: tuple[int, ...],
    expected_seed: int,
    expected_tolerance: float,
) -> None:
    metrics = (
        evidence.mean_absolute_error,
        evidence.maximum_absolute_error,
        evidence.reference_mean_absolute,
        evidence.relative_l1_error,
        evidence.relative_l1_tolerance,
    )
    if not all(math.isfinite(metric) for metric in metrics):
        raise ValueError("kernel receipt numerical metrics must be finite")
    if (
        not evidence.output_finite
        or not evidence.reference_finite
        or evidence.execution_dtype != "torch.bfloat16"
        or evidence.reference_dtype != "torch.float32"
        or evidence.shape != expected_shape
        or evidence.seed != expected_seed
        or evidence.relative_l1_tolerance != expected_tolerance
        or evidence.relative_l1_error > expected_tolerance
        or evidence.mean_absolute_error < 0
        or evidence.maximum_absolute_error < 0
        or evidence.reference_mean_absolute <= 0
    ):
        raise ValueError("kernel receipt numerical evidence is not admissible")


def _validate_cuda_evidence(
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    cuda = receipt.cuda
    if (
        cuda.visible_devices != receipt.config.gpu_uuid
        or cuda.gpu_uuid != receipt.config.gpu_uuid
        or cuda.torch_device_uuid != receipt.config.gpu_uuid
        or cuda.logical_device_count != 1
        or cuda.logical_device_index != 0
        or cuda.compute_capability != _EXPECTED_COMPUTE_CAPABILITY
        or cuda.torch_cuda_version != receipt.runtime_identity.torch_cuda_version
    ):
        raise ValueError("kernel receipt CUDA device identity is not exact")
    _validate_numerical_evidence(
        cuda.numerical,
        expected_shape=(128, 96),
        expected_seed=_EXPECTED_RANDOM_SEED,
        expected_tolerance=_EXPECTED_CUDA_RELATIVE_L1_TOLERANCE,
    )


def _validate_direct_amx_evidence(
    evidence: _AmxExecutionEvidence,
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    if (
        evidence.route != "direct"
        or evidence.kernel_class != _EXPECTED_AMX_KERNEL_CLASS
        or evidence.cpu_variant.lower() != "amx"
        or evidence.worker_pool != receipt.config.worker_pool
        or evidence.extension_sha256 != receipt.provenance.kt_extension_sha256
        or evidence.load_submit_count != 1
        or evidence.load_sync_count != 1
        or evidence.direct_forward_submit_count != 1
        or evidence.direct_forward_sync_count != 1
        or evidence.cuda_stream_api_available is not None
        or evidence.cuda_stream_submit_count != 0
        or evidence.cuda_stream_sync_count != 0
        or evidence.cuda_stream_id is not None
        or evidence.default_cuda_stream_id is not None
        or evidence.cuda_input_producer_count != 0
        or evidence.cuda_input_to_cpu_copy_count != 0
        or evidence.cuda_output_from_cpu_copy_count != 0
        or evidence.cuda_output_consumer_count != 0
        or evidence.cuda_input_produced_checksum is not None
        or evidence.cpu_input_observed_checksum is not None
        or evidence.cuda_output_consumed_l1 is not None
        or evidence.cuda_output_reference_l1 is not None
    ):
        raise ValueError("kernel receipt direct AMX evidence is not exact")
    _validate_numerical_evidence(
        evidence.numerical,
        expected_shape=(evidence.qlen, _EXPECTED_HIDDEN_SIZE),
        expected_seed=_EXPECTED_RANDOM_SEED + evidence.qlen,
        expected_tolerance=_EXPECTED_AMX_RELATIVE_L1_TOLERANCE,
    )


def _validate_cuda_stream_amx_evidence(
    evidence: _AmxExecutionEvidence,
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    expected_input_checksum = (
        evidence.qlen * _EXPECTED_HIDDEN_SIZE * _EXPECTED_CUDA_INPUT_VALUE
    )
    output_l1_matches = (
        evidence.cuda_output_consumed_l1 is not None
        and evidence.cuda_output_reference_l1 is not None
        and evidence.cuda_output_reference_l1 > 0
        and abs(evidence.cuda_output_consumed_l1 - evidence.cuda_output_reference_l1)
        / evidence.cuda_output_reference_l1
        <= _EXPECTED_AMX_RELATIVE_L1_TOLERANCE
    )
    if (
        evidence.route != "cuda_stream"
        or evidence.qlen != 1
        or evidence.kernel_class != _EXPECTED_AMX_KERNEL_CLASS
        or evidence.cpu_variant.lower() != "amx"
        or evidence.worker_pool != receipt.config.worker_pool
        or evidence.extension_sha256 != receipt.provenance.kt_extension_sha256
        or evidence.load_submit_count != 1
        or evidence.load_sync_count != 1
        or evidence.direct_forward_submit_count != 0
        or evidence.direct_forward_sync_count != 0
        or evidence.cuda_stream_api_available is not True
        or evidence.cuda_stream_submit_count != 1
        or evidence.cuda_stream_sync_count != 1
        or evidence.cuda_stream_id is None
        or evidence.default_cuda_stream_id is None
        or evidence.cuda_stream_id == evidence.default_cuda_stream_id
        or evidence.cuda_input_producer_count != 1
        or evidence.cuda_input_to_cpu_copy_count != 1
        or evidence.cuda_output_from_cpu_copy_count != 1
        or evidence.cuda_output_consumer_count != 1
        or evidence.cuda_input_produced_checksum != expected_input_checksum
        or evidence.cpu_input_observed_checksum != expected_input_checksum
        or not output_l1_matches
    ):
        raise ValueError("kernel receipt CUDA-stream AMX evidence is not exact")
    _validate_numerical_evidence(
        evidence.numerical,
        expected_shape=(1, _EXPECTED_HIDDEN_SIZE),
        expected_seed=_EXPECTED_RANDOM_SEED + 1,
        expected_tolerance=_EXPECTED_AMX_RELATIVE_L1_TOLERANCE,
    )


def _validate_amx_evidence(
    receipt: _PassedKernelRuntimeValidationReceiptV1,
) -> None:
    if tuple(item.qlen for item in receipt.amx) != _EXPECTED_DIRECT_QLENS:
        raise ValueError("kernel receipt direct AMX qlen set is not exact")
    for evidence in receipt.amx:
        _validate_direct_amx_evidence(evidence, receipt)
    _validate_cuda_stream_amx_evidence(receipt.cuda_stream, receipt)


def _validate_expected_sha256(expected_receipt_sha256: str | None) -> None:
    if (
        expected_receipt_sha256 is not None
        and _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None
    ):
        raise SglangKtKernelRuntimeValidationReceiptError(
            "expected kernel receipt SHA-256 is invalid"
        )


def load_sglang_kt_kernel_runtime_validation_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> SglangKtKernelRuntimeValidationReceiptObservation:
    """Load one stable v1 kernel receipt and derive its narrow capability."""

    _validate_expected_sha256(expected_receipt_sha256)
    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=KERNEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES,
        )
        if (
            expected_receipt_sha256 is not None
            and bound_file.sha256 != expected_receipt_sha256
        ):
            raise SglangKtKernelRuntimeValidationReceiptError(
                "kernel receipt does not match the expected SHA-256"
            )
        parse_sglang_kt_strict_json(bound_file.contents)
        receipt = _PassedKernelRuntimeValidationReceiptV1.model_validate_json(
            bound_file.contents
        )
    except SglangKtKernelRuntimeValidationReceiptError:
        raise
    except (RecursionError, SglangKtReceiptFileError, ValidationError) as error:
        raise SglangKtKernelRuntimeValidationReceiptError(
            f"invalid SGLang-KTransformers kernel runtime receipt: {path}"
        ) from error

    runtime = receipt.runtime_identity
    provenance = receipt.provenance
    cuda = receipt.cuda
    process = receipt.host.process
    worker_pool = receipt.config.worker_pool
    return SglangKtKernelRuntimeValidationReceiptObservation(
        receipt_path=str(bound_file.path),
        receipt_size_bytes=len(bound_file.contents),
        receipt_sha256=bound_file.sha256,
        schema_version=KERNEL_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        generated_at_utc=receipt.generated_at_utc,
        capabilities=(_EXPECTED_KERNEL_CAPABILITY,),
        gpu_uuid=receipt.config.gpu_uuid,
        gpu_compute_capability=(8, 6),
        gpu_pci_bus_id=cuda.pci_bus_id,
        gpu_name=cuda.gpu_name,
        gpu_total_memory_bytes=cuda.total_memory_bytes,
        driver_version=cuda.driver_version,
        hostname=process.hostname,
        executable=process.executable,
        cpu_cores=process.affinity_cpu_ids,
        allowed_memory_nodes=process.allowed_memory_nodes,
        memory_nodes=worker_pool.numa_nodes,
        threads_per_subpool=worker_pool.threads_per_subpool,
        build_receipt_path=provenance.receipt_path,
        build_receipt_sha256=provenance.receipt_sha256,
        runtime_build_id=provenance.build_id,
        builder_sha256=provenance.builder_sha256,
        kt_extension_sha256=provenance.kt_extension_sha256,
        host_profile=provenance.host_profile,
        package_version=provenance.package_version,
        sglang_revision=provenance.sglang_revision,
        ktransformers_revision=provenance.ktransformers_revision,
        torch_version=runtime.torch_module_version,
        cuda_version=runtime.torch_cuda_version,
        transformers_distribution_version=(runtime.transformers_distribution_version),
        transformers_module_version=runtime.transformers_module_version,
        sgl_kernel_build_id=runtime.sgl_kernel_build_id,
        deep_gemm_build_id=runtime.deep_gemm_build_id,
        kt_kernel_build_id=runtime.kt_kernel_build_id,
    )
