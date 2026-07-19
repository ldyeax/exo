#!/usr/bin/env python3
"""Validate the pinned GLM-4.7 SGLang-KTransformers kernel runtime.

This is a correctness gate, not a benchmark and not a profiler. It writes one
fail-closed JSON receipt after checking exact build provenance, the selected
SM86 CUDA device, AMX-BF16 MoE decode/prefill numerics, and KT's non-default
CUDA-stream submit/sync bridge.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

if __package__ in {None, ""}:
    sys.dont_write_bytecode = True
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.worker.sglang_kt.artifact_identity import (
    calculate_sglang_kt_artifact_build_id,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from scripts.build_sglang_kt_runtime import (
    BOOTSTRAP_WHEEL_PINS,
    HOST_BUILD_PROFILES,
    KTRANSFORMERS_BUILD_SUBMODULE_PATHS,
    RuntimeBuildLayout,
    RuntimeSourceObservation,
    RuntimeSourcePins,
    RuntimeToolchainObservation,
    SubmoduleObservation,
    ToolObservation,
    calculate_runtime_build_id,
    runtime_build_commands,
    runtime_build_environment,
)

SCHEMA_VERSION = 1
EXPECTED_PACKAGE_VERSION = "0.6.3.post1"
EXPECTED_TORCH_MODULE_VERSION = "2.9.1+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_TRANSFORMERS_MODULE_VERSION = "5.6.0"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_QLENS = (1, 16)
EXPECTED_WHEEL_DISTRIBUTIONS = frozenset({"ktransformers", "kt-kernel", "sglang-kt"})
EXPECTED_DISTRIBUTION_VERSIONS: Mapping[str, str] = {
    "flashinfer-cubin": "0.6.3",
    "flashinfer-python": "0.6.3",
    "ktransformers": EXPECTED_PACKAGE_VERSION,
    "kt-kernel": EXPECTED_PACKAGE_VERSION,
    "sgl-kernel": "0.3.21",
    "sglang-kt": EXPECTED_PACKAGE_VERSION,
    "torch": "2.9.1",
    "transformers-kt": "5.6.0.post1",
}
REQUIRED_CPU_FEATURES = frozenset(
    {"amx_bf16", "amx_int8", "amx_tile", "avx512_bf16", "avx512f"}
)
REQUIRED_BUILD_ENVIRONMENT: Mapping[str, str] = {
    "CPUINFER_BUILD_ALL_VARIANTS": "0",
    "CPUINFER_CPU_INSTRUCT": "NATIVE",
    "CPUINFER_CUDA_ARCHS": "86",
    "CPUINFER_CUDA_STATIC_RUNTIME": "ON",
    "CPUINFER_ENABLE_AMX": "ON",
    "CPUINFER_USE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.6",
}
GPU_UUID_PATTERN = re.compile(r"GPU-[0-9A-Fa-f-]{16,}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
HOST_ENVIRONMENT_NAMES = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "KMP_AFFINITY",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
)
AMX_RELATIVE_L1_TOLERANCE = 0.02
CUDA_RELATIVE_L1_TOLERANCE = 0.02
CUDA_STREAM_INPUT_VALUE = 0.125
CUDA_STREAM_OUTPUT_MULTIPLIER = 2.0
EXPERT_COUNT = 8
TOP_K = 2
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
MAX_QLEN = 128
RANDOM_SEED = 20_260_719

ReceiptStatus = Literal["passed", "failed"]
ExecutionRoute = Literal["direct", "cuda_stream"]


class RuntimeValidationError(RuntimeError):
    """Raised when validation evidence cannot be collected exactly."""


@dataclass(frozen=True)
class WorkerPoolAssignment:
    numa_nodes: tuple[int, ...]
    threads_per_subpool: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.numa_nodes:
            raise ValueError("at least one NUMA node is required")
        if len(self.numa_nodes) != len(self.threads_per_subpool):
            raise ValueError("NUMA nodes and thread counts must have equal lengths")
        if len(set(self.numa_nodes)) != len(self.numa_nodes):
            raise ValueError("NUMA nodes must be unique")
        if any(node < 0 for node in self.numa_nodes):
            raise ValueError("NUMA nodes must be nonnegative")
        if any(count <= 0 for count in self.threads_per_subpool):
            raise ValueError("thread counts must be positive")


@dataclass(frozen=True)
class ValidationConfig:
    gpu_uuid: str
    build_receipt_path: str
    worker_pool: WorkerPoolAssignment

    def __post_init__(self) -> None:
        if GPU_UUID_PATTERN.fullmatch(self.gpu_uuid) is None:
            raise ValueError("gpu_uuid must be a complete NVIDIA GPU UUID")
        if not Path(self.build_receipt_path).is_absolute():
            raise ValueError("build_receipt_path must be absolute")


@dataclass(frozen=True)
class WheelArtifactEvidence:
    distribution: str
    version: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BuildProvenanceEvidence:
    receipt_path: str
    receipt_sha256: str
    build_id: str
    builder_sha256: str
    host_profile: str
    hostname: str
    ktransformers_revision: str
    sglang_revision: str
    package_version: str
    cuda_architectures: str
    cpu_features: tuple[str, ...]
    build_environment: tuple[tuple[str, str], ...]
    bootstrap_wheels: tuple[WheelArtifactEvidence, ...]
    wheels: tuple[WheelArtifactEvidence, ...]
    embedded_provenance: tuple[EmbeddedProvenanceEvidence, ...]
    kt_extension_member: str
    kt_extension_sha256: str
    verified: bool


@dataclass(frozen=True)
class PackageEvidence:
    distribution: str
    expected_version: str
    observed_version: str | None
    location: str | None
    installer: str | None
    direct_url_sha256: str | None


@dataclass(frozen=True)
class EmbeddedProvenanceEvidence:
    distribution: str
    location: str
    sha256: str
    schema_version: int
    ktransformers_revision: str
    sglang_revision: str


@dataclass(frozen=True)
class RuntimeIdentityEvidence:
    torch_module_version: str | None
    torch_cuda_version: str | None
    transformers_distribution_version: str | None
    transformers_module_version: str | None
    sgl_kernel_build_id: str | None
    deep_gemm_build_id: str | None
    kt_kernel_build_id: str | None
    embedded_provenance: tuple[EmbeddedProvenanceEvidence, ...]


@dataclass(frozen=True)
class NumaNodeEvidence:
    node: int
    cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProcessEvidence:
    hostname: str
    executable: str
    python_implementation: str
    python_version: tuple[int, int, int]
    pid: int
    parent_pid: int
    uid: int
    gid: int
    cwd: str
    argv: tuple[str, ...]
    affinity_cpu_ids: tuple[int, ...]
    allowed_memory_nodes: tuple[int, ...]
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HostEvidence:
    process: ProcessEvidence
    operating_system: str
    machine: str
    cpu_features: tuple[str, ...]
    numa_nodes: tuple[NumaNodeEvidence, ...]
    collection_method: Literal["direct_execution_only"]
    profiler: Literal["none"]


@dataclass(frozen=True)
class NumericalEvidence:
    shape: tuple[int, ...]
    seed: int
    execution_dtype: str
    reference_dtype: str
    output_finite: bool
    reference_finite: bool
    mean_absolute_error: float | None
    maximum_absolute_error: float | None
    reference_mean_absolute: float | None
    relative_l1_error: float | None
    relative_l1_tolerance: float


@dataclass(frozen=True)
class CudaExecutionEvidence:
    visible_devices: str | None
    logical_device_count: int
    logical_device_index: int
    gpu_uuid: str
    pci_bus_id: str
    gpu_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    driver_version: str
    torch_cuda_version: str | None
    torch_device_uuid_raw: str | None
    torch_device_uuid: str | None
    numerical: NumericalEvidence


@dataclass(frozen=True)
class AmxExecutionEvidence:
    route: ExecutionRoute
    qlen: int
    kernel_class: str
    cpu_variant: str
    extension_path: str
    extension_sha256: str
    worker_pool: WorkerPoolAssignment
    load_submit_count: int
    load_sync_count: int
    direct_forward_submit_count: int
    direct_forward_sync_count: int
    cuda_stream_api_available: bool | None
    cuda_stream_submit_count: int
    cuda_stream_sync_count: int
    cuda_stream_id: int | None
    default_cuda_stream_id: int | None
    cuda_input_producer_count: int
    cuda_input_to_cpu_copy_count: int
    cuda_output_from_cpu_copy_count: int
    cuda_output_consumer_count: int
    cuda_input_produced_checksum: float | None
    cpu_input_observed_checksum: float | None
    cuda_output_consumed_l1: float | None
    cuda_output_reference_l1: float | None
    numerical: NumericalEvidence | None


@dataclass(frozen=True)
class _CudaStreamBridgeCounts:
    input_producer: int
    input_to_cpu_copy: int
    output_from_cpu_copy: int
    output_consumer: int


def _enqueue_cuda_stream_bridge(
    *,
    produce_cuda_input: Callable[[], Any],
    copy_cuda_input_to_cpu: Callable[[], Any],
    submit_cpu_work: Callable[[], Any],
    sync_cpu_work: Callable[[], Any],
    copy_cpu_output_to_cuda: Callable[[], Any],
    consume_cuda_output: Callable[[], Any],
) -> tuple[Any, _CudaStreamBridgeCounts]:
    produce_cuda_input()
    copy_cuda_input_to_cpu()
    submit_cpu_work()
    sync_cpu_work()
    copy_cpu_output_to_cuda()
    consumed_output = consume_cuda_output()
    return consumed_output, _CudaStreamBridgeCounts(
        input_producer=1,
        input_to_cpu_copy=1,
        output_from_cpu_copy=1,
        output_consumer=1,
    )


@dataclass(frozen=True)
class RuntimeValidationReceipt:
    schema_version: int
    status: ReceiptStatus
    generated_at_utc: str
    profiler: Literal["none"]
    config: ValidationConfig
    provenance: BuildProvenanceEvidence | None
    packages: tuple[PackageEvidence, ...]
    runtime_identity: RuntimeIdentityEvidence | None
    host: HostEvidence | None
    cuda: CudaExecutionEvidence | None
    amx: tuple[AmxExecutionEvidence, ...]
    cuda_stream: AmxExecutionEvidence | None
    capabilities: tuple[str, ...]
    failures: tuple[str, ...]


class ValidationBackend(Protocol):
    def observe_provenance(
        self, config: ValidationConfig
    ) -> BuildProvenanceEvidence: ...

    def observe_packages(self) -> tuple[PackageEvidence, ...]: ...

    def observe_runtime_identity(self) -> RuntimeIdentityEvidence: ...

    def observe_host(self) -> HostEvidence: ...

    def run_cuda(self, config: ValidationConfig) -> CudaExecutionEvidence: ...

    def run_amx(
        self,
        config: ValidationConfig,
        qlen: int,
        route: ExecutionRoute,
    ) -> AmxExecutionEvidence: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeValidationError(f"{description} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _required_sequence(value: object, description: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RuntimeValidationError(f"{description} must be a JSON array")
    return cast(Sequence[Any], value)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeValidationError(f"build receipt {name} must be a string")
    return value


def _required_integer(values: Mapping[str, Any], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeValidationError(f"build receipt {name} must be an integer")
    return value


def _required_boolean(values: Mapping[str, Any], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise RuntimeValidationError(f"build receipt {name} must be a boolean")
    return value


def _require_exact_keys(
    values: Mapping[str, Any], expected: frozenset[str], description: str
) -> None:
    observed = frozenset(values)
    if observed != expected:
        raise RuntimeValidationError(
            f"{description} keys are not exact; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _parse_source_observation(values: Mapping[str, Any]) -> RuntimeSourceObservation:
    _require_exact_keys(
        values,
        frozenset(
            {
                "ktransformers_source",
                "ktransformers_revision",
                "sglang_revision",
                "package_version",
                "source_date_epoch",
                "submodules",
            }
        ),
        "build receipt source",
    )
    source_path = _required_string(values, "ktransformers_source")
    if not Path(source_path).is_absolute():
        raise RuntimeValidationError("build receipt source path must be absolute")
    submodule_items = _required_sequence(values.get("submodules"), "source submodules")
    submodules: list[SubmoduleObservation] = []
    for item in submodule_items:
        submodule = _required_mapping(item, "source submodule")
        _require_exact_keys(
            submodule,
            frozenset({"path", "revision"}),
            "source submodule",
        )
        path = _required_string(submodule, "path")
        revision = _required_string(submodule, "revision")
        relative_path = PurePosixPath(path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise RuntimeValidationError("build receipt source submodule is invalid")
        submodules.append(SubmoduleObservation(path=path, revision=revision))
    if tuple(sorted(submodules, key=lambda item: item.path)) != tuple(submodules):
        raise RuntimeValidationError("build receipt source submodules are not sorted")
    expected_paths = {
        *(path.as_posix() for path in KTRANSFORMERS_BUILD_SUBMODULE_PATHS),
        RuntimeSourcePins.admitted().sglang_submodule_path.as_posix(),
    }
    if (
        len(submodules) != len(expected_paths)
        or {item.path for item in submodules} != expected_paths
    ):
        raise RuntimeValidationError("build receipt source submodule set is not exact")
    source_date_epoch = _required_integer(values, "source_date_epoch")
    if source_date_epoch <= 0:
        raise RuntimeValidationError("build receipt source_date_epoch must be positive")
    return RuntimeSourceObservation(
        ktransformers_source=source_path,
        ktransformers_revision=_required_string(values, "ktransformers_revision"),
        sglang_revision=_required_string(values, "sglang_revision"),
        package_version=_required_string(values, "package_version"),
        source_date_epoch=source_date_epoch,
        submodules=tuple(submodules),
    )


def _parse_toolchain_observation(
    values: Mapping[str, Any],
) -> RuntimeToolchainObservation:
    _require_exact_keys(
        values,
        frozenset(
            {
                "host_profile",
                "hostname",
                "operating_system",
                "machine",
                "python_version",
                "python_implementation",
                "python_soabi",
                "cuda_root",
                "cuda_version",
                "cuda_version_manifest_sha256",
                "cuda_architectures",
                "cpu_features",
                "tools",
            }
        ),
        "build receipt toolchain",
    )
    host_profile = _required_string(values, "host_profile")
    if host_profile not in HOST_BUILD_PROFILES:
        raise RuntimeValidationError("build receipt host profile is not admitted")
    cpu_feature_items = _required_sequence(
        values.get("cpu_features"), "toolchain CPU features"
    )
    if not all(isinstance(item, str) and item for item in cpu_feature_items):
        raise RuntimeValidationError("toolchain CPU features must be strings")
    cpu_features = tuple(cast(Sequence[str], cpu_feature_items))
    if cpu_features != tuple(sorted(set(cpu_features))):
        raise RuntimeValidationError("toolchain CPU features must be sorted and unique")

    tool_items = _required_sequence(values.get("tools"), "toolchain tools")
    tools: list[ToolObservation] = []
    for item in tool_items:
        tool = _required_mapping(item, "toolchain tool")
        _require_exact_keys(
            tool,
            frozenset({"name", "path", "sha256", "version"}),
            "toolchain tool",
        )
        name = _required_string(tool, "name")
        path = _required_string(tool, "path")
        sha256 = _required_string(tool, "sha256")
        if not Path(path).is_absolute() or SHA256_PATTERN.fullmatch(sha256) is None:
            raise RuntimeValidationError(
                f"build receipt tool {name} identity is invalid"
            )
        tools.append(
            ToolObservation(
                name=name,
                path=path,
                sha256=sha256,
                version=_required_string(tool, "version"),
            )
        )
    expected_tool_names = {
        "python",
        "nvcc",
        "cmake",
        "ninja",
        "cc",
        "cxx",
        "cudart_static",
    }
    if (
        tuple(sorted(tools, key=lambda item: item.name)) != tuple(tools)
        or len(tools) != len(expected_tool_names)
        or {item.name for item in tools} != expected_tool_names
    ):
        raise RuntimeValidationError("build receipt toolchain tool set is not exact")

    manifest_sha256 = _required_string(values, "cuda_version_manifest_sha256")
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise RuntimeValidationError("CUDA version manifest SHA-256 is invalid")
    return RuntimeToolchainObservation(
        host_profile=cast(Literal["dwagon", "fwuff"], host_profile),
        hostname=_required_string(values, "hostname"),
        operating_system=_required_string(values, "operating_system"),
        machine=_required_string(values, "machine"),
        python_version=_required_string(values, "python_version"),
        python_implementation=_required_string(values, "python_implementation"),
        python_soabi=_required_string(values, "python_soabi"),
        cuda_root=_required_string(values, "cuda_root"),
        cuda_version=_required_string(values, "cuda_version"),
        cuda_version_manifest_sha256=manifest_sha256,
        cuda_architectures=_required_string(values, "cuda_architectures"),
        cpu_features=cpu_features,
        tools=tuple(tools),
    )


def _parse_layout(values: Mapping[str, Any]) -> RuntimeBuildLayout:
    _require_exact_keys(
        values,
        frozenset({"build_root", "state", "cache", "logs", "wheels", "receipt"}),
        "build receipt layout",
    )
    return RuntimeBuildLayout(
        build_root=_required_string(values, "build_root"),
        state=_required_string(values, "state"),
        cache=_required_string(values, "cache"),
        logs=_required_string(values, "logs"),
        wheels=_required_string(values, "wheels"),
        receipt=_required_string(values, "receipt"),
    )


def _parse_string_pairs(value: object, description: str) -> tuple[tuple[str, str], ...]:
    items = _required_sequence(value, description)
    pairs: list[tuple[str, str]] = []
    for item in items:
        pair = _required_sequence(item, f"{description} entry")
        if len(pair) != 2 or not all(isinstance(component, str) for component in pair):
            raise RuntimeValidationError(f"{description} entries must be string pairs")
        pairs.append((cast(str, pair[0]), cast(str, pair[1])))
    if len({name for name, _value in pairs}) != len(pairs):
        raise RuntimeValidationError(f"{description} contains duplicate names")
    return tuple(pairs)


def _parse_commands(value: object) -> tuple[tuple[str, ...], ...]:
    items = _required_sequence(value, "build receipt commands")
    commands: list[tuple[str, ...]] = []
    for item in items:
        command = _required_sequence(item, "build receipt command")
        if not command or not all(isinstance(argument, str) for argument in command):
            raise RuntimeValidationError("build receipt commands must contain strings")
        commands.append(tuple(cast(Sequence[str], command)))
    return tuple(commands)


def _safe_receipt_artifact(build_root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeValidationError("build receipt contains an unsafe wheel path")
    path = (build_root / Path(*relative.parts)).resolve(strict=True)
    try:
        path.relative_to(build_root)
    except ValueError as error:
        raise RuntimeValidationError("build receipt wheel escapes its root") from error
    if not path.is_file():
        raise RuntimeValidationError(f"build receipt wheel is not a file: {path}")
    return path


def _parse_embedded_provenance(
    contents: bytes,
    *,
    distribution: str,
    location: str,
) -> EmbeddedProvenanceEvidence:
    try:
        tree = ast.parse(contents.decode("ascii"), filename=location)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise RuntimeValidationError(
            f"{distribution} embedded provenance is not valid ASCII Python"
        ) from error
    values: dict[str, object] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            try:
                values[statement.targets[0].id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as error:
                raise RuntimeValidationError(
                    f"{distribution} embedded provenance has a non-literal value"
                ) from error
    schema_version = values.get("SCHEMA_VERSION")
    ktransformers_revision = values.get("KTRANSFORMERS_REVISION")
    sglang_revision = values.get("SGLANG_REVISION")
    if schema_version != 1:
        raise RuntimeValidationError(
            f"{distribution} embedded provenance schema is not 1"
        )
    if ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION:
        raise RuntimeValidationError(
            f"{distribution} embedded KTransformers revision is not admitted"
        )
    if sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        raise RuntimeValidationError(
            f"{distribution} embedded SGLang revision is not admitted"
        )
    return EmbeddedProvenanceEvidence(
        distribution=distribution,
        location=location,
        sha256=hashlib.sha256(contents).hexdigest(),
        schema_version=cast(int, schema_version),
        ktransformers_revision=cast(str, ktransformers_revision),
        sglang_revision=cast(str, sglang_revision),
    )


def observe_build_provenance(receipt_path: Path) -> BuildProvenanceEvidence:
    resolved_receipt = receipt_path.resolve(strict=True)
    receipt_bytes = resolved_receipt.read_bytes()
    try:
        root = _required_mapping(json.loads(receipt_bytes), "build receipt")
    except json.JSONDecodeError as error:
        raise RuntimeValidationError(f"invalid build receipt JSON: {error}") from error
    _require_exact_keys(
        root,
        frozenset(
            {
                "schema_version",
                "status",
                "build_id",
                "builder_sha256",
                "source",
                "toolchain",
                "layout",
                "build_environment",
                "commands",
                "bootstrap_wheels",
                "runtime_wheels",
                "completed_at_utc",
            }
        ),
        "build receipt",
    )
    if root.get("schema_version") != 2:
        raise RuntimeValidationError("build receipt schema_version must be 2")
    if root.get("status") != "wheel_build_complete":
        raise RuntimeValidationError("build receipt is not wheel_build_complete")
    try:
        completed_at = datetime.fromisoformat(
            _required_string(root, "completed_at_utc")
        )
    except ValueError as error:
        raise RuntimeValidationError(
            "build receipt completion time is invalid"
        ) from error
    if completed_at.tzinfo is None:
        raise RuntimeValidationError("build receipt completion time has no timezone")

    build_id = _required_string(root, "build_id")
    if SHA256_PATTERN.fullmatch(build_id) is None:
        raise RuntimeValidationError("build receipt build_id is not a SHA-256 digest")
    builder_sha256 = _required_string(root, "builder_sha256")
    builder_path = (
        Path(__file__).with_name("build_sglang_kt_runtime.py").resolve(strict=True)
    )
    if (
        SHA256_PATTERN.fullmatch(builder_sha256) is None
        or _sha256(builder_path) != builder_sha256
    ):
        raise RuntimeValidationError(
            "build receipt builder SHA-256 does not match the current builder"
        )
    source = _parse_source_observation(
        _required_mapping(root.get("source"), "build receipt source")
    )
    toolchain = _parse_toolchain_observation(
        _required_mapping(root.get("toolchain"), "build receipt toolchain")
    )
    layout = _parse_layout(
        _required_mapping(root.get("layout"), "build receipt layout")
    )
    build_root = Path(layout.build_root).resolve(strict=True)
    expected_layout = RuntimeBuildLayout(
        build_root=str(build_root),
        state=str(build_root / "state"),
        cache=str(build_root / "cache"),
        logs=str(build_root / "state/logs"),
        wheels=str(build_root / "wheels"),
        receipt=str(build_root / "build-receipt.json"),
    )
    if layout != expected_layout:
        raise RuntimeValidationError("build receipt layout is not canonical")
    if (
        build_root.name != build_id
        or build_root.parent.name != toolchain.host_profile
        or Path(layout.receipt).resolve(strict=True) != resolved_receipt
    ):
        raise RuntimeValidationError("build receipt path disagrees with its identity")

    if source.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION:
        raise RuntimeValidationError(
            "build receipt KTransformers revision is not admitted"
        )
    if source.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        raise RuntimeValidationError("build receipt SGLang revision is not admitted")
    if source.package_version != EXPECTED_PACKAGE_VERSION:
        raise RuntimeValidationError("build receipt package version is not admitted")
    sglang_submodule_path = (
        RuntimeSourcePins.admitted().sglang_submodule_path.as_posix()
    )
    if (
        next(
            (
                item.revision
                for item in source.submodules
                if item.path == sglang_submodule_path
            ),
            None,
        )
        != source.sglang_revision
    ):
        raise RuntimeValidationError(
            "SGLang source revision disagrees with its gitlink"
        )

    profile = HOST_BUILD_PROFILES[toolchain.host_profile]
    environment = _parse_string_pairs(
        root.get("build_environment"), "build receipt environment"
    )
    expected_environment = runtime_build_environment(source, toolchain, profile, layout)
    if environment != expected_environment:
        raise RuntimeValidationError("build receipt environment is not canonical")
    commands = _parse_commands(root.get("commands"))
    expected_commands = runtime_build_commands(
        Path(toolchain.tool("python").path), layout
    )
    if commands != expected_commands:
        raise RuntimeValidationError("build receipt commands are not canonical")
    expected_build_id = calculate_runtime_build_id(
        source, toolchain, profile, builder_sha256
    )
    if build_id != expected_build_id:
        raise RuntimeValidationError(
            "build receipt build_id does not match its contract"
        )

    bootstrap_items = _required_sequence(
        root.get("bootstrap_wheels"), "bootstrap wheels"
    )
    bootstrap_wheels: list[WheelArtifactEvidence] = []
    pins_by_distribution = {
        _normalized_distribution(pin.distribution): pin for pin in BOOTSTRAP_WHEEL_PINS
    }
    for item in bootstrap_items:
        artifact = _required_mapping(item, "bootstrap wheel")
        _require_exact_keys(
            artifact,
            frozenset(
                {
                    "distribution",
                    "version",
                    "filename",
                    "path",
                    "size_bytes",
                    "sha256",
                    "root_is_purelib",
                    "tags",
                }
            ),
            "bootstrap wheel",
        )
        distribution = _normalized_distribution(
            _required_string(artifact, "distribution")
        )
        pin = pins_by_distribution.get(distribution)
        if pin is None:
            raise RuntimeValidationError(
                f"bootstrap wheel distribution is not pinned: {distribution}"
            )
        filename = _required_string(artifact, "filename")
        relative_path = _required_string(artifact, "path")
        expected_size = _required_integer(artifact, "size_bytes")
        expected_sha256 = _required_string(artifact, "sha256")
        tags = _required_sequence(artifact.get("tags"), "bootstrap wheel tags")
        if (
            _required_string(artifact, "version") != pin.version
            or filename != pin.filename
            or relative_path != f"state/bootstrap-wheels/{pin.filename}"
            or expected_sha256 != pin.sha256
            or _required_boolean(artifact, "root_is_purelib") is not True
            or tuple(tags) != ("py3-none-any",)
        ):
            raise RuntimeValidationError(
                f"bootstrap wheel {distribution} does not match its exact pin"
            )
        path = _safe_receipt_artifact(build_root, relative_path)
        if path.name != filename or path.stat().st_size != expected_size:
            raise RuntimeValidationError(
                f"bootstrap wheel {distribution} does not match receipt"
            )
        if _sha256(path) != expected_sha256:
            raise RuntimeValidationError(
                f"bootstrap wheel {distribution} does not match its pinned SHA-256"
            )
        bootstrap_wheels.append(
            WheelArtifactEvidence(
                distribution=distribution,
                version=pin.version,
                path=str(path),
                size_bytes=expected_size,
                sha256=expected_sha256,
            )
        )
    if (
        len(bootstrap_wheels) != len(BOOTSTRAP_WHEEL_PINS)
        or {wheel.distribution for wheel in bootstrap_wheels}
        != set(pins_by_distribution)
        or {path.name for path in (build_root / "state/bootstrap-wheels").glob("*.whl")}
        != {pin.filename for pin in BOOTSTRAP_WHEEL_PINS}
    ):
        raise RuntimeValidationError("bootstrap wheel set is not exact")

    wheel_items = _required_sequence(root.get("runtime_wheels"), "runtime wheels")
    wheels: list[WheelArtifactEvidence] = []
    embedded_provenance: list[EmbeddedProvenanceEvidence] = []
    kt_extension_member = ""
    kt_extension_sha256 = ""
    for item in wheel_items:
        artifact = _required_mapping(item, "runtime wheel")
        _require_exact_keys(
            artifact,
            frozenset(
                {
                    "distribution",
                    "version",
                    "filename",
                    "path",
                    "size_bytes",
                    "sha256",
                    "root_is_purelib",
                    "tags",
                }
            ),
            "runtime wheel",
        )
        distribution = _normalized_distribution(
            _required_string(artifact, "distribution")
        )
        version = _required_string(artifact, "version")
        filename = _required_string(artifact, "filename")
        relative_path = _required_string(artifact, "path")
        expected_size = _required_integer(artifact, "size_bytes")
        expected_sha256 = _required_string(artifact, "sha256")
        if version != EXPECTED_PACKAGE_VERSION:
            raise RuntimeValidationError(
                f"{distribution} wheel version is not admitted"
            )
        if SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise RuntimeValidationError(f"{distribution} wheel SHA-256 is invalid")
        path = _safe_receipt_artifact(build_root, relative_path)
        if (
            path.name != filename
            or relative_path != f"wheels/{filename}"
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha256
        ):
            raise RuntimeValidationError(f"{distribution} wheel does not match receipt")
        wheels.append(
            WheelArtifactEvidence(
                distribution=distribution,
                version=version,
                path=str(path),
                size_bytes=expected_size,
                sha256=expected_sha256,
            )
        )
        if distribution in {"kt-kernel", "sglang-kt"}:
            try:
                with zipfile.ZipFile(path) as archive:
                    package = "kt_kernel" if distribution == "kt-kernel" else "sglang"
                    provenance_member = f"{package}/_exo_build_provenance.py"
                    if archive.namelist().count(provenance_member) != 1:
                        raise RuntimeValidationError(
                            f"{distribution} wheel has no unique embedded provenance"
                        )
                    embedded_provenance.append(
                        _parse_embedded_provenance(
                            archive.read(provenance_member),
                            distribution=distribution,
                            location=provenance_member,
                        )
                    )
                    if distribution == "kt-kernel":
                        candidates = tuple(
                            name
                            for name in archive.namelist()
                            if PurePosixPath(name).name.startswith("kt_kernel_ext")
                            and name.endswith(".so")
                        )
                        if len(candidates) != 1:
                            raise RuntimeValidationError(
                                "kt-kernel wheel must contain one selected native extension"
                            )
                        kt_extension_member = candidates[0]
                        kt_extension_sha256 = _zip_member_sha256(archive, candidates[0])
            except zipfile.BadZipFile as error:
                raise RuntimeValidationError(
                    "kt-kernel wheel is not a ZIP archive"
                ) from error

    if (
        len(wheels) != len(EXPECTED_WHEEL_DISTRIBUTIONS)
        or frozenset(wheel.distribution for wheel in wheels)
        != EXPECTED_WHEEL_DISTRIBUTIONS
    ):
        raise RuntimeValidationError("runtime wheel set is not exact")
    if {item.distribution for item in embedded_provenance} != {
        "kt-kernel",
        "sglang-kt",
    }:
        raise RuntimeValidationError("embedded wheel provenance set is not exact")
    if not kt_extension_member or not kt_extension_sha256:
        raise RuntimeValidationError("kt-kernel native extension identity is missing")

    if toolchain.cuda_architectures != "86":
        raise RuntimeValidationError("build receipt CUDA architecture must be 86")
    if not REQUIRED_CPU_FEATURES.issubset(toolchain.cpu_features):
        raise RuntimeValidationError("build receipt lacks required AMX CPU features")
    if toolchain.hostname != toolchain.host_profile:
        raise RuntimeValidationError("build receipt host identity is not admitted")
    return BuildProvenanceEvidence(
        receipt_path=str(resolved_receipt),
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        build_id=build_id,
        builder_sha256=builder_sha256,
        host_profile=toolchain.host_profile,
        hostname=toolchain.hostname,
        ktransformers_revision=source.ktransformers_revision,
        sglang_revision=source.sglang_revision,
        package_version=source.package_version,
        cuda_architectures=toolchain.cuda_architectures,
        cpu_features=toolchain.cpu_features,
        build_environment=environment,
        bootstrap_wheels=tuple(
            sorted(bootstrap_wheels, key=lambda wheel: wheel.distribution)
        ),
        wheels=tuple(sorted(wheels, key=lambda wheel: wheel.distribution)),
        embedded_provenance=tuple(
            sorted(embedded_provenance, key=lambda item: item.distribution)
        ),
        kt_extension_member=kt_extension_member,
        kt_extension_sha256=kt_extension_sha256,
        verified=True,
    )


def _direct_url_sha256(distribution: importlib.metadata.Distribution) -> str | None:
    text = distribution.read_text("direct_url.json")
    if text is None:
        return None
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict):
        return None
    archive = document.get("archive_info")
    if not isinstance(archive, dict):
        return None
    hashes = archive.get("hashes")
    if isinstance(hashes, dict):
        value = hashes.get("sha256")
        if isinstance(value, str) and SHA256_PATTERN.fullmatch(value):
            return value
    value = archive.get("hash")
    if isinstance(value, str) and value.startswith("sha256="):
        digest = value.removeprefix("sha256=")
        if SHA256_PATTERN.fullmatch(digest):
            return digest
    return None


def observe_packages() -> tuple[PackageEvidence, ...]:
    observations: list[PackageEvidence] = []
    for name, expected_version in EXPECTED_DISTRIBUTION_VERSIONS.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            observations.append(
                PackageEvidence(name, expected_version, None, None, None, None)
            )
            continue
        installer_text = distribution.read_text("INSTALLER")
        installer = installer_text.strip() if installer_text else None
        observations.append(
            PackageEvidence(
                distribution=name,
                expected_version=expected_version,
                observed_version=distribution.version,
                location=str(Path(distribution.locate_file("")).resolve()),
                installer=installer,
                direct_url_sha256=_direct_url_sha256(distribution),
            )
        )
    return tuple(observations)


def _installed_embedded_provenance(
    distribution_name: str,
    relative_path: str,
) -> EmbeddedProvenanceEvidence:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeValidationError(
            f"{distribution_name} is not installed for provenance inspection"
        ) from error
    path = Path(distribution.locate_file(relative_path)).resolve(strict=True)
    if not path.is_file():
        raise RuntimeValidationError(
            f"{distribution_name} installed provenance is not a file"
        )
    return _parse_embedded_provenance(
        path.read_bytes(),
        distribution=distribution_name,
        location=str(path),
    )


def _module_version(module_name: str) -> str | None:
    module = importlib.import_module(module_name)
    value = getattr(module, "__version__", None)
    return value if isinstance(value, str) and value else None


def observe_runtime_identity() -> RuntimeIdentityEvidence:
    torch = importlib.import_module("torch")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return RuntimeIdentityEvidence(
        torch_module_version=_module_version("torch"),
        torch_cuda_version=(
            cuda_version if isinstance(cuda_version, str) and cuda_version else None
        ),
        transformers_distribution_version=next(
            (
                item.observed_version
                for item in observe_packages()
                if item.distribution == "transformers-kt"
            ),
            None,
        ),
        transformers_module_version=_module_version("transformers"),
        sgl_kernel_build_id=calculate_sglang_kt_artifact_build_id(
            "sgl_kernel", "common_ops"
        ),
        deep_gemm_build_id=calculate_sglang_kt_artifact_build_id(
            "deep_gemm", "deep_gemm"
        ),
        kt_kernel_build_id=calculate_sglang_kt_artifact_build_id(
            "kt_kernel", "kt_kernel_ext", ("kt_kernel_ext",)
        ),
        embedded_provenance=(
            _installed_embedded_provenance(
                "kt-kernel", "kt_kernel/_exo_build_provenance.py"
            ),
            _installed_embedded_provenance(
                "sglang-kt", "sglang/_exo_build_provenance.py"
            ),
        ),
    )


def _parse_index_list(value: str) -> tuple[int, ...]:
    indexes: set[int] = set()
    for component in value.strip().split(","):
        if not component:
            continue
        bounds = component.split("-", maxsplit=1)
        try:
            start = int(bounds[0])
            end = int(bounds[-1])
        except ValueError as error:
            raise RuntimeValidationError(
                f"invalid Linux index list: {value}"
            ) from error
        if start < 0 or end < start:
            raise RuntimeValidationError(f"invalid Linux index range: {component}")
        indexes.update(range(start, end + 1))
    return tuple(sorted(indexes))


def _common_cpu_features() -> tuple[str, ...]:
    sets: list[set[str]] = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name.strip().lower() in {"flags", "features"}:
                sets.append(set(value.strip().lower().split()))
    except OSError as error:
        raise RuntimeValidationError(f"cannot read CPU features: {error}") from error
    if not sets:
        raise RuntimeValidationError("no CPU feature records were found")
    return tuple(sorted(set.intersection(*sets)))


def _allowed_memory_nodes() -> tuple[int, ...]:
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError as error:
        raise RuntimeValidationError(f"cannot read process status: {error}") from error
    value = next(
        (
            line.partition(":")[2].strip()
            for line in lines
            if line.startswith("Mems_allowed_list:")
        ),
        None,
    )
    if value is None:
        raise RuntimeValidationError("process memory-node allowance is unavailable")
    return _parse_index_list(value)


def _invoked_python_executable() -> str:
    executable = Path(sys.executable)
    if not executable.is_absolute():
        raise RuntimeValidationError("Python executable path is not absolute")
    return str(executable)


def observe_host() -> HostEvidence:
    node_paths = tuple(
        sorted(
            Path("/sys/devices/system/node").glob("node[0-9]*"),
            key=lambda path: int(path.name.removeprefix("node")),
        )
    )
    if not node_paths:
        raise RuntimeValidationError("no Linux NUMA nodes were found")
    numa_nodes = tuple(
        NumaNodeEvidence(
            node=int(path.name.removeprefix("node")),
            cpu_ids=_parse_index_list((path / "cpulist").read_text()),
        )
        for path in node_paths
    )
    process = ProcessEvidence(
        hostname=socket.gethostname().split(".", maxsplit=1)[0].lower(),
        # Preserve the invoked venv path. Resolving its interpreter symlink would
        # discard the overlay identity needed by the launch specification.
        executable=_invoked_python_executable(),
        python_implementation=platform.python_implementation(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        pid=os.getpid(),
        parent_pid=os.getppid(),
        uid=os.getuid(),
        gid=os.getgid(),
        cwd=str(Path.cwd()),
        argv=tuple(sys.argv),
        affinity_cpu_ids=tuple(sorted(os.sched_getaffinity(0))),
        allowed_memory_nodes=_allowed_memory_nodes(),
        environment=tuple(
            (name, os.environ[name])
            for name in HOST_ENVIRONMENT_NAMES
            if name in os.environ
        ),
    )
    return HostEvidence(
        process=process,
        operating_system=platform.platform(),
        machine=platform.machine(),
        cpu_features=_common_cpu_features(),
        numa_nodes=numa_nodes,
        collection_method="direct_execution_only",
        profiler="none",
    )


def _finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _numerical_evidence(
    torch: Any,
    actual: Any,
    reference: Any,
    *,
    seed: int,
    tolerance: float,
) -> NumericalEvidence:
    actual_float = actual.to(dtype=torch.float32)
    reference_float = reference.to(dtype=torch.float32)
    output_finite = bool(torch.isfinite(actual_float).all().item())
    reference_finite = bool(torch.isfinite(reference_float).all().item())
    if output_finite and reference_finite:
        absolute_error = torch.abs(actual_float - reference_float)
        mean_error = float(absolute_error.mean().item())
        maximum_error = float(absolute_error.max().item())
        reference_mean = float(torch.abs(reference_float).mean().item())
        relative_error = mean_error / max(reference_mean, 1e-12)
    else:
        mean_error = math.nan
        maximum_error = math.nan
        reference_mean = math.nan
        relative_error = math.nan
    return NumericalEvidence(
        shape=tuple(int(dimension) for dimension in actual.shape),
        seed=seed,
        execution_dtype=str(actual.dtype),
        reference_dtype="torch.float32",
        output_finite=output_finite,
        reference_finite=reference_finite,
        mean_absolute_error=_finite_float(mean_error),
        maximum_absolute_error=_finite_float(maximum_error),
        reference_mean_absolute=_finite_float(reference_mean),
        relative_l1_error=_finite_float(relative_error),
        relative_l1_tolerance=tolerance,
    )


def _nvidia_gpu_row(gpu_uuid: str) -> tuple[str, str, str, int, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeValidationError("nvidia-smi is unavailable")
    result = subprocess.run(
        (
            executable,
            "--query-gpu=uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeValidationError(f"nvidia-smi failed: {detail}")
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    matching = tuple(row for row in rows if len(row) == 5 and row[0] == gpu_uuid)
    if len(matching) != 1:
        raise RuntimeValidationError("selected GPU UUID was not uniquely reported")
    row = matching[0]
    try:
        memory_bytes = int(row[3]) * 1024 * 1024
    except ValueError as error:
        raise RuntimeValidationError(
            "nvidia-smi returned invalid GPU memory"
        ) from error
    return row[0], row[1], row[2], memory_bytes, row[4]


def _canonical_torch_device_uuid(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    raw_value = str(value)
    candidate = raw_value if raw_value.startswith("GPU-") else f"GPU-{raw_value}"
    canonical_value = (
        candidate if GPU_UUID_PATTERN.fullmatch(candidate) is not None else None
    )
    return raw_value, canonical_value


def _module_file_sha256(module: Any) -> tuple[str, str]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeValidationError("KT native extension has no file identity")
    path = Path(module_file).resolve(strict=True)
    return str(path), _sha256(path)


def _moe_reference(
    torch: Any,
    input_tensor: Any,
    expert_ids: Any,
    routing_weights: Any,
    gate_proj: Any,
    up_proj: Any,
    down_proj: Any,
) -> Any:
    input_float = input_tensor.float()
    gate_float = gate_proj.float()
    up_float = up_proj.float()
    down_float = down_proj.float()
    output = torch.zeros(
        (input_tensor.shape[0], input_tensor.shape[1]), dtype=torch.float32
    )
    for token in range(input_tensor.shape[0]):
        hidden = input_float[token]
        for slot in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, slot].item())
            gate = torch.nn.functional.silu(torch.mv(gate_float[expert], hidden))
            up = torch.mv(up_float[expert], hidden)
            expert_output = torch.mv(down_float[expert], gate * up)
            output[token].add_(expert_output, alpha=float(routing_weights[token, slot]))
    return output


class DirectRuntimeBackend:
    """Live backend. Imports Torch and KT only when numerical checks start."""

    def observe_provenance(self, config: ValidationConfig) -> BuildProvenanceEvidence:
        return observe_build_provenance(Path(config.build_receipt_path))

    def observe_packages(self) -> tuple[PackageEvidence, ...]:
        return observe_packages()

    def observe_runtime_identity(self) -> RuntimeIdentityEvidence:
        return observe_runtime_identity()

    def observe_host(self) -> HostEvidence:
        return observe_host()

    def run_cuda(self, config: ValidationConfig) -> CudaExecutionEvidence:
        torch = importlib.import_module("torch")
        if not bool(torch.cuda.is_available()):
            raise RuntimeValidationError("Torch CUDA is unavailable")
        device_count = int(torch.cuda.device_count())
        if device_count < 1:
            raise RuntimeValidationError("Torch reports no CUDA device")
        uuid, pci_bus_id, name, total_memory, driver = _nvidia_gpu_row(config.gpu_uuid)
        properties = torch.cuda.get_device_properties(0)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
        property_uuid = getattr(properties, "uuid", None)
        torch_device_uuid_raw, torch_device_uuid = _canonical_torch_device_uuid(
            property_uuid
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(RANDOM_SEED)
        left = (
            torch.randn((128, 192), generator=generator, dtype=torch.float32) / 4
        ).to(torch.bfloat16)
        right = (
            torch.randn((192, 96), generator=generator, dtype=torch.float32) / 4
        ).to(torch.bfloat16)
        reference = torch.matmul(left.float(), right.float())
        actual = torch.matmul(left.to(device="cuda:0"), right.to(device="cuda:0"))
        torch.cuda.synchronize(0)
        numerical = _numerical_evidence(
            torch,
            actual.cpu(),
            reference,
            seed=RANDOM_SEED,
            tolerance=CUDA_RELATIVE_L1_TOLERANCE,
        )
        return CudaExecutionEvidence(
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            logical_device_count=device_count,
            logical_device_index=0,
            gpu_uuid=uuid,
            pci_bus_id=pci_bus_id,
            gpu_name=name,
            compute_capability=cast(tuple[int, int], capability),
            total_memory_bytes=total_memory,
            driver_version=driver,
            torch_cuda_version=getattr(torch.version, "cuda", None),
            torch_device_uuid_raw=torch_device_uuid_raw,
            torch_device_uuid=torch_device_uuid,
            numerical=numerical,
        )

    def run_amx(
        self,
        config: ValidationConfig,
        qlen: int,
        route: ExecutionRoute,
    ) -> AmxExecutionEvidence:
        torch = importlib.import_module("torch")
        kt_kernel = importlib.import_module("kt_kernel")
        extension = importlib.import_module("kt_kernel.kt_kernel_ext")
        extension_path, extension_sha256 = _module_file_sha256(extension)
        cpu_variant = str(getattr(kt_kernel, "__cpu_variant__", ""))

        worker_config = extension.WorkerPoolConfig()
        worker_config.subpool_count = len(config.worker_pool.numa_nodes)
        worker_config.subpool_numa_map = list(config.worker_pool.numa_nodes)
        worker_config.subpool_thread_count = list(
            config.worker_pool.threads_per_subpool
        )
        observed_worker_pool = WorkerPoolAssignment(
            numa_nodes=tuple(int(node) for node in worker_config.subpool_numa_map),
            threads_per_subpool=tuple(
                int(count) for count in worker_config.subpool_thread_count
            ),
        )
        cpu_infer = extension.CPUInfer(worker_config)

        generator = torch.Generator(device="cpu")
        seed = RANDOM_SEED + qlen
        generator.manual_seed(seed)
        gate_proj = (
            (
                torch.randn(
                    (EXPERT_COUNT, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    generator=generator,
                    dtype=torch.float32,
                )
                / 10
            )
            .to(torch.bfloat16)
            .contiguous()
        )
        up_proj = (
            (
                torch.randn(
                    (EXPERT_COUNT, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    generator=generator,
                    dtype=torch.float32,
                )
                / 10
            )
            .to(torch.bfloat16)
            .contiguous()
        )
        down_proj = (
            (
                torch.randn(
                    (EXPERT_COUNT, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                    generator=generator,
                    dtype=torch.float32,
                )
                / 10
            )
            .to(torch.bfloat16)
            .contiguous()
        )
        moe_config = extension.moe.MOEConfig(
            EXPERT_COUNT, TOP_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0
        )
        moe_config.max_len = MAX_QLEN
        moe_config.gate_proj = gate_proj.data_ptr()
        moe_config.up_proj = up_proj.data_ptr()
        moe_config.down_proj = down_proj.data_ptr()
        moe_config.gate_scale = 0
        moe_config.up_scale = 0
        moe_config.down_scale = 0
        moe_config.pool = cpu_infer.backend_
        moe = extension.moe.AMXBF16_MOE(moe_config)
        mapping = torch.arange(EXPERT_COUNT, dtype=torch.int64).contiguous()
        cpu_infer.submit(moe.load_weights_task(mapping.data_ptr()))
        cpu_infer.sync()

        expert_ids = (
            torch.stack(
                [
                    torch.randperm(EXPERT_COUNT, generator=generator)[:TOP_K]
                    for _ in range(qlen)
                ]
            )
            .to(torch.int64)
            .contiguous()
        )
        routing_weights = torch.rand(
            (qlen, TOP_K), generator=generator, dtype=torch.float32
        ).contiguous()
        routing_weights.div_(routing_weights.sum(dim=1, keepdim=True))
        if route == "direct":
            input_tensor = (
                (
                    torch.randn(
                        (qlen, HIDDEN_SIZE),
                        generator=generator,
                        dtype=torch.float32,
                    )
                    / 100
                )
                .to(torch.bfloat16)
                .contiguous()
            )
            reference_input = input_tensor
            output = torch.full(
                (qlen, HIDDEN_SIZE), math.nan, dtype=torch.bfloat16
            ).contiguous()
        else:
            input_tensor = torch.full(
                (qlen, HIDDEN_SIZE),
                math.nan,
                dtype=torch.bfloat16,
                pin_memory=True,
            ).contiguous()
            reference_input = torch.full(
                (qlen, HIDDEN_SIZE),
                CUDA_STREAM_INPUT_VALUE,
                dtype=torch.bfloat16,
            ).contiguous()
            output = torch.full(
                (qlen, HIDDEN_SIZE),
                math.nan,
                dtype=torch.bfloat16,
                pin_memory=True,
            ).contiguous()
        qlen_tensor = torch.tensor([qlen], dtype=torch.int32).contiguous()
        reference = _moe_reference(
            torch,
            reference_input,
            expert_ids,
            routing_weights,
            gate_proj,
            up_proj,
            down_proj,
        )
        task = moe.forward_task(
            qlen_tensor.data_ptr(),
            TOP_K,
            expert_ids.data_ptr(),
            routing_weights.data_ptr(),
            input_tensor.data_ptr(),
            output.data_ptr(),
            False,
        )

        direct_submit_count = 0
        direct_sync_count = 0
        stream_submit_count = 0
        stream_sync_count = 0
        stream_api_available: bool | None = None
        stream_id: int | None = None
        default_stream_id: int | None = None
        cuda_input_producer_count = 0
        cuda_input_to_cpu_copy_count = 0
        cuda_output_from_cpu_copy_count = 0
        cuda_output_consumer_count = 0
        cuda_input_produced_checksum: float | None = None
        cpu_input_observed_checksum: float | None = None
        cuda_output_consumed_l1: float | None = None
        cuda_output_reference_l1: float | None = None
        numerical_actual = output
        numerical_reference = reference
        if route == "direct":
            cpu_infer.submit(task)
            direct_submit_count = 1
            cpu_infer.sync()
            direct_sync_count = 1
        else:
            submit_with_stream = getattr(cpu_infer, "submit_with_cuda_stream", None)
            sync_with_stream = getattr(cpu_infer, "sync_with_cuda_stream", None)
            if callable(submit_with_stream) and callable(sync_with_stream):
                stream_api_available = True
                stream = torch.cuda.Stream(device=0)
                default_stream = torch.cuda.default_stream(device=0)
                stream_id = int(stream.cuda_stream)
                default_stream_id = int(default_stream.cuda_stream)
                with torch.cuda.stream(stream):
                    cuda_input = torch.empty(
                        (qlen, HIDDEN_SIZE),
                        device="cuda:0",
                        dtype=torch.bfloat16,
                    )
                    cuda_output = torch.empty_like(cuda_input)
                    consumed_output, bridge_counts = _enqueue_cuda_stream_bridge(
                        produce_cuda_input=lambda: cuda_input.fill_(
                            CUDA_STREAM_INPUT_VALUE
                        ),
                        copy_cuda_input_to_cpu=lambda: input_tensor.copy_(
                            cuda_input, non_blocking=True
                        ),
                        submit_cpu_work=lambda: submit_with_stream(stream_id, task),
                        sync_cpu_work=lambda: sync_with_stream(stream_id),
                        copy_cpu_output_to_cuda=lambda: cuda_output.copy_(
                            output, non_blocking=True
                        ),
                        consume_cuda_output=lambda: cuda_output.mul(
                            CUDA_STREAM_OUTPUT_MULTIPLIER
                        ),
                    )
                    stream_submit_count = 1
                    stream_sync_count = 1
                    cuda_input_producer_count = bridge_counts.input_producer
                    cuda_input_to_cpu_copy_count = bridge_counts.input_to_cpu_copy
                    cuda_output_from_cpu_copy_count = bridge_counts.output_from_cpu_copy
                    cuda_output_consumer_count = bridge_counts.output_consumer
                stream.synchronize()
                cuda_input_produced_checksum = _finite_float(
                    float(cuda_input.float().sum().item())
                )
                cpu_input_observed_checksum = _finite_float(
                    float(input_tensor.float().sum().item())
                )
                numerical_actual = consumed_output.cpu()
                numerical_reference = reference * CUDA_STREAM_OUTPUT_MULTIPLIER
                cuda_output_consumed_l1 = _finite_float(
                    float(numerical_actual.float().abs().sum().item())
                )
                cuda_output_reference_l1 = _finite_float(
                    float(numerical_reference.float().abs().sum().item())
                )
            else:
                stream_api_available = False

        numerical = _numerical_evidence(
            torch,
            numerical_actual,
            numerical_reference,
            seed=seed,
            tolerance=AMX_RELATIVE_L1_TOLERANCE,
        )
        return AmxExecutionEvidence(
            route=route,
            qlen=qlen,
            kernel_class=type(moe).__name__,
            cpu_variant=cpu_variant,
            extension_path=extension_path,
            extension_sha256=extension_sha256,
            worker_pool=observed_worker_pool,
            load_submit_count=1,
            load_sync_count=1,
            direct_forward_submit_count=direct_submit_count,
            direct_forward_sync_count=direct_sync_count,
            cuda_stream_api_available=stream_api_available,
            cuda_stream_submit_count=stream_submit_count,
            cuda_stream_sync_count=stream_sync_count,
            cuda_stream_id=stream_id,
            default_cuda_stream_id=default_stream_id,
            cuda_input_producer_count=cuda_input_producer_count,
            cuda_input_to_cpu_copy_count=cuda_input_to_cpu_copy_count,
            cuda_output_from_cpu_copy_count=cuda_output_from_cpu_copy_count,
            cuda_output_consumer_count=cuda_output_consumer_count,
            cuda_input_produced_checksum=cuda_input_produced_checksum,
            cpu_input_observed_checksum=cpu_input_observed_checksum,
            cuda_output_consumed_l1=cuda_output_consumed_l1,
            cuda_output_reference_l1=cuda_output_reference_l1,
            numerical=numerical,
        )


def _version_matches(observed: str | None, expected: str) -> bool:
    return observed is not None and observed.partition("+")[0] == expected


def _numerical_failures(
    name: str,
    evidence: NumericalEvidence | None,
    *,
    expected_shape: tuple[int, ...],
    expected_seed: int,
    expected_tolerance: float,
) -> list[str]:
    if evidence is None:
        return [f"{name}: numerical evidence is missing"]
    failures: list[str] = []
    if not evidence.output_finite or not evidence.reference_finite:
        failures.append(f"{name}: numerical output/reference is non-finite")
    if evidence.execution_dtype != "torch.bfloat16":
        failures.append(f"{name}: execution dtype is not torch.bfloat16")
    if evidence.reference_dtype != "torch.float32":
        failures.append(f"{name}: reference dtype is not torch.float32")
    if evidence.shape != expected_shape:
        failures.append(f"{name}: numerical shape is not exact")
    if evidence.seed != expected_seed:
        failures.append(f"{name}: numerical seed is not exact")
    if evidence.relative_l1_tolerance != expected_tolerance:
        failures.append(f"{name}: numerical tolerance is not exact")
    if (
        evidence.relative_l1_error is None
        or evidence.relative_l1_error > expected_tolerance
    ):
        failures.append(f"{name}: relative L1 error exceeds tolerance")
    if (
        evidence.mean_absolute_error is None
        or evidence.mean_absolute_error < 0
        or evidence.maximum_absolute_error is None
        or evidence.maximum_absolute_error < 0
        or evidence.reference_mean_absolute is None
        or evidence.reference_mean_absolute <= 0
    ):
        failures.append(f"{name}: numerical error metrics are incomplete")
    return failures


def _static_failures(
    config: ValidationConfig,
    provenance: BuildProvenanceEvidence,
    packages: tuple[PackageEvidence, ...],
    host: HostEvidence,
    runtime_identity: RuntimeIdentityEvidence,
) -> list[str]:
    failures: list[str] = []
    if not provenance.verified:
        failures.append("provenance: build receipt was not verified")
    if provenance.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION:
        failures.append("provenance: KTransformers revision is not admitted")
    if provenance.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        failures.append("provenance: SGLang revision is not admitted")
    if provenance.package_version != EXPECTED_PACKAGE_VERSION:
        failures.append("provenance: runtime package version is not admitted")
    if provenance.cuda_architectures != "86":
        failures.append("provenance: CUDA architecture is not SM86")
    if provenance.hostname != host.process.hostname:
        failures.append("provenance: build host differs from execution host")
    if SHA256_PATTERN.fullmatch(provenance.kt_extension_sha256) is None:
        failures.append("provenance: KT extension digest is invalid")
    if SHA256_PATTERN.fullmatch(provenance.builder_sha256) is None:
        failures.append("provenance: builder digest is invalid")
    build_environment = dict(provenance.build_environment)
    if any(
        build_environment.get(name) != expected
        for name, expected in REQUIRED_BUILD_ENVIRONMENT.items()
    ):
        failures.append("provenance: native build environment is not exact")
    if (
        len(provenance.wheels) != len(EXPECTED_WHEEL_DISTRIBUTIONS)
        or frozenset(wheel.distribution for wheel in provenance.wheels)
        != EXPECTED_WHEEL_DISTRIBUTIONS
    ):
        failures.append("provenance: runtime wheel set is not exact")
    bootstrap_by_name = {
        wheel.distribution: wheel for wheel in provenance.bootstrap_wheels
    }
    if (
        len(provenance.bootstrap_wheels) != len(BOOTSTRAP_WHEEL_PINS)
        or set(bootstrap_by_name)
        != {_normalized_distribution(pin.distribution) for pin in BOOTSTRAP_WHEEL_PINS}
        or any(
            bootstrap_by_name[_normalized_distribution(pin.distribution)].version
            != pin.version
            or bootstrap_by_name[_normalized_distribution(pin.distribution)].sha256
            != pin.sha256
            for pin in BOOTSTRAP_WHEEL_PINS
            if _normalized_distribution(pin.distribution) in bootstrap_by_name
        )
    ):
        failures.append("provenance: bootstrap wheel set is not exact")

    package_by_name = {item.distribution: item for item in packages}
    if len(packages) != len(EXPECTED_DISTRIBUTION_VERSIONS) or set(
        package_by_name
    ) != set(EXPECTED_DISTRIBUTION_VERSIONS):
        failures.append("packages: evidence set is not exact")
    wheel_by_name = {wheel.distribution: wheel for wheel in provenance.wheels}
    for name, expected_version in EXPECTED_DISTRIBUTION_VERSIONS.items():
        package = package_by_name.get(name)
        if package is None or not _version_matches(
            package.observed_version, expected_version
        ):
            failures.append(f"packages: {name} version is not {expected_version}")
            continue
        wheel = wheel_by_name.get(name)
        if wheel is not None and package.direct_url_sha256 != wheel.sha256:
            failures.append(f"packages: {name} install is not bound to build wheel")

    if runtime_identity.torch_module_version != EXPECTED_TORCH_MODULE_VERSION:
        failures.append(
            f"runtime identity: Torch module is not {EXPECTED_TORCH_MODULE_VERSION}"
        )
    if runtime_identity.torch_cuda_version != EXPECTED_TORCH_CUDA_VERSION:
        failures.append(
            f"runtime identity: Torch CUDA is not {EXPECTED_TORCH_CUDA_VERSION}"
        )
    if (
        runtime_identity.transformers_distribution_version
        != EXPECTED_DISTRIBUTION_VERSIONS["transformers-kt"]
    ):
        failures.append("runtime identity: transformers distribution is not exact")
    if (
        runtime_identity.transformers_module_version
        != EXPECTED_TRANSFORMERS_MODULE_VERSION
    ):
        failures.append("runtime identity: transformers module version is not exact")
    for name, build_id in (
        ("sgl_kernel", runtime_identity.sgl_kernel_build_id),
        ("deep_gemm", runtime_identity.deep_gemm_build_id),
        ("kt_kernel", runtime_identity.kt_kernel_build_id),
    ):
        if build_id is None or SHA256_PATTERN.fullmatch(build_id) is None:
            failures.append(f"runtime identity: {name} build ID is unavailable")
    wheel_embedded = {
        item.distribution: item for item in provenance.embedded_provenance
    }
    installed_embedded = {
        item.distribution: item for item in runtime_identity.embedded_provenance
    }
    if set(wheel_embedded) != {"kt-kernel", "sglang-kt"} or set(installed_embedded) != {
        "kt-kernel",
        "sglang-kt",
    }:
        failures.append("runtime identity: embedded provenance set is not exact")
    else:
        for distribution in sorted(wheel_embedded):
            wheel_item = wheel_embedded[distribution]
            installed_item = installed_embedded[distribution]
            if (
                wheel_item.sha256 != installed_item.sha256
                or installed_item.schema_version != 1
                or installed_item.ktransformers_revision
                != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
                or installed_item.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
            ):
                failures.append(
                    f"runtime identity: {distribution} embedded provenance disagrees"
                )

    features = set(host.cpu_features)
    missing_features = sorted(REQUIRED_CPU_FEATURES - features)
    if missing_features:
        failures.append("host: missing CPU features " + ", ".join(missing_features))
    if host.profiler != "none" or host.collection_method != "direct_execution_only":
        failures.append("host: profiler-backed evidence is forbidden")
    if host.process.python_implementation != "CPython" or (
        host.process.python_version[0:2] != (3, 12)
    ):
        failures.append("host: runtime must use CPython 3.12")
    environment = dict(host.process.environment)
    if environment.get("CUDA_VISIBLE_DEVICES") != config.gpu_uuid:
        failures.append("host: CUDA_VISIBLE_DEVICES is not the exact requested UUID")
    node_by_id = {node.node: node for node in host.numa_nodes}
    affinity = set(host.process.affinity_cpu_ids)
    allowed_memory = set(host.process.allowed_memory_nodes)
    for node, threads in zip(
        config.worker_pool.numa_nodes,
        config.worker_pool.threads_per_subpool,
        strict=True,
    ):
        node_evidence = node_by_id.get(node)
        if node_evidence is None:
            failures.append(f"host: requested NUMA node {node} does not exist")
            continue
        if node not in allowed_memory:
            failures.append(f"host: requested NUMA node {node} is not memory-allowed")
        available_cpus = affinity.intersection(node_evidence.cpu_ids)
        if len(available_cpus) < threads:
            failures.append(
                f"host: NUMA node {node} has {len(available_cpus)} affinity CPUs "
                f"for {threads} worker threads"
            )
    return failures


def _runtime_failures(
    config: ValidationConfig,
    provenance: BuildProvenanceEvidence,
    runtime_identity: RuntimeIdentityEvidence,
    cuda: CudaExecutionEvidence | None,
    amx: tuple[AmxExecutionEvidence, ...],
    stream: AmxExecutionEvidence | None,
) -> list[str]:
    failures: list[str] = []
    if cuda is None:
        failures.append("cuda: execution evidence is missing")
    else:
        if cuda.visible_devices != config.gpu_uuid or cuda.gpu_uuid != config.gpu_uuid:
            failures.append("cuda: selected GPU UUID is not exact")
        if cuda.logical_device_count != 1 or cuda.logical_device_index != 0:
            failures.append(
                "cuda: runtime must expose exactly one logical GPU at index 0"
            )
        if cuda.compute_capability != EXPECTED_COMPUTE_CAPABILITY:
            failures.append("cuda: selected GPU is not compute capability 8.6")
        if cuda.torch_device_uuid != config.gpu_uuid:
            failures.append("cuda: Torch logical device 0 UUID is not exact")
        if cuda.torch_cuda_version != runtime_identity.torch_cuda_version:
            failures.append("cuda: Torch CUDA version changed after static collection")
        failures.extend(
            _numerical_failures(
                "cuda",
                cuda.numerical,
                expected_shape=(128, 96),
                expected_seed=RANDOM_SEED,
                expected_tolerance=CUDA_RELATIVE_L1_TOLERANCE,
            )
        )

    if tuple(sorted(item.qlen for item in amx)) != EXPECTED_QLENS:
        failures.append("amx: direct qlen evidence must be exactly 1 and 16")
    for item in amx:
        name = f"amx qlen={item.qlen}"
        if item.route != "direct":
            failures.append(f"{name}: route is not direct")
        if item.kernel_class != "AMXBF16_MOE" or item.cpu_variant.lower() != "amx":
            failures.append(f"{name}: AMXBF16_MOE was not selected")
        if item.worker_pool != config.worker_pool:
            failures.append(f"{name}: WorkerPoolConfig assignment changed")
        if item.extension_sha256 != provenance.kt_extension_sha256:
            failures.append(f"{name}: loaded KT extension differs from build wheel")
        if (
            item.load_submit_count != 1
            or item.load_sync_count != 1
            or item.direct_forward_submit_count != 1
            or item.direct_forward_sync_count != 1
            or item.cuda_stream_submit_count != 0
            or item.cuda_stream_sync_count != 0
            or item.cuda_input_producer_count != 0
            or item.cuda_input_to_cpu_copy_count != 0
            or item.cuda_output_from_cpu_copy_count != 0
            or item.cuda_output_consumer_count != 0
        ):
            failures.append(f"{name}: direct submit/sync counts are not exact")
        if any(
            value is not None
            for value in (
                item.cuda_input_produced_checksum,
                item.cpu_input_observed_checksum,
                item.cuda_output_consumed_l1,
                item.cuda_output_reference_l1,
            )
        ):
            failures.append(f"{name}: CUDA-stream value evidence must be absent")
        failures.extend(
            _numerical_failures(
                name,
                item.numerical,
                expected_shape=(item.qlen, HIDDEN_SIZE),
                expected_seed=RANDOM_SEED + item.qlen,
                expected_tolerance=AMX_RELATIVE_L1_TOLERANCE,
            )
        )

    if stream is None:
        failures.append("cuda stream: execution evidence is missing")
    else:
        if stream.route != "cuda_stream":
            failures.append("cuda stream: route is not cuda_stream")
        if stream.qlen != 1:
            failures.append("cuda stream: validation qlen must be 1")
        if stream.kernel_class != "AMXBF16_MOE" or stream.cpu_variant.lower() != "amx":
            failures.append("cuda stream: AMXBF16_MOE was not selected")
        if stream.worker_pool != config.worker_pool:
            failures.append("cuda stream: WorkerPoolConfig assignment changed")
        if stream.extension_sha256 != provenance.kt_extension_sha256:
            failures.append("cuda stream: loaded KT extension differs from build wheel")
        if stream.cuda_stream_api_available is not True:
            failures.append("cuda stream: KT CUDA-stream API is unavailable")
        if (
            stream.load_submit_count != 1
            or stream.load_sync_count != 1
            or stream.direct_forward_submit_count != 0
            or stream.direct_forward_sync_count != 0
            or stream.cuda_stream_submit_count != 1
            or stream.cuda_stream_sync_count != 1
        ):
            failures.append("cuda stream: submit/sync counts are not exact")
        if (
            stream.cuda_input_producer_count != 1
            or stream.cuda_input_to_cpu_copy_count != 1
            or stream.cuda_output_from_cpu_copy_count != 1
            or stream.cuda_output_consumer_count != 1
        ):
            failures.append("cuda stream: two-way ordering counts are not exact")
        if (
            stream.cuda_stream_id is None
            or stream.default_cuda_stream_id is None
            or stream.cuda_stream_id == stream.default_cuda_stream_id
        ):
            failures.append("cuda stream: stream was not non-default")
        expected_input_checksum = stream.qlen * HIDDEN_SIZE * CUDA_STREAM_INPUT_VALUE
        if (
            stream.cuda_input_produced_checksum != expected_input_checksum
            or stream.cpu_input_observed_checksum != expected_input_checksum
        ):
            failures.append("cuda stream: CUDA-produced input did not reach AMX input")
        if (
            stream.cuda_output_consumed_l1 is None
            or stream.cuda_output_reference_l1 is None
            or stream.cuda_output_reference_l1 <= 0
            or abs(stream.cuda_output_consumed_l1 - stream.cuda_output_reference_l1)
            / stream.cuda_output_reference_l1
            > AMX_RELATIVE_L1_TOLERANCE
        ):
            failures.append("cuda stream: AMX output was not consumed by CUDA")
        failures.extend(
            _numerical_failures(
                "cuda stream",
                stream.numerical,
                expected_shape=(1, HIDDEN_SIZE),
                expected_seed=RANDOM_SEED + 1,
                expected_tolerance=AMX_RELATIVE_L1_TOLERANCE,
            )
        )
    return failures


def _backend_failure(step: str, error: Exception) -> str:
    detail = str(error).strip() or "no diagnostic"
    return f"{step}: {type(error).__name__}: {detail}"


def execute_validation(
    config: ValidationConfig,
    backend: ValidationBackend | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RuntimeValidationReceipt:
    selected_backend = backend or DirectRuntimeBackend()
    now = clock or (lambda: datetime.now(UTC))
    provenance: BuildProvenanceEvidence | None = None
    packages: tuple[PackageEvidence, ...] = ()
    runtime_identity: RuntimeIdentityEvidence | None = None
    host: HostEvidence | None = None
    cuda: CudaExecutionEvidence | None = None
    amx: list[AmxExecutionEvidence] = []
    stream: AmxExecutionEvidence | None = None
    failures: list[str] = []

    try:
        provenance = selected_backend.observe_provenance(config)
    except Exception as error:
        failures.append(_backend_failure("provenance", error))
    try:
        packages = selected_backend.observe_packages()
    except Exception as error:
        failures.append(_backend_failure("packages", error))
    try:
        runtime_identity = selected_backend.observe_runtime_identity()
    except Exception as error:
        failures.append(_backend_failure("runtime identity", error))
    try:
        host = selected_backend.observe_host()
    except Exception as error:
        failures.append(_backend_failure("host", error))

    if provenance is not None and runtime_identity is not None and host is not None:
        failures.extend(
            _static_failures(
                config,
                provenance,
                packages,
                host,
                runtime_identity,
            )
        )
    if failures:
        return RuntimeValidationReceipt(
            schema_version=SCHEMA_VERSION,
            status="failed",
            generated_at_utc=now().astimezone(UTC).isoformat(),
            profiler="none",
            config=config,
            provenance=provenance,
            packages=packages,
            runtime_identity=runtime_identity,
            host=host,
            cuda=None,
            amx=(),
            cuda_stream=None,
            capabilities=(),
            failures=tuple(failures),
        )

    try:
        cuda = selected_backend.run_cuda(config)
    except Exception as error:
        failures.append(_backend_failure("cuda", error))
    for qlen in EXPECTED_QLENS:
        try:
            amx.append(selected_backend.run_amx(config, qlen, "direct"))
        except Exception as error:
            failures.append(_backend_failure(f"amx qlen={qlen}", error))
    try:
        stream = selected_backend.run_amx(config, 1, "cuda_stream")
    except Exception as error:
        failures.append(_backend_failure("cuda stream", error))

    if provenance is not None and runtime_identity is not None:
        failures.extend(
            _runtime_failures(
                config,
                provenance,
                runtime_identity,
                cuda,
                tuple(amx),
                stream,
            )
        )
    status: ReceiptStatus = "failed" if failures else "passed"
    capabilities = ("kt_bf16_amx_executed_v1",) if status == "passed" else ()
    return RuntimeValidationReceipt(
        schema_version=SCHEMA_VERSION,
        status=status,
        generated_at_utc=now().astimezone(UTC).isoformat(),
        profiler="none",
        config=config,
        provenance=provenance,
        packages=packages,
        runtime_identity=runtime_identity,
        host=host,
        cuda=cuda,
        amx=tuple(amx),
        cuda_stream=stream,
        capabilities=capabilities,
        failures=tuple(failures),
    )


def receipt_json(receipt: RuntimeValidationReceipt) -> str:
    return json.dumps(asdict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_receipt(path: Path, receipt: RuntimeValidationReceipt) -> None:
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(receipt_json(receipt))
    os.replace(temporary, destination)


class _CliArguments(argparse.Namespace):
    gpu_uuid: str
    build_receipt: Path
    numa_nodes: str
    threads_per_subpool: str
    output: Path


def _comma_separated_integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(component) for component in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("value must not be empty")
    return result


def _parse_arguments() -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument(
        "--numa-nodes",
        required=True,
        help="Comma-separated physical NUMA node IDs for WorkerPoolConfig",
    )
    parser.add_argument(
        "--threads-per-subpool",
        required=True,
        help="Comma-separated thread counts aligned with --numa-nodes",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON receipt path; required because native KT may write to stdout",
    )
    arguments = _CliArguments()
    parser.parse_args(namespace=arguments)
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    try:
        config = ValidationConfig(
            gpu_uuid=arguments.gpu_uuid,
            build_receipt_path=str(arguments.build_receipt.resolve(strict=True)),
            worker_pool=WorkerPoolAssignment(
                numa_nodes=_comma_separated_integers(arguments.numa_nodes),
                threads_per_subpool=_comma_separated_integers(
                    arguments.threads_per_subpool
                ),
            ),
        )
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(f"invalid validation configuration: {error}") from error
    receipt = execute_validation(config)
    write_receipt(arguments.output, receipt)
    print(
        f"GLM-4.7 KT runtime validation {receipt.status}; receipt: "
        f"{arguments.output.resolve(strict=False)}",
        file=sys.stderr,
    )
    return 0 if receipt.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
