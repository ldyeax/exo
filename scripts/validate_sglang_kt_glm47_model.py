#!/usr/bin/env python3
"""Produce admission-grade evidence for a pinned GLM-4.7 model smoke test.

The parent binds immutable inputs without importing Torch, SGLang, or
KTransformers. A NUMA-bound disposable child loads those runtimes, executes the
model probe, cleans up, and returns canonical evidence through a parent-owned
anonymous file. The parent publishes only after a clean child exit.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import stat
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, cast

from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.dont_write_bytecode = True
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CONTEXT_LENGTH,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_LAYER_COUNT,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_ROUTED_EXPERT_COUNT,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import (
    SglangKtLoadedModelContract,
    SglangKtModelContractError,
    SglangKtVerifiedModelSnapshot,
    load_sglang_kt_model_contract,
    verify_sglang_kt_model_snapshot,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
    SglangKtModelRuntimeValidationReceiptError,
    canonicalize_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtBoundFile,
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from scripts.sglang_kt_glm47_live import (
    MINIMUM_GPU_MEMORY_HEADROOM_BYTES,
    MINIMUM_HOST_MEMORY_HEADROOM_BYTES,
    Glm47LiveValidationError,
    ValidatorBundleIdentity,
    available_host_memory_bytes,
    build_live_child_environment,
    calculate_memory_headroom,
    calculate_validator_bundle,
    load_bound_kernel_runtime_receipt,
    require_current_process_binding,
    require_disposable_child_evidence_transport,
    run_disposable_live_child,
    validator_bundle_paths,
    write_disposable_child_evidence,
)
from scripts.sglang_kt_glm47_receipt import (
    Glm47ReceiptAssemblyError,
    build_glm47_model_runtime_validation_receipt_payload,
    require_glm47_model_runtime_validation_receipt_parent_bindings,
)

if TYPE_CHECKING:
    from scripts.sglang_kt_glm47_backend import Glm47BackendEvidence

SCHEMA_VERSION: Final = 1
PROFILER: Final[Literal["none"]] = "none"
GLM_4_7_FLASH_BF16_INDEX_SHA256: Final = (
    "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
)
_PROCESS_SPEC_MAXIMUM_BYTES: Final = 1024 * 1024
_KERNEL_RECEIPT_MAXIMUM_BYTES: Final = 8 * 1024 * 1024
_PUBLISHED_RECEIPT_MAXIMUM_BYTES: Final = 4 * 1024 * 1024
_MAXIMUM_VALIDATION_RESIDENT_GPU_EXPERTS: Final = 4
_INTERNAL_EVIDENCE_DESCRIPTOR_ARGUMENT: Final = "--internal-evidence-fd"
_NUMACTL_EXECUTABLE: Final = Path("/usr/bin/numactl")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class Glm47ModelValidationError(RuntimeError):
    """Raised when software preflight cannot bind every required input."""


class _CliArguments(argparse.Namespace):
    process_spec: Path
    expected_process_spec_sha256: str
    model_contract: Path
    expected_model_contract_receipt_sha256: str
    kernel_runtime_receipt: Path
    expected_kernel_receipt_sha256: str
    output: Path
    internal_evidence_fd: int | None


@dataclass(frozen=True)
class BoundProcessSpec:
    path: Path
    receipt_size_bytes: int
    receipt_sha256: str
    process_spec_sha256: str
    process_spec: SglangKtProcessLaunchSpec


@dataclass(frozen=True)
class ResourceHeadroom:
    available_bytes: int
    required_bytes: int
    minimum_headroom_bytes: int
    remaining_bytes: int
    sufficient: bool


@dataclass(frozen=True)
class ResidentExpertRoute:
    resident_gpu_expert_ids: tuple[int, ...]
    selected_expert_ids: tuple[int, ...]
    cpu_expert_ids: tuple[int, ...]
    gpu_expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class Glm47ModelSoftwarePreflight:
    validator: ValidatorBundleIdentity
    process: BoundProcessSpec
    model_contract: SglangKtLoadedModelContract
    kernel_runtime_receipt: SglangKtBoundFile
    output: Path
    route: ResidentExpertRoute


@dataclass(frozen=True)
class Glm47ModelExecutionPreflight(Glm47ModelSoftwarePreflight):
    model_snapshot: SglangKtVerifiedModelSnapshot


def _absolute_normalized_path(value: str) -> Path:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or path.as_posix() != value
        or value != os.path.normpath(value)
    ):
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return Path(value)


def _sha256_digest(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("SHA-256 must be lowercase 64-hex")
    return value


def _inherited_file_descriptor(value: str) -> int:
    try:
        descriptor = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "internal evidence descriptor must be an integer"
        ) from error
    if descriptor < 3:
        raise argparse.ArgumentTypeError(
            "internal evidence descriptor must be inherited"
        )
    return descriptor


def parse_arguments(arguments: list[str] | None = None) -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-spec", required=True, type=_absolute_normalized_path)
    parser.add_argument(
        "--expected-process-spec-sha256", required=True, type=_sha256_digest
    )
    parser.add_argument(
        "--model-contract", required=True, type=_absolute_normalized_path
    )
    parser.add_argument(
        "--expected-model-contract-receipt-sha256",
        required=True,
        type=_sha256_digest,
    )
    parser.add_argument(
        "--kernel-runtime-receipt", required=True, type=_absolute_normalized_path
    )
    parser.add_argument(
        "--expected-kernel-receipt-sha256", required=True, type=_sha256_digest
    )
    parser.add_argument("--output", required=True, type=_absolute_normalized_path)
    parser.add_argument(
        _INTERNAL_EVIDENCE_DESCRIPTOR_ARGUMENT,
        type=_inherited_file_descriptor,
        default=None,
        help=argparse.SUPPRESS,
    )
    namespace = _CliArguments()
    parser.parse_args(arguments, namespace=namespace)
    return namespace


def calculate_resource_headroom(
    *,
    available_bytes: int,
    required_bytes: int,
    minimum_headroom_bytes: int,
) -> ResourceHeadroom:
    """Calculate whether a planned allocation preserves explicit headroom."""

    values = (available_bytes, required_bytes, minimum_headroom_bytes)
    if any(type(value) is not int for value in values):
        raise TypeError("resource byte counts must be integers")
    if available_bytes < 0 or required_bytes < 0 or minimum_headroom_bytes <= 0:
        raise ValueError(
            "available and required bytes must be nonnegative and headroom positive"
        )
    remaining_bytes = available_bytes - required_bytes
    return ResourceHeadroom(
        available_bytes=available_bytes,
        required_bytes=required_bytes,
        minimum_headroom_bytes=minimum_headroom_bytes,
        remaining_bytes=remaining_bytes,
        sufficient=remaining_bytes >= minimum_headroom_bytes,
    )


def require_resource_headroom(
    resource_name: str,
    *,
    available_bytes: int,
    required_bytes: int,
    minimum_headroom_bytes: int,
) -> ResourceHeadroom:
    """Return exact headroom evidence or fail before a live model load."""

    if not resource_name:
        raise ValueError("resource_name must be nonempty")
    headroom = calculate_resource_headroom(
        available_bytes=available_bytes,
        required_bytes=required_bytes,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )
    if not headroom.sufficient:
        raise Glm47ModelValidationError(
            f"{resource_name} would retain {headroom.remaining_bytes} bytes, below "
            f"the required {headroom.minimum_headroom_bytes}-byte headroom"
        )
    return headroom


def select_resident_expert_route(resident_gpu_experts: int) -> ResidentExpertRoute:
    """Select four deterministic experts while preserving a CPU route.

    The pinned uniform placement makes expert IDs ``0..N-1`` resident on every
    wrapped layer.  Counts 1 through 63 must exercise both backends; count zero
    is the explicit CPU-only control.
    """

    if type(resident_gpu_experts) is not int:
        raise TypeError("resident_gpu_experts must be an integer")
    if not 0 <= resident_gpu_experts < GLM_4_7_FLASH_ROUTED_EXPERT_COUNT:
        raise ValueError("resident_gpu_experts must be between 0 and 63")

    resident_ids = tuple(range(resident_gpu_experts))
    if resident_gpu_experts <= 1:
        selected_ids = (0, 1, 2, 3)
    elif resident_gpu_experts <= GLM_4_7_FLASH_ROUTED_EXPERT_COUNT - 2:
        selected_ids = (0, 1, resident_gpu_experts, resident_gpu_experts + 1)
    else:
        selected_ids = (0, 1, 2, GLM_4_7_FLASH_ROUTED_EXPERT_COUNT - 1)

    resident_set = frozenset(resident_ids)
    gpu_ids = tuple(
        expert_id for expert_id in selected_ids if expert_id in resident_set
    )
    cpu_ids = tuple(
        expert_id for expert_id in selected_ids if expert_id not in resident_set
    )
    if len(selected_ids) != 4 or len(set(selected_ids)) != 4 or not cpu_ids:
        raise AssertionError("deterministic expert-route construction is invalid")
    if resident_gpu_experts > 0 and not gpu_ids:
        raise AssertionError("mixed expert-route construction omitted the GPU")
    return ResidentExpertRoute(
        resident_gpu_expert_ids=resident_ids,
        selected_expert_ids=selected_ids,
        cpu_expert_ids=cpu_ids,
        gpu_expert_ids=gpu_ids,
    )


def calculate_validator_source_bundle(
    repository_root: Path | None = None,
) -> ValidatorBundleIdentity:
    """Bind the exact validator implementation admitted by schema v1."""

    root = repository_root or Path(os.path.abspath(Path(__file__).parent.parent))
    try:
        return calculate_validator_bundle(validator_bundle_paths(root))
    except (Glm47LiveValidationError, ValueError, OSError) as error:
        raise Glm47ModelValidationError(
            f"cannot bind validator source bundle under: {root}"
        ) from error


def require_immutable_validator_source_deployment(
    repository_root: Path,
    validator: ValidatorBundleIdentity,
    *,
    required_uid: int = 0,
) -> None:
    """Require an exact root-owned, read-only tree for every executed local byte."""

    if type(required_uid) is not int or required_uid < 0:
        raise ValueError("required deployment UID must be a nonnegative integer")
    if (
        not repository_root.is_absolute()
        or repository_root != Path(os.path.normpath(repository_root))
        or repository_root.is_symlink()
    ):
        raise Glm47ModelValidationError(
            "validator deployment root must be normalized, absolute, and direct"
        )
    expected_files = frozenset(
        repository_root / relative_path
        for relative_path in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    )
    if frozenset(Path(source.path) for source in validator.sources) != expected_files:
        raise Glm47ModelValidationError(
            "validator deployment does not match its bound source paths"
        )

    observed_files: set[Path] = set()
    paths = (repository_root, *repository_root.rglob("*"))
    for path in paths:
        try:
            observed = path.lstat()
        except OSError as error:
            raise Glm47ModelValidationError(
                f"cannot inspect validator deployment path: {path}"
            ) from error
        if stat.S_ISLNK(observed.st_mode):
            raise Glm47ModelValidationError(
                f"validator deployment contains a symbolic link: {path}"
            )
        if observed.st_uid != required_uid or observed.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise Glm47ModelValidationError(
                f"validator deployment is not root-owned and read-only: {path}"
            )
        if stat.S_ISREG(observed.st_mode):
            if observed.st_nlink != 1:
                raise Glm47ModelValidationError(
                    f"validator source is not singly linked: {path}"
                )
            observed_files.add(path)
        elif not stat.S_ISDIR(observed.st_mode):
            raise Glm47ModelValidationError(
                f"validator deployment contains a special file: {path}"
            )
    if frozenset(observed_files) != expected_files:
        raise Glm47ModelValidationError(
            "validator deployment must contain exactly the bound source closure"
        )


def require_immutable_model_snapshot(
    snapshot_path: Path,
    *,
    required_uid: int = 0,
) -> None:
    """Require a root-owned snapshot protected by mount or file permissions."""

    if type(required_uid) is not int or required_uid < 0:
        raise ValueError("required model UID must be a nonnegative integer")
    try:
        filesystem_read_only = bool(os.statvfs(snapshot_path).f_flag & os.ST_RDONLY)
    except OSError as error:
        raise Glm47ModelValidationError(
            f"cannot inspect model snapshot filesystem: {snapshot_path}"
        ) from error
    paths = (snapshot_path, *snapshot_path.rglob("*"))
    for path in paths:
        try:
            observed = path.lstat()
        except OSError as error:
            raise Glm47ModelValidationError(
                f"cannot inspect model snapshot path: {path}"
            ) from error
        if stat.S_ISLNK(observed.st_mode):
            raise Glm47ModelValidationError(
                f"model snapshot contains a symbolic link: {path}"
            )
        if observed.st_uid != required_uid:
            raise Glm47ModelValidationError(f"model snapshot is not root-owned: {path}")
        if not filesystem_read_only and observed.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise Glm47ModelValidationError(f"model snapshot is writable: {path}")
        if not (stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)):
            raise Glm47ModelValidationError(
                f"model snapshot contains a special file: {path}"
            )


def perform_execution_preflight(
    arguments: _CliArguments,
    *,
    validator_repository_root: Path | None = None,
) -> Glm47ModelExecutionPreflight:
    """Verify every source and model byte before the live runtime can import."""

    repository_root = validator_repository_root or Path(
        os.path.abspath(Path(__file__).parent.parent)
    )
    software = perform_software_preflight(
        arguments,
        validator_repository_root=repository_root,
    )
    require_immutable_validator_source_deployment(repository_root, software.validator)
    process_spec = software.process.process_spec
    load_bound_kernel_runtime_receipt(
        process_spec,
        software.kernel_runtime_receipt.path,
        expected_receipt_sha256=software.kernel_runtime_receipt.sha256,
    )
    snapshot_path = Path(process_spec.model_path)
    require_immutable_model_snapshot(snapshot_path)
    try:
        verified = verify_sglang_kt_model_snapshot(
            snapshot_path,
            Path(software.model_contract.path),
            expected_contract_sha256=software.model_contract.contract_sha256,
            expected_model_id=process_spec.model_id,
            expected_revision=process_spec.expected_model_revision,
            expected_ktransformers_method=process_spec.ktransformers_method,
        )
    except SglangKtModelContractError as error:
        raise Glm47ModelValidationError(
            f"model snapshot does not match its exact contract: {snapshot_path}"
        ) from error
    loaded = software.model_contract
    if (
        verified.model_path != process_spec.model_path
        or verified.contract_path != loaded.path
        or verified.contract_receipt_sha256 != loaded.receipt_sha256
        or verified.contract_sha256 != loaded.contract_sha256
        or verified.config_sha256 != _contract_file_sha256(loaded, "config")
        or verified.index_sha256 != _contract_file_sha256(loaded, "safetensors_index")
        or verified.weight_map_entries != loaded.contract.weight_map_entries
        or verified.shard_count
        != sum(file.role == "weight_shard" for file in loaded.contract.files)
        or verified.physical_weight_bytes != loaded.contract.physical_weight_bytes
    ):
        raise Glm47ModelValidationError(
            "verified model snapshot identity changed across contract loading"
        )
    return Glm47ModelExecutionPreflight(
        validator=software.validator,
        process=software.process,
        model_contract=software.model_contract,
        kernel_runtime_receipt=software.kernel_runtime_receipt,
        output=software.output,
        route=software.route,
        model_snapshot=verified,
    )


def _validate_glm47_process_spec(process_spec: SglangKtProcessLaunchSpec) -> None:
    plan = process_spec.plan
    if plan.target_profile not in {
        GLM_4_7_FLASH_TARGET_PROFILE,
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    }:
        raise Glm47ModelValidationError("process spec is not a GLM-4.7 profile")
    if process_spec.pipeline_rank != 0 or len(plan.stages) != 1:
        raise Glm47ModelValidationError("GLM-4.7 model validation requires rank 0 PP=1")
    stage = process_spec.stage
    if (
        stage.pipeline_rank != 0
        or stage.start_layer != 0
        or stage.end_layer != GLM_4_7_FLASH_LAYER_COUNT
        or plan.total_layers != GLM_4_7_FLASH_LAYER_COUNT
    ):
        raise Glm47ModelValidationError(
            "GLM-4.7 model validation requires the complete 0:47 layer range"
        )
    if (
        plan.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
        or plan.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or plan.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or plan.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        or process_spec.model_contract_sha256
        != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    ):
        raise Glm47ModelValidationError(
            "process spec does not use pinned GLM-4.7 inputs"
        )
    if (
        plan.context_length != GLM_4_7_FLASH_CONTEXT_LENGTH
        or plan.max_total_tokens != GLM_4_7_FLASH_MAX_TOTAL_TOKENS
        or plan.max_concurrent_requests != 1
        or plan.static_memory_fraction != 0.8
    ):
        raise Glm47ModelValidationError("process spec launch limits are not pinned")
    if (
        stage.model_path != stage.ktransformers_weight_path
        or stage.ktransformers_method != "BF16"
        or stage.max_deferred_experts_per_token != 0
        or stage.hca_devices
        or process_spec.attention_backend != "flashinfer"
        or process_spec.kv_cache_dtype != "bfloat16"
    ):
        raise Glm47ModelValidationError(
            "process spec backend invariants are not pinned"
        )
    if plan.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE:
        if stage.resident_gpu_experts != 0:
            raise Glm47ModelValidationError(
                "CPU-routed control must have zero resident GPU experts"
            )
    elif (
        not 1
        <= stage.resident_gpu_experts
        <= (_MAXIMUM_VALIDATION_RESIDENT_GPU_EXPERTS)
    ):
        raise Glm47ModelValidationError(
            "initial mixed validation requires between 1 and 4 resident GPU experts"
        )


def load_bound_process_spec(
    path: Path,
    *,
    expected_process_spec_sha256: str,
) -> BoundProcessSpec:
    """Load a stable process-spec file and verify its canonical identity."""

    if _SHA256_PATTERN.fullmatch(expected_process_spec_sha256) is None:
        raise Glm47ModelValidationError("expected process-spec SHA-256 is invalid")
    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=_PROCESS_SPEC_MAXIMUM_BYTES,
        )
        parse_sglang_kt_strict_json(bound_file.contents)
        process_spec = SglangKtProcessLaunchSpec.model_validate_json(
            bound_file.contents
        )
    except (RecursionError, SglangKtReceiptFileError, ValidationError) as error:
        raise Glm47ModelValidationError(f"invalid process-spec file: {path}") from error
    process_spec_sha256 = calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    if process_spec_sha256 != expected_process_spec_sha256:
        raise Glm47ModelValidationError(
            "process spec does not match the expected canonical SHA-256"
        )
    _validate_glm47_process_spec(process_spec)
    return BoundProcessSpec(
        path=bound_file.path,
        receipt_size_bytes=len(bound_file.contents),
        receipt_sha256=bound_file.sha256,
        process_spec_sha256=process_spec_sha256,
        process_spec=process_spec,
    )


def _contract_file_sha256(
    loaded_contract: SglangKtLoadedModelContract,
    role: Literal["config", "safetensors_index"],
) -> str:
    matches = tuple(
        file.sha256 for file in loaded_contract.contract.files if file.role == role
    )
    if len(matches) != 1:
        raise Glm47ModelValidationError(
            f"model contract does not contain exactly one {role} file"
        )
    return matches[0]


def load_bound_model_contract(
    path: Path,
    *,
    expected_receipt_sha256: str,
    process_spec: SglangKtProcessLaunchSpec,
) -> SglangKtLoadedModelContract:
    """Load the small contract document without hashing the model snapshot."""

    if _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None:
        raise Glm47ModelValidationError("expected model-contract SHA-256 is invalid")
    expected_contract_sha256 = process_spec.model_contract_sha256
    if expected_contract_sha256 is None:
        raise Glm47ModelValidationError("process spec does not pin a model contract")
    try:
        loaded = load_sglang_kt_model_contract(
            path,
            expected_contract_sha256=expected_contract_sha256,
        )
    except SglangKtModelContractError as error:
        raise Glm47ModelValidationError(f"invalid model contract: {path}") from error
    if loaded.receipt_sha256 != expected_receipt_sha256:
        raise Glm47ModelValidationError(
            "model contract does not match the expected raw receipt SHA-256"
        )
    contract = loaded.contract
    if (
        contract.model_id != process_spec.model_id
        or contract.revision != process_spec.expected_model_revision
        or contract.ktransformers_method != process_spec.ktransformers_method
        or contract.full_indexer_layer_starts != (0,)
        or _contract_file_sha256(loaded, "config") != GLM_4_7_FLASH_BF16_CONFIG_SHA256
        or _contract_file_sha256(loaded, "safetensors_index")
        != GLM_4_7_FLASH_BF16_INDEX_SHA256
    ):
        raise Glm47ModelValidationError(
            "model contract identity disagrees with the GLM-4.7 launch"
        )
    return loaded


def _bind_kernel_runtime_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> SglangKtBoundFile:
    if _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None:
        raise Glm47ModelValidationError("expected kernel receipt SHA-256 is invalid")
    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=_KERNEL_RECEIPT_MAXIMUM_BYTES,
        )
        root = parse_sglang_kt_strict_json(bound_file.contents)
    except (RecursionError, SglangKtReceiptFileError) as error:
        raise Glm47ModelValidationError(
            f"invalid kernel receipt file: {path}"
        ) from error
    if not isinstance(root, dict):
        raise Glm47ModelValidationError("kernel receipt must be a JSON object")
    if bound_file.sha256 != expected_receipt_sha256:
        raise Glm47ModelValidationError(
            "kernel receipt does not match the expected raw SHA-256"
        )
    return bound_file


def perform_software_preflight(
    arguments: _CliArguments,
    *,
    validator_repository_root: Path | None = None,
) -> Glm47ModelSoftwarePreflight:
    """Bind software-only inputs without creating model execution evidence."""

    input_paths = (
        arguments.process_spec,
        arguments.model_contract,
        arguments.kernel_runtime_receipt,
    )
    if len(set(input_paths)) != len(input_paths) or arguments.output in input_paths:
        raise Glm47ModelValidationError("input and output paths must be distinct")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise Glm47ModelValidationError(
            f"refusing to replace existing output: {arguments.output}"
        )
    process = load_bound_process_spec(
        arguments.process_spec,
        expected_process_spec_sha256=arguments.expected_process_spec_sha256,
    )
    model_contract = load_bound_model_contract(
        arguments.model_contract,
        expected_receipt_sha256=arguments.expected_model_contract_receipt_sha256,
        process_spec=process.process_spec,
    )
    kernel_receipt = _bind_kernel_runtime_receipt(
        arguments.kernel_runtime_receipt,
        expected_receipt_sha256=arguments.expected_kernel_receipt_sha256,
    )
    validator = calculate_validator_source_bundle(validator_repository_root)
    route = select_resident_expert_route(
        process.process_spec.stage.resident_gpu_experts
    )
    return Glm47ModelSoftwarePreflight(
        validator=validator,
        process=process,
        model_contract=model_contract,
        kernel_runtime_receipt=kernel_receipt,
        output=arguments.output,
        route=route,
    )


def _public_argument_vector(arguments: _CliArguments) -> tuple[str, ...]:
    """Reconstruct the exact public invocation passed to the disposable child."""

    return (
        "--process-spec",
        str(arguments.process_spec),
        "--expected-process-spec-sha256",
        arguments.expected_process_spec_sha256,
        "--model-contract",
        str(arguments.model_contract),
        "--expected-model-contract-receipt-sha256",
        arguments.expected_model_contract_receipt_sha256,
        "--kernel-runtime-receipt",
        str(arguments.kernel_runtime_receipt),
        "--expected-kernel-receipt-sha256",
        arguments.expected_kernel_receipt_sha256,
        "--output",
        str(arguments.output),
    )


def build_disposable_child_command(
    arguments: _CliArguments,
    preflight: Glm47ModelSoftwarePreflight,
    *,
    validator_script: Path | None = None,
    numactl_executable: Path = _NUMACTL_EXECUTABLE,
) -> tuple[str, ...]:
    """Build a resource-bound command using the launch spec's exact Python."""

    script = validator_script or Path(__file__).resolve()
    process_spec = preflight.process.process_spec
    if (
        not script.is_absolute()
        or not numactl_executable.is_absolute()
        or Path(process_spec.executable) == Path("/")
    ):
        raise Glm47ModelValidationError(
            "live child executables and validator source must be absolute"
        )
    cpu_list = ",".join(str(cpu_id) for cpu_id in process_spec.cpu_cores)
    memory_node_list = ",".join(
        str(memory_node) for memory_node in process_spec.memory_nodes
    )
    if not cpu_list or not memory_node_list:
        raise Glm47ModelValidationError("live child resource bindings are empty")
    return (
        str(numactl_executable),
        "--physcpubind",
        cpu_list,
        "--membind",
        memory_node_list,
        process_spec.executable,
        str(script),
        *_public_argument_vector(arguments),
    )


def _require_gpu_memory_headroom(
    torch_module: object,
    *,
    static_memory_fraction: float,
) -> None:
    cuda_module = getattr(torch_module, "cuda", None)
    memory_info = getattr(cuda_module, "mem_get_info", None)
    if not callable(memory_info):
        raise Glm47ModelValidationError("Torch CUDA memory information is unavailable")
    observed = cast(Callable[[], object], memory_info)()
    if not isinstance(observed, tuple):
        raise Glm47ModelValidationError("Torch CUDA memory information is invalid")
    observed_values = cast(tuple[object, ...], observed)
    if len(observed_values) != 2:
        raise Glm47ModelValidationError("Torch CUDA memory information is invalid")
    free_value = observed_values[0]
    total_value = observed_values[1]
    if (
        type(free_value) is not int
        or free_value <= 0
        or type(total_value) is not int
        or total_value <= 0
    ):
        raise Glm47ModelValidationError("Torch CUDA memory information is invalid")
    free_bytes = free_value
    total_bytes = total_value
    required_bytes = math.ceil(total_bytes * static_memory_fraction)
    try:
        calculate_memory_headroom(
            available_bytes=free_bytes,
            required_bytes=required_bytes,
            minimum_headroom_bytes=MINIMUM_GPU_MEMORY_HEADROOM_BYTES,
        )
    except Glm47LiveValidationError as error:
        raise Glm47ModelValidationError(
            "GPU memory headroom is insufficient for the pinned static allocation"
        ) from error


def available_numa_memory_bytes(
    memory_nodes: tuple[int, ...],
    sysfs_node_root: Path = Path("/sys/devices/system/node"),
) -> int:
    """Return conservative free bytes on the exact nodes used for model weights."""

    if not memory_nodes or memory_nodes != tuple(sorted(set(memory_nodes))):
        raise ValueError("memory nodes must be nonempty, sorted, and unique")
    total_kibibytes = 0
    for memory_node in memory_nodes:
        path = sysfs_node_root / f"node{memory_node}" / "meminfo"
        try:
            lines = path.read_text().splitlines()
        except OSError as error:
            raise Glm47ModelValidationError(
                f"cannot read NUMA memory information: {path}"
            ) from error
        prefix = f"Node {memory_node} MemFree:"
        matches = tuple(
            line.removeprefix(prefix).strip()
            for line in lines
            if line.startswith(prefix)
        )
        if len(matches) != 1:
            raise Glm47ModelValidationError(
                f"NUMA memory information has no exact MemFree value: {path}"
            )
        amount, separator, unit = matches[0].partition(" ")
        if (
            not separator
            or not amount.isascii()
            or not amount.isdecimal()
            or unit.strip() != "kB"
        ):
            raise Glm47ModelValidationError(f"NUMA MemFree value is invalid: {path}")
        total_kibibytes += int(amount)
    if total_kibibytes <= 0:
        raise Glm47ModelValidationError("selected NUMA nodes have no free memory")
    return total_kibibytes * 1024


def _run_bound_backend(
    preflight: Glm47ModelSoftwarePreflight,
) -> Glm47BackendEvidence:
    """Import and execute the heavyweight backend only after binding checks."""

    from scripts.sglang_kt_glm47_backend import (
        Glm47BackendError,
        load_glm47_runtime_bindings,
        run_glm47_backend,
    )

    try:
        runtime = load_glm47_runtime_bindings()
        _require_gpu_memory_headroom(
            runtime.torch,
            static_memory_fraction=(
                preflight.process.process_spec.plan.static_memory_fraction
            ),
        )
        return run_glm47_backend(
            preflight.process.process_spec,
            preflight.route,
            runtime_loader=lambda: runtime,
        )
    except Glm47BackendError as error:
        raise Glm47ModelValidationError(
            f"live GLM-4.7 backend failed: {error}"
        ) from error


def perform_internal_live_validation(
    arguments: _CliArguments,
    *,
    generated_at_utc: datetime | None = None,
) -> None:
    """Run the exact live probe and write evidence only to the inherited memfd."""

    descriptor = arguments.internal_evidence_fd
    if descriptor is None:
        raise Glm47ModelValidationError(
            "internal live validation requires an evidence descriptor"
        )
    preflight = perform_execution_preflight(arguments)
    process_spec = preflight.process.process_spec
    kernel_runtime = load_bound_kernel_runtime_receipt(
        process_spec,
        preflight.kernel_runtime_receipt.path,
        expected_receipt_sha256=preflight.kernel_runtime_receipt.sha256,
    )
    require_current_process_binding(process_spec)
    try:
        calculate_memory_headroom(
            available_bytes=available_host_memory_bytes(),
            required_bytes=preflight.model_snapshot.physical_weight_bytes,
            minimum_headroom_bytes=MINIMUM_HOST_MEMORY_HEADROOM_BYTES,
        )
        calculate_memory_headroom(
            available_bytes=available_numa_memory_bytes(process_spec.memory_nodes),
            required_bytes=preflight.model_snapshot.physical_weight_bytes,
            minimum_headroom_bytes=MINIMUM_HOST_MEMORY_HEADROOM_BYTES,
        )
    except Glm47LiveValidationError as error:
        raise Glm47ModelValidationError(
            "host or selected-NUMA memory headroom is insufficient for the pinned model"
        ) from error

    backend = _run_bound_backend(preflight)
    payload = build_glm47_model_runtime_validation_receipt_payload(
        generated_at_utc=generated_at_utc or datetime.now(UTC),
        preflight=preflight,
        kernel_runtime=kernel_runtime,
        backend=backend,
    )
    write_disposable_child_evidence(descriptor, payload)


def run_parent_live_validation(
    arguments: _CliArguments,
) -> dict[str, object]:
    """Run a disposable child and publish its receipt after independent checks."""

    if arguments.internal_evidence_fd is not None:
        raise Glm47ModelValidationError(
            "parent validation cannot accept an internal evidence descriptor"
        )
    require_disposable_child_evidence_transport()
    preflight = perform_execution_preflight(arguments)
    process_spec = preflight.process.process_spec
    kernel_runtime = load_bound_kernel_runtime_receipt(
        process_spec,
        preflight.kernel_runtime_receipt.path,
        expected_receipt_sha256=preflight.kernel_runtime_receipt.sha256,
    )
    child = run_disposable_live_child(
        build_disposable_child_command(arguments, preflight),
        evidence_descriptor_argument=_INTERNAL_EVIDENCE_DESCRIPTOR_ARGUMENT,
        environment=build_live_child_environment(process_spec),
    )
    try:
        payload = parse_sglang_kt_strict_json(child.contents)
        canonical_contents = canonicalize_sglang_kt_model_runtime_validation_receipt(
            payload
        )
    except (
        RecursionError,
        SglangKtReceiptFileError,
        SglangKtModelRuntimeValidationReceiptError,
    ) as error:
        raise Glm47ModelValidationError(
            "disposable child returned invalid model execution evidence"
        ) from error
    if canonical_contents != child.contents:
        raise Glm47ModelValidationError(
            "disposable child evidence is not canonical JSON"
        )

    # Rebind every file after the child exits. This closes the interval between
    # the child's own preflight and publication without reloading the model.
    preflight = perform_execution_preflight(arguments)
    kernel_runtime = load_bound_kernel_runtime_receipt(
        preflight.process.process_spec,
        preflight.kernel_runtime_receipt.path,
        expected_receipt_sha256=preflight.kernel_runtime_receipt.sha256,
    )
    require_glm47_model_runtime_validation_receipt_parent_bindings(
        payload,
        preflight=preflight,
        kernel_runtime=kernel_runtime,
    )
    receipt_sha256 = publish_new_canonical_receipt(preflight.output, payload)
    if receipt_sha256 != child.sha256:
        raise Glm47ModelValidationError(
            "published model receipt differs from disposable child evidence"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "profiler": PROFILER,
        "output": str(preflight.output),
        "receipt_sha256": receipt_sha256,
        "validator_sha256": preflight.validator.sha256,
    }


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise Glm47ModelValidationError(
            f"output parent must be normalized and absolute: {path}"
        )
    descriptor = os.open("/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise Glm47ModelValidationError(
            f"cannot open output parent without following symlinks: {path}"
        ) from error


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise Glm47ModelValidationError("receipt output parent is not a directory")
    return observed.st_dev, observed.st_ino, observed.st_mode


def _require_current_directory_binding(
    path: Path, identity: tuple[int, int, int]
) -> None:
    current_descriptor = _open_absolute_directory(path)
    try:
        if _directory_identity(current_descriptor) != identity:
            raise Glm47ModelValidationError(
                f"receipt output parent was replaced while publishing: {path}"
            )
    finally:
        os.close(current_descriptor)


def publish_new_canonical_receipt(path: Path, payload: object) -> str:
    """Atomically publish canonical JSON at a new normalized absolute path."""

    try:
        normalized_path = _absolute_normalized_path(str(path))
        contents = canonical_sglang_kt_json(payload)
    except (argparse.ArgumentTypeError, SglangKtReceiptFileError) as error:
        raise Glm47ModelValidationError("cannot canonicalize receipt output") from error
    if len(contents) > _PUBLISHED_RECEIPT_MAXIMUM_BYTES:
        raise Glm47ModelValidationError("receipt exceeds the maximum output size")

    directory_descriptor = _open_absolute_directory(normalized_path.parent)
    directory_identity = _directory_identity(directory_descriptor)
    temporary_name = f".{normalized_path.name}.{os.getpid()}.tmp"
    descriptor = -1
    final_linked = False
    publication_complete = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Glm47ModelValidationError("short receipt write")
            view = view[written:]
        os.fsync(descriptor)
        written_stat = os.fstat(descriptor)
        if not stat.S_ISREG(written_stat.st_mode) or written_stat.st_nlink != 1:
            raise Glm47ModelValidationError(
                "temporary receipt is not a singly linked regular file"
            )
        os.close(descriptor)
        descriptor = -1

        _require_current_directory_binding(normalized_path.parent, directory_identity)
        os.link(
            temporary_name,
            normalized_path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final_linked = True
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        _require_current_directory_binding(normalized_path.parent, directory_identity)
        publication_complete = True
    except FileExistsError as error:
        raise Glm47ModelValidationError(
            f"refusing to replace existing receipt: {normalized_path}"
        ) from error
    except OSError as error:
        raise Glm47ModelValidationError(
            f"cannot publish receipt: {normalized_path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if final_linked and not publication_complete:
            with suppress(FileNotFoundError):
                os.unlink(normalized_path.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
        if not final_linked:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)
    return hashlib.sha256(contents).hexdigest()


def software_preflight_payload(
    preflight: Glm47ModelSoftwarePreflight,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "ready_for_runtime_probe",
        "profiler": PROFILER,
        "runtime_probe_executed": False,
        "validator": {
            "bundle_sha256": preflight.validator.sha256,
            "sources": [
                {
                    "path": source.path,
                    "size_bytes": source.size_bytes,
                    "sha256": source.sha256,
                }
                for source in preflight.validator.sources
            ],
        },
        "process_spec": {
            "path": str(preflight.process.path),
            "receipt_sha256": preflight.process.receipt_sha256,
            "process_spec_sha256": preflight.process.process_spec_sha256,
        },
        "model_contract": {
            "path": preflight.model_contract.path,
            "receipt_sha256": preflight.model_contract.receipt_sha256,
            "contract_sha256": preflight.model_contract.contract_sha256,
        },
        "kernel_runtime_receipt": {
            "path": str(preflight.kernel_runtime_receipt.path),
            "receipt_sha256": preflight.kernel_runtime_receipt.sha256,
        },
        "output": str(preflight.output),
        "resident_expert_route": {
            "resident_gpu_expert_ids": list(preflight.route.resident_gpu_expert_ids),
            "selected_expert_ids": list(preflight.route.selected_expert_ids),
            "cpu_expert_ids": list(preflight.route.cpu_expert_ids),
            "gpu_expert_ids": list(preflight.route.gpu_expert_ids),
        },
    }


def main(arguments: list[str] | None = None) -> int:
    try:
        parsed = parse_arguments(arguments)
        if parsed.internal_evidence_fd is not None:
            perform_internal_live_validation(parsed)
            return 0
        result = canonical_sglang_kt_json(run_parent_live_validation(parsed))
    except (
        Glm47LiveValidationError,
        Glm47ModelValidationError,
        Glm47ReceiptAssemblyError,
        OSError,
    ) as error:
        print(f"GLM-4.7 model validation failed: {error}", file=sys.stderr)
        return 1
    print(result.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
