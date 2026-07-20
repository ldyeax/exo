#!/usr/bin/env python3
"""Run and finalize one leased local GLM-4.7 warm-serving benchmark.

The lease child writes measurement evidence only.  ``execute`` waits for the
transient systemd unit to be collected, immediately reacquires the benchmark
lock, proves host cleanup, and only then publishes a performance receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import IO, Literal, TypeAlias, cast, final

import httpx
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.shared.types.common import ModelId, NodeId  # noqa: E402
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
    GLM_4_7_FLASH_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import (  # noqa: E402
    verify_sglang_kt_model_snapshot,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (  # noqa: E402
    SglangKtModelRuntimeValidationReceiptObservation,
    load_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.process_supervisor import (  # noqa: E402
    build_sglang_kt_process_environment,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (  # noqa: E402
    SglangKtKernelRuntimeValidationReceiptObservation,
    load_sglang_kt_kernel_runtime_validation_receipt,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (  # noqa: E402
    SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
    SglangKtServingAdmissionBinding,
    SglangKtServingCleanupEvidence,
    SglangKtServingClientIdentity,
    SglangKtServingCoordinationGuardEvidence,
    SglangKtServingFileIdentity,
    SglangKtServingInvocationEvidence,
    SglangKtServingJitCacheEvidence,
    SglangKtServingModelFilesystemEvidence,
    SglangKtServingModelIdentity,
    SglangKtServingOwnedServerProcessIdentity,
    SglangKtServingProcessSpecIdentity,
    SglangKtServingRuntimeIdentity,
    SglangKtServingSanityEvidence,
    SglangKtServingServerInfoIdentity,
    SglangKtServingSetupEvidence,
    SglangKtServingSourceFileIdentity,
    SglangKtServingSourceIdentity,
    SglangKtServingTopologyIdentity,
    SglangKtServingTopologyStage,
    SglangKtServingTuningIdentity,
    SglangKtServingWorkloadEvidence,
    SglangKtWarmServingRunIdentity,
    WarmServingRunReceiptV2,
    calculate_sglang_kt_serving_coordination_guard_evidence_sha256,
    calculate_sglang_kt_serving_source_bundle_sha256,
    calculate_sglang_kt_warm_serving_run_identity_sha256,
    canonicalize_sglang_kt_warm_serving_run_receipt,
)
from scripts import benchmark_host_guard as host_guard  # noqa: E402
from scripts import run_sglang_kt_glm47_validation as validation  # noqa: E402
from scripts.sglang_kt_glm47_serving_client import (  # noqa: E402
    EndpointCallObservation,
    Glm47NativeServingClient,
    Glm47ServingClientError,
    build_glm47_server_info_identity,
    prepare_glm47_serving_workload,
    run_glm47_serving_invocation,
    run_glm47_serving_sanity,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

SERVING_HARNESS_RELATIVE_PATH = "scripts/run_sglang_kt_glm47_serving_benchmark.py"
MEASUREMENT_FILENAME = "warm-serving-measurement.json"
PERFORMANCE_RECEIPT_FILENAME = "warm-serving-performance-receipt.json"
PROCESS_SPEC_FILENAME = "serving-process-spec.json"
SERVER_STDOUT_FILENAME = "serving-server.stdout.log"
SERVER_STDERR_FILENAME = "serving-server.stderr.log"
MAXIMUM_MEASUREMENT_BYTES = 4 * 1024 * 1024
MAXIMUM_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAXIMUM_TOOL_BYTES = 64 * 1024 * 1024
FIXED_CHILD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CACHE_DIRECTORY_NAMES = (
    "cuda",
    "huggingface",
    "torch-inductor",
    "torch-extensions",
    "triton",
    "xdg",
)
SERVING_SOURCE_SCRIPT_PATHS = (
    "scripts/benchmark_host_guard.py",
    "scripts/benchmark_lease.py",
    "scripts/create_sglang_kt_glm47_validation_process_spec.py",
    "scripts/run_sglang_kt_glm47_validation.py",
    SERVING_HARNESS_RELATIVE_PATH,
    SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
)
_SHA256 = validation.SHA256_PATTERN


class Glm47ServingHarnessError(RuntimeError):
    """Raised when serving evidence cannot be produced unambiguously."""


@final
class ServingAdmissionConfig(validation.StrictModel):
    validator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_runtime_validation_receipt: validation.ArtifactBinding
    kernel_runtime_validation_receipt: validation.ArtifactBinding
    model_contract_receipt: validation.ArtifactBinding


@final
class ServingToolConfig(validation.StrictModel):
    numactl: validation.ArtifactBinding
    nvidia_smi: validation.ArtifactBinding
    systemctl: validation.ArtifactBinding

    @model_validator(mode="after")
    def validate_distinct_tools(self) -> "ServingToolConfig":
        paths = (self.numactl.path, self.nvidia_smi.path, self.systemctl.path)
        if len(set(paths)) != 3:
            raise ValueError("serving helper executable paths must be distinct")
        return self


@final
class ServingLeaseExecutionConfig(validation.StrictModel):
    owner: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    expected_duration_seconds: float = Field(gt=0.0)
    cleanup_grace_seconds: float = Field(gt=0.0)
    heartbeat_seconds: float = Field(gt=0.0)
    metadata_output: str
    lease_path: str
    lock_path: str
    result_root: str

    @field_validator(
        "expected_duration_seconds", "cleanup_grace_seconds", "heartbeat_seconds"
    )
    @classmethod
    def validate_finite_seconds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("lease timing values must be finite")
        return value

    @field_validator("metadata_output", "lease_path", "lock_path", "result_root")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validation.lexical_absolute_path(value, "serving lease path")

    @model_validator(mode="after")
    def validate_text_and_paths(self) -> "ServingLeaseExecutionConfig":
        if (
            not self.owner.strip()
            or not self.purpose.strip()
            or "\0" in self.owner + self.purpose
        ):
            raise ValueError("serving lease owner and purpose are invalid")
        paths = (
            self.metadata_output,
            self.lease_path,
            self.lock_path,
            self.result_root,
        )
        if len(set(paths)) != len(paths):
            raise ValueError("serving lease paths must be distinct")
        return self


@final
class ServingLocalModelFilesystemConfig(validation.StrictModel):
    mount_point: str
    mount_source: str
    filesystem_type: Literal["xfs", "ext4", "btrfs"]
    device_major: int = Field(ge=0)
    device_minor: int = Field(ge=0)

    @field_validator("mount_source")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validation.lexical_absolute_path(value, "model filesystem path")

    @field_validator("mount_point")
    @classmethod
    def validate_mount_point(cls, value: str) -> str:
        if value == "/":
            return value
        return validation.lexical_absolute_path(value, "model filesystem path")

    @model_validator(mode="after")
    def validate_block_source(self) -> "ServingLocalModelFilesystemConfig":
        if not self.mount_source.startswith("/dev/"):
            raise ValueError("model filesystem must bind a local block device")
        return self


@final
class ServingCoordinationGuardConfig(validation.StrictModel):
    idle_peer: host_guard.HostGuardConfig
    local_hca: host_guard.HcaBinding
    model_filesystem: ServingLocalModelFilesystemConfig

    @model_validator(mode="after")
    def require_two_rail_fabric(self) -> "ServingCoordinationGuardConfig":
        if self.idle_peer.cross_host_fabric is None:
            raise ValueError("serving coordination guard requires a two-rail fabric")
        return self


@final
class ServingBenchmarkConfig(validation.ValidationConfig):
    admission: ServingAdmissionConfig
    tools: ServingToolConfig
    lease_execution: ServingLeaseExecutionConfig
    coordination_guard: ServingCoordinationGuardConfig
    request_timeout_seconds: float = Field(gt=0.0)
    readiness_timeout_seconds: float = Field(gt=0.0)

    @field_validator("request_timeout_seconds", "readiness_timeout_seconds")
    @classmethod
    def validate_finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("serving timeout must be finite")
        return value

    @model_validator(mode="after")
    def validate_serving_contract(self) -> "ServingBenchmarkConfig":
        if (
            self.phase != "serving_baseline"
            or self.profiler != "none"
            or self.resident_gpu_experts < 1
        ):
            raise ValueError("warm serving requires an unprofiled resident-GPU run")
        admission_paths = {
            self.admission.model_runtime_validation_receipt.path,
            self.admission.kernel_runtime_validation_receipt.path,
            self.admission.model_contract_receipt.path,
        }
        if len(admission_paths) != 3 or self.build_receipt.path in admission_paths:
            raise ValueError("serving admission artifacts must use distinct paths")
        if self.tools.numactl.path != self.numactl_executable:
            raise ValueError("bound numactl path differs from launch configuration")
        execution = self.lease_execution
        if Path(self.result_directory).parent != Path(
            execution.result_root
        ) or execution.cleanup_grace_seconds < validation.minimum_cleanup_grace_seconds(
            self
        ):
            raise ValueError("serving lease execution contract is inconsistent")
        filesystem = self.coordination_guard.model_filesystem
        model_path = Path(self.model_path)
        mount_point = Path(filesystem.mount_point)
        if mount_point != model_path and mount_point not in model_path.parents:
            raise ValueError("serving model path is outside its bound block mount")
        peer = self.coordination_guard.idle_peer
        fabric = peer.cross_host_fabric
        assert fabric is not None
        if peer.peer.reserved_ports != self.reserved_ports:
            raise ValueError("idle peer reserved ports differ from the serving run")
        local_hca = self.coordination_guard.local_hca
        configured_hca = {
            (binding.device, binding.port, binding.gid)
            for binding in self.host.hca_bindings
        }
        guarded_hca = {
            (local_hca.device, port.port, port.gid) for port in local_hca.ports
        }
        if configured_hca != guarded_hca:
            raise ValueError("local two-port HCA differs from the host binding")
        local_ports = {port.port: port for port in local_hca.ports}
        if any(
            (rail.local_gid, rail.rate)
            != (
                local_ports[rail.local_port].gid,
                local_ports[rail.local_port].expected_rate,
            )
            for rail in fabric.rails
        ):
            raise ValueError("local HCA differs from the cross-host fabric binding")
        if any(rail.subnet_manager_host != "remote" for rail in fabric.rails):
            raise ValueError(
                "serving local baseline requires peer-owned subnet managers"
            )
        return self


@final
class FilesystemObjectIdentity(validation.StrictModel):
    path: str
    kind: Literal["directory", "regular_file"]
    device: int = Field(ge=0)
    inode: int = Field(gt=0)
    owner_uid: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validation.lexical_absolute_path(value, "handoff path")


@final
class ServingHandoffIdentity(validation.StrictModel):
    result_directory: FilesystemObjectIdentity
    coordination_lock: FilesystemObjectIdentity

    @model_validator(mode="after")
    def validate_kinds(self) -> "ServingHandoffIdentity":
        if (
            self.result_directory.kind != "directory"
            or self.coordination_lock.kind != "regular_file"
            or self.result_directory.path == self.coordination_lock.path
        ):
            raise ValueError("serving filesystem handoff identities are invalid")
        return self


@final
class WarmServingMeasurementV2(validation.StrictModel):
    schema_version: Literal[2]
    status: Literal["passed"]
    generated_at_utc: str
    profiler: Literal["none"]
    instrumentation: Literal["none"]
    radix_cache_disabled: Literal[True]
    max_concurrent_requests: Literal[1]
    lease_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity: SglangKtWarmServingRunIdentity
    setup: SglangKtServingSetupEvidence
    sanity: SglangKtServingSanityEvidence
    jit_cache: SglangKtServingJitCacheEvidence
    workloads: tuple[SglangKtServingWorkloadEvidence, ...]
    coordination_guard: SglangKtServingCoordinationGuardEvidence
    owner_token: str = Field(min_length=1)
    server_process: SglangKtServingOwnedServerProcessIdentity
    server_return_code: Literal[-15, 0]
    termination_signal: Literal["SIGTERM"]
    forced: Literal[False]
    owned_processes_absent: Literal[True]
    delegated_cgroup_path: str
    delegated_cgroup_removed: Literal[True]
    transient_unit_name: str
    handoff: ServingHandoffIdentity

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("generated_at_utc")
    @classmethod
    def validate_generated_at_utc(cls, value: str) -> str:
        try:
            observed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("measurement timestamp is not ISO-8601") from error
        if observed.tzinfo is None or observed.utcoffset() != timedelta(0):
            raise ValueError("measurement timestamp must be UTC")
        return value

    @model_validator(mode="after")
    def validate_measurement(self) -> "WarmServingMeasurementV2":
        if self.identity_sha256 != calculate_sglang_kt_warm_serving_run_identity_sha256(
            self.identity
        ):
            raise ValueError("measurement identity SHA-256 does not match")
        if tuple(item.request.kind for item in self.workloads) != (
            "prefill",
            "decode",
        ):
            raise ValueError("measurement workloads are not canonical")
        if self.server_process.executable != self.identity.runtime.executable:
            raise ValueError("measurement server executable does not match identity")
        if (
            self.server_process.argv_sha256
            != self.identity.process_spec.launch_argv_sha256
        ):
            raise ValueError("measurement server argv does not match identity")
        if (
            self.server_process.cpu_affinity != self.identity.process_spec.cpu_cores
            or self.server_process.memory_nodes
            != self.identity.process_spec.memory_nodes
        ):
            raise ValueError("measurement server placement does not match identity")
        if not PurePosixPath(self.delegated_cgroup_path).is_absolute():
            raise ValueError("measurement cgroup path must be absolute")
        if not self.transient_unit_name:
            raise ValueError("measurement transient unit name must be nonempty")
        return self


@dataclass(frozen=True)
class ServingLeasePreparation:
    metadata: JsonObject
    metadata_output: str
    deployment: validation.DeploymentIdentity
    child_argv: tuple[str, ...]
    systemd_argv: tuple[str, ...]
    execute_argv: tuple[str, ...]
    systemd_unit_name: str


@dataclass
class PreservedServingHandoff:
    lock_file: IO[bytes]
    results: validation.ResultDirectory
    identity: ServingHandoffIdentity

    def close(self) -> None:
        self.results.close()
        self.lock_file.close()


@dataclass(frozen=True)
class AdmissionEvidence:
    process_spec: SglangKtProcessLaunchSpec
    process_spec_file: SglangKtServingFileIdentity
    model_receipt: SglangKtModelRuntimeValidationReceiptObservation
    kernel_receipt: SglangKtKernelRuntimeValidationReceiptObservation
    admission_seconds: float


@dataclass(frozen=True)
class ServerProcess:
    process: subprocess.Popen[bytes]
    owned: validation.OwnedProcess
    command: tuple[str, ...]
    environment: dict[str, str]
    working_directory: str
    launch_argv_sha256: str
    launch_seconds: float


@dataclass(frozen=True)
class CompletedMeasurement:
    identity: SglangKtWarmServingRunIdentity
    setup: SglangKtServingSetupEvidence
    sanity: SglangKtServingSanityEvidence
    jit_cache: SglangKtServingJitCacheEvidence
    workloads: tuple[SglangKtServingWorkloadEvidence, ...]
    server: ServerProcess
    server_identity: SglangKtServingOwnedServerProcessIdentity
    server_return_code: Literal[-15, 0]


@dataclass
class RetainedExecutable:
    descriptor: int
    identity: SglangKtServingFileIdentity

    @property
    def execution_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(frozen=True)
class BoundMeasurementSnapshot:
    measurement: WarmServingMeasurementV2
    contents: bytes
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class CoordinationPreflight:
    remote: host_guard.HostGuardSnapshot
    local_hca: host_guard.HcaObservation
    model_filesystem: SglangKtServingModelFilesystemEvidence


@dataclass(frozen=True)
class LocalFabricObservation:
    hca: host_guard.HcaObservation
    opensm_units: tuple[host_guard.OpenSmUnitObservation, ...] = ()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_sglang_kt_json(value)).hexdigest()


def _file_identity(path: Path, *, maximum_bytes: int) -> SglangKtServingFileIdentity:
    bound = read_sglang_kt_bound_file(path, maximum_bytes=maximum_bytes)
    return SglangKtServingFileIdentity(
        path=str(bound.path),
        size_bytes=len(bound.contents),
        sha256=bound.sha256,
    )


def open_bound_executable(
    binding: validation.ArtifactBinding,
    description: str,
) -> RetainedExecutable:
    path = Path(binding.path)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or before.st_nlink != 1
            or current.st_nlink != 1
            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
            or before.st_size <= 0
            or before.st_size > MAXIMUM_TOOL_BYTES
            or before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) == 0
        ):
            raise Glm47ServingHarnessError(
                f"bound {description} executable identity changed"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Glm47ServingHarnessError(
                    f"bound {description} executable was truncated"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise Glm47ServingHarnessError(
            f"cannot inspect bound {description} executable"
        ) from error
    except Glm47ServingHarnessError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    if digest.hexdigest() != binding.sha256 or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        if descriptor is not None:
            os.close(descriptor)
        raise Glm47ServingHarnessError(
            f"bound {description} executable identity changed"
        )
    assert descriptor is not None
    return RetainedExecutable(
        descriptor=descriptor,
        identity=SglangKtServingFileIdentity(
            path=str(path),
            size_bytes=before.st_size,
            sha256=binding.sha256,
        ),
    )


def bound_executable_identity(
    binding: validation.ArtifactBinding,
    description: str,
) -> SglangKtServingFileIdentity:
    retained = open_bound_executable(binding, description)
    try:
        return retained.identity
    finally:
        retained.close()


def run_bound_tool_text(
    binding: validation.ArtifactBinding,
    arguments: Sequence[str],
    description: str,
    *,
    timeout_seconds: float = 15.0,
) -> str:
    retained = open_bound_executable(binding, description)
    try:
        observed = subprocess.run(
            (binding.path, *arguments),
            executable=retained.execution_path,
            pass_fds=(retained.descriptor,),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": FIXED_CHILD_PATH},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Glm47ServingHarnessError(f"cannot execute bound {description}") from error
    finally:
        retained.close()
    if observed.returncode != 0:
        raise Glm47ServingHarnessError(f"bound {description} command failed")
    return observed.stdout


def filesystem_object_identity(
    path: Path,
    descriptor: int,
    kind: Literal["directory", "regular_file"],
) -> FilesystemObjectIdentity:
    retained = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    expected_mode = stat.S_IFDIR if kind == "directory" else stat.S_IFREG
    if (
        stat.S_IFMT(retained.st_mode) != expected_mode
        or stat.S_IFMT(current.st_mode) != expected_mode
        or (retained.st_dev, retained.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise Glm47ServingHarnessError(f"{kind} handoff identity changed: {path}")
    return FilesystemObjectIdentity(
        path=str(path),
        kind=kind,
        device=retained.st_dev,
        inode=retained.st_ino,
        owner_uid=retained.st_uid,
    )


def _load_serving_config(path: Path) -> ServingBenchmarkConfig:
    if not path.is_absolute():
        raise Glm47ServingHarnessError("serving config path must be absolute")
    try:
        bound = read_sglang_kt_bound_file(
            path, maximum_bytes=validation.MAXIMUM_JSON_BYTES
        )
        parse_sglang_kt_strict_json(bound.contents)
        return ServingBenchmarkConfig.model_validate_json(bound.contents)
    except (OSError, ValueError, ValidationError) as error:
        raise Glm47ServingHarnessError(
            f"invalid GLM-4.7 serving config: {error}"
        ) from error


def _resource_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _decode_mountinfo_path(value: str, *, require_absolute: bool = True) -> str:
    decoded = value
    for escaped, replacement in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        decoded = decoded.replace(escaped, replacement)
    if "\\" in decoded or "\0" in decoded:
        raise Glm47ServingHarnessError("mountinfo contains an unsafe path escape")
    if not require_absolute:
        if not decoded:
            raise Glm47ServingHarnessError("mountinfo contains an empty source")
        return decoded
    if decoded == "/":
        return decoded
    return validation.lexical_absolute_path(decoded, "observed mount path")


def observe_local_model_filesystem(
    config: ServingBenchmarkConfig,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    stat_path: Callable[[Path], os.stat_result] = os.stat,
) -> SglangKtServingModelFilesystemEvidence:
    try:
        lines = mountinfo_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise Glm47ServingHarnessError("cannot read local mount provenance") from error
    model_path = Path(config.model_path)
    candidates: list[tuple[int, int, str, str, str, int, int]] = []
    for line in lines:
        left, separator, right = line.partition(" - ")
        left_fields = left.split()
        right_fields = right.split()
        if separator != " - " or len(left_fields) < 6 or len(right_fields) < 3:
            raise Glm47ServingHarnessError("mountinfo contains a malformed record")
        mount_point = Path(_decode_mountinfo_path(left_fields[4]))
        if mount_point != model_path and mount_point not in model_path.parents:
            continue
        device_fields = left_fields[2].split(":", maxsplit=1)
        try:
            mount_id = int(left_fields[0])
            major, minor = (int(value) for value in device_fields)
        except (TypeError, ValueError) as error:
            raise Glm47ServingHarnessError(
                "mountinfo device identity is invalid"
            ) from error
        if mount_id <= 0 or major < 0 or minor < 0:
            raise Glm47ServingHarnessError("mountinfo device identity is invalid")
        candidates.append(
            (
                len(mount_point.parts),
                mount_id,
                str(mount_point),
                _decode_mountinfo_path(right_fields[1], require_absolute=False),
                right_fields[0],
                major,
                minor,
            )
        )
    if not candidates:
        raise Glm47ServingHarnessError("model path has no observed mount provenance")
    (
        _,
        _,
        mount_point,
        mount_source,
        filesystem_type,
        major,
        minor,
    ) = max(candidates, key=lambda item: (item[0], item[1]))
    expected = config.coordination_guard.model_filesystem
    try:
        model_stat = stat_path(model_path)
    except OSError as error:
        raise Glm47ServingHarnessError(
            "cannot stat the configured model path"
        ) from error
    if (
        (mount_point, mount_source, filesystem_type, major, minor)
        != (
            expected.mount_point,
            expected.mount_source,
            expected.filesystem_type,
            expected.device_major,
            expected.device_minor,
        )
        or not mount_source.startswith("/dev/")
        or filesystem_type not in {"xfs", "ext4", "btrfs"}
        or (os.major(model_stat.st_dev), os.minor(model_stat.st_dev)) != (major, minor)
    ):
        raise Glm47ServingHarnessError(
            "model path is not on its config-bound local block filesystem"
        )
    return SglangKtServingModelFilesystemEvidence(
        model_path=str(model_path),
        mount_point=mount_point,
        mount_source=mount_source,
        filesystem_type=cast(Literal["xfs", "ext4", "btrfs"], filesystem_type),
        device_major=major,
        device_minor=minor,
        local_block_filesystem=True,
    )


def _validate_cross_host_fabric(
    config: ServingBenchmarkConfig,
    local_hca: host_guard.HcaObservation,
    remote: host_guard.HostGuardSnapshot,
) -> None:
    fabric = config.coordination_guard.idle_peer.cross_host_fabric
    assert fabric is not None
    local = cast(
        host_guard.HostObservation,
        cast(object, LocalFabricObservation(hca=local_hca)),
    )
    host_guard.validate_cross_host_fabric(local, remote, fabric)


def _validate_local_hca_observation(
    observation: host_guard.HcaObservation,
    binding: host_guard.HcaBinding,
) -> None:
    if (
        observation.device != binding.device
        or observation.node_guid != binding.node_guid
    ):
        raise Glm47ServingHarnessError("local HCA identity differs from its binding")
    for observed, expected in zip(observation.ports, binding.ports, strict=True):
        if (
            observed.port,
            observed.gid,
            observed.rate,
            observed.counter_device,
            observed.counter_port,
        ) != (
            expected.port,
            expected.gid,
            expected.expected_rate,
            binding.device,
            expected.port,
        ):
            raise Glm47ServingHarnessError(
                "local HCA port or counter source differs from its binding"
            )
        for name, maximum in expected.health_counter_maximums.items():
            value = observed.counters.get(name)
            if value is None or value > maximum:
                raise Glm47ServingHarnessError(
                    f"local HCA health counter exceeds its bound: {name}"
                )


def collect_coordination_preflight(
    config: ServingBenchmarkConfig,
) -> CoordinationPreflight:
    filesystem = observe_local_model_filesystem(config)
    remote = host_guard.collect_remote_snapshot(
        config.coordination_guard.idle_peer, "preflight"
    )
    local_hca = host_guard.collect_hca_observation(config.coordination_guard.local_hca)
    _validate_local_hca_observation(local_hca, config.coordination_guard.local_hca)
    _validate_cross_host_fabric(config, local_hca, remote)
    return CoordinationPreflight(
        remote=remote,
        local_hca=local_hca,
        model_filesystem=filesystem,
    )


def _validate_local_hca_unchanged(
    before: host_guard.HcaObservation,
    after: host_guard.HcaObservation,
    binding: host_guard.HcaBinding,
) -> None:
    if (
        before.device != after.device
        or before.node_guid != after.node_guid
        or len(before.ports) != len(after.ports)
    ):
        raise Glm47ServingHarnessError("local HCA identity changed during serving")
    for before_port, after_port in zip(before.ports, after.ports, strict=True):
        expected_port = binding.ports[before_port.port - 1]
        if (
            before_port.port,
            before_port.gid,
            before_port.state,
            before_port.physical_state,
            before_port.rate,
            before_port.lid,
            before_port.sm_lid,
        ) != (
            after_port.port,
            after_port.gid,
            after_port.state,
            after_port.physical_state,
            after_port.rate,
            after_port.lid,
            after_port.sm_lid,
        ):
            raise Glm47ServingHarnessError(
                "local HCA fabric identity changed during serving"
            )
        for name in host_guard.HEALTH_COUNTER_NAMES:
            first = before_port.counters.get(name)
            final = after_port.counters.get(name)
            if first is None and final is None:
                continue
            if first is None or final is None or final != first:
                raise Glm47ServingHarnessError(
                    f"local HCA health counter changed during serving: {name}"
                )
        for (
            name,
            maximum_delta,
        ) in expected_port.idle_data_counter_maximum_deltas.items():
            first = before_port.counters.get(name)
            final = after_port.counters.get(name)
            if (
                first is None
                or final is None
                or final < first
                or final - first > maximum_delta
            ):
                raise Glm47ServingHarnessError(
                    f"local HCA data counter exceeded idle bound: {name}"
                )


def collect_coordination_postflight(
    config: ServingBenchmarkConfig,
    preflight: CoordinationPreflight,
) -> SglangKtServingCoordinationGuardEvidence:
    postflight = host_guard.collect_remote_snapshot(
        config.coordination_guard.idle_peer, "postflight"
    )
    comparison = host_guard.compare_snapshots(preflight.remote, postflight)
    if not comparison.stable:
        raise Glm47ServingHarnessError(
            "idle peer changed during serving: " + "; ".join(comparison.failures)
        )
    local_hca = host_guard.collect_hca_observation(config.coordination_guard.local_hca)
    _validate_local_hca_unchanged(
        preflight.local_hca,
        local_hca,
        config.coordination_guard.local_hca,
    )
    _validate_cross_host_fabric(config, local_hca, postflight)
    current_filesystem = observe_local_model_filesystem(config)
    if current_filesystem != preflight.model_filesystem:
        raise Glm47ServingHarnessError(
            "model filesystem provenance changed during serving"
        )
    guard_config = config.coordination_guard.idle_peer
    remote_preflight = cast(
        JsonObject, cast(object, preflight.remote.model_dump(mode="json"))
    )
    remote_postflight = cast(
        JsonObject, cast(object, postflight.model_dump(mode="json"))
    )
    comparison_payload = cast(
        JsonObject, cast(object, comparison.model_dump(mode="json"))
    )
    local_preflight = cast(
        JsonObject, cast(object, preflight.local_hca.model_dump(mode="json"))
    )
    local_postflight = cast(JsonObject, cast(object, local_hca.model_dump(mode="json")))
    config_sha256 = host_guard.calculate_host_guard_config_sha256(guard_config)
    peer_sha256 = host_guard.calculate_coordination_peer_binding_sha256(
        guard_config.peer
    )
    evidence_sha256 = calculate_sglang_kt_serving_coordination_guard_evidence_sha256(
        host_guard_config_sha256=config_sha256,
        peer_binding_sha256=peer_sha256,
        remote_preflight=remote_preflight,
        remote_postflight=remote_postflight,
        comparison=comparison_payload,
        local_hca_preflight=local_preflight,
        local_hca_postflight=local_postflight,
        model_filesystem=preflight.model_filesystem,
    )
    return SglangKtServingCoordinationGuardEvidence(
        peer_role="idle_nonparticipant",
        host_guard_config_sha256=config_sha256,
        peer_binding_sha256=peer_sha256,
        remote_preflight=remote_preflight,
        remote_postflight=remote_postflight,
        comparison=comparison_payload,
        local_hca_preflight=local_preflight,
        local_hca_postflight=local_postflight,
        model_filesystem=preflight.model_filesystem,
        remote_peer_unchanged=True,
        cross_host_fabric_validated=True,
        evidence_sha256=evidence_sha256,
    )


def _systemd_unit_name(config: ServingBenchmarkConfig) -> str:
    return validation.systemd_unit_name(config)


def require_admitted_validator(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
) -> None:
    if deployment.validator_sha256 != config.admission.validator_sha256:
        raise Glm47ServingHarnessError(
            "immutable validator differs from the admitted validator"
        )


def _serving_child_argv(
    config: ServingBenchmarkConfig, *, lease_path: Path, lock_path: Path
) -> tuple[str, ...]:
    root = Path(config.source.deployment_root)
    return (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
        config.runtime_python.path,
        str(root / "orchestrator" / SERVING_HARNESS_RELATIVE_PATH),
        "--config",
        str(root / validation.IMMUTABLE_CONFIG_RELATIVE_PATH),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    )


def _benchmark_lease_argv(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
) -> tuple[str, ...]:
    execution = config.lease_execution
    child_argv = _serving_child_argv(
        config,
        lease_path=Path(execution.lease_path),
        lock_path=Path(execution.lock_path),
    )
    benchmark_lease_script = (
        Path(deployment.root) / "orchestrator/scripts/benchmark_lease.py"
    )
    return (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
        config.runtime_python.path,
        str(benchmark_lease_script),
        f"--owner={execution.owner}",
        f"--purpose={execution.purpose}",
        "--run-id",
        config.run_id,
        "--namespace",
        config.namespace,
        "--port",
        ",".join(str(port) for port in config.reserved_ports),
        "--metadata-json",
        execution.metadata_output,
        "--heartbeat-seconds",
        str(execution.heartbeat_seconds),
        "--expected-duration-seconds",
        str(execution.expected_duration_seconds),
        "--cleanup-grace-seconds",
        str(execution.cleanup_grace_seconds),
        "--lock-path",
        execution.lock_path,
        "--lease-path",
        execution.lease_path,
        "--result-root",
        execution.result_root,
        "--",
        *child_argv,
    )


def _prepared_systemd_argv(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
) -> tuple[str, ...]:
    return validation.systemd_benchmark_argv(
        config,
        _benchmark_lease_argv(config, deployment),
        expected_duration_seconds=config.lease_execution.expected_duration_seconds,
        cleanup_grace_seconds=config.lease_execution.cleanup_grace_seconds,
    )


def _validate_prepared_metadata(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    config_path: Path,
) -> JsonObject:
    metadata = validation.read_json_regular(
        Path(config.lease_execution.metadata_output), "prepared serving metadata"
    )
    expected = build_serving_static_metadata(
        config,
        _serving_child_argv(
            config,
            lease_path=Path(config.lease_execution.lease_path),
            lock_path=Path(config.lease_execution.lock_path),
        ),
        validation.config_sha256(config_path),
        deployment,
    )
    generated_at = metadata.get("generated_at")
    if not isinstance(generated_at, str):
        raise Glm47ServingHarnessError("prepared serving metadata lacks generated_at")
    expected["generated_at"] = generated_at
    if metadata != expected:
        raise Glm47ServingHarnessError(
            "prepared serving metadata differs from immutable authorization"
        )
    return metadata


def _source_files(
    deployment: validation.DeploymentIdentity,
) -> tuple[SglangKtServingSourceFileIdentity, ...]:
    root = Path(deployment.root) / "orchestrator"
    paths = [
        path
        for path in (root / "src").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    paths.extend(root / relative for relative in SERVING_SOURCE_SCRIPT_PATHS)
    identities: list[SglangKtServingSourceFileIdentity] = []
    for path in sorted(set(paths)):
        bound = read_sglang_kt_bound_file(path, maximum_bytes=MAXIMUM_SOURCE_FILE_BYTES)
        identities.append(
            SglangKtServingSourceFileIdentity(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=len(bound.contents),
                sha256=bound.sha256,
            )
        )
    return tuple(identities)


def _source_identity(
    deployment: validation.DeploymentIdentity,
) -> SglangKtServingSourceIdentity:
    root = Path(deployment.root) / "orchestrator"
    files = _source_files(deployment)
    dirty_paths = set(deployment.source.dirty_file_hashes)
    dirty_files = tuple(item for item in files if item.relative_path in dirty_paths)
    return SglangKtServingSourceIdentity(
        repository_root=str(root),
        commit=deployment.source.commit,
        source_bundle_sha256=calculate_sglang_kt_serving_source_bundle_sha256(files),
        files=files,
        dirty_files=dirty_files,
    )


def _load_process_spec(
    results: validation.ResultDirectory,
    generator: validation.CommandOutcome,
) -> tuple[SglangKtProcessLaunchSpec, SglangKtServingFileIdentity]:
    generated = validation.parse_command_json(results, generator)
    process_spec_sha256 = generated.get("process_spec_sha256")
    if (
        generated.get("schema_version") != 1
        or generated.get("output") != str(results.path_for(PROCESS_SPEC_FILENAME))
        or not isinstance(process_spec_sha256, str)
        or _SHA256.fullmatch(process_spec_sha256) is None
    ):
        raise Glm47ServingHarnessError("process-spec generator result is not bound")
    bound = read_sglang_kt_bound_file(
        results.path_for(PROCESS_SPEC_FILENAME), maximum_bytes=1024 * 1024
    )
    if generated.get("receipt_sha256") != bound.sha256:
        raise Glm47ServingHarnessError("generated process-spec file hash changed")
    try:
        parse_sglang_kt_strict_json(bound.contents)
        process_spec = SglangKtProcessLaunchSpec.model_validate_json(bound.contents)
    except (ValueError, ValidationError) as error:
        raise Glm47ServingHarnessError("generated process spec is invalid") from error
    if (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
        != process_spec_sha256
    ):
        raise Glm47ServingHarnessError("generated process-spec identity changed")
    return process_spec, SglangKtServingFileIdentity(
        path=str(bound.path),
        size_bytes=len(bound.contents),
        sha256=bound.sha256,
    )


def _validate_admission_cross_bindings(
    config: ServingBenchmarkConfig,
    process_spec: SglangKtProcessLaunchSpec,
    model: SglangKtModelRuntimeValidationReceiptObservation,
    kernel: SglangKtKernelRuntimeValidationReceiptObservation,
) -> None:
    stage = process_spec.stage
    if (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
        != config.admission.process_spec_sha256
        or process_spec.target_profile != GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE
        or process_spec.executable != config.runtime_python.path
        or process_spec.model_path != config.model_path
        or process_spec.node_id != NodeId(config.host.node_id)
        or process_spec.gpu_uuid != config.host.gpu.uuid
        or process_spec.cpu_cores != config.host.cpu_cores
        or process_spec.memory_nodes != config.host.memory_nodes
        or stage.resident_gpu_experts != config.resident_gpu_experts
        or process_spec.service_endpoint.ip != config.service_endpoint.ip
        or process_spec.service_endpoint.port != config.service_endpoint.port
        or process_spec.distributed_coordinator.ip != config.distributed_coordinator.ip
        or process_spec.distributed_coordinator.port
        != config.distributed_coordinator.port
        or "--disable-radix-cache" not in process_spec.arguments
        or "--disable-cuda-graph" not in process_spec.arguments
    ):
        raise Glm47ServingHarnessError(
            "fresh serving process spec differs from the admitted launch"
        )
    if (
        model.process_spec_sha256 != config.admission.process_spec_sha256
        or model.target_profile != GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE
        or model.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
        or model.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or model.model_path != config.model_path
        or model.model_config_sha256 != GLM_4_7_FLASH_BF16_CONFIG_SHA256
        or model.gpu_uuid != config.host.gpu.uuid
        or model.cpu_cores != config.host.cpu_cores
        or model.memory_nodes != config.host.memory_nodes
        or model.resident_gpu_experts != config.resident_gpu_experts
        or model.executed_cpu_backend != "AMX_BF16"
        or model.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or model.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        or model.kernel_runtime_validation_receipt_path
        != config.admission.kernel_runtime_validation_receipt.path
        or model.kernel_runtime_validation_receipt_sha256
        != config.admission.kernel_runtime_validation_receipt.sha256
        or model.model_contract_path != config.admission.model_contract_receipt.path
        or model.model_contract_receipt_sha256
        != config.admission.model_contract_receipt.sha256
        or model.model_contract_sha256 != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    ):
        raise Glm47ServingHarnessError(
            "model admission receipt differs from the serving process spec"
        )
    if (
        kernel.receipt_path != config.admission.kernel_runtime_validation_receipt.path
        or kernel.receipt_sha256
        != config.admission.kernel_runtime_validation_receipt.sha256
        or kernel.executable != config.runtime_python.path
        or kernel.gpu_uuid != config.host.gpu.uuid
        or kernel.cpu_cores != config.host.cpu_cores
        or kernel.memory_nodes != config.host.memory_nodes
        or kernel.build_receipt_path != config.build_receipt.path
        or kernel.build_receipt_sha256 != config.build_receipt.sha256
        or kernel.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or kernel.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        or kernel.sgl_kernel_build_id != model.sgl_kernel_build_id
        or kernel.deep_gemm_build_id != model.deep_gemm_build_id
        or kernel.kt_kernel_build_id != model.kt_kernel_build_id
        or kernel.torch_version != model.torch_version
        or kernel.cuda_version != model.cuda_version
    ):
        raise Glm47ServingHarnessError(
            "kernel admission receipt differs from the serving runtime"
        )


def collect_admission_evidence(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    results: validation.ResultDirectory,
    registry: validation.OwnershipRegistry,
    owned_cgroup: validation.OwnedCgroup,
    latch: validation.SignalLatch,
    environment: Mapping[str, str],
) -> AdmissionEvidence:
    started = time.monotonic()
    generator = validation.run_owned_command(
        config,
        results,
        registry,
        owned_cgroup,
        latch,
        name="serving-process-spec-generator",
        command=validation.build_generator_command(
            config, deployment, results.path_for(PROCESS_SPEC_FILENAME)
        ),
        timeout_seconds=config.timeouts.generator_seconds,
        environment=environment,
    )
    validation.require_success(generator)
    process_spec, process_spec_file = _load_process_spec(results, generator)
    if (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
        != config.admission.process_spec_sha256
    ):
        raise Glm47ServingHarnessError(
            "fresh process spec is not bound to the admitted canonical digest"
        )

    kernel = load_sglang_kt_kernel_runtime_validation_receipt(
        Path(config.admission.kernel_runtime_validation_receipt.path),
        expected_receipt_sha256=(
            config.admission.kernel_runtime_validation_receipt.sha256
        ),
    )
    model = load_sglang_kt_model_runtime_validation_receipt(
        Path(config.admission.model_runtime_validation_receipt.path),
        expected_validator_sha256=config.admission.validator_sha256,
        expected_process_spec_sha256=config.admission.process_spec_sha256,
        expected_model_contract_receipt_sha256=(
            config.admission.model_contract_receipt.sha256
        ),
        expected_kernel_receipt_sha256=(
            config.admission.kernel_runtime_validation_receipt.sha256
        ),
        expected_receipt_sha256=(
            config.admission.model_runtime_validation_receipt.sha256
        ),
    )
    _validate_admission_cross_bindings(config, process_spec, model, kernel)
    verified_model = verify_sglang_kt_model_snapshot(
        Path(config.model_path),
        Path(config.model_contract.path),
        expected_contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        expected_model_id=ModelId(GLM_4_7_FLASH_BF16_MODEL_ID),
        expected_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        expected_ktransformers_method="BF16",
    )
    if (
        verified_model.contract_receipt_sha256 != config.model_contract.sha256
        or verified_model.config_sha256 != model.model_config_sha256
        or verified_model.index_sha256 != model.model_index_sha256
        or verified_model.weight_map_entries != model.model_weight_map_entries
        or verified_model.shard_count != model.model_shard_count
        or verified_model.physical_weight_bytes != model.model_physical_weight_bytes
    ):
        raise Glm47ServingHarnessError(
            "contracted model snapshot differs from admitted model evidence"
        )
    validation.verify_runtime_python(config.runtime_python)
    validation.verify_artifact(config.build_receipt, "runtime build receipt")
    validation.verify_artifact(config.model_contract, "model contract")
    validation.verify_artifact(
        config.admission.model_contract_receipt,
        "admitted model contract receipt",
    )
    validation.verify_artifact(
        config.admission.model_runtime_validation_receipt,
        "model runtime validation receipt",
    )
    validation.verify_artifact(
        config.admission.kernel_runtime_validation_receipt,
        "kernel runtime validation receipt",
    )
    return AdmissionEvidence(
        process_spec=process_spec,
        process_spec_file=process_spec_file,
        model_receipt=model,
        kernel_receipt=kernel,
        admission_seconds=time.monotonic() - started,
    )


def create_cache_directories(
    scratch: validation.OwnedScratchDirectory,
) -> tuple[Path, ...]:
    cache_root = scratch.path / "cache"
    cache_root.mkdir(mode=0o700)
    directories: list[Path] = []
    for name in CACHE_DIRECTORY_NAMES:
        path = cache_root / name
        path.mkdir(mode=0o700)
        directories.append(path)
    home = scratch.path / "home"
    home.mkdir(mode=0o700)
    temporary = scratch.path / "tmp"
    temporary.mkdir(mode=0o700)
    return tuple(sorted((scratch.path, home, temporary, *directories)))


def build_serving_environment(
    config: ServingBenchmarkConfig,
    process_spec: SglangKtProcessLaunchSpec,
    owner_token: str,
    scratch: validation.OwnedScratchDirectory,
    cache_directories: tuple[Path, ...],
    *,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    base = validation.build_child_environment(
        config, owner_token, parent_environment=parent_environment
    )
    base["PATH"] = FIXED_CHILD_PATH
    environment = build_sglang_kt_process_environment(process_spec, base)
    cache_by_name = {path.name: path for path in cache_directories}
    environment.update(
        {
            "CUDA_CACHE_PATH": str(cache_by_name["cuda"]),
            "HF_HOME": str(cache_by_name["huggingface"]),
            "HOME": str(scratch.path / "home"),
            "TEMP": str(scratch.path / "tmp"),
            "TMP": str(scratch.path / "tmp"),
            "TMPDIR": str(scratch.path / "tmp"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_by_name["torch-inductor"]),
            "TORCH_EXTENSIONS_DIR": str(cache_by_name["torch-extensions"]),
            "TRITON_CACHE_DIR": str(cache_by_name["triton"]),
            "XDG_CACHE_HOME": str(cache_by_name["xdg"]),
        }
    )
    validation.require_no_profiler_state(environment, process_spec.command, "")
    if any(name.startswith(("NCCL_", "SGLANG_")) for name in environment):
        raise Glm47ServingHarnessError(
            "untuned local baseline inherited an SGLANG/NCCL variable"
        )
    return environment


def _cache_manifest_once(cache_directories: tuple[Path, ...]) -> str:
    entries: list[dict[str, JsonValue]] = []
    for root in cache_directories:
        root_stat = root.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_ISLNK(root_stat.st_mode)
            or root_stat.st_uid != os.geteuid()
        ):
            raise Glm47ServingHarnessError("owned cache root identity is unsafe")
        entries.append(
            {
                "path": root.name,
                "kind": "directory",
            }
        )
        for path in sorted(root.rglob("*")):
            observed = path.lstat()
            if observed.st_uid != os.geteuid() or stat.S_ISLNK(observed.st_mode):
                raise Glm47ServingHarnessError("owned cache contains a foreign link")
            relative = f"{root.name}/{path.relative_to(root).as_posix()}"
            if stat.S_ISDIR(observed.st_mode):
                entries.append({"path": relative, "kind": "directory"})
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise Glm47ServingHarnessError(
                    "owned cache contains a special or multiply linked file"
                )
            bound = read_sglang_kt_bound_file(path, maximum_bytes=1024 * 1024 * 1024)
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": len(bound.contents),
                    "sha256": bound.sha256,
                }
            )
    return _canonical_sha256(
        {
            "canonicalization": "exo-glm47-owned-jit-cache-manifest-v1",
            "entries": entries,
        }
    )


def stable_cache_manifest(
    cache_directories: tuple[Path, ...],
    *,
    timeout_seconds: float = 5.0,
    quiet_seconds: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    if timeout_seconds <= 0.0 or quiet_seconds <= 0.0:
        raise ValueError("cache stability timings must be positive")
    deadline = time.monotonic() + timeout_seconds
    previous = _cache_manifest_once(cache_directories)
    while time.monotonic() < deadline:
        sleep(quiet_seconds)
        current = _cache_manifest_once(cache_directories)
        if current == previous:
            return current
        previous = current
    raise Glm47ServingHarnessError("owned JIT cache did not become stable")


def _server_command(
    config: ServingBenchmarkConfig,
    process_spec: SglangKtProcessLaunchSpec,
) -> tuple[str, ...]:
    return (
        config.tools.numactl.path,
        "--physcpubind",
        _resource_list(process_spec.cpu_cores),
        "--membind",
        _resource_list(process_spec.memory_nodes),
        *process_spec.command,
    )


def _parse_linux_resource_list(value: str, description: str) -> tuple[int, ...]:
    resources: set[int] = set()
    try:
        for component in value.strip().split(","):
            if not component:
                raise ValueError("empty component")
            bounds = component.split("-", maxsplit=1)
            start = int(bounds[0])
            end = int(bounds[-1])
            if start < 0 or end < start:
                raise ValueError("invalid range")
            resources.update(range(start, end + 1))
    except ValueError as error:
        raise Glm47ServingHarnessError(
            f"cannot parse {description} resource list"
        ) from error
    if not resources:
        raise Glm47ServingHarnessError(f"{description} resource list is empty")
    return tuple(sorted(resources))


def _proc_status_resource_list(contents: str, field_name: str) -> tuple[int, ...]:
    prefix = f"{field_name}:"
    matches = [
        line.removeprefix(prefix).strip()
        for line in contents.splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise Glm47ServingHarnessError(f"process status lacks exact {field_name}")
    return _parse_linux_resource_list(matches[0], field_name)


def _parse_numa_maps_policy(value: str) -> tuple[int, ...]:
    policy, separator, resources = value.partition(":")
    if separator != ":" or policy != "bind":
        raise Glm47ServingHarnessError(
            "server placement NUMA map does not use a bind policy"
        )
    return _parse_linux_resource_list(resources, "NUMA bind policy")


def observe_running_server_process(
    config: ServingBenchmarkConfig,
    server: ServerProcess,
    *,
    proc_root: Path = Path("/proc"),
    affinity_reader: Callable[[int], set[int]] | None = None,
) -> SglangKtServingOwnedServerProcessIdentity:
    if server.process.poll() is not None:
        raise Glm47ServingHarnessError(
            f"serving process exited early with {server.process.returncode}"
        )
    process_root = proc_root / str(server.process.pid)
    if affinity_reader is None:
        affinity_reader = cast(
            Callable[[int], set[int]], os.__dict__["sched_getaffinity"]
        )
    try:
        executable = (process_root / "exe").resolve(strict=True)
        expected_executable = Path(config.runtime_python.path).resolve(strict=True)
        command_line = tuple(
            item.decode("utf-8", errors="strict")
            for item in (process_root / "cmdline").read_bytes().split(b"\0")
            if item
        )
        working_directory = (process_root / "cwd").resolve(strict=True)
        stat_value = (process_root / "stat").read_text(encoding="ascii")
        close_parenthesis = stat_value.rfind(")")
        stat_fields = tuple(stat_value[close_parenthesis + 2 :].split())
        if close_parenthesis < 0 or len(stat_fields) < 20:
            raise ValueError("truncated process stat")
        start_time_ticks = int(stat_fields[19])
        cpu_affinity = tuple(sorted(affinity_reader(server.process.pid)))
        status = (process_root / "status").read_text(encoding="ascii")
        status_cpu_affinity = _proc_status_resource_list(status, "Cpus_allowed_list")
        allowed_memory_nodes = _proc_status_resource_list(status, "Mems_allowed_list")
        numa_lines = tuple(
            line.split()
            for line in (process_root / "numa_maps")
            .read_text(encoding="ascii")
            .splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise Glm47ServingHarnessError(
            "cannot verify the running server process identity"
        ) from error
    expected_cpu_affinity = tuple(config.host.cpu_cores)
    expected_memory_nodes = tuple(config.host.memory_nodes)
    observed_policies = tuple(fields[1] for fields in numa_lines if len(fields) >= 2)
    if (
        executable != expected_executable
        or command_line != server.command[5:]
        or working_directory != Path(server.working_directory)
        or start_time_ticks != server.owned.start_time_ticks
        or cpu_affinity != expected_cpu_affinity
        or status_cpu_affinity != expected_cpu_affinity
        or not set(expected_memory_nodes).issubset(allowed_memory_nodes)
        or not observed_policies
        or any(
            _parse_numa_maps_policy(policy) != expected_memory_nodes
            for policy in observed_policies
        )
    ):
        raise Glm47ServingHarnessError(
            "running server process differs from its admitted placement"
        )
    return SglangKtServingOwnedServerProcessIdentity(
        pid=server.process.pid,
        proc_start_time_ticks=start_time_ticks,
        executable=config.runtime_python.path,
        argv_sha256=server.launch_argv_sha256,
        cpu_affinity=cpu_affinity,
        memory_nodes=expected_memory_nodes,
    )


def launch_server(
    config: ServingBenchmarkConfig,
    process_spec: SglangKtProcessLaunchSpec,
    environment: dict[str, str],
    results: validation.ResultDirectory,
    registry: validation.OwnershipRegistry,
    owned_cgroup: validation.OwnedCgroup,
    working_directory: Path,
) -> ServerProcess:
    retained_numactl = open_bound_executable(config.tools.numactl, "numactl")
    command = _server_command(config, process_spec)
    command_sha256 = _canonical_sha256(list(command))
    cgroup_descriptor: int | None = None
    started = time.monotonic()
    stdout: IO[bytes] | None = None
    stderr: IO[bytes] | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        validation.require_single_threaded_harness()
        stdout = results.create_log(SERVER_STDOUT_FILENAME)
        stderr = results.create_log(SERVER_STDERR_FILENAME)
        cgroup_descriptor = validation.open_owned_cgroup_control(
            owned_cgroup, "cgroup.procs", os.O_WRONLY
        )
        process = subprocess.Popen(
            command,
            executable=retained_numactl.execution_path,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            cwd=working_directory,
            pass_fds=(cgroup_descriptor, retained_numactl.descriptor),
            preexec_fn=functools.partial(
                validation.enter_owned_cgroup, cgroup_descriptor
            ),
            start_new_session=True,
        )
        os.close(cgroup_descriptor)
        cgroup_descriptor = None
        owned = validation.process_identity(
            config,
            process.pid,
            registry.owner_token,
            str(results.path_for(SERVER_STDERR_FILENAME)),
        )
        registry.register(owned)
        return ServerProcess(
            process=process,
            owned=owned,
            command=command,
            environment=environment,
            working_directory=str(working_directory),
            launch_argv_sha256=command_sha256,
            launch_seconds=time.monotonic() - started,
        )
    except BaseException:
        if process is not None:
            validation.terminate_owned_group(
                process.pid, registry.owner_token, config.timeouts.cleanup_seconds
            )
        raise
    finally:
        retained_numactl.close()
        if cgroup_descriptor is not None:
            os.close(cgroup_descriptor)
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()


def wait_for_server(
    config: ServingBenchmarkConfig,
    server: ServerProcess,
    latch: validation.SignalLatch,
) -> tuple[
    Glm47NativeServingClient,
    float,
    EndpointCallObservation,
    SglangKtServingServerInfoIdentity,
    float,
    SglangKtServingOwnedServerProcessIdentity,
]:
    client = Glm47NativeServingClient(
        f"http://{config.service_endpoint.argument}",
        timeout_seconds=config.request_timeout_seconds,
    )
    started = time.monotonic()
    deadline = started + config.readiness_timeout_seconds
    health: EndpointCallObservation | None = None
    while time.monotonic() < deadline:
        latch.checkpoint()
        if server.process.poll() is not None:
            client.close()
            raise Glm47ServingHarnessError(
                f"serving process exited before readiness: {server.process.returncode}"
            )
        try:
            health = client.health_generate()
            break
        except (Glm47ServingClientError, httpx.HTTPError, OSError):
            time.sleep(0.25)
    if health is None:
        client.close()
        raise Glm47ServingHarnessError("serving process did not become ready")
    ready_seconds = time.monotonic() - started
    process_identity = observe_running_server_process(config, server)
    server_info = client.server_info()
    identity = build_glm47_server_info_identity(
        server_info,
        node_id=NodeId(config.host.node_id),
        host=config.service_endpoint.ip,
        port=config.service_endpoint.port,
    )
    return (
        client,
        ready_seconds,
        health,
        identity,
        server_info.call.elapsed_seconds,
        process_identity,
    )


def build_run_identity(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    admission: AdmissionEvidence,
    server_info: SglangKtServingServerInfoIdentity,
    server: ServerProcess,
    server_identity: SglangKtServingOwnedServerProcessIdentity,
) -> SglangKtWarmServingRunIdentity:
    source = _source_identity(deployment)
    source_by_path = {item.relative_path: item for item in source.files}
    client_source = source_by_path[SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH]
    model = admission.model_receipt
    kernel = admission.kernel_receipt
    process_spec = admission.process_spec
    return SglangKtWarmServingRunIdentity(
        admission=SglangKtServingAdmissionBinding(
            model_runtime_validation_receipt=_file_identity(
                Path(config.admission.model_runtime_validation_receipt.path),
                maximum_bytes=MAXIMUM_MEASUREMENT_BYTES,
            ),
            kernel_runtime_validation_receipt=_file_identity(
                Path(config.admission.kernel_runtime_validation_receipt.path),
                maximum_bytes=MAXIMUM_MEASUREMENT_BYTES,
            ),
            model_contract_receipt=_file_identity(
                Path(config.admission.model_contract_receipt.path),
                maximum_bytes=MAXIMUM_MEASUREMENT_BYTES,
            ),
        ),
        runtime=SglangKtServingRuntimeIdentity(
            executable=config.runtime_python.path,
            runtime_build_receipt=_file_identity(
                Path(config.build_receipt.path),
                maximum_bytes=validation.MAXIMUM_JSON_BYTES,
            ),
            numactl_executable=bound_executable_identity(
                config.tools.numactl, "numactl"
            ),
            nvidia_smi_executable=bound_executable_identity(
                config.tools.nvidia_smi, "nvidia-smi"
            ),
            systemctl_executable=bound_executable_identity(
                config.tools.systemctl, "systemctl"
            ),
            runtime_build_id=kernel.runtime_build_id,
            python_version=platform.python_version(),
            torch_version=kernel.torch_version,
            cuda_version=kernel.cuda_version,
            sglang_revision=kernel.sglang_revision,
            ktransformers_revision=kernel.ktransformers_revision,
            sgl_kernel_build_id=kernel.sgl_kernel_build_id,
            deep_gemm_build_id=kernel.deep_gemm_build_id,
            kt_kernel_build_id=kernel.kt_kernel_build_id,
        ),
        model=SglangKtServingModelIdentity(
            model_id=model.model_id,
            model_revision=model.model_revision,
            model_path=model.model_path,
            model_config_sha256=model.model_config_sha256,
            model_index_sha256=model.model_index_sha256,
            physical_weight_bytes=model.model_physical_weight_bytes,
        ),
        process_spec=SglangKtServingProcessSpecIdentity(
            receipt=admission.process_spec_file,
            process_spec_sha256=config.admission.process_spec_sha256,
            launch_argv_sha256=server_identity.argv_sha256,
            launch_environment_sha256=_canonical_sha256(
                sorted(server.environment.items())
            ),
            target_profile=process_spec.target_profile,
            resident_gpu_experts=process_spec.stage.resident_gpu_experts,
            cpu_cores=process_spec.cpu_cores,
            memory_nodes=process_spec.memory_nodes,
        ),
        tuning=SglangKtServingTuningIdentity(
            mode="untuned",
            config_directory=None,
            manifest_sha256=None,
            config_files=(),
        ),
        client=SglangKtServingClientIdentity(
            source_file=SglangKtServingFileIdentity(
                path=str(
                    Path(source.repository_root)
                    / SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH
                ),
                size_bytes=client_source.size_bytes,
                sha256=client_source.sha256,
            ),
            source_bundle_sha256=source.source_bundle_sha256,
            protocol_version=1,
            http_library="httpx",
            http_library_version=importlib.metadata.version("httpx"),
            request_timeout_seconds=config.request_timeout_seconds,
        ),
        source=source,
        topology=SglangKtServingTopologyIdentity(
            deployment="local",
            interconnect="none",
            stages=(
                SglangKtServingTopologyStage(
                    pipeline_rank=0,
                    node_id=NodeId(config.host.node_id),
                    host=config.service_endpoint.ip,
                    port=config.service_endpoint.port,
                    gpu_uuid=config.host.gpu.uuid,
                    hca_devices=(),
                ),
            ),
        ),
        server_info=(server_info,),
    )


def collect_interleaved_workloads(
    client: Glm47NativeServingClient,
    server: ServerProcess,
    config: ServingBenchmarkConfig,
    cache_directories: tuple[Path, ...],
    latch: validation.SignalLatch,
    server_identity: SglangKtServingOwnedServerProcessIdentity,
) -> tuple[
    tuple[SglangKtServingWorkloadEvidence, SglangKtServingWorkloadEvidence],
    SglangKtServingJitCacheEvidence,
]:
    prefill = prepare_glm47_serving_workload("prefill")
    decode = prepare_glm47_serving_workload("decode")
    warmups: dict[str, list[SglangKtServingInvocationEvidence]] = {
        "prefill": [],
        "decode": [],
    }
    samples: dict[str, list[SglangKtServingInvocationEvidence]] = {
        "prefill": [],
        "decode": [],
    }
    warmup_manifests: list[str] = []
    for ordinal in range(1, 3):
        for workload in (prefill, decode):
            latch.checkpoint()
            if observe_running_server_process(config, server) != server_identity:
                raise Glm47ServingHarnessError(
                    "owned serving process identity changed during warmup"
                )
            warmups[workload.receipt_request.kind].append(
                run_glm47_serving_invocation(client, workload, ordinal)
            )
        warmup_manifests.append(stable_cache_manifest(cache_directories))
    if warmup_manifests[0] != warmup_manifests[1]:
        raise Glm47ServingHarnessError(
            "JIT cache changed between the two interleaved warmup rounds"
        )

    for ordinal in range(1, 4):
        for workload in (prefill, decode):
            latch.checkpoint()
            if observe_running_server_process(config, server) != server_identity:
                raise Glm47ServingHarnessError(
                    "owned serving process identity changed during measurement"
                )
            samples[workload.receipt_request.kind].append(
                run_glm47_serving_invocation(client, workload, ordinal)
            )
    measured_manifest = stable_cache_manifest(cache_directories)
    if measured_manifest != warmup_manifests[-1]:
        raise Glm47ServingHarnessError("JIT cache changed during measured requests")

    workloads = (
        SglangKtServingWorkloadEvidence(
            request=prefill.receipt_request,
            warmups=tuple(warmups["prefill"]),
            samples=tuple(samples["prefill"]),
        ),
        SglangKtServingWorkloadEvidence(
            request=decode.receipt_request,
            warmups=tuple(warmups["decode"]),
            samples=tuple(samples["decode"]),
        ),
    )
    return workloads, SglangKtServingJitCacheEvidence(
        cache_directories=tuple(str(path) for path in cache_directories),
        after_penultimate_warmup_manifest_sha256=warmup_manifests[0],
        after_final_warmup_manifest_sha256=warmup_manifests[1],
        after_measurement_manifest_sha256=measured_manifest,
    )


def terminate_server_unforced(
    config: ServingBenchmarkConfig,
    server: ServerProcess,
    owner_token: str,
    server_identity: SglangKtServingOwnedServerProcessIdentity,
) -> Literal[-15, 0]:
    if observe_running_server_process(config, server) != server_identity:
        raise Glm47ServingHarnessError(
            "owned serving process changed before SIGTERM delivery"
        )
    if (
        validation.live_group_ownership(server.owned.process_group_id, owner_token)
        != "owned"
    ):
        raise Glm47ServingHarnessError(
            "owned serving group was not live before SIGTERM delivery"
        )
    try:
        os.killpg(server.owned.process_group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as error:
        raise Glm47ServingHarnessError(
            "could not positively deliver SIGTERM to the owned serving group"
        ) from error
    deadline = time.monotonic() + config.timeouts.cleanup_seconds
    while time.monotonic() < deadline:
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.process.wait(timeout=0.05)
        validation.reap_adopted_children()
        ownership = validation.live_group_ownership(
            server.owned.process_group_id, owner_token
        )
        owned_processes = validation.owned_token_processes(owner_token)
        if (
            server.process.returncode is not None
            and ownership == "absent"
            and owned_processes == {}
        ):
            if server.process.returncode not in {-15, 0}:
                raise Glm47ServingHarnessError(
                    "server returned a noncanonical code after SIGTERM: "
                    f"{server.process.returncode}"
                )
            return cast(Literal[-15, 0], server.process.returncode)
        if ownership in {"foreign", "unknown"} or owned_processes is None:
            raise Glm47ServingHarnessError(
                "owned serving process cleanup became ambiguous"
            )
        time.sleep(0.05)
    raise Glm47ServingHarnessError("owned serving group exceeded SIGTERM grace")


def collect_serving_measurement(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    results: validation.ResultDirectory,
    registry: validation.OwnershipRegistry,
    owned_cgroup: validation.OwnedCgroup,
    scratch: validation.OwnedScratchDirectory,
    latch: validation.SignalLatch,
    base_environment: Mapping[str, str],
) -> CompletedMeasurement:
    admission = collect_admission_evidence(
        config,
        deployment,
        results,
        registry,
        owned_cgroup,
        latch,
        base_environment,
    )
    cache_directories = create_cache_directories(scratch)
    environment = build_serving_environment(
        config,
        admission.process_spec,
        registry.owner_token,
        scratch,
        cache_directories,
        parent_environment=base_environment,
    )
    server = launch_server(
        config,
        admission.process_spec,
        environment,
        results,
        registry,
        owned_cgroup,
        scratch.path,
    )
    client: Glm47NativeServingClient | None = None
    try:
        (
            client,
            ready_seconds,
            health,
            server_info,
            server_info_seconds,
            server_identity,
        ) = wait_for_server(config, server, latch)
        identity = build_run_identity(
            config, deployment, admission, server_info, server, server_identity
        )
        sanity = run_glm47_serving_sanity(client, config.model_path)
        if observe_running_server_process(config, server) != server_identity:
            raise Glm47ServingHarnessError(
                "owned serving process identity changed during sanity generation"
            )
        workloads, jit_cache = collect_interleaved_workloads(
            client,
            server,
            config,
            cache_directories,
            latch,
            server_identity,
        )
    finally:
        if client is not None:
            client.close()
    return_code = terminate_server_unforced(
        config, server, registry.owner_token, server_identity
    )
    return CompletedMeasurement(
        identity=identity,
        setup=SglangKtServingSetupEvidence(
            process_launch_seconds=server.launch_seconds,
            health_ready_seconds=ready_seconds,
            admission_seconds=admission.admission_seconds,
            server_info_fetch_seconds=server_info_seconds,
            health_generate_status_code=cast(Literal[200], health.status_code),
            health_generate_response_sha256=health.response_sha256,
        ),
        sanity=sanity,
        jit_cache=jit_cache,
        workloads=workloads,
        server=server,
        server_identity=server_identity,
        server_return_code=return_code,
    )


def build_serving_static_metadata(
    config: ServingBenchmarkConfig,
    command: Sequence[str],
    config_sha256: str,
    deployment: validation.DeploymentIdentity,
) -> JsonObject:
    metadata = validation.build_static_metadata(
        config, command, config_sha256, deployment
    )
    metadata["validation_contract"] = {
        "kind": "glm47_sglang_kt_warm_serving_benchmark",
        "phase": "serving_baseline",
        "profiler": "none",
        "hca_requirement": config.hca_requirement,
        "config_sha256": config_sha256,
        "orchestrator_sha256": deployment.orchestrator_sha256,
        "validator_sha256": deployment.validator_sha256,
        "runtime_python": config.runtime_python.model_dump(mode="json"),
        "build_receipt": config.build_receipt.model_dump(mode="json"),
        "model_contract": config.model_contract.model_dump(mode="json"),
        "admission": config.admission.model_dump(mode="json"),
        "tools": config.tools.model_dump(mode="json"),
        "lease_execution": config.lease_execution.model_dump(mode="json"),
        "coordination_guard": {
            "peer_role": "idle_nonparticipant",
            "host_guard_config_sha256": host_guard.calculate_host_guard_config_sha256(
                config.coordination_guard.idle_peer
            ),
            "peer_binding_sha256": (
                host_guard.calculate_coordination_peer_binding_sha256(
                    config.coordination_guard.idle_peer.peer
                )
            ),
            "local_hca": config.coordination_guard.local_hca.model_dump(mode="json"),
            "model_filesystem": (
                config.coordination_guard.model_filesystem.model_dump(mode="json")
            ),
            "preflight_inside_lease": True,
            "postflight_before_release": True,
        },
        "resident_gpu_experts": config.resident_gpu_experts,
        "semantic_sanity": {
            "required": True,
            "before_warmups": True,
            "local_tokenizer_and_chat_template": True,
            "post_sanity_cache_flush": True,
        },
        "warmup_rounds": 2,
        "measurement_rounds": 3,
        "workload_order": ["prefill", "decode"],
        "request_timeout_seconds": config.request_timeout_seconds,
        "readiness_timeout_seconds": config.readiness_timeout_seconds,
        "radix_cache_disabled": True,
        "cuda_graph_disabled": True,
        "instrumentation": "none",
        "outer_finalization_required": True,
        "python_bytecode_policy": "PYTHONDONTWRITEBYTECODE=1",
        "scratch_directory": config.scratch_directory,
    }
    return metadata


def validate_active_serving_lease(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    results: validation.ResultDirectory,
    *,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int | None = None,
) -> tuple[JsonObject, ServingHandoffIdentity]:
    lock_identity: FilesystemObjectIdentity | None = None
    try:
        lock_descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(lock_descriptor, "rb", closefd=True) as lock_file:
            lock_identity = filesystem_object_identity(
                lock_path, lock_file.fileno(), "regular_file"
            )
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                raise Glm47ServingHarnessError(
                    "benchmark coordination lock is not held"
                )
    except OSError as error:
        raise Glm47ServingHarnessError(
            f"cannot inspect benchmark lock: {error}"
        ) from error

    expected_pid = os.getpid() if process_id is None else process_id
    deadline = time.monotonic() + validation.LEASE_CHILD_BIND_SECONDS
    while True:
        record = validation.read_json_regular(
            lease_path, "active serving benchmark lease"
        )
        child_pid = record.get("child_pid")
        if child_pid == expected_pid:
            break
        if child_pid is not None or time.monotonic() >= deadline:
            raise Glm47ServingHarnessError(
                "active serving lease does not belong to this child"
            )
        time.sleep(0.05)

    expected_command = _serving_child_argv(
        config, lease_path=lease_path, lock_path=lock_path
    )
    if record.get("command") != list(expected_command):
        raise Glm47ServingHarnessError("active serving lease command changed")
    expected_values: dict[str, object] = {
        "owner": config.lease_execution.owner,
        "purpose": config.lease_execution.purpose,
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "wrapper_pid": os.getppid(),
        "child_cleanup_confirmation_required": True,
        "cleanup_grace_seconds": config.lease_execution.cleanup_grace_seconds,
    }
    for name, value in expected_values.items():
        if record.get(name) != value:
            raise Glm47ServingHarnessError(
                f"active serving lease {name} differs from config"
            )
    lease_id = record.get("lease_id")
    if not isinstance(lease_id, str) or re.fullmatch(r"[0-9a-f]{32}", lease_id) is None:
        raise Glm47ServingHarnessError("active serving lease ID is invalid")
    metadata = validation.json_object(record.get("metadata"), "lease metadata")
    expected_metadata = build_serving_static_metadata(
        config,
        expected_command,
        validation.config_sha256(config_path),
        deployment,
    )
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise Glm47ServingHarnessError(
                f"active serving lease metadata.{name} changed"
            )
    if set(metadata) != {*expected_metadata, "generated_at"} or not isinstance(
        metadata.get("generated_at"), str
    ):
        raise Glm47ServingHarnessError(
            "active serving lease metadata has unauthorized fields"
        )
    try:
        heartbeat = datetime.fromisoformat(
            cast(str, record["heartbeat"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError) as error:
        raise Glm47ServingHarnessError("active lease heartbeat is invalid") from error
    now = _utc_now()
    if (
        heartbeat < now - validation.LEASE_HEARTBEAT_MAX_AGE
        or heartbeat > now + timedelta(minutes=1)
    ):
        raise Glm47ServingHarnessError("active lease heartbeat is stale or future")
    return record, ServingHandoffIdentity(
        result_directory=filesystem_object_identity(
            results.path, results.descriptor, "directory"
        ),
        coordination_lock=lock_identity,
    )


def _publish_measurement(
    config: ServingBenchmarkConfig,
    results: validation.ResultDirectory,
    registry: validation.OwnershipRegistry,
    completed: CompletedMeasurement,
    cgroup_path: Path,
    handoff: ServingHandoffIdentity,
    coordination_guard: SglangKtServingCoordinationGuardEvidence,
    lease_id: str,
) -> WarmServingMeasurementV2:
    measurement = WarmServingMeasurementV2(
        schema_version=2,
        status="passed",
        generated_at_utc=_utc_now().isoformat(timespec="microseconds"),
        profiler="none",
        instrumentation="none",
        radix_cache_disabled=True,
        max_concurrent_requests=1,
        lease_id=lease_id,
        identity_sha256=calculate_sglang_kt_warm_serving_run_identity_sha256(
            completed.identity
        ),
        identity=completed.identity,
        setup=completed.setup,
        sanity=completed.sanity,
        jit_cache=completed.jit_cache,
        workloads=completed.workloads,
        coordination_guard=coordination_guard,
        owner_token=registry.owner_token,
        server_process=completed.server_identity,
        server_return_code=completed.server_return_code,
        termination_signal="SIGTERM",
        forced=False,
        owned_processes_absent=True,
        delegated_cgroup_path=str(cgroup_path),
        delegated_cgroup_removed=True,
        transient_unit_name=_systemd_unit_name(config),
        handoff=handoff,
    )
    results.write_json(
        MEASUREMENT_FILENAME,
        cast(
            Mapping[str, object],
            cast(object, measurement.model_dump(mode="json")),
        ),
        replace=False,
    )
    persisted = WarmServingMeasurementV2.model_validate_json(
        results.read_bytes(MEASUREMENT_FILENAME, MAXIMUM_MEASUREMENT_BYTES)
    )
    if persisted != measurement:
        raise Glm47ServingHarnessError("persisted serving measurement changed")
    return measurement


def run_serving_benchmark(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    results: validation.ResultDirectory,
    latch: validation.SignalLatch,
    handoff: ServingHandoffIdentity,
    lease_id: str,
) -> JsonObject:
    owner_token = f"{config.run_id}:{uuid.uuid4().hex}"
    registry = validation.OwnershipRegistry(config, results, owner_token)
    caught_error: BaseException | None = None
    completed: CompletedMeasurement | None = None
    coordination_preflight: CoordinationPreflight | None = None
    coordination_evidence: SglangKtServingCoordinationGuardEvidence | None = None
    preflight: JsonObject | None = None
    scratch: validation.OwnedScratchDirectory | None = None
    owned_cgroup: validation.OwnedCgroup | None = None
    cgroup_evidence: JsonObject | None = None
    cgroup_path: Path | None = None
    cleanup_succeeded = True
    cleanup_errors: list[str] = []
    try:
        validation.enable_child_subreaper()
        latch.checkpoint()
        owned_cgroup = validation.create_owned_cgroup(config, owner_token)
        cgroup_path = owned_cgroup.path
        cgroup_evidence = validation.owned_cgroup_evidence(owned_cgroup)
        registry.bind_cgroup(owned_cgroup)

        def gpu_probe(command: Sequence[str], description: str) -> str:
            if not command or command[0] != config.tools.nvidia_smi.path:
                raise Glm47ServingHarnessError(
                    "validation requested an unbound GPU helper"
                )
            return run_bound_tool_text(
                config.tools.nvidia_smi, command[1:], description
            )

        preflight = validation.collect_live_preflight(config, gpu_probe=gpu_probe)
        coordination_preflight = collect_coordination_preflight(config)
        scratch = validation.create_scratch(config)
        base_environment = validation.build_child_environment(config, owner_token)
        completed = collect_serving_measurement(
            config,
            deployment,
            results,
            registry,
            owned_cgroup,
            scratch,
            latch,
            base_environment,
        )
        if validation.load_deployment_identity(Path(deployment.root)) != deployment:
            raise Glm47ServingHarnessError(
                "immutable serving deployment changed during measurement"
            )
    except BaseException as error:
        caught_error = error
    finally:
        latch.begin_cleanup()
        try:
            process_cleanup = validation.cleanup_all_owned_processes(
                config, registry, str(results.path_for(SERVER_STDERR_FILENAME))
            )
            cleanup_succeeded = process_cleanup and cleanup_succeeded
            if not process_cleanup:
                cleanup_errors.append("owned process cleanup was not proven complete")
        except BaseException as cleanup_error:
            cleanup_succeeded = False
            cleanup_errors.append(
                "owned process cleanup raised "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if owned_cgroup is None:
            cleanup_succeeded = False
            cleanup_errors.append("owned serving cgroup was not established")
        else:
            try:
                cgroup_cleanup = validation.cleanup_owned_cgroup(
                    owned_cgroup, config.timeouts.cleanup_seconds
                )
                cleanup_succeeded = cgroup_cleanup and cleanup_succeeded
                if not cgroup_cleanup:
                    cleanup_errors.append("owned serving cgroup was not removed")
            except BaseException as cleanup_error:
                cleanup_succeeded = False
                cleanup_errors.append(
                    "owned cgroup cleanup raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if scratch is not None:
            try:
                scratch_cleanup = validation.cleanup_owned_scratch(scratch)
                cleanup_succeeded = scratch_cleanup and cleanup_succeeded
                if not scratch_cleanup:
                    cleanup_errors.append("owned serving scratch was not removed")
            except BaseException as cleanup_error:
                cleanup_succeeded = False
                cleanup_errors.append(
                    "owned scratch cleanup raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    if coordination_preflight is not None:
        try:
            coordination_evidence = collect_coordination_postflight(
                config, coordination_preflight
            )
        except BaseException as guard_error:
            if caught_error is None:
                caught_error = guard_error
            else:
                caught_error = Glm47ServingHarnessError(
                    f"{type(caught_error).__name__}: {caught_error}; "
                    "coordination postflight failed: "
                    f"{type(guard_error).__name__}: {guard_error}"
                )

    measurement: WarmServingMeasurementV2 | None = None
    if (
        caught_error is None
        and cleanup_succeeded
        and completed is not None
        and cgroup_path is not None
        and coordination_evidence is not None
    ):
        try:
            measurement = _publish_measurement(
                config,
                results,
                registry,
                completed,
                cgroup_path,
                handoff,
                coordination_evidence,
                lease_id,
            )
        except BaseException as error:
            caught_error = error

    status = (
        "completed"
        if caught_error is None and cleanup_succeeded and measurement is not None
        else ("cleanup_failed" if not cleanup_succeeded else "benchmark_failed")
    )
    owned_processes: list[JsonValue] = [
        cast(JsonObject, cast(object, asdict(process)))
        for process in registry.processes
    ]
    result: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "status": status,
        "phase": "serving_baseline",
        "reportable": status == "completed",
        "performance_comparable": False,
        "outer_finalization_required": True,
        "completed_normally": caught_error is None,
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_errors": cast(list[JsonValue], cast(object, cleanup_errors)),
        "interrupted_signal": latch.signal_number,
        "error": (
            None
            if caught_error is None
            else f"{type(caught_error).__name__}: {caught_error}"
        ),
        "profiler": "none",
        "unsafe_profiler_drivers_used": False,
        "containment": cgroup_evidence,
        "preflight": preflight,
        "measurement_file": MEASUREMENT_FILENAME if measurement is not None else None,
        "measurement_sha256": (
            results.sha256(MEASUREMENT_FILENAME) if measurement is not None else None
        ),
        "coordination_guard_evidence_sha256": (
            coordination_evidence.evidence_sha256
            if measurement is not None and coordination_evidence is not None
            else None
        ),
        "owned_processes": owned_processes,
    }
    child_manifest: dict[str, object] = {
        **cast(dict[str, object], cast(object, result)),
        "manifest_writer": Path(__file__).name,
        "config": config.model_dump(mode="json"),
        "deployment": asdict(deployment),
        "python_bytecode_policy": "PYTHONDONTWRITEBYTECODE=1",
    }
    results.write_json(
        validation.CHILD_MANIFEST_FILENAME, child_manifest, replace=False
    )
    results.write_json(
        validation.BENCHMARK_RESULT_FILENAME,
        cast(Mapping[str, object], cast(object, result)),
        replace=False,
    )
    return result


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or path == Path("/"):
        raise Glm47ServingHarnessError("result directory path is not canonical")
    descriptor = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."} or "\0" in component:
                raise Glm47ServingHarnessError(
                    "result directory path contains an unsafe component"
                )
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_lock(path: Path) -> IO[bytes]:
    if not path.is_absolute() or path == Path("/"):
        raise Glm47ServingHarnessError("coordination lock path is not canonical")
    parent_descriptor = _open_absolute_directory(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            os.close(descriptor)
            raise Glm47ServingHarnessError(
                "coordination lock is not a singly linked regular file"
            )
        os.fsync(parent_descriptor)
        return os.fdopen(descriptor, "r+b", closefd=True)
    finally:
        os.close(parent_descriptor)


def run_systemd_with_preserved_handoff(
    config: ServingBenchmarkConfig,
    command: Sequence[str],
    *,
    lock_path: Path,
) -> tuple[int, PreservedServingHandoff]:
    lock_file = _open_or_create_lock(lock_path)
    result_root_descriptor: int | None = None
    result_descriptor: int | None = None
    process: subprocess.Popen[bytes] | None = None
    results: validation.ResultDirectory | None = None
    handoff_transferred = False
    try:
        lock_identity = filesystem_object_identity(
            lock_path, lock_file.fileno(), "regular_file"
        )
        result_path = Path(config.result_directory)
        result_root_descriptor = _open_absolute_directory(result_path.parent)
        try:
            os.stat(
                result_path.name,
                dir_fd=result_root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise Glm47ServingHarnessError(
                "serving result directory existed before wrapper launch"
            )
        process = subprocess.Popen(tuple(command))
        while result_descriptor is None:
            try:
                result_descriptor = os.open(
                    result_path.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=result_root_descriptor,
                )
            except FileNotFoundError:
                if process.poll() is not None:
                    raise Glm47ServingHarnessError(
                        "lease wrapper exited without creating its result directory"
                    ) from None
                time.sleep(0.05)
        results = validation.ResultDirectory(result_path, result_descriptor)
        os.close(result_descriptor)
        result_descriptor = None
        return_code = process.wait()
        identity = ServingHandoffIdentity(
            result_directory=filesystem_object_identity(
                results.path, results.descriptor, "directory"
            ),
            coordination_lock=lock_identity,
        )
        handoff = PreservedServingHandoff(
            lock_file=lock_file,
            results=results,
            identity=identity,
        )
        results = None
        handoff_transferred = True
        return return_code, handoff
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5.0)
            if process.poll() is None:
                process.kill()
                process.wait()
        raise
    finally:
        if results is not None:
            results.close()
        if result_descriptor is not None:
            os.close(result_descriptor)
        if result_root_descriptor is not None:
            os.close(result_root_descriptor)
        if not handoff_transferred:
            lock_file.close()


def write_new_bytes(
    results: validation.ResultDirectory, name: str, contents: bytes
) -> None:
    results.validate_name(name)
    results.validate_identity()
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=results.descriptor,
        )
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Glm47ServingHarnessError("short performance-receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            name,
            src_dir_fd=results.descriptor,
            dst_dir_fd=results.descriptor,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=results.descriptor)
        os.fsync(results.descriptor)
    except FileExistsError as error:
        raise Glm47ServingHarnessError(
            f"refusing to replace existing final result {name}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=results.descriptor)
        if linked:
            results.validate_identity()


def _load_measurement(
    results: validation.ResultDirectory,
) -> BoundMeasurementSnapshot:
    descriptor: int | None = None
    try:
        results.validate_identity()
        descriptor = os.open(
            MEASUREMENT_FILENAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=results.descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAXIMUM_MEASUREMENT_BYTES
        ):
            raise Glm47ServingHarnessError(
                "serving measurement is not a bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Glm47ServingHarnessError("serving measurement was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise Glm47ServingHarnessError(
                "serving measurement changed while it was read"
            )
        parse_sglang_kt_strict_json(contents)
        measurement = WarmServingMeasurementV2.model_validate_json(contents)
        final = os.fstat(descriptor)
        current = os.stat(
            MEASUREMENT_FILENAME,
            dir_fd=results.descriptor,
            follow_symlinks=False,
        )
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise Glm47ServingHarnessError(
                "serving measurement path changed during its descriptor snapshot"
            )
        return BoundMeasurementSnapshot(
            measurement=measurement,
            contents=contents,
            sha256=hashlib.sha256(contents).hexdigest(),
            device=before.st_dev,
            inode=before.st_ino,
        )
    except (OSError, ValueError, ValidationError) as error:
        raise Glm47ServingHarnessError("serving measurement is invalid") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_wrapper_manifest(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    results: validation.ResultDirectory,
    snapshot: BoundMeasurementSnapshot,
) -> JsonObject:
    measurement = snapshot.measurement
    manifest = results.read_json(validation.CHILD_MANIFEST_FILENAME)
    benchmark_result = validation.json_object(
        manifest.get("benchmark_result"), "wrapper benchmark result"
    )
    child_manifest = validation.json_object(
        manifest.get("child_manifest"), "wrapper child manifest"
    )
    runtime_metadata = validation.json_object(
        manifest.get("runtime_metadata"), "wrapper runtime metadata"
    )
    metadata = validation.json_object(manifest.get("metadata"), "wrapper metadata")
    expected_command = _serving_child_argv(
        config,
        lease_path=Path(config.lease_execution.lease_path),
        lock_path=Path(config.lease_execution.lock_path),
    )
    immutable_config = Path(deployment.root) / validation.IMMUTABLE_CONFIG_RELATIVE_PATH
    expected_metadata = build_serving_static_metadata(
        config,
        expected_command,
        validation.config_sha256(immutable_config),
        deployment,
    )
    generated_at = metadata.get("generated_at")
    expected_metadata["generated_at"] = cast(JsonValue, generated_at)
    lease_id = manifest.get("lease_id")
    coordination_sha256 = measurement.coordination_guard.evidence_sha256
    if (
        manifest.get("manifest_writer") != "benchmark_lease.py"
        or not isinstance(lease_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", lease_id) is None
        or lease_id != measurement.lease_id
        or manifest.get("owner") != config.lease_execution.owner
        or manifest.get("purpose") != config.lease_execution.purpose
        or manifest.get("run_id") != config.run_id
        or manifest.get("exo_namespace") != config.namespace
        or manifest.get("ports") != list(config.reserved_ports)
        or manifest.get("result_directory") != config.result_directory
        or manifest.get("command") != list(expected_command)
        or manifest.get("cleanup_grace_seconds")
        != config.lease_execution.cleanup_grace_seconds
        or manifest.get("child_cleanup_confirmation_required") is not True
        or not isinstance(generated_at, str)
        or metadata != expected_metadata
        or manifest.get("status") != "completed"
        or manifest.get("return_code") != 0
        or manifest.get("command_return_code") != 0
        or manifest.get("cleanup_succeeded") is not True
        or manifest.get("cleanup_forced") is not False
        or benchmark_result.get("status") != "completed"
        or benchmark_result.get("run_id") != config.run_id
        or benchmark_result.get("namespace") != config.namespace
        or benchmark_result.get("cleanup_succeeded") is not True
        or benchmark_result.get("performance_comparable") is not False
        or benchmark_result.get("outer_finalization_required") is not True
        or benchmark_result.get("measurement_file") != MEASUREMENT_FILENAME
        or benchmark_result.get("measurement_sha256") != snapshot.sha256
        or child_manifest.get("measurement_sha256") != snapshot.sha256
        or benchmark_result.get("coordination_guard_evidence_sha256")
        != coordination_sha256
        or child_manifest.get("coordination_guard_evidence_sha256")
        != coordination_sha256
        or child_manifest.get("run_id") != config.run_id
        or child_manifest.get("namespace") != config.namespace
        or child_manifest.get("config") != config.model_dump(mode="json")
        or child_manifest.get("deployment") != asdict(deployment)
        or runtime_metadata.get("owner_token") != measurement.owner_token
        or runtime_metadata.get("run_id") != config.run_id
        or runtime_metadata.get("namespace") != config.namespace
        or runtime_metadata.get("containment") != benchmark_result.get("containment")
        or child_manifest.get("containment") != benchmark_result.get("containment")
    ):
        raise Glm47ServingHarnessError(
            "lease wrapper did not finalize the measurement transaction cleanly"
        )
    return manifest


def _validate_identity_files(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    measurement: WarmServingMeasurementV2,
) -> None:
    identity = measurement.identity
    if validation.load_deployment_identity(Path(deployment.root)) != deployment:
        raise Glm47ServingHarnessError("immutable deployment changed before finalizing")
    if _source_identity(deployment) != identity.source:
        raise Glm47ServingHarnessError(
            "serving source bundle changed before finalizing"
        )
    file_identities = (
        identity.admission.model_runtime_validation_receipt,
        identity.admission.kernel_runtime_validation_receipt,
        identity.admission.model_contract_receipt,
        identity.runtime.runtime_build_receipt,
        identity.runtime.numactl_executable,
        identity.runtime.nvidia_smi_executable,
        identity.runtime.systemctl_executable,
        identity.process_spec.receipt,
        identity.client.source_file,
    )
    for expected in file_identities:
        maximum = (
            MAXIMUM_SOURCE_FILE_BYTES
            if expected == identity.client.source_file
            else validation.MAXIMUM_JSON_BYTES
        )
        if _file_identity(Path(expected.path), maximum_bytes=maximum) != expected:
            raise Glm47ServingHarnessError(
                f"bound serving artifact changed before finalizing: {expected.path}"
            )

    process_contents = read_sglang_kt_bound_file(
        Path(identity.process_spec.receipt.path), maximum_bytes=1024 * 1024
    ).contents
    process_spec = SglangKtProcessLaunchSpec.model_validate_json(process_contents)
    kernel = load_sglang_kt_kernel_runtime_validation_receipt(
        Path(config.admission.kernel_runtime_validation_receipt.path),
        expected_receipt_sha256=config.admission.kernel_runtime_validation_receipt.sha256,
    )
    model = load_sglang_kt_model_runtime_validation_receipt(
        Path(config.admission.model_runtime_validation_receipt.path),
        expected_validator_sha256=config.admission.validator_sha256,
        expected_process_spec_sha256=config.admission.process_spec_sha256,
        expected_model_contract_receipt_sha256=(
            config.admission.model_contract_receipt.sha256
        ),
        expected_kernel_receipt_sha256=(
            config.admission.kernel_runtime_validation_receipt.sha256
        ),
        expected_receipt_sha256=(
            config.admission.model_runtime_validation_receipt.sha256
        ),
    )
    _validate_admission_cross_bindings(config, process_spec, model, kernel)


def _require_processes_absent(
    manifest: JsonObject, measurement: WarmServingMeasurementV2
) -> None:
    runtime_metadata = validation.json_object(
        manifest.get("runtime_metadata"), "wrapper runtime metadata"
    )
    process_values = runtime_metadata.get("owned_processes")
    if not isinstance(process_values, list):
        raise Glm47ServingHarnessError("wrapper owned-process evidence is invalid")
    for raw in process_values:
        process = validation.json_object(raw, "wrapper owned process")
        process_id = process.get("pid")
        start_time_ticks = process.get("start_time_ticks")
        if type(process_id) is not int or type(start_time_ticks) is not int:
            raise Glm47ServingHarnessError("wrapper process identity is invalid")
        try:
            observed_start = int(validation.process_stat_fields(process_id)[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if observed_start == start_time_ticks:
            raise Glm47ServingHarnessError(
                f"owned benchmark process {process_id} remains after systemd cleanup"
            )
    if validation.owned_token_processes(measurement.owner_token) != {}:
        raise Glm47ServingHarnessError(
            "an owner-token process remains after systemd cleanup"
        )


def require_port_clear(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            raise Glm47ServingHarnessError(
                f"serving endpoint remains reachable at {host}:{port}"
            )


def require_gpu_clear(config: ServingBenchmarkConfig) -> None:
    stdout = run_bound_tool_text(
        config.tools.nvidia_smi,
        (
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        "nvidia-smi final GPU process probe",
    )
    selected = config.host.gpu.uuid.lower()
    for line in stdout.splitlines():
        gpu_uuid, separator, process_id = line.partition(",")
        if not separator or not process_id.strip().isdigit():
            raise Glm47ServingHarnessError(
                "nvidia-smi compute process output is invalid"
            )
        if gpu_uuid.strip().lower() == selected:
            raise Glm47ServingHarnessError(
                f"GPU process {process_id.strip()} remains on the selected GPU"
            )


def _require_unit_removed(config: ServingBenchmarkConfig, unit_name: str) -> None:
    stdout = run_bound_tool_text(
        config.tools.systemctl,
        ("show", unit_name, "--property=LoadState", "--value"),
        "systemctl transient unit probe",
    )
    if stdout.strip() != "not-found":
        raise Glm47ServingHarnessError(
            f"transient serving unit is not removed: {unit_name}"
        )


def acquire_finalization_lock(handoff: PreservedServingHandoff) -> None:
    try:
        observed = filesystem_object_identity(
            Path(handoff.identity.coordination_lock.path),
            handoff.lock_file.fileno(),
            "regular_file",
        )
        if observed != handoff.identity.coordination_lock:
            raise Glm47ServingHarnessError(
                "coordination lock changed across the wrapper handoff"
            )
        fcntl.flock(handoff.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise Glm47ServingHarnessError(
            "another benchmark won the post-lease finalization race"
        ) from error
    except OSError as error:
        raise Glm47ServingHarnessError("cannot acquire finalization lock") from error


def _require_terminal_absent(results: validation.ResultDirectory) -> None:
    try:
        os.stat(
            PERFORMANCE_RECEIPT_FILENAME,
            dir_fd=results.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise Glm47ServingHarnessError(
        "serving transaction already has an authoritative terminal result"
    )


def _terminal_failure_contents(
    config: ServingBenchmarkConfig, error: Exception
) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed_closed",
                "run_id": config.run_id,
                "namespace": config.namespace,
                "performance_comparable": False,
                "failed_at_utc": _utc_now().isoformat(timespec="microseconds"),
                "error": f"{type(error).__name__}: {error}",
                "retry_permitted": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_performance_receipt(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    *,
    lease_path: Path,
    handoff: PreservedServingHandoff,
) -> WarmServingRunReceiptV2:
    results = handoff.results
    try:
        os.lstat(lease_path)
    except FileNotFoundError:
        pass
    else:
        raise Glm47ServingHarnessError(
            "benchmark lease still exists during outer finalization"
        )
    observed_result = filesystem_object_identity(
        results.path, results.descriptor, "directory"
    )
    observed_lock = filesystem_object_identity(
        Path(handoff.identity.coordination_lock.path),
        handoff.lock_file.fileno(),
        "regular_file",
    )
    if (
        observed_result != handoff.identity.result_directory
        or observed_lock != handoff.identity.coordination_lock
    ):
        raise Glm47ServingHarnessError(
            "filesystem identity changed across the wrapper handoff"
        )
    snapshot = _load_measurement(results)
    measurement = snapshot.measurement
    if measurement.handoff != handoff.identity:
        raise Glm47ServingHarnessError(
            "lease child and outer wrapper observed different filesystem identities"
        )
    manifest = _validate_wrapper_manifest(config, deployment, results, snapshot)
    _validate_identity_files(config, deployment, measurement)
    _require_processes_absent(manifest, measurement)
    if os.path.lexists(measurement.delegated_cgroup_path):
        raise Glm47ServingHarnessError(
            "delegated serving cgroup remains after transient-unit cleanup"
        )
    if measurement.transient_unit_name != _systemd_unit_name(config):
        raise Glm47ServingHarnessError("measurement transient unit name changed")
    _require_unit_removed(config, measurement.transient_unit_name)
    stage = measurement.identity.topology.stages[0]
    require_port_clear(stage.host, stage.port)
    require_gpu_clear(config)

    cleanup_completed_at = _utc_now()
    cleanup = SglangKtServingCleanupEvidence(
        lease_id=measurement.lease_id,
        benchmark_completed_normally=True,
        server_process=measurement.server_process,
        termination_signal=measurement.termination_signal,
        server_return_code=measurement.server_return_code,
        forced=False,
        owned_processes_absent=True,
        service_host=stage.host,
        service_port=stage.port,
        service_port_clear=True,
        gpu_uuid=stage.gpu_uuid,
        gpu_process_clear=True,
        delegated_cgroup_path=measurement.delegated_cgroup_path,
        delegated_cgroup_removed=True,
        transient_unit_name=measurement.transient_unit_name,
        transient_unit_removed=True,
        cleanup_completed_at_utc=cleanup_completed_at.isoformat(
            timespec="microseconds"
        ),
        lease_cleanup_scope="outer_benchmark_wrapper",
    )
    generated_at = _utc_now()
    if generated_at <= cleanup_completed_at:
        generated_at = cleanup_completed_at + timedelta(microseconds=1)
    return WarmServingRunReceiptV2(
        schema_version=2,
        status="passed",
        generated_at_utc=generated_at.isoformat(timespec="microseconds"),
        evidence_class="performance",
        performance_comparable=True,
        profiler="none",
        instrumentation="none",
        radix_cache_disabled=True,
        max_concurrent_requests=1,
        measurement_sha256=snapshot.sha256,
        identity_sha256=measurement.identity_sha256,
        identity=measurement.identity,
        setup=measurement.setup,
        sanity=measurement.sanity,
        jit_cache=measurement.jit_cache,
        workloads=measurement.workloads,
        coordination_guard=measurement.coordination_guard,
        cleanup=cleanup,
    )


def finalize_serving_benchmark(
    config: ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    *,
    handoff: PreservedServingHandoff,
    lease_path: Path,
    wrapper_return_code: int,
) -> WarmServingRunReceiptV2:
    validation.require_no_profiler_state(os.environ, sys.argv, "")
    acquire_finalization_lock(handoff)
    try:
        _require_terminal_absent(handoff.results)
        try:
            if type(wrapper_return_code) is not int or wrapper_return_code != 0:
                raise Glm47ServingHarnessError(
                    f"leased systemd serving transaction returned {wrapper_return_code}"
                )
            receipt = _build_performance_receipt(
                config,
                deployment,
                lease_path=lease_path,
                handoff=handoff,
            )
        except Exception as error:
            try:
                write_new_bytes(
                    handoff.results,
                    PERFORMANCE_RECEIPT_FILENAME,
                    _terminal_failure_contents(config, error),
                )
            except Exception as commit_error:
                raise Glm47ServingHarnessError(
                    f"{error}; terminal failure could not be committed: "
                    f"{type(commit_error).__name__}: {commit_error}"
                ) from error
            raise
        receipt_contents = canonicalize_sglang_kt_warm_serving_run_receipt(
            receipt.model_dump(mode="json")
        )
        write_new_bytes(handoff.results, PERFORMANCE_RECEIPT_FILENAME, receipt_contents)
        return receipt
    finally:
        fcntl.flock(handoff.lock_file.fileno(), fcntl.LOCK_UN)


def prepare_serving_lease(
    *,
    config_path: Path,
    metadata_output: Path,
    owner: str,
    purpose: str,
    expected_duration_seconds: float,
    cleanup_grace_seconds: float,
    heartbeat_seconds: float,
    lease_path: Path,
    lock_path: Path,
    result_root: Path,
    now: Callable[[], datetime] = _utc_now,
) -> ServingLeasePreparation:
    for path in (
        config_path,
        metadata_output,
        lease_path,
        lock_path,
        result_root,
    ):
        if not path.is_absolute():
            raise Glm47ServingHarnessError(
                "all serving preparation paths must be absolute"
            )
    for value, description in (
        (expected_duration_seconds, "expected duration"),
        (cleanup_grace_seconds, "cleanup grace"),
        (heartbeat_seconds, "heartbeat"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise Glm47ServingHarnessError(
                f"serving {description} must be finite and positive"
            )
    if not owner.strip() or not purpose.strip() or "\0" in owner + purpose:
        raise Glm47ServingHarnessError("serving lease owner and purpose are invalid")

    config_contents = read_sglang_kt_bound_file(
        config_path, maximum_bytes=validation.MAXIMUM_JSON_BYTES
    ).contents
    config = _load_serving_config(config_path)
    execution = config.lease_execution
    if (
        str(metadata_output),
        owner,
        purpose,
        expected_duration_seconds,
        cleanup_grace_seconds,
        heartbeat_seconds,
        str(lease_path),
        str(lock_path),
        str(result_root),
    ) != (
        execution.metadata_output,
        execution.owner,
        execution.purpose,
        execution.expected_duration_seconds,
        execution.cleanup_grace_seconds,
        execution.heartbeat_seconds,
        execution.lease_path,
        execution.lock_path,
        execution.result_root,
    ):
        raise Glm47ServingHarnessError(
            "prepare arguments differ from the config-bound lease execution contract"
        )
    repository = Path(config.source.repository)
    if Path(__file__).resolve() != repository / SERVING_HARNESS_RELATIVE_PATH:
        raise Glm47ServingHarnessError(
            "prepare-lease must run from the configured mutable repository"
        )
    if Path(config.result_directory).parent != result_root:
        raise Glm47ServingHarnessError(
            "serving result directory must be a direct result-root child"
        )
    minimum_grace = validation.minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise Glm47ServingHarnessError(
            f"serving cleanup grace must be at least {minimum_grace} seconds"
        )
    for executable in (
        config.runtime_python.path,
        config.tools.numactl.path,
        config.tools.nvidia_smi.path,
        config.tools.systemctl.path,
        str(validation.SYSTEMD_RUN_EXECUTABLE),
    ):
        if not os.access(executable, os.X_OK):
            raise Glm47ServingHarnessError(
                f"required executable is unavailable: {executable}"
            )
    for binding, description in (
        (config.tools.numactl, "numactl"),
        (config.tools.nvidia_smi, "nvidia-smi"),
        (config.tools.systemctl, "systemctl"),
    ):
        bound_executable_identity(binding, description)

    source_identity = validation.read_source_identity(repository)
    deployment = validation.create_immutable_deployment(
        config, config_contents, source_identity
    )
    require_admitted_validator(config, deployment)
    for artifact, description in (
        (config.build_receipt, "runtime build receipt"),
        (config.model_contract, "deployed model contract"),
        (config.admission.model_runtime_validation_receipt, "model admission receipt"),
        (
            config.admission.kernel_runtime_validation_receipt,
            "kernel admission receipt",
        ),
        (config.admission.model_contract_receipt, "admitted model contract receipt"),
    ):
        validation.verify_artifact(artifact, description)
    if validation.read_source_identity(repository) != source_identity:
        raise Glm47ServingHarnessError(
            "source identity changed during immutable serving deployment"
        )

    immutable_config = Path(deployment.root) / validation.IMMUTABLE_CONFIG_RELATIVE_PATH
    child_argv = _serving_child_argv(config, lease_path=lease_path, lock_path=lock_path)
    generated_at = now().astimezone(timezone.utc)
    metadata = build_serving_static_metadata(
        config,
        child_argv,
        validation.config_sha256(immutable_config),
        deployment,
    )
    metadata["generated_at"] = generated_at.isoformat(timespec="seconds")
    benchmark_lease_script = (
        Path(deployment.root) / "orchestrator/scripts/benchmark_lease.py"
    )
    validation.validate_metadata(benchmark_lease_script, metadata, generated_at)
    systemd_argv = _prepared_systemd_argv(config, deployment)
    immutable_harness = (
        Path(deployment.root) / "orchestrator" / SERVING_HARNESS_RELATIVE_PATH
    )
    execute_argv = (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
        config.runtime_python.path,
        str(immutable_harness),
        "execute",
        "--config",
        str(immutable_config),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--",
        *systemd_argv,
    )
    validation.write_new_json(
        metadata_output, cast(Mapping[str, object], metadata), 0o444
    )
    return ServingLeasePreparation(
        metadata=metadata,
        metadata_output=str(metadata_output),
        deployment=deployment,
        child_argv=child_argv,
        systemd_argv=systemd_argv,
        execute_argv=execute_argv,
        systemd_unit_name=_systemd_unit_name(config),
    )


class RunArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path | None


class PrepareArguments(argparse.Namespace):
    config: Path
    metadata_output: Path
    owner: str
    purpose: str
    expected_duration_seconds: float
    cleanup_grace_seconds: float
    heartbeat_seconds: float
    lease_path: Path
    lock_path: Path
    result_root: Path


class ExecuteArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    command: list[str]


class FinalizeArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path


def _run_parser(arguments: Sequence[str]) -> RunArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--lease-path", type=Path, default=validation.DEFAULT_LEASE_PATH
    )
    parser.add_argument("--lock-path", type=Path, default=validation.DEFAULT_LOCK_PATH)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args(arguments, namespace=RunArguments())


def _prepare_parser(arguments: Sequence[str]) -> PrepareArguments:
    parser = argparse.ArgumentParser(
        description="Prepare one immutable leased GLM-4.7 serving benchmark"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-duration-seconds", required=True, type=float)
    parser.add_argument("--cleanup-grace-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--lease-path", type=Path, default=validation.DEFAULT_LEASE_PATH
    )
    parser.add_argument("--lock-path", type=Path, default=validation.DEFAULT_LOCK_PATH)
    parser.add_argument(
        "--result-root", type=Path, default=validation.DEFAULT_RESULT_ROOT
    )
    return parser.parse_args(arguments, namespace=PrepareArguments())


def _execute_parser(arguments: Sequence[str]) -> ExecuteArguments:
    parser = argparse.ArgumentParser(
        description="Execute the transient service and finalize after collection"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--lease-path", type=Path, default=validation.DEFAULT_LEASE_PATH
    )
    parser.add_argument("--lock-path", type=Path, default=validation.DEFAULT_LOCK_PATH)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(arguments, namespace=ExecuteArguments())
    if parsed.command and parsed.command[0] == "--":
        parsed.command = parsed.command[1:]
    return parsed


def _finalize_parser(arguments: Sequence[str]) -> FinalizeArguments:
    parser = argparse.ArgumentParser(
        description="Finalize an already collected GLM-4.7 serving transaction"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--lease-path", type=Path, default=validation.DEFAULT_LEASE_PATH
    )
    parser.add_argument("--lock-path", type=Path, default=validation.DEFAULT_LOCK_PATH)
    return parser.parse_args(arguments, namespace=FinalizeArguments())


def prepare_main(arguments: Sequence[str]) -> int:
    parsed = _prepare_parser(arguments)
    prepared = prepare_serving_lease(
        config_path=parsed.config,
        metadata_output=parsed.metadata_output,
        owner=parsed.owner,
        purpose=parsed.purpose,
        expected_duration_seconds=parsed.expected_duration_seconds,
        cleanup_grace_seconds=parsed.cleanup_grace_seconds,
        heartbeat_seconds=parsed.heartbeat_seconds,
        lease_path=parsed.lease_path,
        lock_path=parsed.lock_path,
        result_root=parsed.result_root,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "metadata_output": prepared.metadata_output,
                "deployment": asdict(prepared.deployment),
                "child_argv": list(prepared.child_argv),
                "systemd_argv": list(prepared.systemd_argv),
                "execute_argv": list(prepared.execute_argv),
                "systemd_unit_name": prepared.systemd_unit_name,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _load_immutable_context(
    config_path: Path,
) -> tuple[ServingBenchmarkConfig, validation.DeploymentIdentity]:
    config = _load_serving_config(config_path)
    expected_config = (
        Path(config.source.deployment_root) / validation.IMMUTABLE_CONFIG_RELATIVE_PATH
    )
    if config_path != expected_config:
        raise Glm47ServingHarnessError(
            "serving config is not the immutable deployed config"
        )
    deployment = validation.load_deployment_identity(
        Path(config.source.deployment_root)
    )
    return config, deployment


def run_main(arguments: Sequence[str]) -> int:
    parsed = _run_parser(arguments)
    if not all(
        path.is_absolute()
        for path in (parsed.config, parsed.lease_path, parsed.lock_path)
    ):
        raise Glm47ServingHarnessError("serving run paths must be absolute")
    config, deployment = _load_immutable_context(parsed.config)
    if (
        str(parsed.lease_path),
        str(parsed.lock_path),
    ) != (
        config.lease_execution.lease_path,
        config.lease_execution.lock_path,
    ):
        raise Glm47ServingHarnessError(
            "serving child paths differ from immutable lease authorization"
        )
    if (
        parsed.result_dir is not None
        and str(parsed.result_dir) != config.result_directory
    ):
        raise Glm47ServingHarnessError("--result-dir differs from immutable config")
    results = validation.ResultDirectory.inherited(Path(config.result_directory))
    latch = validation.SignalLatch()
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    try:
        lease_record, handoff = validate_active_serving_lease(
            config,
            deployment,
            results,
            config_path=parsed.config,
            lease_path=parsed.lease_path,
            lock_path=parsed.lock_path,
        )
        validation.require_no_profiler_state(os.environ, sys.argv, "")
        if (
            os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or not sys.dont_write_bytecode
        ):
            raise Glm47ServingHarnessError(
                "immutable serving execution requires bytecode writes disabled"
            )
        for managed in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[managed] = signal.getsignal(managed)
            signal.signal(managed, latch.handle)
        result = run_serving_benchmark(
            config,
            deployment,
            results,
            latch,
            handoff,
            cast(str, lease_record["lease_id"]),
        )
        return 0 if result.get("status") == "completed" else 2
    finally:
        for managed, previous in previous_handlers.items():
            signal.signal(managed, previous)
        results.close()


def execute_main(arguments: Sequence[str]) -> int:
    parsed = _execute_parser(arguments)
    if not all(
        path.is_absolute()
        for path in (parsed.config, parsed.lease_path, parsed.lock_path)
    ):
        raise Glm47ServingHarnessError("serving execute paths must be absolute")
    if not parsed.command:
        raise Glm47ServingHarnessError("execute requires the prepared systemd command")
    config, deployment = _load_immutable_context(parsed.config)
    if (
        str(parsed.lease_path),
        str(parsed.lock_path),
    ) != (
        config.lease_execution.lease_path,
        config.lease_execution.lock_path,
    ):
        raise Glm47ServingHarnessError(
            "execute paths differ from immutable lease authorization"
        )
    expected_command = _prepared_systemd_argv(config, deployment)
    if tuple(parsed.command) != expected_command:
        raise Glm47ServingHarnessError(
            "execute command differs from the exact prepared systemd transaction"
        )
    _validate_prepared_metadata(config, deployment, parsed.config)
    validation.require_no_profiler_state(os.environ, parsed.command, "")
    return_code, handoff = run_systemd_with_preserved_handoff(
        config, expected_command, lock_path=parsed.lock_path
    )
    try:
        receipt = finalize_serving_benchmark(
            config,
            deployment,
            handoff=handoff,
            lease_path=parsed.lease_path,
            wrapper_return_code=return_code,
        )
    finally:
        handoff.close()
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "receipt": str(
                    Path(config.result_directory) / PERFORMANCE_RECEIPT_FILENAME
                ),
                "identity_sha256": receipt.identity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def finalize_main(arguments: Sequence[str]) -> int:
    parsed = _finalize_parser(arguments)
    if not all(
        path.is_absolute()
        for path in (parsed.config, parsed.lease_path, parsed.lock_path)
    ):
        raise Glm47ServingHarnessError("serving finalizer paths must be absolute")
    raise Glm47ServingHarnessError(
        "standalone finalization cannot recover preserved wrapper descriptors"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if arguments is None else arguments)
    if normalized and normalized[0] == "prepare-lease":
        return prepare_main(normalized[1:])
    if normalized and normalized[0] == "execute":
        return execute_main(normalized[1:])
    if normalized and normalized[0] == "finalize":
        return finalize_main(normalized[1:])
    return run_main(normalized)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        Glm47ServingHarnessError,
        validation.Glm47HarnessError,
        ValidationError,
        ValueError,
        OSError,
    ) as error:
        print(
            f"GLM-4.7 serving benchmark failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
