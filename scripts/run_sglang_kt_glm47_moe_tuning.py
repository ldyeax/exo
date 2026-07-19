#!/usr/bin/env python3
"""Prepare and run one leased GLM-4.7 fused-MoE tuning transaction.

``prepare-lease`` creates a read-only source capsule and metadata for
``benchmark_lease.py``.  The default mode is the lease child: it performs
fail-closed local preflight, starts the tuner as an owned child, and publishes
cleanup evidence only after every tagged child has exited.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import re
import shlex
import signal
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType, ModuleType
from typing import Final, Literal, TextIO, TypeAlias, TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_LOCK_PATH: Final = Path("/var/lock/fwuffydwagon-benchmark.lock")
DEFAULT_LEASE_PATH: Final = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_RESULT_ROOT: Final = Path("/var/lib/exo/benchmarks")
RESULT_DIRECTORY_FD_ENVIRONMENT: Final = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
TUNER_AUTHORIZATION_FD_ENVIRONMENT: Final = "EXO_GLM47_TUNER_AUTHORIZATION_FD"
TUNER_RESULT_DIRECTORY_FD_ENVIRONMENT: Final = "EXO_GLM47_TUNER_RESULT_DIRECTORY_FD"
TUNER_OUTPUT_DIRECTORY_FD_ENVIRONMENT: Final = "EXO_GLM47_TUNER_OUTPUT_DIRECTORY_FD"
TUNER_OWNER_TOKEN_ENVIRONMENT: Final = "EXO_GLM47_TUNER_OWNER_TOKEN"
BENCHMARK_RESULT_FILENAME: Final = "benchmark-result.json"
RUNTIME_METADATA_FILENAME: Final = "runtime-metadata.json"
AUTHORIZATION_FILENAME: Final = "tuner-authorization.json"
COMPLETION_FILENAME: Final = "tuning-completion.json"
PREFLIGHT_FILENAME: Final = "preflight.json"
TELEMETRY_BEFORE_FILENAME: Final = "telemetry-before.json"
TELEMETRY_AFTER_FILENAME: Final = "telemetry-after.json"
TUNER_STDOUT_FILENAME: Final = "tuner.stdout.log"
TUNER_STDERR_FILENAME: Final = "tuner.stderr.log"
TUNING_OUTPUT_DIRECTORY_NAME: Final = "tuning-bundle"
SOURCE_IDENTITY_FILENAME: Final = "source-identity.json"
IMMUTABLE_CONFIG_FILENAME: Final = "tuning-config.json"
SOURCE_FILES: Final = (
    Path("scripts/benchmark_lease.py"),
    Path("scripts/run_sglang_kt_glm47_moe_tuning.py"),
    Path("scripts/tune_sglang_kt_glm47_moe.py"),
)
TUNER_RELATIVE_PATH: Final = Path("scripts/tune_sglang_kt_glm47_moe.py")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
MAXIMUM_CONFIG_JSON_BYTES: Final = 256 * 1024
MAXIMUM_AUTHORIZATION_BYTES: Final = 64 * 1024
MAXIMUM_RECEIPT_JSON_BYTES: Final = 4 * 1024 * 1024
MAXIMUM_MANIFEST_JSON_BYTES: Final = 32 * 1024 * 1024
MAXIMUM_PATH_LENGTH: Final = 4_096
MAXIMUM_STRING_LENGTH: Final = 1_024
MAXIMUM_BATCH_ANCHORS: Final = 16
MAXIMUM_BATCH_SIZE: Final = 65_536
MAXIMUM_CPU_CORES: Final = 2_048
MAXIMUM_MEMORY_NODES: Final = 64
MAXIMUM_HCA_BINDINGS: Final = 16
MAXIMUM_RESERVED_PORTS: Final = 32
MAXIMUM_SYMLINK_HOPS: Final = 16
GLM47_HIDDEN_SIZE: Final = 2_048
GLM47_INTERMEDIATE_SIZE: Final = 1_536
GLM47_GLOBAL_EXPERTS: Final = 64
GLM47_TOP_K: Final = 4
MINIMUM_SERVING_IMPROVEMENT: Final = 0.03
MINIMUM_SYNTHETIC_IMPROVEMENT: Final = 0.05
MAXIMUM_RELATIVE_L1_ERROR: Final = 0.02
MAXIMUM_ABSOLUTE_ERROR: Final = 0.02
_BANNED_PROFILER_TOKENS: Final = ("sep5", "pax")
_CONFLICTING_EXECUTABLES: Final = frozenset(
    {
        "all_gather_perf",
        "all_reduce_perf",
        "alltoall_perf",
        "amplxe-cl",
        "exo",
        "ib_read_bw",
        "ib_send_bw",
        "ib_write_bw",
        "llama-server",
        "nccl-tests",
        "ncu",
        "nsys",
        "ollama",
        "text-generation-launcher",
        "uvicorn",
        "vtune",
    }
)
_CONFLICTING_COMMAND_FRAGMENTS: Final = (
    "sglang.launch_server",
    "vllm.entrypoints",
    "scripts/tune_sglang_kt_glm47_moe.py",
    "python -m exo",
    "uv run exo",
)
_STORAGE_COMMAND_FRAGMENTS: Final = (
    "hf download",
    "huggingface-cli download",
    "snapshot_download",
    "aria2c",
    "b2sum",
    "b3sum",
    "mdadm --action=check",
    "rclone",
    "rsync",
    "sha256sum",
    "zpool scrub",
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class LeaseMetadataValidator(ModuleType):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


class TunerSearchSpaceModule(ModuleType):
    def build_rtx3090_search_space(
        self, profile: Literal["quick", "balanced"]
    ) -> Sequence[Mapping[str, int]]: ...


class HarnessError(RuntimeError):
    """Raised when a tuning run cannot be proven safe or reproducible."""


@dataclass
class ManagedSignalState:
    first_signal_number: int | None = None

    def record(self, signal_number: int) -> None:
        if self.first_signal_number is None:
            self.first_signal_number = signal_number

    def raise_if_interrupted(self) -> None:
        if self.first_signal_number is not None:
            raise HarnessError(
                f"tuning harness received signal {self.first_signal_number}"
            )


@dataclass(frozen=True)
class ValidatedStageMeasurement:
    evidence: JsonObject
    config: JsonObject
    median_microseconds: float
    stably_faster: bool


class FileBinding(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    path: str = Field(min_length=1, max_length=MAXIMUM_PATH_LENGTH)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if "\0" in value or not Path(value).is_absolute():
            raise ValueError("path must be absolute")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return value


class ExecutableBinding(FileBinding):
    symlink_chain: tuple[str, ...] = Field(max_length=MAXIMUM_SYMLINK_HOPS)

    @field_validator("symlink_chain")
    @classmethod
    def validate_symlink_chain(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not target or len(target) > MAXIMUM_PATH_LENGTH for target in value):
            raise ValueError("symlink targets must be nonempty and bounded")
        if any("\0" in target for target in value):
            raise ValueError("symlink targets must be NUL-free")
        return value


class RuntimeReceiptBindings(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    install: FileBinding
    build: FileBinding
    kernel_validation: FileBinding


class ContextualModelBinding(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str = Field(min_length=1, max_length=MAXIMUM_STRING_LENGTH)
    revision: str
    path: str = Field(min_length=1, max_length=MAXIMUM_PATH_LENGTH)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("revision must be an exact commit hash")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("path must be absolute")
        return value


class GpuBinding(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    uuid: str = Field(min_length=1, max_length=128)
    pci_address: str = Field(min_length=1, max_length=128)
    expected_name: str = Field(min_length=1, max_length=256)


class HcaBinding(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    device: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    port: int = Field(gt=0, le=255)
    gid: str = Field(max_length=128)

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if address.version != 6 or address.is_unspecified:
            raise ValueError("gid must be a non-unspecified IPv6 address")
        return value


class HostBinding(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    hostname: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    gpu: GpuBinding
    cpu_cores: tuple[int, ...] = Field(max_length=MAXIMUM_CPU_CORES)
    memory_nodes: tuple[int, ...] = Field(max_length=MAXIMUM_MEMORY_NODES)
    hca_bindings: tuple[HcaBinding, ...] = Field(max_length=MAXIMUM_HCA_BINDINGS)

    @model_validator(mode="after")
    def validate_resources(self) -> HostBinding:
        if not self.cpu_cores or min(self.cpu_cores) < 0:
            raise ValueError("cpu_cores must contain nonnegative CPU identifiers")
        if len(set(self.cpu_cores)) != len(self.cpu_cores):
            raise ValueError("cpu_cores must not contain duplicates")
        if not self.memory_nodes or min(self.memory_nodes) < 0:
            raise ValueError("memory_nodes must contain nonnegative identifiers")
        if len(set(self.memory_nodes)) != len(self.memory_nodes):
            raise ValueError("memory_nodes must not contain duplicates")
        if not self.hca_bindings:
            raise ValueError("hca_bindings must not be empty")
        return self


class TuningSpec(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    resident_experts: int = Field(gt=0, le=GLM47_GLOBAL_EXPERTS - GLM47_TOP_K)
    batch_sizes: tuple[int, ...] = Field(
        default=(1, 8, 32, 128, 512, 1_024, 4_096),
        max_length=MAXIMUM_BATCH_ANCHORS,
    )
    warmup_iterations: int = Field(gt=0, le=10_000)
    measurement_iterations: int = Field(gt=0, le=100_000)
    independent_samples: int = Field(ge=3, le=100)
    search_profile: Literal["quick", "balanced"]
    seed: int = Field(ge=0, le=2**63 - 1)
    expected_sglang_revision: str
    expected_ktransformers_revision: str
    expected_torch_version: str = Field(min_length=1, max_length=128)
    expected_triton_version: str = Field(min_length=1, max_length=128)

    @field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(size <= 0 or size > MAXIMUM_BATCH_SIZE for size in value):
            raise ValueError("batch_sizes must contain bounded positive integers")
        if tuple(sorted(set(value))) != value:
            raise ValueError("batch_sizes must be unique and increasing")
        return value

    @field_validator("expected_sglang_revision", "expected_ktransformers_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("runtime revision must be an exact commit hash")
        return value


class TimeoutSpec(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    tuning_seconds: float = Field(gt=0, le=7 * 24 * 60 * 60)
    cleanup_seconds: float = Field(gt=0, le=10 * 60)

    @model_validator(mode="after")
    def validate_finite(self) -> TimeoutSpec:
        if not math.isfinite(self.tuning_seconds) or not math.isfinite(
            self.cleanup_seconds
        ):
            raise ValueError("timeouts must be finite")
        return self


class TuningRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[3]
    run_id: str = Field(max_length=128)
    namespace: str = Field(max_length=128)
    profiler: Literal["none"]
    result_directory: str = Field(max_length=MAXIMUM_PATH_LENGTH)
    source_repository: str = Field(max_length=MAXIMUM_PATH_LENGTH)
    source_deployment_root: str = Field(max_length=MAXIMUM_PATH_LENGTH)
    runtime_python: ExecutableBinding
    runtime_receipts: RuntimeReceiptBindings
    numactl_executable: FileBinding
    contextual_model: ContextualModelBinding
    host: HostBinding
    reserved_ports: tuple[int, ...] = Field(max_length=MAXIMUM_RESERVED_PORTS)
    tuning: TuningSpec
    timeouts: TimeoutSpec

    @field_validator("run_id", "namespace")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError("value is not a valid identifier")
        return value

    @field_validator("result_directory", "source_repository", "source_deployment_root")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if "\0" in value or not Path(value).is_absolute():
            raise ValueError("path must be absolute")
        return value

    @field_validator("reserved_ports")
    @classmethod
    def validate_ports(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(port < 1 or port > 65_535 for port in value):
            raise ValueError("reserved_ports must contain valid ports")
        if len(set(value)) != len(value):
            raise ValueError("reserved_ports must not contain duplicates")
        return value


class ProcessIdentity(TypedDict):
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    transport_pid: int
    namespace: str
    owner_token: str
    log_path: str


class CompletedCommand(TypedDict):
    argv: list[str]
    return_code: int
    stdout: str
    stderr: str


class LeasePreparation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    metadata_path: str
    source_deployment_root: str
    source_commit: str
    source_file_sha256: dict[str, str]
    benchmark_lease_argv: tuple[str, ...]


def _canonical_json_bytes(value: object, *, pretty: bool = True) -> bytes:
    separators = None if pretty else (",", ":")
    rendered = json.dumps(
        value,
        indent=2 if pretty else None,
        separators=separators,
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def _json_object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise HarnessError(f"{description} must be a JSON object")
    return cast(JsonObject, value)


def _read_bounded_regular_file(
    path: Path, description: str, maximum_bytes: int
) -> bytes:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size > maximum_bytes:
            raise HarnessError(f"{description} is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            contents = source.read(maximum_bytes + 1)
    except OSError as error:
        raise HarnessError(f"cannot read {description}: {error}") from error
    if len(contents) > maximum_bytes:
        raise HarnessError(f"{description} exceeds its size limit")
    return contents


def _read_json(
    path: Path,
    description: str,
    maximum_bytes: int = MAXIMUM_RECEIPT_JSON_BYTES,
) -> JsonObject:
    try:
        return _json_object(
            cast(
                object,
                json.loads(
                    _read_bounded_regular_file(path, description, maximum_bytes)
                ),
            ),
            description,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {description}: {error}") from error


def _write_new_file(path: Path, contents: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _atomic_write_json_at(
    directory_descriptor: int, filename: str, value: Mapping[str, object]
) -> None:
    if Path(filename).name != filename:
        raise HarnessError("result filename must be a basename")
    temporary_name = f".{filename}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(_canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.rename(
            temporary_name,
            filename,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_descriptor)


def _make_read_only_at(directory_descriptor: int, filename: str) -> None:
    os.chmod(
        filename,
        0o444,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _open_new_log_at(directory_descriptor: int, filename: str) -> tuple[TextIO, str]:
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    return os.fdopen(descriptor, "w", encoding="utf-8"), filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise HarnessError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_file_at(directory_descriptor: int, filename: str) -> str:
    if Path(filename).name != filename:
        raise HarnessError("descriptor-relative filename must be a basename")
    descriptor = os.open(
        filename,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise HarnessError(f"{filename} is not a regular file")
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _create_directory_at(directory_descriptor: int, name: str) -> int:
    if Path(name).name != name or name in {".", ".."}:
        raise HarnessError("anchored directory name must be a safe basename")
    try:
        os.mkdir(name, mode=0o700, dir_fd=directory_descriptor)
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise HarnessError(
            f"cannot create anchored directory {name}: {error}"
        ) from error
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        os.close(descriptor)
        raise HarnessError(f"anchored path {name} is not a directory")
    return descriptor


def _directory_identity(descriptor: int) -> JsonObject:
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise HarnessError("inherited descriptor is not a directory")
    return {"device": status.st_dev, "inode": status.st_ino}


def _verify_file(binding: FileBinding, description: str) -> None:
    path = Path(binding.path)
    if not path.is_file() or path.is_symlink():
        raise HarnessError(f"{description} must be a regular non-symlink file")
    observed = _sha256_file(path)
    if observed != binding.sha256:
        raise HarnessError(
            f"{description} hash mismatch: expected {binding.sha256}, got {observed}"
        )


def _verify_executable(binding: ExecutableBinding, description: str) -> None:
    current = Path(binding.path)
    for expected_target in binding.symlink_chain:
        try:
            status = current.lstat()
            if not stat.S_ISLNK(status.st_mode):
                raise HarnessError(f"{description} symlink chain changed")
            observed_target = os.readlink(current)
        except OSError as error:
            raise HarnessError(
                f"cannot inspect {description} symlink: {error}"
            ) from error
        if observed_target != expected_target:
            raise HarnessError(f"{description} symlink target changed")
        target = Path(expected_target)
        current = target if target.is_absolute() else current.parent / target
    try:
        final_status = current.lstat()
    except OSError as error:
        raise HarnessError(
            f"{description} final target is unavailable: {error}"
        ) from error
    if (
        not stat.S_ISREG(final_status.st_mode)
        or stat.S_ISLNK(final_status.st_mode)
        or _sha256_file(current) != binding.sha256
        or not os.access(binding.path, os.X_OK)
    ):
        raise HarnessError(f"{description} final target identity changed")


def _required_json_list(value: JsonValue | None, description: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise HarnessError(f"{description} must be a JSON array")
    return value


def _required_bounded_string(value: JsonValue | None, description: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAXIMUM_PATH_LENGTH:
        raise HarnessError(f"{description} must be a bounded nonempty string")
    return value


def _verify_distribution_record(
    *, install_root: Path, metadata_path: Path, expected_sha256: str
) -> int:
    try:
        resolved_root = install_root.resolve(strict=True)
        resolved_metadata = metadata_path.resolve(strict=True)
    except OSError as error:
        raise HarnessError(f"runtime install path is unavailable: {error}") from error
    if not resolved_metadata.is_relative_to(resolved_root):
        raise HarnessError("installed distribution metadata escapes install root")
    record_path = resolved_metadata.parent / "RECORD"
    if _sha256_file(record_path) != expected_sha256:
        raise HarnessError("installed distribution RECORD hash changed")
    site_packages = resolved_metadata.parent.parent
    rows = _read_bounded_regular_file(
        record_path, "installed distribution RECORD", MAXIMUM_RECEIPT_JSON_BYTES
    ).decode("utf-8")
    file_count = 0
    for row in csv.reader(rows.splitlines()):
        if len(row) != 3 or not row[0]:
            raise HarnessError("installed distribution RECORD row is invalid")
        recorded_path, recorded_hash, recorded_size = row
        path_parts = Path(recorded_path).parts
        if path_parts[:3] == ("..", "..", "bin"):
            lexical_candidate = site_packages / "bin" / Path(*path_parts[3:])
        else:
            lexical_candidate = site_packages / recorded_path
        candidate = lexical_candidate.resolve(strict=True)
        lexical_status = lexical_candidate.lstat()
        if (
            not candidate.is_relative_to(resolved_root)
            or stat.S_ISLNK(lexical_status.st_mode)
            or not stat.S_ISREG(lexical_status.st_mode)
        ):
            raise HarnessError("installed distribution file escapes install root")
        if recorded_hash:
            algorithm, separator, encoded_digest = recorded_hash.partition("=")
            if algorithm != "sha256" or not separator or not encoded_digest:
                raise HarnessError("installed distribution RECORD hash is unsupported")
            padding = "=" * (-len(encoded_digest) % 4)
            try:
                expected_digest = base64.urlsafe_b64decode(
                    encoded_digest + padding
                ).hex()
            except (ValueError, TypeError) as error:
                raise HarnessError("installed RECORD digest is invalid") from error
            if _sha256_file(candidate) != expected_digest:
                raise HarnessError(f"installed runtime file changed: {recorded_path}")
        if recorded_size:
            try:
                expected_size = int(recorded_size)
            except ValueError as error:
                raise HarnessError("installed RECORD size is invalid") from error
            if candidate.stat().st_size != expected_size:
                raise HarnessError(
                    f"installed runtime file size changed: {recorded_path}"
                )
        file_count += 1
    if file_count == 0:
        raise HarnessError("installed distribution RECORD is empty")
    return file_count


def _pip_freeze_version(lines: Sequence[str], distribution: str) -> str:
    prefix = f"{distribution}=="
    matches = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise HarnessError(f"install receipt does not pin {distribution}")
    return matches[0]


def _required_sha256(value: JsonValue | None, description: str) -> str:
    digest = _required_bounded_string(value, description)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise HarnessError(f"{description} is not a SHA-256 digest")
    return digest


def _required_revision(value: JsonValue | None, description: str) -> str:
    revision = _required_bounded_string(value, description)
    if _COMMIT_PATTERN.fullmatch(revision) is None:
        raise HarnessError(f"{description} is not an exact revision")
    return revision


def _artifact_digest_map(
    values: Sequence[JsonValue], description: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        artifact = _json_object(value, description)
        distribution = _required_bounded_string(
            artifact.get("distribution"), f"{description} distribution"
        )
        digest = _required_sha256(
            artifact.get("sha256"), f"{description} {distribution} digest"
        )
        if distribution in result:
            raise HarnessError(f"{description} contains duplicate {distribution}")
        result[distribution] = digest
    if not result:
        raise HarnessError(f"{description} is empty")
    return result


def _verified_runtime_contract(config: TuningRunConfig) -> JsonObject:
    receipts = config.runtime_receipts
    _verify_executable(config.runtime_python, "runtime Python")
    for binding, description in (
        (receipts.install, "runtime install receipt"),
        (receipts.build, "runtime build receipt"),
        (receipts.kernel_validation, "kernel validation receipt"),
    ):
        _verify_file(binding, description)

    install = _read_json(Path(receipts.install.path), "runtime install receipt")
    if (
        install.get("schema_version") != 1
        or install.get("status") != "install_complete"
    ):
        raise HarnessError("runtime install receipt is not complete schema 1")
    install_id = _required_sha256(install.get("install_id"), "runtime install id")
    install_layout = _json_object(install.get("layout"), "runtime install layout")
    install_root = Path(
        _required_bounded_string(
            install_layout.get("install_root"), "runtime install root"
        )
    )
    if (
        not install_root.is_absolute()
        or install_root.name != install_id
        or install_root.is_symlink()
        or not install_root.is_dir()
        or install_layout.get("python") != config.runtime_python.path
        or install_layout.get("receipt") != receipts.install.path
    ):
        raise HarnessError("runtime install layout does not bind this install")

    install_build = _json_object(install.get("build"), "runtime install build")
    build_id = _required_sha256(install_build.get("build_id"), "runtime build id")
    sglang_revision = _required_revision(
        install_build.get("sglang_revision"), "installed SGLang revision"
    )
    ktransformers_revision = _required_revision(
        install_build.get("ktransformers_revision"),
        "installed KTransformers revision",
    )
    package_version = _required_bounded_string(
        install_build.get("package_version"), "runtime package version"
    )
    if (
        install_build.get("receipt_path") != receipts.build.path
        or install_build.get("receipt_sha256") != receipts.build.sha256
        or sglang_revision != config.tuning.expected_sglang_revision
        or ktransformers_revision != config.tuning.expected_ktransformers_revision
    ):
        raise HarnessError("runtime install build binding differs from tuning config")
    install_wheels = _artifact_digest_map(
        _required_json_list(install_build.get("wheels"), "installed build wheels"),
        "installed build wheel",
    )

    base_runtime = _json_object(install.get("base_runtime"), "base runtime")
    base_python_sha256 = _required_sha256(
        base_runtime.get("python_sha256"), "base runtime Python digest"
    )
    if base_python_sha256 != config.runtime_python.sha256:
        raise HarnessError("base runtime Python differs from configured runtime")
    base_pip_freeze_sha256 = _required_sha256(
        base_runtime.get("pip_freeze_sha256"), "base runtime package-set digest"
    )
    raw_freeze = _required_json_list(base_runtime.get("pip_freeze"), "pip freeze")
    if len(raw_freeze) > 4_096 or not all(isinstance(line, str) for line in raw_freeze):
        raise HarnessError("runtime pip freeze is not a bounded string list")
    pip_freeze = cast(list[str], cast(object, raw_freeze))
    if hashlib.sha256(("\n".join(pip_freeze) + "\n").encode()).hexdigest() != (
        base_pip_freeze_sha256
    ):
        raise HarnessError("runtime pip freeze digest differs from receipt")
    triton_version = _pip_freeze_version(pip_freeze, "triton")
    if triton_version != config.tuning.expected_triton_version:
        raise HarnessError("installed Triton version differs from tuning config")

    installed_distributions = _required_json_list(
        install.get("installed_distributions"), "installed distributions"
    )
    if not installed_distributions or len(installed_distributions) > 32:
        raise HarnessError("installed distribution evidence is not bounded")
    installed_record_sha256: dict[str, str] = {}
    installed_file_counts: dict[str, int] = {}
    for value in installed_distributions:
        distribution = _json_object(value, "installed distribution")
        name = _required_bounded_string(
            distribution.get("distribution"), "installed distribution name"
        )
        record_sha256 = _required_sha256(
            distribution.get("record_sha256"), f"{name} RECORD digest"
        )
        metadata_path = Path(
            _required_bounded_string(
                distribution.get("metadata_path"), f"{name} metadata path"
            )
        )
        if name in installed_record_sha256:
            raise HarnessError(f"duplicate installed distribution {name}")
        installed_record_sha256[name] = record_sha256
        installed_file_counts[name] = _verify_distribution_record(
            install_root=install_root,
            metadata_path=metadata_path,
            expected_sha256=record_sha256,
        )
    if set(installed_record_sha256) != {"kt-kernel", "ktransformers", "sglang-kt"}:
        raise HarnessError("runtime overlay distribution set is unexpected")

    build = _read_json(Path(receipts.build.path), "runtime build receipt")
    build_source = _json_object(build.get("source"), "runtime build source")
    build_layout = _json_object(build.get("layout"), "runtime build layout")
    if (
        build.get("schema_version") != 2
        or build.get("status") != "wheel_build_complete"
        or build.get("build_id") != build_id
        or build_layout.get("receipt") != receipts.build.path
        or build_source.get("sglang_revision") != sglang_revision
        or build_source.get("ktransformers_revision") != ktransformers_revision
        or build_source.get("package_version") != package_version
    ):
        raise HarnessError("runtime build receipt differs from installed build")
    build_wheel_values = _required_json_list(
        build.get("runtime_wheels"), "runtime build wheels"
    )
    build_wheels = _artifact_digest_map(build_wheel_values, "runtime build wheel")
    if build_wheels != install_wheels:
        raise HarnessError("built and installed wheel identities differ")
    build_root = Path(receipts.build.path).parent
    for value in build_wheel_values:
        wheel = _json_object(value, "runtime build wheel")
        wheel_path = Path(
            _required_bounded_string(wheel.get("path"), "runtime wheel path")
        )
        if not wheel_path.is_absolute():
            wheel_path = build_root / wheel_path
        binding = FileBinding(
            path=str(wheel_path),
            sha256=_required_sha256(wheel.get("sha256"), "runtime wheel digest"),
        )
        _verify_file(binding, "runtime wheel")

    kernel = _read_json(
        Path(receipts.kernel_validation.path), "kernel validation receipt"
    )
    failures = _required_json_list(kernel.get("failures"), "kernel validation failures")
    capabilities_values = _required_json_list(
        kernel.get("capabilities"), "kernel validation capabilities"
    )
    if not all(isinstance(value, str) for value in capabilities_values):
        raise HarnessError("kernel validation capabilities are invalid")
    capabilities = cast(list[str], cast(object, capabilities_values))
    kernel_config = _json_object(kernel.get("config"), "kernel validation config")
    provenance = _json_object(kernel.get("provenance"), "kernel provenance")
    runtime_identity = _json_object(
        kernel.get("runtime_identity"), "kernel runtime identity"
    )
    cuda = _json_object(kernel.get("cuda"), "kernel CUDA identity")
    if (
        kernel.get("schema_version") != 1
        or kernel.get("status") != "passed"
        or failures
        or kernel.get("profiler") != "none"
        or "kt_bf16_amx_executed_v1" not in capabilities
        or kernel_config.get("build_receipt_path") != receipts.build.path
        or kernel_config.get("gpu_uuid") != config.host.gpu.uuid
        or provenance.get("verified") is not True
        or provenance.get("build_id") != build_id
        or provenance.get("receipt_path") != receipts.build.path
        or provenance.get("receipt_sha256") != receipts.build.sha256
        or provenance.get("sglang_revision") != sglang_revision
        or provenance.get("ktransformers_revision") != ktransformers_revision
        or cuda.get("gpu_uuid") != config.host.gpu.uuid
        or cuda.get("gpu_name") != config.host.gpu.expected_name
        or cuda.get("compute_capability") != [8, 6]
    ):
        raise HarnessError("kernel validation receipt does not bind this runtime/GPU")
    if (
        _artifact_digest_map(
            _required_json_list(provenance.get("wheels"), "kernel provenance wheels"),
            "kernel provenance wheel",
        )
        != build_wheels
    ):
        raise HarnessError("kernel validation wheel identities differ from build")
    torch_version = _required_bounded_string(
        runtime_identity.get("torch_module_version"), "runtime Torch version"
    )
    torch_cuda_version = _required_bounded_string(
        runtime_identity.get("torch_cuda_version"), "runtime Torch CUDA version"
    )
    if (
        torch_version != config.tuning.expected_torch_version
        or cuda.get("torch_cuda_version") != torch_cuda_version
    ):
        raise HarnessError("validated Torch runtime differs from tuning config")
    extension_sha256 = _required_sha256(
        provenance.get("kt_extension_sha256"), "KTransformers extension digest"
    )
    embedded_values = _required_json_list(
        provenance.get("embedded_provenance"), "embedded provenance"
    )
    embedded_sha256: dict[str, str] = {}
    for value in embedded_values:
        embedded = _json_object(value, "embedded provenance entry")
        distribution = _required_bounded_string(
            embedded.get("distribution"), "embedded provenance distribution"
        )
        if (
            embedded.get("sglang_revision") != sglang_revision
            or embedded.get("ktransformers_revision") != ktransformers_revision
            or distribution in embedded_sha256
        ):
            raise HarnessError("embedded runtime provenance differs from build")
        embedded_sha256[distribution] = _required_sha256(
            embedded.get("sha256"), "embedded provenance digest"
        )
    if set(embedded_sha256) != {"kt-kernel", "sglang-kt"}:
        raise HarnessError("embedded runtime provenance set is unexpected")

    return {
        "schema_version": 1,
        "install_id": install_id,
        "install_root": str(install_root),
        "runtime_python": cast(
            JsonObject, cast(object, config.runtime_python.model_dump(mode="json"))
        ),
        "base_runtime_python_sha256": base_python_sha256,
        "base_runtime_pip_freeze_sha256": base_pip_freeze_sha256,
        "installed_distribution_record_sha256": cast(
            JsonObject, cast(object, installed_record_sha256)
        ),
        "installed_distribution_file_counts": cast(
            JsonObject, cast(object, installed_file_counts)
        ),
        "build_id": build_id,
        "sglang_revision": sglang_revision,
        "ktransformers_revision": ktransformers_revision,
        "package_version": package_version,
        "runtime_wheel_sha256": cast(JsonObject, cast(object, build_wheels)),
        "fused_moe_distribution_sha256": build_wheels["sglang-kt"],
        "kt_extension_sha256": extension_sha256,
        "embedded_provenance_sha256": cast(JsonObject, cast(object, embedded_sha256)),
        "torch_version": torch_version,
        "triton_version": triton_version,
        "torch_cuda_version": torch_cuda_version,
        "gpu_uuid": config.host.gpu.uuid,
        "gpu_name": config.host.gpu.expected_name,
        "compute_capability": [8, 6],
        "capabilities": cast(list[JsonValue], cast(object, capabilities)),
        "receipt_bindings": {
            "runtime_install": cast(
                JsonObject, cast(object, receipts.install.model_dump(mode="json"))
            ),
            "runtime_build": cast(
                JsonObject, cast(object, receipts.build.model_dump(mode="json"))
            ),
            "kernel_validation": cast(
                JsonObject,
                cast(object, receipts.kernel_validation.model_dump(mode="json")),
            ),
        },
    }


def _run_capture(argv: Sequence[str], description: str) -> CompletedCommand:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError(f"cannot run {description}: {error}") from error
    return {
        "argv": list(argv),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _require_success(command: CompletedCommand, description: str) -> None:
    if command["return_code"] != 0:
        detail = command["stderr"].strip() or command["stdout"].strip()
        raise HarnessError(f"{description} failed: {detail}")


def _git_output(repository: Path, arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repository), *arguments), stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HarnessError(f"cannot inspect source repository: {error}") from error


def _source_commit(repository: Path) -> str:
    commit = _git_output(repository, ("rev-parse", "HEAD")).decode("ascii").strip()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise HarnessError("source HEAD is not an exact commit")
    return commit


def _head_file(repository: Path, relative_path: Path) -> bytes | None:
    try:
        return _git_output(repository, ("show", f"HEAD:{relative_path.as_posix()}"))
    except HarnessError:
        return None


def _source_file_identity(
    repository: Path,
) -> tuple[str, dict[str, str], dict[str, bytes]]:
    commit = _source_commit(repository)
    dirty_hashes: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    for relative_path in SOURCE_FILES:
        source_path = repository / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise HarnessError(f"required source file is unavailable: {source_path}")
        data = source_path.read_bytes()
        relative_name = relative_path.as_posix()
        contents[relative_name] = data
        if _head_file(repository, relative_path) != data:
            dirty_hashes[relative_name] = hashlib.sha256(data).hexdigest()
    return commit, dirty_hashes, contents


def _load_config(path: Path) -> TuningRunConfig:
    if not path.is_absolute():
        raise HarnessError("tuning config path must be absolute")
    try:
        contents = _read_bounded_regular_file(
            path, "tuning config", MAXIMUM_CONFIG_JSON_BYTES
        )
        return TuningRunConfig.model_validate_json(contents)
    except (OSError, ValidationError, ValueError) as error:
        raise HarnessError(f"cannot load tuning config: {error}") from error


def _create_source_capsule(
    config: TuningRunConfig,
    config_contents: bytes,
) -> tuple[str, dict[str, str], dict[str, str]]:
    repository = Path(config.source_repository)
    deployment = Path(config.source_deployment_root)
    if deployment.exists():
        raise HarnessError("source deployment path must not exist")
    commit, dirty_hashes, source_contents = _source_file_identity(repository)
    parent = deployment.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{deployment.name}.", dir=parent))
    file_hashes: dict[str, str] = {}
    try:
        for relative_name, contents in source_contents.items():
            destination = temporary / relative_name
            _write_new_file(destination, contents)
            file_hashes[relative_name] = hashlib.sha256(contents).hexdigest()
        _write_new_file(temporary / IMMUTABLE_CONFIG_FILENAME, config_contents)
        identity: JsonObject = {
            "schema_version": 1,
            "commit": commit,
            "dirty_file_hashes": cast(JsonObject, cast(object, dirty_hashes)),
            "file_sha256": cast(JsonObject, cast(object, file_hashes)),
            "config_sha256": hashlib.sha256(config_contents).hexdigest(),
        }
        _write_new_file(
            temporary / SOURCE_IDENTITY_FILENAME, _canonical_json_bytes(identity)
        )
        observed_commit, observed_dirty_hashes, observed_contents = (
            _source_file_identity(repository)
        )
        if (
            observed_commit != commit
            or observed_dirty_hashes != dirty_hashes
            or observed_contents != source_contents
        ):
            raise HarnessError("source changed while creating immutable deployment")
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()), reverse=True
        ):
            os.chmod(directory, 0o555)
        os.chmod(temporary, 0o555)
        temporary.rename(deployment)
    except BaseException:
        for path in sorted(temporary.rglob("*"), reverse=True):
            with contextlib.suppress(OSError):
                if path.is_dir():
                    os.chmod(path, 0o755)
                    path.rmdir()
                else:
                    path.unlink()
        with contextlib.suppress(OSError):
            os.chmod(temporary, 0o755)
            temporary.rmdir()
        raise
    return commit, dirty_hashes, file_hashes


def _immutable_child_argv(
    config: TuningRunConfig, *, lease_path: Path, lock_path: Path
) -> tuple[str, ...]:
    deployment = Path(config.source_deployment_root)
    return (
        config.runtime_python.path,
        str(deployment / "scripts/run_sglang_kt_glm47_moe_tuning.py"),
        "--config",
        str(deployment / IMMUTABLE_CONFIG_FILENAME),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    )


def _build_metadata(
    config: TuningRunConfig,
    *,
    child_argv: Sequence[str],
    source_commit: str,
    dirty_hashes: Mapping[str, str],
    source_file_hashes: Mapping[str, str],
    config_sha256: str,
    source_identity_sha256: str,
    generated_at: datetime,
) -> JsonObject:
    host = config.host.hostname
    hca_bindings: list[JsonValue] = [
        {"device": binding.device, "port": binding.port, "gid": binding.gid}
        for binding in config.host.hca_bindings
    ]
    return {
        "schema_version": 1,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "command": list(child_argv),
        "git": {
            "commit": source_commit,
            "dirty": bool(dirty_hashes),
            "dirty_file_hashes": dict(dirty_hashes),
        },
        "hosts": [host],
        "models": [
            {
                "model_id": config.contextual_model.model_id,
                "revision": config.contextual_model.revision,
                "paths": {host: config.contextual_model.path},
            }
        ],
        "gpu_bindings": {
            host: [
                {
                    "uuid": config.host.gpu.uuid,
                    "pci_address": config.host.gpu.pci_address,
                }
            ]
        },
        "cpu_bindings": {
            host: {
                "cpu_set": ",".join(str(cpu) for cpu in config.host.cpu_cores),
                "numa_nodes": list(config.host.memory_nodes),
                "memory_policy": "bind:"
                + ",".join(str(node) for node in config.host.memory_nodes),
            }
        },
        "hca_bindings": {host: hca_bindings},
        "source_deployments": {
            host: {
                "path": config.source_deployment_root,
                "commit": source_commit,
                "dirty_file_hashes": dict(dirty_hashes),
            }
        },
        "owner_pids": {host: []},
        "tuning_contract": {
            "kind": "glm47_sglang_kt_fused_moe_tuning",
            "profiler": config.profiler,
            "source_file_sha256": dict(source_file_hashes),
            "config_sha256": config_sha256,
            "source_identity_sha256": source_identity_sha256,
            "runtime_python": config.runtime_python.model_dump(mode="json"),
            "runtime_receipts": config.runtime_receipts.model_dump(mode="json"),
            "numactl_executable": config.numactl_executable.model_dump(mode="json"),
            "contextual_model_binding": config.contextual_model.model_dump(mode="json"),
            "contextual_model_snapshot_consumed": False,
            "contextual_model_binding_is_verification": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "performance_comparable": False,
            "candidate_only": True,
            "adoption_gate": {
                "kind": "warm_serving",
                "minimum_improvement_percent": 3.0,
                "status": "required_not_run",
            },
            "tuning": config.tuning.model_dump(mode="json"),
            "authorization_schema": "bounded-inherited-pipe-active-lease-v3",
        },
    }


def _validate_metadata(
    benchmark_lease_script: Path, metadata: JsonObject, now: datetime
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_glm47_tuning_benchmark_lease", benchmark_lease_script
    )
    if specification is None or specification.loader is None:
        raise HarnessError("cannot import immutable benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    validator = cast(LeaseMetadataValidator, module)
    validated = validator.validate_run_metadata(
        cast(Mapping[str, object], metadata), now=now
    )
    if validated != metadata:
        raise HarnessError("benchmark lease changed generated metadata")


def prepare_lease(
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
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> LeasePreparation:
    paths = (config_path, metadata_output, lease_path, lock_path, result_root)
    if any(not path.is_absolute() for path in paths):
        raise HarnessError("all preparation paths must be absolute")
    if not owner.strip() or not purpose.strip() or "\0" in owner + purpose:
        raise HarnessError("owner and purpose must be nonempty and NUL-free")
    for value, description in (
        (expected_duration_seconds, "expected duration"),
        (cleanup_grace_seconds, "cleanup grace"),
        (heartbeat_seconds, "heartbeat"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise HarnessError(f"{description} must be finite and positive")

    config_contents = _read_bounded_regular_file(
        config_path, "tuning config", MAXIMUM_CONFIG_JSON_BYTES
    )
    config = _load_config(config_path)
    repository = Path(config.source_repository)
    if (
        Path(__file__).resolve()
        != (repository / "scripts/run_sglang_kt_glm47_moe_tuning.py").resolve()
    ):
        raise HarnessError("prepare-lease must run from the configured repository")
    if Path(config.result_directory) != result_root / config.run_id:
        raise HarnessError("result directory must equal result_root/run_id")
    if Path(config.result_directory).exists() or metadata_output.exists():
        raise HarnessError("result directory and metadata output must both be new")
    if Path(config.source_deployment_root).exists():
        raise HarnessError("source deployment must be new")
    _verify_executable(config.runtime_python, "runtime Python")
    _verified_runtime_contract(config)
    _verify_file(config.numactl_executable, "numactl executable")
    source_commit, dirty_hashes, source_file_hashes = _create_source_capsule(
        config, config_contents
    )
    child_argv = _immutable_child_argv(
        config, lease_path=lease_path, lock_path=lock_path
    )
    generated_at = now().astimezone(timezone.utc)
    metadata = _build_metadata(
        config,
        child_argv=child_argv,
        source_commit=source_commit,
        dirty_hashes=dirty_hashes,
        source_file_hashes=source_file_hashes,
        config_sha256=hashlib.sha256(config_contents).hexdigest(),
        source_identity_sha256=_sha256_file(
            Path(config.source_deployment_root) / SOURCE_IDENTITY_FILENAME
        ),
        generated_at=generated_at,
    )
    benchmark_lease_script = (
        Path(config.source_deployment_root) / "scripts/benchmark_lease.py"
    )
    _validate_metadata(benchmark_lease_script, metadata, generated_at)
    _write_new_file(metadata_output, _canonical_json_bytes(metadata))
    lease_argv = (
        config.runtime_python.path,
        str(benchmark_lease_script),
        f"--owner={owner}",
        f"--purpose={purpose}",
        "--run-id",
        config.run_id,
        "--namespace",
        config.namespace,
        "--port",
        ",".join(str(port) for port in config.reserved_ports),
        "--metadata-json",
        str(metadata_output),
        "--heartbeat-seconds",
        str(heartbeat_seconds),
        "--expected-duration-seconds",
        str(expected_duration_seconds),
        "--cleanup-grace-seconds",
        str(cleanup_grace_seconds),
        "--lock-path",
        str(lock_path),
        "--lease-path",
        str(lease_path),
        "--result-root",
        str(result_root),
        "--",
        *child_argv,
    )
    return LeasePreparation(
        metadata_path=str(metadata_output),
        source_deployment_root=config.source_deployment_root,
        source_commit=source_commit,
        source_file_sha256=source_file_hashes,
        benchmark_lease_argv=lease_argv,
    )


def _result_descriptor(config: TuningRunConfig, result_dir: Path) -> int:
    descriptor_value = os.environ.get(RESULT_DIRECTORY_FD_ENVIRONMENT)
    if descriptor_value is None:
        raise HarnessError("benchmark lease did not pass a result directory FD")
    try:
        inherited_descriptor = int(descriptor_value)
    except ValueError as error:
        raise HarnessError("result directory FD is invalid") from error
    descriptor = os.dup(inherited_descriptor)
    expected = os.fstat(descriptor)
    observed = os.stat(result_dir, follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode) or (
        expected.st_dev,
        expected.st_ino,
    ) != (observed.st_dev, observed.st_ino):
        os.close(descriptor)
        raise HarnessError("result directory FD does not match configured path")
    if result_dir != Path(config.result_directory):
        os.close(descriptor)
        raise HarnessError("result directory argument does not match config")
    return descriptor


def _process_stat_identity(process_id: int) -> tuple[str, int, int, int]:
    try:
        contents = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
    except OSError as error:
        raise HarnessError(
            f"cannot read process identity for {process_id}: {error}"
        ) from error
    closing_parenthesis = contents.rfind(")")
    if closing_parenthesis < 0:
        raise HarnessError(f"process identity for {process_id} is truncated")
    fields = contents[closing_parenthesis + 1 :].split()
    if len(fields) <= 19:
        raise HarnessError(f"process identity for {process_id} is truncated")
    try:
        return fields[0], int(fields[2]), int(fields[3]), int(fields[19])
    except ValueError as error:
        raise HarnessError(f"process identity for {process_id} is invalid") from error


def _process_start_time_ticks(process_id: int) -> int:
    return _process_stat_identity(process_id)[3]


def _ancestor_process_ids() -> frozenset[int]:
    ancestors: set[int] = {os.getpid()}
    process_id = os.getppid()
    while process_id > 1 and process_id not in ancestors:
        ancestors.add(process_id)
        try:
            fields = (
                Path(f"/proc/{process_id}/stat").read_text(encoding="ascii").split()
            )
            process_id = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    return frozenset(ancestors)


def _assert_standard_lock_held(lock_path: Path) -> None:
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    raise HarnessError("standard benchmark flock is not held")


def _expected_child_argv(
    config: TuningRunConfig, lease_path: Path, lock_path: Path
) -> tuple[str, ...]:
    return _immutable_child_argv(config, lease_path=lease_path, lock_path=lock_path)


def _validate_active_lease(
    config: TuningRunConfig,
    *,
    lease_path: Path,
    lock_path: Path,
    result_dir: Path,
) -> JsonObject:
    _assert_standard_lock_held(lock_path)
    lease = _read_json(lease_path, "active benchmark lease")
    expected_values: tuple[tuple[str, object], ...] = (
        ("run_id", config.run_id),
        ("exo_namespace", config.namespace),
        ("result_directory", str(result_dir)),
        ("child_pid", os.getpid()),
        ("command", list(_expected_child_argv(config, lease_path, lock_path))),
    )
    for field_name, expected in expected_values:
        if lease.get(field_name) != expected:
            raise HarnessError(f"active lease {field_name} does not match this child")
    if lease.get("child_cleanup_confirmation_required") is not True:
        raise HarnessError("active lease does not require child cleanup confirmation")
    metadata = _json_object(lease.get("metadata"), "active lease metadata")
    if metadata.get("run_id") != config.run_id:
        raise HarnessError("active lease metadata run_id changed")
    return lease


def _verify_source_capsule(config: TuningRunConfig, lease: JsonObject) -> JsonObject:
    deployment = Path(config.source_deployment_root)
    identity = _read_json(deployment / SOURCE_IDENTITY_FILENAME, "source identity")
    file_hashes = _json_object(identity.get("file_sha256"), "source file hashes")
    for relative_path, digest_value in file_hashes.items():
        if not isinstance(digest_value, str):
            raise HarnessError("source file digest is invalid")
        observed = _sha256_file(deployment / relative_path)
        if observed != digest_value:
            raise HarnessError(f"immutable source changed: {relative_path}")
    config_digest = identity.get("config_sha256")
    if config_digest != _sha256_file(deployment / IMMUTABLE_CONFIG_FILENAME):
        raise HarnessError("immutable tuning config changed")
    metadata = _json_object(lease.get("metadata"), "lease metadata")
    contract = _json_object(metadata.get("tuning_contract"), "tuning contract")
    if contract.get("source_file_sha256") != file_hashes:
        raise HarnessError("source capsule hashes differ from lease metadata")
    if contract.get("config_sha256") != identity.get("config_sha256"):
        raise HarnessError("immutable config hash differs from lease metadata")
    if contract.get("source_identity_sha256") != _sha256_file(
        deployment / SOURCE_IDENTITY_FILENAME
    ):
        raise HarnessError("source identity hash differs from lease metadata")
    return identity


def _read_proc_command(process_directory: Path) -> tuple[str, str] | None:
    try:
        raw = (process_directory / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    arguments = tuple(
        part.decode("utf-8", errors="replace")
        for part in raw.rstrip(b"\0").split(b"\0")
        if part
    )
    if not arguments:
        return None
    return Path(arguments[0]).name, " ".join(arguments)


def find_conflicting_processes(
    proc_root: Path = Path("/proc"),
    *,
    ignored_process_ids: frozenset[int] | None = None,
) -> tuple[JsonObject, ...]:
    ignored = ignored_process_ids or _ancestor_process_ids()
    conflicts: list[JsonObject] = []
    entries = sorted(
        (entry for entry in proc_root.iterdir() if entry.name.isdigit()),
        key=lambda entry: int(entry.name),
    )
    for entry in entries:
        process_id = int(entry.name)
        if process_id in ignored:
            continue
        command = _read_proc_command(entry)
        if command is None:
            continue
        executable, rendered = command
        lowered = rendered.lower()
        reason: str | None = None
        if executable in _CONFLICTING_EXECUTABLES:
            reason = "benchmark process"
        elif any(fragment in lowered for fragment in _CONFLICTING_COMMAND_FRAGMENTS):
            reason = "inference or tuning process"
        elif any(fragment in lowered for fragment in _STORAGE_COMMAND_FRAGMENTS):
            reason = "storage-intensive process"
        if reason is not None:
            conflicts.append({"pid": process_id, "reason": reason, "command": rendered})
    return tuple(conflicts)


def _probe_ports_unused(ports: Sequence[int]) -> list[JsonValue]:
    evidence: list[JsonValue] = []
    for port in ports:
        sockets: list[socket.socket] = []
        try:
            for socket_type, protocol in (
                (socket.SOCK_STREAM, socket.IPPROTO_TCP),
                (socket.SOCK_DGRAM, socket.IPPROTO_UDP),
            ):
                probe = socket.socket(socket.AF_INET, socket_type, protocol)
                sockets.append(probe)
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                probe.bind(("0.0.0.0", port))
            evidence.append({"port": port, "tcp": "unused", "udp": "unused"})
        except OSError as error:
            raise HarnessError(f"reserved port {port} is in use: {error}") from error
        finally:
            for probe in sockets:
                probe.close()
    return evidence


def _nvidia_query(config: TuningRunConfig) -> JsonObject:
    fields = (
        "uuid",
        "pci.bus_id",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "clocks.sm",
        "power.draw",
        "power.limit",
    )
    command = _run_capture(
        (
            "/usr/bin/nvidia-smi",
            f"--id={config.host.gpu.uuid}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        "GPU telemetry",
    )
    _require_success(command, "GPU telemetry")
    rows = tuple(line for line in command["stdout"].splitlines() if line.strip())
    if len(rows) != 1:
        raise HarnessError("GPU telemetry did not identify exactly one device")
    values = tuple(value.strip() for value in rows[0].split(","))
    if len(values) != len(fields):
        raise HarnessError("GPU telemetry schema is unexpected")
    telemetry = dict(zip(fields, values, strict=True))
    if telemetry["uuid"] != config.host.gpu.uuid:
        raise HarnessError("visible GPU UUID differs from config")
    if telemetry["pci.bus_id"].lower() != config.host.gpu.pci_address.lower():
        raise HarnessError("visible GPU PCI address differs from config")
    if telemetry["name"] != config.host.gpu.expected_name:
        raise HarnessError("visible GPU name differs from config")
    compute = _run_capture(
        (
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        "GPU process query",
    )
    _require_success(compute, "GPU process query")
    active_rows = [
        row
        for row in compute["stdout"].splitlines()
        if row.strip() and row.split(",", maxsplit=1)[0].strip() == config.host.gpu.uuid
    ]
    if active_rows:
        raise HarnessError("selected GPU has unowned compute processes")
    driver_summary = _run_capture(("/usr/bin/nvidia-smi",), "NVIDIA driver summary")
    _require_success(driver_summary, "NVIDIA driver summary")
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", driver_summary["stdout"])
    if cuda_match is None:
        raise HarnessError("NVIDIA driver summary did not expose a CUDA version")
    return {
        "fields": cast(JsonObject, cast(object, telemetry)),
        "compute_process_query": cast(JsonObject, cast(object, compute)),
        "driver_summary": cast(JsonObject, cast(object, driver_summary)),
        "cuda_version": cuda_match.group(1),
    }


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _hca_telemetry(config: TuningRunConfig) -> list[JsonValue]:
    telemetry: list[JsonValue] = []
    for binding in config.host.hca_bindings:
        root = (
            Path("/sys/class/infiniband") / binding.device / "ports" / str(binding.port)
        )
        state = _read_optional_text(root / "state")
        if state is None or "ACTIVE" not in state.upper():
            raise HarnessError(
                f"HCA {binding.device} port {binding.port} is not ACTIVE"
            )
        counters: JsonObject = {}
        counters_root = root / "counters"
        if counters_root.is_dir():
            for counter in sorted(counters_root.iterdir()):
                value = _read_optional_text(counter)
                if value is not None:
                    counters[counter.name] = value
        telemetry.append(
            {
                "device": binding.device,
                "port": binding.port,
                "configured_gid": binding.gid,
                "state": state,
                "physical_state": _read_optional_text(root / "phys_state"),
                "rate": _read_optional_text(root / "rate"),
                "lid": _read_optional_text(root / "lid"),
                "counters": counters,
            }
        )
    return telemetry


def _cpu_telemetry(config: TuningRunConfig) -> JsonObject:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    flags_line = next(
        (line for line in cpuinfo.splitlines() if line.startswith("flags")), ""
    )
    flags = set(flags_line.partition(":")[2].split())
    required_amx = {"amx_bf16", "amx_int8", "amx_tile"}
    if not required_amx <= flags:
        raise HarnessError("required AMX CPU features are unavailable")
    governors: JsonObject = {}
    for cpu in config.host.cpu_cores:
        governor = _read_optional_text(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        )
        governors[str(cpu)] = governor
    return {
        "loadavg": Path("/proc/loadavg").read_text(encoding="ascii").strip(),
        "meminfo": cast(
            list[JsonValue],
            cast(
                object,
                Path("/proc/meminfo").read_text(encoding="ascii").splitlines(),
            ),
        ),
        "amx_features": cast(list[JsonValue], cast(object, sorted(required_amx))),
        "scaling_governors": governors,
    }


def _storage_telemetry(config: TuningRunConfig, result_dir: Path) -> JsonObject:
    mdstat = _read_optional_text(Path("/proc/mdstat")) or ""
    lowered = mdstat.lower()
    if any(word in lowered for word in ("resync", "recovery", "reshape", "check")):
        raise HarnessError("software RAID maintenance is active")
    values: JsonObject = {"mdstat": mdstat, "paths": {}}
    paths = cast(JsonObject, values["paths"])
    for path in (result_dir.parent,):
        stats = os.statvfs(path)
        paths[str(path)] = {
            "available_bytes": stats.f_bavail * stats.f_frsize,
            "free_bytes": stats.f_bfree * stats.f_frsize,
        }
    return values


def _profiler_preflight(
    proc_root: Path = Path("/proc"),
    *,
    ignored_process_ids: frozenset[int] | None = None,
) -> JsonObject:
    command = " ".join(
        part.decode("utf-8", errors="replace")
        for part in (proc_root / "self" / "cmdline").read_bytes().split(b"\0")
        if part
    ).lower()
    if any(token in command for token in _BANNED_PROFILER_TOKENS):
        raise HarnessError("command references prohibited sep5/pax profiler support")
    modules = _read_optional_text(proc_root / "modules") or ""
    loaded = {
        line.split(maxsplit=1)[0].lower()
        for line in modules.splitlines()
        if line.strip()
    }
    prohibited_loaded = sorted(loaded.intersection(_BANNED_PROFILER_TOKENS))
    active_users: list[int] = []
    ignored_processes: frozenset[int] = (
        frozenset[int]() if ignored_process_ids is None else ignored_process_ids
    )
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in ignored_processes:
            continue
        process_id = int(entry.name)
        command = _read_proc_command(entry)
        command_uses_driver = command is not None and any(
            token in command[1].lower() for token in _BANNED_PROFILER_TOKENS
        )
        descriptor_uses_driver = False
        try:
            for descriptor in (entry / "fd").iterdir():
                try:
                    target = os.readlink(descriptor).lower()
                except OSError:
                    continue
                if any(
                    marker in target
                    for marker in ("/dev/sep", "/dev/pax", "/dev/socperf")
                ):
                    descriptor_uses_driver = True
                    break
        except OSError:
            pass
        if command_uses_driver or descriptor_uses_driver:
            active_users.append(process_id)
    if active_users:
        raise HarnessError(
            "prohibited sep5/pax profiler use is active in processes: "
            + ",".join(str(process_id) for process_id in sorted(active_users))
        )
    return {
        "mode": "none",
        "prohibited_modules_loaded": cast(
            list[JsonValue], cast(object, prohibited_loaded)
        ),
        "prohibited_profiler_processes": [],
    }


def _collect_preflight(config: TuningRunConfig, result_dir: Path) -> JsonObject:
    observed_hostname = socket.gethostname().split(".", maxsplit=1)[0]
    if observed_hostname != config.host.hostname:
        raise HarnessError(
            f"configured host {config.host.hostname!r} differs from "
            f"local host {observed_hostname!r}"
        )
    conflicts = find_conflicting_processes()
    if conflicts:
        raise HarnessError(f"conflicting unowned processes are active: {conflicts}")
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": observed_hostname,
        "profiler": _profiler_preflight(),
        "ports": _probe_ports_unused(config.reserved_ports),
        "gpu": _nvidia_query(config),
        "cpu": _cpu_telemetry(config),
        "hca": _hca_telemetry(config),
        "storage": _storage_telemetry(config, result_dir),
        "conflicting_processes": [],
    }


def _tuner_argv(config: TuningRunConfig, output_directory: Path) -> tuple[str, ...]:
    cpu_set = ",".join(str(cpu) for cpu in config.host.cpu_cores)
    memory_nodes = ",".join(str(node) for node in config.host.memory_nodes)
    tuning = config.tuning
    return (
        config.numactl_executable.path,
        "--physcpubind",
        cpu_set,
        "--membind",
        memory_nodes,
        config.runtime_python.path,
        str(Path(config.source_deployment_root) / TUNER_RELATIVE_PATH),
        "--output-dir",
        str(output_directory),
        "--resident-experts",
        str(tuning.resident_experts),
        "--batch-sizes",
        ",".join(str(size) for size in tuning.batch_sizes),
        "--seed",
        str(tuning.seed),
        "--warmup-iters",
        str(tuning.warmup_iterations),
        "--measurement-iters",
        str(tuning.measurement_iterations),
        "--samples",
        str(tuning.independent_samples),
        "--search-profile",
        tuning.search_profile,
    )


def _child_environment(
    config: TuningRunConfig,
    *,
    owner_token: str,
    authorization_descriptor: int,
    result_descriptor: int,
    output_descriptor: int,
    cache_descriptors: Mapping[str, int],
) -> tuple[dict[str, str], JsonObject]:
    source = os.environ
    thread_count = len(config.host.cpu_cores)
    environment = {
        "HOME": source.get("HOME", "/root"),
        "LANG": source.get("LANG", "C.UTF-8"),
        "LC_ALL": source.get("LC_ALL", "C.UTF-8"),
        "PATH": source.get("PATH", "/usr/bin:/bin"),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": config.host.gpu.uuid,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "MKL_NUM_THREADS": str(thread_count),
        "OMP_NUM_THREADS": str(thread_count),
        "OPENBLAS_NUM_THREADS": str(thread_count),
        "PYTHONHASHSEED": str(config.tuning.seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
        "CUDA_CACHE_PATH": f"/proc/self/fd/{cache_descriptors['cuda-cache']}",
        "TRITON_CACHE_DIR": f"/proc/self/fd/{cache_descriptors['triton-cache']}",
        "TMPDIR": f"/proc/self/fd/{cache_descriptors['tmp']}",
        TUNER_AUTHORIZATION_FD_ENVIRONMENT: str(authorization_descriptor),
        TUNER_RESULT_DIRECTORY_FD_ENVIRONMENT: str(result_descriptor),
        TUNER_OUTPUT_DIRECTORY_FD_ENVIRONMENT: str(output_descriptor),
        TUNER_OWNER_TOKEN_ENVIRONMENT: owner_token,
    }
    evidence: JsonObject = {
        "inheritance_policy": "allowlist-v1",
        "environment": cast(JsonObject, cast(object, environment)),
    }
    return environment, evidence


def _authorization_payload(
    config: TuningRunConfig,
    *,
    lease: JsonObject,
    result_descriptor: int,
    output_descriptor: int,
    cache_descriptors: Mapping[str, int],
    result_dir: Path,
    owner_token: str,
    tuner_argv: Sequence[str],
    lease_path: Path,
    lock_path: Path,
    environment_evidence: JsonObject,
    runtime_contract: JsonObject,
) -> JsonObject:
    result_identity = os.fstat(result_descriptor)
    return {
        "schema_version": 3,
        "authorization_kind": "active-benchmark-lease-inherited-pipe-v3",
        "lease_id": lease.get("lease_id"),
        "run_id": config.run_id,
        "namespace": config.namespace,
        "lock_path": str(lock_path),
        "lease_path": str(lease_path),
        "result_directory": {
            "path": str(result_dir),
            "device": result_identity.st_dev,
            "inode": result_identity.st_ino,
        },
        "harness_process": {
            "pid": os.getpid(),
            "start_time_ticks": _process_start_time_ticks(os.getpid()),
        },
        "process_ownership": {
            "mode": "inherit-lease-child-process-group",
            "lease_child_pid": os.getpid(),
            "process_group_id": os.getpgrp(),
            "outer_wrapper_cleanup": "killpg",
        },
        "owner_token": owner_token,
        "launcher_argv": list(tuner_argv),
        "tuner_process_argv": list(tuner_argv[5:]),
        "tuning_output_directory": str(result_dir / TUNING_OUTPUT_DIRECTORY_NAME),
        "output_directory_descriptor": output_descriptor,
        "output_directory_identity": _directory_identity(output_descriptor),
        "cache_directories": {
            name: {
                "descriptor": descriptor,
                "identity": _directory_identity(descriptor),
            }
            for name, descriptor in sorted(cache_descriptors.items())
        },
        "child_environment": environment_evidence,
        "runtime_contract": runtime_contract,
        "receipt_bindings": {
            "runtime_python": config.runtime_python.model_dump(mode="json"),
            "runtime_install": config.runtime_receipts.install.model_dump(mode="json"),
            "runtime_build": config.runtime_receipts.build.model_dump(mode="json"),
            "kernel_validation": (
                config.runtime_receipts.kernel_validation.model_dump(mode="json")
            ),
            "tuner_script": {
                "path": str(Path(config.source_deployment_root) / TUNER_RELATIVE_PATH),
                "sha256": _sha256_file(
                    Path(config.source_deployment_root) / TUNER_RELATIVE_PATH
                ),
            },
            "source_identity": {
                "path": str(
                    Path(config.source_deployment_root) / SOURCE_IDENTITY_FILENAME
                ),
                "sha256": _sha256_file(
                    Path(config.source_deployment_root) / SOURCE_IDENTITY_FILENAME
                ),
            },
            "tuning_config": {
                "path": str(
                    Path(config.source_deployment_root) / IMMUTABLE_CONFIG_FILENAME
                ),
                "sha256": _sha256_file(
                    Path(config.source_deployment_root) / IMMUTABLE_CONFIG_FILENAME
                ),
            },
            "preflight": {
                "path": str(result_dir / PREFLIGHT_FILENAME),
                "sha256": _sha256_file_at(result_descriptor, PREFLIGHT_FILENAME),
            },
            "telemetry_before": {
                "path": str(result_dir / TELEMETRY_BEFORE_FILENAME),
                "sha256": _sha256_file_at(result_descriptor, TELEMETRY_BEFORE_FILENAME),
            },
        },
        "experiment_context": {
            "contextual_model": config.contextual_model.model_dump(mode="json"),
            "model_snapshot_consumed": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "binding_establishes_model_verification": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _environment_descriptor(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise HarnessError(f"{name} was not inherited from the tuning harness")
    try:
        descriptor = int(value)
    except ValueError as error:
        raise HarnessError(f"{name} is not a valid descriptor") from error
    if descriptor < 0:
        raise HarnessError(f"{name} is not a valid descriptor")
    try:
        os.fstat(descriptor)
    except OSError as error:
        raise HarnessError(f"{name} is not open: {error}") from error
    return descriptor


def _read_pipe_json(descriptor: int) -> tuple[JsonObject, bytes]:
    status = os.fstat(descriptor)
    if not stat.S_ISFIFO(status.st_mode):
        raise HarnessError("tuner authorization descriptor is not a pipe")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 16_384)
        if not chunk:
            break
        total += len(chunk)
        if total > MAXIMUM_AUTHORIZATION_BYTES:
            raise HarnessError("tuner authorization exceeds its size limit")
        chunks.append(chunk)
    contents = b"".join(chunks)
    try:
        value = cast(object, json.loads(contents))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"tuner authorization is invalid JSON: {error}") from error
    return _json_object(value, "tuner authorization"), contents


def _read_json_at(
    directory_descriptor: int,
    filename: str,
    description: str,
    maximum_bytes: int = MAXIMUM_RECEIPT_JSON_BYTES,
    *,
    require_read_only: bool = False,
) -> JsonObject:
    if Path(filename).name != filename:
        raise HarnessError(f"{description} filename must be a basename")
    descriptor = os.open(
        filename,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_size > maximum_bytes
            or (require_read_only and status.st_mode & 0o222)
        ):
            raise HarnessError(f"{description} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise HarnessError(f"{description} exceeds its size limit")
    finally:
        os.close(descriptor)
    try:
        value = cast(object, json.loads(b"".join(chunks)))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"{description} is invalid JSON: {error}") from error
    return _json_object(value, description)


def _require_json_string(value: JsonValue | None, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessError(f"{description} must be a nonempty string")
    return value


def _require_json_integer(value: JsonValue | None, description: str) -> int:
    if type(value) is not int:
        raise HarnessError(f"{description} must be an integer")
    return value


def _verify_authorized_file(value: JsonValue | None, description: str) -> JsonObject:
    binding = _json_object(value, description)
    path = Path(_require_json_string(binding.get("path"), f"{description}.path"))
    digest = _require_json_string(binding.get("sha256"), f"{description}.sha256")
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise HarnessError(f"{description}.sha256 is invalid")
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
        raise HarnessError(f"{description} binding does not match the filesystem")
    return binding


def _verify_authorized_executable(
    value: JsonValue | None, description: str
) -> JsonObject:
    raw_binding = _json_object(value, description)
    try:
        binding = ExecutableBinding.model_validate_json(
            _canonical_json_bytes(raw_binding, pretty=False)
        )
    except ValidationError as error:
        raise HarnessError(f"{description} binding is invalid: {error}") from error
    _verify_executable(binding, description)
    return cast(JsonObject, cast(object, binding.model_dump(mode="json")))


def _write_pipe_payload(
    descriptor: int,
    contents: bytes,
    process: subprocess.Popen[str],
    timeout_seconds: float = 5.0,
) -> None:
    if len(contents) > MAXIMUM_AUTHORIZATION_BYTES:
        raise HarnessError("tuner authorization exceeds its size limit")
    deadline = time.monotonic() + timeout_seconds
    offset = 0
    os.set_blocking(descriptor, False)
    while offset < len(contents):
        if process.poll() is not None:
            raise HarnessError("tuner exited before consuming authorization")
        if time.monotonic() >= deadline:
            raise HarnessError("timed out delivering tuner authorization")
        try:
            written = os.write(descriptor, contents[offset:])
        except BlockingIOError:
            time.sleep(0.01)
            continue
        except BrokenPipeError as error:
            raise HarnessError("tuner closed its authorization pipe") from error
        if written <= 0:
            raise HarnessError("authorization pipe made no forward progress")
        offset += written


def _current_process_argv() -> list[JsonValue]:
    try:
        command_line = Path("/proc/self/cmdline").read_bytes()
    except OSError as error:
        raise HarnessError(f"cannot read tuner command line: {error}") from error
    return [
        part.decode("utf-8", errors="strict")
        for part in command_line.rstrip(b"\0").split(b"\0")
        if part
    ]


def validate_tuner_authorization(*, output_directory: Path) -> JsonObject:
    """Consume and validate the one-run authorization before CUDA is imported.

    This is intentionally callable by the sibling tuner module.  The pipe is
    consumed once; a second validation attempt fails closed.
    """

    authorization_descriptor = _environment_descriptor(
        TUNER_AUTHORIZATION_FD_ENVIRONMENT
    )
    result_descriptor = _environment_descriptor(TUNER_RESULT_DIRECTORY_FD_ENVIRONMENT)
    output_descriptor = _environment_descriptor(TUNER_OUTPUT_DIRECTORY_FD_ENVIRONMENT)
    authorization, authorization_contents = _read_pipe_json(authorization_descriptor)
    if (
        authorization.get("schema_version") != 3
        or authorization.get("authorization_kind")
        != "active-benchmark-lease-inherited-pipe-v3"
    ):
        raise HarnessError("tuner authorization schema is unsupported")
    stored_authorization = _read_json_at(
        result_descriptor,
        AUTHORIZATION_FILENAME,
        "stored tuner authorization",
        MAXIMUM_AUTHORIZATION_BYTES,
        require_read_only=True,
    )
    if stored_authorization != authorization:
        raise HarnessError("pipe authorization differs from stored authorization")

    owner_token = _require_json_string(
        authorization.get("owner_token"), "authorization owner_token"
    )
    if (
        _TOKEN_PATTERN.fullmatch(owner_token) is None
        or os.environ.get(TUNER_OWNER_TOKEN_ENVIRONMENT) != owner_token
    ):
        raise HarnessError("tuner owner token does not match authorization")

    result_binding = _json_object(
        authorization.get("result_directory"), "authorization result directory"
    )
    result_path = Path(
        _require_json_string(result_binding.get("path"), "result directory path")
    )
    descriptor_status = os.fstat(result_descriptor)
    path_status = os.stat(result_path, follow_symlinks=False)
    expected_device = _require_json_integer(
        result_binding.get("device"), "result directory device"
    )
    expected_inode = _require_json_integer(
        result_binding.get("inode"), "result directory inode"
    )
    if (
        not stat.S_ISDIR(descriptor_status.st_mode)
        or not stat.S_ISDIR(path_status.st_mode)
        or (descriptor_status.st_dev, descriptor_status.st_ino)
        != (expected_device, expected_inode)
        or (path_status.st_dev, path_status.st_ino) != (expected_device, expected_inode)
    ):
        raise HarnessError("authorized result directory identity changed")
    authorized_output = Path(
        _require_json_string(
            authorization.get("tuning_output_directory"),
            "authorized tuning output directory",
        )
    )
    if output_directory != authorized_output or output_directory.parent != result_path:
        raise HarnessError("requested tuning output differs from authorization")
    authorized_output_descriptor = _require_json_integer(
        authorization.get("output_directory_descriptor"),
        "authorized output directory descriptor",
    )
    output_identity = _json_object(
        authorization.get("output_directory_identity"),
        "authorized output directory identity",
    )
    output_status = os.fstat(output_descriptor)
    output_path_status = os.stat(output_directory, follow_symlinks=False)
    expected_output_identity = (
        _require_json_integer(output_identity.get("device"), "output directory device"),
        _require_json_integer(output_identity.get("inode"), "output directory inode"),
    )
    if (
        authorized_output_descriptor != output_descriptor
        or not stat.S_ISDIR(output_status.st_mode)
        or not stat.S_ISDIR(output_path_status.st_mode)
        or (output_status.st_dev, output_status.st_ino) != expected_output_identity
        or (output_path_status.st_dev, output_path_status.st_ino)
        != expected_output_identity
        or os.listdir(output_descriptor)
    ):
        raise HarnessError(
            "authorized tuning output is not the fresh anchored directory"
        )

    cache_bindings = _json_object(
        authorization.get("cache_directories"), "authorized cache directories"
    )
    if set(cache_bindings) != {"cuda-cache", "triton-cache", "tmp"}:
        raise HarnessError("authorized cache directory set is invalid")
    for cache_name, cache_value in cache_bindings.items():
        cache_binding = _json_object(cache_value, f"authorized {cache_name}")
        cache_descriptor = _require_json_integer(
            cache_binding.get("descriptor"), f"authorized {cache_name} descriptor"
        )
        cache_identity = _json_object(
            cache_binding.get("identity"), f"authorized {cache_name} identity"
        )
        cache_status = os.fstat(cache_descriptor)
        if not stat.S_ISDIR(cache_status.st_mode) or (
            cache_status.st_dev,
            cache_status.st_ino,
        ) != (
            _require_json_integer(
                cache_identity.get("device"), f"authorized {cache_name} device"
            ),
            _require_json_integer(
                cache_identity.get("inode"), f"authorized {cache_name} inode"
            ),
        ):
            raise HarnessError(f"authorized {cache_name} directory identity changed")

    parent_binding = _json_object(
        authorization.get("harness_process"), "authorization harness process"
    )
    parent_process_id = _require_json_integer(
        parent_binding.get("pid"), "harness process pid"
    )
    parent_start_time = _require_json_integer(
        parent_binding.get("start_time_ticks"), "harness process start time"
    )
    if os.getppid() != parent_process_id:
        raise HarnessError("tuner parent PID differs from authorization")
    if _process_start_time_ticks(parent_process_id) != parent_start_time:
        raise HarnessError("tuner parent identity changed")
    process_ownership = _json_object(
        authorization.get("process_ownership"), "authorization process ownership"
    )
    expected_process_group = _require_json_integer(
        process_ownership.get("process_group_id"), "owned process group"
    )
    if (
        process_ownership
        != {
            "mode": "inherit-lease-child-process-group",
            "lease_child_pid": parent_process_id,
            "process_group_id": expected_process_group,
            "outer_wrapper_cleanup": "killpg",
        }
        or expected_process_group != parent_process_id
    ):
        raise HarnessError("authorization process ownership is invalid")
    if os.getpgrp() != expected_process_group:
        raise HarnessError("tuner escaped the lease child process group")

    lock_path = Path(
        _require_json_string(authorization.get("lock_path"), "authorization lock path")
    )
    lease_path = Path(
        _require_json_string(
            authorization.get("lease_path"), "authorization lease path"
        )
    )
    if os.environ.get("EXO_TESTS") != "1" and (
        lock_path != DEFAULT_LOCK_PATH or lease_path != DEFAULT_LEASE_PATH
    ):
        raise HarnessError("live tuning requires the standard lock and lease paths")
    _assert_standard_lock_held(lock_path)
    lease = _read_json(lease_path, "active benchmark lease")
    expected_lease_values: tuple[tuple[str, object], ...] = (
        ("lease_id", authorization.get("lease_id")),
        ("run_id", authorization.get("run_id")),
        ("exo_namespace", authorization.get("namespace")),
        ("result_directory", str(result_path)),
        ("child_pid", parent_process_id),
        ("child_cleanup_confirmation_required", True),
    )
    for field_name, expected in expected_lease_values:
        if lease.get(field_name) != expected:
            raise HarnessError(f"active lease {field_name} differs from authorization")

    tuner_process_argv = authorization.get("tuner_process_argv")
    if tuner_process_argv != _current_process_argv():
        raise HarnessError("live tuner argv differs from authorization")
    launcher_argv = authorization.get("launcher_argv")
    if not isinstance(launcher_argv, list) or not launcher_argv:
        raise HarnessError("authorized launcher argv is invalid")

    receipt_bindings = _json_object(
        authorization.get("receipt_bindings"), "authorization receipt bindings"
    )
    runtime_binding = _verify_authorized_executable(
        receipt_bindings.get("runtime_python"), "runtime Python"
    )
    install_binding = _verify_authorized_file(
        receipt_bindings.get("runtime_install"), "runtime install receipt"
    )
    build_binding = _verify_authorized_file(
        receipt_bindings.get("runtime_build"), "runtime build receipt"
    )
    kernel_binding = _verify_authorized_file(
        receipt_bindings.get("kernel_validation"), "kernel validation receipt"
    )
    tuner_binding = _verify_authorized_file(
        receipt_bindings.get("tuner_script"), "tuner script"
    )
    source_binding = _verify_authorized_file(
        receipt_bindings.get("source_identity"), "source identity"
    )
    config_binding = _verify_authorized_file(
        receipt_bindings.get("tuning_config"), "tuning config"
    )
    preflight_binding = _verify_authorized_file(
        receipt_bindings.get("preflight"), "preflight evidence"
    )
    telemetry_binding = _verify_authorized_file(
        receipt_bindings.get("telemetry_before"), "pre-tuning telemetry"
    )
    current_argv = _current_process_argv()
    if (
        len(current_argv) < 2
        or current_argv[0] != runtime_binding.get("path")
        or current_argv[1] != tuner_binding.get("path")
    ):
        raise HarnessError("runtime or tuner path differs from receipt binding")
    authorized_config = _load_config(Path(cast(str, config_binding["path"])))
    expected_runtime_receipts = {
        "runtime_install": authorized_config.runtime_receipts.install.model_dump(
            mode="json"
        ),
        "runtime_build": authorized_config.runtime_receipts.build.model_dump(
            mode="json"
        ),
        "kernel_validation": (
            authorized_config.runtime_receipts.kernel_validation.model_dump(mode="json")
        ),
    }
    if {
        "runtime_install": install_binding,
        "runtime_build": build_binding,
        "kernel_validation": kernel_binding,
    } != expected_runtime_receipts:
        raise HarnessError("authorized runtime receipts differ from tuning config")
    runtime_contract = _verified_runtime_contract(authorized_config)
    if authorization.get("runtime_contract") != runtime_contract:
        raise HarnessError("authorized runtime contract differs from verified receipts")
    experiment_context = _json_object(
        authorization.get("experiment_context"), "authorization experiment context"
    )
    if experiment_context != {
        "contextual_model": authorized_config.contextual_model.model_dump(mode="json"),
        "model_snapshot_consumed": False,
        "contextual_model_snapshot_weights_loaded": False,
        "synthetic_kernel_weights_generated": True,
        "binding_establishes_model_verification": False,
    }:
        raise HarnessError("authorized experiment context differs from tuning config")

    lease_after_receipts = _read_json(lease_path, "active benchmark lease")
    if lease_after_receipts != lease:
        raise HarnessError("active lease changed during tuner authorization")
    final_result_path_status = os.stat(result_path, follow_symlinks=False)
    final_output_path_status = os.stat(output_directory, follow_symlinks=False)
    if (final_result_path_status.st_dev, final_result_path_status.st_ino) != (
        expected_device,
        expected_inode,
    ) or (
        final_output_path_status.st_dev,
        final_output_path_status.st_ino,
    ) != expected_output_identity:
        raise HarnessError("authorized directory pathname changed during validation")

    authorization_sha256 = hashlib.sha256(authorization_contents).hexdigest()
    return {
        "schema_version": 1,
        "authorization_sha256": authorization_sha256,
        "lease_id": authorization.get("lease_id"),
        "run_id": authorization.get("run_id"),
        "namespace": authorization.get("namespace"),
        "result_directory": result_binding,
        "tuning_output_directory": str(output_directory),
        "output_directory_descriptor": output_descriptor,
        "output_directory_identity": output_identity,
        "tuner_process_argv": tuner_process_argv,
        "launcher_argv": launcher_argv,
        "receipt_sha256": {
            "runtime_python": runtime_binding.get("sha256"),
            "runtime_install": install_binding.get("sha256"),
            "runtime_build": build_binding.get("sha256"),
            "kernel_validation": kernel_binding.get("sha256"),
            "tuner_script": tuner_binding.get("sha256"),
            "source_identity": source_binding.get("sha256"),
            "tuning_config": config_binding.get("sha256"),
            "preflight": preflight_binding.get("sha256"),
            "telemetry_before": telemetry_binding.get("sha256"),
        },
        "child_environment": authorization.get("child_environment"),
        "parent_process": parent_binding,
        "lock_path": str(lock_path),
        "lease_path": str(lease_path),
        "output_created_by_harness": True,
        "runtime_contract": runtime_contract,
        "verified_authorization": authorization,
    }


def _tagged_processes(
    owner_token: str,
    config: TuningRunConfig,
    log_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[tuple[int, int], ProcessIdentity]:
    processes: dict[tuple[int, int], ProcessIdentity] = {}
    expected = f"{TUNER_OWNER_TOKEN_ENVIRONMENT}={owner_token}".encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        try:
            before = _process_stat_identity(process_id)
            environment = (entry / "environ").read_bytes().split(b"\0")
            if expected not in environment:
                continue
            after = _process_stat_identity(process_id)
        except (FileNotFoundError, PermissionError, ProcessLookupError, HarnessError):
            continue
        if before != after:
            continue
        state, process_group_id, _session_id, start_time = after
        if state == "Z":
            continue
        identity: ProcessIdentity = {
            "host_name": config.host.hostname,
            "pid": process_id,
            "process_group_id": process_group_id,
            "start_time_ticks": start_time,
            "transport_pid": process_id,
            "namespace": config.namespace,
            "owner_token": owner_token,
            "log_path": str(log_path),
        }
        processes[(process_id, start_time)] = identity
    return processes


def _same_process_group_processes(
    process_group_id: int,
    *,
    excluded_process_id: int,
    owner_token: str,
    config: TuningRunConfig,
    log_path: Path,
    proc_root: Path = Path("/proc"),
) -> dict[tuple[int, int], ProcessIdentity]:
    processes: dict[tuple[int, int], ProcessIdentity] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        if process_id == excluded_process_id:
            continue
        try:
            state, observed_group, _session_id, start_time = _process_stat_identity(
                process_id
            )
        except HarnessError:
            continue
        if state == "Z" or observed_group != process_group_id:
            continue
        identity: ProcessIdentity = {
            "host_name": config.host.hostname,
            "pid": process_id,
            "process_group_id": observed_group,
            "start_time_ticks": start_time,
            "transport_pid": process_id,
            "namespace": config.namespace,
            "owner_token": owner_token,
            "log_path": str(log_path),
        }
        processes[(process_id, start_time)] = identity
    return processes


def _discover_owned_processes(
    owner_token: str,
    config: TuningRunConfig,
    log_path: Path,
    *,
    process_group_id: int,
    harness_process_id: int,
) -> dict[tuple[int, int], ProcessIdentity]:
    processes = _tagged_processes(owner_token, config, log_path)
    processes.update(
        _same_process_group_processes(
            process_group_id,
            excluded_process_id=harness_process_id,
            owner_token=owner_token,
            config=config,
            log_path=log_path,
        )
    )
    return processes


def _write_runtime_metadata(
    result_descriptor: int,
    config: TuningRunConfig,
    owner_token: str,
    owned_processes: Mapping[tuple[int, int], ProcessIdentity],
) -> None:
    value: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "owner_token": owner_token,
        "owned_processes": cast(
            list[JsonValue], cast(object, list(owned_processes.values()))
        ),
    }
    _atomic_write_json_at(result_descriptor, RUNTIME_METADATA_FILENAME, value)


def _identity_alive(identity: ProcessIdentity) -> bool:
    try:
        state, _group, _session, start_time = _process_stat_identity(identity["pid"])
        return state != "Z" and start_time == identity["start_time_ticks"]
    except HarnessError:
        return False


def _signal_process_identity(identity: ProcessIdentity, signal_number: int) -> bool:
    if not _identity_alive(identity):
        return False
    try:
        descriptor = os.pidfd_open(identity["pid"], 0)
    except (AttributeError, OSError):
        return False
    try:
        if not _identity_alive(identity):
            return False
        signal.pidfd_send_signal(descriptor, signal_number)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)


def _terminate_owned_identities(
    owned_processes: dict[tuple[int, int], ProcessIdentity],
    timeout_seconds: float,
    *,
    discover_owned_processes: Callable[[], Mapping[tuple[int, int], ProcessIdentity]],
) -> bool:
    def run_phase(signal_number: int, deadline: float) -> bool:
        signaled: set[tuple[int, int]] = set()
        quiescent_scans = 0
        while True:
            owned_processes.update(discover_owned_processes())
            alive = {
                key: identity
                for key, identity in owned_processes.items()
                if _identity_alive(identity)
            }
            if not alive:
                quiescent_scans += 1
                if quiescent_scans >= 2:
                    return True
            else:
                quiescent_scans = 0
                for key, identity in alive.items():
                    if key not in signaled and _signal_process_identity(
                        identity, signal_number
                    ):
                        signaled.add(key)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    if run_phase(signal.SIGTERM, time.monotonic() + timeout_seconds):
        return True
    kill_timeout = min(5.0, max(0.1, timeout_seconds))
    return run_phase(signal.SIGKILL, time.monotonic() + kill_timeout)


@dataclass(frozen=True)
class OwnedTunerOutcome:
    return_code: int | None
    completion: JsonObject
    owned_processes: tuple[ProcessIdentity, ...]
    cleanup_succeeded: bool
    error: str | None
    output_descriptor: int


def _run_owned_tuner(
    config: TuningRunConfig,
    *,
    lease: JsonObject,
    result_descriptor: int,
    result_dir: Path,
    lease_path: Path,
    lock_path: Path,
    owner_token: str,
    signal_state: ManagedSignalState,
) -> OwnedTunerOutcome:
    output_directory = result_dir / TUNING_OUTPUT_DIRECTORY_NAME
    tuner_argv = _tuner_argv(config, output_directory)
    owned_processes: dict[tuple[int, int], ProcessIdentity] = {}
    output_descriptor: int | None = None
    cache_descriptors: dict[str, int] = {}
    read_descriptor = -1
    write_descriptor = -1
    stdout: TextIO | None = None
    stderr: TextIO | None = None
    process: subprocess.Popen[str] | None = None
    environment_evidence: JsonObject = {}
    timed_out = False
    run_error: str | None = None
    return_code: int | None = None
    cleanup_succeeded = True
    harness_process_id = os.getpid()
    lease_process_group = os.getpgrp()

    def record_error(error: BaseException) -> None:
        nonlocal run_error
        detail = f"{type(error).__name__}: {error}"
        run_error = detail if run_error is None else f"{run_error}; {detail}"

    def discover() -> dict[tuple[int, int], ProcessIdentity]:
        return _discover_owned_processes(
            owner_token,
            config,
            result_dir / TUNER_STDOUT_FILENAME,
            process_group_id=lease_process_group,
            harness_process_id=harness_process_id,
        )

    try:
        if _TOKEN_PATTERN.fullmatch(owner_token) is None:
            raise HarnessError("generated owner token is invalid")
        if lease_process_group != harness_process_id:
            raise HarnessError("tuning harness is not the lease process-group leader")
        output_descriptor = _create_directory_at(
            result_descriptor, TUNING_OUTPUT_DIRECTORY_NAME
        )
        for name in ("cuda-cache", "triton-cache", "tmp"):
            cache_descriptors[name] = _create_directory_at(result_descriptor, name)
        read_descriptor, write_descriptor = os.pipe()
        environment, environment_evidence = _child_environment(
            config,
            owner_token=owner_token,
            authorization_descriptor=read_descriptor,
            result_descriptor=result_descriptor,
            output_descriptor=output_descriptor,
            cache_descriptors=cache_descriptors,
        )
        authorization = _authorization_payload(
            config,
            lease=lease,
            result_descriptor=result_descriptor,
            output_descriptor=output_descriptor,
            cache_descriptors=cache_descriptors,
            result_dir=result_dir,
            owner_token=owner_token,
            tuner_argv=tuner_argv,
            lease_path=lease_path,
            lock_path=lock_path,
            environment_evidence=environment_evidence,
            runtime_contract=_verified_runtime_contract(config),
        )
        authorization_bytes = _canonical_json_bytes(authorization, pretty=False)
        if len(authorization_bytes) > MAXIMUM_AUTHORIZATION_BYTES:
            raise HarnessError("tuner authorization exceeds its size limit")
        _atomic_write_json_at(result_descriptor, AUTHORIZATION_FILENAME, authorization)
        _make_read_only_at(result_descriptor, AUTHORIZATION_FILENAME)
        _write_runtime_metadata(result_descriptor, config, owner_token, owned_processes)
        stdout, _ = _open_new_log_at(result_descriptor, TUNER_STDOUT_FILENAME)
        stderr, _ = _open_new_log_at(result_descriptor, TUNER_STDERR_FILENAME)
        process = subprocess.Popen(
            tuner_argv,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=environment,
            pass_fds=(
                read_descriptor,
                result_descriptor,
                output_descriptor,
                *cache_descriptors.values(),
            ),
            start_new_session=False,
        )
        if os.getpgid(process.pid) != lease_process_group:
            raise HarnessError("tuner did not inherit the lease process group")
        direct_identity: ProcessIdentity = {
            "host_name": config.host.hostname,
            "pid": process.pid,
            "process_group_id": lease_process_group,
            "start_time_ticks": _process_start_time_ticks(process.pid),
            "transport_pid": process.pid,
            "namespace": config.namespace,
            "owner_token": owner_token,
            "log_path": str(result_dir / TUNER_STDOUT_FILENAME),
        }
        owned_processes[(process.pid, direct_identity["start_time_ticks"])] = (
            direct_identity
        )
        _write_runtime_metadata(result_descriptor, config, owner_token, owned_processes)
        os.close(read_descriptor)
        read_descriptor = -1
        _write_pipe_payload(write_descriptor, authorization_bytes, process)
        os.close(write_descriptor)
        write_descriptor = -1
        deadline = time.monotonic() + config.timeouts.tuning_seconds
        while (
            process.poll() is None
            and time.monotonic() < deadline
            and signal_state.first_signal_number is None
        ):
            discovered = discover()
            if not discovered.keys() <= owned_processes.keys():
                owned_processes.update(discovered)
                _write_runtime_metadata(
                    result_descriptor, config, owner_token, owned_processes
                )
            time.sleep(0.25)
        if (
            signal_state.first_signal_number is None
            and process.poll() is None
            and time.monotonic() >= deadline
        ):
            timed_out = True
        return_code = process.poll()
    except BaseException as error:
        record_error(error)
    finally:
        if process is not None:
            try:
                owned_processes.update(discover())
                cleanup_succeeded = _terminate_owned_identities(
                    owned_processes,
                    config.timeouts.cleanup_seconds,
                    discover_owned_processes=discover,
                )
            except BaseException as error:
                cleanup_succeeded = False
                record_error(error)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=min(1.0, config.timeouts.cleanup_seconds))
            return_code = process.poll()
            try:
                _write_runtime_metadata(
                    result_descriptor, config, owner_token, owned_processes
                )
            except BaseException as error:
                record_error(error)
        if read_descriptor >= 0:
            os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
        for descriptor in cache_descriptors.values():
            os.close(descriptor)

    if output_descriptor is None:
        raise HarnessError(run_error or "tuning output setup failed")
    completion: JsonObject = {
        "schema_version": 1,
        "command": list(tuner_argv),
        "environment": environment_evidence,
        "return_code": return_code,
        "timed_out": timed_out,
        "interrupted_signal": signal_state.first_signal_number,
        "output_directory": str(output_directory),
        "output_directory_identity": _directory_identity(output_descriptor),
        "output_tree_entry_count": len(os.listdir(output_descriptor)),
        "error": run_error,
    }
    return OwnedTunerOutcome(
        return_code=return_code,
        completion=completion,
        owned_processes=tuple(owned_processes.values()),
        cleanup_succeeded=cleanup_succeeded,
        error=run_error,
        output_descriptor=output_descriptor,
    )


def _read_relative_file_at(
    root_descriptor: int,
    relative_path: str,
    description: str,
    maximum_bytes: int,
) -> bytes:
    path = Path(relative_path)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HarnessError(f"{description} path is not a safe relative path")
    directory_descriptor = os.dup(root_descriptor)
    try:
        for part in path.parts[:-1]:
            child_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        descriptor = os.open(
            path.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_size > maximum_bytes:
                raise HarnessError(f"{description} is not a bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                contents = source.read(maximum_bytes + 1)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise HarnessError(f"cannot read {description}: {error}") from error
    finally:
        os.close(directory_descriptor)
    if len(contents) > maximum_bytes:
        raise HarnessError(f"{description} exceeds its size limit")
    return contents


def _output_tree(
    root_descriptor: int,
) -> tuple[dict[str, tuple[str, int]], set[str]]:
    files: dict[str, tuple[str, int]] = {}
    directories: set[str] = set()

    def visit(directory_descriptor: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            raise HarnessError(f"cannot enumerate tuning output: {error}") from error
        for name in names:
            if Path(name).name != name or name in {".", ".."}:
                raise HarnessError("tuning output contains an invalid entry name")
            relative_path = f"{prefix}/{name}" if prefix else name
            try:
                status = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
            except OSError as error:
                raise HarnessError(f"cannot inspect tuning output: {error}") from error
            if stat.S_ISLNK(status.st_mode):
                raise HarnessError("tuning output contains a symlink")
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative_path)
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                try:
                    visit(child, relative_path)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(status.st_mode):
                raise HarnessError("tuning output contains a non-regular entry")
            contents = _read_relative_file_at(
                root_descriptor,
                relative_path,
                "tuning output file",
                MAXIMUM_MANIFEST_JSON_BYTES,
            )
            files[relative_path] = (hashlib.sha256(contents).hexdigest(), len(contents))
            os.chmod(
                name,
                0o444,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )

    visit(root_descriptor, "")
    return files, directories


def _require_exact_keys(
    value: JsonObject, expected: set[str], description: str
) -> None:
    if set(value) != expected:
        raise HarnessError(
            f"{description} keys differ: expected {sorted(expected)}, "
            f"got {sorted(value)}"
        )


def _validate_kernel_config_file(
    contents: bytes, config: TuningRunConfig
) -> dict[int, JsonObject]:
    try:
        value = _json_object(cast(object, json.loads(contents)), "kernel config file")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"kernel config file is invalid JSON: {error}") from error
    expected_batches = {str(batch) for batch in config.tuning.batch_sizes}
    if set(value) != expected_batches:
        raise HarnessError("kernel config file does not exactly cover batch anchors")
    parsed: dict[int, JsonObject] = {}
    for batch in config.tuning.batch_sizes:
        raw_kernel_config = value[str(batch)]
        _validate_kernel_config_object(
            raw_kernel_config, f"kernel config batch {batch}"
        )
        parsed[batch] = _json_object(raw_kernel_config, f"kernel config batch {batch}")
    return parsed


def _validate_kernel_config_object(value: JsonValue | None, description: str) -> None:
    kernel_config = _json_object(value, description)
    expected_config_keys = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    _require_exact_keys(kernel_config, expected_config_keys, description)
    if any(
        type(setting) is not int or setting <= 0 for setting in kernel_config.values()
    ):
        raise HarnessError("kernel config values must be positive integers")


def _is_fallback_kernel_config(config: JsonObject) -> bool:
    return config == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def _kernel_config_key(config: JsonObject) -> tuple[int, int, int, int, int, int]:
    return cast(
        tuple[int, int, int, int, int, int],
        tuple(
            cast(int, config[name])
            for name in (
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
                "GROUP_SIZE_M",
                "num_warps",
                "num_stages",
            )
        ),
    )


def _canonical_search_configs(
    config: TuningRunConfig, tuner_path: Path, expected_sha256: str
) -> tuple[JsonObject, ...]:
    if _sha256_file(tuner_path) != expected_sha256:
        raise HarnessError("canonical search helper differs from authenticated tuner")
    module_name = f"_glm47_tuner_search_{expected_sha256}"
    specification = importlib.util.spec_from_file_location(module_name, tuner_path)
    if specification is None or specification.loader is None:
        raise HarnessError("cannot import authenticated tuner search helper")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise HarnessError(
            f"cannot load authenticated tuner search helper: {error}"
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    tuner = cast(TunerSearchSpaceModule, module)
    try:
        raw_configs = tuner.build_rtx3090_search_space(config.tuning.search_profile)
    except Exception as error:
        raise HarnessError(
            f"cannot derive canonical tuner search space: {error}"
        ) from error
    configs: list[JsonObject] = []
    for index, raw_config in enumerate(raw_configs):
        candidate = cast(JsonObject, cast(object, dict(raw_config)))
        _validate_kernel_config_object(candidate, f"canonical search config {index}")
        configs.append(candidate)
    if not configs:
        raise HarnessError("canonical tuner search space is empty")
    return tuple(configs)


def _finite_numbers(
    value: JsonValue | None,
    *,
    description: str,
    expected_length: int | None = None,
    positive: bool,
) -> list[float]:
    raw_values = _required_json_list(value, description)
    if expected_length is not None and len(raw_values) != expected_length:
        raise HarnessError(f"{description} has the wrong sample count")
    numbers: list[float] = []
    for raw_value in raw_values:
        if (
            not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
            or not math.isfinite(raw_value)
            or (positive and raw_value <= 0)
        ):
            raise HarnessError(f"{description} contains an invalid number")
        numbers.append(float(raw_value))
    return numbers


def _validate_numerical_evidence(
    value: JsonValue | None,
    *,
    description: str,
    scenario: str | None = None,
) -> None:
    evidence = _json_object(value, description)
    expected_keys = {"relative_l1", "max_absolute", "repeat_exact"}
    if scenario is not None:
        expected_keys.add("scenario")
    _require_exact_keys(evidence, expected_keys, description)
    if scenario is not None and evidence.get("scenario") != scenario:
        raise HarnessError(f"{description} scenario order is invalid")
    for field_name in ("relative_l1", "max_absolute"):
        observed = evidence.get(field_name)
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(observed)
            or observed < 0
        ):
            raise HarnessError(f"{description} {field_name} is invalid")
    if (
        cast(float, evidence["relative_l1"]) > MAXIMUM_RELATIVE_L1_ERROR
        or cast(float, evidence["max_absolute"]) > MAXIMUM_ABSOLUTE_ERROR
    ):
        raise HarnessError(f"{description} exceeds the declared tolerance")
    if evidence.get("repeat_exact") is not True:
        raise HarnessError(f"{description} repeat evidence is invalid")


def _validate_stage_measurement(
    value: JsonValue | None,
    *,
    expected_stage: Literal["gate_up", "down"],
    expected_config: JsonObject,
    config: TuningRunConfig,
    timing_route_strata: Sequence[JsonObject],
) -> ValidatedStageMeasurement:
    measurement = _json_object(value, f"{expected_stage} measurement")
    _require_exact_keys(
        measurement,
        {
            "stage",
            "config",
            "sample_microseconds",
            "fallback_before_microseconds",
            "fallback_after_microseconds",
            "fallback_reference_microseconds",
            "relative_improvement_samples",
            "stable_minimum_five_percent_improvement",
            "numerical_evidence",
            "timing_strata",
        },
        f"{expected_stage} measurement",
    )
    if measurement.get("stage") != expected_stage:
        raise HarnessError("stage measurement label is invalid")
    observed_config = _json_object(
        measurement.get("config"), f"{expected_stage} measurement config"
    )
    _validate_kernel_config_object(observed_config, "stage measurement config")
    if observed_config != expected_config:
        raise HarnessError("stage measurement config differs from candidate")
    sample_count = config.tuning.independent_samples
    samples = _finite_numbers(
        measurement.get("sample_microseconds"),
        description="candidate timing samples",
        expected_length=sample_count,
        positive=True,
    )
    fallback_before = _finite_numbers(
        measurement.get("fallback_before_microseconds"),
        description="fallback-before samples",
        expected_length=sample_count,
        positive=True,
    )
    fallback_after = _finite_numbers(
        measurement.get("fallback_after_microseconds"),
        description="fallback-after samples",
        expected_length=sample_count,
        positive=True,
    )
    fallback_reference = _finite_numbers(
        measurement.get("fallback_reference_microseconds"),
        description="fallback-reference samples",
        expected_length=sample_count,
        positive=True,
    )
    improvements = _finite_numbers(
        measurement.get("relative_improvement_samples"),
        description="relative improvement samples",
        expected_length=sample_count,
        positive=False,
    )
    for index in range(sample_count):
        expected_fallback = (fallback_before[index] + fallback_after[index]) / 2
        expected_improvement = (expected_fallback - samples[index]) / expected_fallback
        if not math.isclose(
            fallback_reference[index], expected_fallback, rel_tol=1e-12, abs_tol=1e-9
        ) or not math.isclose(
            improvements[index], expected_improvement, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise HarnessError("derived timing samples are inconsistent")
    stable = measurement.get("stable_minimum_five_percent_improvement")
    expected_stable = not _is_fallback_kernel_config(observed_config) and all(
        improvement >= MINIMUM_SYNTHETIC_IMPROVEMENT for improvement in improvements
    )
    if type(stable) is not bool or stable is not expected_stable:
        raise HarnessError("stable-improvement decision is invalid")

    scenario_values = _required_json_list(
        measurement.get("numerical_evidence"), "scenario numerical evidence"
    )
    scenarios = ("uniform", "zero_resident", "mixed", "resident_skew")
    if len(scenario_values) != len(scenarios):
        raise HarnessError("scenario numerical evidence is incomplete")
    for scenario_value, scenario_name in zip(scenario_values, scenarios, strict=True):
        _validate_numerical_evidence(
            scenario_value,
            description="scenario numerical evidence",
            scenario=scenario_name,
        )

    timing_values = _required_json_list(
        measurement.get("timing_strata"), "stage timing strata"
    )
    if len(timing_values) != len(timing_route_strata):
        raise HarnessError("stage timing strata do not match route strata")
    stratum_samples: list[tuple[float, list[float], list[float], list[float]]] = []
    for timing_value, route_stratum in zip(
        timing_values, timing_route_strata, strict=True
    ):
        timing = _json_object(timing_value, "stage timing stratum")
        _require_exact_keys(
            timing,
            {
                "name",
                "resident_route_count",
                "resident_routes_per_token",
                "probability_weight",
                "sample_microseconds",
                "fallback_before_microseconds",
                "fallback_after_microseconds",
                "numerical_evidence",
            },
            "stage timing stratum",
        )
        for name in (
            "name",
            "resident_route_count",
            "resident_routes_per_token",
            "probability_weight",
        ):
            if timing.get(name) != route_stratum.get(name):
                raise HarnessError("stage timing stratum identity changed")
        stratum_candidate = _finite_numbers(
            timing.get("sample_microseconds"),
            description="stratum candidate samples",
            expected_length=sample_count,
            positive=True,
        )
        stratum_before = _finite_numbers(
            timing.get("fallback_before_microseconds"),
            description="stratum fallback-before samples",
            expected_length=sample_count,
            positive=True,
        )
        stratum_after = _finite_numbers(
            timing.get("fallback_after_microseconds"),
            description="stratum fallback-after samples",
            expected_length=sample_count,
            positive=True,
        )
        _validate_numerical_evidence(
            timing.get("numerical_evidence"),
            description="stratum numerical evidence",
        )
        stratum_samples.append(
            (
                cast(float, route_stratum["probability_weight"]),
                stratum_candidate,
                stratum_before,
                stratum_after,
            )
        )
    for aggregate, group_index in (
        (samples, 1),
        (fallback_before, 2),
        (fallback_after, 3),
    ):
        for sample_index, observed in enumerate(aggregate):
            expected = sum(
                weight * cast(list[float], stratum[group_index])[sample_index]
                for stratum in stratum_samples
                for weight in (stratum[0],)
            )
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9):
                raise HarnessError("aggregate timing differs from weighted strata")
    return ValidatedStageMeasurement(
        evidence=measurement,
        config=observed_config,
        median_microseconds=float(statistics.median(samples)),
        stably_faster=expected_stable,
    )


def _validate_anchor(
    value: JsonValue,
    *,
    expected_batch_size: int,
    config: TuningRunConfig,
    expected_gate_up_config: JsonObject,
    expected_down_config: JsonObject,
    expected_candidate_configs: Sequence[JsonObject],
) -> None:
    anchor = _json_object(value, "tuning anchor")
    _require_exact_keys(
        anchor,
        {
            "batch_size",
            "route_scenarios",
            "timing_route_strata",
            "selected",
            "candidate_records",
        },
        "tuning anchor",
    )
    if anchor.get("batch_size") != expected_batch_size:
        raise HarnessError("tuning anchor batch order differs from config")
    route_scenario_values = _required_json_list(
        anchor.get("route_scenarios"), "anchor route scenarios"
    )
    scenario_names = ("uniform", "zero_resident", "mixed", "resident_skew")
    if len(route_scenario_values) != len(scenario_names):
        raise HarnessError("anchor route scenarios are incomplete")
    for raw_scenario, expected_name in zip(
        route_scenario_values, scenario_names, strict=True
    ):
        scenario = _json_object(raw_scenario, "route scenario")
        _require_exact_keys(
            scenario,
            {
                "name",
                "global_route_sha256",
                "masked_route_sha256",
                "resident_route_count",
                "masked_cpu_route_count",
                "resident_routes_per_token",
            },
            "route scenario",
        )
        resident_count = scenario.get("resident_route_count")
        cpu_count = scenario.get("masked_cpu_route_count")
        resident_per_token = scenario.get("resident_routes_per_token")
        if (
            scenario.get("name") != expected_name
            or type(resident_count) is not int
            or type(cpu_count) is not int
            or resident_count < 0
            or cpu_count < 0
            or resident_count + cpu_count != expected_batch_size * GLM47_TOP_K
            or not isinstance(resident_per_token, (int, float))
            or isinstance(resident_per_token, bool)
            or not math.isclose(
                resident_per_token,
                resident_count / expected_batch_size,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or any(
                _SHA256_PATTERN.fullmatch(cast(str, scenario.get(hash_name, "")))
                is None
                for hash_name in ("global_route_sha256", "masked_route_sha256")
            )
        ):
            raise HarnessError("route scenario evidence is invalid")

    timing_values = _required_json_list(
        anchor.get("timing_route_strata"), "anchor timing route strata"
    )
    numerator = expected_batch_size * GLM47_TOP_K * config.tuning.resident_experts
    lower_count, remainder = divmod(numerator, GLM47_GLOBAL_EXPERTS)
    expected_strata = (
        ((lower_count, 1.0),)
        if remainder == 0
        else (
            (lower_count, 1 - remainder / GLM47_GLOBAL_EXPERTS),
            (lower_count + 1, remainder / GLM47_GLOBAL_EXPERTS),
        )
    )
    if len(timing_values) != len(expected_strata):
        raise HarnessError("anchor timing strata are incomplete")
    timing_strata: list[JsonObject] = []
    weights: list[float] = []
    for raw_stratum, (resident_count, expected_probability) in zip(
        timing_values, expected_strata, strict=True
    ):
        stratum = _json_object(raw_stratum, "anchor timing route stratum")
        _require_exact_keys(
            stratum,
            {
                "name",
                "resident_route_count",
                "resident_routes_per_token",
                "probability_weight",
                "global_route_sha256",
                "masked_route_sha256",
            },
            "anchor timing route stratum",
        )
        probability = stratum.get("probability_weight")
        resident_routes_per_token = stratum.get("resident_routes_per_token")
        if (
            stratum.get("name") != f"resident_total_{resident_count}"
            or stratum.get("resident_route_count") != resident_count
            or not isinstance(resident_routes_per_token, (int, float))
            or isinstance(resident_routes_per_token, bool)
            or not math.isclose(
                resident_routes_per_token,
                resident_count / expected_batch_size,
                rel_tol=0,
                abs_tol=1e-15,
            )
            or not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not math.isclose(
                probability, expected_probability, rel_tol=0, abs_tol=1e-15
            )
            or any(
                _SHA256_PATTERN.fullmatch(cast(str, stratum.get(name, ""))) is None
                for name in ("global_route_sha256", "masked_route_sha256")
            )
        ):
            raise HarnessError("anchor timing route stratum is invalid")
        weights.append(float(probability))
        timing_strata.append(stratum)
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise HarnessError("anchor timing stratum weights do not sum to one")

    candidate_values = _required_json_list(
        anchor.get("candidate_records"), "anchor candidate records"
    )
    if len(candidate_values) != len(expected_candidate_configs):
        raise HarnessError(
            "anchor candidate records do not cover the full search space"
        )
    candidate_keys: set[bytes] = set()
    stage_measurements: dict[
        Literal["gate_up", "down"], list[ValidatedStageMeasurement]
    ] = {"gate_up": [], "down": []}
    for candidate_index, (raw_candidate, expected_candidate_config) in enumerate(
        zip(candidate_values, expected_candidate_configs, strict=True)
    ):
        candidate = _json_object(raw_candidate, "candidate record")
        _require_exact_keys(
            candidate, {"config", "measurements", "rejections"}, "candidate record"
        )
        candidate_config = _json_object(candidate.get("config"), "candidate config")
        _validate_kernel_config_object(candidate_config, "candidate config")
        if candidate_config != expected_candidate_config:
            raise HarnessError(
                f"anchor candidate {candidate_index} differs from canonical search order"
            )
        candidate_key = _canonical_json_bytes(candidate_config, pretty=False)
        if candidate_key in candidate_keys:
            raise HarnessError("anchor contains duplicate candidate configs")
        candidate_keys.add(candidate_key)
        measurements = _required_json_list(
            candidate.get("measurements"), "candidate measurements"
        )
        rejections = _required_json_list(
            candidate.get("rejections"), "candidate rejections"
        )
        outcomes: list[str] = []
        for raw_measurement in measurements:
            measurement = _json_object(raw_measurement, "candidate measurement")
            stage = measurement.get("stage")
            if stage not in {"gate_up", "down"}:
                raise HarnessError("candidate measurement stage is invalid")
            outcomes.append(stage)
            typed_stage = cast(Literal["gate_up", "down"], stage)
            stage_measurements[typed_stage].append(
                _validate_stage_measurement(
                    measurement,
                    expected_stage=typed_stage,
                    expected_config=candidate_config,
                    config=config,
                    timing_route_strata=timing_strata,
                )
            )
        for raw_rejection in rejections:
            rejection = _json_object(raw_rejection, "candidate rejection")
            _require_exact_keys(
                rejection, {"stage", "category", "reason"}, "candidate rejection"
            )
            stage = rejection.get("stage")
            if (
                stage not in {"gate_up", "down"}
                or rejection.get("category") not in {"out_of_resources", "numerical"}
                or not isinstance(rejection.get("reason"), str)
                or not rejection.get("reason")
            ):
                raise HarnessError("candidate rejection is invalid")
            outcomes.append(stage)
        if sorted(outcomes) != ["down", "gate_up"]:
            raise HarnessError("candidate record lacks exactly one outcome per stage")

    best_by_stage: dict[
        Literal["gate_up", "down"], dict[int, ValidatedStageMeasurement]
    ] = {"gate_up": {}, "down": {}}
    for stage in cast(tuple[Literal["gate_up", "down"], ...], ("gate_up", "down")):
        for measurement in stage_measurements[stage]:
            if not (
                _is_fallback_kernel_config(measurement.config)
                or measurement.stably_faster
            ):
                continue
            block_size_m = cast(int, measurement.config["BLOCK_SIZE_M"])
            current = best_by_stage[stage].get(block_size_m)
            if (
                current is None
                or measurement.median_microseconds < current.median_microseconds
            ):
                best_by_stage[stage][block_size_m] = measurement
    shared_block_sizes = set(best_by_stage["gate_up"]) & set(best_by_stage["down"])
    if not shared_block_sizes:
        raise HarnessError("anchor has no admissible paired candidate")
    expected_block_size = min(
        shared_block_sizes,
        key=lambda block_size: (
            best_by_stage["gate_up"][block_size].median_microseconds
            + best_by_stage["down"][block_size].median_microseconds,
            block_size,
            _kernel_config_key(best_by_stage["gate_up"][block_size].config),
            _kernel_config_key(best_by_stage["down"][block_size].config),
        ),
    )
    expected_selected = {
        "gate_up": best_by_stage["gate_up"][expected_block_size],
        "down": best_by_stage["down"][expected_block_size],
    }

    selected = _json_object(anchor.get("selected"), "selected candidate")
    _require_exact_keys(
        selected,
        {
            "shared_block_size_m",
            "gate_up",
            "down",
            "gate_up_retained_fallback",
            "down_retained_fallback",
        },
        "selected candidate",
    )
    block_size_m = selected.get("shared_block_size_m")
    if type(block_size_m) is not int or block_size_m <= 0:
        raise HarnessError("selected shared BLOCK_SIZE_M is invalid")
    if block_size_m != expected_block_size:
        raise HarnessError("selected pair is not the deterministic admitted minimum")
    for stage in cast(tuple[Literal["gate_up", "down"], ...], ("gate_up", "down")):
        selected_measurement = _json_object(
            selected.get(stage), f"selected {stage} measurement"
        )
        selected_config = _json_object(
            selected_measurement.get("config"), f"selected {stage} config"
        )
        if selected_config.get("BLOCK_SIZE_M") != block_size_m:
            raise HarnessError("selected stages do not share BLOCK_SIZE_M")
        validated_selected = _validate_stage_measurement(
            selected_measurement,
            expected_stage=stage,
            expected_config=selected_config,
            config=config,
            timing_route_strata=timing_strata,
        )
        if validated_selected.evidence != expected_selected[stage].evidence:
            raise HarnessError("selected measurement is not the admitted candidate")
        retained = selected.get(f"{stage}_retained_fallback")
        if type(retained) is not bool or retained is not _is_fallback_kernel_config(
            selected_config
        ):
            raise HarnessError("selected fallback decision is invalid")
        expected_file_config = (
            expected_gate_up_config if stage == "gate_up" else expected_down_config
        )
        if selected_config != expected_file_config:
            raise HarnessError("selected config differs from emitted config file")


def _tuning_output_evidence(
    config: TuningRunConfig,
    result_descriptor: int,
    output_descriptor: int,
) -> JsonObject:
    if not stat.S_ISDIR(os.fstat(output_descriptor).st_mode):
        raise HarnessError("tuning output descriptor is not a directory")
    manifest_contents = _read_relative_file_at(
        output_descriptor,
        "manifest.json",
        "tuning manifest",
        MAXIMUM_MANIFEST_JSON_BYTES,
    )
    try:
        manifest = _json_object(
            cast(object, json.loads(manifest_contents)), "tuning manifest"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"tuning manifest is invalid JSON: {error}") from error
    stored_authorization = _read_json_at(
        result_descriptor,
        AUTHORIZATION_FILENAME,
        "stored tuner authorization",
        MAXIMUM_AUTHORIZATION_BYTES,
        require_read_only=True,
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "candidate",
            "deployment_admitted",
            "performance_comparable",
            "benchmark_scope",
            "production_path_reproduced",
            "limitations",
            "serving_admission_required",
            "tuner",
            "authorization",
            "output_contract",
            "shape",
            "workload",
            "route_contract",
            "measurement_contract",
            "runtime_contract",
            "runtime_parent_receipts",
            "runtime_observed",
            "config_files",
            "anchors",
        },
        "tuning manifest",
    )
    expected_flags = {
        "schema_version": 3,
        "artifact_type": "glm47_sglang_kt_fused_moe_candidate_v3",
        "candidate": True,
        "deployment_admitted": False,
        "performance_comparable": False,
        "benchmark_scope": "synthetic_separate_stage_kernels_only",
        "production_path_reproduced": False,
    }
    for field_name, expected in expected_flags.items():
        if manifest.get(field_name) != expected:
            raise HarnessError(f"tuning manifest {field_name} is invalid")
    if manifest.get("limitations") != [
        "alignment is prepared outside timed regions",
        "the pinned filtered activation and final reduction are not timed",
        "concurrent CPU AMX expert execution and resource contention are not reproduced",
        "resident-route strata preserve the uniform expected count but are synthetic",
        "stratum weights assume uniform global top-k routing, not a captured trace",
        "the untuned serving baseline profile does not consume this bundle",
        "the current serving_baseline profile strips SGLANG_* variables",
    ]:
        raise HarnessError("tuning manifest limitations changed")

    serving_gate = _json_object(
        manifest.get("serving_admission_required"), "serving admission gate"
    )
    if serving_gate != {
        "minimum_improvement": MINIMUM_SERVING_IMPROVEMENT,
        "metric": "matched end-to-end serving performance",
        "baseline_target_profile": "GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE",
        "baseline_profile_is_untuned": True,
        "baseline_profile_consumes_this_bundle": False,
        "profile": "future tuned profile with exact config SHA-256 binding",
        "consumer_chain": "KTEP gpu_method.apply -> SGLang fused_moe config loader",
        "required_environment_binding": (
            "SGLANG_MOE_CONFIG_DIR with exact config file SHA-256"
        ),
        "current_serving_baseline_strips_sglang_environment": True,
    }:
        raise HarnessError("tuning manifest serving admission gate is invalid")

    runtime_contract = _verified_runtime_contract(config)
    if manifest.get("runtime_contract") != runtime_contract:
        raise HarnessError("tuning manifest runtime contract differs from receipts")
    contract_receipts = _json_object(
        runtime_contract.get("receipt_bindings"), "runtime contract receipts"
    )
    runtime_parent_receipts = _json_object(
        manifest.get("runtime_parent_receipts"), "runtime parent receipts"
    )
    if set(runtime_parent_receipts) != {
        "runtime_install",
        "runtime_build",
        "kernel_validation",
    }:
        raise HarnessError("runtime parent receipt set is invalid")
    for receipt_name in sorted(runtime_parent_receipts):
        receipt = _json_object(
            runtime_parent_receipts.get(receipt_name),
            f"{receipt_name} parent receipt",
        )
        if receipt != {
            "sha256": _json_object(
                contract_receipts.get(receipt_name),
                f"{receipt_name} contract binding",
            ).get("sha256"),
            "binding": contract_receipts.get(receipt_name),
        }:
            raise HarnessError(f"{receipt_name} parent receipt binding is invalid")
    runtime_observed = _json_object(
        manifest.get("runtime_observed"), "observed tuning runtime"
    )
    if runtime_observed != {
        "torch_version": runtime_contract["torch_version"],
        "cuda_version": runtime_contract["torch_cuda_version"],
        "triton_version": runtime_contract["triton_version"],
        "sglang_revision": runtime_contract["sglang_revision"],
        "ktransformers_revision": runtime_contract["ktransformers_revision"],
        "device_name": runtime_contract["gpu_name"],
        "gpu_uuid": runtime_contract["gpu_uuid"],
    }:
        raise HarnessError("tuning manifest observed runtime differs from contract")

    shape = _json_object(manifest.get("shape"), "tuning shape")
    if shape != {
        "H": GLM47_HIDDEN_SIZE,
        "N": GLM47_INTERMEDIATE_SIZE,
        "E": config.tuning.resident_experts,
        "global_experts": GLM47_GLOBAL_EXPERTS,
        "top_k": GLM47_TOP_K,
    }:
        raise HarnessError("tuning manifest shape differs from GLM-4.7 contract")
    workload = _json_object(manifest.get("workload"), "tuning workload")
    if workload != {
        "resident_experts": config.tuning.resident_experts,
        "batch_sizes": list(config.tuning.batch_sizes),
        "global_experts": GLM47_GLOBAL_EXPERTS,
        "top_k": GLM47_TOP_K,
    }:
        raise HarnessError("tuning manifest workload differs from config")
    route_contract = _json_object(manifest.get("route_contract"), "route contract")
    if route_contract != {
        "cpu_experts_are_masked_to": -1,
        "resident_global_expert_ids": list(range(config.tuning.resident_experts)),
        "seed": config.tuning.seed,
        "timed_scenarios": "deterministic expected total resident-route count strata",
        "decode_actual_resident_gemm_timed": True,
        "correctness_scenarios": [
            "uniform",
            "zero_resident",
            "mixed",
            "resident_skew",
        ],
        "uniform_expected_resident_routes_per_token": (
            GLM47_TOP_K * config.tuning.resident_experts / GLM47_GLOBAL_EXPERTS
        ),
    }:
        raise HarnessError("tuning manifest route contract is invalid")
    measurement = _json_object(
        manifest.get("measurement_contract"), "measurement contract"
    )
    expected_measurement_values = {
        "batch_sizes": list(config.tuning.batch_sizes),
        "warmup_iterations": config.tuning.warmup_iterations,
        "measurement_iterations": config.tuning.measurement_iterations,
        "independent_samples": config.tuning.independent_samples,
        "search_profile": config.tuning.search_profile,
        "jit_compilation_excluded": True,
        "timing_source": "CUDA events",
        "candidate_order": "canonical fixed order",
        "drift_mitigation": "fallback-before/candidate/fallback-after",
        "minimum_stable_synthetic_improvement": 0.05,
        "selection_fallback": (
            "retain deployed fallback unless every paired sample meets threshold"
        ),
    }
    if any(
        measurement.get(name) != expected
        for name, expected in expected_measurement_values.items()
    ) or set(measurement) != set(expected_measurement_values) | {
        "relative_l1_tolerance",
        "max_absolute_tolerance",
    }:
        raise HarnessError("tuning manifest measurement contract is invalid")
    for tolerance_name in ("relative_l1_tolerance", "max_absolute_tolerance"):
        tolerance = measurement.get(tolerance_name)
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not (0 < tolerance <= 1)
        ):
            raise HarnessError("tuning manifest numerical tolerance is invalid")

    authorization = _json_object(
        manifest.get("authorization"), "tuning authorization evidence"
    )
    _require_exact_keys(
        authorization,
        {
            "authorization_sha256",
            "evidence",
            "evidence_sha256",
            "output_directory_identity",
        },
        "tuning authorization evidence",
    )
    evidence = _json_object(authorization.get("evidence"), "authorization evidence")
    _require_exact_keys(
        evidence,
        {
            "schema_version",
            "authorization_sha256",
            "lease_id",
            "run_id",
            "namespace",
            "result_directory",
            "tuning_output_directory",
            "output_directory_descriptor",
            "output_directory_identity",
            "tuner_process_argv",
            "launcher_argv",
            "receipt_sha256",
            "child_environment",
            "parent_process",
            "lock_path",
            "lease_path",
            "output_created_by_harness",
            "runtime_contract",
            "verified_authorization",
        },
        "authorization evidence",
    )
    verified_authorization = _json_object(
        evidence.get("verified_authorization"), "verified authorization"
    )
    if verified_authorization != stored_authorization:
        raise HarnessError(
            "manifest authorization differs from stored read-only authorization"
        )
    _require_exact_keys(
        verified_authorization,
        {
            "schema_version",
            "authorization_kind",
            "lease_id",
            "run_id",
            "namespace",
            "lock_path",
            "lease_path",
            "result_directory",
            "harness_process",
            "process_ownership",
            "owner_token",
            "launcher_argv",
            "tuner_process_argv",
            "tuning_output_directory",
            "output_directory_descriptor",
            "output_directory_identity",
            "cache_directories",
            "child_environment",
            "runtime_contract",
            "receipt_bindings",
            "experiment_context",
            "created_at",
        },
        "verified authorization",
    )
    receipt_bindings = _json_object(
        verified_authorization.get("receipt_bindings"),
        "verified authorization receipt bindings",
    )
    expected_receipt_names = {
        "runtime_python",
        "runtime_install",
        "runtime_build",
        "kernel_validation",
        "tuner_script",
        "source_identity",
        "tuning_config",
        "preflight",
        "telemetry_before",
    }
    if set(receipt_bindings) != expected_receipt_names:
        raise HarnessError("verified authorization receipt binding set is invalid")
    expected_receipt_sha256: JsonObject = {
        name: _json_object(
            receipt_bindings.get(name), f"verified {name} receipt binding"
        ).get("sha256")
        for name in expected_receipt_names
    }
    observed_receipt_sha256 = _json_object(
        evidence.get("receipt_sha256"), "authorization receipt digests"
    )
    directory_identity = _directory_identity(output_descriptor)
    if (
        authorization.get("evidence_sha256")
        != hashlib.sha256(_canonical_json_bytes(evidence, pretty=False)).hexdigest()
        or authorization.get("authorization_sha256")
        != evidence.get("authorization_sha256")
        or authorization.get("output_directory_identity") != directory_identity
        or evidence.get("output_directory_identity") != directory_identity
        or evidence.get("output_directory_descriptor") != output_descriptor
        or evidence.get("runtime_contract") != runtime_contract
        or evidence.get("output_created_by_harness") is not True
        or evidence.get("schema_version") != 1
        or evidence.get("run_id") != config.run_id
        or evidence.get("lease_id") != verified_authorization.get("lease_id")
        or evidence.get("namespace") != verified_authorization.get("namespace")
        or evidence.get("result_directory")
        != verified_authorization.get("result_directory")
        or evidence.get("tuning_output_directory")
        != str(Path(config.result_directory) / TUNING_OUTPUT_DIRECTORY_NAME)
        or evidence.get("authorization_sha256")
        != hashlib.sha256(
            _canonical_json_bytes(verified_authorization, pretty=False)
        ).hexdigest()
        or observed_receipt_sha256 != expected_receipt_sha256
        or verified_authorization.get("schema_version") != 3
        or verified_authorization.get("authorization_kind")
        != "active-benchmark-lease-inherited-pipe-v3"
        or verified_authorization.get("run_id") != config.run_id
        or verified_authorization.get("namespace") != config.namespace
        or verified_authorization.get("tuning_output_directory")
        != evidence.get("tuning_output_directory")
        or verified_authorization.get("output_directory_descriptor")
        != output_descriptor
        or verified_authorization.get("output_directory_identity") != directory_identity
        or verified_authorization.get("runtime_contract") != runtime_contract
        or verified_authorization.get("tuner_process_argv")
        != evidence.get("tuner_process_argv")
        or verified_authorization.get("launcher_argv") != evidence.get("launcher_argv")
        or verified_authorization.get("child_environment")
        != evidence.get("child_environment")
        or verified_authorization.get("harness_process")
        != evidence.get("parent_process")
        or verified_authorization.get("lock_path") != evidence.get("lock_path")
        or verified_authorization.get("lease_path") != evidence.get("lease_path")
        or verified_authorization.get("experiment_context")
        != {
            "contextual_model": config.contextual_model.model_dump(mode="json"),
            "model_snapshot_consumed": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "binding_establishes_model_verification": False,
        }
        or verified_authorization.get("process_ownership")
        != {
            "mode": "inherit-lease-child-process-group",
            "lease_child_pid": _json_object(
                verified_authorization.get("harness_process"),
                "verified harness process",
            ).get("pid"),
            "process_group_id": _json_object(
                verified_authorization.get("harness_process"),
                "verified harness process",
            ).get("pid"),
            "outer_wrapper_cleanup": "killpg",
        }
    ):
        raise HarnessError("tuning manifest authorization evidence is invalid")
    output_contract = _json_object(
        manifest.get("output_contract"), "tuning output contract"
    )
    if output_contract != {
        "descriptor_anchored": True,
        "harness_created_empty_directory": True,
        "semantic_path": evidence.get("tuning_output_directory"),
        "identity": _directory_identity(output_descriptor),
    }:
        raise HarnessError("tuning manifest output contract is invalid")

    tuner = _json_object(manifest.get("tuner"), "tuner identity")
    expected_tuner_path = str(Path(config.source_deployment_root) / TUNER_RELATIVE_PATH)
    if tuner != {
        "path": expected_tuner_path,
        "sha256": _sha256_file(Path(expected_tuner_path)),
    }:
        raise HarnessError("tuning manifest tuner identity differs from capsule")
    if tuner != _json_object(
        receipt_bindings.get("tuner_script"), "authorized tuner binding"
    ):
        raise HarnessError("manifest tuner differs from stored authorization")
    expected_candidate_configs = _canonical_search_configs(
        config,
        Path(expected_tuner_path),
        cast(str, tuner["sha256"]),
    )

    config_files = _json_object(manifest.get("config_files"), "config files")
    _require_exact_keys(config_files, {"gate_up", "down"}, "config files")
    expected_paths: set[str] = {"manifest.json"}
    parsed_config_files: dict[str, dict[int, JsonObject]] = {}
    for stage in ("gate_up", "down"):
        binding = _json_object(config_files.get(stage), f"{stage} config binding")
        _require_exact_keys(
            binding,
            {"relative_path", "sha256", "shape", "batch_keys"},
            f"{stage} config binding",
        )
        relative_path = _required_bounded_string(
            binding.get("relative_path"), f"{stage} config path"
        )
        expected_name = (
            f"E={config.tuning.resident_experts},N={GLM47_INTERMEDIATE_SIZE},"
            f"device_name={config.host.gpu.expected_name.replace(' ', '_')}"
            f"{'_down' if stage == 'down' else ''}.json"
        )
        expected_relative_path = (
            f"configs/triton_{config.tuning.expected_triton_version.replace('.', '_')}/"
            f"{expected_name}"
        )
        if (
            relative_path != expected_relative_path
            or binding.get("shape")
            != {"E": config.tuning.resident_experts, "N": GLM47_INTERMEDIATE_SIZE}
            or binding.get("batch_keys") != list(config.tuning.batch_sizes)
        ):
            raise HarnessError(f"{stage} config binding differs from workload")
        contents = _read_relative_file_at(
            output_descriptor,
            relative_path,
            f"{stage} config file",
            MAXIMUM_CONFIG_JSON_BYTES,
        )
        if binding.get("sha256") != hashlib.sha256(contents).hexdigest():
            raise HarnessError(f"{stage} config digest differs from manifest")
        parsed_config_files[stage] = _validate_kernel_config_file(contents, config)
        expected_paths.add(relative_path)

    anchors = _required_json_list(manifest.get("anchors"), "tuning anchors")
    if len(anchors) != len(config.tuning.batch_sizes):
        raise HarnessError("tuning anchors do not exactly cover batch sizes")
    for anchor_value, batch_size in zip(
        anchors, config.tuning.batch_sizes, strict=True
    ):
        _validate_anchor(
            anchor_value,
            expected_batch_size=batch_size,
            config=config,
            expected_gate_up_config=parsed_config_files["gate_up"][batch_size],
            expected_down_config=parsed_config_files["down"][batch_size],
            expected_candidate_configs=expected_candidate_configs,
        )

    files, directories = _output_tree(output_descriptor)
    expected_directories = {
        "configs",
        f"configs/triton_{config.tuning.expected_triton_version.replace('.', '_')}",
    }
    if set(files) != expected_paths or directories != expected_directories:
        raise HarnessError("tuning output tree contains missing or unexpected entries")
    file_evidence: list[JsonValue] = [
        {"relative_path": path, "sha256": digest, "size_bytes": size}
        for path, (digest, size) in sorted(files.items())
    ]
    return {
        "manifest_sha256": hashlib.sha256(manifest_contents).hexdigest(),
        "files": file_evidence,
        "performance_comparable": False,
        "candidate_only": True,
        "adoption_gate": {
            "kind": "warm_serving",
            "minimum_improvement_percent": 100 * MINIMUM_SERVING_IMPROVEMENT,
            "status": "required_not_run",
        },
    }


def _telemetry(config: TuningRunConfig) -> JsonObject:
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": _nvidia_query_without_exclusivity(config),
        "cpu": _cpu_telemetry(config),
        "hca": _hca_telemetry(config),
    }


def _nvidia_query_without_exclusivity(config: TuningRunConfig) -> JsonObject:
    fields = (
        "uuid",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "clocks.sm",
        "power.draw",
    )
    command = _run_capture(
        (
            "/usr/bin/nvidia-smi",
            f"--id={config.host.gpu.uuid}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        "GPU telemetry",
    )
    _require_success(command, "GPU telemetry")
    return cast(JsonObject, cast(object, command))


def _empty_runtime_metadata(
    result_descriptor: int, config: TuningRunConfig, owner_token: str
) -> None:
    _write_runtime_metadata(result_descriptor, config, owner_token, {})


def _write_benchmark_result(
    result_descriptor: int,
    config: TuningRunConfig,
    *,
    cleanup_succeeded: bool,
    owned_processes: Sequence[ProcessIdentity],
) -> None:
    value: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "cleanup_succeeded": cleanup_succeeded,
        "owned_processes": cast(list[JsonValue], cast(object, list(owned_processes))),
    }
    _atomic_write_json_at(result_descriptor, BENCHMARK_RESULT_FILENAME, value)


def run_tuning_child(
    config: TuningRunConfig,
    *,
    lease_path: Path,
    lock_path: Path,
    result_dir: Path,
    signal_state: ManagedSignalState | None = None,
) -> int:
    managed_signal_state = signal_state or ManagedSignalState()
    result_descriptor = _result_descriptor(config, result_dir)
    owner_token = os.urandom(32).hex()
    owned_processes: list[ProcessIdentity] = []
    output_descriptor: int | None = None
    cleanup_succeeded = True
    completion: JsonObject = {
        "schema_version": 1,
        "status": "preflight_failed",
        "error": None,
    }
    return_code = 1
    try:
        _empty_runtime_metadata(result_descriptor, config, owner_token)
        lease = _validate_active_lease(
            config,
            lease_path=lease_path,
            lock_path=lock_path,
            result_dir=result_dir,
        )
        managed_signal_state.raise_if_interrupted()
        source_before = _verify_source_capsule(config, lease)
        _verify_executable(config.runtime_python, "runtime Python")
        _verify_file(config.numactl_executable, "numactl executable")
        runtime_contract = _verified_runtime_contract(config)
        preflight = _collect_preflight(config, result_dir)
        managed_signal_state.raise_if_interrupted()
        preflight["source_identity"] = source_before
        preflight["experiment_context"] = {
            "contextual_model": cast(
                JsonObject,
                cast(object, config.contextual_model.model_dump(mode="json")),
            ),
            "model_snapshot_consumed": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "binding_establishes_model_verification": False,
        }
        preflight["runtime_contract"] = runtime_contract
        _atomic_write_json_at(result_descriptor, PREFLIGHT_FILENAME, preflight)
        _make_read_only_at(result_descriptor, PREFLIGHT_FILENAME)
        _atomic_write_json_at(
            result_descriptor, TELEMETRY_BEFORE_FILENAME, _telemetry(config)
        )
        _make_read_only_at(result_descriptor, TELEMETRY_BEFORE_FILENAME)
        outcome = _run_owned_tuner(
            config,
            lease=lease,
            result_descriptor=result_descriptor,
            result_dir=result_dir,
            lease_path=lease_path,
            lock_path=lock_path,
            owner_token=owner_token,
            signal_state=managed_signal_state,
        )
        owned_processes = list(outcome.owned_processes)
        output_descriptor = outcome.output_descriptor
        cleanup_succeeded = outcome.cleanup_succeeded
        if not cleanup_succeeded:
            raise HarnessError("owned tuner processes remain after cleanup")
        if outcome.error is not None:
            raise HarnessError(f"owned tuner execution failed: {outcome.error}")
        if outcome.return_code != 0:
            raise HarnessError(f"tuner exited with return code {outcome.return_code}")
        managed_signal_state.raise_if_interrupted()
        output_evidence = _tuning_output_evidence(
            config, result_descriptor, output_descriptor
        )
        if _verify_source_capsule(config, lease) != source_before:
            raise HarnessError("source identity changed during tuning")
        _atomic_write_json_at(
            result_descriptor, TELEMETRY_AFTER_FILENAME, _telemetry(config)
        )
        _make_read_only_at(result_descriptor, TELEMETRY_AFTER_FILENAME)
        completion = {
            "schema_version": 1,
            "status": "completed",
            "error": None,
            "tuner": outcome.completion,
            "output": output_evidence,
            "cleanup_succeeded": True,
            "candidate_only": True,
            "performance_comparable": False,
            "adoption_gate": {
                "kind": "warm_serving",
                "minimum_improvement_percent": 3.0,
                "status": "required_not_run",
            },
            "receipt_authority": "lease_parent_harness",
            "evidence_sha256": {
                "preflight": _sha256_file_at(result_descriptor, PREFLIGHT_FILENAME),
                "telemetry_before": _sha256_file_at(
                    result_descriptor, TELEMETRY_BEFORE_FILENAME
                ),
                "telemetry_after": _sha256_file_at(
                    result_descriptor, TELEMETRY_AFTER_FILENAME
                ),
                "authorization": _sha256_file_at(
                    result_descriptor, AUTHORIZATION_FILENAME
                ),
            },
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return_code = 0
    except BaseException as error:
        interrupted_signal = managed_signal_state.first_signal_number
        completion = {
            "schema_version": 1,
            "status": (
                "cleanup_failed"
                if not cleanup_succeeded
                else ("interrupted" if interrupted_signal is not None else "failed")
            ),
            "error": f"{type(error).__name__}: {error}",
            "interrupted_signal": interrupted_signal,
            "cleanup_succeeded": cleanup_succeeded,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return_code = 1 if interrupted_signal is None else 128 + interrupted_signal
    finally:
        with contextlib.suppress(Exception):
            _atomic_write_json_at(result_descriptor, COMPLETION_FILENAME, completion)
            _make_read_only_at(result_descriptor, COMPLETION_FILENAME)
        _write_benchmark_result(
            result_descriptor,
            config,
            cleanup_succeeded=cleanup_succeeded,
            owned_processes=owned_processes,
        )
        _make_read_only_at(result_descriptor, BENCHMARK_RESULT_FILENAME)
        if output_descriptor is not None:
            os.close(output_descriptor)
        os.close(result_descriptor)
    return return_code


class ChildArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path


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


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an immutable leased GLM-4.7 fused-MoE tuning run"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-duration-seconds", type=float, required=True)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    normalized = tuple(arguments if arguments is not None else sys.argv[1:])
    try:
        if normalized and normalized[0] == "prepare-lease":
            parsed = _prepare_parser().parse_args(
                normalized[1:], namespace=PrepareArguments()
            )
            preparation = prepare_lease(
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
                _canonical_json_bytes(
                    {
                        **preparation.model_dump(mode="json"),
                        "benchmark_lease_shell_command": shlex.join(
                            preparation.benchmark_lease_argv
                        ),
                    }
                ).decode("utf-8"),
                end="",
            )
            return 0
        parsed = _child_parser().parse_args(normalized, namespace=ChildArguments())
        config = _load_config(parsed.config)
        signal_state = ManagedSignalState()
        managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous_handlers = {
            signal_number: signal.getsignal(signal_number)
            for signal_number in managed_signals
        }

        def record_signal(signal_number: int, _frame: FrameType | None) -> None:
            signal_state.record(signal_number)

        try:
            for signal_number in managed_signals:
                signal.signal(signal_number, record_signal)
            return run_tuning_child(
                config,
                lease_path=parsed.lease_path,
                lock_path=parsed.lock_path,
                result_dir=parsed.result_dir,
                signal_state=signal_state,
            )
        finally:
            for signal_number, previous_handler in previous_handlers.items():
                signal.signal(signal_number, previous_handler)
    except (HarnessError, OSError, ValueError) as error:
        print(f"GLM-4.7 tuning harness failed: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
