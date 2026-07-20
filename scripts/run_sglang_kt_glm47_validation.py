#!/usr/bin/env python3
"""Prepare and run one leased GLM-4.7 kernel/model validation transaction.

``prepare-lease`` creates two immutable deployments from the current source
snapshot, emits strict metadata for ``benchmark_lease.py``, and prints the
exact wrapper command.  The default mode is the direct lease child.  It creates
a fresh kernel receipt, creates a pinned process specification, and optionally
runs the CPU-control or hybrid model validator.

This harness never enables a profiler.  It rejects the unsafe VTune SEP/PAX
drivers, profiler-related process state, and inherited profiler controls before
starting any validation child.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import functools
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import IO, Literal, Protocol, TypeAlias, cast

from pydantic import (
    BaseModel,
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

from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (  # noqa: E402
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
)
from scripts.sglang_kt_glm47_live import (  # noqa: E402
    calculate_validator_bundle,
    validator_bundle_paths,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
ValidationPhase: TypeAlias = Literal[
    "kernel", "cpu_control", "hybrid", "serving_baseline"
]

RESULT_DIRECTORY_FD_ENVIRONMENT = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
RUNTIME_METADATA_FILENAME = "runtime-metadata.json"
BENCHMARK_RESULT_FILENAME = "benchmark-result.json"
CHILD_MANIFEST_FILENAME = "manifest.json"
DEPLOYMENT_RECEIPT_FILENAME = "deployment-receipt.json"
IMMUTABLE_CONFIG_RELATIVE_PATH = Path("orchestrator/run-config.json")
DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_RESULT_ROOT = Path("/var/lib/exo/benchmarks")
REQUIRED_SCRATCH_ROOT = Path("/var/lib/exo/validation-scratch")
CGROUP_FILESYSTEM_ROOT = Path("/sys/fs/cgroup")
SYSTEMD_RUN_EXECUTABLE = Path("/usr/bin/systemd-run")
SYSTEMD_SLICE = "system.slice"
SYSTEMD_DELEGATE_SUBGROUP = "supervisor"
MAXIMUM_JSON_BYTES = 16 * 1024 * 1024
MINIMUM_CLEANUP_GRACE_SECONDS = 180.0
LEASE_CHILD_BIND_SECONDS = 5.0
LEASE_HEARTBEAT_MAX_AGE = timedelta(minutes=2)
CLEANUP_QUIET_SECONDS = 0.5
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}")
_SYSTEMD_UNIT = re.compile(r"exo-glm47-[0-9a-f]{32}\.service")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_PCI_BDF = re.compile(
    r"(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):"
    r"(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})\."
    r"(?P<function>[0-7])"
)
_PROFILER_ENVIRONMENT_MARKERS = (
    "AMPLXE",
    "INTEL_LIBITTNOTIFY",
    "ITT_",
    "PROFILER",
    "SEP5",
    "SGLANG_TORCH_PROFILER",
    "VTUNE",
)
_PROFILER_COMMAND_MARKERS = (
    "amplxe",
    "sep5",
    "vtune",
)
_CONFLICTING_EXECUTABLES = frozenset(
    {
        "all_gather_perf",
        "all_reduce_perf",
        "aria2c",
        "b3sum",
        "exo",
        "fio",
        "hf",
        "ib_read_bw",
        "ib_send_bw",
        "ib_write_bw",
        "llama-server",
        "nccl-tests",
        "ollama",
        "rclone",
        "rsync",
        "sglang",
        "sha256sum",
        "text-generation-launcher",
        "tritonserver",
        "uvicorn",
        "vllm",
    }
)
_CONFLICTING_PYTHON_MODULE_PREFIXES = (
    "exo",
    "ktransformers.server",
    "sglang.launch_server",
    "vllm.entrypoints",
)
_REQUIRED_AMX_FEATURES = frozenset(
    {"amx_bf16", "amx_int8", "amx_tile", "avx512_bf16", "avx512f"}
)
_ORCHESTRATOR_SCRIPT_PATHS = (
    Path("scripts/benchmark_host_guard.py"),
    Path("scripts/benchmark_lease.py"),
    Path("scripts/build_sglang_kt_runtime.py"),
    Path("scripts/create_sglang_kt_glm47_validation_process_spec.py"),
    Path("scripts/prepare_sglang_kt_source.py"),
    Path("scripts/run_sglang_kt_glm47_serving_benchmark.py"),
    Path("scripts/run_sglang_kt_glm47_validation.py"),
    Path("scripts/sglang_kt_glm47_live.py"),
    Path("scripts/sglang_kt_glm47_serving_client.py"),
    Path("scripts/validate_sglang_kt_runtime.py"),
)
MODEL_CONTRACT_RELATIVE_PATH = (
    Path("src/exo/worker/sglang_kt/manifests")
    / GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME
)


class Glm47HarnessError(RuntimeError):
    """Expected fail-closed harness error."""


class ManagedSignalError(Glm47HarnessError):
    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received managed signal {signal_number}")
        self.signal_number = signal_number


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _lexical_absolute_path(value: str, description: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\0" in value
        or value.startswith("//")
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or path.as_posix() != value
        or os.path.normpath(value) != value
    ):
        raise ValueError(f"{description} must be a normalized absolute path")
    return value


class Endpoint(StrictModel):
    ip: str
    port: int = Field(ge=1, le=65535)

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        address = ipaddress.IPv4Address(value)
        if str(address) != value or address.is_unspecified or address.is_multicast:
            raise ValueError("endpoint must use a concrete canonical IPv4 address")
        return value

    @property
    def argument(self) -> str:
        return f"{self.ip}:{self.port}"


class ArtifactBinding(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _lexical_absolute_path(value, "artifact path")


class ExecutableBinding(ArtifactBinding):
    symlink_chain: tuple[str, ...]

    @model_validator(mode="after")
    def require_explicit_chain(self) -> "ExecutableBinding":
        if not self.symlink_chain or any(
            not value or "\0" in value for value in self.symlink_chain
        ):
            raise ValueError("runtime executable requires an explicit symlink chain")
        return self


class SourceConfig(StrictModel):
    repository: str
    deployment_root: str

    @field_validator("repository", "deployment_root")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _lexical_absolute_path(value, "source path")

    @model_validator(mode="after")
    def distinct_paths(self) -> "SourceConfig":
        repository = Path(self.repository)
        deployment = Path(self.deployment_root)
        if repository == deployment or repository in deployment.parents:
            raise ValueError("source deployment must be outside the mutable repository")
        return self


class GpuBinding(StrictModel):
    uuid: str
    pci_address: str

    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        if _GPU_UUID.fullmatch(value) is None:
            raise ValueError("GPU UUID must be complete")
        return "GPU-" + value.removeprefix("GPU-").lower()

    @field_validator("pci_address")
    @classmethod
    def validate_pci_address(cls, value: str) -> str:
        normalized = value.lower()
        match = _PCI_BDF.fullmatch(normalized)
        if match is None:
            raise ValueError("GPU PCI address must be canonical")
        domain = int(match.group("domain"), 16)
        if domain > 0xFFFF:
            raise ValueError("GPU PCI domain exceeds the Linux PCI domain range")
        return (
            f"{domain:08x}:{match.group('bus')}:"
            f"{match.group('device')}.{match.group('function')}"
        )


class HcaBinding(StrictModel):
    device: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    port: int = Field(gt=0)
    gid_index: int = Field(default=0, ge=0)
    gid: str

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.IPv6Address(value)
        if address.is_unspecified or int(address) & ((1 << 64) - 1) == 0:
            raise ValueError("HCA GID must be port-specific")
        return address.exploded


class HostBinding(StrictModel):
    hostname: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    node_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    gpu: GpuBinding
    cpu_cores: tuple[int, ...]
    memory_nodes: tuple[int, ...]
    threads_per_subpool: tuple[int, ...]
    cpu_infer_threads: int = Field(gt=0)
    threadpool_count: int = Field(gt=0)
    hca_bindings: tuple[HcaBinding, ...]

    @model_validator(mode="after")
    def validate_resources(self) -> "HostBinding":
        for name, values in (
            ("cpu_cores", self.cpu_cores),
            ("memory_nodes", self.memory_nodes),
        ):
            if not values or values != tuple(sorted(set(values))) or values[0] < 0:
                raise ValueError(f"{name} must be nonempty, sorted, and unique")
        if (
            len(self.threads_per_subpool) != len(self.memory_nodes)
            or any(value <= 0 for value in self.threads_per_subpool)
            or sum(self.threads_per_subpool) != self.cpu_infer_threads
            or self.threadpool_count != len(self.threads_per_subpool)
        ):
            raise ValueError("thread pools must exactly cover the selected NUMA nodes")
        if not self.hca_bindings or len(
            {(binding.device, binding.port) for binding in self.hca_bindings}
        ) != len(self.hca_bindings):
            raise ValueError("HCA bindings must be nonempty and unique")
        return self


class TimeoutConfig(StrictModel):
    generator_seconds: float = Field(gt=0)
    kernel_seconds: float = Field(gt=0)
    model_seconds: float = Field(gt=0)
    cleanup_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def finite_values(self) -> "TimeoutConfig":
        if not all(
            math.isfinite(value)
            for value in (
                self.generator_seconds,
                self.kernel_seconds,
                self.model_seconds,
                self.cleanup_seconds,
            )
        ):
            raise ValueError("timeouts must be finite")
        return self


class ValidationConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str
    namespace: str
    phase: ValidationPhase
    profiler: Literal["none"]
    hca_requirement: Literal["active", "metadata_only"]
    result_directory: str
    scratch_directory: str
    source: SourceConfig
    runtime_python: ExecutableBinding
    numactl_executable: str
    build_receipt: ArtifactBinding
    model_path: str
    model_contract: ArtifactBinding
    host: HostBinding
    distributed_coordinator: Endpoint
    service_endpoint: Endpoint
    reserved_ports: tuple[int, ...]
    resident_gpu_experts: int = Field(ge=0, le=4)
    timeouts: TimeoutConfig

    @field_validator(
        "run_id",
        "namespace",
    )
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("identifier contains unsupported characters")
        return value

    @field_validator(
        "result_directory",
        "scratch_directory",
        "numactl_executable",
        "model_path",
    )
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        return _lexical_absolute_path(value, "configured path")

    @model_validator(mode="after")
    def validate_contract(self) -> "ValidationConfig":
        if Path(self.result_directory).name != self.run_id:
            raise ValueError("result_directory must end with the exact run_id")
        scratch = Path(self.scratch_directory)
        if scratch.name != self.run_id or scratch.parent != REQUIRED_SCRATCH_ROOT:
            raise ValueError(
                "scratch_directory must equal /var/lib/exo/validation-scratch/run_id"
            )
        expected_ports = tuple(
            sorted(
                {
                    self.distributed_coordinator.port,
                    self.service_endpoint.port,
                }
            )
        )
        if self.reserved_ports != expected_ports or len(self.reserved_ports) != 2:
            raise ValueError(
                "reserved_ports must exactly contain distinct coordinator/service ports"
            )
        if self.phase == "cpu_control" and self.resident_gpu_experts != 0:
            raise ValueError("CPU-control validation requires zero GPU experts")
        if self.phase in {"hybrid", "serving_baseline"} and not (
            1 <= self.resident_gpu_experts <= 4
        ):
            raise ValueError(
                f"{self.phase} validation requires one to four GPU experts"
            )
        expected_contract_path = (
            Path(self.source.deployment_root)
            / "orchestrator"
            / MODEL_CONTRACT_RELATIVE_PATH
        )
        if (
            Path(self.model_contract.path) != expected_contract_path
            or self.model_contract.sha256 != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
        ):
            raise ValueError(
                "model_contract must bind the pinned immutable deployment artifact"
            )
        paths = {
            self.result_directory,
            self.scratch_directory,
            self.source.repository,
            self.source.deployment_root,
            self.runtime_python.path,
            self.numactl_executable,
            self.build_receipt.path,
            self.model_path,
            self.model_contract.path,
        }
        if len(paths) != 9:
            raise ValueError("configured artifact and working paths must be distinct")
        return self


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    dirty_file_hashes: dict[str, str]


@dataclass(frozen=True)
class DeploymentIdentity:
    root: str
    orchestrator_sha256: str
    validator_sha256: str
    validator_files: tuple[dict[str, JsonValue], ...]
    source: SourceIdentity


@dataclass(frozen=True)
class LeasePreparation:
    metadata: JsonObject
    metadata_output: str
    child_argv: tuple[str, ...]
    benchmark_lease_argv: tuple[str, ...]
    systemd_unit_name: str
    deployment: DeploymentIdentity
    minimum_cleanup_grace_seconds: float


@dataclass(frozen=True)
class OwnedProcess:
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    transport_pid: int
    namespace: str
    owner_token: str
    log_path: str


@dataclass(frozen=True)
class CommandOutcome:
    name: str
    argv: tuple[str, ...]
    return_code: int
    stdout_name: str
    stderr_name: str
    elapsed_seconds: float
    cleanup_succeeded: bool
    error: str | None


@dataclass(frozen=True)
class OwnedScratchDirectory:
    path: Path
    parent_descriptor: int
    descriptor: int
    device: int
    inode: int


@dataclass(frozen=True)
class OwnedCgroup:
    path: Path
    parent_descriptor: int
    descriptor: int
    device: int
    inode: int
    owner_uid: int
    invocation_id: str
    systemd_unit_name: str


@dataclass
class SignalLatch:
    signal_number: int | None = None
    cleanup_started: bool = False

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number

    def checkpoint(self) -> None:
        if self.signal_number is not None and not self.cleanup_started:
            raise ManagedSignalError(self.signal_number)

    def begin_cleanup(self) -> None:
        self.cleanup_started = True


class LeaseMetadataValidator(Protocol):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


def _json_object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict):
        raise Glm47HarnessError(f"{description} must be a JSON object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise Glm47HarnessError(f"{description} must have string keys")
    return cast(JsonObject, cast(object, value))


def _read_json_regular(path: Path, description: str) -> JsonObject:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > MAXIMUM_JSON_BYTES:
            raise Glm47HarnessError(f"{description} is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            raw = source.read(MAXIMUM_JSON_BYTES + 1)
    except OSError as error:
        raise Glm47HarnessError(f"cannot read {description}: {error}") from error
    try:
        return _json_object(cast(object, json.loads(raw)), description)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Glm47HarnessError(f"invalid {description}: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise Glm47HarnessError(f"hashed path is not regular: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Glm47HarnessError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _hash_manifest(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256(b"exo-glm47-immutable-bundle-v1\0")
    normalized = tuple(sorted(set(paths), key=lambda path: path.as_posix()))
    if not normalized:
        raise Glm47HarnessError("immutable bundle must not be empty")
    for path in normalized:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise Glm47HarnessError("bundle path escaped its root") from error
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ("/usr/bin/git", "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Glm47HarnessError("cannot inspect source Git identity") from error
    if result.returncode != 0:
        raise Glm47HarnessError(
            "cannot inspect source Git identity: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def read_source_identity(repository: Path) -> SourceIdentity:
    if not repository.is_absolute() or repository.is_symlink():
        raise Glm47HarnessError("source repository must be a direct absolute path")
    commit = _run_git(repository, ("rev-parse", "--verify", "HEAD")).decode().strip()
    if _COMMIT.fullmatch(commit) is None:
        raise Glm47HarnessError("source HEAD is not an exact commit")
    tracked = _run_git(
        repository,
        ("diff", "--name-only", "--diff-filter=ACDMRTUXB", "-z", "HEAD"),
    )
    untracked = _run_git(
        repository, ("ls-files", "--others", "--exclude-standard", "-z")
    )
    names = {
        item.decode("utf-8")
        for item in (*tracked.split(b"\0"), *untracked.split(b"\0"))
        if item
    }
    hashes: dict[str, str] = {}
    for name in sorted(names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise Glm47HarnessError("Git returned an unsafe dirty path")
        path = repository / Path(relative)
        try:
            observed = path.lstat()
        except FileNotFoundError:
            hashes[name] = hashlib.sha256(
                b"exo-deleted-source-v1\0" + name.encode("utf-8")
            ).hexdigest()
            continue
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise Glm47HarnessError(f"dirty source path is not regular: {name}")
        hashes[name] = _file_sha256(path)
    return SourceIdentity(commit, hashes)


def _copy_regular_file(source: Path, destination: Path) -> None:
    try:
        observed = source.lstat()
    except OSError as error:
        raise Glm47HarnessError(
            f"cannot inspect source file {source}: {error}"
        ) from error
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise Glm47HarnessError(f"source deployment input is not regular: {source}")
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with (
            source.open("rb") as input_file,
            os.fdopen(descriptor, "wb", closefd=True) as output,
        ):
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise Glm47HarnessError(
            f"cannot copy immutable source {source}: {error}"
        ) from error


def _write_new_json(path: Path, value: Mapping[str, object], mode: int = 0o600) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise Glm47HarnessError(f"cannot create JSON file {path}: {error}") from error


def _source_files(root: Path) -> tuple[Path, ...]:
    source_root = root / "src"
    files: list[Path] = []
    for path in source_root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        observed = path.lstat()
        if stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            files.append(path)
        elif not stat.S_ISDIR(observed.st_mode):
            raise Glm47HarnessError(f"source tree contains a special entry: {path}")
    if not files:
        raise Glm47HarnessError("source tree is empty")
    return tuple(sorted(files))


def _make_tree_immutable(root: Path, *, required_uid: int = 0) -> None:
    paths = tuple(root.rglob("*"))
    for path in paths:
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise Glm47HarnessError(f"deployment contains a symbolic link: {path}")
        if stat.S_ISREG(observed.st_mode):
            if observed.st_nlink != 1:
                raise Glm47HarnessError(f"deployment file is multiply linked: {path}")
            os.chown(path, required_uid, 0)
            os.chmod(path, 0o444)
        elif stat.S_ISDIR(observed.st_mode):
            os.chown(path, required_uid, 0)
            os.chmod(path, 0o555)
        else:
            raise Glm47HarnessError(f"deployment contains a special file: {path}")
    os.chown(root, required_uid, 0)
    os.chmod(root, 0o555)


def _verify_immutable_tree(root: Path, expected_files: frozenset[Path]) -> None:
    if not root.is_absolute() or root.is_symlink():
        raise Glm47HarnessError("immutable tree root is not direct and absolute")
    observed_files: set[Path] = set()
    for path in (root, *root.rglob("*")):
        observed = path.lstat()
        if (
            stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != 0
            or observed.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise Glm47HarnessError(f"deployment path is mutable: {path}")
        if stat.S_ISREG(observed.st_mode):
            if observed.st_nlink != 1:
                raise Glm47HarnessError(f"deployment file is multiply linked: {path}")
            observed_files.add(path)
        elif not stat.S_ISDIR(observed.st_mode):
            raise Glm47HarnessError(f"deployment path is special: {path}")
    if frozenset(observed_files) != expected_files:
        raise Glm47HarnessError("immutable tree does not have the exact file set")


def create_immutable_deployment(
    config: ValidationConfig,
    config_contents: bytes,
    source_identity: SourceIdentity,
) -> DeploymentIdentity:
    """Create exact validator and orchestration trees without live scratch."""

    repository = Path(config.source.repository)
    root = Path(config.source.deployment_root)
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise Glm47HarnessError(
            "deployment root must be a new path in an existing directory"
        )
    try:
        root.mkdir(mode=0o755)
    except OSError as error:
        raise Glm47HarnessError(f"cannot create deployment root: {error}") from error

    validator_root = root / "validator"
    orchestrator_root = root / "orchestrator"
    validator_root.mkdir(mode=0o755)
    orchestrator_root.mkdir(mode=0o755)

    validator_destinations: list[Path] = []
    for relative in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS:
        source = repository / relative
        destination = validator_root / relative
        _copy_regular_file(source, destination)
        validator_destinations.append(destination)

    orchestrator_destinations: list[Path] = []
    for source in _source_files(repository):
        relative = source.relative_to(repository)
        destination = orchestrator_root / relative
        _copy_regular_file(source, destination)
        orchestrator_destinations.append(destination)
    for relative in _ORCHESTRATOR_SCRIPT_PATHS:
        source = repository / relative
        destination = orchestrator_root / relative
        _copy_regular_file(source, destination)
        orchestrator_destinations.append(destination)
    immutable_config = root / IMMUTABLE_CONFIG_RELATIVE_PATH
    try:
        descriptor = os.open(
            immutable_config,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(config_contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Glm47HarnessError("short immutable config write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
    except OSError as error:
        raise Glm47HarnessError(f"cannot copy immutable config: {error}") from error
    orchestrator_destinations.append(immutable_config)

    validator_identity = calculate_validator_bundle(
        validator_bundle_paths(validator_root)
    )
    orchestrator_sha256 = _hash_manifest(orchestrator_root, orchestrator_destinations)
    validator_files: tuple[dict[str, JsonValue], ...] = tuple(
        cast(
            dict[str, JsonValue],
            {
                "path": str(source.path),
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            },
        )
        for source in validator_identity.sources
    )
    deployment = DeploymentIdentity(
        root=str(root),
        orchestrator_sha256=orchestrator_sha256,
        validator_sha256=validator_identity.sha256,
        validator_files=validator_files,
        source=source_identity,
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "root": deployment.root,
        "orchestrator_sha256": deployment.orchestrator_sha256,
        "validator_sha256": deployment.validator_sha256,
        "validator_files": list(deployment.validator_files),
        "source": {
            "commit": source_identity.commit,
            "dirty_file_hashes": source_identity.dirty_file_hashes,
        },
        "python_bytecode_policy": "PYTHONDONTWRITEBYTECODE=1",
    }
    _write_new_json(root / DEPLOYMENT_RECEIPT_FILENAME, receipt)
    _make_tree_immutable(root)
    _verify_immutable_tree(validator_root, frozenset(validator_destinations))
    _verify_immutable_tree(orchestrator_root, frozenset(orchestrator_destinations))
    root_expected = frozenset(
        {
            *validator_destinations,
            *orchestrator_destinations,
            root / DEPLOYMENT_RECEIPT_FILENAME,
        }
    )
    _verify_immutable_tree(root, root_expected)
    return deployment


def load_deployment_identity(root: Path) -> DeploymentIdentity:
    receipt = _read_json_regular(
        root / DEPLOYMENT_RECEIPT_FILENAME, "deployment receipt"
    )
    try:
        source_raw = _json_object(receipt.get("source"), "deployment source")
        commit = cast(str, source_raw["commit"])
        dirty_raw = _json_object(
            source_raw.get("dirty_file_hashes"), "dirty source hashes"
        )
        dirty = {name: cast(str, digest) for name, digest in dirty_raw.items()}
        files_raw = receipt.get("validator_files")
        if not isinstance(files_raw, list):
            raise Glm47HarnessError("validator_files must be an array")
        validator_files = tuple(
            _json_object(item, "validator file") for item in files_raw
        )
        deployment = DeploymentIdentity(
            root=cast(str, receipt["root"]),
            orchestrator_sha256=cast(str, receipt["orchestrator_sha256"]),
            validator_sha256=cast(str, receipt["validator_sha256"]),
            validator_files=validator_files,
            source=SourceIdentity(commit, dirty),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Glm47HarnessError("deployment receipt has an invalid shape") from error
    if (
        deployment.root != str(root)
        or _COMMIT.fullmatch(deployment.source.commit) is None
        or _SHA256.fullmatch(deployment.orchestrator_sha256) is None
        or _SHA256.fullmatch(deployment.validator_sha256) is None
        or any(_SHA256.fullmatch(value) is None for value in dirty.values())
    ):
        raise Glm47HarnessError("deployment receipt identity is invalid")
    validator_root = root / "validator"
    expected_validator_files = frozenset(
        validator_root / relative
        for relative in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    )
    _verify_immutable_tree(validator_root, expected_validator_files)
    orchestrator_root = root / "orchestrator"
    observed_orchestrator_files = frozenset(
        path for path in orchestrator_root.rglob("*") if path.is_file()
    )
    _verify_immutable_tree(orchestrator_root, observed_orchestrator_files)
    _verify_immutable_tree(
        root,
        frozenset(
            {
                *expected_validator_files,
                *observed_orchestrator_files,
                root / DEPLOYMENT_RECEIPT_FILENAME,
            }
        ),
    )
    if _hash_manifest(orchestrator_root, observed_orchestrator_files) != (
        deployment.orchestrator_sha256
    ):
        raise Glm47HarnessError("orchestration deployment hash changed")
    validator_identity = calculate_validator_bundle(
        validator_bundle_paths(validator_root)
    )
    if validator_identity.sha256 != deployment.validator_sha256:
        raise Glm47HarnessError("validator deployment hash changed")
    return deployment


def _verify_artifact(binding: ArtifactBinding, description: str) -> None:
    path = Path(binding.path)
    if _file_sha256(path) != binding.sha256:
        raise Glm47HarnessError(f"{description} SHA-256 changed")


def verify_runtime_python(binding: ExecutableBinding) -> None:
    current = Path(binding.path)
    for expected_target in binding.symlink_chain:
        try:
            observed = current.lstat()
            target = os.readlink(current)
        except OSError as error:
            raise Glm47HarnessError("runtime Python symlink chain changed") from error
        if not stat.S_ISLNK(observed.st_mode) or target != expected_target:
            raise Glm47HarnessError("runtime Python symlink chain changed")
        target_path = Path(target)
        current = (
            target_path
            if target_path.is_absolute()
            else Path(os.path.normpath(current.parent / target_path))
        )
        if not current.is_absolute():
            raise Glm47HarnessError("runtime Python link target is not absolute")
    try:
        final = current.lstat()
    except OSError as error:
        raise Glm47HarnessError("runtime Python final target is unavailable") from error
    if (
        not stat.S_ISREG(final.st_mode)
        or stat.S_ISLNK(final.st_mode)
        or _file_sha256(current) != binding.sha256
        or not os.access(binding.path, os.X_OK)
    ):
        raise Glm47HarnessError("runtime Python final target identity changed")


def load_config(path: Path) -> ValidationConfig:
    if not path.is_absolute():
        raise Glm47HarnessError("config path must be absolute")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > MAXIMUM_JSON_BYTES:
            raise Glm47HarnessError("GLM-4.7 harness config is not bounded")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            contents = source.read(MAXIMUM_JSON_BYTES + 1)
        return ValidationConfig.model_validate_json(contents)
    except (OSError, ValidationError) as error:
        raise Glm47HarnessError(f"invalid GLM-4.7 harness config: {error}") from error


def _config_sha256(path: Path) -> str:
    return _file_sha256(path)


def _model_metadata(config: ValidationConfig) -> JsonObject:
    return {
        "model_id": GLM_4_7_FLASH_BF16_MODEL_ID,
        "revision": GLM_4_7_FLASH_BF16_MODEL_REVISION,
        "paths": {config.host.hostname: config.model_path},
    }


def build_static_metadata(
    config: ValidationConfig,
    command: Sequence[str],
    config_sha256: str,
    deployment: DeploymentIdentity,
) -> JsonObject:
    host = config.host.hostname
    value: dict[str, object] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "command": list(command),
        "git": {
            "commit": deployment.source.commit,
            "dirty": bool(deployment.source.dirty_file_hashes),
            "dirty_file_hashes": deployment.source.dirty_file_hashes,
        },
        "hosts": [host],
        "models": [_model_metadata(config)],
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
        "hca_bindings": {
            host: [
                {
                    "device": binding.device,
                    "port": binding.port,
                    "gid": binding.gid,
                }
                for binding in config.host.hca_bindings
            ]
        },
        "source_deployments": {
            host: {
                "path": deployment.root,
                "commit": deployment.source.commit,
                "dirty_file_hashes": deployment.source.dirty_file_hashes,
            }
        },
        "owner_pids": {host: []},
        "containment_contract": {
            "schema": "systemd_delegated_cgroup_v1",
            "systemd_unit_name": _systemd_unit_name(config),
            "systemd_slice": SYSTEMD_SLICE,
            "delegate_subgroup": SYSTEMD_DELEGATE_SUBGROUP,
            "validator_cgroup_layout": "delegated-sibling-v1",
            "attach_method": "preexec-cgroup.procs-v1",
            "cleanup_method": "cgroup.kill-v1",
        },
        "validation_contract": {
            "kind": "glm47_sglang_kt_admission_validation",
            "phase": config.phase,
            "profiler": "none",
            "hca_requirement": config.hca_requirement,
            "config_sha256": config_sha256,
            "orchestrator_sha256": deployment.orchestrator_sha256,
            "validator_sha256": deployment.validator_sha256,
            "runtime_python": config.runtime_python.model_dump(mode="json"),
            "build_receipt": config.build_receipt.model_dump(mode="json"),
            "model_contract": config.model_contract.model_dump(mode="json"),
            "resident_gpu_experts": config.resident_gpu_experts,
            "python_bytecode_policy": "PYTHONDONTWRITEBYTECODE=1",
            "scratch_directory": config.scratch_directory,
        },
    }
    return cast(JsonObject, cast(object, value))


def minimum_cleanup_grace_seconds(config: ValidationConfig) -> float:
    return max(
        MINIMUM_CLEANUP_GRACE_SECONDS,
        (2.0 * config.timeouts.cleanup_seconds) + 60.0,
    )


def _validate_metadata(
    benchmark_lease_script: Path,
    metadata: JsonObject,
    now: datetime,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_glm47_benchmark_lease", benchmark_lease_script
    )
    if specification is None or specification.loader is None:
        raise Glm47HarnessError("cannot import immutable benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    validator = cast(LeaseMetadataValidator, cast(object, module))
    if (
        validator.validate_run_metadata(cast(Mapping[str, object], metadata), now=now)
        != metadata
    ):
        raise Glm47HarnessError("benchmark lease changed generated metadata")


def _immutable_child_argv(
    config: ValidationConfig,
    *,
    lease_path: Path,
    lock_path: Path,
) -> tuple[str, ...]:
    root = Path(config.source.deployment_root)
    return (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
        config.runtime_python.path,
        str(root / "orchestrator/scripts/run_sglang_kt_glm47_validation.py"),
        "--config",
        str(root / IMMUTABLE_CONFIG_RELATIVE_PATH),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    )


def _systemd_unit_name(config: ValidationConfig) -> str:
    binding = hashlib.sha256(
        f"glm47-validation-v1\0{config.run_id}\0{config.namespace}".encode()
    ).hexdigest()[:32]
    return f"exo-glm47-{binding}.service"


def _systemd_benchmark_argv(
    config: ValidationConfig,
    benchmark_argv: Sequence[str],
    *,
    expected_duration_seconds: float,
    cleanup_grace_seconds: float,
) -> tuple[str, ...]:
    unit_name = _systemd_unit_name(config)
    if _SYSTEMD_UNIT.fullmatch(unit_name) is None:
        raise Glm47HarnessError("generated systemd unit name is invalid")
    runtime_max_seconds = (
        expected_duration_seconds + (2.0 * cleanup_grace_seconds) + 300.0
    )
    return (
        str(SYSTEMD_RUN_EXECUTABLE),
        "--system",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
        "--service-type=exec",
        f"--unit={unit_name}",
        f"--slice={SYSTEMD_SLICE}",
        f"--working-directory={config.source.deployment_root}",
        "--expand-environment=no",
        "--property=Delegate=yes",
        f"--property=DelegateSubgroup={SYSTEMD_DELEGATE_SUBGROUP}",
        "--property=KillMode=control-group",
        "--property=SendSIGKILL=yes",
        f"--property=TimeoutStopSec={cleanup_grace_seconds}s",
        f"--property=RuntimeMaxSec={runtime_max_seconds}s",
        "--",
        *benchmark_argv,
    )


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
    for path in (
        config_path,
        metadata_output,
        lease_path,
        lock_path,
        result_root,
    ):
        if not path.is_absolute():
            raise Glm47HarnessError("all preparation paths must be absolute")
    if not owner.strip() or not purpose.strip() or "\0" in owner + purpose:
        raise Glm47HarnessError("owner and purpose must be nonempty and NUL-free")
    for value, description in (
        (expected_duration_seconds, "expected duration"),
        (cleanup_grace_seconds, "cleanup grace"),
        (heartbeat_seconds, "heartbeat"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise Glm47HarnessError(f"{description} must be finite and positive")

    config_contents = config_path.read_bytes()
    config = load_config(config_path)
    repository = Path(config.source.repository)
    if Path(__file__).resolve() != (
        repository / "scripts/run_sglang_kt_glm47_validation.py"
    ):
        raise Glm47HarnessError("prepare-lease must run from the configured repository")
    if Path(config.result_directory) != result_root / config.run_id:
        raise Glm47HarnessError("result directory must equal result_root/run_id")
    if (
        Path(config.result_directory).exists()
        or Path(config.scratch_directory).exists()
    ):
        raise Glm47HarnessError("result and scratch paths must both be new")
    minimum_grace = minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise Glm47HarnessError(f"cleanup grace must be at least {minimum_grace}")
    verify_runtime_python(config.runtime_python)
    _verify_artifact(config.build_receipt, "build receipt")
    if not os.access(config.numactl_executable, os.X_OK):
        raise Glm47HarnessError("numactl is not executable")
    if not os.access(SYSTEMD_RUN_EXECUTABLE, os.X_OK):
        raise Glm47HarnessError("systemd-run is not executable")

    source_identity = read_source_identity(repository)
    deployment = create_immutable_deployment(config, config_contents, source_identity)
    _verify_artifact(config.model_contract, "model contract")
    if read_source_identity(repository) != source_identity:
        raise Glm47HarnessError("source identity changed during deployment")

    child_argv = _immutable_child_argv(
        config, lease_path=lease_path, lock_path=lock_path
    )
    generated_at = now().astimezone(timezone.utc)
    immutable_config = Path(deployment.root) / IMMUTABLE_CONFIG_RELATIVE_PATH
    metadata = build_static_metadata(
        config, child_argv, _config_sha256(immutable_config), deployment
    )
    metadata["generated_at"] = generated_at.isoformat(timespec="seconds")
    benchmark_lease_script = (
        Path(deployment.root) / "orchestrator/scripts/benchmark_lease.py"
    )
    _validate_metadata(benchmark_lease_script, metadata, generated_at)
    if load_deployment_identity(Path(deployment.root)) != deployment:
        raise Glm47HarnessError(
            "immutable deployment changed during metadata validation"
        )
    lease_argv = (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
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
    benchmark_argv = _systemd_benchmark_argv(
        config,
        lease_argv,
        expected_duration_seconds=expected_duration_seconds,
        cleanup_grace_seconds=cleanup_grace_seconds,
    )
    _write_new_json(metadata_output, cast(Mapping[str, object], metadata), 0o444)
    return LeasePreparation(
        metadata=metadata,
        metadata_output=str(metadata_output),
        child_argv=child_argv,
        benchmark_lease_argv=benchmark_argv,
        systemd_unit_name=_systemd_unit_name(config),
        deployment=deployment,
        minimum_cleanup_grace_seconds=minimum_grace,
    )


class ResultDirectory:
    """Descriptor-anchored access to the lease wrapper's result directory."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = os.dup(descriptor)
        self.validate_identity()

    @classmethod
    def inherited(cls, path: Path) -> "ResultDirectory":
        raw = os.environ.get(RESULT_DIRECTORY_FD_ENVIRONMENT)
        if raw is None or not raw.isascii() or not raw.isdigit() or int(raw) < 3:
            raise Glm47HarnessError(
                f"lease wrapper did not pass {RESULT_DIRECTORY_FD_ENVIRONMENT}"
            )
        return cls(path, int(raw))

    @staticmethod
    def validate_name(name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise Glm47HarnessError("result name must be one safe component")

    def validate_identity(self) -> None:
        try:
            retained = os.fstat(self.descriptor)
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise Glm47HarnessError(
                f"cannot validate result directory identity: {error}"
            ) from error
        if (
            not stat.S_ISDIR(retained.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (retained.st_dev, retained.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise Glm47HarnessError("result directory identity changed")

    def close(self) -> None:
        os.close(self.descriptor)

    def path_for(self, name: str) -> Path:
        self.validate_name(name)
        self.validate_identity()
        return self.path / name

    def create_log(self, name: str) -> IO[bytes]:
        self.validate_name(name)
        self.validate_identity()
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.descriptor,
            )
        except OSError as error:
            raise Glm47HarnessError(
                f"cannot create result log {name}: {error}"
            ) from error
        return os.fdopen(descriptor, "wb", closefd=True)

    def read_bytes(self, name: str, maximum: int = MAXIMUM_JSON_BYTES) -> bytes:
        self.validate_name(name)
        self.validate_identity()
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=self.descriptor,
            )
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
                raise Glm47HarnessError(f"result {name} is not a bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=True) as source:
                return source.read(maximum + 1)
        except OSError as error:
            raise Glm47HarnessError(f"cannot read result {name}: {error}") from error

    def read_json(self, name: str) -> JsonObject:
        try:
            value = cast(object, json.loads(self.read_bytes(name)))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise Glm47HarnessError(f"invalid JSON result {name}: {error}") from error
        return _json_object(value, name)

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.read_bytes(name)).hexdigest()

    def write_json(
        self, name: str, value: Mapping[str, object], *, replace: bool
    ) -> None:
        self.validate_name(name)
        self.validate_identity()
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.descriptor,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            if replace:
                try:
                    current = os.stat(
                        name, dir_fd=self.descriptor, follow_symlinks=False
                    )
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(current.st_mode):
                        raise Glm47HarnessError(
                            f"refusing to replace non-regular result {name}"
                        )
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=self.descriptor,
                    dst_dir_fd=self.descriptor,
                )
            else:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=self.descriptor,
                    dst_dir_fd=self.descriptor,
                    follow_symlinks=False,
                )
            os.fsync(self.descriptor)
        except OSError as error:
            raise Glm47HarnessError(f"cannot publish result {name}: {error}") from error
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self.descriptor)
        self.validate_identity()


def validate_active_lease(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    *,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int | None = None,
) -> JsonObject:
    try:
        lock_descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(lock_descriptor, "rb", closefd=True) as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                raise Glm47HarnessError("benchmark coordination lock is not held")
    except OSError as error:
        raise Glm47HarnessError(f"cannot inspect benchmark lock: {error}") from error
    expected_pid = os.getpid() if process_id is None else process_id
    deadline = time.monotonic() + LEASE_CHILD_BIND_SECONDS
    while True:
        record = _read_json_regular(lease_path, "active benchmark lease")
        child_pid = record.get("child_pid")
        if child_pid == expected_pid:
            break
        if child_pid is not None or time.monotonic() >= deadline:
            raise Glm47HarnessError("active lease does not belong to this child")
        time.sleep(0.05)
    command_raw = record.get("command")
    if not isinstance(command_raw, list) or not all(
        isinstance(argument, str) for argument in command_raw
    ):
        raise Glm47HarnessError("active lease command is invalid")
    command = tuple(cast(list[str], command_raw))
    expected_command = _immutable_child_argv(
        config, lease_path=lease_path, lock_path=lock_path
    )
    if command != expected_command:
        raise Glm47HarnessError(
            "active lease command differs from immutable invocation"
        )
    expected: dict[str, object] = {
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "wrapper_pid": os.getppid(),
        "child_cleanup_confirmation_required": True,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise Glm47HarnessError(f"active lease {name} differs from config")
    cleanup_grace = record.get("cleanup_grace_seconds")
    if (
        not isinstance(cleanup_grace, int | float)
        or isinstance(cleanup_grace, bool)
        or cleanup_grace < minimum_cleanup_grace_seconds(config)
    ):
        raise Glm47HarnessError("active lease cleanup grace is too short")
    metadata = _json_object(record.get("metadata"), "active lease metadata")
    expected_metadata = build_static_metadata(
        config, command, _config_sha256(config_path), deployment
    )
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise Glm47HarnessError(f"active lease metadata.{name} changed")
    try:
        heartbeat = datetime.fromisoformat(
            cast(str, record["heartbeat"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError) as error:
        raise Glm47HarnessError("active lease heartbeat is invalid") from error
    current = datetime.now(timezone.utc)
    if heartbeat < current - LEASE_HEARTBEAT_MAX_AGE or heartbeat > current + timedelta(
        minutes=1
    ):
        raise Glm47HarnessError("active lease heartbeat is stale or future-dated")
    return record


def require_no_profiler_state(
    environment: Mapping[str, str],
    command: Sequence[str],
    modules_text: str,
) -> None:
    del modules_text
    for name, value in environment.items():
        upper_name = name.upper()
        upper_value = value.upper()
        if (
            any(marker in upper_name for marker in _PROFILER_ENVIRONMENT_MARKERS)
            or "SEP5" in upper_value
            or "VTUNE" in upper_value
            or "AMPLXE" in upper_value
        ):
            raise Glm47HarnessError(f"profiler environment is present: {name}")
    lowered = tuple(argument.lower() for argument in command)
    if any(
        marker in argument
        for marker in _PROFILER_COMMAND_MARKERS
        for argument in lowered
    ):
        raise Glm47HarnessError("profiler command is forbidden")
    for argument in lowered:
        if Path(argument).name in {"pax", "sep5"}:
            raise Glm47HarnessError("SEP/PAX command is forbidden")


def loaded_unsafe_profiler_modules(modules_text: str) -> tuple[str, ...]:
    modules = {
        line.split(maxsplit=1)[0].lower()
        for line in modules_text.splitlines()
        if line.strip()
    }
    return tuple(sorted(modules & {"sep5", "pax"}))


def find_active_profiler_use(proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    active: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        process_id = int(entry.name)
        try:
            command = tuple(
                value.decode("utf-8", errors="replace").lower()
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            )
        except OSError:
            continue
        command_uses_profiler = any(
            marker in argument
            for marker in _PROFILER_COMMAND_MARKERS
            for argument in command
        ) or any(Path(argument).name in {"pax", "sep5"} for argument in command)
        driver_open = False
        try:
            descriptors = entry / "fd"
            for descriptor in descriptors.iterdir():
                try:
                    target = os.readlink(descriptor).lower()
                except OSError:
                    continue
                if any(
                    marker in target
                    for marker in ("/dev/sep", "/dev/pax", "/dev/socperf")
                ):
                    driver_open = True
                    break
        except OSError:
            pass
        if command_uses_profiler or driver_open:
            active.append(process_id)
    return tuple(sorted(active))


def build_child_environment(
    config: ValidationConfig,
    owner_token: str,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if parent_environment is None else parent_environment
    require_no_profiler_state(source, (), "")
    path = source.get("PATH", "/usr/bin:/bin")
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": config.host.gpu.uuid,
        "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
        "EXO_NAMESPACE": config.namespace,
        "HOME": source.get("HOME", "/root"),
        "PATH": path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": config.scratch_directory,
    }


def _parse_linux_indices(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for component in text.strip().split(","):
        if not component:
            continue
        start_text, separator, end_text = component.partition("-")
        try:
            start = int(start_text)
            end = int(end_text) if separator else start
        except ValueError as error:
            raise Glm47HarnessError("Linux resource list is invalid") from error
        if start < 0 or end < start:
            raise Glm47HarnessError("Linux resource list is invalid")
        values.extend(range(start, end + 1))
    result = tuple(values)
    if not result or result != tuple(sorted(set(result))):
        raise Glm47HarnessError("Linux resource list is not canonical")
    return result


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise Glm47HarnessError(f"cannot read {description}: {path}") from error


def _process_command(arguments: Sequence[str]) -> bool:
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    if executable in _CONFLICTING_EXECUTABLES:
        return True
    if executable.startswith("python"):
        for index, argument in enumerate(arguments[:-1]):
            if argument == "-m" and arguments[index + 1].startswith(
                _CONFLICTING_PYTHON_MODULE_PREFIXES
            ):
                return True
        if any(
            Path(argument).name in _CONFLICTING_EXECUTABLES
            for argument in arguments[1:]
        ):
            return True
    return False


def find_conflicting_processes(
    proc_root: Path = Path("/proc"),
    ignored_process_ids: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    conflicts: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        process_id = int(entry.name)
        if process_id in ignored_process_ids:
            continue
        try:
            command = tuple(
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            )
        except OSError:
            continue
        if _process_command(command):
            conflicts.append(process_id)
    return tuple(sorted(conflicts))


def _run_probe(command: Sequence[str], description: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Glm47HarnessError(f"cannot run {description}") from error
    if result.returncode != 0:
        raise Glm47HarnessError(f"{description} failed: {result.stderr.strip()}")
    return result.stdout


def _probe_ports_unused(ports: Sequence[int]) -> None:
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            for family, kind, address in (
                (socket.AF_INET, socket.SOCK_STREAM, ("0.0.0.0", port)),
                (socket.AF_INET, socket.SOCK_DGRAM, ("0.0.0.0", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, ("::", port)),
                (socket.AF_INET6, socket.SOCK_DGRAM, ("::", port)),
            ):
                probe = socket.socket(family, kind)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind(address)
                sockets.append(probe)
    except OSError as error:
        raise Glm47HarnessError(f"reserved port is already in use: {error}") from error
    finally:
        for probe in sockets:
            probe.close()


def collect_local_hca_evidence(
    bindings: Sequence[HcaBinding],
    requirement: Literal["active", "metadata_only"],
    sysfs_root: Path = Path("/sys/class/infiniband"),
) -> list[JsonValue]:
    """Verify physical links without requiring an unused local subnet manager."""

    evidence: list[JsonValue] = []
    for binding in bindings:
        port_root = sysfs_root / binding.device / "ports" / str(binding.port)
        state = _read_text(port_root / "state", "HCA port state")
        physical_state = _read_text(port_root / "phys_state", "HCA physical port state")
        rate = _read_text(port_root / "rate", "HCA port rate")
        gid = ipaddress.IPv6Address(
            _read_text(port_root / "gids" / str(binding.gid_index), "HCA GID")
        ).exploded
        subnet_state = state.partition(":")[2].strip()
        if (
            physical_state.partition(":")[2].strip() != "LinkUp"
            or subnet_state not in {"INIT", "ACTIVE"}
            or (requirement == "active" and subnet_state != "ACTIVE")
            or gid != binding.gid
        ):
            raise Glm47HarnessError(
                "configured HCA port is not the expected physical LinkUp/GID"
            )
        evidence.append(
            {
                "device": binding.device,
                "port": binding.port,
                "gid": gid,
                "state": state,
                "physical_state": physical_state,
                "rate": rate,
                "subnet_manager_active": subnet_state == "ACTIVE",
                "requirement": requirement,
                "traffic_expected": False,
            }
        )
    return evidence


def collect_live_preflight(
    config: ValidationConfig,
    *,
    gpu_probe: Callable[[Sequence[str], str], str] = _run_probe,
) -> JsonObject:
    verify_runtime_python(config.runtime_python)
    _verify_artifact(config.build_receipt, "build receipt")
    _verify_artifact(config.model_contract, "model contract")
    modules = _read_text(Path("/proc/modules"), "loaded kernel modules")
    require_no_profiler_state(os.environ, sys.argv, modules)
    active_profiler_processes = find_active_profiler_use()
    if active_profiler_processes:
        raise Glm47HarnessError(
            f"active profiler use is forbidden: {active_profiler_processes}"
        )
    if os.uname().nodename != config.host.hostname:
        raise Glm47HarnessError("hostname differs from configured host")
    conflicts = find_conflicting_processes(
        ignored_process_ids=frozenset({os.getpid(), os.getppid()})
    )
    if conflicts:
        raise Glm47HarnessError(
            f"unowned conflicting processes are active: {conflicts}"
        )
    online_cpus = set(
        _parse_linux_indices(
            _read_text(Path("/sys/devices/system/cpu/online"), "online CPUs")
        )
    )
    if not set(config.host.cpu_cores) <= online_cpus:
        raise Glm47HarnessError("selected CPU set is not online")
    for node in config.host.memory_nodes:
        if not (Path("/sys/devices/system/node") / f"node{node}").is_dir():
            raise Glm47HarnessError(f"selected NUMA node {node} is absent")
    cpuinfo = _read_text(Path("/proc/cpuinfo"), "CPU features")
    feature_lines = [
        line.partition(":")[2].strip().split()
        for line in cpuinfo.splitlines()
        if line.startswith("flags") and ":" in line
    ]
    if not feature_lines or any(
        not set(features).issuperset(_REQUIRED_AMX_FEATURES)
        for features in feature_lines
    ):
        raise Glm47HarnessError("required AMX-BF16 CPU features are unavailable")
    gpu_rows = gpu_probe(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ),
        "GPU identity probe",
    )
    expected_gpu = (config.host.gpu.uuid, config.host.gpu.pci_address)
    observed_gpus: list[tuple[str, str]] = []
    for line in gpu_rows.splitlines():
        values = tuple(value.strip() for value in line.split(","))
        if len(values) == 2:
            try:
                observed = GpuBinding(uuid=values[0], pci_address=values[1])
            except ValidationError:
                continue
            observed_gpus.append((observed.uuid, observed.pci_address))
    if observed_gpus.count(expected_gpu) != 1:
        raise Glm47HarnessError(
            "configured GPU UUID/PCI identity was not found exactly once"
        )
    compute_rows = gpu_probe(
        (
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ),
        "GPU process probe",
    )
    selected_compute = [
        line for line in compute_rows.splitlines() if config.host.gpu.uuid in line
    ]
    if selected_compute:
        raise Glm47HarnessError("selected GPU has an unowned compute process")
    hca_evidence = collect_local_hca_evidence(
        config.host.hca_bindings, config.hca_requirement
    )
    _probe_ports_unused(config.reserved_ports)
    value: dict[str, object] = {
        "hostname": config.host.hostname,
        "cpu_cores": list(config.host.cpu_cores),
        "memory_nodes": list(config.host.memory_nodes),
        "amx_features": sorted(_REQUIRED_AMX_FEATURES),
        "gpu_uuid": config.host.gpu.uuid,
        "gpu_pci_address": config.host.gpu.pci_address,
        "hca_bindings": hca_evidence,
        "reserved_ports_unused": list(config.reserved_ports),
        "profiler": "none",
        "unsafe_profiler_modules_loaded": list(loaded_unsafe_profiler_modules(modules)),
        "active_profiler_processes": [],
        "unsafe_profiler_drivers_used": False,
    }
    return cast(JsonObject, cast(object, value))


def _process_stat_fields(process_id: int) -> tuple[str, ...]:
    try:
        value = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        close_parenthesis = value.rfind(")")
        fields = tuple(value[close_parenthesis + 2 :].split())
        if close_parenthesis < 0 or len(fields) < 20:
            raise ValueError("truncated process stat")
        return fields
    except (FileNotFoundError, ProcessLookupError):
        raise
    except (OSError, ValueError) as error:
        raise Glm47HarnessError(f"cannot identify process {process_id}") from error


def _process_identity(
    config: ValidationConfig,
    process_id: int,
    owner_token: str,
    log_path: str,
) -> OwnedProcess:
    fields = _process_stat_fields(process_id)
    try:
        process_group_id = int(fields[2])
        start_time_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise Glm47HarnessError("cannot parse owned process identity") from error
    if process_group_id <= 0 or start_time_ticks <= 0:
        raise Glm47HarnessError("owned process identity is not positive")
    return OwnedProcess(
        host_name=config.host.hostname,
        pid=process_id,
        process_group_id=process_group_id,
        start_time_ticks=start_time_ticks,
        transport_pid=process_id,
        namespace=config.namespace,
        owner_token=owner_token,
        log_path=log_path,
    )


def _owned_cgroup_evidence(owned: OwnedCgroup) -> JsonObject:
    return {
        "schema_version": 1,
        "path": str(owned.path),
        "device": owned.device,
        "inode": owned.inode,
        "owner_uid": owned.owner_uid,
        "invocation_id": owned.invocation_id,
        "systemd_unit_name": owned.systemd_unit_name,
        "attach_method": "preexec-cgroup.procs-v1",
        "cleanup_method": "cgroup.kill-v1",
    }


class OwnershipRegistry:
    def __init__(
        self,
        config: ValidationConfig,
        results: ResultDirectory,
        owner_token: str,
    ) -> None:
        self.config = config
        self.results = results
        self.owner_token = owner_token
        self.processes: list[OwnedProcess] = []
        self.identities: set[tuple[str, int, int]] = set()
        self.containment: JsonObject | None = None
        self.publish()

    def publish(self) -> None:
        self.results.write_json(
            RUNTIME_METADATA_FILENAME,
            {
                "schema_version": 1,
                "run_id": self.config.run_id,
                "namespace": self.config.namespace,
                "owner_token": self.owner_token,
                "containment": self.containment,
                "owned_processes": [asdict(process) for process in self.processes],
            },
            replace=True,
        )

    def register(self, process: OwnedProcess) -> None:
        if (
            process.host_name != self.config.host.hostname
            or process.namespace != self.config.namespace
            or process.owner_token != self.owner_token
        ):
            raise Glm47HarnessError("owned process has a foreign identity")
        identity = (process.host_name, process.pid, process.start_time_ticks)
        if identity in self.identities:
            return
        self.identities.add(identity)
        self.processes.append(process)
        self.publish()

    def bind_cgroup(self, owned: OwnedCgroup) -> None:
        if self.containment is not None:
            raise Glm47HarnessError("owned cgroup is already bound")
        self.containment = _owned_cgroup_evidence(owned)
        self.publish()


def _owned_token_processes(
    owner_token: str, proc_root: Path = Path("/proc")
) -> dict[int, tuple[int, int]] | None:
    expected_owner = f"EXO_BENCHMARK_OWNER_TOKEN={owner_token}".encode()
    processes: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        process_id = int(entry.name)
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return None
        if expected_owner not in environment:
            continue
        try:
            fields = _process_stat_fields(process_id)
            processes[process_id] = (int(fields[2]), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (Glm47HarnessError, ValueError, IndexError):
            return None
    return processes


def _live_group_ownership(
    process_group_id: int, owner_token: str
) -> Literal["absent", "owned", "foreign", "unknown"]:
    expected_owner = f"EXO_BENCHMARK_OWNER_TOKEN={owner_token}".encode()
    members = 0
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return "unknown"
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        process_id = int(entry.name)
        try:
            fields = _process_stat_fields(process_id)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (Glm47HarnessError, ValueError, IndexError):
            try:
                fields = _process_stat_fields(process_id)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (Glm47HarnessError, ValueError, IndexError):
                return "unknown"
        try:
            if int(fields[2]) != process_group_id or fields[0] in {"Z", "X", "x"}:
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            try:
                refreshed = _process_stat_fields(process_id)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (Glm47HarnessError, ValueError, IndexError):
                return "unknown"
            if int(refreshed[2]) != process_group_id or refreshed[0] in {"Z", "X", "x"}:
                continue
            return "unknown"
        if expected_owner not in environment:
            return "foreign"
        members += 1
    return "owned" if members else "absent"


def _signal_owned_group(
    process_group_id: int, owner_token: str, signal_number: signal.Signals
) -> bool:
    ownership = _live_group_ownership(process_group_id, owner_token)
    if ownership == "absent":
        return True
    if ownership != "owned":
        return False
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _terminate_owned_group(
    process_group_id: int,
    owner_token: str,
    timeout_seconds: float,
) -> bool:
    ownership = _live_group_ownership(process_group_id, owner_token)
    if ownership == "absent":
        return True
    if ownership != "owned" or not _signal_owned_group(
        process_group_id, owner_token, signal.SIGTERM
    ):
        return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ownership = _live_group_ownership(process_group_id, owner_token)
        if ownership == "absent":
            return True
        if ownership != "owned":
            return False
        time.sleep(0.05)
    if not _signal_owned_group(process_group_id, owner_token, signal.SIGKILL):
        return False
    deadline = time.monotonic() + min(5.0, timeout_seconds)
    while time.monotonic() < deadline:
        ownership = _live_group_ownership(process_group_id, owner_token)
        if ownership == "absent":
            return True
        if ownership != "owned":
            return False
        time.sleep(0.05)
    return _live_group_ownership(process_group_id, owner_token) == "absent"


def child_subreaper_enabled() -> bool:
    value = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(37, ctypes.byref(value), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        error_number = ctypes.get_errno()
        raise Glm47HarnessError(f"cannot read child subreaper: errno {error_number}")
    return value.value != 0


def set_child_subreaper(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(36, int(enabled), 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error_number = ctypes.get_errno()
        raise Glm47HarnessError(f"cannot set child subreaper: errno {error_number}")


def _enable_child_subreaper() -> None:
    set_child_subreaper(True)


def _systemd_cgroup_path(
    config: ValidationConfig,
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
) -> PurePosixPath:
    try:
        contents = proc_self_cgroup.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise Glm47HarnessError("cannot read the unified process cgroup") from error
    lines = contents.splitlines()
    if len(lines) != 1 or not lines[0].startswith("0::"):
        raise Glm47HarnessError("process is not in one unified cgroup-v2 hierarchy")
    raw_path = lines[0][3:]
    path = PurePosixPath(raw_path)
    expected = (
        PurePosixPath("/")
        / SYSTEMD_SLICE
        / _systemd_unit_name(config)
        / SYSTEMD_DELEGATE_SUBGROUP
    )
    if (
        not raw_path
        or "\\" in raw_path
        or "\0" in raw_path
        or not path.is_absolute()
        or path.as_posix() != raw_path
        or os.path.normpath(raw_path) != raw_path
        or path != expected
    ):
        raise Glm47HarnessError(
            "validation harness is not in its exact delegated systemd subgroup"
        )
    return path


def _open_directory_components(root: Path, relative: PurePosixPath) -> int:
    if not root.is_absolute() or relative.is_absolute():
        raise Glm47HarnessError("cgroup directory roots must be canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for component in relative.parts:
            if component in {"", ".", ".."} or "\0" in component:
                raise Glm47HarnessError("cgroup path contains an unsafe component")
            child_descriptor = os.open(
                component,
                flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_descriptor_file(
    directory_descriptor: int,
    name: str,
    *,
    expected_device: int,
    expected_uid: int,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptor,
    )
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_dev != expected_device
            or observed.st_uid != expected_uid
        ):
            raise Glm47HarnessError(f"cgroup control {name} changed identity")
        contents = os.read(descriptor, 4097)
        if len(contents) > 4096:
            raise Glm47HarnessError(f"cgroup control {name} is oversized")
        return contents
    finally:
        os.close(descriptor)


def _owned_cgroup_identity_matches(owned: OwnedCgroup) -> bool:
    try:
        retained = os.fstat(owned.descriptor)
        current = os.stat(
            owned.path.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        stat.S_ISDIR(retained.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and retained.st_uid == owned.owner_uid
        and current.st_uid == owned.owner_uid
        and stat.S_IMODE(retained.st_mode) == 0o700
        and stat.S_IMODE(current.st_mode) == 0o700
        and (retained.st_dev, retained.st_ino) == (owned.device, owned.inode)
        and (current.st_dev, current.st_ino) == (owned.device, owned.inode)
    )


def _open_owned_cgroup_control(
    owned: OwnedCgroup,
    name: Literal["cgroup.procs", "cgroup.events", "cgroup.kill"],
    flags: int,
) -> int:
    if not _owned_cgroup_identity_matches(owned):
        raise Glm47HarnessError("owned validator cgroup changed identity")
    descriptor = os.open(
        name,
        flags | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=owned.descriptor,
    )
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_dev != owned.device
            or observed.st_uid != owned.owner_uid
        ):
            raise Glm47HarnessError(f"cgroup control {name} changed identity")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_owned_cgroup_controls(owned: OwnedCgroup) -> None:
    for name, flags in (
        ("cgroup.procs", os.O_WRONLY),
        ("cgroup.events", os.O_RDONLY),
        ("cgroup.kill", os.O_WRONLY),
    ):
        descriptor = _open_owned_cgroup_control(
            owned,
            cast(Literal["cgroup.procs", "cgroup.events", "cgroup.kill"], name),
            flags,
        )
        os.close(descriptor)


def _create_owned_cgroup(
    config: ValidationConfig,
    owner_token: str,
    *,
    cgroup_root: Path = CGROUP_FILESYSTEM_ROOT,
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
    environment: Mapping[str, str] | None = None,
) -> OwnedCgroup:
    invocation_id = (os.environ if environment is None else environment).get(
        "INVOCATION_ID", ""
    )
    if _INVOCATION_ID.fullmatch(invocation_id) is None:
        raise Glm47HarnessError("delegated systemd invocation identity is missing")
    current_path = _systemd_cgroup_path(config, proc_self_cgroup)
    unit_path = current_path.parent
    relative_unit = PurePosixPath(*unit_path.parts[1:])
    parent_descriptor = _open_directory_components(cgroup_root, relative_unit)
    child_descriptor: int | None = None
    child_created = False
    child_name = "validators-" + hashlib.sha256(owner_token.encode()).hexdigest()[:32]
    path = cgroup_root.joinpath(*unit_path.parts[1:], child_name)
    try:
        parent = os.fstat(parent_descriptor)
        supervisor = os.stat(
            SYSTEMD_DELEGATE_SUBGROUP,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(parent.st_mode)
            or not stat.S_ISDIR(supervisor.st_mode)
            or parent.st_uid != os.geteuid()
            or supervisor.st_uid != os.geteuid()
            or parent.st_mode & 0o022
            or supervisor.st_mode & 0o022
        ):
            raise Glm47HarnessError("delegated systemd cgroup identity is unsafe")
        unit_processes = _read_descriptor_file(
            parent_descriptor,
            "cgroup.procs",
            expected_device=parent.st_dev,
            expected_uid=os.geteuid(),
        )
        if unit_processes.strip():
            raise Glm47HarnessError("delegated systemd unit root is not empty")
        try:
            os.stat(child_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise Glm47HarnessError("validator cgroup must be a new direct child")
        os.mkdir(child_name, mode=0o700, dir_fd=parent_descriptor)
        child_created = True
        child_descriptor = os.open(
            child_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        observed = os.fstat(child_descriptor)
        owned = OwnedCgroup(
            path=path,
            parent_descriptor=parent_descriptor,
            descriptor=child_descriptor,
            device=observed.st_dev,
            inode=observed.st_ino,
            owner_uid=os.geteuid(),
            invocation_id=invocation_id,
            systemd_unit_name=_systemd_unit_name(config),
        )
        if not _owned_cgroup_identity_matches(owned):
            raise Glm47HarnessError("new validator cgroup identity is unsafe")
        _validate_owned_cgroup_controls(owned)
        return owned
    except BaseException:
        if child_descriptor is not None:
            os.close(child_descriptor)
        if child_created:
            with contextlib.suppress(OSError):
                os.rmdir(child_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)
        raise


def _cgroup_is_populated(owned: OwnedCgroup) -> bool:
    descriptor = _open_owned_cgroup_control(owned, "cgroup.events", os.O_RDONLY)
    try:
        contents = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    if len(contents) > 4096:
        raise Glm47HarnessError("cgroup.events is oversized")
    values: dict[str, str] = {}
    try:
        for line in contents.decode("ascii").splitlines():
            key, value = line.split()
            if key in values:
                raise ValueError("duplicate key")
            values[key] = value
    except (UnicodeError, ValueError) as error:
        raise Glm47HarnessError("cgroup.events is malformed") from error
    populated = values.get("populated")
    if populated not in {"0", "1"}:
        raise Glm47HarnessError("cgroup.events has no canonical populated state")
    return populated == "1"


def quiesce_owned_cgroup(owned: OwnedCgroup, timeout_seconds: float) -> bool:
    try:
        descriptor = _open_owned_cgroup_control(owned, "cgroup.kill", os.O_WRONLY)
        try:
            if os.write(descriptor, b"1\n") != 2:
                return False
        finally:
            os.close(descriptor)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            _reap_adopted_children()
            if not _cgroup_is_populated(owned):
                return True
            time.sleep(0.05)
        _reap_adopted_children()
        return not _cgroup_is_populated(owned)
    except (Glm47HarnessError, OSError):
        return False


def cleanup_owned_cgroup(owned: OwnedCgroup, timeout_seconds: float) -> bool:
    try:
        if not quiesce_owned_cgroup(owned, timeout_seconds):
            return False
        if not _owned_cgroup_identity_matches(owned):
            return False
        os.rmdir(owned.path.name, dir_fd=owned.parent_descriptor)
        return True
    except OSError:
        return False
    finally:
        os.close(owned.descriptor)
        os.close(owned.parent_descriptor)


def _enter_owned_cgroup(cgroup_procs_descriptor: int) -> None:
    if os.write(cgroup_procs_descriptor, b"0\n") != 2:
        raise OSError("short write while entering validator cgroup")
    os.close(cgroup_procs_descriptor)


def _require_single_threaded_harness() -> None:
    try:
        tasks = tuple(entry.name for entry in Path("/proc/self/task").iterdir())
    except OSError as error:
        raise Glm47HarnessError("cannot verify harness thread count") from error
    if tasks != (str(os.getpid()),):
        raise Glm47HarnessError("pre-exec cgroup entry requires one harness thread")


def _reap_adopted_children() -> None:
    while True:
        try:
            process_id, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if process_id == 0:
            return


def _process_group_has_members(process_group_id: int) -> bool | None:
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            fields = _process_stat_fields(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (Glm47HarnessError, ValueError, IndexError):
            try:
                fields = _process_stat_fields(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (Glm47HarnessError, ValueError, IndexError):
                return None
        if int(fields[2]) == process_group_id:
            return True
    return False


def cleanup_all_owned_processes(
    config: ValidationConfig,
    registry: OwnershipRegistry,
    default_log_path: str,
) -> bool:
    cleanup = True
    all_groups = {process.process_group_id for process in registry.processes}

    def discover() -> dict[int, tuple[int, int]] | None:
        nonlocal cleanup
        observed = _owned_token_processes(registry.owner_token)
        if observed is None:
            return None
        for process_id, (process_group_id, start_time_ticks) in observed.items():
            all_groups.add(process_group_id)
            try:
                registry.register(
                    OwnedProcess(
                        host_name=config.host.hostname,
                        pid=process_id,
                        process_group_id=process_group_id,
                        start_time_ticks=start_time_ticks,
                        transport_pid=process_id,
                        namespace=config.namespace,
                        owner_token=registry.owner_token,
                        log_path=default_log_path,
                    )
                )
            except (Glm47HarnessError, OSError, ValueError):
                # Process termination must survive an evidence-publication failure.
                cleanup = False
        return observed

    term_signaled_groups: set[int] = set()
    term_signaled_processes: set[tuple[int, int]] = set()
    quiet_since: float | None = None
    deadline = time.monotonic() + config.timeouts.cleanup_seconds
    while time.monotonic() < deadline:
        observed = discover()
        if observed is None:
            cleanup = False
            quiet_since = None
            observed = {}
        for group_id in tuple(all_groups):
            current_identities = {
                (process_id, start_time_ticks)
                for process_id, (observed_group, start_time_ticks) in observed.items()
                if observed_group == group_id
            }
            needs_signal = group_id not in term_signaled_groups or not (
                current_identities <= term_signaled_processes
            )
            if not needs_signal:
                continue
            if _signal_owned_group(group_id, registry.owner_token, signal.SIGTERM):
                term_signaled_groups.add(group_id)
                term_signaled_processes.update(current_identities)
            else:
                cleanup = False
        _reap_adopted_children()
        group_states = tuple(
            _live_group_ownership(group_id, registry.owner_token)
            for group_id in all_groups
        )
        if observed == {} and all(state == "absent" for state in group_states):
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= CLEANUP_QUIET_SECONDS:
                break
        else:
            quiet_since = None
        time.sleep(0.05)

    quiet_since = None
    deadline = time.monotonic() + min(5.0, config.timeouts.cleanup_seconds)
    while time.monotonic() < deadline:
        observed = discover()
        if observed is None:
            cleanup = False
            quiet_since = None
            observed = {}
        for group_id in tuple(all_groups):
            if not _signal_owned_group(group_id, registry.owner_token, signal.SIGKILL):
                cleanup = False
        _reap_adopted_children()
        group_states = tuple(
            _live_group_ownership(group_id, registry.owner_token)
            for group_id in all_groups
        )
        if observed == {} and all(state == "absent" for state in group_states):
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= CLEANUP_QUIET_SECONDS:
                break
        else:
            quiet_since = None
        time.sleep(0.05)

    _reap_adopted_children()
    reap_deadline = time.monotonic() + min(1.0, config.timeouts.cleanup_seconds)
    group_members = tuple(
        _process_group_has_members(group_id) for group_id in all_groups
    )
    while any(value is not False for value in group_members) and (
        time.monotonic() < reap_deadline
    ):
        _reap_adopted_children()
        time.sleep(0.01)
        group_members = tuple(
            _process_group_has_members(group_id) for group_id in all_groups
        )
    _reap_adopted_children()
    group_members = tuple(
        _process_group_has_members(group_id) for group_id in all_groups
    )
    cleanup = cleanup and all(value is False for value in group_members)
    remaining = _owned_token_processes(registry.owner_token)
    return cleanup and remaining == {}


def run_owned_command(
    config: ValidationConfig,
    results: ResultDirectory,
    registry: OwnershipRegistry,
    owned_cgroup: OwnedCgroup,
    latch: SignalLatch,
    *,
    name: str,
    command: Sequence[str],
    timeout_seconds: float,
    environment: Mapping[str, str],
) -> CommandOutcome:
    stdout_name = f"{name}.stdout.log"
    stderr_name = f"{name}.stderr.log"
    start = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    cleanup_succeeded = True
    return_code = 75
    caught_error: BaseException | None = None
    cgroup_procs_descriptor: int | None = None
    with (
        results.create_log(stdout_name) as stdout,
        results.create_log(stderr_name) as stderr,
    ):
        try:
            _require_single_threaded_harness()
            cgroup_procs_descriptor = _open_owned_cgroup_control(
                owned_cgroup, "cgroup.procs", os.O_WRONLY
            )
            process = subprocess.Popen(
                tuple(command),
                stdout=stdout,
                stderr=stderr,
                env=dict(environment),
                pass_fds=(cgroup_procs_descriptor,),
                preexec_fn=functools.partial(
                    _enter_owned_cgroup, cgroup_procs_descriptor
                ),
                start_new_session=True,
            )
            os.close(cgroup_procs_descriptor)
            cgroup_procs_descriptor = None
            registry.register(
                _process_identity(
                    config,
                    process.pid,
                    registry.owner_token,
                    str(results.path_for(stderr_name)),
                )
            )
            deadline = time.monotonic() + timeout_seconds
            while True:
                latch.checkpoint()
                observed = process.poll()
                if observed is not None:
                    return_code = observed
                    break
                if time.monotonic() >= deadline:
                    raise Glm47HarnessError(f"{name} exceeded its timeout")
                time.sleep(0.1)
        except BaseException as error:
            caught_error = error
        finally:
            if cgroup_procs_descriptor is not None:
                os.close(cgroup_procs_descriptor)
            if process is not None:
                cleanup_succeeded = _terminate_owned_group(
                    process.pid,
                    registry.owner_token,
                    config.timeouts.cleanup_seconds,
                )
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.1)
                if process.returncode is not None:
                    return_code = process.returncode
            cleanup_succeeded = (
                quiesce_owned_cgroup(
                    owned_cgroup,
                    min(5.0, config.timeouts.cleanup_seconds),
                )
                and cleanup_succeeded
            )
    return CommandOutcome(
        name=name,
        argv=tuple(command),
        return_code=return_code,
        stdout_name=stdout_name,
        stderr_name=stderr_name,
        elapsed_seconds=time.monotonic() - start,
        cleanup_succeeded=cleanup_succeeded,
        error=(
            None
            if caught_error is None
            else f"{type(caught_error).__name__}: {caught_error}"
        ),
    )


def _resource_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def build_generator_command(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    process_spec_output: Path,
) -> tuple[str, ...]:
    script = Path(deployment.root) / (
        "orchestrator/scripts/create_sglang_kt_glm47_validation_process_spec.py"
    )
    launch_mode_arguments = (
        ("--launch-mode", "serving_baseline")
        if config.phase == "serving_baseline"
        else ()
    )
    return (
        config.runtime_python.path,
        str(script),
        *launch_mode_arguments,
        "--model-path",
        config.model_path,
        "--runtime-python",
        config.runtime_python.path,
        "--output",
        str(process_spec_output),
        "--node-id",
        config.host.node_id,
        "--gpu-uuid",
        config.host.gpu.uuid,
        "--cpu-cores",
        _resource_list(config.host.cpu_cores),
        "--memory-nodes",
        _resource_list(config.host.memory_nodes),
        "--cpu-infer-threads",
        str(config.host.cpu_infer_threads),
        "--threadpool-count",
        str(config.host.threadpool_count),
        "--distributed-coordinator",
        config.distributed_coordinator.argument,
        "--service-endpoint",
        config.service_endpoint.argument,
        "--resident-gpu-experts",
        str(config.resident_gpu_experts),
    )


def build_kernel_command(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    kernel_output: Path,
) -> tuple[str, ...]:
    script = (
        Path(deployment.root) / "orchestrator/scripts/validate_sglang_kt_runtime.py"
    )
    return (
        config.numactl_executable,
        "--physcpubind",
        _resource_list(config.host.cpu_cores),
        "--membind",
        _resource_list(config.host.memory_nodes),
        config.runtime_python.path,
        str(script),
        "--gpu-uuid",
        config.host.gpu.uuid,
        "--build-receipt",
        config.build_receipt.path,
        "--numa-nodes",
        _resource_list(config.host.memory_nodes),
        "--threads-per-subpool",
        _resource_list(config.host.threads_per_subpool),
        "--output",
        str(kernel_output),
    )


def build_model_command(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    *,
    process_spec: Path,
    process_spec_sha256: str,
    kernel_receipt: Path,
    kernel_receipt_sha256: str,
    model_output: Path,
) -> tuple[str, ...]:
    if (
        _SHA256.fullmatch(process_spec_sha256) is None
        or _SHA256.fullmatch(kernel_receipt_sha256) is None
    ):
        raise Glm47HarnessError("model validator input digest is invalid")
    script = (
        Path(deployment.root) / "validator/scripts/validate_sglang_kt_glm47_model.py"
    )
    return (
        config.runtime_python.path,
        str(script),
        "--process-spec",
        str(process_spec),
        "--expected-process-spec-sha256",
        process_spec_sha256,
        "--model-contract",
        config.model_contract.path,
        "--expected-model-contract-receipt-sha256",
        config.model_contract.sha256,
        "--kernel-runtime-receipt",
        str(kernel_receipt),
        "--expected-kernel-receipt-sha256",
        kernel_receipt_sha256,
        "--output",
        str(model_output),
    )


def _require_success(outcome: CommandOutcome) -> None:
    if not outcome.cleanup_succeeded:
        raise Glm47HarnessError(f"{outcome.name} cleanup was not confirmed")
    if outcome.error is not None:
        raise Glm47HarnessError(f"{outcome.name} failed: {outcome.error}")
    if outcome.return_code != 0:
        raise Glm47HarnessError(
            f"{outcome.name} failed with return code {outcome.return_code}"
        )


def _parse_command_json(
    results: ResultDirectory, outcome: CommandOutcome
) -> JsonObject:
    try:
        return _json_object(
            cast(object, json.loads(results.read_bytes(outcome.stdout_name))),
            f"{outcome.name} stdout",
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Glm47HarnessError(f"{outcome.name} stdout is invalid JSON") from error


def _validate_kernel_receipt(
    config: ValidationConfig,
    results: ResultDirectory,
    name: str,
) -> JsonObject:
    receipt = results.read_json(name)
    host = _json_object(receipt.get("host"), "kernel host evidence")
    process = _json_object(host.get("process"), "kernel process evidence")
    kernel_config = _json_object(receipt.get("config"), "kernel config evidence")
    provenance = _json_object(receipt.get("provenance"), "kernel provenance")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "passed"
        or receipt.get("profiler") != "none"
        or host.get("profiler") != "none"
        or process.get("hostname") != config.host.hostname
        or process.get("executable") != config.runtime_python.path
        or process.get("affinity_cpu_ids") != list(config.host.cpu_cores)
        or kernel_config.get("build_receipt_path") != config.build_receipt.path
        or provenance.get("receipt_path") != config.build_receipt.path
        or provenance.get("receipt_sha256") != config.build_receipt.sha256
    ):
        raise Glm47HarnessError("kernel receipt does not bind the overlay/resources")
    return receipt


def _create_scratch(config: ValidationConfig) -> OwnedScratchDirectory:
    scratch = Path(config.scratch_directory)
    if scratch.exists() or scratch.is_symlink() or not scratch.parent.is_dir():
        raise Glm47HarnessError("scratch directory must be a new direct child")
    parent_descriptor = os.open(
        scratch.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.mkdir(scratch.name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(
            scratch.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        observed = os.fstat(descriptor)
        return OwnedScratchDirectory(
            path=scratch,
            parent_descriptor=parent_descriptor,
            descriptor=descriptor,
            device=observed.st_dev,
            inode=observed.st_ino,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise


def _scratch_entry_is_unlinkable(mode: int) -> bool:
    return (
        stat.S_ISLNK(mode)
        or stat.S_ISREG(mode)
        or stat.S_ISSOCK(mode)
        or stat.S_ISFIFO(mode)
    )


def _clean_scratch_descriptor(
    descriptor: int,
    *,
    expected_device: int,
    required_uid: int,
) -> bool:
    try:
        entries = tuple(os.scandir(descriptor))
    except OSError:
        return False
    for entry in entries:
        try:
            observed = entry.stat(follow_symlinks=False)
        except OSError:
            return False
        if observed.st_uid != required_uid:
            return False
        if _scratch_entry_is_unlinkable(observed.st_mode):
            try:
                os.unlink(entry.name, dir_fd=descriptor)
            except OSError:
                return False
            continue
        if not stat.S_ISDIR(observed.st_mode) or observed.st_dev != expected_device:
            return False
        try:
            child_descriptor = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
        except OSError:
            return False
        try:
            child_observed = os.fstat(child_descriptor)
            if (
                child_observed.st_dev != observed.st_dev
                or child_observed.st_ino != observed.st_ino
                or not _clean_scratch_descriptor(
                    child_descriptor,
                    expected_device=expected_device,
                    required_uid=required_uid,
                )
            ):
                return False
        finally:
            os.close(child_descriptor)
        try:
            os.rmdir(entry.name, dir_fd=descriptor)
        except OSError:
            return False
    return True


def cleanup_owned_scratch(owned: OwnedScratchDirectory) -> bool:
    try:
        retained = os.fstat(owned.descriptor)
        current = os.stat(
            owned.path.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(retained.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (retained.st_dev, retained.st_ino) != (owned.device, owned.inode)
            or (current.st_dev, current.st_ino) != (owned.device, owned.inode)
        ):
            return False
        if not _clean_scratch_descriptor(
            owned.descriptor,
            expected_device=owned.device,
            required_uid=os.geteuid(),
        ):
            return False
        current = os.stat(
            owned.path.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (owned.device, owned.inode):
            return False
        os.rmdir(owned.path.name, dir_fd=owned.parent_descriptor)
        os.fsync(owned.parent_descriptor)
        return True
    except OSError:
        return False
    finally:
        os.close(owned.descriptor)
        os.close(owned.parent_descriptor)


def run_pipeline(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    results: ResultDirectory,
    registry: OwnershipRegistry,
    owned_cgroup: OwnedCgroup,
    latch: SignalLatch,
    environment: Mapping[str, str],
    evidence: JsonObject,
    outcomes: list[CommandOutcome],
) -> None:
    process_spec_name = "process-spec.json"
    kernel_name = "kernel-runtime-validation-receipt.json"
    model_name = "model-runtime-validation-receipt.json"

    generator = run_owned_command(
        config,
        results,
        registry,
        owned_cgroup,
        latch,
        name="process-spec-generator",
        command=build_generator_command(
            config, deployment, results.path_for(process_spec_name)
        ),
        timeout_seconds=config.timeouts.generator_seconds,
        environment=environment,
    )
    outcomes.append(generator)
    _require_success(generator)
    generator_result = _parse_command_json(results, generator)
    process_spec_sha256 = generator_result.get("process_spec_sha256")
    if (
        generator_result.get("schema_version") != 1
        or generator_result.get("output") != str(results.path_for(process_spec_name))
        or not isinstance(process_spec_sha256, str)
        or _SHA256.fullmatch(process_spec_sha256) is None
    ):
        raise Glm47HarnessError("process-spec generator result is not bound")
    evidence["generator"] = generator_result
    artifacts = _json_object(evidence["artifacts"], "pipeline artifacts")
    artifacts[process_spec_name] = results.sha256(process_spec_name)

    kernel = run_owned_command(
        config,
        results,
        registry,
        owned_cgroup,
        latch,
        name="kernel-validator",
        command=build_kernel_command(config, deployment, results.path_for(kernel_name)),
        timeout_seconds=config.timeouts.kernel_seconds,
        environment=environment,
    )
    outcomes.append(kernel)
    _require_success(kernel)
    _validate_kernel_receipt(config, results, kernel_name)
    kernel_sha256 = results.sha256(kernel_name)
    artifacts[kernel_name] = kernel_sha256

    model_result: JsonObject | None = None
    if config.phase != "kernel":
        model = run_owned_command(
            config,
            results,
            registry,
            owned_cgroup,
            latch,
            name="model-validator",
            command=build_model_command(
                config,
                deployment,
                process_spec=results.path_for(process_spec_name),
                process_spec_sha256=process_spec_sha256,
                kernel_receipt=results.path_for(kernel_name),
                kernel_receipt_sha256=kernel_sha256,
                model_output=results.path_for(model_name),
            ),
            timeout_seconds=config.timeouts.model_seconds,
            environment=environment,
        )
        outcomes.append(model)
        _require_success(model)
        model_result = _parse_command_json(results, model)
        if (
            model_result.get("schema_version") != 1
            or model_result.get("status") != "passed"
            or model_result.get("profiler") != "none"
            or model_result.get("output") != str(results.path_for(model_name))
            or model_result.get("validator_sha256") != deployment.validator_sha256
            or model_result.get("receipt_sha256") != results.sha256(model_name)
        ):
            raise Glm47HarnessError("model validator result is not admission-bound")
        evidence["model_validator"] = model_result
        artifacts[model_name] = results.sha256(model_name)


def run_validation(
    config: ValidationConfig,
    deployment: DeploymentIdentity,
    results: ResultDirectory,
    latch: SignalLatch,
) -> JsonObject:
    owner_token = f"{config.run_id}:{uuid.uuid4().hex}"
    registry = OwnershipRegistry(config, results, owner_token)
    caught_error: BaseException | None = None
    pipeline_evidence: JsonObject = {
        "generator": None,
        "model_validator": None,
        "artifacts": {},
    }
    preflight: JsonObject | None = None
    outcomes: list[CommandOutcome] = []
    scratch: OwnedScratchDirectory | None = None
    owned_cgroup: OwnedCgroup | None = None
    cgroup_evidence: JsonObject | None = None
    cleanup_succeeded = True
    cleanup_errors: list[str] = []
    default_log = str(results.path_for("model-validator.stderr.log"))
    try:
        _enable_child_subreaper()
        latch.checkpoint()
        owned_cgroup = _create_owned_cgroup(config, owner_token)
        cgroup_evidence = _owned_cgroup_evidence(owned_cgroup)
        registry.bind_cgroup(owned_cgroup)
        preflight = collect_live_preflight(config)
        scratch = _create_scratch(config)
        environment = build_child_environment(config, owner_token)
        run_pipeline(
            config,
            deployment,
            results,
            registry,
            owned_cgroup,
            latch,
            environment,
            pipeline_evidence,
            outcomes,
        )
        verify_runtime_python(config.runtime_python)
        _verify_artifact(config.build_receipt, "build receipt")
        _verify_artifact(config.model_contract, "model contract")
        if load_deployment_identity(Path(deployment.root)) != deployment:
            raise Glm47HarnessError("immutable deployment changed during validation")
    except BaseException as error:
        caught_error = error
    finally:
        latch.begin_cleanup()
        try:
            cleanup_succeeded = cleanup_all_owned_processes(
                config, registry, default_log
            )
            if not cleanup_succeeded:
                cleanup_errors.append("owned process cleanup was not proven complete")
        except BaseException as cleanup_error:
            cleanup_succeeded = False
            cleanup_errors.append(
                "owned process cleanup raised "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        if owned_cgroup is not None:
            try:
                cgroup_cleanup_succeeded = cleanup_owned_cgroup(
                    owned_cgroup, config.timeouts.cleanup_seconds
                )
                cleanup_succeeded = cgroup_cleanup_succeeded and cleanup_succeeded
                if not cgroup_cleanup_succeeded:
                    cleanup_errors.append(
                        "owned validator cgroup cleanup was not proven complete"
                    )
            except BaseException as cleanup_error:
                cleanup_succeeded = False
                cleanup_errors.append(
                    "owned validator cgroup cleanup raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        else:
            cleanup_succeeded = False
            cleanup_errors.append("owned validator cgroup was not established")
        if scratch is not None:
            try:
                scratch_cleanup_succeeded = cleanup_owned_scratch(scratch)
                cleanup_succeeded = scratch_cleanup_succeeded and cleanup_succeeded
                if not scratch_cleanup_succeeded:
                    cleanup_errors.append(
                        "owned scratch cleanup was not proven complete"
                    )
            except BaseException as cleanup_error:
                cleanup_succeeded = False
                cleanup_errors.append(
                    "owned scratch cleanup raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    status = (
        "completed"
        if caught_error is None and cleanup_succeeded
        else ("cleanup_failed" if not cleanup_succeeded else "validation_failed")
    )
    model_checkpoint_verified = status == "completed" and config.phase != "kernel"
    reportable = model_checkpoint_verified
    process_values: list[JsonValue] = [
        cast(JsonObject, cast(object, asdict(process)))
        for process in registry.processes
    ]
    result: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "status": status,
        "phase": config.phase,
        "hca_requirement": config.hca_requirement,
        "reportable": reportable,
        "model_checkpoint_verified": model_checkpoint_verified,
        "performance_comparable": False,
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
        "deployment": {
            "root": deployment.root,
            "orchestrator_sha256": deployment.orchestrator_sha256,
            "validator_sha256": deployment.validator_sha256,
        },
        "preflight": preflight,
        "pipeline": pipeline_evidence,
        "commands": [
            {
                **asdict(outcome),
                "argv": list(outcome.argv),
            }
            for outcome in outcomes
        ],
        "owned_processes": process_values,
    }
    manifest: dict[str, object] = {
        **cast(dict[str, object], cast(object, result)),
        "manifest_writer": "run_sglang_kt_glm47_validation.py",
        "config": config.model_dump(mode="json"),
        "python_bytecode_policy": "PYTHONDONTWRITEBYTECODE=1",
        "live_scratch_root": str(REQUIRED_SCRATCH_ROOT),
    }
    results.write_json(CHILD_MANIFEST_FILENAME, manifest, replace=False)
    results.write_json(
        BENCHMARK_RESULT_FILENAME,
        cast(Mapping[str, object], cast(object, result)),
        replace=False,
    )
    return result


# Public support surface for sibling immutable benchmark harnesses.  These
# aliases keep ownership, cgroup, and lease semantics centralized.
SHA256_PATTERN = _SHA256
lexical_absolute_path = _lexical_absolute_path
systemd_unit_name = _systemd_unit_name
parse_command_json = _parse_command_json
require_success = _require_success
verify_artifact = _verify_artifact
require_single_threaded_harness = _require_single_threaded_harness
open_owned_cgroup_control = _open_owned_cgroup_control
enter_owned_cgroup = _enter_owned_cgroup
process_identity = _process_identity
terminate_owned_group = _terminate_owned_group
signal_owned_group = _signal_owned_group
reap_adopted_children = _reap_adopted_children
live_group_ownership = _live_group_ownership
owned_token_processes = _owned_token_processes
read_json_regular = _read_json_regular
json_object = _json_object
config_sha256 = _config_sha256
enable_child_subreaper = _enable_child_subreaper
create_owned_cgroup = _create_owned_cgroup
owned_cgroup_evidence = _owned_cgroup_evidence
create_scratch = _create_scratch
process_stat_fields = _process_stat_fields
validate_metadata = _validate_metadata
systemd_benchmark_argv = _systemd_benchmark_argv
write_new_json = _write_new_json


class RunArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path | None


class PreparationArguments(argparse.Namespace):
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


def parse_run_arguments(arguments: Sequence[str]) -> RunArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args(arguments, namespace=RunArguments())


def parse_preparation_arguments(arguments: Sequence[str]) -> PreparationArguments:
    parser = argparse.ArgumentParser(
        description="Create immutable GLM-4.7 validation deployments and lease metadata"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-duration-seconds", required=True, type=float)
    parser.add_argument("--cleanup-grace-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args(arguments, namespace=PreparationArguments())


def preparation_main(arguments: Sequence[str]) -> int:
    parsed = parse_preparation_arguments(arguments)
    prepared = prepare_lease(
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
                "benchmark_lease_argv": list(prepared.benchmark_lease_argv),
                "systemd_unit_name": prepared.systemd_unit_name,
                "benchmark_lease_shell_command": shlex.join(
                    prepared.benchmark_lease_argv
                ),
                "minimum_cleanup_grace_seconds": (
                    prepared.minimum_cleanup_grace_seconds
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def run_main(arguments: Sequence[str]) -> int:
    parsed = parse_run_arguments(arguments)
    if not all(
        path.is_absolute()
        for path in (parsed.config, parsed.lease_path, parsed.lock_path)
    ):
        raise Glm47HarnessError("config, lease, and lock paths must be absolute")
    config = load_config(parsed.config)
    if (
        parsed.config
        != Path(config.source.deployment_root) / IMMUTABLE_CONFIG_RELATIVE_PATH
    ):
        raise Glm47HarnessError("run config is not the immutable deployed config")
    if (
        parsed.result_dir is not None
        and str(parsed.result_dir) != config.result_directory
    ):
        raise Glm47HarnessError("--result-dir differs from the immutable config")
    deployment = load_deployment_identity(Path(config.source.deployment_root))
    results = ResultDirectory.inherited(Path(config.result_directory))
    latch = SignalLatch()
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    try:
        validate_active_lease(
            config,
            deployment,
            config_path=parsed.config,
            lease_path=parsed.lease_path,
            lock_path=parsed.lock_path,
        )
        require_no_profiler_state(
            os.environ,
            sys.argv,
            _read_text(Path("/proc/modules"), "loaded kernel modules"),
        )
        if (
            os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or not sys.dont_write_bytecode
        ):
            raise Glm47HarnessError(
                "immutable execution requires bytecode writes disabled"
            )
        for managed in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[managed] = signal.getsignal(managed)
            signal.signal(managed, latch.handle)
        result = run_validation(config, deployment, results, latch)
        return 0 if result.get("status") == "completed" else 2
    finally:
        for managed, previous in previous_handlers.items():
            signal.signal(managed, previous)
        results.close()


def main(arguments: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if arguments is None else arguments)
    if normalized and normalized[0] == "prepare-lease":
        return preparation_main(normalized[1:])
    return run_main(normalized)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Glm47HarnessError, ValidationError, OSError, ValueError) as error:
        print(
            f"GLM-4.7 leased validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
