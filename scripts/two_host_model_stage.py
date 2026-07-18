#!/usr/bin/env python3
"""Ownership-safe model staging for the leased two-host Exo proof.

This command is a staging transaction, not a performance benchmark. It must be
the direct child of ``scripts/benchmark_lease.py`` while that wrapper owns the
FwuffyDwagon coordination lock. The command publishes the wrapper's v1 runtime
and result fragments so every local or remote process remains attributable and
cleanup failures retain the lease tombstone.

The remote command receives and executes these exact script bytes. Consequently
both hosts use the same snapshot verifier without requiring a synchronized Exo
source deployment merely to stage the checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import re
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Protocol, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_HEX_REVISION = re.compile(r"[0-9a-f]{40}")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SAFE_SSH_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,254}")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MODEL_RECEIPT = ".exo-huggingface-revision.json"
_INDEX_FILENAME = "model.safetensors.index.json"
_OWNER_TOKEN_ENVIRONMENT = "EXO_BENCHMARK_OWNER_TOKEN"
_NAMESPACE_ENVIRONMENT = "EXO_NAMESPACE"
_RESULT_DIRECTORY_FD_ENVIRONMENT = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
_RUNTIME_METADATA = "runtime-metadata.json"
_BENCHMARK_RESULT = "benchmark-result.json"
_DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
_DEFAULT_LOCK_PATH = Path("/run/lock/fwuffydwagon-benchmark.lock")
_LEASE_BIND_TIMEOUT_SECONDS = 5.0
_LEASE_HEARTBEAT_MAX_AGE = timedelta(minutes=2)
_LEASE_METADATA_MAX_AGE = timedelta(minutes=15)
_MAX_REMOTE_IDENTITY_BYTES = 64 * 1024
_MAX_REMOTE_RESPONSE_BYTES = 64 * 1024 * 1024
_REMOTE_LOADER = (
    "import base64,sys,types;"
    "encoded=sys.stdin.buffer.readline().rstrip(b'\\n');"
    "source=base64.b64decode(encoded,validate=True);"
    "sys.argv=['two_host_model_stage.py','remote-helper'];"
    "module=types.ModuleType('_exo_model_stage_remote');"
    "module.__file__='/dev/stdin';"
    "sys.modules[module.__name__]=module;"
    "exec(compile(source,'two_host_model_stage.py','exec'),module.__dict__);"
    "raise SystemExit(module.remote_helper_main())"
)


class LeaseMetadataValidator(Protocol):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


class StageError(RuntimeError):
    """A fail-closed staging or ownership error."""


class OperationError(StageError):
    """A process operation that also reports whether cleanup was confirmed."""

    def __init__(self, message: str, *, cleanup_confirmed: bool) -> None:
        super().__init__(message)
        self.cleanup_confirmed = cleanup_confirmed


class ManagedSignalError(StageError):
    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received managed signal {signal_number}")
        self.signal_number = signal_number


@dataclass
class SignalLatch:
    signal_number: int | None = None
    cleanup_started: bool = False

    def handle(self, signal_number: int, _frame: object) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number

    def checkpoint(self) -> None:
        if self.signal_number is not None and not self.cleanup_started:
            raise ManagedSignalError(self.signal_number)

    def begin_cleanup(self) -> None:
        self.cleanup_started = True


@dataclass(frozen=True)
class LeasePreparation:
    metadata: JsonObject
    child_argv: tuple[str, ...]
    benchmark_lease_argv: tuple[str, ...]
    generated_at: str
    minimum_cleanup_grace_seconds: float

    def machine_output(self, metadata_output: Path) -> JsonObject:
        return {
            "schema_version": 1,
            "metadata_output": str(metadata_output),
            "generated_at": self.generated_at,
            "minimum_cleanup_grace_seconds": self.minimum_cleanup_grace_seconds,
            "child_argv": list(self.child_argv),
            "benchmark_lease_argv": list(self.benchmark_lease_argv),
        }


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelSpec(StrictModel):
    model_id: str = Field(min_length=3)
    revision: str
    expected_indexed_bytes: int = Field(gt=0)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        parts = value.split("/")
        if (
            len(parts) != 2
            or any(not part or part in {".", ".."} for part in parts)
            or any(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) is None
                for part in parts
            )
        ):
            raise ValueError("model_id must be an exact owner/repository identifier")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _HEX_REVISION.fullmatch(value) is None:
            raise ValueError("revision must be an exact lowercase 40-hex commit")
        return value

    @property
    def directory_name(self) -> str:
        return f"{self.model_id.replace('/', '--')}--{self.revision}"


_REQUIRED_SSH_OPTIONS = frozenset(
    {
        "BatchMode",
        "StrictHostKeyChecking",
        "UserKnownHostsFile",
        "ConnectTimeout",
        "ServerAliveInterval",
        "ServerAliveCountMax",
        "ControlMaster",
        "ControlPath",
        "ControlPersist",
        "RequestTTY",
    }
)


class SshConfig(StrictModel):
    target: str
    executable: str
    remote_python_executable: str
    options: tuple[str, ...]

    @model_validator(mode="after")
    def validate_ssh(self) -> "SshConfig":
        if _SAFE_SSH_TARGET.fullmatch(self.target) is None or self.target.startswith(
            "-"
        ):
            raise ValueError("SSH target contains unsafe command characters")
        _validate_lexical_absolute_path(self.executable, "SSH executable")
        _validate_lexical_absolute_path(
            self.remote_python_executable, "remote Python executable"
        )
        if len(set(self.options)) != len(self.options):
            raise ValueError("SSH options must be unique")
        parsed: dict[str, str] = {}
        for option in self.options:
            if not option.startswith("-o") or "=" not in option[2:]:
                raise ValueError("SSH options must use the exact -oName=value form")
            key, value = option[2:].split("=", 1)
            if key not in _REQUIRED_SSH_OPTIONS or key in parsed:
                raise ValueError(f"unsupported or repeated SSH option {key!r}")
            if not value or any(
                character.isspace() or character == "\0" for character in value
            ):
                raise ValueError(f"SSH option {key} has an unsafe value")
            parsed[key] = value
        if set(parsed) != set(_REQUIRED_SSH_OPTIONS):
            missing = sorted(_REQUIRED_SSH_OPTIONS - set(parsed))
            raise ValueError(
                f"SSH options do not exactly cover required keys: {missing}"
            )
        exact_values = {
            "BatchMode": "yes",
            "StrictHostKeyChecking": "yes",
            "ControlMaster": "no",
            "ControlPath": "none",
            "ControlPersist": "no",
            "RequestTTY": "no",
        }
        for key, expected in exact_values.items():
            if parsed[key] != expected:
                raise ValueError(f"SSH option {key} must equal {expected}")
        known_hosts = parsed["UserKnownHostsFile"]
        _validate_lexical_absolute_path(known_hosts, "SSH known-hosts file")
        for key in ("ConnectTimeout", "ServerAliveInterval", "ServerAliveCountMax"):
            if not parsed[key].isdigit() or int(parsed[key]) <= 0:
                raise ValueError(f"SSH option {key} must be a positive integer")
        return self


class AcquisitionConfig(StrictModel):
    hf_executable: str
    source_snapshot: str | None = None
    environment: dict[str, str]

    @model_validator(mode="after")
    def validate_acquisition(self) -> "AcquisitionConfig":
        _validate_lexical_absolute_path(self.hf_executable, "hf executable")
        if self.source_snapshot is not None:
            _validate_lexical_absolute_path(self.source_snapshot, "source snapshot")
        for name, value in self.environment.items():
            if _ENVIRONMENT_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid environment variable name {name!r}")
            if "\0" in value:
                raise ValueError(f"environment variable {name} contains NUL")
        if {
            _OWNER_TOKEN_ENVIRONMENT,
            _NAMESPACE_ENVIRONMENT,
        } & self.environment.keys():
            raise ValueError(
                "acquisition environment contains a reserved ownership key"
            )
        return self


class TimeoutConfig(StrictModel):
    lease_bind_seconds: float = Field(gt=0)
    local_stage_seconds: float = Field(gt=0)
    remote_probe_seconds: float = Field(gt=0)
    remote_transfer_seconds: float = Field(gt=0)
    cleanup_seconds: float = Field(gt=0)
    poll_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_finite(self) -> "TimeoutConfig":
        values = (
            ("lease_bind_seconds", self.lease_bind_seconds),
            ("local_stage_seconds", self.local_stage_seconds),
            ("remote_probe_seconds", self.remote_probe_seconds),
            ("remote_transfer_seconds", self.remote_transfer_seconds),
            ("cleanup_seconds", self.cleanup_seconds),
            ("poll_seconds", self.poll_seconds),
        )
        for name, value in values:
            if not math.isfinite(value):
                raise ValueError(f"timeout {name} must be finite")
        return self


class GitIdentity(StrictModel):
    commit: str
    dirty: bool
    dirty_file_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_identity(self) -> "GitIdentity":
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.commit) is None:
            raise ValueError("git commit must be an exact lowercase object identifier")
        for relative_path, digest in self.dirty_file_hashes.items():
            _validate_relative_file_name(relative_path)
            if _HEX_SHA256.fullmatch(digest) is None:
                raise ValueError("dirty file hashes must be lowercase SHA-256")
        if self.dirty != bool(self.dirty_file_hashes):
            raise ValueError(
                "git dirty must match whether dirty_file_hashes is nonempty"
            )
        return self


class GpuBinding(StrictModel):
    uuid: str = Field(min_length=1)
    pci_address: str = Field(min_length=1)


class CpuBinding(StrictModel):
    cpu_set: str = Field(min_length=1)
    numa_nodes: tuple[int, ...]
    memory_policy: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cpu_binding(self) -> "CpuBinding":
        if not self.numa_nodes or any(node < 0 for node in self.numa_nodes):
            raise ValueError("NUMA nodes must contain nonnegative integers")
        return self


class HcaBinding(StrictModel):
    device: str = Field(min_length=1)
    port: int = Field(gt=0)
    ip_address: str | None = None
    gid: str | None = None

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("HCA device must be a safe kernel device name")
        return value

    @model_validator(mode="after")
    def validate_hca_identity(self) -> "HcaBinding":
        if (self.ip_address is None) == (self.gid is None):
            raise ValueError(
                "HCA binding must contain exactly one of ip_address or gid"
            )
        field_name = "ip_address" if self.ip_address is not None else "gid"
        raw_address = self.ip_address if self.ip_address is not None else self.gid
        assert raw_address is not None
        if "%" in raw_address:
            raise ValueError(f"HCA {field_name} must not contain a scope zone")
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as error:
            raise ValueError(f"HCA {field_name} must be an IP address") from error
        if str(address) != raw_address:
            raise ValueError(f"HCA {field_name} must use canonical compressed form")
        if self.ip_address is not None and address.is_unspecified:
            raise ValueError("HCA ip_address must not be unspecified")
        if self.gid is not None and (
            address.version != 6
            or address.is_unspecified
            or int(address) & ((1 << 64) - 1) == 0
        ):
            raise ValueError("HCA gid must be a port-specific IPv6 GID")
        return self


class SourceDeployment(StrictModel):
    path: str
    commit: str
    dirty_file_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_deployment(self) -> "SourceDeployment":
        _validate_lexical_absolute_path(self.path, "source deployment")
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.commit) is None:
            raise ValueError(
                "source deployment commit must be an exact object identifier"
            )
        for relative_path, digest in self.dirty_file_hashes.items():
            _validate_relative_file_name(relative_path)
            if _HEX_SHA256.fullmatch(digest) is None:
                raise ValueError("source dirty file hashes must be lowercase SHA-256")
        return self


class LeaseMetadataInputs(StrictModel):
    reserved_ports: tuple[int, ...]
    git: GitIdentity
    staging_script_sha256: str
    gpu_bindings: dict[str, tuple[GpuBinding, ...]]
    cpu_bindings: dict[str, CpuBinding]
    hca_bindings: dict[str, tuple[HcaBinding, ...]]
    source_deployments: dict[str, SourceDeployment]
    owner_pids: dict[str, tuple[int, ...]]

    @model_validator(mode="after")
    def validate_inputs(self) -> "LeaseMetadataInputs":
        if (
            not self.reserved_ports
            or len(set(self.reserved_ports)) != len(self.reserved_ports)
            or any(port < 1 or port > 65535 for port in self.reserved_ports)
        ):
            raise ValueError("reserved ports must be unique values between 1 and 65535")
        if _HEX_SHA256.fullmatch(self.staging_script_sha256) is None:
            raise ValueError("staging script SHA-256 must be lowercase SHA-256")
        return self


class StageConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str
    namespace: str
    result_directory: str
    local_host_name: str
    remote_host_name: str
    model: ModelSpec
    local_destination: str
    remote_destination: str
    ssh: SshConfig
    acquisition: AcquisitionConfig
    timeouts: TimeoutConfig
    lease_metadata: LeaseMetadataInputs

    @model_validator(mode="after")
    def validate_stage(self) -> "StageConfig":
        for value, description in (
            (self.run_id, "run_id"),
            (self.namespace, "namespace"),
            (self.local_host_name, "local_host_name"),
            (self.remote_host_name, "remote_host_name"),
        ):
            if _SAFE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{description} is not a safe identifier")
        if self.run_id not in self.namespace:
            raise ValueError("namespace must contain the unique run_id")
        if self.local_host_name == self.remote_host_name:
            raise ValueError("local and remote host names must differ")
        ssh_host = self.ssh.target.rsplit("@", 1)[-1]
        if ssh_host != self.remote_host_name:
            raise ValueError("SSH target must name the exact remote_host_name")
        _validate_lexical_absolute_path(self.result_directory, "result directory")
        for raw_path, description in (
            (self.local_destination, "local destination"),
            (self.remote_destination, "remote destination"),
        ):
            _validate_lexical_absolute_path(raw_path, description)
            if Path(raw_path).name != self.model.directory_name:
                raise ValueError(
                    f"{description} must end in the exact revision-pinned Exo name "
                    f"{self.model.directory_name}"
                )
        if self.acquisition.source_snapshot in {
            self.local_destination,
            self.remote_destination,
        }:
            raise ValueError("source snapshot must differ from both destinations")
        if self.acquisition.source_snapshot is not None:
            source = Path(self.acquisition.source_snapshot)
            local_destination = Path(self.local_destination)
            if (
                source in local_destination.parents
                or local_destination in source.parents
            ):
                raise ValueError(
                    "source snapshot and local destination must not overlap"
                )
        result_directory = Path(self.result_directory)
        local_destination = Path(self.local_destination)
        if (
            result_directory == local_destination
            or result_directory in local_destination.parents
            or local_destination in result_directory.parents
        ):
            raise ValueError("result directory and local destination must not overlap")
        if self.acquisition.source_snapshot is not None:
            source = Path(self.acquisition.source_snapshot)
            if (
                result_directory == source
                or result_directory in source.parents
                or source in result_directory.parents
            ):
                raise ValueError(
                    "result directory and source snapshot must not overlap"
                )
        host_names = {self.local_host_name, self.remote_host_name}
        for section_name, section in (
            ("gpu_bindings", self.lease_metadata.gpu_bindings),
            ("cpu_bindings", self.lease_metadata.cpu_bindings),
            ("hca_bindings", self.lease_metadata.hca_bindings),
            ("source_deployments", self.lease_metadata.source_deployments),
            ("owner_pids", self.lease_metadata.owner_pids),
        ):
            if set(section) != host_names:
                raise ValueError(
                    f"lease {section_name} must exactly match configured hosts"
                )
        for host_name in host_names:
            if not self.lease_metadata.hca_bindings[host_name]:
                raise ValueError(
                    f"lease HCA bindings for {host_name} must not be empty"
                )
            hca_bindings = self.lease_metadata.hca_bindings[host_name]
            if len({(binding.device, binding.port) for binding in hca_bindings}) != len(
                hca_bindings
            ):
                raise ValueError(
                    f"lease HCA bindings for {host_name} repeat a device port"
                )
            deployment = self.lease_metadata.source_deployments[host_name]
            if (
                deployment.commit != self.lease_metadata.git.commit
                or deployment.dirty_file_hashes
                != self.lease_metadata.git.dirty_file_hashes
            ):
                raise ValueError(
                    f"source deployment identity for {host_name} must match git"
                )
            if any(
                process_id <= 0
                for process_id in self.lease_metadata.owner_pids[host_name]
            ):
                raise ValueError("owner PIDs must be positive")
        return self


class SnapshotVerification(StrictModel):
    model_id: str
    revision: str
    indexed_bytes: int = Field(gt=0)
    shard_count: int = Field(gt=0)
    manifest: dict[str, str]

    @field_validator("manifest")
    @classmethod
    def validate_manifest(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("snapshot manifest must not be empty")
        for relative_path, digest in value.items():
            _validate_relative_file_name(relative_path)
            if _HEX_SHA256.fullmatch(digest) is None:
                raise ValueError("manifest values must be lowercase SHA-256")
        return value


class OwnedProcess(StrictModel):
    host_name: str
    pid: int = Field(gt=0)
    process_group_id: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)
    owner_token: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    transport_pid: int = Field(gt=0)
    log_path: str

    @field_validator("log_path")
    @classmethod
    def validate_log_path(cls, value: str) -> str:
        _validate_lexical_absolute_path(value, "owned process log")
        return value


class ProcessOperationResult(StrictModel):
    verification: SnapshotVerification | None
    cleanup_confirmed: bool
    installed: bool


class RemoteRequest(StrictModel):
    schema_version: Literal[1]
    operation: Literal["probe", "receive"]
    run_id: str
    namespace: str
    owner_token: str
    expected_host_name: str
    destination: str
    temporary_path: str
    model: ModelSpec

    @model_validator(mode="after")
    def validate_request(self) -> "RemoteRequest":
        if _SAFE_IDENTIFIER.fullmatch(self.expected_host_name) is None:
            raise ValueError("expected remote host name is not a safe identifier")
        _validate_lexical_absolute_path(self.destination, "remote destination")
        _validate_lexical_absolute_path(self.temporary_path, "remote temporary path")
        destination = Path(self.destination)
        temporary = Path(self.temporary_path)
        if temporary.parent != destination.parent or temporary == destination:
            raise ValueError("remote temporary path must be a distinct sibling")
        if destination.name != self.model.directory_name:
            raise ValueError("remote destination has the wrong revision suffix")
        expected_prefix = f".{destination.name}.{self.run_id}."
        if not temporary.name.startswith(
            expected_prefix
        ) or not temporary.name.endswith(".stage"):
            raise ValueError("remote temporary path is not owned by this run")
        return self


class RemoteIdentity(StrictModel):
    schema_version: Literal[1]
    kind: Literal["identity"]
    pid: int = Field(gt=0)
    process_group_id: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)
    owner_token: str
    namespace: str
    host_name: str


class RemoteResponse(StrictModel):
    schema_version: Literal[1]
    kind: Literal["result"]
    status: Literal["completed", "failed"]
    verification: SnapshotVerification | None
    installed: bool
    cleanup_succeeded: bool
    error: str | None


@dataclass
class OwnedTreeEntry:
    relative_path: str
    descriptor: int
    device: int
    inode: int
    kind: Literal["directory", "regular"]

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass
class OwnedDirectory:
    path: Path
    parent_descriptor: int
    descriptor: int
    device: int
    inode: int
    entries: dict[str, OwnedTreeEntry] = field(default_factory=dict)

    def close(self) -> None:
        for entry in self.entries.values():
            entry.close()
        self.entries.clear()
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


def reconcile_remote_cleanup(
    local_transport_cleanup_succeeded: bool,
    response: RemoteResponse | None,
) -> bool:
    """Require a validated remote receipt in addition to local SSH cleanup."""
    return (
        local_transport_cleanup_succeeded
        and response is not None
        and response.cleanup_succeeded
    )


class StagingEffects(Protocol):
    def inspect_local_destination(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification | None: ...

    def create_local_temporary(self, destination: Path, owned_name: str) -> Path: ...

    def copy_preverified_snapshot(
        self,
        source: Path,
        destination: Path,
        model: ModelSpec,
        latch: SignalLatch,
    ) -> None: ...

    def download_snapshot(
        self,
        config: StageConfig,
        temporary_path: Path,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> bool: ...

    def write_revision_receipt(self, path: Path, model: ModelSpec) -> None: ...

    def verify_local_snapshot(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification: ...

    def install_local_snapshot(
        self, temporary_path: Path, destination: Path
    ) -> None: ...

    def cleanup_local_temporary(self, path: Path) -> bool: ...

    def probe_remote_destination(
        self,
        config: StageConfig,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult: ...

    def transfer_remote_snapshot(
        self,
        config: StageConfig,
        local_snapshot: Path,
        local_verification: SnapshotVerification,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult: ...

    def write_result_json(self, filename: str, value: Mapping[str, object]) -> None: ...


def _validate_lexical_absolute_path(raw_path: str, description: str) -> None:
    if not raw_path or "\0" in raw_path:
        raise ValueError(f"{description} must be a nonempty path without NUL")
    path = Path(raw_path)
    if not path.is_absolute() or str(path) != raw_path or ".." in path.parts:
        raise ValueError(f"{description} must be absolute and lexically canonical")


def _validate_relative_file_name(raw_path: str) -> PurePosixPath:
    if not raw_path or "\0" in raw_path or "\\" in raw_path:
        raise StageError(f"unsafe snapshot file path {raw_path!r}")
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or str(path) != raw_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise StageError(f"unsafe snapshot file path {raw_path!r}")
    return path


def _parse_json_object(raw_value: str | bytes, description: str) -> JsonObject:
    try:
        parsed = cast(
            object,
            json.loads(
                raw_value,
                parse_constant=lambda value: (_raise_invalid_constant(value)),
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise StageError(f"{description} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], parsed)
    ):
        raise StageError(f"{description} must be a JSON object")
    return cast(JsonObject, parsed)


def _raise_invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_directory_entry_name(filename: str) -> str:
    if not filename or filename in {".", ".."} or "/" in filename or "\0" in filename:
        raise StageError(f"unsafe directory entry name {filename!r}")
    return filename


def _open_directory_without_symlinks(path: Path) -> int:
    _validate_lexical_absolute_path(str(path), "directory path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            _validate_directory_entry_name(component)
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise StageError(
            f"cannot open directory without symlinks {path}: {error}"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _validate_result_directory_descriptor(path: Path, descriptor: int) -> None:
    """Bind the wrapper-provided descriptor to the configured result path."""
    try:
        retained = os.fstat(descriptor)
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise StageError(
            f"cannot validate trusted result directory: {error}"
        ) from error
    if not stat.S_ISDIR(retained.st_mode):
        raise StageError("trusted result descriptor is not a directory")
    if not stat.S_ISDIR(observed.st_mode) or (
        observed.st_dev,
        observed.st_ino,
    ) != (retained.st_dev, retained.st_ino):
        raise StageError("trusted result directory path identity changed")


def _inherited_result_directory_descriptor(path: Path) -> int:
    raw_descriptor = os.environ.get(_RESULT_DIRECTORY_FD_ENVIRONMENT)
    if (
        raw_descriptor is None
        or not raw_descriptor.isascii()
        or not raw_descriptor.isdigit()
    ):
        raise StageError(
            f"lease wrapper did not pass {_RESULT_DIRECTORY_FD_ENVIRONMENT}"
        )
    descriptor = int(raw_descriptor)
    _validate_result_directory_descriptor(path, descriptor)
    fcntl.fcntl(descriptor, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    return descriptor


def _open_owned_directory(path: Path) -> OwnedDirectory:
    parent_descriptor = _open_directory_without_symlinks(path.parent)
    try:
        descriptor = os.open(
            _validate_directory_entry_name(path.name),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    status_value = os.fstat(descriptor)
    return OwnedDirectory(
        path=path,
        parent_descriptor=parent_descriptor,
        descriptor=descriptor,
        device=status_value.st_dev,
        inode=status_value.st_ino,
    )


def _owned_directory_path_matches(directory: OwnedDirectory) -> bool:
    try:
        observed = os.stat(
            directory.path.name,
            dir_fd=directory.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return stat.S_ISDIR(observed.st_mode) and (
        observed.st_dev,
        observed.st_ino,
    ) == (directory.device, directory.inode)


def _create_owned_directory(path: Path) -> OwnedDirectory:
    """Create and retain a directory, reporting truthful cleanup on setup failure."""
    parent_descriptor = _open_directory_without_symlinks(path.parent)
    name = _validate_directory_entry_name(path.name)
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
    except FileExistsError as error:
        os.close(parent_descriptor)
        raise StageError(f"owned temporary path already exists: {path}") from error
    except BaseException:
        os.close(parent_descriptor)
        raise

    descriptor: int | None = None
    try:
        created = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
            raise StageError("created temporary is not a real directory")
        created_identity = (created.st_dev, created.st_ino)
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        retained = os.fstat(descriptor)
        if (retained.st_dev, retained.st_ino) != created_identity:
            raise StageError("created temporary identity changed while opening")
        return OwnedDirectory(
            path=path,
            parent_descriptor=parent_descriptor,
            descriptor=descriptor,
            device=retained.st_dev,
            inode=retained.st_ino,
        )
    except BaseException as error:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise OperationError(
            f"cannot retain owned temporary {path}: {type(error).__name__}: {error}",
            # Without a retained descriptor there is no conditional unlink primitive.
            # Preserve the path and require manual tombstone clearance.
            cleanup_confirmed=False,
        ) from error


def _atomic_write_json_at(
    directory_descriptor: int, filename: str, value: Mapping[str, object]
) -> None:
    entry_name = _validate_directory_entry_name(filename)
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = f".{entry_name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(serialized)
            output.flush()
            os.fchmod(output.fileno(), 0o644)
            os.fsync(output.fileno())
        try:
            existing = os.stat(
                entry_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise StageError(
                    "refusing to replace symlink or non-regular JSON entry "
                    f"{entry_name!r}"
                )
        os.replace(
            temporary,
            entry_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_descriptor)


def _create_owned_json_at(
    directory: OwnedDirectory, filename: str, value: Mapping[str, object]
) -> None:
    """Create a private-tree JSON file and journal it before writing bytes."""
    entry_name = _validate_directory_entry_name(filename)
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        entry_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory.descriptor,
    )
    try:
        _retain_owned_entry(directory, entry_name, descriptor, "regular")
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(serialized)
            output.flush()
            os.fchmod(output.fileno(), 0o644)
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def _atomic_create_json(path: Path, value: Mapping[str, object]) -> None:
    directory_descriptor = _open_directory_without_symlinks(path.parent)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        serialized = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(serialized)
            output.flush()
            os.fchmod(output.fileno(), 0o644)
            os.fsync(output.fileno())
        try:
            os.link(
                temporary,
                _validate_directory_entry_name(path.name),
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise StageError(f"JSON output already exists: {path}") from error
        os.fsync(directory_descriptor)
    finally:
        if created:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _read_regular_file_path(path: Path, description: str) -> bytes:
    """Read one regular file through a symlink-free retained parent."""
    parent_descriptor = _open_directory_without_symlinks(path.parent)
    try:
        descriptor = os.open(
            _validate_directory_entry_name(path.name),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_descriptor,
        )
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise StageError(f"{description} is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as input_file:
                return input_file.read()
        finally:
            os.close(descriptor)
    except OSError as error:
        raise StageError(f"cannot safely read {description} {path}: {error}") from error
    finally:
        os.close(parent_descriptor)


def _read_json_object_path(path: Path, description: str) -> JsonObject:
    return _parse_json_object(_read_regular_file_path(path, description), description)


def _require_canonical_directory(path: Path, description: str) -> None:
    try:
        path_status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise StageError(f"cannot inspect {description} {path}: {error}") from error
    if not stat.S_ISDIR(path_status.st_mode) or stat.S_ISLNK(path_status.st_mode):
        raise StageError(f"{description} is not a real directory: {path}")
    if resolved != path:
        raise StageError(f"{description} is not canonical: {path}")


def _require_canonical_regular_file(
    path: Path, description: str, *, executable: bool = False
) -> None:
    try:
        path_status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise StageError(f"cannot inspect {description} {path}: {error}") from error
    if not stat.S_ISREG(path_status.st_mode) or stat.S_ISLNK(path_status.st_mode):
        raise StageError(f"{description} is not a regular file: {path}")
    if resolved != path:
        raise StageError(f"{description} is not canonical: {path}")
    if executable and not os.access(path, os.X_OK):
        raise StageError(f"{description} is not executable: {path}")


def _open_relative_regular_file(root_descriptor: int, relative_path: str) -> int:
    relative = _validate_relative_file_name(relative_path)
    directory_descriptor = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            os.close(descriptor)
            raise StageError(f"snapshot entry is not a regular file: {relative_path}")
        return descriptor
    except OSError as error:
        raise StageError(
            f"cannot safely open snapshot file {relative_path}: {error}"
        ) from error
    finally:
        os.close(directory_descriptor)


def _read_relative_file(root_descriptor: int, relative_path: str) -> bytes:
    descriptor = _open_relative_regular_file(root_descriptor, relative_path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def _sha256_file_at(
    root_descriptor: int,
    relative_path: str,
    checkpoint: Callable[[], None] = lambda: None,
) -> str:
    digest = hashlib.sha256()
    descriptor = _open_relative_regular_file(root_descriptor, relative_path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while chunk := source.read(1024 * 1024):
                checkpoint()
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _snapshot_regular_files(
    root_descriptor: int,
    checkpoint: Callable[[], None] = lambda: None,
) -> tuple[str, ...]:
    files: list[str] = []

    def walk(directory_descriptor: int, prefix: PurePosixPath | None) -> None:
        with os.scandir(directory_descriptor) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                checkpoint()
                relative = (
                    PurePosixPath(entry.name) if prefix is None else prefix / entry.name
                )
                relative_name = str(relative)
                _validate_relative_file_name(relative_name)
                item_status = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(item_status.st_mode):
                    raise StageError(f"snapshot contains symlink {relative_name}")
                if stat.S_ISDIR(item_status.st_mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_descriptor,
                    )
                    try:
                        walk(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(item_status.st_mode):
                    files.append(relative_name)
                else:
                    raise StageError(
                        f"snapshot contains a non-regular file {relative_name}"
                    )

    walk(root_descriptor, None)
    return tuple(files)


def _retain_owned_entry(
    directory: OwnedDirectory,
    relative_path: str,
    descriptor: int,
    kind: Literal["directory", "regular"],
) -> None:
    relative = _validate_relative_file_name(relative_path)
    normalized = str(relative)
    if normalized in directory.entries:
        raise StageError(f"owned tree entry was journaled twice: {normalized}")
    if len(relative.parts) > 1:
        parent_name = str(PurePosixPath(*relative.parts[:-1]))
        parent = directory.entries.get(parent_name)
        if parent is None or parent.kind != "directory":
            raise StageError(f"owned tree parent is not journaled: {parent_name}")
    retained_descriptor = os.dup(descriptor)
    retained = os.fstat(retained_descriptor)
    expected_kind = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if not expected_kind(retained.st_mode):
        os.close(retained_descriptor)
        raise StageError(f"owned tree entry has the wrong type: {normalized}")
    directory.entries[normalized] = OwnedTreeEntry(
        relative_path=normalized,
        descriptor=retained_descriptor,
        device=retained.st_dev,
        inode=retained.st_ino,
        kind=kind,
    )


def _journal_existing_owned_tree(
    directory: OwnedDirectory,
    checkpoint: Callable[[], None] = lambda: None,
) -> None:
    """Adopt a stopped owned child's output and retain every entry identity."""
    if directory.entries:
        raise StageError("owned tree adoption requires an empty journal")

    def walk(parent_descriptor: int, prefix: PurePosixPath | None) -> None:
        with os.scandir(parent_descriptor) as scanned:
            entries = sorted(scanned, key=lambda value: value.name)
        for entry in entries:
            checkpoint()
            relative = (
                PurePosixPath(entry.name) if prefix is None else prefix / entry.name
            )
            relative_name = str(relative)
            _validate_relative_file_name(relative_name)
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
                kind: Literal["directory", "regular"] = "directory"
            elif stat.S_ISREG(observed.st_mode):
                flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
                kind = "regular"
            else:
                raise StageError(
                    f"owned child produced a non-regular entry: {relative_name}"
                )
            descriptor = os.open(entry.name, flags, dir_fd=parent_descriptor)
            try:
                retained = os.fstat(descriptor)
                if (retained.st_dev, retained.st_ino) != (
                    observed.st_dev,
                    observed.st_ino,
                ):
                    raise StageError(
                        f"owned child entry changed while journaling: {relative_name}"
                    )
                _retain_owned_entry(directory, relative_name, descriptor, kind)
                if kind == "directory":
                    walk(descriptor, relative)
            finally:
                os.close(descriptor)

    walk(directory.descriptor, None)


def _validate_owned_tree(directory: OwnedDirectory) -> bool:
    """Compare the live tree to every retained journal entry without mutation."""
    observed_paths: set[str] = set()

    def walk(parent_descriptor: int, prefix: PurePosixPath | None) -> bool:
        try:
            with os.scandir(parent_descriptor) as scanned:
                entries = sorted(scanned, key=lambda value: value.name)
        except OSError:
            return False
        for scanned_entry in entries:
            relative = (
                PurePosixPath(scanned_entry.name)
                if prefix is None
                else prefix / scanned_entry.name
            )
            relative_name = str(relative)
            expected = directory.entries.get(relative_name)
            if expected is None:
                return False
            try:
                live = scanned_entry.stat(follow_symlinks=False)
                retained = os.fstat(expected.descriptor)
            except OSError:
                return False
            expected_mode = (
                stat.S_ISDIR(live.st_mode)
                if expected.kind == "directory"
                else stat.S_ISREG(live.st_mode)
            )
            if (
                not expected_mode
                or (live.st_dev, live.st_ino) != (expected.device, expected.inode)
                or (retained.st_dev, retained.st_ino)
                != (expected.device, expected.inode)
                or retained.st_nlink <= 0
            ):
                return False
            observed_paths.add(relative_name)
            if expected.kind == "directory" and not walk(expected.descriptor, relative):
                return False
        return True

    return walk(directory.descriptor, None) and observed_paths == set(directory.entries)


def _ensure_owned_relative_parent(
    directory: OwnedDirectory, relative_path: str
) -> tuple[int, str]:
    relative = _validate_relative_file_name(relative_path)
    parent_descriptor = os.dup(directory.descriptor)
    prefix: PurePosixPath | None = None
    try:
        for component in relative.parts[:-1]:
            prefix = PurePosixPath(component) if prefix is None else prefix / component
            relative_parent = str(prefix)
            journaled = directory.entries.get(relative_parent)
            if journaled is None:
                os.mkdir(component, mode=0o700, dir_fd=parent_descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    _retain_owned_entry(directory, relative_parent, child, "directory")
                except BaseException:
                    os.close(child)
                    raise
            else:
                if journaled.kind != "directory":
                    raise StageError(
                        f"owned tree parent is not a directory: {relative_parent}"
                    )
                child = os.dup(journaled.descriptor)
            os.close(parent_descriptor)
            parent_descriptor = child
        return parent_descriptor, relative.parts[-1]
    except BaseException:
        os.close(parent_descriptor)
        raise


def _restore_quarantined_name(
    parent_descriptor: int, quarantine_name: str, original_name: str
) -> bool:
    try:
        _rename_entry_noreplace_at(parent_descriptor, quarantine_name, original_name)
    except (OSError, StageError):
        return False
    return True


def _move_name_to_quarantine(parent_descriptor: int, name: str) -> str:
    quarantine_name = f".exo-cleanup-{uuid.uuid4().hex}"
    _rename_entry_noreplace_at(parent_descriptor, name, quarantine_name)
    return quarantine_name


def _quarantine_matches_entry(
    parent_descriptor: int, quarantine_name: str, entry: OwnedTreeEntry
) -> bool:
    try:
        observed = os.stat(
            quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        retained = os.fstat(entry.descriptor)
    except OSError:
        return False
    return (observed.st_dev, observed.st_ino) == (entry.device, entry.inode) and (
        retained.st_dev,
        retained.st_ino,
    ) == (entry.device, entry.inode)


def _cleanup_owned_directory(directory: OwnedDirectory) -> bool:
    """Preserve a failed staging tree because Linux lacks conditional unlink."""
    original_path = directory.path
    root_quarantine: str | None = None
    try:
        root_quarantine = _move_name_to_quarantine(
            directory.parent_descriptor, original_path.name
        )
        root_entry = OwnedTreeEntry(
            relative_path=".",
            descriptor=directory.descriptor,
            device=directory.device,
            inode=directory.inode,
            kind="directory",
        )
        if not _quarantine_matches_entry(
            directory.parent_descriptor, root_quarantine, root_entry
        ):
            _restore_quarantined_name(
                directory.parent_descriptor, root_quarantine, original_path.name
            )
            root_quarantine = None
            return False
        directory.path = original_path.parent / root_quarantine
        if not _validate_owned_tree(directory):
            return False
        # unlinkat(2) and rmdir(2) accept only a parent/name pair, not an
        # expected inode. A same-UID actor can replace that name after any
        # userspace identity check. Automatic deletion would therefore be an
        # unverifiable claim: restore the known path and force a lease
        # tombstone for explicit, quiesced manual cleanup.
        return False
    except (OSError, StageError):
        return False
    finally:
        if root_quarantine is not None and _restore_quarantined_name(
            directory.parent_descriptor,
            root_quarantine,
            original_path.name,
        ):
            directory.path = original_path
        directory.close()


def _validate_receipt_bytes(raw_value: bytes, model: ModelSpec) -> None:
    receipt = _parse_json_object(raw_value, "Exo revision receipt")
    if set(receipt) != {"repo_id", "revision"}:
        raise StageError("Exo revision receipt has an invalid schema")
    if receipt["repo_id"] != model.model_id or receipt["revision"] != model.revision:
        raise StageError("Exo revision receipt does not match the configured model")


def _verify_snapshot_descriptor(
    descriptor: int,
    model: ModelSpec,
    checkpoint: Callable[[], None] = lambda: None,
) -> SnapshotVerification:
    files = _snapshot_regular_files(descriptor, checkpoint)
    file_set = set(files)
    if "config.json" not in file_set:
        raise StageError("snapshot is missing config.json")
    if _MODEL_RECEIPT not in file_set:
        raise StageError("snapshot is missing the exact Exo revision receipt")
    _validate_receipt_bytes(_read_relative_file(descriptor, _MODEL_RECEIPT), model)
    if _INDEX_FILENAME not in file_set:
        raise StageError(f"snapshot is missing {_INDEX_FILENAME}")
    index = _parse_json_object(
        _read_relative_file(descriptor, _INDEX_FILENAME), "safetensors index"
    )
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise StageError(
            "safetensors index must contain metadata and weight_map objects"
        )
    indexed_bytes = cast(dict[object, object], metadata).get("total_size")
    if type(indexed_bytes) is not int or indexed_bytes != model.expected_indexed_bytes:
        raise StageError(
            "safetensors index total_size does not match expected indexed bytes"
        )
    raw_weight_map = cast(dict[object, object], weight_map)
    if not raw_weight_map or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_weight_map.items()
    ):
        raise StageError("safetensors weight_map must contain string entries")
    shard_names = sorted({cast(str, value) for value in raw_weight_map.values()})
    for shard_name in shard_names:
        _validate_relative_file_name(shard_name)
        if shard_name not in file_set or not shard_name.endswith(".safetensors"):
            raise StageError(f"referenced safetensors shard is missing: {shard_name}")
    indexed_shards = set(shard_names)
    extra_shards = sorted(
        relative
        for relative in files
        if relative.endswith(".safetensors") and relative not in indexed_shards
    )
    if extra_shards:
        raise StageError(f"unindexed safetensors files are present: {extra_shards}")
    manifest = {
        relative: _sha256_file_at(descriptor, relative, checkpoint)
        for relative in sorted(files)
    }
    return SnapshotVerification(
        model_id=model.model_id,
        revision=model.revision,
        indexed_bytes=indexed_bytes,
        shard_count=len(shard_names),
        manifest=manifest,
    )


def _validate_hugging_face_download_metadata(
    descriptor: int,
    model: ModelSpec,
    checkpoint: Callable[[], None] = lambda: None,
) -> None:
    files = set(_snapshot_regular_files(descriptor, checkpoint))
    metadata_prefix = ".cache/huggingface/download/"
    model_files = sorted(
        relative
        for relative in files
        if not relative.startswith(".cache/huggingface/") and relative != _MODEL_RECEIPT
    )
    if not model_files:
        raise StageError("Hugging Face download produced no model files")
    metadata_files = sorted(
        relative
        for relative in files
        if relative.startswith(metadata_prefix) and relative.endswith(".metadata")
    )
    if not metadata_files:
        raise StageError("Hugging Face download omitted local-dir revision metadata")
    for metadata_path in metadata_files:
        checkpoint()
        lines = _read_relative_file(descriptor, metadata_path).splitlines()
        if not lines or lines[0].decode("ascii", errors="replace") != model.revision:
            raise StageError(
                f"Hugging Face metadata is not pinned to {model.revision}: {metadata_path}"
            )
    missing = [
        relative
        for relative in model_files
        if f"{metadata_prefix}{relative}.metadata" not in files
    ]
    if missing:
        raise StageError(
            f"Hugging Face download file lacks exact revision metadata: {missing[0]}"
        )


def verify_snapshot(
    path: Path,
    model: ModelSpec,
    checkpoint: Callable[[], None] = lambda: None,
) -> SnapshotVerification:
    """Verify one canonical, symlink-free, revision-bound model snapshot."""
    _validate_lexical_absolute_path(str(path), "snapshot path")
    try:
        directory = _open_owned_directory(path)
    except OSError as error:
        raise StageError(f"cannot inspect snapshot {path}: {error}") from error
    try:
        if not _owned_directory_path_matches(directory):
            raise StageError("snapshot root identity changed while opening")
        return _verify_snapshot_descriptor(directory.descriptor, model, checkpoint)
    finally:
        directory.close()


def build_hf_download_argv(
    hf_executable: Path, model: ModelSpec, temporary_path: Path
) -> tuple[str, ...]:
    """Return the only permitted Hugging Face acquisition command."""
    return (
        str(hf_executable),
        "download",
        model.model_id,
        "--revision",
        model.revision,
        "--local-dir",
        str(temporary_path),
    )


def _process_stat_fields(process_id: int) -> list[str]:
    try:
        raw_stat = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        close_parenthesis = raw_stat.rfind(")")
        fields_after_name = raw_stat[close_parenthesis + 2 :].split()
        if close_parenthesis < 0 or len(fields_after_name) < 20:
            raise ValueError("truncated process stat")
        return fields_after_name
    except (OSError, ValueError) as error:
        raise StageError(
            f"cannot establish process identity for PID {process_id}"
        ) from error


def _process_start_time_ticks(process_id: int) -> int:
    try:
        start_time_ticks = int(_process_stat_fields(process_id)[19])
    except (ValueError, IndexError) as error:
        raise StageError(
            f"cannot establish process identity for PID {process_id}"
        ) from error
    if start_time_ticks <= 0:
        raise StageError("process start time must be positive")
    return start_time_ticks


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owned_process_group_members(
    process_group_id: int, owner_token: str
) -> tuple[int, ...] | None:
    expected_entry = f"{_OWNER_TOKEN_ENVIRONMENT}={owner_token}".encode("utf-8")
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        try:
            observed_group = int(_process_stat_fields(process_id)[2])
        except (StageError, ValueError, IndexError):
            continue
        if observed_group != process_group_id:
            continue
        try:
            environment_entries = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            return None
        if expected_entry not in environment_entries:
            return None
        members.append(process_id)
    return tuple(sorted(members))


def _terminate_process_group(
    process: subprocess.Popen[bytes], timeout_seconds: float, owner_token: str
) -> bool:
    process_group_id = process.pid
    if _process_group_exists(process_group_id):
        members = _owned_process_group_members(process_group_id, owner_token)
        if not members:
            return False
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
        deadline = time.monotonic() + timeout_seconds
        while _process_group_exists(process_group_id) and time.monotonic() < deadline:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.05)
            time.sleep(0.01)
        if _process_group_exists(process_group_id):
            members = _owned_process_group_members(process_group_id, owner_token)
            if not members:
                return False
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=min(1.0, timeout_seconds))
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0.1)
    return not _process_group_exists(process_group_id) and process.poll() is not None


def _wait_for_process(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
    poll_seconds: float,
    latch: SignalLatch,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        latch.checkpoint()
        return_code = process.poll()
        if return_code is not None:
            return return_code
        if time.monotonic() >= deadline:
            raise StageError(f"owned process {process.pid} exceeded its timeout")
        time.sleep(min(poll_seconds, max(0.01, deadline - time.monotonic())))


def _rename_entry_noreplace_at(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    validated_source = _validate_directory_entry_name(source_name)
    validated_destination = _validate_directory_entry_name(destination_name)
    library = ctypes.CDLL(None, use_errno=True)
    symbol = getattr(library, "renameat2", None)
    if symbol is None:
        raise StageError("Linux renameat2 is required for no-replace model install")
    renameat2 = cast(Callable[[int, bytes, int, bytes, int], int], symbol)
    result = renameat2(
        parent_descriptor,
        os.fsencode(validated_source),
        parent_descriptor,
        os.fsencode(validated_destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise StageError(
                f"destination appeared during atomic install: {validated_destination}"
            )
        raise StageError(
            f"cannot atomically install {validated_destination}: "
            f"{os.strerror(error_number)}"
        )
    os.fsync(parent_descriptor)


def _rename_directory_noreplace_at(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    _rename_entry_noreplace_at(parent_descriptor, source_name, destination_name)


def owned_temporary_name(config: StageConfig, owner_token: str) -> str:
    digest = hashlib.sha256(owner_token.encode("utf-8")).hexdigest()[:16]
    return f".{config.model.directory_name}.{config.run_id}.{digest}.stage"


def _timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str):
        raise StageError(f"{description} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise StageError(f"{description} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StageError(f"{description} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _command_option(command: Sequence[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) > 1:
        raise StageError(f"active lease command repeats {option}")
    if not positions:
        return None
    position = positions[0]
    if position + 1 >= len(command):
        raise StageError(f"active lease command omits the value for {option}")
    return command[position + 1]


def _lock_is_held(lock_path: Path) -> bool:
    parent_descriptor = _open_directory_without_symlinks(lock_path.parent)
    try:
        descriptor = os.open(
            _validate_directory_entry_name(lock_path.name),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise StageError("benchmark lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return False
        finally:
            os.close(descriptor)
    except OSError as error:
        raise StageError(
            f"cannot inspect benchmark lock {lock_path}: {error}"
        ) from error
    finally:
        os.close(parent_descriptor)


def staging_metadata_contract(config: StageConfig) -> JsonObject:
    """Return the complete immutable staging config binding for lease metadata."""
    return cast(JsonObject, config.model_dump(mode="json", exclude_none=True))


def minimum_cleanup_grace_seconds(config: StageConfig) -> float:
    """Return a conservative wrapper grace bound for child-owned cleanup."""
    cooperative_checkpoint = max(config.timeouts.poll_seconds, 0.25)
    local_cleanup = config.timeouts.cleanup_seconds
    remote_cleanup = 3 * config.timeouts.cleanup_seconds
    return max(300.0, cooperative_checkpoint + local_cleanup + remote_cleanup + 30.0)


def _require_absolute_executable(path: Path, description: str) -> None:
    _validate_lexical_absolute_path(str(path), description)
    try:
        target_status = path.stat()
    except OSError as error:
        raise StageError(f"cannot inspect {description} {path}: {error}") from error
    if not stat.S_ISREG(target_status.st_mode) or not os.access(path, os.X_OK):
        raise StageError(f"{description} is not an executable regular file: {path}")


def _canonical_prospective_path(path: Path, description: str) -> Path:
    _validate_lexical_absolute_path(str(path), description)
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise StageError(f"cannot resolve {description} {path}: {error}") from error
    return resolved


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise StageError(f"cannot inspect path {path}: {error}") from error
    return True


def _require_outside_directory(path: Path, directory: Path, description: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError:
        return
    raise StageError(f"{description} must be outside the source deployment")


def _format_cli_number(value: float, description: str) -> str:
    if not math.isfinite(value) or value <= 0:
        raise StageError(f"{description} must be finite and positive")
    return str(value)


def _git_output(source_directory: Path, arguments: Sequence[str]) -> bytes:
    git_executable = Path("/usr/bin/git")
    _require_absolute_executable(git_executable, "git executable")
    try:
        completed = subprocess.run(
            (str(git_executable), "-C", str(source_directory), *arguments),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=30.0,
            env={"PATH": os.defpath, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StageError(f"cannot inspect source git identity: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-500:]
        raise StageError(
            f"git identity command failed with {completed.returncode}: {diagnostic}"
        )
    return completed.stdout


def _relative_entry_status(
    root_descriptor: int, relative_path: str
) -> os.stat_result | None:
    relative = _validate_relative_file_name(relative_path)
    directory_descriptor = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child
        return os.stat(
            relative.parts[-1],
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise StageError(
            f"cannot safely inspect dirty source path {relative_path}: {error}"
        ) from error
    finally:
        os.close(directory_descriptor)


def source_identity(
    source_directory: Path,
    checkpoint: Callable[[], None] = lambda: None,
) -> GitIdentity:
    """Hash every dirty source path below one retained, symlink-free root."""
    _require_canonical_directory(source_directory, "source deployment")
    try:
        commit = (
            _git_output(source_directory, ("rev-parse", "HEAD")).decode("ascii").strip()
        )
        tracked = _git_output(source_directory, ("diff", "--name-only", "-z", "HEAD"))
        untracked = _git_output(
            source_directory,
            ("ls-files", "--others", "--exclude-standard", "-z"),
        )
    except UnicodeDecodeError as error:
        raise StageError("source git identity is not ASCII/UTF-8") from error
    dirty_paths: set[str] = set()
    for raw_path in (*tracked.split(b"\0"), *untracked.split(b"\0")):
        if not raw_path:
            continue
        try:
            relative_path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise StageError("source contains a non-UTF-8 dirty path") from error
        _validate_relative_file_name(relative_path)
        dirty_paths.add(relative_path)

    deleted_digest = hashlib.sha256(b"<deleted>").hexdigest()
    root = _open_owned_directory(source_directory)
    try:
        dirty_hashes: dict[str, str] = {}
        for relative_path in sorted(dirty_paths):
            checkpoint()
            try:
                dirty_hashes[relative_path] = _sha256_file_at(
                    root.descriptor, relative_path, checkpoint
                )
            except StageError as error:
                if _relative_entry_status(root.descriptor, relative_path) is not None:
                    raise error
                dirty_hashes[relative_path] = deleted_digest
        if not _owned_directory_path_matches(root):
            raise StageError("source deployment identity changed while hashing")
    finally:
        root.close()
    return GitIdentity(
        commit=commit,
        dirty=bool(dirty_hashes),
        dirty_file_hashes=dirty_hashes,
    )


def _lease_static_metadata(
    config: StageConfig, command: Sequence[str], generated_at: str
) -> JsonObject:
    lease_inputs = cast(
        JsonObject,
        config.lease_metadata.model_dump(mode="json", exclude_none=True),
    )
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": list(config.lease_metadata.reserved_ports),
        "result_directory": config.result_directory,
        "command": list(command),
        "git": lease_inputs["git"],
        "hosts": [config.local_host_name, config.remote_host_name],
        "models": [
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "paths": {
                    config.local_host_name: config.local_destination,
                    config.remote_host_name: config.remote_destination,
                },
            }
        ],
        "gpu_bindings": lease_inputs["gpu_bindings"],
        "cpu_bindings": lease_inputs["cpu_bindings"],
        "hca_bindings": lease_inputs["hca_bindings"],
        "source_deployments": lease_inputs["source_deployments"],
        "owner_pids": lease_inputs["owner_pids"],
        "model_staging": staging_metadata_contract(config),
    }


def _validate_metadata_with_benchmark_lease(
    benchmark_lease_script: Path,
    metadata: JsonObject,
    generated_at: datetime,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_exo_model_stage_benchmark_lease_validation", benchmark_lease_script
    )
    if specification is None or specification.loader is None:
        raise StageError("cannot import the exact benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
        validator = cast(LeaseMetadataValidator, cast(object, module))
        validated = validator.validate_run_metadata(metadata, now=generated_at)
    except Exception as error:
        raise StageError(
            f"benchmark lease rejected generated metadata: {error}"
        ) from error
    if validated != metadata:
        raise StageError("benchmark lease metadata validation changed the document")


def prepare_lease_metadata(
    *,
    config_path: Path,
    metadata_output: Path,
    wrapper_python: Path,
    child_python: Path,
    benchmark_lease_script: Path,
    staging_script: Path,
    owner: str,
    purpose: str,
    expected_duration_seconds: float,
    heartbeat_seconds: float,
    cleanup_grace_seconds: float,
    lease_path: Path,
    lock_path: Path,
    result_root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    identity_reader: Callable[[Path], GitIdentity] = source_identity,
) -> LeasePreparation:
    """Create fresh, wrapper-validated metadata and its exact launch argv."""
    for path, description in (
        (config_path, "config path"),
        (metadata_output, "metadata output"),
        (wrapper_python, "wrapper Python"),
        (child_python, "child Python"),
        (benchmark_lease_script, "benchmark lease script"),
        (staging_script, "staging script"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
        (result_root, "result root"),
    ):
        _validate_lexical_absolute_path(str(path), description)
    _require_canonical_regular_file(config_path, "config path")
    _require_canonical_regular_file(benchmark_lease_script, "benchmark lease script")
    _require_canonical_regular_file(staging_script, "staging script")
    _require_absolute_executable(wrapper_python, "wrapper Python")
    _require_absolute_executable(child_python, "child Python")
    try:
        same_python = wrapper_python.samefile(child_python)
    except OSError as error:
        raise StageError("cannot compare wrapper and child Python") from error
    if not same_python:
        raise StageError("wrapper Python must match child Python")

    lease_path = _canonical_prospective_path(lease_path, "lease path")
    lock_path = _canonical_prospective_path(lock_path, "lock path")
    _require_canonical_directory(result_root, "result root")
    _require_canonical_directory(metadata_output.parent, "metadata output parent")
    if _path_exists_without_following(metadata_output):
        raise StageError(f"metadata output already exists: {metadata_output}")

    config = load_config(config_path)
    local_source = Path(
        config.lease_metadata.source_deployments[config.local_host_name].path
    )
    _require_canonical_directory(local_source, "local source deployment")
    expected_staging_script = local_source / "scripts" / Path(__file__).name
    expected_wrapper_script = local_source / "scripts" / "benchmark_lease.py"
    if (
        staging_script != expected_staging_script
        or staging_script != Path(__file__).resolve()
    ):
        raise StageError("staging script does not match the local source deployment")
    if benchmark_lease_script != expected_wrapper_script:
        raise StageError(
            "benchmark lease script does not match the local source deployment"
        )
    staging_script_bytes = _read_regular_file_path(
        staging_script, "model staging script"
    )
    if (
        hashlib.sha256(staging_script_bytes).hexdigest()
        != config.lease_metadata.staging_script_sha256
    ):
        raise StageError("staging script differs from its configured SHA-256")

    result_directory = Path(config.result_directory)
    if result_directory != result_root / config.run_id:
        raise StageError("config result_directory must equal result_root/run_id")
    generated_paths = (metadata_output, result_directory, lease_path, lock_path)
    if len(set(generated_paths)) != len(generated_paths):
        raise StageError(
            "metadata output, result directory, lease path, and lock path must "
            "be pairwise distinct"
        )
    if _path_exists_without_following(result_directory):
        raise StageError(f"result directory already exists: {result_directory}")
    for generated_path, description in (
        (metadata_output, "metadata output"),
        (result_directory, "result directory"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
    ):
        _require_outside_directory(generated_path, local_source, description)

    if not owner.strip() or "\0" in owner:
        raise StageError("owner must be nonempty and must not contain NUL")
    if not purpose.strip() or "\0" in purpose:
        raise StageError("purpose must be nonempty and must not contain NUL")
    expected_duration = _format_cli_number(
        expected_duration_seconds, "expected duration"
    )
    heartbeat = _format_cli_number(heartbeat_seconds, "heartbeat interval")
    cleanup_grace = _format_cli_number(cleanup_grace_seconds, "cleanup grace")
    minimum_grace = minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise StageError(
            "cleanup grace is shorter than the staging cleanup bound "
            f"({minimum_grace} seconds)"
        )

    observed_identity = identity_reader(local_source)
    if observed_identity != config.lease_metadata.git:
        raise StageError("strict config source identity is stale")
    child_argv = (
        str(child_python),
        str(staging_script),
        "--config",
        str(config_path),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    )
    generated_at_value = now()
    if generated_at_value.tzinfo is None or generated_at_value.utcoffset() is None:
        raise StageError("metadata clock must return an offset-aware timestamp")
    generated_at_utc = generated_at_value.astimezone(timezone.utc)
    generated_at = generated_at_utc.isoformat(timespec="seconds")
    metadata = _lease_static_metadata(config, child_argv, generated_at)
    _validate_metadata_with_benchmark_lease(
        benchmark_lease_script, metadata, generated_at_utc
    )
    benchmark_lease_argv = (
        str(wrapper_python),
        str(benchmark_lease_script),
        f"--owner={owner}",
        f"--purpose={purpose}",
        "--run-id",
        config.run_id,
        "--namespace",
        config.namespace,
        "--port",
        ",".join(str(port) for port in config.lease_metadata.reserved_ports),
        "--metadata-json",
        str(metadata_output),
        "--heartbeat-seconds",
        heartbeat,
        "--expected-duration-seconds",
        expected_duration,
        "--cleanup-grace-seconds",
        cleanup_grace,
        "--lock-path",
        str(lock_path),
        "--lease-path",
        str(lease_path),
        "--result-root",
        str(result_root),
        "--",
        *child_argv,
    )
    if identity_reader(local_source) != observed_identity:
        raise StageError("source identity changed while preparing lease metadata")
    if (
        _read_regular_file_path(staging_script, "model staging script")
        != staging_script_bytes
    ):
        raise StageError("staging script changed while preparing lease metadata")
    _atomic_create_json(metadata_output, metadata)
    return LeasePreparation(
        metadata=metadata,
        child_argv=child_argv,
        benchmark_lease_argv=benchmark_lease_argv,
        generated_at=generated_at,
        minimum_cleanup_grace_seconds=minimum_grace,
    )


def validate_active_lease(
    config: StageConfig,
    *,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int | None = None,
    parent_process_id: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    lock_is_held: Callable[[Path], bool] = _lock_is_held,
) -> JsonObject:
    """Bind this exact staging child and static model contract to the lease."""
    for path, description in (
        (config_path, "config path"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
    ):
        if not path.is_absolute():
            raise StageError(f"{description} must be absolute")
    if not lock_is_held(lock_path):
        raise StageError("benchmark coordination lock is not held")
    expected_process_id = os.getpid() if process_id is None else process_id
    expected_parent_id = (
        os.getppid() if parent_process_id is None else parent_process_id
    )
    deadline = monotonic() + config.timeouts.lease_bind_seconds
    while True:
        record = _read_json_object_path(lease_path, "active lease record")
        child_pid = record.get("child_pid")
        if child_pid == expected_process_id:
            break
        if child_pid is not None:
            raise StageError("active lease belongs to a different child process")
        if monotonic() >= deadline:
            raise StageError("active lease did not publish this child PID")
        sleep(0.05)

    raw_command = record.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or not all(isinstance(argument, str) for argument in raw_command)
    ):
        raise StageError("active lease command must be a nonempty string array")
    command = tuple(cast(list[str], raw_command))
    script_path = Path(__file__).resolve()
    if not any(Path(argument).resolve() == script_path for argument in command):
        raise StageError("active lease command does not name this exact staging script")
    if _command_option(command, "--config") != str(config_path):
        raise StageError("active lease command uses a different config path")
    command_lease = _command_option(command, "--lease-path")
    if command_lease is None:
        if lease_path != _DEFAULT_LEASE_PATH:
            raise StageError("nonstandard lease path must be explicit in the command")
    elif command_lease != str(lease_path):
        raise StageError("active lease command uses a different lease path")
    command_lock = _command_option(command, "--lock-path")
    if command_lock is None:
        if lock_path != _DEFAULT_LOCK_PATH:
            raise StageError("nonstandard lock path must be explicit in the command")
    elif command_lock != str(lock_path):
        raise StageError("active lease command uses a different lock path")
    if _command_option(command, "--result-dir") != config.result_directory:
        raise StageError("active lease command uses a different result directory")

    expected_record_values: tuple[tuple[str, object], ...] = (
        ("run_id", config.run_id),
        ("exo_namespace", config.namespace),
        ("result_directory", config.result_directory),
        ("wrapper_pid", expected_parent_id),
        ("child_cleanup_confirmation_required", True),
    )
    for name, expected in expected_record_values:
        if record.get(name) != expected:
            raise StageError(f"active lease {name} does not match the staging child")
    cleanup_grace = record.get("cleanup_grace_seconds")
    minimum_cleanup_grace = minimum_cleanup_grace_seconds(config)
    if (
        not isinstance(cleanup_grace, int | float)
        or isinstance(cleanup_grace, bool)
        or not math.isfinite(cleanup_grace)
        or cleanup_grace < minimum_cleanup_grace
    ):
        raise StageError("active lease cleanup grace is shorter than the staging bound")
    current_time_value = now()
    if current_time_value.tzinfo is None or current_time_value.utcoffset() is None:
        raise StageError("lease validation clock must be offset-aware")
    current_time = current_time_value.astimezone(timezone.utc)
    heartbeat = _timestamp(record.get("heartbeat"), "lease heartbeat")
    if (
        heartbeat < current_time - _LEASE_HEARTBEAT_MAX_AGE
        or heartbeat > current_time + timedelta(minutes=1)
    ):
        raise StageError("active lease heartbeat is stale or in the future")
    metadata_value = record.get("metadata")
    if not isinstance(metadata_value, dict):
        raise StageError("active lease metadata must be an object")
    metadata = cast(JsonObject, metadata_value)
    raw_ports = record.get("ports")
    if (
        not isinstance(raw_ports, list)
        or not raw_ports
        or not all(type(port) is int and 1 <= port <= 65535 for port in raw_ports)
        or raw_ports != list(config.lease_metadata.reserved_ports)
        or metadata.get("reserved_ports") != raw_ports
    ):
        raise StageError("active lease ports differ from lease metadata")
    generated_at = _timestamp(metadata.get("generated_at"), "metadata generated_at")
    if (
        generated_at < current_time - _LEASE_METADATA_MAX_AGE
        or generated_at > current_time + timedelta(minutes=1)
    ):
        raise StageError("active lease metadata is stale or in the future")
    lease_inputs = config.lease_metadata.model_dump(mode="json", exclude_none=True)
    expected_metadata: tuple[tuple[str, object], ...] = (
        ("run_id", config.run_id),
        ("namespace", config.namespace),
        ("result_directory", config.result_directory),
        ("command", list(command)),
        ("hosts", [config.local_host_name, config.remote_host_name]),
        ("git", lease_inputs["git"]),
        ("gpu_bindings", lease_inputs["gpu_bindings"]),
        ("cpu_bindings", lease_inputs["cpu_bindings"]),
        ("hca_bindings", lease_inputs["hca_bindings"]),
        ("source_deployments", lease_inputs["source_deployments"]),
        ("owner_pids", lease_inputs["owner_pids"]),
        ("model_staging", staging_metadata_contract(config)),
    )
    for name, expected in expected_metadata:
        if metadata.get(name) != expected:
            raise StageError(
                f"active lease metadata.{name} differs from staging config"
            )
    expected_models = [
        {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "paths": {
                config.local_host_name: config.local_destination,
                config.remote_host_name: config.remote_destination,
            },
        }
    ]
    if metadata.get("models") != expected_models:
        raise StageError("active lease model binding differs from staging config")
    return record


def _validate_local_source_identity(
    config: StageConfig,
    checkpoint: Callable[[], None] = lambda: None,
) -> None:
    source_path = Path(
        config.lease_metadata.source_deployments[config.local_host_name].path
    )
    if source_identity(source_path, checkpoint) != config.lease_metadata.git:
        raise StageError("local source deployment differs from the active lease")


class SystemEffects:
    def __init__(
        self,
        config: StageConfig,
        result_directory_descriptor: int,
        *,
        observed_local_host_name: str | None = None,
    ) -> None:
        self._config = config
        self._result_directory = Path(config.result_directory)
        self._result_directory_descriptor = result_directory_descriptor
        self._owned_local_temporaries: dict[Path, OwnedDirectory] = {}
        self._installed_local_directories: dict[Path, OwnedDirectory] = {}
        script_path = Path(__file__).resolve()
        self._script_bytes = _read_regular_file_path(
            script_path, "model staging script"
        )
        if (
            hashlib.sha256(self._script_bytes).hexdigest()
            != config.lease_metadata.staging_script_sha256
        ):
            raise StageError("model staging script differs from its config digest")
        _validate_result_directory_descriptor(
            self._result_directory, self._result_directory_descriptor
        )
        observed_host = observed_local_host_name or socket.gethostname()
        if observed_host != config.local_host_name:
            raise StageError(
                f"local host identity {observed_host!r} does not match "
                f"{config.local_host_name!r}"
            )

    def write_result_json(self, filename: str, value: Mapping[str, object]) -> None:
        if filename not in {_RUNTIME_METADATA, _BENCHMARK_RESULT}:
            raise StageError(f"unsupported result fragment {filename!r}")
        _atomic_write_json_at(self._result_directory_descriptor, filename, value)

    def inspect_local_destination(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification | None:
        parent_descriptor = _open_directory_without_symlinks(path.parent)
        try:
            try:
                path_status = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISDIR(
                path_status.st_mode
            ):
                raise StageError("local destination exists but is not a real directory")
            directory_descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            try:
                if not _snapshot_regular_files(directory_descriptor, latch.checkpoint):
                    raise StageError(
                        "refusing to replace a pre-existing empty destination"
                    )
                return _verify_snapshot_descriptor(
                    directory_descriptor, model, latch.checkpoint
                )
            finally:
                os.close(directory_descriptor)
        finally:
            os.close(parent_descriptor)

    def create_local_temporary(self, destination: Path, owned_name: str) -> Path:
        temporary = destination.parent / owned_name
        if temporary.parent != destination.parent:
            raise StageError("owned temporary path escaped the destination parent")
        directory = _create_owned_directory(temporary)
        try:
            self._owned_local_temporaries[temporary] = directory
        except BaseException as error:
            cleanup_confirmed = _cleanup_owned_directory(directory)
            raise OperationError(
                "cannot record owned local temporary",
                cleanup_confirmed=cleanup_confirmed,
            ) from error
        return temporary

    def _local_temporary_identity_matches(self, path: Path) -> bool:
        directory = self._owned_local_temporaries.get(path)
        return directory is not None and _owned_directory_path_matches(directory)

    def copy_preverified_snapshot(
        self,
        source: Path,
        destination: Path,
        model: ModelSpec,
        latch: SignalLatch,
    ) -> None:
        source_directory = _open_owned_directory(source)
        try:
            destination_directory = self._owned_local_temporaries.get(destination)
            if destination_directory is None or not _owned_directory_path_matches(
                destination_directory
            ):
                raise StageError("copy destination is not the retained owned temporary")
            source_verification = _verify_snapshot_descriptor(
                source_directory.descriptor, model, latch.checkpoint
            )
            for relative_path in sorted(source_verification.manifest):
                if relative_path == _MODEL_RECEIPT:
                    continue
                latch.checkpoint()
                source_descriptor = _open_relative_regular_file(
                    source_directory.descriptor, relative_path
                )
                parent_descriptor, filename = _ensure_owned_relative_parent(
                    destination_directory, relative_path
                )
                descriptor: int | None = None
                try:
                    source_status = os.fstat(source_descriptor)
                    descriptor = os.open(
                        filename,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        source_status.st_mode & 0o777,
                        dir_fd=parent_descriptor,
                    )
                    _retain_owned_entry(
                        destination_directory,
                        relative_path,
                        descriptor,
                        "regular",
                    )
                    with (
                        os.fdopen(source_descriptor, "rb", closefd=False) as input_file,
                        os.fdopen(descriptor, "wb", closefd=False) as output_file,
                    ):
                        while chunk := input_file.read(1024 * 1024):
                            latch.checkpoint()
                            output_file.write(chunk)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
                    os.close(source_descriptor)
                    os.close(parent_descriptor)
            if (
                _verify_snapshot_descriptor(
                    source_directory.descriptor, model, latch.checkpoint
                )
                != source_verification
            ):
                raise StageError("source snapshot changed while it was copied")
        finally:
            source_directory.close()

    def _local_owned_process(
        self,
        process: subprocess.Popen[bytes],
        owner_token: str,
        log_path: Path,
    ) -> OwnedProcess:
        process_group_id = os.getpgid(process.pid)
        if process_group_id != process.pid:
            raise StageError(
                "local download process is not its own process-group leader"
            )
        return OwnedProcess(
            host_name=self._config.local_host_name,
            pid=process.pid,
            process_group_id=process_group_id,
            start_time_ticks=_process_start_time_ticks(process.pid),
            owner_token=owner_token,
            namespace=self._config.namespace,
            transport_pid=process.pid,
            log_path=str(log_path),
        )

    def download_snapshot(
        self,
        config: StageConfig,
        temporary_path: Path,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> bool:
        executable = Path(config.acquisition.hf_executable)
        _require_canonical_regular_file(executable, "hf executable", executable=True)
        argv = build_hf_download_argv(executable, config.model, temporary_path)
        log_path = self._result_directory / "model-stage-local-download.log"
        log_descriptor = os.open(
            log_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._result_directory_descriptor,
        )
        process: subprocess.Popen[bytes] | None = None
        caught_error: BaseException | None = None
        try:
            with os.fdopen(log_descriptor, "wb", closefd=True) as log_file:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env={
                        **config.acquisition.environment,
                        _OWNER_TOKEN_ENVIRONMENT: owner_token,
                        _NAMESPACE_ENVIRONMENT: config.namespace,
                    },
                    start_new_session=True,
                    shell=False,
                )
                register_process(
                    self._local_owned_process(process, owner_token, log_path)
                )
                return_code = _wait_for_process(
                    process,
                    config.timeouts.local_stage_seconds,
                    config.timeouts.poll_seconds,
                    latch,
                )
                if return_code != 0:
                    raise StageError(f"hf download exited with status {return_code}")
        except BaseException as error:
            caught_error = error
        assert process is not None or caught_error is not None
        cleanup_confirmed = (
            True
            if process is None
            else _terminate_process_group(
                process, config.timeouts.cleanup_seconds, owner_token
            )
        )
        temporary = self._owned_local_temporaries.get(temporary_path)
        if cleanup_confirmed and temporary is not None:
            try:
                _journal_existing_owned_tree(temporary, latch.checkpoint)
            except BaseException as journal_error:
                if caught_error is None:
                    caught_error = journal_error
        if caught_error is not None:
            raise OperationError(
                f"local acquisition failed: {type(caught_error).__name__}: {caught_error}",
                cleanup_confirmed=cleanup_confirmed,
            ) from caught_error
        if not cleanup_confirmed:
            raise OperationError(
                "local acquisition left an owned process group",
                cleanup_confirmed=False,
            )
        if temporary is None or not _owned_directory_path_matches(temporary):
            raise OperationError(
                "local acquisition temporary identity changed",
                cleanup_confirmed=True,
            )
        _validate_hugging_face_download_metadata(
            temporary.descriptor, config.model, latch.checkpoint
        )
        return True

    def write_revision_receipt(self, path: Path, model: ModelSpec) -> None:
        directory = self._owned_local_temporaries.get(path)
        if directory is None or not _owned_directory_path_matches(directory):
            raise StageError("revision receipt destination is not an owned temporary")
        _create_owned_json_at(
            directory,
            _MODEL_RECEIPT,
            {"repo_id": model.model_id, "revision": model.revision},
        )

    def verify_local_snapshot(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification:
        directory = self._owned_local_temporaries.get(path)
        installed = False
        if directory is None:
            directory = self._installed_local_directories.get(path)
            installed = directory is not None
        if directory is not None:
            try:
                if not _owned_directory_path_matches(directory):
                    raise StageError("owned model root changed before verification")
                if not _validate_owned_tree(directory):
                    raise StageError(
                        "owned model tree differs from its creation journal"
                    )
                verification = _verify_snapshot_descriptor(
                    directory.descriptor, model, latch.checkpoint
                )
                if not _validate_owned_tree(directory):
                    raise StageError("owned model tree changed during verification")
                return verification
            finally:
                if installed:
                    self._installed_local_directories.pop(path, None)
                    directory.close()
        return verify_snapshot(path, model, latch.checkpoint)

    def install_local_snapshot(self, temporary_path: Path, destination: Path) -> None:
        directory = self._owned_local_temporaries.get(temporary_path)
        if directory is None or not _owned_directory_path_matches(directory):
            raise StageError("refusing to install an unowned local temporary path")
        if not _validate_owned_tree(directory):
            raise StageError("refusing to install a model tree outside its journal")
        _rename_directory_noreplace_at(
            directory.parent_descriptor, temporary_path.name, destination.name
        )
        del self._owned_local_temporaries[temporary_path]
        directory.path = destination
        try:
            observed = os.stat(
                destination.name,
                dir_fd=directory.parent_descriptor,
                follow_symlinks=False,
            )
            if (observed.st_dev, observed.st_ino) != (
                directory.device,
                directory.inode,
            ):
                raise StageError(
                    "installed local destination identity differs from owned temp"
                )
            if not _validate_owned_tree(directory):
                raise StageError("installed local model tree changed during rename")
            self._installed_local_directories[destination] = directory
        except BaseException:
            directory.close()
            raise

    def cleanup_local_temporary(self, path: Path) -> bool:
        directory = self._owned_local_temporaries.pop(path, None)
        if directory is None:
            return False
        return _cleanup_owned_directory(directory)

    def build_remote_command(self, config: StageConfig) -> tuple[str, ...]:
        ssh_executable = Path(config.ssh.executable)
        _require_canonical_regular_file(
            ssh_executable, "SSH executable", executable=True
        )
        remote_command = shlex.join(
            (config.ssh.remote_python_executable, "-c", _REMOTE_LOADER)
        )
        return (
            str(ssh_executable),
            *config.ssh.options,
            "--",
            config.ssh.target,
            remote_command,
        )

    def _script_payload(self) -> bytes:
        return base64.b64encode(self._script_bytes) + b"\n"

    def _remote_owned_process(
        self,
        identity: RemoteIdentity,
        transport_pid: int,
        log_path: Path,
        config: StageConfig,
        owner_token: str,
    ) -> OwnedProcess:
        if (
            identity.owner_token != owner_token
            or identity.namespace != config.namespace
            or identity.host_name != config.remote_host_name
        ):
            raise StageError("remote process identity is bound to a different owner")
        if identity.process_group_id != identity.pid:
            raise StageError("remote helper is not its own process-group leader")
        return OwnedProcess(
            host_name=config.remote_host_name,
            pid=identity.pid,
            process_group_id=identity.process_group_id,
            start_time_ticks=identity.start_time_ticks,
            owner_token=owner_token,
            namespace=config.namespace,
            transport_pid=transport_pid,
            log_path=str(log_path),
        )

    def _read_protocol_line(
        self,
        process: subprocess.Popen[bytes],
        timeout_seconds: float,
        poll_seconds: float,
        latch: SignalLatch,
    ) -> bytes:
        assert process.stdout is not None
        deadline = time.monotonic() + timeout_seconds
        line = bytearray()
        while True:
            latch.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StageError("remote helper receipt timed out")
            ready, _, _ = select.select(
                [process.stdout.fileno()],
                [],
                [],
                min(poll_seconds, 0.25, remaining),
            )
            if ready:
                chunk = os.read(process.stdout.fileno(), 1)
                if not chunk:
                    raise StageError("remote helper closed stdout before its receipt")
                line.extend(chunk)
                if len(line) > _MAX_REMOTE_IDENTITY_BYTES:
                    raise StageError("remote helper identity receipt is too large")
                if chunk == b"\n":
                    return bytes(line)
            if process.poll() is not None:
                raise StageError("remote helper exited before its receipt")

    def _write_remote_header(
        self,
        output: IO[bytes],
        payload: bytes,
        errors: list[BaseException],
    ) -> None:
        try:
            output.write(payload)
            output.flush()
        except BaseException as error:
            errors.append(error)

    def _write_tar_stream(
        self,
        output: IO[bytes],
        root: Path,
        verification: SnapshotVerification,
        latch: SignalLatch,
        errors: list[BaseException],
    ) -> None:
        directory: OwnedDirectory | None = None
        try:
            directory = _open_owned_directory(root)
            if not _owned_directory_path_matches(directory):
                raise StageError("local transfer snapshot path identity changed")
            if (
                _verify_snapshot_descriptor(
                    directory.descriptor,
                    self._config.model,
                    latch.checkpoint,
                )
                != verification
            ):
                raise StageError("local transfer snapshot changed before streaming")
            with tarfile.open(fileobj=output, mode="w|") as archive:
                for relative_path in sorted(verification.manifest):
                    if relative_path == _MODEL_RECEIPT:
                        continue
                    latch.checkpoint()
                    source_descriptor = _open_relative_regular_file(
                        directory.descriptor, relative_path
                    )
                    source_status = os.fstat(source_descriptor)
                    information = tarfile.TarInfo(relative_path)
                    information.size = source_status.st_size
                    information.mode = source_status.st_mode & 0o777
                    information.mtime = 0
                    information.uid = 0
                    information.gid = 0
                    information.uname = ""
                    information.gname = ""
                    with os.fdopen(source_descriptor, "rb", closefd=True) as source:
                        archive.addfile(information, source)
            if (
                _verify_snapshot_descriptor(
                    directory.descriptor,
                    self._config.model,
                    latch.checkpoint,
                )
                != verification
            ):
                raise StageError("local transfer snapshot changed while streaming")
            output.close()
        except BaseException as error:
            errors.append(error)
            with contextlib.suppress(OSError):
                output.close()
        finally:
            if directory is not None:
                directory.close()

    def _read_remote_response(
        self,
        input_file: IO[bytes],
        chunks: list[bytes],
        errors: list[BaseException],
    ) -> None:
        try:
            total = 0
            exceeded_limit = False
            while chunk := input_file.read(64 * 1024):
                total += len(chunk)
                if total <= _MAX_REMOTE_RESPONSE_BYTES:
                    chunks.append(chunk)
                else:
                    exceeded_limit = True
            if exceeded_limit:
                errors.append(StageError("remote response exceeds the size limit"))
        except BaseException as error:
            errors.append(error)

    def _parse_remote_response(self, chunks: Sequence[bytes]) -> RemoteResponse:
        raw_response = b"".join(chunks)
        if not raw_response:
            raise StageError("remote helper omitted its final receipt")
        if len(raw_response) > _MAX_REMOTE_RESPONSE_BYTES:
            raise StageError("remote response exceeds the size limit")
        return RemoteResponse.model_validate_json(raw_response)

    def _run_remote(
        self,
        config: StageConfig,
        request: RemoteRequest,
        timeout_seconds: float,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
        local_snapshot: Path | None = None,
        local_verification: SnapshotVerification | None = None,
    ) -> ProcessOperationResult:
        remote_command = self.build_remote_command(config)
        header_payload = (
            self._script_payload() + request.model_dump_json().encode("utf-8") + b"\n"
        )
        log_suffix = "probe" if request.operation == "probe" else "transfer"
        log_path = self._result_directory / f"model-stage-remote-{log_suffix}.log"
        log_descriptor = os.open(
            log_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._result_directory_descriptor,
        )
        process: subprocess.Popen[bytes] | None = None
        header_writer: threading.Thread | None = None
        writer: threading.Thread | None = None
        reader: threading.Thread | None = None
        header_errors: list[BaseException] = []
        writer_errors: list[BaseException] = []
        reader_errors: list[BaseException] = []
        response_chunks: list[bytes] = []
        caught_error: BaseException | None = None
        response: RemoteResponse | None = None
        try:
            with os.fdopen(log_descriptor, "wb", closefd=True) as log_file:
                process = subprocess.Popen(
                    remote_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env={
                        **config.acquisition.environment,
                        _OWNER_TOKEN_ENVIRONMENT: request.owner_token,
                        _NAMESPACE_ENVIRONMENT: config.namespace,
                    },
                    start_new_session=True,
                    shell=False,
                )
                assert process.stdin is not None
                header_writer = threading.Thread(
                    target=self._write_remote_header,
                    args=(process.stdin, header_payload, header_errors),
                    name="model-stage-header-writer",
                    daemon=True,
                )
                header_writer.start()
                first_receipt = self._read_protocol_line(
                    process,
                    min(timeout_seconds, config.timeouts.remote_probe_seconds),
                    config.timeouts.poll_seconds,
                    latch,
                )
                header_writer.join(timeout=config.timeouts.cleanup_seconds)
                if header_writer.is_alive():
                    raise StageError("remote request header writer did not finish")
                if header_errors:
                    raise StageError(
                        f"remote request header writer failed: {header_errors[0]}"
                    )
                try:
                    identity = RemoteIdentity.model_validate_json(first_receipt)
                except ValidationError as identity_error:
                    try:
                        response = RemoteResponse.model_validate_json(first_receipt)
                    except ValidationError:
                        raise identity_error from None
                    raise StageError(
                        response.error or "remote helper failed before identity"
                    ) from identity_error
                assert process.stdout is not None
                reader = threading.Thread(
                    target=self._read_remote_response,
                    args=(process.stdout, response_chunks, reader_errors),
                    name="model-stage-response-reader",
                    daemon=True,
                )
                reader.start()
                register_process(
                    self._remote_owned_process(
                        identity, process.pid, log_path, config, request.owner_token
                    )
                )
                if request.operation == "receive":
                    if local_snapshot is None or local_verification is None:
                        raise StageError("remote receive omitted the local snapshot")
                    writer = threading.Thread(
                        target=self._write_tar_stream,
                        args=(
                            process.stdin,
                            local_snapshot,
                            local_verification,
                            latch,
                            writer_errors,
                        ),
                        name="model-stage-tar-writer",
                        daemon=True,
                    )
                    writer.start()
                else:
                    process.stdin.close()
                return_code = _wait_for_process(
                    process, timeout_seconds, config.timeouts.poll_seconds, latch
                )
                if writer is not None:
                    writer.join(timeout=config.timeouts.cleanup_seconds)
                    if writer.is_alive():
                        raise StageError("local tar writer did not finish")
                    if writer_errors:
                        raise StageError(f"local tar writer failed: {writer_errors[0]}")
                reader.join(timeout=config.timeouts.cleanup_seconds)
                if reader.is_alive():
                    raise StageError("remote response reader did not finish")
                if reader_errors:
                    raise StageError(
                        f"remote response reader failed: {reader_errors[0]}"
                    )
                response = self._parse_remote_response(response_chunks)
                if return_code not in {0, 1}:
                    raise StageError(
                        f"foreground SSH helper exited with status {return_code}"
                    )
                if response.status != "completed":
                    raise StageError(response.error or "remote helper failed")
                if return_code != 0:
                    raise StageError(
                        "remote helper reported success with a failing exit status"
                    )
        except BaseException as error:
            caught_error = error
        cleanup_confirmed = True
        if process is not None:
            writer_may_hold_stdin = (
                header_writer is not None and header_writer.is_alive()
            ) or (writer is not None and writer.is_alive())
            if not writer_may_hold_stdin:
                with contextlib.suppress(OSError, ValueError):
                    if process.stdin is not None:
                        process.stdin.close()
            cleanup_confirmed = _terminate_process_group(
                process, config.timeouts.cleanup_seconds, request.owner_token
            )
        if header_writer is not None and header_writer.is_alive():
            header_writer.join(timeout=config.timeouts.cleanup_seconds)
            cleanup_confirmed = cleanup_confirmed and not header_writer.is_alive()
        if writer is not None and writer.is_alive():
            writer.join(timeout=config.timeouts.cleanup_seconds)
            cleanup_confirmed = cleanup_confirmed and not writer.is_alive()
        if reader is not None and reader.is_alive():
            reader.join(timeout=config.timeouts.cleanup_seconds)
            cleanup_confirmed = cleanup_confirmed and not reader.is_alive()
        if process is not None:
            with contextlib.suppress(OSError, ValueError):
                if process.stdin is not None:
                    process.stdin.close()
        if response is None and reader is not None and not reader.is_alive():
            try:
                if not reader_errors:
                    response = self._parse_remote_response(response_chunks)
            except BaseException as error:
                if caught_error is None:
                    caught_error = error
        if process is not None:
            # Reaping local SSH does not prove that its remote helper exited.
            cleanup_confirmed = reconcile_remote_cleanup(cleanup_confirmed, response)
        if caught_error is not None:
            raise OperationError(
                f"remote {request.operation} failed: {type(caught_error).__name__}: {caught_error}",
                cleanup_confirmed=cleanup_confirmed,
            ) from caught_error
        assert response is not None
        if not cleanup_confirmed:
            raise OperationError(
                f"remote {request.operation} cleanup was not confirmed",
                cleanup_confirmed=False,
            )
        return ProcessOperationResult(
            verification=response.verification,
            cleanup_confirmed=True,
            installed=response.installed,
        )

    def _remote_request(
        self,
        config: StageConfig,
        owner_token: str,
        operation: Literal["probe", "receive"],
    ) -> RemoteRequest:
        temporary = Path(config.remote_destination).parent / owned_temporary_name(
            config, owner_token
        )
        return RemoteRequest(
            schema_version=1,
            operation=operation,
            run_id=config.run_id,
            namespace=config.namespace,
            owner_token=owner_token,
            expected_host_name=config.remote_host_name,
            destination=config.remote_destination,
            temporary_path=str(temporary),
            model=config.model,
        )

    def probe_remote_destination(
        self,
        config: StageConfig,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult:
        return self._run_remote(
            config,
            self._remote_request(config, owner_token, "probe"),
            config.timeouts.remote_probe_seconds,
            register_process,
            latch,
        )

    def transfer_remote_snapshot(
        self,
        config: StageConfig,
        local_snapshot: Path,
        local_verification: SnapshotVerification,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult:
        return self._run_remote(
            config,
            self._remote_request(config, owner_token, "receive"),
            config.timeouts.remote_transfer_seconds,
            register_process,
            latch,
            local_snapshot,
            local_verification,
        )


def _runtime_fragment(
    config: StageConfig, owner_token: str, processes: Sequence[OwnedProcess]
) -> JsonObject:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "owner_token": owner_token,
        "owned_processes": [process.model_dump(mode="json") for process in processes],
    }


def run_staging(
    config: StageConfig,
    effects: StagingEffects,
    latch: SignalLatch | None = None,
    owner_token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> JsonObject:
    """Stage both snapshots and always reconcile ownership in the result fragment."""
    signal_latch = latch or SignalLatch()
    owner_token = f"{config.run_id}:{owner_token_factory()}"
    processes: list[OwnedProcess] = []
    process_identities: set[tuple[str, int, int]] = set()
    cleanup_confirmations: list[bool] = []
    local_temporary: Path | None = None
    local_verification: SnapshotVerification | None = None
    remote_verification: SnapshotVerification | None = None
    local_installed = False
    remote_installed = False
    acquisition = "existing"
    caught_error: BaseException | None = None

    def publish_runtime() -> None:
        effects.write_result_json(
            _RUNTIME_METADATA,
            cast(
                Mapping[str, object], _runtime_fragment(config, owner_token, processes)
            ),
        )

    def register_process(process: OwnedProcess) -> None:
        expected_hosts = {config.local_host_name, config.remote_host_name}
        if process.host_name not in expected_hosts:
            raise StageError("owned process belongs to an unconfigured host")
        if process.owner_token != owner_token or process.namespace != config.namespace:
            raise StageError("owned process binding differs from the staging run")
        identity = (process.host_name, process.pid, process.start_time_ticks)
        if identity in process_identities:
            raise StageError("owned process identity was repeated")
        process_identities.add(identity)
        processes.append(process)
        publish_runtime()

    publish_runtime()
    try:
        signal_latch.checkpoint()
        local_destination = Path(config.local_destination)
        local_verification = effects.inspect_local_destination(
            local_destination, config.model, signal_latch
        )
        if local_verification is None:
            local_temporary = effects.create_local_temporary(
                local_destination, owned_temporary_name(config, owner_token)
            )
            if config.acquisition.source_snapshot is not None:
                acquisition = "preverified_snapshot_copy"
                effects.copy_preverified_snapshot(
                    Path(config.acquisition.source_snapshot),
                    local_temporary,
                    config.model,
                    signal_latch,
                )
            else:
                acquisition = "huggingface_cli"
                cleanup_confirmations.append(
                    effects.download_snapshot(
                        config,
                        local_temporary,
                        owner_token,
                        register_process,
                        signal_latch,
                    )
                )
            effects.write_revision_receipt(local_temporary, config.model)
            local_verification = effects.verify_local_snapshot(
                local_temporary, config.model, signal_latch
            )
            signal_latch.checkpoint()
            effects.install_local_snapshot(local_temporary, local_destination)
            local_temporary = None
            local_installed = True
            if (
                effects.verify_local_snapshot(
                    local_destination, config.model, signal_latch
                )
                != local_verification
            ):
                raise StageError("local snapshot changed during atomic installation")

        assert local_verification is not None
        remote_probe = effects.probe_remote_destination(
            config, owner_token, register_process, signal_latch
        )
        cleanup_confirmations.append(remote_probe.cleanup_confirmed)
        remote_verification = remote_probe.verification
        if remote_verification is None:
            remote_result = effects.transfer_remote_snapshot(
                config,
                local_destination,
                local_verification,
                owner_token,
                register_process,
                signal_latch,
            )
            cleanup_confirmations.append(remote_result.cleanup_confirmed)
            remote_verification = remote_result.verification
            remote_installed = remote_result.installed
        if remote_verification is None:
            raise StageError("both hosts must return verified snapshot manifests")
        if local_verification != remote_verification:
            raise StageError("local and remote full SHA-256 manifests differ")
    except BaseException as error:
        caught_error = error
        if isinstance(error, OperationError):
            cleanup_confirmations.append(error.cleanup_confirmed)
    finally:
        signal_latch.begin_cleanup()
        if local_temporary is not None:
            cleanup_confirmations.append(
                effects.cleanup_local_temporary(local_temporary)
            )

    cleanup_succeeded = all(cleanup_confirmations)
    status = (
        "completed"
        if caught_error is None and cleanup_succeeded
        else ("cleanup_failed" if not cleanup_succeeded else "staging_failed")
    )
    owned_process_values: list[JsonValue] = [
        cast(JsonObject, process.model_dump(mode="json")) for process in processes
    ]
    result: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "status": status,
        "reportable": False,
        "result_scope": "model_staging",
        "performance_comparable": False,
        "performance_claim": None,
        "completed_normally": caught_error is None,
        "cleanup_succeeded": cleanup_succeeded,
        "interrupted_signal": signal_latch.signal_number,
        "error": (
            None
            if caught_error is None
            else f"{type(caught_error).__name__}: {caught_error}"
        ),
        "model": config.model.model_dump(mode="json"),
        "paths": {
            config.local_host_name: config.local_destination,
            config.remote_host_name: config.remote_destination,
        },
        "acquisition": acquisition,
        "local_installed": local_installed,
        "remote_installed": remote_installed,
        "local_verification": (
            None
            if local_verification is None
            else local_verification.model_dump(mode="json")
        ),
        "remote_verification": (
            None
            if remote_verification is None
            else remote_verification.model_dump(mode="json")
        ),
        "staging_config": config.model_dump(mode="json", exclude_none=True),
        "owned_processes": owned_process_values,
    }
    effects.write_result_json(_BENCHMARK_RESULT, cast(Mapping[str, object], result))
    return result


def extract_tar_stream(
    root: Path, input_file: IO[bytes], checkpoint: Callable[[], None]
) -> None:
    directory = _open_owned_directory(root)
    try:
        if not _owned_directory_path_matches(directory):
            raise StageError("archive root identity changed")
        _extract_tar_stream_at(directory, input_file, checkpoint)
    finally:
        directory.close()


def _extract_tar_stream_at(
    directory: OwnedDirectory,
    input_file: IO[bytes],
    checkpoint: Callable[[], None],
) -> None:
    seen: set[str] = set()
    with tarfile.open(fileobj=input_file, mode="r|*") as archive:
        for member in archive:
            checkpoint()
            relative = _validate_relative_file_name(member.name)
            relative_name = str(relative)
            if relative_name in seen:
                raise StageError(f"archive repeats {relative_name}")
            seen.add(relative_name)
            if not member.isfile():
                raise StageError(
                    f"archive member is not a regular file: {relative_name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise StageError(f"cannot read archive member {relative_name}")
            parent_descriptor, filename = _ensure_owned_relative_parent(
                directory, relative_name
            )
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    filename,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    member.mode & 0o777,
                    dir_fd=parent_descriptor,
                )
                _retain_owned_entry(directory, relative_name, descriptor, "regular")
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                raise
            finally:
                os.close(parent_descriptor)
            assert descriptor is not None
            written = 0
            with source, os.fdopen(descriptor, "wb", closefd=True) as output:
                while chunk := source.read(1024 * 1024):
                    checkpoint()
                    output.write(chunk)
                    written += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written != member.size:
                raise StageError(f"archive member size changed: {relative_name}")


def _remote_process_identity() -> tuple[int, int, int]:
    process_id = os.getpid()
    with contextlib.suppress(PermissionError):
        os.setsid()
    process_group_id = os.getpgid(process_id)
    if process_group_id != process_id:
        raise StageError("remote helper could not establish an owned process group")
    return process_id, process_group_id, _process_start_time_ticks(process_id)


def _remote_destination_probe(
    destination: Path, model: ModelSpec
) -> SnapshotVerification | None:
    parent_descriptor = _open_directory_without_symlinks(destination.parent)
    try:
        try:
            destination_status = os.stat(
                destination.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(destination_status.st_mode) or stat.S_ISLNK(
            destination_status.st_mode
        ):
            raise StageError("remote destination exists but is not a real directory")
        descriptor = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            if not _snapshot_regular_files(descriptor):
                raise StageError(
                    "refusing to replace a pre-existing empty remote destination"
                )
            return _verify_snapshot_descriptor(descriptor, model)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def remote_helper_main(
    create_owned_directory: Callable[[Path], OwnedDirectory] = _create_owned_directory,
) -> int:
    """Run the stdin-framed remote verifier/receiver under foreground SSH."""
    response: RemoteResponse
    temporary: Path | None = None
    temporary_directory: OwnedDirectory | None = None
    temporary_owned = False
    cleanup_succeeded = True
    installed = False
    verification: SnapshotVerification | None = None
    request: RemoteRequest | None = None
    managed_signal: int | None = None

    def handle_signal(signal_number: int, _frame: object) -> None:
        nonlocal managed_signal
        managed_signal = signal_number
        raise ManagedSignalError(signal_number)

    try:
        request = RemoteRequest.model_validate_json(sys.stdin.buffer.readline())
        os.environ[_OWNER_TOKEN_ENVIRONMENT] = request.owner_token
        os.environ[_NAMESPACE_ENVIRONMENT] = request.namespace
        observed_host_name = socket.gethostname()
        if observed_host_name != request.expected_host_name:
            raise StageError(
                f"remote host identity {observed_host_name!r} does not match "
                f"{request.expected_host_name!r}"
            )
        process_id, process_group_id, start_time_ticks = _remote_process_identity()
        print(
            RemoteIdentity(
                schema_version=1,
                kind="identity",
                pid=process_id,
                process_group_id=process_group_id,
                start_time_ticks=start_time_ticks,
                owner_token=request.owner_token,
                namespace=request.namespace,
                host_name=observed_host_name,
            ).model_dump_json(),
            flush=True,
        )
        previous_handlers = {
            signal_number: signal.getsignal(signal_number)
            for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            for signal_number in previous_handlers:
                signal.signal(signal_number, handle_signal)
            destination = Path(request.destination)
            existing = _remote_destination_probe(destination, request.model)
            if request.operation == "probe":
                verification = existing
            else:
                if existing is not None:
                    raise StageError("remote destination appeared before transfer")
                temporary = Path(request.temporary_path)
                temporary_directory = create_owned_directory(temporary)
                temporary_owned = True
                _extract_tar_stream_at(
                    temporary_directory,
                    sys.stdin.buffer,
                    lambda: (
                        None
                        if managed_signal is None
                        else (_raise_managed_signal(managed_signal))
                    ),
                )
                _create_owned_json_at(
                    temporary_directory,
                    _MODEL_RECEIPT,
                    {
                        "repo_id": request.model.model_id,
                        "revision": request.model.revision,
                    },
                )
                verification = _verify_snapshot_descriptor(
                    temporary_directory.descriptor, request.model
                )
                if not _owned_directory_path_matches(temporary_directory):
                    raise StageError("remote owned temporary identity changed")
                if not _validate_owned_tree(temporary_directory):
                    raise StageError(
                        "remote model tree differs from its creation journal"
                    )
                _rename_directory_noreplace_at(
                    temporary_directory.parent_descriptor,
                    temporary.name,
                    destination.name,
                )
                temporary_owned = False
                installed = True
                temporary_directory.path = destination
                observed_destination = os.stat(
                    destination.name,
                    dir_fd=temporary_directory.parent_descriptor,
                    follow_symlinks=False,
                )
                if (observed_destination.st_dev, observed_destination.st_ino) != (
                    temporary_directory.device,
                    temporary_directory.inode,
                ):
                    raise StageError("installed remote destination identity changed")
                if (
                    _verify_snapshot_descriptor(
                        temporary_directory.descriptor, request.model
                    )
                    != verification
                ):
                    raise StageError(
                        "remote snapshot changed during atomic installation"
                    )
                if not _validate_owned_tree(temporary_directory):
                    raise StageError(
                        "installed remote model tree changed during rename"
                    )
                temporary_directory.close()
                temporary_directory = None
        finally:
            for signal_number, previous_handler in previous_handlers.items():
                signal.signal(signal_number, previous_handler)
        response = RemoteResponse(
            schema_version=1,
            kind="result",
            status="completed",
            verification=verification,
            installed=installed,
            cleanup_succeeded=True,
            error=None,
        )
    except BaseException as error:
        if isinstance(error, OperationError):
            cleanup_succeeded = cleanup_succeeded and error.cleanup_confirmed
        if temporary_owned and temporary_directory is not None:
            cleanup_succeeded = _cleanup_owned_directory(temporary_directory)
            temporary_directory = None
            temporary_owned = False
        elif installed and temporary_directory is not None:
            temporary_directory.close()
            temporary_directory = None
            cleanup_succeeded = False
        response = RemoteResponse(
            schema_version=1,
            kind="result",
            status="failed",
            verification=None,
            installed=installed,
            cleanup_succeeded=cleanup_succeeded and not temporary_owned,
            error=f"{type(error).__name__}: {error}",
        )
    print(response.model_dump_json(), flush=True)
    return 0 if response.status == "completed" else 1


def _raise_managed_signal(signal_number: int) -> None:
    raise ManagedSignalError(signal_number)


class CliArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path


class LeasePreparationCliArguments(argparse.Namespace):
    config: Path
    metadata_output: Path
    wrapper_python: Path
    child_python: Path
    benchmark_lease_script: Path
    staging_script: Path
    owner: str
    purpose: str
    expected_duration_seconds: float
    heartbeat_seconds: float
    cleanup_grace_seconds: float
    lease_path: Path
    lock_path: Path
    result_root: Path


def parse_args(arguments: Sequence[str] | None = None) -> CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lease-path", type=Path, default=_DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=_DEFAULT_LOCK_PATH)
    parser.add_argument("--result-dir", required=True, type=Path)
    return cast(CliArguments, parser.parse_args(arguments))


def parse_lease_preparation_args(
    arguments: Sequence[str] | None = None,
) -> LeasePreparationCliArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a strict model-staging config and create fresh benchmark "
            "lease metadata plus the exact wrapper command."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--wrapper-python", required=True, type=Path)
    parser.add_argument("--child-python", required=True, type=Path)
    parser.add_argument("--benchmark-lease-script", required=True, type=Path)
    parser.add_argument("--staging-script", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-duration-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--cleanup-grace-seconds", required=True, type=float)
    parser.add_argument("--lease-path", type=Path, default=_DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=_DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", required=True, type=Path)
    return cast(LeasePreparationCliArguments, parser.parse_args(arguments))


def load_config(path: Path) -> StageConfig:
    _require_canonical_regular_file(path, "staging config")
    try:
        return StageConfig.model_validate_json(
            _read_regular_file_path(path, "staging config")
        )
    except ValidationError as error:
        raise StageError(f"invalid staging config: {error}") from error


def prepare_lease_main(arguments: Sequence[str] | None = None) -> int:
    args = parse_lease_preparation_args(arguments)
    prepared = prepare_lease_metadata(
        config_path=args.config,
        metadata_output=args.metadata_output,
        wrapper_python=args.wrapper_python,
        child_python=args.child_python,
        benchmark_lease_script=args.benchmark_lease_script,
        staging_script=args.staging_script,
        owner=args.owner,
        purpose=args.purpose,
        expected_duration_seconds=args.expected_duration_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        lease_path=args.lease_path,
        lock_path=args.lock_path,
        result_root=args.result_root,
    )
    print(
        json.dumps(
            prepared.machine_output(args.metadata_output),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if arguments is None else arguments)
    if normalized == ["remote-helper"]:
        return remote_helper_main()
    if normalized and normalized[0] == "prepare-lease":
        return prepare_lease_main(normalized[1:])
    args = parse_args(normalized)
    if not all(
        path.is_absolute()
        for path in (args.config, args.lease_path, args.lock_path, args.result_dir)
    ):
        raise StageError("all CLI paths must be absolute")
    config = load_config(args.config)
    if args.result_dir != Path(config.result_directory):
        raise StageError("--result-dir must exactly match config.result_directory")
    validate_active_lease(
        config,
        config_path=args.config,
        lease_path=args.lease_path,
        lock_path=args.lock_path,
    )
    result_directory_descriptor = _inherited_result_directory_descriptor(
        Path(config.result_directory)
    )
    latch = SignalLatch()
    managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in managed_signals
    }
    try:
        for signal_number in managed_signals:
            signal.signal(signal_number, latch.handle)
        _validate_local_source_identity(config, latch.checkpoint)
        effects = SystemEffects(config, result_directory_descriptor)
        _validate_local_source_identity(config, latch.checkpoint)
        result = run_staging(
            config,
            effects,
            latch,
        )
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
        os.close(result_directory_descriptor)
    if latch.signal_number is not None:
        return 128 + latch.signal_number
    if result["status"] != "completed":
        print(result["error"] or result["status"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
