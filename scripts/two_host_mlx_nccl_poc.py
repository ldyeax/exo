#!/usr/bin/env python3
"""Leased two-host, three-rank MLX/NCCL proof-of-concept harness.

This command is intentionally narrower than the general benchmark tooling.  It
accepts one exact model snapshot, starts exactly two Exo nodes, validates a
two-GPU plus one-GPU Tensor/NCCL placement, and owns the resulting instance for
the complete create/benchmark/delete lifecycle.

This first hardware-specific proof requires the complete physical GPU inventory
to be exactly two coordinator GPUs plus one worker GPU. General selected-subset
support is deliberately outside this harness.

The command must be run underneath ``scripts/benchmark_lease.py``.  It does not
acquire the global benchmark lease itself.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import http.client
import importlib.machinery
import importlib.metadata
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
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import IO, Literal, Protocol, TypeAlias, cast
from urllib.parse import urlencode

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
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_NVIDIA_RESOURCE_PREFIX = "nvidia-gpu:"
_MODEL_RECEIPT = ".exo-huggingface-revision.json"
_RESULT_FRAGMENT = "benchmark-result.json"
_RUNTIME_METADATA = "runtime-metadata.json"
_RESULT_DIRECTORY_FD_ENVIRONMENT = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
_DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
_DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
_LEASE_BIND_TIMEOUT_SECONDS = 5.0
_LEASE_METADATA_MAX_AGE = timedelta(minutes=15)
_MINIMUM_CLEANUP_GRACE_SECONDS = 300.0


class HarnessError(RuntimeError):
    """An expected fail-closed harness error."""


class HttpResponseError(HarnessError):
    def __init__(self, status: int, reason: str, body: str) -> None:
        super().__init__(f"HTTP {status} {reason}: {body[:300]}")
        self.status = status


class ManagedSignalError(HarnessError):
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


class RemoteOwnerReceipt(StrictModel):
    pid: int = Field(gt=0)
    process_group_id: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)
    owner_token: str = Field(min_length=1)
    namespace: str = Field(min_length=1)


class ProcessCleanupReceipt(StrictModel):
    host_name: str = Field(min_length=1)
    ownership_verified: bool
    terminated: bool
    forced: bool
    error: str | None = None


class GpuIdentity(StrictModel):
    device_uuid: str = Field(min_length=1)
    pci_bus_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @property
    def resource_id(self) -> str:
        return f"{_NVIDIA_RESOURCE_PREFIX}{self.device_uuid}"


class HcaPort(StrictModel):
    device: str = Field(min_length=1)
    port: int = Field(ge=1)
    gid: str

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        parsed = ipaddress.ip_address(value)
        if (
            parsed.version != 6
            or parsed.is_unspecified
            or int(parsed) & ((1 << 64) - 1) == 0
        ):
            raise ValueError("gid must be a port-specific IPv6 address")
        return str(parsed)


class SourceIdentity(StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_file_hashes: dict[str, str]

    @field_validator("dirty_file_hashes")
    @classmethod
    def validate_dirty_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for relative_path, digest in value.items():
            if not relative_path or PurePosixPath(relative_path).is_absolute():
                raise ValueError("dirty source paths must be nonempty and relative")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("dirty source hashes must be lowercase SHA-256")
        return value


class ModelSnapshot(StrictModel):
    model_id: str = Field(min_length=3)
    revision: str
    expected_weight_bytes: int = Field(gt=0)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _HEX_REVISION.fullmatch(value) is None:
            raise ValueError("model revision must be an exact lowercase 40-hex commit")
        return value


class HostConfig(StrictModel):
    name: str
    role: Literal["coordinator", "worker"]
    transport: Literal["local", "ssh"]
    ssh_target: str | None = None
    python_executable: str = Field(min_length=1)
    source_directory: str = Field(min_length=1)
    source: SourceIdentity
    model_path: str = Field(min_length=1)
    gpus: tuple[GpuIdentity, ...]
    hca_ports: tuple[HcaPort, ...]
    cpu_set: tuple[int, ...]
    numa_nodes: tuple[int, ...]
    launch_argv: tuple[str, ...]
    environment: dict[str, str]
    zenoh_port: int = Field(ge=1, le=65535)
    discovery_port: int = Field(ge=1, le=65535)
    launch_order: int = Field(ge=0)
    forbidden_process_substrings: tuple[str, ...] = (
        "uvicorn",
        "sglang.launch_server",
        "vllm.entrypoints",
        "mlx_nccl_smoke",
        "nccl-tests",
        "nccl_test",
        "hf download",
        "huggingface-cli download",
    )

    @model_validator(mode="after")
    def validate_host(self) -> "HostConfig":
        if _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("host name must be safe for result file names")
        if self.transport == "ssh" and not self.ssh_target:
            raise ValueError("ssh transport requires ssh_target")
        if self.transport == "local" and self.ssh_target is not None:
            raise ValueError("local transport must not set ssh_target")
        if not Path(self.python_executable).is_absolute():
            raise ValueError("python_executable must be absolute")
        if not Path(self.source_directory).is_absolute():
            raise ValueError("source_directory must be absolute")
        if not Path(self.model_path).is_absolute():
            raise ValueError("model_path must be absolute")
        if not self.launch_argv:
            raise ValueError("launch_argv must not be empty")
        for name, value in self.environment.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"invalid environment variable name {name!r}")
            if "\0" in value:
                raise ValueError(f"environment variable {name} contains NUL")
        if not self.cpu_set or min(self.cpu_set) < 0:
            raise ValueError("cpu_set must contain nonnegative CPU IDs")
        if tuple(sorted(set(self.cpu_set))) != self.cpu_set:
            raise ValueError("cpu_set must be sorted and unique")
        if not self.numa_nodes or min(self.numa_nodes) < 0:
            raise ValueError("numa_nodes must contain nonnegative node IDs")
        if tuple(sorted(set(self.numa_nodes))) != self.numa_nodes:
            raise ValueError("numa_nodes must be sorted and unique")
        if len({gpu.device_uuid for gpu in self.gpus}) != len(self.gpus):
            raise ValueError("GPU UUIDs must be unique within a host")
        if not self.hca_ports:
            raise ValueError("hca_ports must not be empty")
        if len({(port.device, port.port) for port in self.hca_ports}) != len(
            self.hca_ports
        ):
            raise ValueError("HCA device/port selections must be unique within a host")
        return self


class ApiConfig(StrictModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class BenchmarkConfig(StrictModel):
    prompt: str = Field(min_length=1)
    expected_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    warmup_count: int = Field(ge=2)
    sample_count: int = Field(ge=3)
    max_tokens: int = Field(ge=1)
    seed: int
    temperature: float

    @field_validator("temperature")
    @classmethod
    def require_deterministic_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("proof benchmark temperature must be 0")
        return value


class TimeoutConfig(StrictModel):
    process_start_seconds: float = Field(gt=0)
    api_start_seconds: float = Field(gt=0)
    cluster_seconds: float = Field(gt=0)
    runner_ready_seconds: float = Field(gt=0)
    request_seconds: float = Field(gt=0)
    cleanup_seconds: float = Field(gt=0)
    poll_seconds: float = Field(gt=0)


class PythonAbiIdentity(StrictModel):
    implementation: Literal["cpython"]
    major: int = Field(ge=3)
    minor: int = Field(ge=0)
    cache_tag: str = Field(min_length=1)
    soabi: str = Field(min_length=1)
    abiflags: str


class HostRuntimePin(StrictModel):
    exo_rs_native_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeRequirements(StrictModel):
    cuda_major: Literal[13]
    minimum_nvidia_driver_version: str
    python_abi: PythonAbiIdentity
    host_pins: dict[str, HostRuntimePin]

    @field_validator("minimum_nvidia_driver_version")
    @classmethod
    def validate_driver_version(cls, value: str) -> str:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value) is None:
            raise ValueError("minimum NVIDIA driver must be a dotted numeric version")
        return value


class HarnessConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str
    namespace: str
    result_directory: str
    api: ApiConfig
    nccl_coordinator_port: int = Field(ge=1, le=65535)
    reserved_ports: tuple[int, ...]
    model: ModelSnapshot
    hosts: tuple[HostConfig, ...]
    benchmark: BenchmarkConfig
    timeouts: TimeoutConfig
    runtime: RuntimeRequirements

    @model_validator(mode="after")
    def validate_run(self) -> "HarnessConfig":
        if _SAFE_NAME.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be safe for paths and namespaces")
        if not self.namespace or self.run_id not in self.namespace:
            raise ValueError("namespace must contain the unique run_id")
        if not Path(self.result_directory).is_absolute():
            raise ValueError("result_directory must be absolute")
        if len(self.hosts) != 2:
            raise ValueError("proof requires exactly two hosts")
        if len({host.name for host in self.hosts}) != 2:
            raise ValueError("host names must be unique")
        if {host.role for host in self.hosts} != {"coordinator", "worker"}:
            raise ValueError("proof requires one coordinator and one worker")
        coordinator = next(host for host in self.hosts if host.role == "coordinator")
        worker = next(host for host in self.hosts if host.role == "worker")
        if coordinator.transport != "local" or worker.transport != "ssh":
            raise ValueError("coordinator must be local and worker must use ssh")
        if len(coordinator.gpus) != 2 or len(worker.gpus) != 1:
            raise ValueError("proof requires a coordinator 2-GPU + worker 1-GPU layout")
        all_gpu_uuids = [gpu.device_uuid for host in self.hosts for gpu in host.gpus]
        if len(set(all_gpu_uuids)) != 3:
            raise ValueError("all three GPU UUIDs must be unique")
        if len({host.launch_order for host in self.hosts}) != 2:
            raise ValueError("host launch_order values must be unique")
        if len({host.discovery_port for host in self.hosts}) != 1:
            raise ValueError("both hosts must use one shared multicast discovery port")
        if set(self.runtime.host_pins) != {host.name for host in self.hosts}:
            raise ValueError("runtime host_pins must exactly match configured hosts")
        all_ports = (
            self.api.port,
            self.nccl_coordinator_port,
            *(host.zenoh_port for host in self.hosts),
            self.hosts[0].discovery_port,
        )
        if any(port < 1 or port > 65535 for port in self.reserved_ports):
            raise ValueError("reserved ports must be in 1..65535")
        if len(set(self.reserved_ports)) != len(self.reserved_ports):
            raise ValueError("reserved_ports must be unique")
        if not set(all_ports).issubset(self.reserved_ports):
            raise ValueError(
                "every API, NCCL, Zenoh, and discovery port must be reserved"
            )
        if len(set(all_ports)) != len(all_ports):
            raise ValueError("service ports must be globally unique for the proof")
        for host in self.hosts:
            _validate_launch_contract(host, self)
        return self


class ModelProbeResult(StrictModel):
    host_name: str
    path: str
    model_id: str
    revision: str
    weight_bytes: int = Field(ge=0)
    physical_weight_bytes: int = Field(ge=0)
    weight_files: int = Field(ge=0)
    sha256_manifest: dict[str, str]
    receipt_kind: Literal["exo", "huggingface"]
    verified: bool
    error: str | None = None

    @field_validator("sha256_manifest")
    @classmethod
    def validate_manifest(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in value.values()
        ):
            raise ValueError("model manifest values must be lowercase SHA-256")
        return value


class HcaPortObservation(StrictModel):
    device: str
    port: int
    state: str
    rate: str
    physical_state: str = "unknown"
    link_layer: str = "unknown"
    lid: str = "unknown"
    gids: tuple[str, ...] = ()
    net_devices: tuple[str, ...] = ()
    ip_addresses: tuple[str, ...] = ()
    counters: dict[str, str] = Field(default_factory=dict)


class HostPreflightReport(StrictModel):
    schema_version: Literal[1]
    run_id: str
    host_name: str
    passed: bool
    conflicts: tuple[str, ...]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_file_hashes: dict[str, str]
    gpus: tuple[GpuIdentity, ...]
    checked_tcp_ports: tuple[int, ...]
    checked_udp_ports: tuple[int, ...]
    hca_ports: tuple[HcaPortObservation, ...]
    amx_flags: tuple[str, ...]
    facts: dict[
        str,
        JsonScalar | list[JsonScalar] | dict[str, JsonScalar],
    ]


class HostPreflightRequest(StrictModel):
    schema_version: Literal[1]
    run_id: str
    host: HostConfig
    reserved_ports: tuple[int, ...]


class HostProbe(Protocol):
    def source_identity(self, source_directory: str) -> SourceIdentity: ...

    def gpu_identities(self) -> tuple[GpuIdentity, ...]: ...

    def gpu_compute_processes(self) -> tuple[str, ...]: ...

    def busy_ports(
        self, ports: Sequence[int], socket_type: Literal["tcp", "udp"]
    ) -> tuple[int, ...]: ...

    def hca_port_observations(
        self, ports: Sequence[HcaPort]
    ) -> tuple[HcaPortObservation, ...]: ...

    def amx_flags(self) -> tuple[str, ...]: ...

    def online_cpu_ids(self) -> tuple[int, ...]: ...

    def numa_node_ids(self) -> tuple[int, ...]: ...

    def process_conflicts(self, substrings: Sequence[str]) -> tuple[str, ...]: ...

    def raid_operations(self) -> tuple[str, ...]: ...

    def facts(
        self,
    ) -> dict[str, JsonScalar | list[JsonScalar] | dict[str, JsonScalar]]: ...


class LeaseMetadataValidator(Protocol):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


def _parse_id_ranges(value: str) -> tuple[int, ...]:
    result: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start, end = int(raw_start), int(raw_end)
            if start > end:
                raise ValueError(f"invalid descending ID range {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return tuple(sorted(result))


class LinuxHostProbe:
    """Read-only Linux facts used by the internal host-preflight subprocess."""

    def __init__(
        self,
        infiniband_class_path: Path = Path("/sys/class/infiniband"),
    ) -> None:
        self._infiniband_class_path = infiniband_class_path

    @staticmethod
    def _command(
        arguments: Sequence[str],
        *,
        working_directory: str | None = None,
        allow_failure: bool = False,
    ) -> str:
        completed = subprocess.run(
            arguments,
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        if completed.returncode != 0 and not allow_failure:
            raise HarnessError(
                f"probe command {arguments[0]} failed with {completed.returncode}: "
                f"{completed.stderr[-500:]}"
            )
        return completed.stdout.strip()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def native_extension_identity(self, module_name: str) -> dict[str, JsonScalar]:
        identity: dict[str, JsonScalar] = {
            "module": module_name,
            "path": None,
            "sha256": None,
            "is_native_extension": False,
        }
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError):
            return identity
        if spec is None or spec.origin is None:
            return identity

        artifact_path = Path(spec.origin).resolve()
        identity["path"] = str(artifact_path)
        is_native_extension = isinstance(
            spec.loader, importlib.machinery.ExtensionFileLoader
        ) and any(
            artifact_path.name.endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
        identity["is_native_extension"] = is_native_extension
        if is_native_extension:
            with contextlib.suppress(OSError):
                identity["sha256"] = self._sha256_file(artifact_path)
        return identity

    def source_identity(self, source_directory: str) -> SourceIdentity:
        commit = self._command(("git", "-C", source_directory, "rev-parse", "HEAD"))
        tracked = self._command(
            (
                "git",
                "-C",
                source_directory,
                "diff",
                "--name-only",
                "-z",
                "HEAD",
            )
        )
        untracked = self._command(
            (
                "git",
                "-C",
                source_directory,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
        )
        dirty_paths = sorted(
            {path for path in (*tracked.split("\0"), *untracked.split("\0")) if path}
        )
        root = Path(source_directory)
        deleted_digest = hashlib.sha256(b"<deleted>").hexdigest()
        dirty_hashes = {
            relative_path: (
                self._sha256_file(root / relative_path)
                if (root / relative_path).is_file()
                else deleted_digest
            )
            for relative_path in dirty_paths
        }
        return SourceIdentity(commit=commit, dirty_file_hashes=dirty_hashes)

    def gpu_identities(self) -> tuple[GpuIdentity, ...]:
        output = self._command(
            (
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name",
                "--format=csv,noheader",
            )
        )
        rows = csv.reader(output.splitlines())
        return tuple(
            GpuIdentity(
                device_uuid=row[0].strip(),
                pci_bus_id=row[1].strip(),
                model_name=row[2].strip(),
            )
            for row in rows
            if len(row) == 3
        )

    def gpu_compute_processes(self) -> tuple[str, ...]:
        output = self._command(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ),
            allow_failure=True,
        )
        return tuple(
            line.strip()
            for line in output.splitlines()
            if line.strip() and "No running processes" not in line
        )

    def busy_ports(
        self, ports: Sequence[int], socket_type: Literal["tcp", "udp"]
    ) -> tuple[int, ...]:
        kind = socket.SOCK_STREAM if socket_type == "tcp" else socket.SOCK_DGRAM
        busy: list[int] = []
        for port in ports:
            probe = socket.socket(socket.AF_INET6, kind)
            try:
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                probe.bind(("::", port))
                if socket_type == "tcp":
                    probe.listen(1)
            except OSError:
                busy.append(port)
            finally:
                probe.close()
        return tuple(busy)

    def hca_port_observations(
        self, ports: Sequence[HcaPort]
    ) -> tuple[HcaPortObservation, ...]:
        observations: list[HcaPortObservation] = []
        for port in ports:
            root = self._infiniband_class_path / port.device / "ports" / str(port.port)
            try:
                state = (root / "state").read_text().strip()
                rate = (root / "rate").read_text().strip()
                physical_state = (root / "phys_state").read_text().strip()
                link_layer = (root / "link_layer").read_text().strip()
                lid = (root / "lid").read_text().strip()
                observed_gids: set[str] = set()
                for path in sorted((root / "gids").glob("[0-9]*")):
                    try:
                        parsed_gid = ipaddress.ip_address(path.read_text().strip())
                    except (OSError, ValueError):
                        continue
                    if (
                        parsed_gid.version == 6
                        and not parsed_gid.is_unspecified
                        and int(parsed_gid) & ((1 << 64) - 1) != 0
                    ):
                        observed_gids.add(str(parsed_gid))
                gids = tuple(sorted(observed_gids))
                observed_net_devices: set[str] = set()
                for path in sorted((root / "gid_attrs" / "ndevs").glob("[0-9]*")):
                    try:
                        net_device = path.read_text().strip()
                    except OSError:
                        continue
                    if net_device:
                        observed_net_devices.add(net_device)
                net_devices = tuple(sorted(observed_net_devices))
                observed_addresses: set[str] = set()
                for net_device in net_devices:
                    raw_addresses = self._command(
                        ("ip", "-o", "address", "show", "dev", net_device),
                        allow_failure=True,
                    )
                    for line in raw_addresses.splitlines():
                        fields = line.split()
                        for family in ("inet", "inet6"):
                            if family in fields and fields.index(family) + 1 < len(
                                fields
                            ):
                                address = fields[fields.index(family) + 1].split(
                                    "/", 1
                                )[0]
                                observed_addresses.add(
                                    str(ipaddress.ip_address(address))
                                )
                ip_addresses = tuple(sorted(observed_addresses))
                counters = {
                    path.name: path.read_text().strip()
                    for path in sorted((root / "counters").glob("*"))
                    if path.is_file()
                }
            except OSError as error:
                state = f"MISSING: {error}"
                rate = "unknown"
                physical_state = "unknown"
                link_layer = "unknown"
                lid = "unknown"
                gids = ()
                net_devices = ()
                ip_addresses = ()
                counters = {}
            observations.append(
                HcaPortObservation(
                    device=port.device,
                    port=port.port,
                    state=state,
                    rate=rate,
                    physical_state=physical_state,
                    link_layer=link_layer,
                    lid=lid,
                    gids=gids,
                    net_devices=net_devices,
                    ip_addresses=ip_addresses,
                    counters=counters,
                )
            )
        return tuple(observations)

    def amx_flags(self) -> tuple[str, ...]:
        cpu_info = Path("/proc/cpuinfo").read_text()
        flags_line = next(
            (line for line in cpu_info.splitlines() if line.startswith("flags")), ""
        )
        flags = set(flags_line.partition(":")[2].split())
        return tuple(sorted(flags.intersection(("amx_bf16", "amx_int8", "amx_tile"))))

    def online_cpu_ids(self) -> tuple[int, ...]:
        return _parse_id_ranges(Path("/sys/devices/system/cpu/online").read_text())

    def numa_node_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(path.name.removeprefix("node"))
                for path in Path("/sys/devices/system/node").glob("node[0-9]*")
            )
        )

    @staticmethod
    def _ancestor_pids() -> set[int]:
        ancestors: set[int] = set()
        pid = os.getpid()
        while pid > 1 and pid not in ancestors:
            ancestors.add(pid)
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                pid = int(fields[1])
            except (OSError, IndexError, ValueError):
                break
        return ancestors

    @staticmethod
    def _command_is_conflict(
        arguments: Sequence[str], substrings: Sequence[str]
    ) -> bool:
        if not arguments:
            return False
        exact_executables = {
            "exo",
            "ib_read_bw",
            "ib_write_bw",
            "ib_send_bw",
            "ib_read_lat",
            "ib_write_lat",
            "ib_send_lat",
            "all_reduce_perf",
            "aria2c",
            "b3sum",
            "hf",
            "huggingface-cli",
            "md5sum",
            "rsync",
            "sha256sum",
        }
        command_line = " ".join(arguments)
        executable = Path(arguments[0]).name
        argument_basenames = {Path(argument).name for argument in arguments}
        python_module_exo = any(
            argument == "-m"
            and index + 1 < len(arguments)
            and arguments[index + 1].split(".", 1)[0] == "exo"
            for index, argument in enumerate(arguments)
        )
        return (
            executable in exact_executables
            or python_module_exo
            or "exo" in argument_basenames
            or any(substring in command_line for substring in substrings)
        )

    def process_conflicts(self, substrings: Sequence[str]) -> tuple[str, ...]:
        ignored_pids = self._ancestor_pids()
        conflicts: list[str] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) in ignored_pids:
                continue
            try:
                raw_command = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            arguments = [
                part.decode("utf-8", errors="replace")
                for part in raw_command.split(b"\0")
                if part
            ]
            if not arguments:
                continue
            command_line = " ".join(arguments)
            if self._command_is_conflict(arguments, substrings):
                conflicts.append(f"pid={entry.name} command={command_line[:500]}")
        return tuple(sorted(conflicts))

    def raid_operations(self) -> tuple[str, ...]:
        operations: list[str] = []
        for path in Path("/sys/block").glob("md*/md/sync_action"):
            try:
                action = path.read_text().strip()
            except OSError:
                continue
            if action not in {"idle", "frozen"}:
                operations.append(f"{path}: {action}")
        return tuple(operations)

    def facts(
        self,
    ) -> dict[str, JsonScalar | list[JsonScalar] | dict[str, JsonScalar]]:
        raw_driver_versions = self._command(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
            allow_failure=True,
        )
        driver_versions = ",".join(
            sorted({line.strip() for line in raw_driver_versions.splitlines() if line})
        )
        runtime_versions: dict[str, JsonScalar] = {"python": sys.version}
        for distribution in (
            "exo",
            "mlx",
            "mlx-cuda-12",
            "mlx-cuda-13",
            "nvidia-nccl-cu12",
            "nvidia-nccl-cu13",
        ):
            try:
                runtime_versions[distribution] = importlib.metadata.version(
                    distribution
                )
            except importlib.metadata.PackageNotFoundError:
                runtime_versions[distribution] = None
        soabi = cast(object, sysconfig.get_config_var("SOABI"))
        python_abi: dict[str, JsonScalar] = {
            "implementation": sys.implementation.name,
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "cache_tag": sys.implementation.cache_tag,
            "soabi": soabi if isinstance(soabi, str) else None,
            "abiflags": sys.abiflags,
        }
        exo_rs_identity = self.native_extension_identity("exo_rs.exo_rs")
        exo_import_origin: JsonScalar = None
        exo_spec = importlib.util.find_spec("exo")
        if exo_spec is not None and exo_spec.origin is not None:
            exo_import_origin = str(Path(exo_spec.origin).resolve())
        frequency_policy: dict[str, JsonScalar] = {}
        for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
            for field_name in (
                "scaling_driver",
                "scaling_governor",
                "energy_performance_preference",
                "scaling_cur_freq",
                "scaling_min_freq",
                "scaling_max_freq",
            ):
                path = policy / field_name
                if path.is_file():
                    with contextlib.suppress(OSError):
                        frequency_policy[f"{policy.name}.{field_name}"] = (
                            path.read_text().strip()
                        )
        return {
            "load_average": Path("/proc/loadavg").read_text().strip(),
            "memory": "\n".join(Path("/proc/meminfo").read_text().splitlines()[:5]),
            "kernel": self._command(("uname", "-a"), allow_failure=True),
            "gpu_telemetry_csv": self._command(
                (
                    "nvidia-smi",
                    "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used,"
                    "memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,"
                    "power.limit,driver_version",
                    "--format=csv,noheader,nounits",
                ),
                allow_failure=True,
            ),
            "nvidia_smi_banner": self._command(("nvidia-smi",), allow_failure=True),
            "nvidia_driver_versions": driver_versions,
            "ip_addresses_json": self._command(
                ("ip", "-j", "address", "show"), allow_failure=True
            ),
            "ip_routes_json": self._command(
                ("ip", "-j", "route", "show"), allow_failure=True
            ),
            "opensm_processes": self._command(
                ("pgrep", "-a", "opensm"), allow_failure=True
            ),
            "cpu_frequency_policy": frequency_policy,
            "runtime_versions": runtime_versions,
            "python_abi": python_abi,
            "exo_rs_artifact": exo_rs_identity,
            "exo_import_origin": exo_import_origin,
        }


def _ib_state_is(value: str, expected: str) -> bool:
    return value.rpartition(":")[2].strip().upper() == expected.upper()


def collect_host_preflight(
    request: HostPreflightRequest, probe: HostProbe
) -> HostPreflightReport:
    host = request.host
    conflicts: list[str] = []

    try:
        source = probe.source_identity(host.source_directory)
    except Exception as error:
        source = SourceIdentity(commit="0" * 40, dirty_file_hashes={})
        conflicts.append(f"source probe failed: {type(error).__name__}: {error}")
    if source != host.source:
        conflicts.append("source commit or dirty-file hashes differ from config")

    try:
        gpus = probe.gpu_identities()
    except Exception as error:
        gpus = ()
        conflicts.append(f"GPU inventory probe failed: {type(error).__name__}: {error}")
    if set(gpus) != set(host.gpus):
        conflicts.append("GPU UUID/PCI/model inventory differs from config")

    gpu_processes = probe.gpu_compute_processes()
    if gpu_processes:
        conflicts.append("selected host has active GPU compute processes")
    process_conflicts = probe.process_conflicts(host.forbidden_process_substrings)
    if process_conflicts:
        conflicts.append("unowned inference, benchmark, or model-I/O process is active")
    raid_operations = probe.raid_operations()
    if raid_operations:
        conflicts.append("RAID maintenance is active")

    busy_tcp_ports = probe.busy_ports(request.reserved_ports, "tcp")
    busy_udp_ports = probe.busy_ports(request.reserved_ports, "udp")
    if busy_tcp_ports:
        conflicts.append(f"reserved TCP ports are busy: {list(busy_tcp_ports)}")
    if busy_udp_ports:
        conflicts.append(f"reserved UDP ports are busy: {list(busy_udp_ports)}")

    hca_observations = probe.hca_port_observations(host.hca_ports)
    if any(not _ib_state_is(port.state, "ACTIVE") for port in hca_observations):
        conflicts.append("one or more selected InfiniBand ports are not ACTIVE")
    if any(
        not _ib_state_is(port.physical_state, "LINKUP")
        or port.link_layer.lower() != "infiniband"
        or port.lid in {"unknown", "0", "0x0", "0x0000"}
        or not port.gids
        or not port.counters
        or configured.gid not in port.gids
        for port, configured in zip(hca_observations, host.hca_ports, strict=True)
    ):
        conflicts.append("selected InfiniBand port detail is incomplete or not LinkUp")
    amx_flags = probe.amx_flags()
    if not {"amx_bf16", "amx_int8", "amx_tile"}.issubset(amx_flags):
        conflicts.append("AMX BF16/INT8/tile capability is incomplete")
    online_cpu_ids = probe.online_cpu_ids()
    if not set(host.cpu_set).issubset(online_cpu_ids):
        conflicts.append("configured CPU set is not fully online")
    numa_node_ids = probe.numa_node_ids()
    if not set(host.numa_nodes).issubset(numa_node_ids):
        conflicts.append("configured NUMA memory nodes are unavailable")

    facts = probe.facts()
    facts.update(
        {
            "gpu_compute_processes": list(gpu_processes),
            "process_conflicts": list(process_conflicts),
            "raid_operations": list(raid_operations),
            "busy_tcp_ports": list(busy_tcp_ports),
            "busy_udp_ports": list(busy_udp_ports),
            "online_cpu_ids": list(online_cpu_ids),
            "numa_node_ids": list(numa_node_ids),
        }
    )
    return HostPreflightReport(
        schema_version=1,
        run_id=request.run_id,
        host_name=host.name,
        passed=not conflicts,
        conflicts=tuple(conflicts),
        source_commit=source.commit,
        dirty_file_hashes=source.dirty_file_hashes,
        gpus=gpus,
        checked_tcp_ports=request.reserved_ports,
        checked_udp_ports=request.reserved_ports,
        hca_ports=hca_observations,
        amx_flags=amx_flags,
        facts=facts,
    )


@dataclass(frozen=True)
class OwnedProcess:
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    owner_token: str
    namespace: str
    transport_pid: int
    log_path: str


@dataclass(frozen=True)
class ProcessCleanup:
    host_name: str
    ownership_verified: bool
    terminated: bool
    forced: bool
    error: str | None = None


class StartNodeError(HarnessError):
    def __init__(
        self,
        host_name: str,
        cause: BaseException,
        cleanup: ProcessCleanup,
        receipt: OwnedProcess | None,
    ) -> None:
        super().__init__(
            f"node start failed on {host_name}: {type(cause).__name__}: {cause}; "
            f"transactional cleanup={cleanup}"
        )
        self.cleanup = cleanup
        self.receipt = receipt


def _argument_value(arguments: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"launch argument {flag} has no value")
            values.append(arguments[index + 1])
        elif argument.startswith(flag + "="):
            values.append(argument.partition("=")[2])
    if len(values) > 1:
        raise ValueError(f"launch arguments repeat {flag}")
    if not values:
        return None
    if not values[0]:
        raise ValueError(f"launch argument {flag} has an empty value")
    return values[0]


def _validate_launch_contract(host: HostConfig, config: HarnessConfig) -> None:
    arguments = host.launch_argv
    for flag in (
        "--offline",
        "--no-downloads",
        "--force-master",
        "--no-api",
    ):
        occurrences = [
            argument
            for argument in arguments
            if argument == flag or argument.startswith(flag + "=")
        ]
        if any(argument != flag for argument in occurrences):
            raise ValueError(f"{host.name} launch uses a value for boolean {flag}")
        if len(occurrences) > 1:
            raise ValueError(f"{host.name} launch repeats {flag}")
    expected_prefix = (
        "numactl",
        f"--physcpubind={','.join(str(cpu) for cpu in host.cpu_set)}",
        f"--membind={','.join(str(node) for node in host.numa_nodes)}",
    )
    executable_index = len(expected_prefix)
    expected_exo_command = (host.python_executable, "-m", "exo")
    if (
        len(arguments) < executable_index + len(expected_exo_command)
        or not Path(arguments[0]).is_absolute()
        or Path(arguments[0]).name != expected_prefix[0]
        or tuple(arguments[1 : len(expected_prefix)]) != expected_prefix[1:]
        or tuple(
            arguments[executable_index : executable_index + len(expected_exo_command)]
        )
        != expected_exo_command
    ):
        raise ValueError(
            f"{host.name} launch must use exact numactl bindings and "
            "configured Python -m exo"
        )
    if _argument_value(arguments, "--namespace") != config.namespace:
        raise ValueError(f"{host.name} launch must use the configured namespace")
    if _argument_value(arguments, "--zenoh-port") != str(host.zenoh_port):
        raise ValueError(f"{host.name} launch must use its reserved Zenoh port")
    if _argument_value(arguments, "--discovery-port") != str(host.discovery_port):
        raise ValueError(f"{host.name} launch must use its reserved discovery port")
    api_port = _argument_value(arguments, "--api-port")
    if "--offline" not in arguments:
        raise ValueError(f"{host.name} launch must run offline after model staging")
    if "--no-downloads" not in arguments:
        raise ValueError(f"{host.name} launch must disable model downloads")
    if host.role == "coordinator":
        if "--force-master" not in arguments:
            raise ValueError("coordinator launch must force master")
        if "--no-api" in arguments:
            raise ValueError("coordinator must expose the API")
        if api_port != str(config.api.port):
            raise ValueError("coordinator launch must use the reserved API port")
    else:
        if "--no-api" not in arguments:
            raise ValueError("worker launch must disable its API")
        if "--force-master" in arguments or api_port is not None:
            raise ValueError("worker launch must not configure coordinator API flags")
    model_path = Path(host.model_path)
    model_parent = str(model_path.parent)
    expected_model_directory = (
        f"{config.model.model_id.replace('/', '--')}--{config.model.revision}"
    )
    if model_path.name != expected_model_directory:
        raise ValueError(
            f"{host.name} model_path must name the exact revision-suffixed snapshot"
        )
    if host.environment.get("EXO_MODELS_READ_ONLY_DIRS") != model_parent:
        raise ValueError(
            f"{host.name} must configure exactly one canonical read-only model root"
        )
    visible_devices = host.environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible_devices != [gpu.device_uuid for gpu in host.gpus]:
        raise ValueError(
            f"{host.name} CUDA_VISIBLE_DEVICES must use configured UUID order"
        )
    exo_home = Path(host.environment.get("EXO_HOME", ""))
    if not exo_home.is_absolute() or config.run_id not in str(exo_home):
        raise ValueError(f"{host.name} EXO_HOME must be absolute and contain run_id")
    required_environment = {
        "NCCL_NET": "IB",
        "NCCL_GIN_ENABLE": "0",
        "NCCL_GIN_TYPE": "0",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_IB_MERGE_NICS": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "EXO_OFFLINE": "true",
        "EXO_MAX_CONCURRENT_REQUESTS": "1",
        "ENABLE_DISAGGREGATION": "false",
        "EXO_MLX_VISION_LOADING": "disabled",
        "PYTHONHASHSEED": str(config.benchmark.seed),
    }
    for name, expected in required_environment.items():
        if host.environment.get(name) != expected:
            raise ValueError(f"{host.name} must set {name}={expected}")
    path_value = host.environment.get("PATH")
    if not path_value or any(
        not Path(entry).is_absolute() for entry in path_value.split(":")
    ):
        raise ValueError(f"{host.name} must set an explicit absolute PATH")
    home_value = host.environment.get("HOME")
    if not home_value or not Path(home_value).is_absolute():
        raise ValueError(f"{host.name} must set an explicit absolute HOME")
    if "LD_LIBRARY_PATH" not in host.environment:
        raise ValueError(f"{host.name} must explicitly set LD_LIBRARY_PATH")
    if any(
        entry and not Path(entry).is_absolute()
        for entry in host.environment["LD_LIBRARY_PATH"].split(":")
    ):
        raise ValueError(f"{host.name} LD_LIBRARY_PATH entries must be absolute")
    expected_hca_selection = "=" + ",".join(
        f"{port.device}:{port.port}" for port in host.hca_ports
    )
    if host.environment.get("NCCL_IB_HCA") != expected_hca_selection:
        raise ValueError(f"{host.name} must set NCCL_IB_HCA={expected_hca_selection}")


def load_config(path: Path) -> HarnessConfig:
    try:
        return HarnessConfig.model_validate_json(path.read_text())
    except OSError as error:
        raise HarnessError(f"cannot read config {path}: {error}") from error
    except ValidationError as error:
        raise HarnessError(f"invalid strict harness config: {error}") from error


def _object(value: JsonValue, description: str) -> JsonObject:
    if not isinstance(value, dict):
        raise HarnessError(f"{description} must be a JSON object")
    return value


def _validated_json_value(value: object, description: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessError(f"{description} contains a non-finite number")
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        return [
            _validated_json_value(item, f"{description}[{index}]")
            for index, item in enumerate(items)
        ]
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        result: JsonObject = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise HarnessError(f"{description} contains a non-string object key")
            result[key] = _validated_json_value(item, f"{description}.{key}")
        return result
    raise HarnessError(f"{description} contains a non-JSON value")


def _parse_json_value(raw: str, description: str) -> JsonValue:
    try:
        value = cast(object, json.loads(raw))
    except json.JSONDecodeError as error:
        raise HarnessError(f"{description} is not valid JSON: {error}") from error
    return _validated_json_value(value, description)


def _string(value: JsonValue | None, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise HarnessError(f"{description} must be a nonempty string")
    return value


def _positive_float(value: JsonValue | None, description: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise HarnessError(f"{description} must be finite and positive")
    return float(value)


def _positive_integer(value: JsonValue | None, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HarnessError(f"{description} must be a positive integer")
    return value


def _array(value: JsonValue | None, description: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise HarnessError(f"{description} must be a JSON array")
    return value


def _parse_lease_timestamp(value: JsonValue | None, description: str) -> datetime:
    raw_timestamp = _string(value, description)
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise HarnessError(f"{description} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HarnessError(f"{description} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _command_option(command: Sequence[str], option: str) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(command):
        if argument == option:
            if index + 1 >= len(command):
                raise HarnessError(f"active lease command has no value for {option}")
            values.append(command[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument.partition("=")[2])
    if len(values) > 1:
        raise HarnessError(f"active lease command repeats {option}")
    return values[0] if values else None


def _lease_static_metadata(config: HarnessConfig, command: Sequence[str]) -> JsonObject:
    source = config.hosts[0].source
    if any(host.source != source for host in config.hosts[1:]):
        raise HarnessError(
            "lease v1 requires identical source identities on both hosts"
        )
    hosts: list[JsonValue] = [host.name for host in config.hosts]
    ports: list[JsonValue] = [port for port in config.reserved_ports]
    command_json: list[JsonValue] = [argument for argument in command]
    source_hashes: JsonObject = {
        relative_path: digest
        for relative_path, digest in source.dirty_file_hashes.items()
    }
    model_paths: JsonObject = {host.name: host.model_path for host in config.hosts}
    models: list[JsonValue] = [
        {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "paths": model_paths,
        }
    ]
    gpu_bindings: JsonObject = {
        host.name: [
            {
                "uuid": gpu.device_uuid,
                "pci_address": gpu.pci_bus_id,
            }
            for gpu in host.gpus
        ]
        for host in config.hosts
    }
    cpu_bindings: JsonObject = {
        host.name: {
            "cpu_set": ",".join(str(cpu) for cpu in host.cpu_set),
            "numa_nodes": [node for node in host.numa_nodes],
            "memory_policy": "bind:" + ",".join(str(node) for node in host.numa_nodes),
        }
        for host in config.hosts
    }
    hca_bindings: JsonObject = {
        host.name: [
            {
                "device": port.device,
                "port": port.port,
                "gid": port.gid,
            }
            for port in host.hca_ports
        ]
        for host in config.hosts
    }
    source_deployments: JsonObject = {
        host.name: {
            "path": host.source_directory,
            "commit": host.source.commit,
            "dirty_file_hashes": {
                relative_path: digest
                for relative_path, digest in host.source.dirty_file_hashes.items()
            },
        }
        for host in config.hosts
    }
    execution_contracts: JsonObject = {
        host.name: {
            "launch_argv": [argument for argument in host.launch_argv],
            "probe_environment": {
                name: value for name, value in sorted(host.environment.items())
            },
        }
        for host in config.hosts
    }
    owner_pids: JsonObject = {host.name: [] for host in config.hosts}
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": ports,
        "result_directory": config.result_directory,
        "command": command_json,
        "git": {
            "commit": source.commit,
            "dirty": bool(source.dirty_file_hashes),
            "dirty_file_hashes": source_hashes,
        },
        "hosts": hosts,
        "models": models,
        "runtime_requirements": cast(JsonValue, config.runtime.model_dump(mode="json")),
        "correctness_oracle": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "prompt_sha256": hashlib.sha256(
                config.benchmark.prompt.encode("utf-8")
            ).hexdigest(),
            "expected_content_sha256": (config.benchmark.expected_content_sha256),
        },
        "gpu_bindings": gpu_bindings,
        "cpu_bindings": cpu_bindings,
        "hca_bindings": hca_bindings,
        "source_deployments": source_deployments,
        "execution_contracts": execution_contracts,
        "owner_pids": owner_pids,
    }


def minimum_cleanup_grace_seconds(config: HarnessConfig) -> float:
    remote_start_checkpoint_delay = (
        2 * config.timeouts.process_start_seconds
        + config.timeouts.cleanup_seconds
        + 5.0
    )
    maximum_signal_checkpoint_delay = max(
        config.timeouts.api_start_seconds,
        config.timeouts.request_seconds,
        remote_start_checkpoint_delay,
        config.timeouts.poll_seconds,
    )
    delete_and_verify_bound = (
        2 * config.timeouts.request_seconds
        + config.timeouts.cleanup_seconds
        + config.timeouts.poll_seconds
    )
    sequential_node_stop_bound = 2 * (2 * config.timeouts.cleanup_seconds + 17.0)
    return max(
        _MINIMUM_CLEANUP_GRACE_SECONDS,
        maximum_signal_checkpoint_delay
        + delete_and_verify_bound
        + sequential_node_stop_bound
        + 30.0,
    )


def validate_active_lease(
    config: HarnessConfig,
    *,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int | None = None,
    timeout_seconds: float = _LEASE_BIND_TIMEOUT_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> JsonObject:
    """Bind this child and every static proof resource to the active lease."""
    if not all(path.is_absolute() for path in (config_path, lease_path, lock_path)):
        raise HarnessError("config, lease, and lock paths must be absolute")
    try:
        with lock_path.open("r") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                raise HarnessError("benchmark coordination lock is not held")
    except OSError as error:
        raise HarnessError(
            f"cannot inspect benchmark lock {lock_path}: {error}"
        ) from error
    expected_process_id = os.getpid() if process_id is None else process_id
    deadline = monotonic() + timeout_seconds
    while True:
        try:
            record = _object(
                _parse_json_value(lease_path.read_text(), "active lease record"),
                "active lease record",
            )
        except OSError as error:
            raise HarnessError(
                f"cannot read active lease record {lease_path}: {error}"
            ) from error
        child_pid = record.get("child_pid")
        if child_pid == expected_process_id:
            break
        if child_pid is not None:
            raise HarnessError("active lease belongs to a different child process")
        if monotonic() >= deadline:
            raise HarnessError("active lease did not publish this child PID")
        sleep(0.05)

    command_values = _array(record.get("command"), "active lease command")
    if not all(isinstance(value, str) for value in command_values):
        raise HarnessError("active lease command must contain only strings")
    command = tuple(value for value in command_values if isinstance(value, str))
    if not any(
        Path(argument).resolve() == Path(__file__).resolve() for argument in command
    ):
        raise HarnessError("active lease command does not name this exact harness")
    command_config = _command_option(command, "--config")
    if command_config != str(config_path):
        raise HarnessError("active lease command uses a different config path")
    command_lease = _command_option(command, "--lease-path")
    if command_lease is None:
        if lease_path != _DEFAULT_LEASE_PATH:
            raise HarnessError("nonstandard lease path must be explicit in the command")
    elif command_lease != str(lease_path):
        raise HarnessError("active lease command uses a different lease path")
    command_lock = _command_option(command, "--lock-path")
    if command_lock is None:
        if lock_path != _DEFAULT_LOCK_PATH:
            raise HarnessError("nonstandard lock path must be explicit in the command")
    elif command_lock != str(lock_path):
        raise HarnessError("active lease command uses a different lock path")
    command_result = _command_option(command, "--result-dir")
    if command_result is not None and command_result != config.result_directory:
        raise HarnessError("active lease command uses a different result directory")

    if record.get("run_id") != config.run_id:
        raise HarnessError("active lease run_id differs from the config")
    if record.get("exo_namespace") != config.namespace:
        raise HarnessError("active lease namespace differs from the config")
    if record.get("ports") != [port for port in config.reserved_ports]:
        raise HarnessError("active lease reserved ports differ from the config")
    if record.get("result_directory") != config.result_directory:
        raise HarnessError("active lease result directory differs from the config")
    if record.get("wrapper_pid") != os.getppid():
        raise HarnessError("active lease wrapper_pid is not this process's parent")
    if record.get("child_cleanup_confirmation_required") is not True:
        raise HarnessError("active lease does not require child cleanup confirmation")
    cleanup_grace = record.get("cleanup_grace_seconds")
    if (
        not isinstance(cleanup_grace, int | float)
        or isinstance(cleanup_grace, bool)
        or not math.isfinite(cleanup_grace)
    ):
        raise HarnessError("active lease cleanup grace is missing")
    worst_case_cleanup = minimum_cleanup_grace_seconds(config)
    if cleanup_grace < worst_case_cleanup:
        raise HarnessError(
            "active lease cleanup grace is shorter than the proof cleanup bound"
        )
    heartbeat = _parse_lease_timestamp(record.get("heartbeat"), "lease heartbeat")
    now = datetime.now(timezone.utc)
    if heartbeat < now - timedelta(minutes=2) or heartbeat > now + timedelta(minutes=1):
        raise HarnessError("active lease heartbeat is stale or in the future")

    metadata = _object(record.get("metadata"), "active lease metadata")
    generated_at = _parse_lease_timestamp(
        metadata.get("generated_at"), "lease metadata generated_at"
    )
    if generated_at < now - _LEASE_METADATA_MAX_AGE or generated_at > now + timedelta(
        minutes=1
    ):
        raise HarnessError("active lease metadata is stale or in the future")
    expected_metadata = _lease_static_metadata(config, command)
    for field_name, expected_value in expected_metadata.items():
        if metadata.get(field_name) != expected_value:
            raise HarnessError(
                f"active lease metadata.{field_name} differs from the proof config"
            )
    return record


def _tagged(value: JsonValue, expected_tag: str, description: str) -> JsonObject:
    outer = _object(value, description)
    if set(outer) != {expected_tag}:
        raise HarnessError(f"{description} must contain only tag {expected_tag}")
    return _object(outer[expected_tag], f"{description}.{expected_tag}")


def _assignment_object(instance: JsonObject) -> JsonObject:
    return _object(instance.get("shardAssignments"), "instance.shardAssignments")


def validate_and_patch_placement(
    placement: JsonValue,
    config: HarnessConfig,
    runtime_node_ids: Mapping[str, str],
) -> tuple[JsonObject, str, tuple[str, ...], tuple[str, ...]]:
    """Validate the exact 2+1 topology and change only the NCCL port."""
    outer = _object(placement, "placement")
    inner = _tagged(outer, "MlxNcclInstance", "placement")
    instance_id = _string(inner.get("instanceId"), "instance ID")
    assignments = _assignment_object(inner)
    if assignments.get("modelId") != config.model.model_id:
        raise HarnessError("placement model ID does not match the pinned model")

    runners = _object(assignments.get("runnerToShard"), "runnerToShard")
    nodes = _object(assignments.get("nodeToRunner"), "nodeToRunner")
    resource_to_runner = _object(
        assignments.get("computeResourceToRunner"), "computeResourceToRunner"
    )
    resource_to_node = _object(
        assignments.get("computeResourceToNode"), "computeResourceToNode"
    )
    expected_host_names = {host.name for host in config.hosts}
    if set(runtime_node_ids) != expected_host_names:
        raise HarnessError("runtime node map does not contain exactly the two hosts")
    if len(set(runtime_node_ids.values())) != 2:
        raise HarnessError("runtime node IDs must be unique")
    if set(nodes) != set(runtime_node_ids.values()):
        raise HarnessError("placement does not use the two discovered node IDs")
    node_representatives = {
        node_id: _string(value, f"representative for node {node_id}")
        for node_id, value in nodes.items()
    }
    if len(runners) != 3 or len(set(node_representatives.values())) != 2:
        raise HarnessError("placement must contain three ranks and two representatives")

    expected_resources = {
        gpu.resource_id: runtime_node_ids[host.name]
        for host in config.hosts
        for gpu in host.gpus
    }
    if set(resource_to_runner) != set(expected_resources):
        raise HarnessError("placement compute resources do not match configured GPUs")
    if resource_to_node != expected_resources:
        raise HarnessError("placement GPU ownership does not match the 2+1 host layout")
    if set(resource_to_runner.values()) != set(runners):
        raise HarnessError("placement must bind exactly one compute resource per rank")
    for node_id, representative in node_representatives.items():
        if (
            resource_to_node.get(
                next(
                    (
                        resource_id
                        for resource_id, runner_id in resource_to_runner.items()
                        if runner_id == representative
                    ),
                    "",
                )
            )
            != node_id
        ):
            raise HarnessError("node representative must own its configured GPU")

    ranks: set[int] = set()
    for runner_id, tagged_shard in runners.items():
        shard = _tagged(tagged_shard, "TensorShardMetadata", f"shard {runner_id}")
        card = _object(shard.get("modelCard"), f"shard {runner_id} modelCard")
        if card.get("modelId") != config.model.model_id:
            raise HarnessError(f"runner {runner_id} has the wrong model ID")
        if card.get("revision") != config.model.revision:
            raise HarnessError(f"runner {runner_id} has the wrong model revision")
        rank = shard.get("deviceRank")
        world_size = shard.get("worldSize")
        if not isinstance(rank, int) or isinstance(rank, bool) or world_size != 3:
            raise HarnessError(f"runner {runner_id} has invalid tensor rank metadata")
        if shard.get("startLayer") != 0 or shard.get("endLayer") != shard.get(
            "nLayers"
        ):
            raise HarnessError(f"runner {runner_id} is not a full-layer tensor shard")
        ranks.add(rank)
    if ranks != {0, 1, 2}:
        raise HarnessError("tensor ranks must be contiguous 0, 1, 2")

    coordinator = _object(inner.get("ncclCoordinator"), "ncclCoordinator")
    _string(coordinator.get("ip"), "NCCL coordinator IP")
    patched = copy.deepcopy(outer)
    patched_inner = _tagged(patched, "MlxNcclInstance", "patched placement")
    patched_coordinator = _object(
        patched_inner.get("ncclCoordinator"), "patched ncclCoordinator"
    )
    patched_coordinator["port"] = config.nccl_coordinator_port
    return (
        patched,
        instance_id,
        tuple(runners),
        tuple(expected_resources),
    )


def validate_cluster_inventory(
    resources_value: JsonValue, backends_value: JsonValue, config: HarnessConfig
) -> dict[str, str]:
    resources_by_node = _object(resources_value, "node compute resources")
    backends_by_node = _object(backends_value, "node backends")
    if len(resources_by_node) != 2 or set(backends_by_node) != set(resources_by_node):
        raise HarnessError("cluster compute inventory must contain exactly two nodes")
    expected_by_host = {
        host.name: {
            (gpu.device_uuid, gpu.pci_bus_id, gpu.model_name) for gpu in host.gpus
        }
        for host in config.hosts
    }
    runtime_node_ids: dict[str, str] = {}
    for node_id, raw_resources in resources_by_node.items():
        if not isinstance(raw_resources, list):
            raise HarnessError(f"compute resources for node {node_id} must be a list")
        observed: set[tuple[str, str, str]] = set()
        for raw_resource in raw_resources:
            resource = _tagged(
                raw_resource,
                "NvidiaGpuComputeResource",
                f"compute resource on node {node_id}",
            )
            observed.add(
                (
                    _string(resource.get("deviceUuid"), "GPU UUID"),
                    _string(resource.get("pciBusId"), "GPU PCI bus ID"),
                    _string(resource.get("modelName"), "GPU model"),
                )
            )
        matching_hosts = [
            host_name
            for host_name, expected in expected_by_host.items()
            if observed == expected
        ]
        if len(matching_hosts) != 1:
            raise HarnessError(
                f"node {node_id} GPU inventory does not identify exactly one host"
            )
        host_name = matching_hosts[0]
        if host_name in runtime_node_ids:
            raise HarnessError(f"GPU inventory mapped two nodes to {host_name}")
        runtime_node_ids[host_name] = node_id
        node_backends = backends_by_node.get(node_id)
        if not isinstance(node_backends, list) or "MlxCuda" not in node_backends:
            raise HarnessError(f"{host_name} does not advertise MlxCuda")
    if set(runtime_node_ids) != set(expected_by_host):
        raise HarnessError("cluster inventory did not identify both configured hosts")
    return runtime_node_ids


def _numeric_version(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is None:
        raise HarnessError(f"invalid dotted numeric version: {value}")
    return tuple(int(component) for component in value.split("."))


def _cuda_driver_major_from_nvidia_smi(banner: str) -> int | None:
    match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+)(?:\.[0-9]+)?", banner)
    return None if match is None else int(match.group(1))


def validated_runtime_identity(
    report: HostPreflightReport, host: HostConfig, config: HarnessConfig
) -> JsonObject:
    runtime_versions = report.facts.get("runtime_versions")
    if not isinstance(runtime_versions, dict):
        raise HarnessError(f"runtime versions are missing on {host.name}")
    cuda_distribution = f"mlx-cuda-{config.runtime.cuda_major}"
    nccl_distribution = f"nvidia-nccl-cu{config.runtime.cuda_major}"
    required_distributions = ("exo", "mlx", cuda_distribution, nccl_distribution)
    full_python_version = runtime_versions.get("python")
    if not isinstance(full_python_version, str) or not full_python_version:
        raise HarnessError(f"required runtime python is missing on {host.name}")
    for distribution in required_distributions:
        version = runtime_versions.get(distribution)
        if not isinstance(version, str) or not version:
            raise HarnessError(
                f"required runtime {distribution} is missing on {host.name}"
            )

    raw_python_abi = report.facts.get("python_abi")
    if not isinstance(raw_python_abi, dict):
        raise HarnessError(f"Python ABI identity is missing on {host.name}")
    try:
        python_abi = PythonAbiIdentity.model_validate(raw_python_abi)
    except ValidationError as error:
        raise HarnessError(f"Python ABI identity is invalid on {host.name}") from error
    if python_abi != config.runtime.python_abi:
        raise HarnessError(
            f"Python ABI identity does not match the configured ABI on {host.name}"
        )

    artifact = report.facts.get("exo_rs_artifact")
    if not isinstance(artifact, dict):
        raise HarnessError(f"exo_rs identity is missing on {host.name}")
    if artifact.get("module") != "exo_rs.exo_rs":
        raise HarnessError(f"exo_rs native module identity is invalid on {host.name}")
    if artifact.get("is_native_extension") is not True:
        raise HarnessError(f"exo_rs artifact is not a native extension on {host.name}")
    artifact_path = artifact.get("path")
    expected_extension_suffix = f".{python_abi.soabi}.so"
    if (
        not isinstance(artifact_path, str)
        or not Path(artifact_path).is_absolute()
        or not Path(artifact_path).name.endswith(expected_extension_suffix)
    ):
        raise HarnessError(
            f"exo_rs native extension path does not match the Python ABI on {host.name}"
        )
    artifact_sha = artifact.get("sha256")
    if (
        not isinstance(artifact_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_sha) is None
    ):
        raise HarnessError(f"exo_rs SHA-256 is missing on {host.name}")
    expected_artifact_sha = config.runtime.host_pins[host.name].exo_rs_native_sha256
    if artifact_sha != expected_artifact_sha:
        raise HarnessError(f"exo_rs SHA-256 does not match the host pin on {host.name}")

    origin = report.facts.get("exo_import_origin")
    if not isinstance(origin, str) or not Path(origin).is_absolute():
        raise HarnessError(f"exo import origin is missing on {host.name}")
    source_directory = Path(host.source_directory).resolve()
    resolved_origin = Path(origin).resolve()
    if not resolved_origin.is_relative_to(source_directory):
        raise HarnessError(
            f"exo import origin is outside the configured source on {host.name}"
        )

    driver_versions = report.facts.get("nvidia_driver_versions")
    if not isinstance(driver_versions, str) or not driver_versions:
        raise HarnessError(f"NVIDIA driver version is missing on {host.name}")
    minimum_driver = _numeric_version(config.runtime.minimum_nvidia_driver_version)
    for driver_version in driver_versions.split(","):
        if _numeric_version(driver_version) < minimum_driver:
            raise HarnessError(
                f"NVIDIA driver {driver_version} is below the configured minimum "
                f"on {host.name}"
            )
    banner = report.facts.get("nvidia_smi_banner")
    if not isinstance(banner, str) or (
        _cuda_driver_major_from_nvidia_smi(banner) != config.runtime.cuda_major
    ):
        raise HarnessError(
            f"nvidia-smi does not advertise CUDA {config.runtime.cuda_major} on "
            f"{host.name}"
        )
    relative_origin = str(resolved_origin.relative_to(source_directory))
    runtime_versions_json: JsonObject = {
        distribution: runtime_versions[distribution]
        for distribution in required_distributions
    }
    return {
        "runtime_versions": runtime_versions_json,
        "python_abi": cast(JsonValue, python_abi.model_dump(mode="json")),
        "exo_rs_module": "exo_rs.exo_rs",
        "exo_import_relative_origin": relative_origin,
        "cuda_major": config.runtime.cuda_major,
    }


def validate_preflight(
    report: HostPreflightReport,
    model: ModelProbeResult,
    host: HostConfig,
    config: HarnessConfig,
) -> None:
    if report.run_id != config.run_id or report.host_name != host.name:
        raise HarnessError(f"preflight identity mismatch for {host.name}")
    if report.source_commit != host.source.commit:
        raise HarnessError(f"source commit mismatch for {host.name}")
    if report.dirty_file_hashes != host.source.dirty_file_hashes:
        raise HarnessError(f"dirty source hashes mismatch for {host.name}")
    if set(report.checked_tcp_ports) != set(config.reserved_ports):
        raise HarnessError(f"preflight did not check every TCP port on {host.name}")
    if set(report.checked_udp_ports) != set(config.reserved_ports):
        raise HarnessError(f"preflight did not check every UDP port on {host.name}")
    observed_gpus = {
        (gpu.device_uuid, gpu.pci_bus_id, gpu.model_name) for gpu in report.gpus
    }
    expected_gpus = {
        (gpu.device_uuid, gpu.pci_bus_id, gpu.model_name) for gpu in host.gpus
    }
    if observed_gpus != expected_gpus:
        raise HarnessError(f"preflight GPU identity mismatch for {host.name}")
    observed_hca_ports = {(port.device, port.port) for port in report.hca_ports}
    if observed_hca_ports != {(port.device, port.port) for port in host.hca_ports}:
        raise HarnessError(f"preflight HCA selection mismatch for {host.name}")
    if any(not _ib_state_is(port.state, "ACTIVE") for port in report.hca_ports):
        raise HarnessError(
            f"one or more selected IB ports are not ACTIVE on {host.name}"
        )
    if any(
        not _ib_state_is(port.physical_state, "LINKUP")
        or port.link_layer.lower() != "infiniband"
        or port.lid in {"unknown", "0", "0x0", "0x0000"}
        or not port.gids
        or not port.counters
        or configured.gid not in port.gids
        for port, configured in zip(report.hca_ports, host.hca_ports, strict=True)
    ):
        raise HarnessError(f"InfiniBand detail is incomplete on {host.name}")
    if not {"amx_bf16", "amx_int8", "amx_tile"}.issubset(report.amx_flags):
        raise HarnessError(f"AMX capability was not verified on {host.name}")
    if not report.passed or report.conflicts:
        conflicts = "; ".join(report.conflicts) or "unspecified preflight failure"
        raise HarnessError(f"preflight failed on {host.name}: {conflicts}")
    for fact_name in (
        "load_average",
        "memory",
        "kernel",
        "gpu_telemetry_csv",
        "nvidia_smi_banner",
        "nvidia_driver_versions",
        "ip_addresses_json",
        "ip_routes_json",
        "exo_import_origin",
    ):
        fact = report.facts.get(fact_name)
        if not isinstance(fact, str) or not fact:
            raise HarnessError(f"preflight fact {fact_name} is missing on {host.name}")
    for fact_name in (
        "cpu_frequency_policy",
        "runtime_versions",
        "python_abi",
        "exo_rs_artifact",
    ):
        fact = report.facts.get(fact_name)
        if not isinstance(fact, dict) or not fact:
            raise HarnessError(f"preflight fact {fact_name} is missing on {host.name}")
    validated_runtime_identity(report, host, config)
    if (
        not model.verified
        or model.host_name != host.name
        or model.path != host.model_path
        or model.model_id != config.model.model_id
        or model.revision != config.model.revision
        or model.weight_bytes != config.model.expected_weight_bytes
        or model.physical_weight_bytes <= 0
        or model.weight_files < 1
        or not model.sha256_manifest
    ):
        raise HarnessError(f"exact model snapshot verification failed on {host.name}")


class HarnessEffects(Protocol):
    def run_preflight(
        self, host: HostConfig, config: HarnessConfig
    ) -> HostPreflightReport: ...

    def probe_model(
        self, host: HostConfig, model: ModelSnapshot
    ) -> ModelProbeResult: ...

    def start_node(
        self, host: HostConfig, config: HarnessConfig, owner_token: str
    ) -> OwnedProcess: ...

    def process_alive(self, process: OwnedProcess) -> bool: ...

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup: ...

    def read_owned_log(self, process: OwnedProcess) -> str: ...

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def write_result_json(self, filename: str, value: JsonObject) -> None: ...


class SignalAwareEffects:
    def __init__(self, effects: HarnessEffects, latch: SignalLatch) -> None:
        self._effects = effects
        self._latch = latch

    def run_preflight(
        self, host: HostConfig, config: HarnessConfig
    ) -> HostPreflightReport:
        self._latch.checkpoint()
        result = self._effects.run_preflight(host, config)
        self._latch.checkpoint()
        return result

    def probe_model(self, host: HostConfig, model: ModelSnapshot) -> ModelProbeResult:
        self._latch.checkpoint()
        result = self._effects.probe_model(host, model)
        self._latch.checkpoint()
        return result

    def start_node(
        self, host: HostConfig, config: HarnessConfig, owner_token: str
    ) -> OwnedProcess:
        self._latch.checkpoint()
        return self._effects.start_node(host, config, owner_token)

    def process_alive(self, process: OwnedProcess) -> bool:
        self._latch.checkpoint()
        result = self._effects.process_alive(process)
        self._latch.checkpoint()
        return result

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        return self._effects.stop_node(process, timeout_seconds)

    def read_owned_log(self, process: OwnedProcess) -> str:
        return self._effects.read_owned_log(process)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue:
        self._latch.checkpoint()
        result = self._effects.request_json(method, path, params=params, body=body)
        self._latch.checkpoint()
        return result

    def monotonic(self) -> float:
        return self._effects.monotonic()

    def sleep(self, seconds: float) -> None:
        self._effects.sleep(seconds)
        self._latch.checkpoint()

    def write_result_json(self, filename: str, value: JsonObject) -> None:
        self._effects.write_result_json(filename, value)


_MODEL_PROBE_PROGRAM = r"""
import json
import sys
from pathlib import Path

host_name, raw_path, model_id, revision, raw_expected_bytes = sys.argv[1:]
path = Path(raw_path)
expected_bytes = int(raw_expected_bytes)
result = {
    "host_name": host_name,
    "path": raw_path,
    "model_id": model_id,
    "revision": revision,
    "weight_bytes": 0,
    "physical_weight_bytes": 0,
    "weight_files": 0,
    "sha256_manifest": {},
    "receipt_kind": "exo",
    "verified": False,
    "error": None,
}
try:
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError("model path must be absolute, canonical, and not a symlink")
    if not path.is_dir():
        raise RuntimeError("model path is not an absolute directory")
    if not (path / "config.json").is_file():
        raise RuntimeError("model config.json is missing")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise RuntimeError("model snapshot contains a symlink")
    receipt_path = path / ".exo-huggingface-revision.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if set(receipt) != {"repo_id", "revision"}:
            raise RuntimeError("Exo revision receipt has an invalid schema")
        if receipt["repo_id"] != model_id or receipt["revision"] != revision:
            raise RuntimeError("Exo revision receipt does not match model")
        receipt_kind = "exo"
    else:
        receipt_kind = "huggingface"
        metadata_root = path / ".cache" / "huggingface" / "download"
        model_files = []
        for item in path.rglob("*"):
            relative = item.relative_to(path)
            if relative.parts and relative.parts[0] == ".cache":
                continue
            if item.is_file():
                model_files.append(relative)
        if not model_files:
            raise RuntimeError("model snapshot has no downloaded files")
        for relative in model_files:
            metadata = metadata_root / f"{relative}.metadata"
            if not metadata.is_file():
                raise RuntimeError(f"missing Hugging Face metadata for {relative}")
            lines = metadata.read_text().splitlines()
            if not lines or lines[0] != revision:
                raise RuntimeError(f"wrong Hugging Face revision for {relative}")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("proof checkpoint requires model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise RuntimeError("safetensors index is missing metadata or weight_map")
    indexed_weight_bytes = metadata.get("total_size")
    if not isinstance(indexed_weight_bytes, int) or isinstance(indexed_weight_bytes, bool):
        raise RuntimeError("safetensors index total_size must be an integer")
    if indexed_weight_bytes != expected_bytes:
        raise RuntimeError(
            f"indexed weight bytes {indexed_weight_bytes} do not match expected {expected_bytes}"
        )
    shard_names = sorted(set(weight_map.values()))
    if not shard_names or not all(isinstance(name, str) for name in shard_names):
        raise RuntimeError("safetensors index has no valid shard targets")
    weights = []
    for shard_name in shard_names:
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe weight shard path {shard_name}")
        shard_path = path / relative
        if not shard_path.is_file() or shard_path.is_symlink():
            raise RuntimeError(f"referenced weight shard is missing: {shard_name}")
        weights.append(shard_path)
    snapshot_files = []
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if item.is_file() and item != receipt_path:
            snapshot_files.append(item)
    indexed_weights = {item.resolve() for item in weights}
    unexpected_weights = [
        str(item.relative_to(path))
        for item in snapshot_files
        if item.suffix == ".safetensors" and item.resolve() not in indexed_weights
    ]
    if unexpected_weights:
        raise RuntimeError(
            f"unindexed safetensors files are present: {sorted(unexpected_weights)}"
        )
    physical_weight_bytes = sum(item.stat().st_size for item in weights)

    def sha256_file(file_path):
        digest = __import__("hashlib").sha256()
        with file_path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    manifest_paths = sorted(snapshot_files)
    sha256_manifest = {
        str(item.relative_to(path)): sha256_file(item) for item in manifest_paths
    }
    result.update({
        "weight_bytes": indexed_weight_bytes,
        "physical_weight_bytes": physical_weight_bytes,
        "weight_files": len(weights),
        "sha256_manifest": sha256_manifest,
        "receipt_kind": receipt_kind,
        "verified": True,
    })
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, sort_keys=True))
"""


_REMOTE_LAUNCH_PROGRAM = r"""
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

owner_token, namespace, working_directory, raw_environment, raw_argv = sys.argv[1:]
environment = json.loads(raw_environment)
arguments = json.loads(raw_argv)
child_environment = dict(environment)
child_environment["EXO_BENCHMARK_OWNER_TOKEN"] = owner_token
child = None
receipt = None
supervisor_error = None
pump = None

def pump_output(source):
    for line in source:
        sys.stdout.write(line)
        sys.stdout.flush()

def identity(process_pid):
    fields = Path(f"/proc/{process_pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[2]), int(fields[19])

def ownership_matches():
    if child is None or receipt is None or child.poll() is not None:
        return True
    process_group_id, start_time_ticks = identity(child.pid)
    environment_entries = Path(f"/proc/{child.pid}/environ").read_bytes().split(b"\0")
    owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={owner_token}".encode()
    return (
        process_group_id == child.pid == receipt["process_group_id"]
        and start_time_ticks == receipt["start_time_ticks"]
        and owner_entry in environment_entries
    )

def terminate_child(timeout):
    if child is None or child.poll() is not None:
        return
    if not ownership_matches():
        raise RuntimeError("remote supervisor refused to signal an unowned child")
    os.killpg(receipt["process_group_id"], signal.SIGTERM)
    try:
        child.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    if not ownership_matches():
        raise RuntimeError("remote child ownership changed before SIGKILL")
    os.killpg(receipt["process_group_id"], signal.SIGKILL)
    child.wait(timeout=min(timeout, 5.0))

print("EXO_SUPERVISOR_READY", flush=True)
try:
    if sys.stdin.readline().strip() != "START":
        raise RuntimeError("launch acknowledgement was not received")
    child = subprocess.Popen(
        arguments,
        cwd=working_directory,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process_group_id, start_time_ticks = identity(child.pid)
    if process_group_id != child.pid:
        raise RuntimeError("remote Exo child is not its process-group leader")
    receipt = {
        "pid": child.pid,
        "process_group_id": process_group_id,
        "start_time_ticks": start_time_ticks,
        "owner_token": owner_token,
        "namespace": namespace,
    }
    print("EXO_OWNER " + json.dumps(receipt, sort_keys=True), flush=True)
    assert child.stdout is not None
    pump = threading.Thread(target=pump_output, args=(child.stdout,), daemon=True)
    pump.start()
    while child.poll() is None:
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if readable and sys.stdin.readline() == "":
            terminate_child(10.0)
            break
except BaseException as error:
    supervisor_error = error
finally:
    if child is not None and child.poll() is None:
        terminate_child(10.0)
if child is not None and child.poll() is None:
    raise RuntimeError("remote child survived supervisor cleanup")
if pump is not None:
    pump.join(timeout=5.0)
    if pump.is_alive():
        raise RuntimeError("remote child output pump did not finish")
print("EXO_SUPERVISOR_LOG_COMPLETE", flush=True)
print("EXO_SUPERVISOR_CLEAN", flush=True)
if supervisor_error is not None:
    raise supervisor_error
raise SystemExit(0 if child is None or child.returncode is None else child.returncode)
"""


_REMOTE_STOP_PROGRAM = r"""
import json
import os
import signal
import sys
import time
from pathlib import Path

receipt = json.loads(sys.argv[1])
timeout = float(sys.argv[2])
pid = int(receipt["pid"])
result = {
    "host_name": receipt["host_name"],
    "ownership_verified": False,
    "terminated": False,
    "forced": False,
    "error": None,
}
try:
    process_group_id = int(receipt["process_group_id"])
    owner_entry = (
        "EXO_BENCHMARK_OWNER_TOKEN=" + receipt["owner_token"]
    ).encode()

    def group_members():
        members = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                state = fields[0]
                observed_group = int(fields[2])
                start_ticks = int(fields[19])
            except (OSError, IndexError, ValueError):
                continue
            if observed_group != process_group_id:
                continue
            member_pid = int(entry.name)
            members.append(member_pid)
            if member_pid == pid and start_ticks != int(receipt["start_time_ticks"]):
                raise RuntimeError("PID start time no longer matches ownership receipt")
            if state == "Z":
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
            if owner_entry not in environment:
                raise RuntimeError(
                    f"process-group member {member_pid} has a different owner token"
                )
            if member_pid == pid:
                command_line = (entry / "cmdline").read_bytes().replace(
                    b"\0", b" "
                ).decode("utf-8", errors="replace")
                if receipt["namespace"] not in command_line:
                    raise RuntimeError(
                        "process command line no longer has owned namespace"
                    )
        return members

    def wait_group_empty(wait_seconds):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if not group_members():
                return True
            time.sleep(0.1)
        return not group_members()

    members = group_members()
    if not members:
        result.update({"ownership_verified": True, "terminated": True})
    else:
        if process_group_id != pid:
            raise RuntimeError("owned process is no longer its process-group leader")
        result["ownership_verified"] = True
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not wait_group_empty(timeout):
            result["forced"] = True
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result["terminated"] = wait_group_empty(min(timeout, 5.0))
        else:
            result["terminated"] = True
        if not result["terminated"]:
            result["error"] = "owned process group still exists after SIGKILL"
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, sort_keys=True))
"""


@dataclass
class _RunningHandle:
    transport_process: subprocess.Popen[str]
    log_file: IO[str]
    pump_thread: threading.Thread | None = None


class SystemEffects:
    """Default network/process adapter; tests use an in-memory implementation."""

    def __init__(
        self, config: HarnessConfig, result_directory_descriptor: int | None = None
    ) -> None:
        self._config = config
        self._running: dict[str, _RunningHandle] = {}
        self._result_directory = Path(config.result_directory)
        self._result_directory_descriptor = (
            _open_directory_without_symlinks(self._result_directory)
            if result_directory_descriptor is None
            else result_directory_descriptor
        )
        _validate_result_directory_descriptor(
            self._result_directory, self._result_directory_descriptor
        )

    @staticmethod
    def _transport_argv(host: HostConfig, command: Sequence[str]) -> list[str]:
        if host.transport == "local":
            return list(command)
        assert host.ssh_target is not None
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "--",
            host.ssh_target,
            shlex.join(command),
        ]

    @staticmethod
    def _exact_environment_command(
        host: HostConfig, command: Sequence[str]
    ) -> tuple[str, ...]:
        assignments = tuple(
            f"{name}={value}" for name, value in sorted(host.environment.items())
        )
        return ("/usr/bin/env", "-i", *assignments, *command)

    def _run_text_command(
        self,
        host: HostConfig,
        command: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float,
    ) -> str:
        transported_command: Sequence[str] = command
        if host.transport == "ssh":
            transported_command = self._exact_environment_command(host, command)
        client_environment = dict(host.environment)
        completed = subprocess.run(
            self._transport_argv(host, transported_command),
            check=False,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=client_environment,
        )
        if completed.returncode != 0:
            raise HarnessError(
                f"command failed on {host.name} with {completed.returncode}: "
                f"{completed.stderr[-1000:]}"
            )
        return completed.stdout

    def run_preflight(
        self, host: HostConfig, config: HarnessConfig
    ) -> HostPreflightReport:
        request = HostPreflightRequest(
            schema_version=1,
            run_id=config.run_id,
            host=host,
            reserved_ports=config.reserved_ports,
        )
        script_path = Path(host.source_directory) / "scripts" / Path(__file__).name
        command = (host.python_executable, str(script_path), "host-preflight")
        output = self._run_text_command(
            host,
            command,
            stdin=request.model_dump_json(),
            timeout=config.timeouts.api_start_seconds,
        )
        try:
            return HostPreflightReport.model_validate_json(output)
        except ValidationError as error:
            raise HarnessError(
                f"invalid preflight JSON returned by {host.name}: {error}"
            ) from error

    def probe_model(self, host: HostConfig, model: ModelSnapshot) -> ModelProbeResult:
        command = (
            host.python_executable,
            "-c",
            _MODEL_PROBE_PROGRAM,
            host.name,
            host.model_path,
            model.model_id,
            model.revision,
            str(model.expected_weight_bytes),
        )
        output = self._run_text_command(
            host,
            command,
            timeout=self._config.timeouts.api_start_seconds,
        )
        try:
            return ModelProbeResult.model_validate_json(output)
        except ValidationError as error:
            raise HarnessError(
                f"invalid model probe JSON returned by {host.name}: {error}"
            ) from error

    @staticmethod
    def _read_process_identity(pid: int) -> tuple[int, int]:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
        fields_after_name = stat_text.rsplit(")", 1)[1].split()
        return int(fields_after_name[2]), int(fields_after_name[19])

    @staticmethod
    def _read_start_marker(
        process: subprocess.Popen[str], timeout_seconds: float
    ) -> str:
        if process.stdout is None:
            raise HarnessError("remote launch has no stdout ownership channel")
        ready, _, _ = select.select([process.stdout], [], [], timeout_seconds)
        if not ready:
            raise HarnessError("remote launch did not emit an ownership receipt")
        line = process.stdout.readline()
        if not line:
            raise HarnessError("remote launch closed before ownership receipt")
        return line.rstrip("\n")

    @staticmethod
    def _pump_output(source: IO[str], destination: IO[str]) -> None:
        for line in source:
            destination.write(line)
            destination.flush()

    def _terminate_unregistered_transport(
        self,
        host: HostConfig,
        process: subprocess.Popen[str],
        *,
        owner_token: str,
        namespace: str,
        log_path: Path,
        timeout_seconds: float,
    ) -> ProcessCleanup:
        if host.transport == "ssh":
            forced = False
            try:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    forced = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=2.0)
                remaining_output = ""
                if process.stdout is not None:
                    remaining_output = process.stdout.read()
                confirmed = "EXO_SUPERVISOR_CLEAN" in remaining_output.splitlines()
                return ProcessCleanup(
                    host_name=host.name,
                    ownership_verified=confirmed,
                    terminated=confirmed and process.poll() is not None,
                    forced=forced,
                    error=(
                        None
                        if confirmed
                        else "remote supervisor did not confirm child cleanup"
                    ),
                )
            except BaseException as error:
                return ProcessCleanup(
                    host.name,
                    False,
                    False,
                    forced,
                    f"{type(error).__name__}: {error}",
                )

        try:
            if process.poll() is not None:
                process.wait(timeout=1.0)
                return ProcessCleanup(host.name, True, True, False)
            process_group_id, start_ticks = self._read_process_identity(process.pid)
            receipt = OwnedProcess(
                host_name=host.name,
                pid=process.pid,
                process_group_id=process_group_id,
                start_time_ticks=start_ticks,
                owner_token=owner_token,
                namespace=namespace,
                transport_pid=process.pid,
                log_path=str(log_path),
            )
            if process_group_id != process.pid:
                raise HarnessError("unregistered local process is not its group leader")
            ownership_matches, members = self._local_group_ownership(receipt)
            if not ownership_matches:
                raise HarnessError("unregistered local process ownership is unverified")
            if not members:
                process.wait(timeout=1.0)
                return ProcessCleanup(host.name, True, True, False)
            os.killpg(process_group_id, signal.SIGTERM)
            if self._wait_local_group_gone(receipt, process, timeout_seconds):
                process.wait(timeout=1.0)
                return ProcessCleanup(host.name, True, True, False)
            ownership_matches, _ = self._local_group_ownership(receipt)
            if not ownership_matches:
                raise HarnessError("local process ownership changed before SIGKILL")
            os.killpg(process_group_id, signal.SIGKILL)
            terminated = self._wait_local_group_gone(
                receipt, process, min(timeout_seconds, 5.0)
            )
            process.wait(timeout=1.0)
            return ProcessCleanup(
                host.name,
                True,
                terminated,
                True,
                None if terminated else "local process group survived SIGKILL",
            )
        except BaseException as error:
            return ProcessCleanup(
                host.name,
                False,
                False,
                False,
                f"{type(error).__name__}: {error}",
            )

    def start_node(
        self, host: HostConfig, config: HarnessConfig, owner_token: str
    ) -> OwnedProcess:
        if host.name in self._running:
            raise HarnessError(f"node process already started for {host.name}")
        log_path = self._result_directory / f"exo-{host.name}.log"
        log_descriptor = os.open(
            log_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._result_directory_descriptor,
        )
        log_file = os.fdopen(log_descriptor, mode="w", encoding="utf-8")
        environment = dict(host.environment)
        environment["EXO_BENCHMARK_OWNER_TOKEN"] = owner_token
        log_file.write(
            "EXO_POC_LAUNCH "
            + json.dumps(
                {
                    "host_name": host.name,
                    "environment": environment,
                    "argv": list(host.launch_argv),
                },
                sort_keys=True,
            )
            + "\n"
        )
        log_file.flush()
        started_process: subprocess.Popen[str] | None = None
        owned_receipt: OwnedProcess | None = None
        try:
            if host.transport == "local":
                process = subprocess.Popen(
                    host.launch_argv,
                    cwd=host.source_directory,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                started_process = process
                process_group_id, start_ticks = self._read_process_identity(process.pid)
                if process_group_id != process.pid:
                    raise HarnessError("local node is not its process-group leader")
                receipt = OwnedProcess(
                    host_name=host.name,
                    pid=process.pid,
                    process_group_id=process_group_id,
                    start_time_ticks=start_ticks,
                    owner_token=owner_token,
                    namespace=config.namespace,
                    transport_pid=process.pid,
                    log_path=str(log_path),
                )
                owned_receipt = receipt
                self._running[host.name] = _RunningHandle(process, log_file)
                return receipt

            assert host.ssh_target is not None
            remote_command = (
                host.python_executable,
                "-c",
                _REMOTE_LAUNCH_PROGRAM,
                owner_token,
                config.namespace,
                host.source_directory,
                json.dumps(host.environment, sort_keys=True),
                json.dumps(host.launch_argv),
            )
            process = subprocess.Popen(
                self._transport_argv(host, remote_command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            started_process = process
            supervisor_marker = self._read_start_marker(
                process, config.timeouts.process_start_seconds
            )
            if supervisor_marker != "EXO_SUPERVISOR_READY":
                raise HarnessError(
                    f"invalid supervisor marker from {host.name}: {supervisor_marker}"
                )
            if process.stdin is None:
                raise HarnessError("remote launch has no acknowledgement channel")
            process.stdin.write("START\n")
            process.stdin.flush()
            marker = self._read_start_marker(
                process, config.timeouts.process_start_seconds
            )
            if not marker.startswith("EXO_OWNER "):
                raise HarnessError(
                    f"invalid ownership receipt from {host.name}: {marker}"
                )
            raw_receipt = RemoteOwnerReceipt.model_validate_json(
                marker.removeprefix("EXO_OWNER ")
            )
            pid = raw_receipt.pid
            process_group_id = raw_receipt.process_group_id
            start_ticks = raw_receipt.start_time_ticks
            if process_group_id != pid:
                raise HarnessError("remote node is not its process-group leader")
            if raw_receipt.owner_token != owner_token:
                raise HarnessError("remote ownership token mismatch")
            if raw_receipt.namespace != config.namespace:
                raise HarnessError("remote ownership namespace mismatch")
            receipt = OwnedProcess(
                host_name=host.name,
                pid=pid,
                process_group_id=process_group_id,
                start_time_ticks=start_ticks,
                owner_token=owner_token,
                namespace=config.namespace,
                transport_pid=process.pid,
                log_path=str(log_path),
            )
            owned_receipt = receipt
            self._running[host.name] = _RunningHandle(process, log_file)
            assert process.stdout is not None
            pump = threading.Thread(
                target=self._pump_output,
                args=(process.stdout, log_file),
                daemon=True,
                name=f"exo-log-{host.name}",
            )
            pump.start()
            self._running[host.name].pump_thread = pump
            return receipt
        except BaseException as error:
            if owned_receipt is not None and host.name in self._running:
                try:
                    cleanup = self.stop_node(
                        owned_receipt, config.timeouts.cleanup_seconds
                    )
                except BaseException as cleanup_error:
                    cleanup = ProcessCleanup(
                        host.name,
                        False,
                        False,
                        False,
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    )
            elif started_process is not None:
                cleanup = self._terminate_unregistered_transport(
                    host,
                    started_process,
                    owner_token=owner_token,
                    namespace=config.namespace,
                    log_path=log_path,
                    timeout_seconds=config.timeouts.cleanup_seconds,
                )
                log_file.close()
            else:
                log_file.close()
                cleanup = ProcessCleanup(host.name, True, True, False)
            raise StartNodeError(host.name, error, cleanup, owned_receipt) from error

    def process_alive(self, process: OwnedProcess) -> bool:
        handle = self._running.get(process.host_name)
        return handle is not None and handle.transport_process.poll() is None

    @staticmethod
    def _local_group_ownership(
        process: OwnedProcess,
    ) -> tuple[bool, tuple[int, ...]]:
        members: list[int] = []
        leader_seen = False
        owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={process.owner_token}".encode()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat_text = (entry / "stat").read_text()
                fields_after_name = stat_text.rsplit(")", 1)[1].split()
                process_state = fields_after_name[0]
                process_group_id = int(fields_after_name[2])
                start_time_ticks = int(fields_after_name[19])
            except (OSError, IndexError, ValueError):
                continue
            if process_group_id != process.process_group_id:
                continue
            pid = int(entry.name)
            members.append(pid)
            if pid == process.pid:
                leader_seen = True
                if start_time_ticks != process.start_time_ticks:
                    return False, tuple(sorted(members))
            if process_state == "Z":
                continue
            try:
                environment = (entry / "environ").read_bytes().split(b"\0")
            except OSError:
                return False, tuple(sorted(members))
            if owner_entry not in environment:
                return False, tuple(sorted(members))
            if pid == process.pid:
                try:
                    command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
                except OSError:
                    return False, tuple(sorted(members))
                if process.namespace.encode() not in command_line:
                    return False, tuple(sorted(members))
        if leader_seen and process.process_group_id != process.pid:
            return False, tuple(sorted(members))
        return True, tuple(sorted(members))

    def _wait_local_group_gone(
        self,
        process: OwnedProcess,
        transport_process: subprocess.Popen[str],
        timeout_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            transport_process.poll()
            ownership_matches, members = self._local_group_ownership(process)
            if not members:
                return True
            if not ownership_matches:
                return False
            time.sleep(0.1)
        transport_process.poll()
        _, members = self._local_group_ownership(process)
        return not members

    def _stop_local(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        handle = self._running.get(process.host_name)
        if handle is None:
            return ProcessCleanup(
                process.host_name,
                False,
                False,
                False,
                "local process handle is missing",
            )
        handle.transport_process.poll()
        ownership_matches, members = self._local_group_ownership(process)
        if not members:
            return ProcessCleanup(process.host_name, True, True, False)
        if not ownership_matches:
            return ProcessCleanup(
                process.host_name,
                False,
                False,
                False,
                "local process ownership no longer matches the receipt",
            )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.process_group_id, signal.SIGTERM)
        if self._wait_local_group_gone(
            process, handle.transport_process, timeout_seconds
        ):
            return ProcessCleanup(process.host_name, True, True, False)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.process_group_id, signal.SIGKILL)
        terminated = self._wait_local_group_gone(
            process, handle.transport_process, min(timeout_seconds, 5.0)
        )
        return ProcessCleanup(
            process.host_name,
            True,
            terminated,
            True,
            None if terminated else "local process group survived SIGKILL",
        )

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        host = next(
            host for host in self._config.hosts if host.name == process.host_name
        )
        handle = self._running.get(process.host_name)
        cleanup: ProcessCleanup
        try:
            if host.transport == "local":
                cleanup = self._stop_local(process, timeout_seconds)
            else:
                command = (
                    host.python_executable,
                    "-c",
                    _REMOTE_STOP_PROGRAM,
                    json.dumps(asdict(process), sort_keys=True),
                    str(timeout_seconds),
                )
                output = self._run_text_command(
                    host,
                    command,
                    timeout=timeout_seconds + 10.0,
                )
                cleanup_receipt = ProcessCleanupReceipt.model_validate_json(output)
                if cleanup_receipt.host_name != process.host_name:
                    raise HarnessError("remote cleanup host identity mismatch")
                cleanup = ProcessCleanup(
                    host_name=cleanup_receipt.host_name,
                    ownership_verified=cleanup_receipt.ownership_verified,
                    terminated=cleanup_receipt.terminated,
                    forced=cleanup_receipt.forced,
                    error=cleanup_receipt.error,
                )
        except Exception as error:
            cleanup = ProcessCleanup(
                process.host_name,
                False,
                False,
                False,
                f"{type(error).__name__}: {error}",
            )
        finalization_errors: list[str] = []
        if handle is not None:
            try:
                try:
                    handle.transport_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    if handle.transport_process.pid != process.pid:
                        os.killpg(handle.transport_process.pid, signal.SIGTERM)
                        try:
                            handle.transport_process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            os.killpg(handle.transport_process.pid, signal.SIGKILL)
                            handle.transport_process.wait(timeout=2.0)
                    else:
                        finalization_errors.append(
                            "local process transport was not reaped after cleanup"
                        )
            except BaseException as error:
                finalization_errors.append(
                    f"transport finalization failed: {type(error).__name__}: {error}"
                )
            if handle.pump_thread is not None:
                handle.pump_thread.join(timeout=5.0)
                if handle.pump_thread.is_alive():
                    finalization_errors.append(
                        "SSH output pump did not finish; owned log is incomplete"
                    )
                else:
                    handle.log_file.close()
            else:
                handle.log_file.close()
            self._running.pop(process.host_name, None)
        if finalization_errors:
            prior_error = cleanup.error
            combined_error = "; ".join(
                part for part in (prior_error, *finalization_errors) if part
            )
            cleanup = ProcessCleanup(
                host_name=process.host_name,
                ownership_verified=cleanup.ownership_verified,
                terminated=False,
                forced=cleanup.forced,
                error=combined_error,
            )
        return cleanup

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue:
        if not path.startswith("/"):
            path = "/" + path
        if params:
            path = f"{path}?{urlencode(params)}"
        connection = http.client.HTTPConnection(
            self._config.api.host,
            self._config.api.port,
            timeout=self._config.timeouts.request_seconds,
        )
        headers = {"Accept": "application/json"}
        payload: str | None = None
        if body is not None:
            payload = json.dumps(body, sort_keys=True)
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise HttpResponseError(response.status, response.reason, raw)
            if not raw:
                return None
            return _parse_json_value(raw, f"HTTP response from {method} {path}")
        finally:
            connection.close()

    def read_owned_log(self, process: OwnedProcess) -> str:
        path = Path(process.log_path)
        if (
            path.parent != self._result_directory
            or path.name != f"exo-{process.host_name}.log"
        ):
            raise HarnessError("owned log path does not match the result directory")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=self._result_directory_descriptor,
        )
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise HarnessError(f"owned log is not a regular file: {path}")
            if file_status.st_size > 64 * 1024 * 1024:
                raise HarnessError(f"owned log is unexpectedly large: {path}")
            with os.fdopen(
                descriptor,
                mode="r",
                encoding="utf-8",
                errors="replace",
                closefd=False,
            ) as input_file:
                return input_file.read()
        finally:
            os.close(descriptor)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def write_result_json(self, filename: str, value: JsonObject) -> None:
        if PurePosixPath(filename).name != filename:
            raise HarnessError("result filename must not contain a path")
        temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
        encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._result_directory_descriptor,
            )
            with os.fdopen(
                descriptor, mode="w", encoding="utf-8", closefd=False
            ) as output:
                output.write(encoded)
                output.flush()
                os.fchmod(descriptor, 0o644)
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                temporary,
                filename,
                src_dir_fd=self._result_directory_descriptor,
                dst_dir_fd=self._result_directory_descriptor,
            )
            os.fsync(self._result_directory_descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self._result_directory_descriptor)


def _check_processes_alive(
    effects: HarnessEffects, processes: Sequence[OwnedProcess]
) -> None:
    dead_hosts = [
        process.host_name for process in processes if not effects.process_alive(process)
    ]
    if dead_hosts:
        raise HarnessError(f"owned Exo process exited on {', '.join(dead_hosts)}")


def _wait_until(
    effects: HarnessEffects,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    description: str,
    processes: Sequence[OwnedProcess],
    check: Callable[[], bool],
) -> None:
    deadline = effects.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while effects.monotonic() < deadline:
        _check_processes_alive(effects, processes)
        try:
            if check():
                return
            last_error = None
        except HttpResponseError as error:
            last_error = error
        except (ConnectionError, OSError) as error:
            last_error = error
        effects.sleep(poll_seconds)
    suffix = f": last error was {last_error}" if last_error is not None else ""
    raise HarnessError(f"timed out waiting for {description}{suffix}")


def wait_for_api(
    effects: HarnessEffects,
    config: HarnessConfig,
    processes: Sequence[OwnedProcess],
) -> str:
    observed_node_id: str | None = None

    def api_is_ready() -> bool:
        nonlocal observed_node_id
        value = effects.request_json("GET", "/node_id")
        if not isinstance(value, str) or not value:
            return False
        observed_node_id = value
        return True

    _wait_until(
        effects,
        timeout_seconds=config.timeouts.api_start_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="the owned coordinator API",
        processes=processes,
        check=api_is_ready,
    )
    assert observed_node_id is not None
    return observed_node_id


def wait_for_cluster(
    effects: HarnessEffects,
    config: HarnessConfig,
    processes: Sequence[OwnedProcess],
) -> dict[str, str]:
    runtime_node_ids: dict[str, str] | None = None

    def cluster_is_ready() -> bool:
        nonlocal runtime_node_ids
        resources = effects.request_json("GET", "/state/nodeComputeResources")
        backends = effects.request_json("GET", "/state/nodeBackends")
        try:
            runtime_node_ids = validate_cluster_inventory(resources, backends, config)
        except HarnessError:
            return False
        return True

    _wait_until(
        effects,
        timeout_seconds=config.timeouts.cluster_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="the exact two-host CUDA inventory",
        processes=processes,
        check=cluster_is_ready,
    )
    assert runtime_node_ids is not None
    return runtime_node_ids


def _get_optional_state(effects: HarnessEffects, path: str) -> JsonValue | None:
    try:
        return effects.request_json("GET", path)
    except HttpResponseError as error:
        if error.status == 404:
            return None
        raise


def assert_instance_id_unused(effects: HarnessEffects, instance_id: str) -> None:
    existing = _get_optional_state(effects, f"/state/instances/{instance_id}")
    if existing is not None:
        raise HarnessError(f"proposed instance ID {instance_id} already exists")


def wait_for_owned_runners_ready(
    effects: HarnessEffects,
    config: HarnessConfig,
    processes: Sequence[OwnedProcess],
    instance_id: str,
    runner_ids: Sequence[str],
) -> None:
    expected_runner_ids = set(runner_ids)

    def runners_are_ready() -> bool:
        instance = _get_optional_state(effects, f"/state/instances/{instance_id}")
        if instance is None:
            return False
        inner = _tagged(instance, "MlxNcclInstance", "owned instance")
        assignments = _assignment_object(inner)
        observed_runners = set(
            _object(assignments.get("runnerToShard"), "owned runnerToShard")
        )
        if observed_runners != expected_runner_ids:
            raise HarnessError("owned instance runner set changed after submission")

        all_ready = True
        for runner_id in runner_ids:
            status_value = _get_optional_state(effects, f"/state/runners/{runner_id}")
            if status_value is None:
                all_ready = False
                continue
            status = _object(status_value, f"runner {runner_id} status")
            if len(status) != 1:
                raise HarnessError(f"runner {runner_id} status is not tagged")
            tag = next(iter(status))
            if tag == "RunnerFailed":
                failure = _object(status[tag], f"runner {runner_id} failure")
                message = failure.get("errorMessage")
                raise HarnessError(f"runner {runner_id} failed: {message}")
            if tag != "RunnerReady":
                all_ready = False
        return all_ready

    _wait_until(
        effects,
        timeout_seconds=config.timeouts.runner_ready_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="all three owned runners to reach RunnerReady",
        processes=processes,
        check=runners_are_ready,
    )


def deterministic_request(config: HarnessConfig) -> JsonObject:
    return {
        "model": config.model.model_id,
        "messages": [{"role": "user", "content": config.benchmark.prompt}],
        "max_tokens": config.benchmark.max_tokens,
        "temperature": config.benchmark.temperature,
        "seed": config.benchmark.seed,
        "stream": False,
        "use_prefix_cache": False,
        "logprobs": False,
    }


def _completion_result(
    response_value: JsonValue,
    *,
    elapsed_seconds: float,
    iteration: int,
    expected_model_id: str,
    expected_content_sha256: str,
) -> JsonObject:
    response = _object(response_value, "benchmark completion response")
    if response.get("object") != "chat.completion":
        raise HarnessError("benchmark completion returned the wrong object type")
    if response.get("model") != expected_model_id:
        raise HarnessError("benchmark completion returned the wrong model ID")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise HarnessError("benchmark completion must contain exactly one choice")
    choice = _object(choices[0], "benchmark completion choice")
    finish_reason = choice.get("finish_reason", choice.get("finishReason"))
    if finish_reason not in {"stop", "length", "tool_calls"}:
        raise HarnessError("benchmark completion has an invalid finish reason")
    message = _object(choice.get("message"), "benchmark completion message")
    content_value = message.get("content")
    if not isinstance(content_value, str) or not content_value:
        raise HarnessError("benchmark completion content must be nonempty text")
    content_sha256 = hashlib.sha256(content_value.encode()).hexdigest()
    if content_sha256 != expected_content_sha256:
        raise HarnessError("benchmark completion differs from the correctness oracle")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise HarnessError(
            "benchmark completion elapsed time must be finite and positive"
        )
    statistics_value = response.get("generation_stats")
    if statistics_value is None:
        statistics_value = response.get("generationStats")
    generation_statistics = _object(statistics_value, "benchmark generation statistics")
    for snake_key, camel_key in (
        ("prompt_tps", "promptTps"),
        ("generation_tps", "generationTps"),
    ):
        _positive_float(
            generation_statistics.get(snake_key, generation_statistics.get(camel_key)),
            f"generation statistic {snake_key}",
        )
    for snake_key, camel_key in (
        ("prompt_tokens", "promptTokens"),
        ("generation_tokens", "generationTokens"),
    ):
        _positive_integer(
            generation_statistics.get(snake_key, generation_statistics.get(camel_key)),
            f"generation statistic {snake_key}",
        )
    return {
        "iteration": iteration,
        "elapsed_seconds": elapsed_seconds,
        "response_id": response.get("id"),
        "finish_reason": finish_reason,
        "content_sha256": content_sha256,
        "content_bytes": len(content_value.encode()),
        "generation_stats": cast(JsonValue, generation_statistics),
        "power_usage": response.get("power_usage", response.get("powerUsage")),
    }


def run_completions(
    effects: HarnessEffects,
    config: HarnessConfig,
    processes: Sequence[OwnedProcess],
) -> tuple[list[JsonObject], list[JsonObject]]:
    request = deterministic_request(config)
    warmups: list[JsonObject] = []
    samples: list[JsonObject] = []
    for iteration in range(
        config.benchmark.warmup_count + config.benchmark.sample_count
    ):
        _check_processes_alive(effects, processes)
        start = effects.monotonic()
        response = effects.request_json("POST", "/bench/chat/completions", body=request)
        elapsed = effects.monotonic() - start
        result = _completion_result(
            response,
            elapsed_seconds=elapsed,
            iteration=iteration,
            expected_model_id=config.model.model_id,
            expected_content_sha256=config.benchmark.expected_content_sha256,
        )
        if iteration < config.benchmark.warmup_count:
            warmups.append(result)
        else:
            samples.append(result)
    return warmups, samples


def _instance_uses_resources(instance_value: JsonValue, resource_ids: set[str]) -> bool:
    if not isinstance(instance_value, dict) or len(instance_value) != 1:
        return True
    inner_value = next(iter(instance_value.values()))
    if not isinstance(inner_value, dict):
        return True
    assignments = inner_value.get("shardAssignments")
    if not isinstance(assignments, dict):
        return True
    resource_map = assignments.get("computeResourceToRunner")
    if not isinstance(resource_map, dict):
        return True
    return bool(resource_ids.intersection(resource_map))


def cleanup_state_is_clear(
    state_value: JsonValue,
    instance_id: str,
    runner_ids: Sequence[str],
    resource_ids: Sequence[str],
) -> bool:
    state = _object(state_value, "cluster cleanup state")
    instances = _object(state.get("instances"), "cleanup instances")
    runners = _object(state.get("runners"), "cleanup runners")
    retiring = _object(
        state.get("retiringComputeResources"), "retiring compute resources"
    )
    prefill_ports = _object(state.get("prefillServerPorts"), "prefill server ports")
    owned_runners = set(runner_ids)
    owned_resources = set(resource_ids)
    return (
        instance_id not in instances
        and owned_runners.isdisjoint(runners)
        and owned_runners.isdisjoint(prefill_ports)
        and owned_resources.isdisjoint(retiring)
        and owned_runners.isdisjoint(retiring.values())
        and not any(
            _instance_uses_resources(instance, owned_resources)
            for instance in instances.values()
        )
    )


def delete_and_verify_owned_instance(
    effects: HarnessEffects,
    config: HarnessConfig,
    processes: Sequence[OwnedProcess],
    instance_id: str,
    runner_ids: Sequence[str],
    resource_ids: Sequence[str],
) -> None:
    try:
        effects.request_json("DELETE", f"/instance/{instance_id}")
    except HttpResponseError as error:
        if error.status != 404:
            raise

    def cleanup_is_clear() -> bool:
        state = effects.request_json("GET", "/state")
        return cleanup_state_is_clear(state, instance_id, runner_ids, resource_ids)

    _wait_until(
        effects,
        timeout_seconds=config.timeouts.cleanup_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="owned instance, runners, and resource leases to disappear",
        processes=processes,
        check=cleanup_is_clear,
    )


def _sample_aggregate(samples: Sequence[JsonObject]) -> JsonObject:
    latencies = [
        _positive_float(sample.get("elapsed_seconds"), "sample elapsed_seconds")
        for sample in samples
    ]
    prompt_rates: list[float] = []
    generation_rates: list[float] = []
    prompt_tokens: list[int] = []
    generation_tokens: list[int] = []
    for sample in samples:
        stats = _object(sample["generation_stats"], "sample generation statistics")
        prompt_rates.append(
            _positive_float(
                stats.get("prompt_tps", stats.get("promptTps")), "sample prompt TPS"
            )
        )
        generation_rates.append(
            _positive_float(
                stats.get("generation_tps", stats.get("generationTps")),
                "sample generation TPS",
            )
        )
        prompt_tokens.append(
            _positive_integer(
                stats.get("prompt_tokens", stats.get("promptTokens")),
                "sample prompt token count",
            )
        )
        generation_tokens.append(
            _positive_integer(
                stats.get("generation_tokens", stats.get("generationTokens")),
                "sample generation token count",
            )
        )
    deterministic_fingerprints = {
        (
            sample["content_sha256"],
            prompt_token_count,
            generation_token_count,
        )
        for sample, prompt_token_count, generation_token_count in zip(
            samples, prompt_tokens, generation_tokens, strict=True
        )
    }
    if len(deterministic_fingerprints) != 1:
        raise HarnessError(
            "deterministic measured samples disagree on content hash or token counts"
        )
    input_tokens_json: list[JsonValue] = [value for value in prompt_tokens]
    output_tokens_json: list[JsonValue] = [value for value in generation_tokens]
    return {
        "samples": len(samples),
        "latency_seconds_mean": statistics.fmean(latencies),
        "latency_seconds_median": statistics.median(latencies),
        "prefill_tokens_per_second_mean": statistics.fmean(prompt_rates),
        "decode_tokens_per_second_mean": statistics.fmean(generation_rates),
        "input_tokens_per_sample": input_tokens_json,
        "output_tokens_per_sample": output_tokens_json,
    }


def validate_nccl_logs(
    config: HarnessConfig,
    placement: JsonObject,
    logs_by_host: Mapping[str, str],
    owner_token: str,
) -> JsonObject:
    """Require stable NCCL transport/rank evidence before a run is reportable."""
    if set(logs_by_host) != {host.name for host in config.hosts}:
        raise HarnessError("owned NCCL logs do not cover both configured hosts")
    inner = _tagged(placement, "MlxNcclInstance", "logged placement")
    assignments = _assignment_object(inner)
    runners = _object(assignments.get("runnerToShard"), "logged runnerToShard")
    resource_to_runner = _object(
        assignments.get("computeResourceToRunner"), "logged computeResourceToRunner"
    )
    rank_by_runner: dict[str, int] = {}
    for runner_id, tagged_shard in runners.items():
        shard = _tagged(tagged_shard, "TensorShardMetadata", f"logged {runner_id}")
        rank = shard.get("deviceRank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise HarnessError("logged placement has a non-integer rank")
        rank_by_runner[runner_id] = rank
    rank_bindings: list[JsonValue] = []
    expected_ranks_by_host: dict[str, set[int]] = {
        host.name: set() for host in config.hosts
    }
    for host in config.hosts:
        for gpu in host.gpus:
            runner_id = _string(
                resource_to_runner.get(gpu.resource_id),
                f"runner for logged resource {gpu.resource_id}",
            )
            rank_bindings.append(
                {
                    "host_name": host.name,
                    "gpu_uuid": gpu.device_uuid,
                    "resource_id": gpu.resource_id,
                    "runner_id": runner_id,
                    "rank": rank_by_runner[runner_id],
                    "world_size": 3,
                }
            )
            expected_ranks_by_host[host.name].add(rank_by_runner[runner_id])

    combined_logs = "\n".join(logs_by_host.values())
    fatal_patterns = (
        r"\bNCCL\s+WARN\b",
        r"\bNET/Socket\b",
        r"\busing\s+network\s+Socket\b",
        r"\bNCCL[^\n]{0,80}\b(?:fatal|abort(?:ed)?|unhandled error)\b",
    )
    for pattern in fatal_patterns:
        if re.search(pattern, combined_logs, flags=re.IGNORECASE):
            raise HarnessError(f"owned NCCL logs contain forbidden marker {pattern}")

    rank_pattern = re.compile(
        r"\brank\s*[:=]?\s*([0-2])\b[^\n]{0,200}"
        r"\b(?:nranks|world(?:_size|\s+size)?)\s*[:=]?\s*3\b",
        flags=re.IGNORECASE,
    )
    reverse_rank_pattern = re.compile(
        r"\b(?:nranks|world(?:_size|\s+size)?)\s*[:=]?\s*3\b"
        r"[^\n]{0,200}\brank\s*[:=]?\s*([0-2])\b",
        flags=re.IGNORECASE,
    )
    observed_ranks = {
        int(match.group(1))
        for pattern in (rank_pattern, reverse_rank_pattern)
        for match in pattern.finditer(combined_logs)
    }
    initialized_ranks: set[int] = set()
    for line in combined_logs.splitlines():
        if re.search(r"\bInit\s+COMPLETE\b", line, flags=re.IGNORECASE):
            for pattern in (rank_pattern, reverse_rank_pattern):
                match = pattern.search(line)
                if match is not None:
                    initialized_ranks.add(int(match.group(1)))
    if observed_ranks != {0, 1, 2} or initialized_ranks != {0, 1, 2}:
        raise HarnessError("NCCL logs do not prove initialized ranks 0, 1, 2 of 3")

    host_evidence: JsonObject = {}
    for host in config.hosts:
        log_text = logs_by_host[host.name]
        host_observed_ranks = {
            int(match.group(1))
            for pattern in (rank_pattern, reverse_rank_pattern)
            for match in pattern.finditer(log_text)
        }
        host_initialized_ranks: set[int] = set()
        for line in log_text.splitlines():
            if re.search(r"\bInit\s+COMPLETE\b", line, flags=re.IGNORECASE):
                for pattern in (rank_pattern, reverse_rank_pattern):
                    match = pattern.search(line)
                    if match is not None:
                        host_initialized_ranks.add(int(match.group(1)))
        expected_host_ranks = expected_ranks_by_host[host.name]
        if (
            host_observed_ranks != expected_host_ranks
            or host_initialized_ranks != expected_host_ranks
        ):
            raise HarnessError(
                f"{host.name} log does not prove its placement-assigned ranks"
            )
        launch_lines = [
            line.removeprefix("EXO_POC_LAUNCH ")
            for line in log_text.splitlines()
            if line.startswith("EXO_POC_LAUNCH ")
        ]
        if len(launch_lines) != 1:
            raise HarnessError(f"{host.name} log has no unique launch receipt")
        launch = _object(
            _parse_json_value(launch_lines[0], f"{host.name} launch receipt"),
            f"{host.name} launch receipt",
        )
        expected_environment = {
            **host.environment,
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
        }
        if launch.get("host_name") != host.name:
            raise HarnessError(f"{host.name} launch receipt has the wrong host")
        if launch.get("environment") != expected_environment:
            raise HarnessError(f"{host.name} launch environment receipt differs")
        if launch.get("argv") != list(host.launch_argv):
            raise HarnessError(f"{host.name} launch argv receipt differs")
        if host.transport == "ssh" and (
            "EXO_SUPERVISOR_LOG_COMPLETE" not in log_text.splitlines()
        ):
            raise HarnessError(
                f"{host.name} log does not prove remote stream completion"
            )
        net_ib_lines = [
            line for line in log_text.splitlines() if "NET/IB" in line.upper()
        ]
        if not net_ib_lines or "NCCL INFO" not in log_text.upper():
            raise HarnessError(f"{host.name} log does not prove NCCL NET/IB")
        hca_tokens = [f"{port.device}:{port.port}" for port in host.hca_ports]
        for token in hca_tokens:
            if not any(token.lower() in line.lower() for line in net_ib_lines):
                raise HarnessError(
                    f"{host.name} NCCL log does not show selected HCA {token}"
                )
        merged_rail_evidence = len(hca_tokens) == 1 or any(
            all(token.lower() in line.lower() for token in hca_tokens)
            for line in net_ib_lines
        )
        if not merged_rail_evidence:
            raise HarnessError(f"{host.name} log does not prove merged HCA rails")
        net_ib_lines_json: list[JsonValue] = [line for line in net_ib_lines[:20]]
        hca_tokens_json: list[JsonValue] = [token for token in hca_tokens]
        evidence: JsonObject = {
            "sha256": hashlib.sha256(log_text.encode()).hexdigest(),
            "bytes": len(log_text.encode()),
            "net_ib_lines": net_ib_lines_json,
            "selected_hcas": hca_tokens_json,
            "merged_rail_evidence": merged_rail_evidence,
            "observed_ranks": [rank for rank in sorted(host_observed_ranks)],
            "initialized_ranks": [rank for rank in sorted(host_initialized_ranks)],
            "gin_disabled": (
                host.environment["NCCL_GIN_ENABLE"] == "0"
                and host.environment["NCCL_GIN_TYPE"] == "0"
            ),
        }
        host_evidence[host.name] = evidence
    return {
        "evidence_scope": "functional_transport_and_rank_initialization",
        "performance_comparable": False,
        "hca_payload_counter_deltas_verified": False,
        "world_size": 3,
        "observed_ranks": [rank for rank in sorted(observed_ranks)],
        "initialized_ranks": [rank for rank in sorted(initialized_ranks)],
        "rank_bindings": rank_bindings,
        "hosts": host_evidence,
        "socket_fallback_absent": True,
        "fatal_markers_absent": True,
    }


def _process_metadata(
    config: HarnessConfig, owner_token: str, processes: Sequence[OwnedProcess]
) -> JsonObject:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "owner_token": owner_token,
        "owned_processes": [cast(JsonValue, asdict(process)) for process in processes],
    }


def run_harness(
    config: HarnessConfig,
    effects: HarnessEffects,
    signal_latch: SignalLatch | None = None,
) -> JsonObject:
    """Run one proof and always emit an ownership/cleanup result fragment."""
    latch = signal_latch or SignalLatch()
    effects = SignalAwareEffects(effects, latch)
    owner_token = f"{config.run_id}:{uuid.uuid4().hex}"
    processes: list[OwnedProcess] = []
    failed_start_receipts: list[OwnedProcess] = []
    failed_start_cleanup_count = 0
    preflights: dict[str, JsonValue] = {}
    model_probes: dict[str, JsonValue] = {}
    model_sha256_manifest: dict[str, str] | None = None
    runtime_identity: JsonObject | None = None
    process_cleanups: list[JsonObject] = []
    owned_instance_id: str | None = None
    owned_runner_ids: tuple[str, ...] = ()
    owned_resource_ids: tuple[str, ...] = ()
    submission_attempted = False
    preflight_complete = False
    benchmark_complete = False
    instance_cleanup_complete = True
    caught_error: BaseException | None = None
    placement: JsonObject | None = None
    warmups: list[JsonObject] = []
    samples: list[JsonObject] = []
    aggregate: JsonObject | None = None
    nccl_log_evidence: JsonObject | None = None
    runtime_node_ids: dict[str, str] = {}
    started_at = time.time()

    try:
        for host in config.hosts:
            report = effects.run_preflight(host, config)
            model_probe = effects.probe_model(host, config.model)
            preflights[host.name] = cast(JsonValue, report.model_dump(mode="json"))
            model_probes[host.name] = cast(
                JsonValue, model_probe.model_dump(mode="json")
            )
            validate_preflight(report, model_probe, host, config)
            if model_sha256_manifest is None:
                model_sha256_manifest = model_probe.sha256_manifest
            elif model_probe.sha256_manifest != model_sha256_manifest:
                raise HarnessError(
                    "model SHA-256 manifests differ between the two hosts"
                )
            observed_runtime_identity = validated_runtime_identity(report, host, config)
            if runtime_identity is None:
                runtime_identity = observed_runtime_identity
            elif observed_runtime_identity != runtime_identity:
                raise HarnessError(
                    "common Python/package runtime identity differs between the two hosts"
                )
        preflight_complete = True

        for host in sorted(config.hosts, key=lambda item: item.launch_order):
            process = effects.start_node(host, config, owner_token)
            processes.append(process)
            effects.write_result_json(
                _RUNTIME_METADATA, _process_metadata(config, owner_token, processes)
            )
            latch.checkpoint()

        api_node_id = wait_for_api(effects, config, processes)
        runtime_node_ids = wait_for_cluster(effects, config, processes)
        coordinator = next(host for host in config.hosts if host.role == "coordinator")
        if runtime_node_ids[coordinator.name] != api_node_id:
            raise HarnessError(
                "API node ID does not map to the coordinator GPU inventory"
            )
        raw_placement = effects.request_json(
            "GET",
            "/instance/placement",
            params={
                "model_id": config.model.model_id,
                "sharding": "Tensor",
                "instance_meta": "MlxNccl",
                "min_nodes": "2",
                "use_all_compute_resources": "true",
            },
        )
        (
            placement,
            owned_instance_id,
            owned_runner_ids,
            owned_resource_ids,
        ) = validate_and_patch_placement(raw_placement, config, runtime_node_ids)
        assert_instance_id_unused(effects, owned_instance_id)
        submission_attempted = True
        effects.request_json(
            "POST",
            "/instance",
            body={"instance": cast(JsonValue, placement)},
        )
        wait_for_owned_runners_ready(
            effects,
            config,
            processes,
            owned_instance_id,
            owned_runner_ids,
        )
        warmups, samples = run_completions(effects, config, processes)
        aggregate = _sample_aggregate(samples)
        benchmark_complete = True
    except StartNodeError as error:
        caught_error = error
        failed_start_cleanup_count = 1
        process_cleanups.append(
            _object(
                _validated_json_value(asdict(error.cleanup), "start cleanup receipt"),
                "start cleanup receipt",
            )
        )
        if error.receipt is not None:
            failed_start_receipts.append(error.receipt)
            try:
                effects.write_result_json(
                    _RUNTIME_METADATA,
                    _process_metadata(
                        config,
                        owner_token,
                        (*processes, *failed_start_receipts),
                    ),
                )
            except BaseException as metadata_error:
                process_cleanups[-1]["ownership_verified"] = False
                process_cleanups[-1]["terminated"] = False
                process_cleanups[-1]["error"] = (
                    "failed to persist parsed start receipt: "
                    f"{type(metadata_error).__name__}: {metadata_error}"
                )
    except BaseException as error:
        caught_error = error
    finally:
        latch.begin_cleanup()
        if submission_attempted and owned_instance_id is not None:
            try:
                delete_and_verify_owned_instance(
                    effects,
                    config,
                    processes,
                    owned_instance_id,
                    owned_runner_ids,
                    owned_resource_ids,
                )
            except BaseException as error:
                instance_cleanup_complete = False
                if caught_error is None:
                    caught_error = error
        for process in reversed(processes):
            try:
                cleanup = effects.stop_node(process, config.timeouts.cleanup_seconds)
            except BaseException as error:
                cleanup = ProcessCleanup(
                    host_name=process.host_name,
                    ownership_verified=False,
                    terminated=False,
                    forced=False,
                    error=f"{type(error).__name__}: {error}",
                )
            process_cleanups.append(cast(JsonObject, asdict(cleanup)))
        if benchmark_complete and placement is not None:
            try:
                logs_by_host = {
                    process.host_name: effects.read_owned_log(process)
                    for process in processes
                }
                nccl_log_evidence = validate_nccl_logs(
                    config, placement, logs_by_host, owner_token
                )
            except BaseException as error:
                benchmark_complete = False
                if caught_error is None:
                    caught_error = error

    process_cleanup_complete = (
        all(
            cleanup.get("ownership_verified") is True
            and cleanup.get("terminated") is True
            for cleanup in process_cleanups
        )
        and len(process_cleanups) == len(processes) + failed_start_cleanup_count
    )
    cleanup_complete = instance_cleanup_complete and process_cleanup_complete
    if not preflight_complete:
        status = "preflight_failed"
    elif not cleanup_complete:
        status = "cleanup_failed"
    elif not benchmark_complete or caught_error is not None:
        status = "benchmark_failed"
    else:
        status = "completed"

    fragment_value = _validated_json_value(
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "namespace": config.namespace,
            "status": status,
            "reportable": status == "completed",
            "result_scope": "functional_proof_of_concept",
            "performance_comparable": False,
            "dual_rail_payload_verified": False,
            "requires_exact_physical_gpu_inventory": True,
            "completed_normally": benchmark_complete and caught_error is None,
            "cleanup_succeeded": cleanup_complete,
            "interrupted_signal": latch.signal_number,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": time.time(),
            "error": (
                None
                if caught_error is None
                else f"{type(caught_error).__name__}: {caught_error}"
            ),
            "model": config.model.model_dump(mode="json"),
            "correctness_oracle": {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "prompt_sha256": hashlib.sha256(
                    config.benchmark.prompt.encode("utf-8")
                ).hexdigest(),
                "expected_content_sha256": (config.benchmark.expected_content_sha256),
            },
            "model_paths": {host.name: host.model_path for host in config.hosts},
            "preflight": preflights,
            "model_probes": model_probes,
            "commands": {host.name: list(host.launch_argv) for host in config.hosts},
            "environment": {
                host.name: {
                    **host.environment,
                    "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
                }
                for host in config.hosts
            },
            "probe_environment": {
                host.name: {
                    name: value for name, value in sorted(host.environment.items())
                }
                for host in config.hosts
            },
            "resource_bindings": {
                host.name: {
                    "runtime_node_id": runtime_node_ids.get(host.name),
                    "gpu_uuids": [gpu.device_uuid for gpu in host.gpus],
                    "gpu_pci_bus_ids": [gpu.pci_bus_id for gpu in host.gpus],
                    "cpu_set": list(host.cpu_set),
                    "numa_nodes": list(host.numa_nodes),
                    "hca_ports": [
                        port.model_dump(mode="json") for port in host.hca_ports
                    ],
                }
                for host in config.hosts
            },
            "reserved_ports": list(config.reserved_ports),
            "owned_processes": [
                asdict(process) for process in (*processes, *failed_start_receipts)
            ],
            "placement": placement,
            "owned_instance_id": owned_instance_id,
            "owned_runner_ids": list(owned_runner_ids),
            "owned_compute_resource_ids": list(owned_resource_ids),
            "request": deterministic_request(config),
            "warmup_count": config.benchmark.warmup_count,
            "sample_count": config.benchmark.sample_count,
            "warmups": warmups,
            "samples": samples,
            "aggregate": aggregate,
            "nccl_log_evidence": nccl_log_evidence,
            "instance_cleanup_succeeded": instance_cleanup_complete,
            "process_cleanup": process_cleanups,
        },
        "benchmark result fragment",
    )
    fragment = _object(fragment_value, "benchmark result fragment")
    effects.write_result_json(_RESULT_FRAGMENT, fragment)
    return fragment


class CliArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path | None


class LeasePreparationCliArguments(argparse.Namespace):
    config: Path
    metadata_output: Path
    wrapper_python: Path
    child_python: Path
    benchmark_lease_script: Path
    harness_script: Path
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
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="must exactly match result_directory in the strict config",
    )
    return parser.parse_args(arguments, namespace=CliArguments())


def parse_lease_preparation_args(
    arguments: Sequence[str] | None = None,
) -> LeasePreparationCliArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete strict POC config and create canonical, fresh "
            "benchmark-lease metadata. This does not generate the hardware config."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--wrapper-python", required=True, type=Path)
    parser.add_argument("--child-python", required=True, type=Path)
    parser.add_argument("--benchmark-lease-script", required=True, type=Path)
    parser.add_argument("--harness-script", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-duration-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--cleanup-grace-seconds", required=True, type=float)
    parser.add_argument("--lease-path", type=Path, default=_DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=_DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", required=True, type=Path)
    return parser.parse_args(arguments, namespace=LeasePreparationCliArguments())


def _require_absolute_path(path: Path, description: str) -> None:
    if not path.is_absolute():
        raise HarnessError(f"{description} must be an absolute path")
    if "\0" in str(path):
        raise HarnessError(f"{description} must not contain NUL")


def _require_canonical_regular_file(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HarnessError(f"cannot resolve {description} {path}: {error}") from error
    if resolved != path:
        raise HarnessError(f"{description} must be canonical and must not use symlinks")
    if not path.is_file():
        raise HarnessError(f"{description} must be a regular file")


def _require_canonical_directory(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HarnessError(f"cannot resolve {description} {path}: {error}") from error
    if resolved != path:
        raise HarnessError(f"{description} must be canonical and must not use symlinks")
    if not path.is_dir():
        raise HarnessError(f"{description} must be a directory")


def _require_absolute_executable(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise HarnessError(f"{description} must be an executable file")


def _require_python_interpreter(path: Path) -> None:
    marker = uuid.uuid4().hex
    probe_script = (
        "import json,sys;"
        "print(json.dumps({"
        f"'marker':{marker!r},"
        "'implementation':sys.implementation.name,"
        "'version':list(sys.version_info[:2]),"
        "'executable':sys.executable"
        "},sort_keys=True,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            (str(path), "-I", "-S", "-c", probe_script),
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
            env={
                "PATH": os.defpath,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError(
            f"cannot validate configured Python interpreter: {error}"
        ) from error
    if completed.returncode != 0:
        raise HarnessError(
            "configured Python interpreter probe failed with return code "
            f"{completed.returncode}: {completed.stderr[-300:]}"
        )
    try:
        value = _object(
            _parse_json_value(completed.stdout, "configured Python identity probe"),
            "configured Python identity probe",
        )
    except HarnessError as error:
        raise HarnessError(
            "configured Python interpreter did not return the identity probe"
        ) from error
    expected_identity: dict[str, object] = {
        "marker": marker,
        "implementation": "cpython",
        "version": [3, 13],
    }
    if any(value.get(name) != expected for name, expected in expected_identity.items()):
        raise HarnessError(
            "configured Python interpreter returned an incompatible identity"
        )
    executable = value.get("executable")
    if not isinstance(executable, str):
        raise HarnessError("configured Python interpreter omitted sys.executable")
    try:
        same_executable = Path(executable).samefile(path)
    except OSError as error:
        raise HarnessError(
            "cannot bind configured Python interpreter to sys.executable"
        ) from error
    if not same_executable:
        raise HarnessError(
            "configured Python interpreter does not match its sys.executable"
        )


def _canonical_prospective_path(path: Path, description: str) -> Path:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise HarnessError(f"cannot resolve {description} {path}: {error}") from error
    if not resolved.is_absolute():
        raise HarnessError(f"resolved {description} must remain absolute")
    return resolved


def _require_outside_directory(path: Path, directory: Path, description: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError:
        return
    raise HarnessError(f"{description} must be outside the source deployment")


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise HarnessError(f"cannot inspect path {path}: {error}") from error
    return True


def _format_cli_number(value: float, description: str) -> str:
    if not math.isfinite(value) or value <= 0:
        raise HarnessError(f"{description} must be finite and positive")
    return str(value)


def _open_directory_without_symlinks(path: Path) -> int:
    """Open an absolute directory without following any ancestor symlink."""
    _require_absolute_path(path, "directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            if component in ("", ".", "..") or "\0" in component:
                raise HarnessError(f"directory path is not canonical: {path}")
            child_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except HarnessError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise HarnessError(
            f"cannot open directory without symlinks {path}: {error}"
        ) from error


def _validate_result_directory_descriptor(path: Path, descriptor: int) -> None:
    try:
        expected = os.fstat(descriptor)
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise HarnessError(
            f"cannot validate trusted result directory: {error}"
        ) from error
    if not stat.S_ISDIR(expected.st_mode):
        raise HarnessError("trusted result descriptor is not a directory")
    if not stat.S_ISDIR(observed.st_mode) or (
        observed.st_dev,
        observed.st_ino,
    ) != (expected.st_dev, expected.st_ino):
        raise HarnessError("trusted result directory path identity changed")


def _inherited_result_directory_descriptor(path: Path) -> int:
    raw_descriptor = os.environ.get(_RESULT_DIRECTORY_FD_ENVIRONMENT)
    if (
        raw_descriptor is None
        or not raw_descriptor.isascii()
        or not raw_descriptor.isdigit()
    ):
        raise HarnessError(
            f"lease wrapper did not pass {_RESULT_DIRECTORY_FD_ENVIRONMENT}"
        )
    descriptor = int(raw_descriptor)
    _validate_result_directory_descriptor(path, descriptor)
    return descriptor


def _atomic_write_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Install a new JSON file atomically without following or replacing symlinks."""
    _require_absolute_path(path, "metadata output")
    if not path.name:
        raise HarnessError("metadata output must name a file")
    _require_canonical_directory(path.parent, "metadata output parent")
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    directory_descriptor = _open_directory_without_symlinks(path.parent)
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary_created = False
    try:
        try:
            os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise HarnessError(f"metadata output already exists: {path}")
        temporary_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        temporary_descriptor = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_created = True
        with os.fdopen(temporary_descriptor, "wb", closefd=True) as output:
            output.write(serialized)
            output.flush()
            os.fchmod(output.fileno(), 0o644)
            os.fsync(output.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise HarnessError(f"metadata output already exists: {path}") from error
        os.fsync(directory_descriptor)
    except OSError as error:
        raise HarnessError(
            f"cannot atomically create metadata {path}: {error}"
        ) from error
    finally:
        if temporary_created:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _validate_metadata_with_benchmark_lease(
    benchmark_lease_script: Path,
    metadata: JsonObject,
    generated_at: datetime,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_exo_poc_benchmark_lease_validation", benchmark_lease_script
    )
    if specification is None or specification.loader is None:
        raise HarnessError("cannot import the exact benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
        validator = cast(LeaseMetadataValidator, cast(object, module))
        validated = validator.validate_run_metadata(
            cast(Mapping[str, object], metadata), now=generated_at
        )
    except Exception as error:
        raise HarnessError(
            f"benchmark lease rejected generated metadata: {error}"
        ) from error
    if validated != metadata:
        raise HarnessError("benchmark lease metadata validation changed the document")


def _validate_preparation_source(
    config: HarnessConfig,
    source_identity: Callable[[str], SourceIdentity],
) -> SourceIdentity:
    configured_source = config.hosts[0].source
    if any(host.source != configured_source for host in config.hosts[1:]):
        raise HarnessError(
            "lease v1 requires identical source identities on both hosts"
        )
    coordinator = next(host for host in config.hosts if host.role == "coordinator")
    observed_source = source_identity(coordinator.source_directory)
    if observed_source != configured_source:
        raise HarnessError(
            "strict config source identity is stale for the coordinator deployment"
        )
    return observed_source


def prepare_lease_metadata(
    *,
    config_path: Path,
    metadata_output: Path,
    wrapper_python: Path,
    child_python: Path,
    benchmark_lease_script: Path,
    harness_script: Path,
    owner: str,
    purpose: str,
    expected_duration_seconds: float,
    heartbeat_seconds: float,
    cleanup_grace_seconds: float,
    lease_path: Path,
    lock_path: Path,
    result_root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    source_identity: Callable[[str], SourceIdentity] | None = None,
) -> LeasePreparation:
    path_fields = (
        (config_path, "config path"),
        (metadata_output, "metadata output"),
        (wrapper_python, "wrapper Python"),
        (child_python, "child Python"),
        (benchmark_lease_script, "benchmark lease script"),
        (harness_script, "harness script"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
        (result_root, "result root"),
    )
    for path, description in path_fields:
        _require_absolute_path(path, description)
    _require_canonical_regular_file(config_path, "config path")
    _require_canonical_regular_file(benchmark_lease_script, "benchmark lease script")
    _require_canonical_regular_file(harness_script, "harness script")
    _require_absolute_executable(wrapper_python, "wrapper Python")
    _require_absolute_executable(child_python, "child Python")
    lease_path = _canonical_prospective_path(lease_path, "lease path")
    lock_path = _canonical_prospective_path(lock_path, "lock path")
    _require_canonical_directory(result_root, "result root")
    _require_canonical_directory(metadata_output.parent, "metadata output parent")
    if _path_exists_without_following(metadata_output):
        raise HarnessError(f"metadata output already exists: {metadata_output}")

    config = load_config(config_path)
    coordinator = next(host for host in config.hosts if host.role == "coordinator")
    source_directory = Path(coordinator.source_directory)
    _require_canonical_directory(source_directory, "coordinator source directory")
    expected_harness_script = source_directory / "scripts" / Path(__file__).name
    expected_wrapper_script = source_directory / "scripts" / "benchmark_lease.py"
    if (
        harness_script != expected_harness_script
        or harness_script != Path(__file__).resolve()
    ):
        raise HarnessError(
            "harness script does not match the coordinator config source deployment"
        )
    if benchmark_lease_script != expected_wrapper_script:
        raise HarnessError(
            "benchmark lease script does not match the coordinator config source "
            "deployment"
        )
    if child_python != Path(coordinator.python_executable):
        raise HarnessError(
            "child Python does not match the coordinator config python_executable"
        )
    if wrapper_python != child_python:
        raise HarnessError(
            "wrapper Python must match the configured child Python interpreter"
        )
    _require_python_interpreter(wrapper_python)
    expected_result_directory = result_root / config.run_id
    result_directory = Path(config.result_directory)
    if result_directory != expected_result_directory:
        raise HarnessError(
            "config result_directory must exactly equal result_root/run_id"
        )
    generated_paths = (metadata_output, result_directory, lease_path, lock_path)
    if len(set(generated_paths)) != len(generated_paths):
        raise HarnessError(
            "metadata output, result directory, lease path, and lock path must be "
            "pairwise distinct"
        )
    if _path_exists_without_following(result_directory):
        raise HarnessError(f"result directory already exists: {result_directory}")
    for generated_path, description in (
        (metadata_output, "metadata output"),
        (result_directory, "result directory"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
    ):
        _require_outside_directory(generated_path, source_directory, description)

    owner_value = owner
    purpose_value = purpose
    if not owner.strip() or "\0" in owner:
        raise HarnessError("owner must be nonempty and must not contain NUL")
    if not purpose.strip() or "\0" in purpose:
        raise HarnessError("purpose must be nonempty and must not contain NUL")
    expected_duration = _format_cli_number(
        expected_duration_seconds, "expected duration"
    )
    heartbeat = _format_cli_number(heartbeat_seconds, "heartbeat interval")
    cleanup_grace = _format_cli_number(cleanup_grace_seconds, "cleanup grace")
    minimum_grace = minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise HarnessError(
            "cleanup grace is shorter than the proof cleanup bound "
            f"({minimum_grace} seconds)"
        )

    identity_reader = source_identity or LinuxHostProbe().source_identity
    observed_source = _validate_preparation_source(config, identity_reader)
    child_argv = (
        str(child_python),
        str(harness_script),
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
        raise HarnessError("metadata clock must return an offset-aware timestamp")
    generated_at_utc = generated_at_value.astimezone(timezone.utc)
    generated_at = generated_at_utc.isoformat(timespec="seconds")
    metadata = _lease_static_metadata(config, child_argv)
    metadata["generated_at"] = generated_at
    _validate_metadata_with_benchmark_lease(
        benchmark_lease_script, metadata, generated_at_utc
    )
    benchmark_lease_argv = (
        str(wrapper_python),
        str(benchmark_lease_script),
        f"--owner={owner_value}",
        f"--purpose={purpose_value}",
        "--run-id",
        config.run_id,
        "--namespace",
        config.namespace,
        "--port",
        ",".join(str(port) for port in config.reserved_ports),
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
    if identity_reader(coordinator.source_directory) != observed_source:
        raise HarnessError("source identity changed while preparing lease metadata")
    _atomic_write_new_json(metadata_output, cast(Mapping[str, object], metadata))
    return LeasePreparation(
        metadata=metadata,
        child_argv=child_argv,
        benchmark_lease_argv=benchmark_lease_argv,
        generated_at=generated_at,
        minimum_cleanup_grace_seconds=minimum_grace,
    )


def prepare_lease_main(arguments: Sequence[str] | None = None) -> int:
    args = parse_lease_preparation_args(arguments)
    prepared = prepare_lease_metadata(
        config_path=args.config,
        metadata_output=args.metadata_output,
        wrapper_python=args.wrapper_python,
        child_python=args.child_python,
        benchmark_lease_script=args.benchmark_lease_script,
        harness_script=args.harness_script,
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


def host_preflight_main() -> int:
    try:
        request = HostPreflightRequest.model_validate_json(sys.stdin.read())
        report = collect_host_preflight(request, LinuxHostProbe())
    except (ValidationError, HarnessError, OSError, ValueError) as error:
        print(
            f"host preflight failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 2
    print(report.model_dump_json())
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    normalized_arguments = list(sys.argv[1:] if arguments is None else arguments)
    if normalized_arguments == ["host-preflight"]:
        return host_preflight_main()
    if normalized_arguments and normalized_arguments[0] == "prepare-lease":
        return prepare_lease_main(normalized_arguments[1:])
    args = parse_args(normalized_arguments)
    if not args.config.is_absolute():
        raise HarnessError("--config must be an absolute path")
    if not args.lease_path.is_absolute() or not args.lock_path.is_absolute():
        raise HarnessError("--lease-path and --lock-path must be absolute")
    config = load_config(args.config)
    if args.result_dir is not None and args.result_dir != Path(config.result_directory):
        raise HarnessError("--result-dir must match config.result_directory exactly")
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
        result = run_harness(
            config,
            SystemEffects(config, result_directory_descriptor),
            latch,
        )
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
    if latch.signal_number is not None:
        return 128 + latch.signal_number
    if result["status"] != "completed":
        print(result["error"] or result["status"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
