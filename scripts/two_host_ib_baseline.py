#!/usr/bin/env python3
"""Strict leased two-host InfiniBand bandwidth baseline.

The harness owns two local OpenSM processes, runs one ``ib_write_bw`` test per
rail and one native dual-port test, and refuses to touch processes it did not
start.  It must be invoked as a direct child of ``benchmark_lease.py``.
"""

from __future__ import annotations

import argparse
import contextlib
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
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

RESULT_DIRECTORY_FD_ENVIRONMENT = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
RUNTIME_METADATA_FILENAME = "runtime-metadata.json"
BENCHMARK_RESULT_FILENAME = "benchmark-result.json"
DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
MINIMUM_CLEANUP_GRACE_SECONDS = 180.0
MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SAFE_SSH_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,254}")
_HEX_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GUID = re.compile(r"(?:[0-9a-f]{4}:){3}[0-9a-f]{4}")
_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_PERFTEST_EXECUTABLES = frozenset(
    {
        "opensm",
        "ib_write_bw",
        "ib_read_bw",
        "ib_send_bw",
        "ib_write_lat",
        "ib_read_lat",
        "ib_send_lat",
    }
)
_INFERENCE_EXECUTABLES = frozenset(
    {
        "exo",
        "uvicorn",
        "all_reduce_perf",
        "all_gather_perf",
        "broadcast_perf",
        "hypercube_perf",
        "reduce_scatter_perf",
        "sendrecv_perf",
        "nccl_test",
        "nccl-tests",
        "llama-server",
        "ollama",
        "sglang",
        "text-generation-launcher",
        "tritonserver",
        "vllm",
    }
)
_INFERENCE_SCRIPT_NAMES = frozenset(
    {
        "mlx_nccl_smoke.py",
        "two_host_mlx_nccl_poc.py",
    }
)
_INFERENCE_MODULE_PREFIXES = (
    "exo",
    "ktransformers.server",
    "mlx_lm.server",
    "sglang.launch_server",
    "text_generation_server",
    "vllm.entrypoints",
)
_STORAGE_EXECUTABLES = frozenset(
    {
        "aria2c",
        "b3sum",
        "fio",
        "md5sum",
        "rclone",
        "rsync",
        "sha1sum",
        "sha256sum",
        "sha512sum",
    }
)
_EXO_PROCESS_ENVIRONMENT_PREFIXES = (
    "EXO_LIBP2P_NAMESPACE=",
    "EXO_MLX_DISTRIBUTED_BACKEND=",
    "EXO_ZENOH_NAMESPACE=",
)
_INACTIVE_RAID_SYNC_ACTIONS = frozenset({"frozen", "idle"})


class BaselineError(RuntimeError):
    """Expected fail-closed harness error."""


class ProcessConflictError(BaselineError):
    """An unowned OpenSM or perftest process is present."""


class PortConflictError(BaselineError):
    """A reserved benchmark port is already bound."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceIdentity(StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_file_hashes: dict[str, str]

    @field_validator("dirty_file_hashes")
    @classmethod
    def validate_dirty_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for path, digest in value.items():
            if (
                not path
                or PurePosixPath(path).is_absolute()
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError(
                    "dirty source identities require relative paths and SHA-256 digests"
                )
        return value


class PciIdentity(StrictModel):
    bdf: str
    vendor_id: str = "0x15b3"
    device_id: str = "0x1003"
    driver: str = "mlx4_core"
    current_width: int = Field(gt=0)
    maximum_width: int = Field(gt=0)
    current_speed_gtps: float = Field(gt=0)
    maximum_speed_gtps: float = Field(gt=0)
    numa_node: int = Field(ge=0)

    @field_validator("bdf")
    @classmethod
    def validate_bdf(cls, value: str) -> str:
        normalized = value.lower()
        if _BDF.fullmatch(normalized) is None:
            raise ValueError("PCI BDF must be canonical domain:bus:device.function")
        return normalized

    @model_validator(mode="after")
    def validate_link(self) -> "PciIdentity":
        if (
            self.current_width > self.maximum_width
            or self.current_speed_gtps > self.maximum_speed_gtps
        ):
            raise ValueError("current PCIe link cannot exceed its capability")
        return self


class PortIdentity(StrictModel):
    port: Literal[1, 2]
    port_guid: str
    gid_index: int = Field(default=0, ge=0)
    gid: str
    expected_rate: str = "40 Gb/sec (4X QDR)"

    @field_validator("port_guid")
    @classmethod
    def validate_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _GUID.fullmatch(normalized) is None:
            raise ValueError("port GUID must contain four lowercase 16-bit groups")
        return normalized

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.IPv6Address(value)
        if address.is_unspecified:
            raise ValueError("port GID must not be unspecified")
        return address.exploded

    @model_validator(mode="after")
    def bind_guid_to_gid(self) -> "PortIdentity":
        if self.gid.replace(":", "")[-16:] != self.port_guid.replace(":", ""):
            raise ValueError("port GUID must equal the interface-ID portion of the GID")
        return self


class HcaIdentity(StrictModel):
    device: str = Field(min_length=1)
    node_guid: str
    pci: PciIdentity
    ports: tuple[PortIdentity, PortIdentity]

    @field_validator("node_guid")
    @classmethod
    def validate_node_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _GUID.fullmatch(normalized) is None:
            raise ValueError("node GUID must contain four lowercase 16-bit groups")
        return normalized

    @model_validator(mode="after")
    def validate_ports(self) -> "HcaIdentity":
        if tuple(port.port for port in self.ports) != (1, 2):
            raise ValueError("HCA ports must be exactly ordered ports 1 and 2")
        if self.ports[0].port_guid == self.ports[1].port_guid:
            raise ValueError("HCA port GUIDs must be unique")
        return self


class HostTools(StrictModel):
    python: str
    harness_script: str
    numactl: str
    ib_write_bw: str
    ib_write_bw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opensm: str | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> "HostTools":
        paths = {
            "python": self.python,
            "harness_script": self.harness_script,
            "numactl": self.numactl,
            "ib_write_bw": self.ib_write_bw,
        }
        for name, value in paths.items():
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be absolute")
        if self.opensm is not None and not Path(self.opensm).is_absolute():
            raise ValueError("opensm must be absolute")
        return self


class HostPreflightPolicy(StrictModel):
    maximum_load_1m_per_online_cpu: float = Field(ge=0.0)
    minimum_available_memory_bytes: int = Field(ge=0)
    allowed_cpu_frequency_governors: tuple[str, ...]
    required_cpu_flags: tuple[str, ...] = ("amx_bf16", "amx_int8", "amx_tile")

    @model_validator(mode="after")
    def validate_sets(self) -> "HostPreflightPolicy":
        if not math.isfinite(self.maximum_load_1m_per_online_cpu):
            raise ValueError("maximum load must be finite")
        for name, values in (
            ("allowed CPU frequency governors", self.allowed_cpu_frequency_governors),
            ("required CPU flags", self.required_cpu_flags),
        ):
            if (
                not values
                or tuple(sorted(set(values))) != values
                or any(_SAFE_IDENTIFIER.fullmatch(value) is None for value in values)
            ):
                raise ValueError(f"{name} must be sorted, unique, and nonempty")
        return self


class HostConfig(StrictModel):
    name: str
    management_address: str
    transport: Literal["local", "ssh"]
    ssh_target: str | None = None
    source_directory: str
    source: SourceIdentity
    preflight: HostPreflightPolicy
    cpu_set: tuple[int, ...]
    numa_nodes: tuple[int, ...]
    hca: HcaIdentity
    tools: HostTools

    @model_validator(mode="after")
    def validate_host(self) -> "HostConfig":
        if _SAFE_IDENTIFIER.fullmatch(self.name) is None:
            raise ValueError("host name is not a safe identifier")
        try:
            ipaddress.ip_address(self.management_address)
        except ValueError as error:
            raise ValueError("management_address must be an IP address") from error
        if self.transport == "local":
            if self.ssh_target is not None or self.tools.opensm is None:
                raise ValueError("local host requires opensm and no ssh_target")
        else:
            if (
                self.ssh_target is None
                or _SAFE_SSH_TARGET.fullmatch(self.ssh_target) is None
                or self.ssh_target.startswith("-")
                or "@" in self.ssh_target
            ):
                raise ValueError("SSH host requires a safe ssh_target")
            if self.tools.opensm is not None:
                raise ValueError("remote host must not configure opensm")
        if not Path(self.source_directory).is_absolute():
            raise ValueError("source_directory must be absolute")
        expected_harness = (
            Path(self.source_directory) / "scripts" / "two_host_ib_baseline.py"
        )
        if Path(self.tools.harness_script) != expected_harness:
            raise ValueError(
                "harness_script must be the exact script in source_directory"
            )
        if (
            not self.cpu_set
            or tuple(sorted(set(self.cpu_set))) != self.cpu_set
            or self.cpu_set[0] < 0
        ):
            raise ValueError("cpu_set must be sorted, unique, and nonempty")
        if (
            not self.numa_nodes
            or tuple(sorted(set(self.numa_nodes))) != self.numa_nodes
            or self.numa_nodes[0] < 0
        ):
            raise ValueError("numa_nodes must be sorted, unique, and nonempty")
        if self.hca.pci.numa_node not in self.numa_nodes:
            raise ValueError("HCA NUMA node must be included in numa_nodes")
        return self


class SshConfig(StrictModel):
    executable: str
    known_hosts_file: str
    identity_file: str
    user: str
    port: int = Field(ge=1, le=65535)
    connect_timeout_seconds: int = Field(ge=1, le=60)
    server_alive_interval_seconds: int = Field(ge=1, le=60)
    server_alive_count_max: int = Field(ge=1, le=10)

    @field_validator("executable", "known_hosts_file", "identity_file")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        if not Path(value).is_absolute() or "\0" in value:
            raise ValueError("SSH paths must be absolute and NUL-free")
        return value

    @field_validator("user")
    @classmethod
    def validate_user(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("SSH user must be a safe identifier")
        return value


class BenchmarkSpec(StrictModel):
    duration_seconds: int = Field(ge=2, le=3600)
    margin_seconds: int = Field(ge=0, le=60)
    message_bytes: int = Field(gt=0)
    tx_depth: int = Field(gt=0)
    queue_pairs: int = Field(gt=0)
    mtu: Literal[256, 512, 1024, 2048, 4096]
    single_port_1_control_port: int = Field(ge=1024, le=65535)
    single_port_2_control_port: int = Field(ge=1024, le=65535)
    dual_port_control_port: int = Field(ge=1024, le=65535)

    @model_validator(mode="after")
    def validate_control_ports(self) -> "BenchmarkSpec":
        values = self.control_ports
        if len(set(values)) != 3:
            raise ValueError("three distinct perftest control ports are required")
        if tuple(sorted(values)) != values:
            raise ValueError("perftest control ports must be in ascending case order")
        return self

    @property
    def control_ports(self) -> tuple[int, int, int]:
        return (
            self.single_port_1_control_port,
            self.single_port_2_control_port,
            self.dual_port_control_port,
        )


class TimeoutConfig(StrictModel):
    probe_seconds: float = Field(gt=0)
    opensm_start_seconds: float = Field(gt=0)
    rail_active_seconds: float = Field(gt=0)
    server_start_seconds: float = Field(gt=0)
    benchmark_seconds: float = Field(gt=0)
    cleanup_seconds: float = Field(gt=0)
    poll_seconds: float = Field(gt=0)


class BaselineConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str
    namespace: str
    result_directory: str
    ssh: SshConfig
    local_host: HostConfig
    remote_host: HostConfig
    benchmark_artifact_id: str = "rdma-core/ib_write_bw"
    benchmark_artifact_revision: str
    benchmark: BenchmarkSpec
    timeouts: TimeoutConfig

    @model_validator(mode="after")
    def validate_config(self) -> "BaselineConfig":
        if (
            _SAFE_IDENTIFIER.fullmatch(self.run_id) is None
            or _SAFE_IDENTIFIER.fullmatch(self.namespace) is None
        ):
            raise ValueError("run_id and namespace must be safe identifiers")
        if not Path(self.result_directory).is_absolute():
            raise ValueError("result_directory must be absolute")
        if self.local_host.transport != "local" or self.remote_host.transport != "ssh":
            raise ValueError("config requires one local coordinator and one SSH peer")
        if self.local_host.name == self.remote_host.name:
            raise ValueError("host names must be unique")
        if self.local_host.source != self.remote_host.source:
            raise ValueError("both hosts must bind the same exact source identity")
        hashes = {
            self.local_host.tools.ib_write_bw_sha256,
            self.remote_host.tools.ib_write_bw_sha256,
        }
        if (
            hashes != {self.benchmark_artifact_revision}
            or _HEX_REVISION.fullmatch(self.benchmark_artifact_revision) is None
        ):
            raise ValueError(
                "artifact revision must equal the common ib_write_bw SHA-256"
            )
        return self

    @property
    def hosts(self) -> tuple[HostConfig, HostConfig]:
        return (self.local_host, self.remote_host)

    @property
    def reserved_ports(self) -> tuple[int, int, int]:
        return self.benchmark.control_ports


CURRENT_CX3_IDENTITIES: JsonObject = {
    "dwagon": {
        "management_address": "192.168.40.24",
        "device": "mlx4_0",
        "node_guid": "0010:e000:0166:3a18",
        "pci": {
            "bdf": "0000:d8:00.0",
            "vendor_id": "0x15b3",
            "device_id": "0x1003",
            "driver": "mlx4_core",
            "current_width": 8,
            "maximum_width": 8,
            "current_speed_gtps": 8.0,
            "maximum_speed_gtps": 8.0,
            "numa_node": 1,
        },
        "ports": [
            {
                "port": 1,
                "port_guid": "0010:e000:0166:3a19",
                "gid_index": 0,
                "gid": "fe80:0000:0000:0000:0010:e000:0166:3a19",
                "expected_rate": "40 Gb/sec (4X QDR)",
            },
            {
                "port": 2,
                "port_guid": "0010:e000:0166:3a1a",
                "gid_index": 0,
                "gid": "fe80:0000:0000:0000:0010:e000:0166:3a1a",
                "expected_rate": "40 Gb/sec (4X QDR)",
            },
        ],
    },
    "fwuff": {
        "management_address": "192.168.40.248",
        "ssh_target": "fwuff",
        "device": "mlx4_0",
        "node_guid": "e41d:2d03:004d:32e0",
        "pci": {
            "bdf": "0000:16:00.0",
            "vendor_id": "0x15b3",
            "device_id": "0x1003",
            "driver": "mlx4_core",
            "current_width": 8,
            "maximum_width": 8,
            "current_speed_gtps": 8.0,
            "maximum_speed_gtps": 8.0,
            "numa_node": 0,
        },
        "ports": [
            {
                "port": 1,
                "port_guid": "e41d:2d03:004d:32e1",
                "gid_index": 0,
                "gid": "fe80:0000:0000:0000:e41d:2d03:004d:32e1",
                "expected_rate": "40 Gb/sec (4X QDR)",
            },
            {
                "port": 2,
                "port_guid": "e41d:2d03:004d:32e2",
                "gid_index": 0,
                "gid": "fe80:0000:0000:0000:e41d:2d03:004d:32e2",
                "expected_rate": "40 Gb/sec (4X QDR)",
            },
        ],
    },
}


def load_config(path: Path) -> BaselineConfig:
    try:
        return BaselineConfig.model_validate_json(path.read_text())
    except (OSError, ValidationError, ValueError) as error:
        raise BaselineError(f"invalid baseline config {path}: {error}") from error


@dataclass(frozen=True)
class LeasePreparation:
    metadata: JsonObject
    child_argv: tuple[str, ...]
    benchmark_lease_argv: tuple[str, ...]
    minimum_cleanup_grace_seconds: float


class LeaseMetadataValidator(Protocol):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


def _config_sha256(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def build_static_metadata(
    config: BaselineConfig, command: Sequence[str], config_digest: str
) -> JsonObject:
    hosts = [host.name for host in config.hosts]
    source = config.local_host.source
    value: dict[str, object] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "command": list(command),
        "git": {
            "commit": source.commit,
            "dirty": bool(source.dirty_file_hashes),
            "dirty_file_hashes": source.dirty_file_hashes,
        },
        "hosts": hosts,
        "models": [
            {
                "model_id": config.benchmark_artifact_id,
                "revision": config.benchmark_artifact_revision,
                "paths": {host.name: host.tools.ib_write_bw for host in config.hosts},
            }
        ],
        "gpu_bindings": {host.name: [] for host in config.hosts},
        "cpu_bindings": {
            host.name: {
                "cpu_set": ",".join(str(cpu) for cpu in host.cpu_set),
                "numa_nodes": list(host.numa_nodes),
                "memory_policy": "bind:"
                + ",".join(str(node) for node in host.numa_nodes),
            }
            for host in config.hosts
        },
        "hca_bindings": {
            host.name: [
                {"device": host.hca.device, "port": port.port, "gid": port.gid}
                for port in host.hca.ports
            ]
            for host in config.hosts
        },
        "source_deployments": {
            host.name: {
                "path": host.source_directory,
                "commit": host.source.commit,
                "dirty_file_hashes": host.source.dirty_file_hashes,
            }
            for host in config.hosts
        },
        "owner_pids": {host.name: [] for host in config.hosts},
        "benchmark_contract": {
            "kind": "ib_write_bw_single_rails_and_native_dualport",
            "config_sha256": config_digest,
            "duration_seconds": config.benchmark.duration_seconds,
            "message_bytes": config.benchmark.message_bytes,
            "control_ports": list(config.reserved_ports),
            "current_profile": "cx3-qdr-x8-x8",
        },
    }
    return cast(JsonObject, cast(object, value))


def minimum_cleanup_grace_seconds(config: BaselineConfig) -> float:
    return max(
        MINIMUM_CLEANUP_GRACE_SECONDS,
        2 * config.timeouts.cleanup_seconds + config.timeouts.probe_seconds + 30.0,
    )


def _atomic_create_json(path: Path, value: Mapping[str, object]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise BaselineError(
            "metadata output must be a new absolute file in an existing directory"
        )
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    parent_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
    except OSError as error:
        raise BaselineError(f"cannot create metadata output: {error}") from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def _validate_metadata(
    wrapper_script: Path, metadata: JsonObject, now: datetime
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_ib_baseline_lease", wrapper_script
    )
    if specification is None or specification.loader is None:
        raise BaselineError("cannot import benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    validator = cast(LeaseMetadataValidator, cast(object, module))
    if (
        validator.validate_run_metadata(cast(Mapping[str, object], metadata), now=now)
        != metadata
    ):
        raise BaselineError("benchmark lease changed generated metadata")


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
        wrapper_python,
        child_python,
        benchmark_lease_script,
        harness_script,
        lease_path,
        lock_path,
        result_root,
    ):
        if not path.is_absolute():
            raise BaselineError("all preparation paths must be absolute")
    if not owner.strip() or not purpose.strip() or "\0" in owner + purpose:
        raise BaselineError("owner and purpose must be nonempty and NUL-free")
    config = load_config(config_path)
    for path, description in (
        (Path(config.ssh.executable), "SSH executable"),
        (Path(config.ssh.known_hosts_file), "SSH known-hosts file"),
        (Path(config.ssh.identity_file), "SSH identity file"),
    ):
        if not path.is_file():
            raise BaselineError(f"{description} is not a regular file: {path}")
    if not os.access(config.ssh.executable, os.X_OK):
        raise BaselineError("configured SSH executable is not executable")
    source_directory = Path(config.local_host.source_directory)
    if (
        harness_script != Path(__file__).resolve()
        or harness_script != source_directory / "scripts" / Path(__file__).name
        or benchmark_lease_script != source_directory / "scripts" / "benchmark_lease.py"
    ):
        raise BaselineError("scripts must match the configured local source deployment")
    if (
        child_python != Path(config.local_host.tools.python)
        or wrapper_python != child_python
    ):
        raise BaselineError("wrapper/child Python must match the local host config")
    if Path(config.local_host.tools.harness_script) != harness_script:
        raise BaselineError("local harness path differs from the preparation script")
    observed_source = read_source_identity(source_directory)
    if observed_source != config.local_host.source:
        raise BaselineError(
            "configured source identity is stale for the local deployment"
        )
    if (
        Path(config.result_directory) != result_root / config.run_id
        or Path(config.result_directory).exists()
    ):
        raise BaselineError("result_directory must be a new result_root/run_id path")
    for value, label in (
        (expected_duration_seconds, "expected duration"),
        (cleanup_grace_seconds, "cleanup grace"),
        (heartbeat_seconds, "heartbeat"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise BaselineError(f"{label} must be finite and positive")
    minimum_grace = minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise BaselineError(f"cleanup grace must be at least {minimum_grace}")
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
    generated = now().astimezone(timezone.utc)
    metadata = build_static_metadata(config, child_argv, _config_sha256(config_path))
    metadata["generated_at"] = generated.isoformat(timespec="seconds")
    _validate_metadata(benchmark_lease_script, metadata, generated)
    wrapper_argv = (
        str(wrapper_python),
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
    _atomic_create_json(metadata_output, cast(Mapping[str, object], metadata))
    if read_source_identity(source_directory) != observed_source:
        raise BaselineError("source identity changed while preparing lease metadata")
    return LeasePreparation(metadata, child_argv, wrapper_argv, minimum_grace)


def _json_object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict):
        raise BaselineError(f"{description} must be a JSON object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise BaselineError(f"{description} must be a JSON object")
    return cast(JsonObject, cast(object, value))


def _parse_json_object(text: str, description: str) -> JsonObject:
    try:
        return _json_object(cast(object, json.loads(text)), description)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise BaselineError(f"invalid {description}: {error}") from error


class ResultDirectory:
    """Descriptor-anchored access to the lease wrapper's result directory."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = os.dup(descriptor)
        self._validate_identity()

    @classmethod
    def inherited(cls, path: Path) -> "ResultDirectory":
        raw = os.environ.get(RESULT_DIRECTORY_FD_ENVIRONMENT)
        if raw is None or not raw.isascii() or not raw.isdigit():
            raise BaselineError(
                f"lease wrapper did not pass {RESULT_DIRECTORY_FD_ENVIRONMENT}"
            )
        return cls(path, int(raw))

    def _validate_identity(self) -> None:
        try:
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise BaselineError(
                f"cannot validate result directory identity: {error}"
            ) from error
        if not stat.S_ISDIR(descriptor_stat.st_mode) or not stat.S_ISDIR(
            path_stat.st_mode
        ):
            raise BaselineError("result directory must remain a directory")
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise BaselineError("result directory path identity changed")

    def close(self) -> None:
        os.close(self.descriptor)

    def log_path(self, name: str) -> str:
        self._validate_name(name)
        return str(self.path / name)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise BaselineError("result filename must be one safe path component")

    def create_log(self, name: str) -> IO[bytes]:
        self._validate_identity()
        self._validate_name(name)
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.descriptor,
            )
        except OSError as error:
            raise BaselineError(f"cannot create result log {name}: {error}") from error
        return os.fdopen(descriptor, "wb", closefd=True)

    def read_log(self, name: str) -> str:
        self._validate_identity()
        self._validate_name(name)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            descriptor = os.open(name, flags, dir_fd=self.descriptor)
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size > MAX_CAPTURE_BYTES
            ):
                raise BaselineError(f"result log {name} is not a bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=True) as input_file:
                return input_file.read(MAX_CAPTURE_BYTES + 1).decode(
                    "utf-8", errors="replace"
                )
        except OSError as error:
            raise BaselineError(f"cannot read result log {name}: {error}") from error

    def overwrite_log(self, name: str, text: str) -> None:
        self._validate_identity()
        self._validate_name(name)
        payload = text.encode("utf-8")
        if len(payload) > MAX_CAPTURE_BYTES:
            raise BaselineError(f"result log {name} exceeds capture limit")
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.descriptor,
            )
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise BaselineError(f"result log {name} is not regular")
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(self.descriptor)
        except OSError as error:
            raise BaselineError(f"cannot update result log {name}: {error}") from error
        self._validate_identity()

    def write_json(
        self, name: str, value: Mapping[str, object], *, replace: bool
    ) -> None:
        self._validate_identity()
        self._validate_name(name)
        payload = (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            if not replace:
                try:
                    os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise BaselineError(f"result fragment already exists: {name}")
            else:
                with contextlib.suppress(FileNotFoundError):
                    current = os.stat(
                        name, dir_fd=self.descriptor, follow_symlinks=False
                    )
                    if not stat.S_ISREG(current.st_mode):
                        raise BaselineError(f"result fragment is not regular: {name}")
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
            raise BaselineError(
                f"cannot write result fragment {name}: {error}"
            ) from error
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self.descriptor)
        self._validate_identity()


class PortObservation(StrictModel):
    port: int
    port_guid: str
    gid: str
    state: str
    physical_state: str
    rate: str
    lid: int = Field(ge=0)
    sm_lid: int = Field(ge=0)
    counters: dict[str, int]


class HostObservation(StrictModel):
    hostname: str
    hca_device: str
    node_guid: str
    pci_bdf: str
    vendor_id: str
    device_id: str
    driver: str
    current_width: int
    maximum_width: int
    current_speed_gtps: float
    maximum_speed_gtps: float
    numa_node: int
    ports: tuple[PortObservation, PortObservation]
    ib_write_bw_sha256: str
    source: SourceIdentity
    numa_cpu_sets: dict[str, tuple[int, ...]]
    online_cpu_count: int = Field(gt=0)
    load_average_1m: float = Field(ge=0.0)
    load_average_5m: float = Field(ge=0.0)
    load_average_15m: float = Field(ge=0.0)
    memory_total_bytes: int = Field(gt=0)
    memory_available_bytes: int = Field(ge=0)
    memory_free_bytes: int = Field(ge=0)
    cpu_frequency_governors: dict[str, str]
    cpu_flags: tuple[str, ...]
    raid_sync_conflicts: tuple[str, ...]
    gpu_bindings: tuple[str, ...]
    unused_reserved_ports: tuple[int, ...]
    conflicts: tuple[str, ...]


class HostProbeRequest(StrictModel):
    host: HostConfig
    reserved_ports: tuple[int, ...]

    @model_validator(mode="after")
    def validate_ports(self) -> "HostProbeRequest":
        if self.reserved_ports and (
            tuple(sorted(set(self.reserved_ports))) != self.reserved_ports
            or self.reserved_ports[0] < 1
            or self.reserved_ports[-1] > 65535
        ):
            raise ValueError(
                "reserved_ports must be empty or sorted, unique, and valid"
            )
        return self


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        raise BaselineError(f"cannot read {description} at {path}: {error}") from error


def _integer_text(path: Path, description: str, *, base: int = 10) -> int:
    text = _read_text(path, description)
    try:
        return int(text, base)
    except ValueError as error:
        raise BaselineError(f"invalid integer in {description}: {text!r}") from error


def _link_width(path: Path, description: str) -> int:
    match = re.search(r"[0-9]+", _read_text(path, description))
    if match is None:
        raise BaselineError(f"cannot parse {description}")
    return int(match.group())


def _link_speed(path: Path, description: str) -> float:
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", _read_text(path, description))
    if match is None:
        raise BaselineError(f"cannot parse {description}")
    return float(match.group())


def parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    try:
        for component in value.strip().split(","):
            if not component:
                raise ValueError("empty CPU-list component")
            if "-" in component:
                first_text, last_text = component.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 0 or last < first:
                    raise ValueError("invalid CPU range")
                cpus.update(range(first, last + 1))
            else:
                cpu = int(component)
                if cpu < 0:
                    raise ValueError("negative CPU")
                cpus.add(cpu)
    except ValueError as error:
        raise BaselineError(f"invalid Linux CPU list {value!r}: {error}") from error
    if not cpus:
        raise BaselineError("Linux CPU list must not be empty")
    return tuple(sorted(cpus))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while block := input_file.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise BaselineError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def read_source_identity(source_directory: Path) -> SourceIdentity:
    def git_output(arguments: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(source_directory), *arguments),
                capture_output=True,
                text=True,
                check=False,
                timeout=30.0,
                env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BaselineError(f"cannot inspect source identity: {error}") from error
        if completed.returncode != 0:
            raise BaselineError(
                f"git {' '.join(arguments)} failed: {completed.stderr[-300:]}"
            )
        return completed.stdout

    commit = git_output(("rev-parse", "HEAD")).strip()
    tracked = git_output(("diff", "--name-only", "-z", "HEAD"))
    untracked = git_output(("ls-files", "--others", "--exclude-standard", "-z"))
    relative_paths = sorted(
        {path for path in (*tracked.split("\0"), *untracked.split("\0")) if path}
    )
    hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise BaselineError(f"git returned an unsafe source path {relative_path!r}")
        path = source_directory / relative_path
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            payload = b"<deleted>"
        except OSError as error:
            raise BaselineError(
                f"cannot inspect dirty source {relative_path}: {error}"
            ) from error
        else:
            if stat.S_ISREG(path_stat.st_mode):
                hashes[relative_path] = file_sha256(path)
                continue
            if stat.S_ISLNK(path_stat.st_mode):
                payload = b"<symlink>\0" + os.readlink(path).encode()
            else:
                raise BaselineError(
                    f"dirty source path is not regular/symlink: {relative_path}"
                )
        hashes[relative_path] = hashlib.sha256(payload).hexdigest()
    try:
        return SourceIdentity(commit=commit, dirty_file_hashes=hashes)
    except ValidationError as error:
        raise BaselineError(f"source identity is invalid: {error}") from error


def classify_process_conflict(
    arguments: Sequence[str], environment: Sequence[str] = ()
) -> tuple[str, ...]:
    """Classify only known benchmark-affecting process command shapes."""
    if not arguments:
        return ()
    executable = Path(arguments[0]).name
    classes: set[str] = set()
    if executable in _PERFTEST_EXECUTABLES or re.fullmatch(
        r"ib_[a-z0-9_]+_(?:bw|lat)", executable
    ):
        classes.add("opensm_or_perftest")
    if executable in _INFERENCE_EXECUTABLES or re.fullmatch(
        r"(?:all_gather|all_reduce|broadcast|gather|hypercube|reduce|"
        r"reduce_scatter|scatter|sendrecv)_perf(?:_mpi)?",
        executable,
    ):
        classes.add("exo_nccl_or_model_server")
    if executable in _STORAGE_EXECUTABLES:
        classes.add("heavy_storage")
    if executable in {"hf", "huggingface-cli"} and "download" in arguments[1:]:
        classes.add("heavy_storage")
    if executable in {"btrfs", "zpool"} and "scrub" in arguments[1:]:
        classes.add("heavy_storage")
    if executable == "mdadm" and any(
        argument in {"--check", "--action=check", "--action=repair"}
        for argument in arguments[1:]
    ):
        classes.add("heavy_storage")
    if any(
        value.startswith(_EXO_PROCESS_ENVIRONMENT_PREFIXES) for value in environment
    ):
        classes.add("exo_nccl_or_model_server")
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", executable):
        for index, argument in enumerate(arguments[:-1]):
            if argument != "-c":
                continue
            spawn_code = " ".join(arguments[index + 1].split())
            if (
                spawn_code.startswith(
                    "from multiprocessing.spawn import spawn_main; spawn_main("
                )
                and "tracker_fd=" in spawn_code
                and "pipe_handle=" in spawn_code
                and "--multiprocessing-fork" in arguments[index + 2 :]
            ):
                classes.add("exo_runner_or_python_multiprocessing_worker")

    for index, argument in enumerate(arguments[:-1]):
        if argument == "-m":
            module = arguments[index + 1]
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _INFERENCE_MODULE_PREFIXES
            ):
                classes.add("exo_nccl_or_model_server")

    script_names = {Path(argument).name for argument in arguments[1:]}
    if script_names & _INFERENCE_SCRIPT_NAMES:
        classes.add("exo_nccl_or_model_server")
    normalized_paths = tuple(argument.replace("\\", "/") for argument in arguments)
    if any(
        path.endswith(
            (
                "/src/exo/main.py",
                "/exo/main.py",
                "/exo/worker/runner/bootstrap.py",
                "/sglang/launch_server.py",
                "/vllm/entrypoints/openai/api_server.py",
            )
        )
        for path in normalized_paths
    ):
        classes.add("exo_nccl_or_model_server")
    if executable in {"uv", "uvx"} and any(
        argument == "exo" or Path(argument).name == "exo" for argument in arguments[1:]
    ):
        classes.add("exo_nccl_or_model_server")
    return tuple(sorted(classes))


def process_conflicts(
    proc_root: Path = Path("/proc"), *, ignored_pids: Sequence[int] = ()
) -> tuple[str, ...]:
    ignored = {*ignored_pids, os.getpid()}
    conflicts: list[str] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise BaselineError(f"cannot scan process table: {error}") from error
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) in ignored:
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise BaselineError(
                f"cannot inspect process {entry.name}: {error}"
            ) from error
        if not command:
            continue
        arguments = [
            part.decode(errors="replace") for part in command.split(b"\0") if part
        ]
        try:
            raw_environment = (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise BaselineError(
                f"cannot inspect process {entry.name} environment: {error}"
            ) from error
        environment = [
            part.decode(errors="replace")
            for part in raw_environment.split(b"\0")
            if part
        ]
        matched = classify_process_conflict(arguments, environment)
        if matched:
            conflicts.append(
                f"pid={entry.name} classes={','.join(matched)} argv={shlex.join(arguments)[:400]}"
            )
    return tuple(conflicts)


def _load_averages(proc_root: Path) -> tuple[float, float, float]:
    fields = _read_text(proc_root / "loadavg", "load average").split()
    try:
        values = tuple(float(field) for field in fields[:3])
    except ValueError as error:
        raise BaselineError("cannot parse load average") from error
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise BaselineError("load average must contain three finite nonnegative values")
    return values


def _memory_bytes(proc_root: Path) -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in _read_text(proc_root / "meminfo", "memory information").splitlines():
        name, separator, raw_value = line.partition(":")
        if name not in {"MemTotal", "MemAvailable", "MemFree"}:
            continue
        fields = raw_value.split()
        if separator != ":" or len(fields) != 2 or fields[1] != "kB":
            raise BaselineError(f"cannot parse {name} in memory information")
        try:
            values[name] = int(fields[0]) * 1024
        except ValueError as error:
            raise BaselineError(f"cannot parse {name} in memory information") from error
    if set(values) != {"MemTotal", "MemAvailable", "MemFree"}:
        raise BaselineError("memory information is missing required fields")
    total, available, free = (
        values["MemTotal"],
        values["MemAvailable"],
        values["MemFree"],
    )
    if min(total, available, free) < 0 or max(available, free) > total:
        raise BaselineError("memory information contains inconsistent values")
    return total, available, free


def _cpu_flags(proc_root: Path, selected_cpus: Sequence[int]) -> tuple[str, ...]:
    if not selected_cpus:
        raise BaselineError("selected CPU set must not be empty")
    records: dict[int, set[str]] = {}
    processor: int | None = None
    flags: set[str] | None = None
    text = _read_text(proc_root / "cpuinfo", "CPU information")
    for line in (*text.splitlines(), ""):
        if line:
            name, separator, raw_value = line.partition(":")
            if separator != ":":
                continue
            if name.strip() == "processor":
                try:
                    processor = int(raw_value.strip())
                except ValueError as error:
                    raise BaselineError("cannot parse CPU processor index") from error
            elif name.strip() in {"flags", "Features"}:
                flags = set(raw_value.split())
            continue
        if processor is not None and flags is not None:
            records[processor] = flags
        processor = None
        flags = None
    missing = sorted(set(selected_cpus) - records.keys())
    if missing:
        raise BaselineError(f"CPU information is missing selected CPUs {missing}")
    common = set(records[selected_cpus[0]])
    for cpu in selected_cpus[1:]:
        common.intersection_update(records[cpu])
    return tuple(sorted(common))


def _cpu_frequency_governors(
    sys_cpu_root: Path, selected_cpus: Sequence[int]
) -> dict[str, str]:
    return {
        str(cpu): _read_text(
            sys_cpu_root / f"cpu{cpu}" / "cpufreq" / "scaling_governor",
            f"CPU {cpu} frequency governor",
        )
        for cpu in selected_cpus
    }


def raid_sync_conflicts(sys_block_root: Path = Path("/sys/block")) -> tuple[str, ...]:
    conflicts: list[str] = []
    try:
        devices = tuple(sys_block_root.glob("md*"))
    except OSError as error:
        raise BaselineError(f"cannot inspect RAID devices: {error}") from error
    for device in devices:
        action_path = device / "md" / "sync_action"
        if not action_path.exists():
            continue
        action = _read_text(action_path, f"{device.name} RAID sync action")
        if action not in _INACTIVE_RAID_SYNC_ACTIONS:
            conflicts.append(f"{device.name}:{action}")
    return tuple(sorted(conflicts))


def probe_reserved_ports_unused(ports: Sequence[int]) -> tuple[int, ...]:
    """Bind every TCP/UDP wildcard endpoint and release only after all succeed.

    The remote supervisor repeats this immediately before ``ib_write_bw`` exec.
    A process winning the unavoidable close/exec race makes perftest fail, so the
    run remains non-reportable rather than silently using an unintended socket.
    """
    normalized = tuple(ports)
    if (
        not normalized
        or tuple(sorted(set(normalized))) != normalized
        or normalized[0] < 1
        or normalized[-1] > 65535
    ):
        raise PortConflictError("reserved ports must be sorted, unique, and valid")
    sockets: list[socket.socket] = []
    try:
        for port in normalized:
            endpoints = (
                (socket.AF_INET, socket.SOCK_STREAM, ("0.0.0.0", port)),
                (socket.AF_INET, socket.SOCK_DGRAM, ("0.0.0.0", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, ("::", port)),
                (socket.AF_INET6, socket.SOCK_DGRAM, ("::", port)),
            )
            for family, socket_type, address in endpoints:
                probe = socket.socket(family, socket_type)
                sockets.append(probe)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind(address)
                if socket_type == socket.SOCK_STREAM:
                    probe.listen(1)
    except OSError as error:
        raise PortConflictError(
            f"reserved port availability probe failed: {error}"
        ) from error
    finally:
        for probe in reversed(sockets):
            probe.close()
    return normalized


def _probe_requested_reserved_ports(ports: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(ports)
    if not normalized:
        return ()
    return probe_reserved_ports_unused(normalized)


def _port_guid_from_gid(gid: str) -> str:
    packed = ipaddress.IPv6Address(gid).packed[-8:].hex()
    return ":".join(packed[index : index + 4] for index in range(0, 16, 4))


def collect_host_observation(
    host: HostConfig,
    *,
    reserved_ports: Sequence[int],
    ignored_pids: Sequence[int] = (),
    sys_infiniband_root: Path = Path("/sys/class/infiniband"),
    sys_pci_root: Path = Path("/sys/bus/pci/devices"),
    sys_node_root: Path = Path("/sys/devices/system/node"),
    sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
    sys_block_root: Path = Path("/sys/block"),
    proc_root: Path = Path("/proc"),
    hostname: Callable[[], str] = socket.gethostname,
) -> HostObservation:
    hca_root = sys_infiniband_root / host.hca.device
    try:
        pci_bdf = (hca_root / "device").resolve(strict=True).name.lower()
    except OSError as error:
        raise BaselineError(f"cannot resolve HCA PCI device: {error}") from error
    pci_root = sys_pci_root / pci_bdf
    try:
        driver = (pci_root / "driver").resolve(strict=True).name
    except OSError as error:
        raise BaselineError(f"cannot resolve HCA driver: {error}") from error
    counter_names = (
        "port_xmit_data",
        "port_rcv_data",
        "port_xmit_packets",
        "port_rcv_packets",
    )
    ports: list[PortObservation] = []
    for configured in host.hca.ports:
        root = hca_root / "ports" / str(configured.port)
        gid = ipaddress.IPv6Address(
            _read_text(root / "gids" / str(configured.gid_index), "port GID")
        ).exploded
        counters = {
            name: _integer_text(
                root / "counters" / name, f"port {configured.port} {name}"
            )
            for name in counter_names
        }
        ports.append(
            PortObservation(
                port=configured.port,
                port_guid=_port_guid_from_gid(gid),
                gid=gid,
                state=_read_text(root / "state", "port state"),
                physical_state=_read_text(root / "phys_state", "port physical state"),
                rate=_read_text(root / "rate", "port rate"),
                lid=_integer_text(root / "lid", "port LID", base=0),
                sm_lid=_integer_text(root / "sm_lid", "port SM LID", base=0),
                counters=counters,
            )
        )
    load_average_1m, load_average_5m, load_average_15m = _load_averages(proc_root)
    memory_total, memory_available, memory_free = _memory_bytes(proc_root)
    online_cpus = parse_cpu_list(_read_text(sys_cpu_root / "online", "online CPUs"))
    return HostObservation(
        hostname=hostname(),
        hca_device=host.hca.device,
        node_guid=_read_text(hca_root / "node_guid", "node GUID").lower(),
        pci_bdf=pci_bdf,
        vendor_id=_read_text(pci_root / "vendor", "PCI vendor ID").lower(),
        device_id=_read_text(pci_root / "device", "PCI device ID").lower(),
        driver=driver,
        current_width=_link_width(
            pci_root / "current_link_width", "current PCIe width"
        ),
        maximum_width=_link_width(pci_root / "max_link_width", "maximum PCIe width"),
        current_speed_gtps=_link_speed(
            pci_root / "current_link_speed", "current PCIe speed"
        ),
        maximum_speed_gtps=_link_speed(
            pci_root / "max_link_speed", "maximum PCIe speed"
        ),
        numa_node=_integer_text(pci_root / "numa_node", "PCI NUMA node"),
        ports=(ports[0], ports[1]),
        ib_write_bw_sha256=file_sha256(Path(host.tools.ib_write_bw)),
        source=read_source_identity(Path(host.source_directory)),
        numa_cpu_sets={
            str(node): parse_cpu_list(
                _read_text(sys_node_root / f"node{node}" / "cpulist", "NUMA CPU list")
            )
            for node in host.numa_nodes
        },
        online_cpu_count=len(online_cpus),
        load_average_1m=load_average_1m,
        load_average_5m=load_average_5m,
        load_average_15m=load_average_15m,
        memory_total_bytes=memory_total,
        memory_available_bytes=memory_available,
        memory_free_bytes=memory_free,
        cpu_frequency_governors=_cpu_frequency_governors(sys_cpu_root, host.cpu_set),
        cpu_flags=_cpu_flags(proc_root, host.cpu_set),
        raid_sync_conflicts=raid_sync_conflicts(sys_block_root),
        gpu_bindings=(),
        unused_reserved_ports=_probe_requested_reserved_ports(reserved_ports),
        conflicts=process_conflicts(proc_root, ignored_pids=ignored_pids),
    )


def validate_host_observation(
    observation: HostObservation,
    host: HostConfig,
    *,
    reserved_ports: Sequence[int],
    require_active: bool,
) -> None:
    pci = host.hca.pci
    expected_scalars: dict[str, object] = {
        "hostname": host.name,
        "hca_device": host.hca.device,
        "node_guid": host.hca.node_guid,
        "pci_bdf": pci.bdf,
        "vendor_id": pci.vendor_id,
        "device_id": pci.device_id,
        "driver": pci.driver,
        "current_width": pci.current_width,
        "maximum_width": pci.maximum_width,
        "current_speed_gtps": pci.current_speed_gtps,
        "maximum_speed_gtps": pci.maximum_speed_gtps,
        "numa_node": pci.numa_node,
        "ib_write_bw_sha256": host.tools.ib_write_bw_sha256,
        "source": host.source,
    }
    for name, expected in expected_scalars.items():
        if getattr(observation, name) != expected:
            raise BaselineError(
                f"{host.name} {name} differs: observed {getattr(observation, name)!r}, expected {expected!r}"
            )
    if observation.conflicts:
        raise ProcessConflictError(
            f"{host.name} has unowned benchmark-affecting processes: "
            f"{'; '.join(observation.conflicts)}"
        )
    if observation.raid_sync_conflicts:
        raise ProcessConflictError(
            f"{host.name} has active RAID work: "
            f"{'; '.join(observation.raid_sync_conflicts)}"
        )
    if observation.gpu_bindings:
        raise BaselineError(f"{host.name} IB-only baseline has nonempty GPU bindings")
    if observation.unused_reserved_ports != tuple(reserved_ports):
        raise PortConflictError(
            f"{host.name} did not prove every exact reserved port unused"
        )
    available_cpus = {
        cpu for cpus in observation.numa_cpu_sets.values() for cpu in cpus
    }
    if set(observation.numa_cpu_sets) != {str(node) for node in host.numa_nodes}:
        raise BaselineError(f"{host.name} NUMA CPU observations are incomplete")
    if not set(host.cpu_set) <= available_cpus:
        raise BaselineError(
            f"{host.name} selected CPU set is outside configured NUMA nodes"
        )
    policy = host.preflight
    normalized_load = observation.load_average_1m / observation.online_cpu_count
    if normalized_load > policy.maximum_load_1m_per_online_cpu:
        raise BaselineError(
            f"{host.name} 1-minute load per online CPU {normalized_load:.4f} exceeds "
            f"{policy.maximum_load_1m_per_online_cpu:.4f}"
        )
    if observation.memory_available_bytes < policy.minimum_available_memory_bytes:
        raise BaselineError(
            f"{host.name} available memory {observation.memory_available_bytes} is below "
            f"{policy.minimum_available_memory_bytes}"
        )
    expected_governor_cpus = {str(cpu) for cpu in host.cpu_set}
    if set(observation.cpu_frequency_governors) != expected_governor_cpus:
        raise BaselineError(f"{host.name} CPU frequency observations are incomplete")
    disallowed_governors = {
        governor
        for governor in observation.cpu_frequency_governors.values()
        if governor not in policy.allowed_cpu_frequency_governors
    }
    if disallowed_governors:
        raise BaselineError(
            f"{host.name} has disallowed CPU frequency governors: "
            f"{sorted(disallowed_governors)}"
        )
    missing_cpu_flags = set(policy.required_cpu_flags) - set(observation.cpu_flags)
    if missing_cpu_flags:
        raise BaselineError(
            f"{host.name} is missing required CPU flags: {sorted(missing_cpu_flags)}"
        )
    for observed, configured in zip(observation.ports, host.hca.ports, strict=True):
        if (observed.port, observed.port_guid, observed.gid, observed.rate) != (
            configured.port,
            configured.port_guid,
            configured.gid,
            configured.expected_rate,
        ):
            raise BaselineError(
                f"{host.name} port {configured.port} identity/rate differs"
            )
        if "LINKUP" not in observed.physical_state.upper():
            raise BaselineError(
                f"{host.name} port {configured.port} is not physically LinkUp"
            )
        if require_active and "ACTIVE" not in observed.state.upper():
            raise BaselineError(f"{host.name} port {configured.port} is not ACTIVE")


def validate_active_rails(
    local: HostObservation, remote: HostObservation, config: BaselineConfig
) -> None:
    validate_host_observation(
        local,
        config.local_host,
        reserved_ports=(),
        require_active=True,
    )
    validate_host_observation(
        remote,
        config.remote_host,
        reserved_ports=(),
        require_active=True,
    )
    for local_port, remote_port in zip(local.ports, remote.ports, strict=True):
        if (
            min(local_port.lid, remote_port.lid, local_port.sm_lid, remote_port.sm_lid)
            <= 0
        ):
            raise BaselineError(
                f"rail {local_port.port} has an unassigned LID or SM LID"
            )
        if local_port.sm_lid != local_port.lid or remote_port.sm_lid != local_port.lid:
            raise BaselineError(
                f"rail {local_port.port} does not agree on the local OpenSM LID"
            )


@dataclass(frozen=True)
class OwnedProcess:
    host_name: str
    kind: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    transport_pid: int
    namespace: str
    owner_token: str
    log_path: str


@dataclass(frozen=True)
class CleanupReceipt:
    host_name: str
    kind: str
    ownership_verified: bool
    terminated: bool
    forced: bool
    error: str | None = None


@dataclass
class LocalHandle:
    receipt: OwnedProcess
    process: subprocess.Popen[bytes]
    log_file: IO[bytes]
    log_name: str


@dataclass
class RemoteHandle:
    receipt: OwnedProcess
    transport_receipt: OwnedProcess
    transport: subprocess.Popen[str]
    log_name: str
    request: JsonObject


def process_identity(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    try:
        text = (proc_root / str(pid) / "stat").read_text()
    except OSError as error:
        raise BaselineError(f"cannot read process {pid} identity: {error}") from error
    try:
        fields = text.rsplit(")", 1)[1].split()
        return int(fields[2]), int(fields[19])
    except (IndexError, ValueError) as error:
        raise BaselineError(f"cannot parse process {pid} identity") from error


def _group_ownership(
    receipt: OwnedProcess, proc_root: Path = Path("/proc")
) -> tuple[bool, tuple[int, ...]]:
    members: list[int] = []
    token = f"EXO_BENCHMARK_OWNER_TOKEN={receipt.owner_token}".encode()
    namespace = f"EXO_BENCHMARK_NAMESPACE={receipt.namespace}".encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            process_group_id, start_ticks = process_identity(pid, proc_root)
        except BaselineError:
            continue
        if process_group_id != receipt.process_group_id:
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            return False, tuple(members)
        if token not in environment or namespace not in environment:
            return False, tuple((*members, pid))
        if pid == receipt.pid and start_ticks != receipt.start_time_ticks:
            return False, tuple((*members, pid))
        members.append(pid)
    return True, tuple(sorted(members))


def _wait_local_gone(handle: LocalHandle, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        handle.process.poll()
        ownership, members = _group_ownership(handle.receipt)
        if ownership and not members and handle.process.poll() is not None:
            handle.process.wait(timeout=1.0)
            return True
        time.sleep(0.05)
    return False


def stop_local_owned(handle: LocalHandle, timeout_seconds: float) -> CleanupReceipt:
    receipt = handle.receipt
    forced = False
    error: str | None = None
    try:
        handle.process.poll()
        ownership, members = _group_ownership(receipt)
        if not ownership:
            return CleanupReceipt(
                receipt.host_name,
                receipt.kind,
                False,
                False,
                False,
                "process group ownership changed",
            )
        if members:
            os.killpg(receipt.process_group_id, signal.SIGTERM)
            if not _wait_local_gone(handle, timeout_seconds):
                ownership, members = _group_ownership(receipt)
                if not ownership:
                    return CleanupReceipt(
                        receipt.host_name,
                        receipt.kind,
                        False,
                        False,
                        False,
                        "ownership changed before SIGKILL",
                    )
                if members:
                    forced = True
                    os.killpg(receipt.process_group_id, signal.SIGKILL)
                if not _wait_local_gone(handle, min(timeout_seconds, 5.0)):
                    error = "owned process group survived SIGKILL"
        elif handle.process.poll() is None:
            error = "process leader is alive outside its recorded process group"
        if handle.process.poll() is not None:
            handle.process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired, BaselineError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        with contextlib.suppress(OSError):
            handle.log_file.flush()
        with contextlib.suppress(OSError):
            handle.log_file.close()
    terminated = (
        error is None
        and handle.process.poll() is not None
        and not _group_ownership(receipt)[1]
    )
    return CleanupReceipt(
        receipt.host_name, receipt.kind, True, terminated, forced, error
    )


def stop_owned_ssh_transport(
    process: subprocess.Popen[str], receipt: OwnedProcess, timeout_seconds: float
) -> CleanupReceipt:
    forced = False
    error: str | None = None
    ownership_verified = False
    final_ownership = False
    final_members: tuple[int, ...] = ()
    try:
        process.poll()
        ownership_verified, members = _group_ownership(receipt)
        if not ownership_verified:
            return CleanupReceipt(
                receipt.host_name,
                receipt.kind,
                False,
                False,
                False,
                "SSH transport group ownership changed",
            )
        if members:
            os.killpg(receipt.process_group_id, signal.SIGTERM)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                process.poll()
                ownership_verified, members = _group_ownership(receipt)
                if not ownership_verified or not members:
                    break
                time.sleep(0.05)
            if not ownership_verified:
                return CleanupReceipt(
                    receipt.host_name,
                    receipt.kind,
                    False,
                    False,
                    False,
                    "SSH transport ownership changed after SIGTERM",
                )
            if members:
                forced = True
                os.killpg(receipt.process_group_id, signal.SIGKILL)
                deadline = time.monotonic() + min(timeout_seconds, 5.0)
                while time.monotonic() < deadline:
                    process.poll()
                    ownership_verified, members = _group_ownership(receipt)
                    if not ownership_verified or not members:
                        break
                    time.sleep(0.05)
        if process.poll() is not None:
            process.wait(timeout=1.0)
        ownership_verified, members = _group_ownership(receipt)
        if not ownership_verified:
            error = "SSH transport ownership changed during cleanup"
        elif members:
            error = "SSH transport process group survived cleanup"
        elif process.poll() is None:
            error = "SSH transport leader survived outside its process group"
        final_ownership = ownership_verified
        final_members = members
    except (OSError, subprocess.TimeoutExpired, BaselineError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    ownership_verified = ownership_verified and final_ownership
    terminated = (
        error is None
        and ownership_verified
        and process.poll() is not None
        and not final_members
    )
    return CleanupReceipt(
        receipt.host_name,
        receipt.kind,
        ownership_verified,
        terminated,
        forced,
        error,
    )


def build_ssh_argv(config: BaselineConfig, remote_subcommand: str) -> tuple[str, ...]:
    remote = config.remote_host
    assert remote.ssh_target is not None
    remote_command = shlex.join(
        (remote.tools.python, remote.tools.harness_script, remote_subcommand)
    )
    return (
        config.ssh.executable,
        "-F",
        "/dev/null",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={config.ssh.known_hosts_file}",
        "-o",
        f"IdentityFile={config.ssh.identity_file}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ProxyCommand=none",
        "-o",
        "ProxyJump=none",
        "-o",
        "CanonicalizeHostname=no",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "RequestTTY=no",
        "-o",
        f"ConnectTimeout={config.ssh.connect_timeout_seconds}",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "PubkeyAuthentication=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"ServerAliveInterval={config.ssh.server_alive_interval_seconds}",
        "-o",
        f"ServerAliveCountMax={config.ssh.server_alive_count_max}",
        "-o",
        f"HostName={remote.ssh_target}",
        "-l",
        config.ssh.user,
        "-p",
        str(config.ssh.port),
        "--",
        remote.ssh_target,
        remote_command,
    )


def _read_line_bounded(
    stream: IO[str], timeout_seconds: float, description: str
) -> str:
    ready, _, _ = select.select([stream], [], [], timeout_seconds)
    if not ready:
        raise BaselineError(f"timed out waiting for {description}")
    line = stream.readline()
    if not line:
        raise BaselineError(f"stream ended before {description}")
    if len(line.encode()) > MAX_CAPTURE_BYTES:
        raise BaselineError(f"{description} exceeded capture limit")
    return line


def build_remote_request(
    host: HostConfig,
    command: Sequence[str],
    owner_token: str,
    namespace: str,
    timeout_seconds: float,
    reserved_ports: Sequence[int],
) -> JsonObject:
    return {
        "schema_version": 1,
        "host": cast(JsonValue, host.model_dump(mode="json")),
        "command": list(command),
        "owner_token": owner_token,
        "namespace": namespace,
        "timeout_seconds": timeout_seconds,
        "reserved_ports": list(reserved_ports),
    }


class BaselineEffects(Protocol):
    def probe_local(
        self,
        host: HostConfig,
        *,
        reserved_ports: Sequence[int],
        ignored_pids: Sequence[int] = (),
    ) -> HostObservation: ...
    def probe_remote(
        self, host: HostConfig, *, reserved_ports: Sequence[int]
    ) -> HostObservation: ...
    def start_opensm(self, port: PortIdentity, owner_token: str) -> LocalHandle: ...
    def start_remote_server(
        self,
        command: Sequence[str],
        kind: str,
        owner_token: str,
        timeout_seconds: float,
    ) -> RemoteHandle: ...
    def start_local_client(
        self, command: Sequence[str], kind: str, owner_token: str
    ) -> LocalHandle: ...
    def wait_local(
        self, handle: LocalHandle, timeout_seconds: float
    ) -> tuple[int, str, CleanupReceipt]: ...
    def wait_remote(
        self, handle: RemoteHandle, timeout_seconds: float
    ) -> tuple[int, str, CleanupReceipt]: ...
    def stop_local(
        self, handle: LocalHandle, timeout_seconds: float
    ) -> CleanupReceipt: ...
    def stop_remote(
        self, handle: RemoteHandle, timeout_seconds: float
    ) -> CleanupReceipt: ...


class SystemEffects:
    def __init__(self, config: BaselineConfig, results: ResultDirectory) -> None:
        self.config = config
        self.results = results

    def probe_local(
        self,
        host: HostConfig,
        *,
        reserved_ports: Sequence[int],
        ignored_pids: Sequence[int] = (),
    ) -> HostObservation:
        return collect_host_observation(
            host,
            reserved_ports=reserved_ports,
            ignored_pids=ignored_pids,
            proc_root=Path("/proc"),
            hostname=socket.gethostname,
        )

    def probe_remote(
        self, host: HostConfig, *, reserved_ports: Sequence[int]
    ) -> HostObservation:
        completed = subprocess.run(
            build_ssh_argv(self.config, "host-probe"),
            input=HostProbeRequest(
                host=host, reserved_ports=tuple(reserved_ports)
            ).model_dump_json()
            + "\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=self.config.timeouts.probe_seconds,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        if completed.returncode != 0:
            raise BaselineError(
                f"remote host probe failed ({completed.returncode}): {completed.stderr[-500:]}"
            )
        return HostObservation.model_validate_json(completed.stdout)

    def _start_local(
        self, command: Sequence[str], kind: str, owner_token: str, log_name: str
    ) -> LocalHandle:
        log_file = self.results.create_log(log_name)
        environment = {
            "PATH": os.defpath,
            "LC_ALL": "C",
            "LANG": "C",
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "EXO_BENCHMARK_NAMESPACE": self.config.namespace,
        }
        process: subprocess.Popen[bytes] | None = None
        receipt: OwnedProcess | None = None
        try:
            process = subprocess.Popen(
                tuple(command),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            process_group_id, start_ticks = process_identity(process.pid)
            if process_group_id != process.pid:
                raise BaselineError(f"{kind} is not its process-group leader")
            receipt = OwnedProcess(
                self.config.local_host.name,
                kind,
                process.pid,
                process_group_id,
                start_ticks,
                process.pid,
                self.config.namespace,
                owner_token,
                self.results.log_path(log_name),
            )
            ownership, members = _group_ownership(receipt)
            if not ownership or process.pid not in members:
                raise BaselineError(f"cannot verify {kind} ownership")
            return LocalHandle(receipt, process, log_file, log_name)
        except BaseException as start_error:
            cleanup_error: str | None = None
            if process is not None:
                if receipt is None and process.poll() is None:
                    with contextlib.suppress(BaselineError):
                        process_group_id, start_ticks = process_identity(process.pid)
                        if process_group_id == process.pid:
                            receipt = OwnedProcess(
                                self.config.local_host.name,
                                kind,
                                process.pid,
                                process_group_id,
                                start_ticks,
                                process.pid,
                                self.config.namespace,
                                owner_token,
                                self.results.log_path(log_name),
                            )
                if receipt is not None:
                    cleanup = _terminate_owned_popen(
                        receipt, process, self.config.timeouts.cleanup_seconds
                    )
                    if not cleanup.ownership_verified or not cleanup.terminated:
                        cleanup_error = (
                            cleanup.error or "startup cleanup was unconfirmed"
                        )
                elif process.poll() is not None:
                    process.wait(timeout=1.0)
                else:
                    cleanup_error = "started child has no verified ownership receipt"
            with contextlib.suppress(OSError):
                log_file.close()
            if cleanup_error is not None:
                raise BaselineError(
                    f"{kind} startup failed and cleanup was unconfirmed: "
                    f"{cleanup_error}; original={start_error}"
                ) from start_error
            raise

    def start_opensm(self, port: PortIdentity, owner_token: str) -> LocalHandle:
        opensm = self.config.local_host.tools.opensm
        assert opensm is not None
        guid = "0x" + port.port_guid.replace(":", "")
        command = (
            *_numactl_prefix(self.config.local_host),
            opensm,
            "--guid",
            guid,
            "--log_file",
            "/dev/stdout",
        )
        handle = self._start_local(
            command,
            f"opensm-port-{port.port}",
            owner_token,
            f"opensm-port-{port.port}.log",
        )
        time.sleep(min(0.2, self.config.timeouts.opensm_start_seconds))
        if handle.process.poll() is not None:
            cleanup = stop_local_owned(handle, self.config.timeouts.cleanup_seconds)
            raise BaselineError(
                f"OpenSM port {port.port} exited during startup: {cleanup.error}"
            )
        return handle

    def start_local_client(
        self, command: Sequence[str], kind: str, owner_token: str
    ) -> LocalHandle:
        control_port = _reserved_control_port_from_command(
            command, self.config.reserved_ports
        )
        probe_reserved_ports_unused((control_port,))
        return self._start_local(command, kind, owner_token, f"{kind}.log")

    def start_remote_server(
        self,
        command: Sequence[str],
        kind: str,
        owner_token: str,
        timeout_seconds: float,
    ) -> RemoteHandle:
        _reserved_control_port_from_command(command, self.config.reserved_ports)
        log_name = f"{kind}.log"
        with self.results.create_log(log_name):
            pass
        request = build_remote_request(
            self.config.remote_host,
            command,
            owner_token,
            self.config.namespace,
            timeout_seconds,
            self.config.reserved_ports,
        )
        environment = {
            "PATH": os.defpath,
            "LC_ALL": "C",
            "LANG": "C",
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "EXO_BENCHMARK_NAMESPACE": self.config.namespace,
        }
        transport = subprocess.Popen(
            build_ssh_argv(self.config, "remote-supervise"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        transport_receipt: OwnedProcess | None = None
        try:
            transport_group_id, transport_start_ticks = process_identity(transport.pid)
            transport_receipt = OwnedProcess(
                host_name=self.config.local_host.name,
                kind=f"{kind}-ssh-transport",
                pid=transport.pid,
                process_group_id=transport_group_id,
                start_time_ticks=transport_start_ticks,
                transport_pid=transport.pid,
                namespace=self.config.namespace,
                owner_token=owner_token,
                log_path=self.results.log_path(log_name),
            )
            transport_ownership, transport_members = _group_ownership(transport_receipt)
            if (
                transport_group_id != transport.pid
                or not transport_ownership
                or transport.pid not in transport_members
            ):
                raise BaselineError("SSH transport ownership could not be verified")
            assert transport.stdin is not None and transport.stdout is not None
            transport.stdin.write(
                json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
            )
            transport.stdin.flush()
            line = _read_line_bounded(
                transport.stdout,
                self.config.timeouts.server_start_seconds,
                "remote owner receipt",
            )
            if not line.startswith("IB_BASELINE_OWNER "):
                raise BaselineError(
                    f"remote server did not return owner receipt: {line[:300]}"
                )
            payload = _parse_json_object(
                line.removeprefix("IB_BASELINE_OWNER "), "remote owner receipt"
            )
            receipt = OwnedProcess(
                host_name=self.config.remote_host.name,
                kind=kind,
                pid=_positive_int(payload.get("pid"), "remote pid"),
                process_group_id=_positive_int(
                    payload.get("process_group_id"), "remote process group"
                ),
                start_time_ticks=_positive_int(
                    payload.get("start_time_ticks"), "remote start time"
                ),
                transport_pid=transport.pid,
                namespace=self.config.namespace,
                owner_token=owner_token,
                log_path=self.results.log_path(log_name),
            )
            if (
                receipt.pid != receipt.process_group_id
                or payload.get("owner_token") != owner_token
                or payload.get("namespace") != self.config.namespace
            ):
                raise BaselineError("remote owner receipt does not match the request")
            return RemoteHandle(
                receipt, transport_receipt, transport, log_name, request
            )
        except BaseException as start_error:
            if transport_receipt is None:
                with contextlib.suppress(OSError):
                    transport.terminate()
                try:
                    transport.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        transport.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        transport.wait(timeout=2.0)
                if transport.poll() is None:
                    raise BaselineError(
                        "remote startup failed before SSH transport identity could be "
                        f"recorded and the transport survived; original={start_error}"
                    ) from start_error
                raise
            cleanup = stop_owned_ssh_transport(
                transport,
                transport_receipt,
                min(timeout_seconds, self.config.timeouts.cleanup_seconds),
            )
            if not cleanup.ownership_verified or not cleanup.terminated:
                raise BaselineError(
                    "remote startup failed and SSH transport cleanup was unconfirmed: "
                    f"{cleanup.error}; original={start_error}"
                ) from start_error
            raise

    def wait_local(
        self, handle: LocalHandle, timeout_seconds: float
    ) -> tuple[int, str, CleanupReceipt]:
        try:
            return_code = handle.process.wait(timeout=timeout_seconds)
            cleanup = stop_local_owned(handle, self.config.timeouts.cleanup_seconds)
            return return_code, self.results.read_log(handle.log_name), cleanup
        except subprocess.TimeoutExpired:
            cleanup = stop_local_owned(handle, self.config.timeouts.cleanup_seconds)
            return 124, self.results.read_log(handle.log_name), cleanup

    def _finish_remote(
        self, handle: RemoteHandle, timeout_seconds: float, *, request_stop: bool
    ) -> tuple[int, str, CleanupReceipt]:
        transport = handle.transport
        protocol_output = ""
        try:
            if (
                request_stop
                and transport.poll() is None
                and transport.stdin is not None
            ):
                transport.stdin.write("STOP\n")
                transport.stdin.flush()
            assert transport.stdout is not None
            deadline = time.monotonic() + timeout_seconds
            final: JsonObject | None = None
            while time.monotonic() < deadline:
                line = _read_line_bounded(
                    transport.stdout,
                    max(0.01, deadline - time.monotonic()),
                    "remote final receipt",
                )
                protocol_output += line
                if line.startswith("IB_BASELINE_FINAL "):
                    final = _parse_json_object(
                        line.removeprefix("IB_BASELINE_FINAL "), "remote final receipt"
                    )
                    break
            if final is None:
                raise BaselineError("remote supervisor omitted final receipt")
            transport.wait(timeout=min(5.0, timeout_seconds))
            transport_cleanup = stop_owned_ssh_transport(
                transport,
                handle.transport_receipt,
                min(timeout_seconds, self.config.timeouts.cleanup_seconds),
            )
            child_output = _string(final.get("output"), "remote server output")
            self._replace_log(handle.log_name, child_output + "\n" + protocol_output)
            remote_error = (
                None if final.get("error") is None else str(final.get("error"))
            )
            errors = tuple(
                error
                for error in (remote_error, transport_cleanup.error)
                if error is not None
            )
            cleanup = CleanupReceipt(
                handle.receipt.host_name,
                handle.receipt.kind,
                final.get("ownership_verified") is True
                and transport_cleanup.ownership_verified,
                final.get("terminated") is True and transport_cleanup.terminated,
                final.get("forced") is True or transport_cleanup.forced,
                None if not errors else "; ".join(errors),
            )
            return (
                _integer(final.get("returncode"), "remote return code"),
                child_output,
                cleanup,
            )
        except (BaselineError, OSError, subprocess.TimeoutExpired) as error:
            cleanup = self._fallback_remote_cleanup(handle, timeout_seconds, error)
            return 124, protocol_output, cleanup

    def _replace_log(self, name: str, text: str) -> None:
        self.results.overwrite_log(name, text)

    def _fallback_remote_cleanup(
        self, handle: RemoteHandle, timeout_seconds: float, cause: BaseException
    ) -> CleanupReceipt:
        transport = handle.transport
        transport_cleanup = stop_owned_ssh_transport(
            transport,
            handle.transport_receipt,
            min(timeout_seconds, self.config.timeouts.cleanup_seconds),
        )
        request = {
            "receipt": asdict(handle.receipt),
            "timeout_seconds": timeout_seconds,
        }
        try:
            completed = subprocess.run(
                build_ssh_argv(self.config, "remote-cleanup"),
                input=json.dumps(request) + "\n",
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
            )
            response = _parse_json_object(completed.stdout, "remote fallback cleanup")
            success = (
                completed.returncode == 0
                and response.get("ownership_verified") is True
                and response.get("terminated") is True
            )
            fully_cleaned = success and transport_cleanup.terminated
            ownership_verified = (
                response.get("ownership_verified") is True
                and transport_cleanup.ownership_verified
            )
            return CleanupReceipt(
                handle.receipt.host_name,
                handle.receipt.kind,
                ownership_verified,
                fully_cleaned,
                response.get("forced") is True or transport_cleanup.forced,
                None
                if fully_cleaned
                else (
                    f"{type(cause).__name__}: {cause}; fallback={response}; "
                    f"transport={asdict(transport_cleanup)}"
                ),
            )
        except (OSError, subprocess.TimeoutExpired, BaselineError) as fallback_error:
            return CleanupReceipt(
                handle.receipt.host_name,
                handle.receipt.kind,
                False,
                False,
                transport_cleanup.forced,
                f"{type(cause).__name__}: {cause}; fallback failed: {fallback_error}; "
                f"transport={asdict(transport_cleanup)}",
            )

    def wait_remote(
        self, handle: RemoteHandle, timeout_seconds: float
    ) -> tuple[int, str, CleanupReceipt]:
        return self._finish_remote(handle, timeout_seconds, request_stop=False)

    def stop_local(self, handle: LocalHandle, timeout_seconds: float) -> CleanupReceipt:
        return stop_local_owned(handle, timeout_seconds)

    def stop_remote(
        self, handle: RemoteHandle, timeout_seconds: float
    ) -> CleanupReceipt:
        return self._finish_remote(handle, timeout_seconds, request_stop=True)[2]


def _positive_int(value: object, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise BaselineError(f"{description} must be a positive integer")
    return value


def _integer(value: object, description: str) -> int:
    if type(value) is not int:
        raise BaselineError(f"{description} must be an integer")
    return value


def _string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise BaselineError(f"{description} must be a string")
    return value


def _cpu_list(cpus: Sequence[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def _numactl_prefix(host: HostConfig) -> tuple[str, str, str]:
    return (
        host.tools.numactl,
        f"--physcpubind={_cpu_list(host.cpu_set)}",
        f"--membind={_cpu_list(host.numa_nodes)}",
    )


class PerftestRow(StrictModel):
    raw_line: str
    message_bytes: int = Field(gt=0)
    iterations: int = Field(gt=0)
    peak_gigabits_per_second: float = Field(ge=0)
    average_gigabits_per_second: float = Field(gt=0)
    message_rate_mpps: float = Field(ge=0)
    port: int | None = None


def parse_ib_write_bw_output(output: str) -> tuple[PerftestRow, ...]:
    rows: list[PerftestRow] = []
    number = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
    pattern = re.compile(rf"^\s*((?:{number}\s+){{4,8}}{number})(?:\s+.*)?$")
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        values = tuple(float(value) for value in match.group(1).split())
        if len(values) not in (5, 9):
            continue
        port_match = re.search(r"\bport\s*[:=]?\s*([12])\b", line, re.IGNORECASE)
        try:
            rows.append(
                PerftestRow(
                    raw_line=line,
                    message_bytes=int(values[0]),
                    iterations=int(values[1]),
                    peak_gigabits_per_second=values[2],
                    average_gigabits_per_second=values[3],
                    message_rate_mpps=values[4],
                    port=None if port_match is None else int(port_match.group(1)),
                )
            )
            if len(values) == 9:
                rows.extend(
                    (
                        PerftestRow(
                            raw_line=line,
                            message_bytes=int(values[0]),
                            iterations=int(values[1]),
                            peak_gigabits_per_second=0.0,
                            average_gigabits_per_second=values[5],
                            message_rate_mpps=values[6],
                            port=1,
                        ),
                        PerftestRow(
                            raw_line=line,
                            message_bytes=int(values[0]),
                            iterations=int(values[1]),
                            peak_gigabits_per_second=0.0,
                            average_gigabits_per_second=values[7],
                            message_rate_mpps=values[8],
                            port=2,
                        ),
                    )
                )
        except ValidationError as error:
            raise BaselineError(
                f"invalid ib_write_bw result row {line!r}: {error}"
            ) from error
    if not rows:
        raise BaselineError("ib_write_bw output contains no parseable bandwidth row")
    return tuple(rows)


def perftest_command(
    config: BaselineConfig,
    host: HostConfig,
    *,
    port: int,
    control_port: int,
    dual_port: bool,
    server: bool,
) -> tuple[str, ...]:
    benchmark = config.benchmark
    command = [
        *_numactl_prefix(host),
        host.tools.ib_write_bw,
        f"--ib-dev={host.hca.device}",
        f"--ib-port={port}",
        f"--port={control_port}",
        f"--duration={benchmark.duration_seconds}",
        f"--margin={benchmark.margin_seconds}",
        f"--size={benchmark.message_bytes}",
        f"--tx-depth={benchmark.tx_depth}",
        f"--qp={benchmark.queue_pairs}",
        f"--mtu={benchmark.mtu}",
        "--report_gbits",
        "--perform_warm_up",
        "--force-link=IB",
    ]
    if dual_port:
        command.extend(("--dualport", "--report-per-port"))
    if not server:
        command.append(config.remote_host.management_address)
    return tuple(command)


def _validate_remote_server_command(command: Sequence[str], host: HostConfig) -> None:
    prefix = (*_numactl_prefix(host), host.tools.ib_write_bw)
    if tuple(command[:4]) != prefix or len(command) < 12:
        raise BaselineError(
            "remote command does not invoke pinned numactl/ib_write_bw with exact bindings"
        )
    allowed_flags = (
        "--ib-dev=",
        "--ib-port=",
        "--port=",
        "--duration=",
        "--margin=",
        "--size=",
        "--tx-depth=",
        "--qp=",
        "--mtu=",
    )
    fixed = {
        "--report_gbits",
        "--perform_warm_up",
        "--force-link=IB",
        "--dualport",
        "--report-per-port",
    }
    for argument in command[4:]:
        if argument in fixed or argument == f"--ib-dev={host.hca.device}":
            continue
        if not any(
            argument.startswith(prefix_value)
            and argument.removeprefix(prefix_value).isdigit()
            for prefix_value in allowed_flags
        ):
            raise BaselineError(
                f"remote perftest server argument is not allowed: {argument!r}"
            )
    if not any(argument == f"--ib-dev={host.hca.device}" for argument in command):
        raise BaselineError("remote perftest command uses the wrong HCA")


def _reserved_control_port_from_command(
    command: Sequence[str], reserved_ports: Sequence[int]
) -> int:
    raw_ports = tuple(
        argument.removeprefix("--port=")
        for argument in command
        if argument.startswith("--port=")
    )
    if len(raw_ports) != 1 or re.fullmatch(r"[0-9]+", raw_ports[0]) is None:
        raise BaselineError(
            "perftest command must use exactly one numeric control port"
        )
    control_port = int(raw_ports[0])
    if control_port not in reserved_ports:
        raise BaselineError("perftest command must use a reserved control port")
    return control_port


def _counter_delta(before: HostObservation, after: HostObservation) -> JsonObject:
    result: JsonObject = {}
    for before_port, after_port in zip(before.ports, after.ports, strict=True):
        deltas: JsonObject = {}
        for name, initial in before_port.counters.items():
            final = after_port.counters.get(name)
            if final is None or final < initial:
                raise BaselineError(
                    f"counter {name} regressed on port {before_port.port}"
                )
            deltas[name] = final - initial
        deltas["estimated_xmit_payload_bytes"] = cast(int, deltas["port_xmit_data"]) * 4
        deltas["estimated_rcv_payload_bytes"] = cast(int, deltas["port_rcv_data"]) * 4
        result[str(before_port.port)] = deltas
    return result


@dataclass
class SignalLatch:
    signal_number: int | None = None
    cleanup_started: bool = False

    def handle(self, signal_number: int, _frame: object) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number

    def checkpoint(self) -> None:
        if self.signal_number is not None and not self.cleanup_started:
            raise BaselineError(f"received managed signal {self.signal_number}")


def _runtime_metadata(
    config: BaselineConfig, owner_token: str, processes: Sequence[OwnedProcess]
) -> JsonObject:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "owner_token": owner_token,
        "owned_processes": [cast(JsonValue, asdict(process)) for process in processes],
    }


def _probe_without_owned_conflicts(
    observation: HostObservation, owned: Sequence[OwnedProcess]
) -> HostObservation:
    owned_pids = {process.pid for process in owned}
    remaining = tuple(
        conflict
        for conflict in observation.conflicts
        if not any(f"pid={pid} " in conflict for pid in owned_pids)
    )
    return observation.model_copy(update={"conflicts": remaining})


def _wait_for_active_rails(
    config: BaselineConfig,
    effects: BaselineEffects,
    owned: Sequence[OwnedProcess],
    latch: SignalLatch,
) -> tuple[HostObservation, HostObservation]:
    deadline = time.monotonic() + config.timeouts.rail_active_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        latch.checkpoint()
        try:
            local = _probe_without_owned_conflicts(
                effects.probe_local(
                    config.local_host,
                    reserved_ports=(),
                    ignored_pids=tuple(process.pid for process in owned),
                ),
                owned,
            )
            remote = effects.probe_remote(config.remote_host, reserved_ports=())
            validate_active_rails(local, remote, config)
            return local, remote
        except ProcessConflictError:
            raise
        except BaselineError as error:
            last_error = error
        time.sleep(config.timeouts.poll_seconds)
    raise BaselineError(f"both QDR rails did not become ACTIVE: {last_error}")


def _case_result(
    config: BaselineConfig,
    effects: BaselineEffects,
    results: ResultDirectory,
    owner_token: str,
    owned: list[OwnedProcess],
    cleanups: list[CleanupReceipt],
    opensm_receipts: Sequence[OwnedProcess],
    latch: SignalLatch,
    *,
    name: str,
    port: int,
    control_port: int,
    dual_port: bool,
) -> JsonObject:
    before_local, before_remote = _wait_for_active_rails(
        config, effects, opensm_receipts, latch
    )
    server_command = perftest_command(
        config,
        config.remote_host,
        port=port,
        control_port=control_port,
        dual_port=dual_port,
        server=True,
    )
    client_command = perftest_command(
        config,
        config.local_host,
        port=port,
        control_port=control_port,
        dual_port=dual_port,
        server=False,
    )
    server: RemoteHandle | None = None
    client: LocalHandle | None = None
    server_cleaned = False
    client_cleaned = False
    try:
        server = effects.start_remote_server(
            server_command,
            f"{name}-server",
            owner_token,
            config.timeouts.benchmark_seconds,
        )
        owned.append(server.receipt)
        results.write_json(
            RUNTIME_METADATA_FILENAME,
            _runtime_metadata(config, owner_token, owned),
            replace=True,
        )
        client = effects.start_local_client(
            client_command, f"{name}-client", owner_token
        )
        owned.append(client.receipt)
        results.write_json(
            RUNTIME_METADATA_FILENAME,
            _runtime_metadata(config, owner_token, owned),
            replace=True,
        )
        latch.checkpoint()
        client_code, client_output, client_cleanup = effects.wait_local(
            client, config.timeouts.benchmark_seconds
        )
        cleanups.append(client_cleanup)
        client_cleaned = True
        server_code, server_output, server_cleanup = effects.wait_remote(
            server, config.timeouts.cleanup_seconds
        )
        cleanups.append(server_cleanup)
        server_cleaned = True
        if client_code != 0 or server_code != 0:
            raise BaselineError(
                f"{name} perftest failed: client={client_code}, server={server_code}"
            )
        if (
            not client_cleanup.terminated
            or not server_cleanup.terminated
            or not client_cleanup.ownership_verified
            or not server_cleanup.ownership_verified
        ):
            raise BaselineError(f"{name} cleanup was not confirmed")
        rows = parse_ib_write_bw_output(client_output)
        if dual_port and {row.port for row in rows} != {None, 1, 2}:
            raise BaselineError("native dual-port output did not report both ports")
        after_local, after_remote = _wait_for_active_rails(
            config, effects, opensm_receipts, latch
        )
        return {
            "name": name,
            "mode": "native_dual_port" if dual_port else "single_port",
            "port": port,
            "control_port": control_port,
            "server_command": list(server_command),
            "client_command": list(client_command),
            "rows": [cast(JsonValue, row.model_dump(mode="json")) for row in rows],
            "maximum_average_gigabits_per_second": max(
                row.average_gigabits_per_second for row in rows
            ),
            "server_output_sha256": hashlib.sha256(server_output.encode()).hexdigest(),
            "counter_deltas": {
                config.local_host.name: _counter_delta(before_local, after_local),
                config.remote_host.name: _counter_delta(before_remote, after_remote),
            },
        }
    finally:
        if client is not None and not client_cleaned:
            cleanups.append(effects.stop_local(client, config.timeouts.cleanup_seconds))
        if server is not None and not server_cleaned:
            cleanups.append(
                effects.stop_remote(server, config.timeouts.cleanup_seconds)
            )


def run_harness(
    config: BaselineConfig,
    effects: BaselineEffects,
    results: ResultDirectory,
    latch: SignalLatch | None = None,
) -> JsonObject:
    signal_latch = latch or SignalLatch()
    owner_token = f"{config.run_id}:{uuid.uuid4().hex}"
    owned: list[OwnedProcess] = []
    open_sm_handles: list[LocalHandle] = []
    cleanups: list[CleanupReceipt] = []
    cases: list[JsonObject] = []
    preflight: JsonObject = {}
    final_observations: JsonObject = {}
    caught: BaseException | None = None
    unsafe_conflict = False
    completed = False
    started = time.time()
    results.write_json(
        RUNTIME_METADATA_FILENAME,
        _runtime_metadata(config, owner_token, owned),
        replace=True,
    )
    try:
        local = effects.probe_local(
            config.local_host, reserved_ports=config.reserved_ports
        )
        remote = effects.probe_remote(
            config.remote_host, reserved_ports=config.reserved_ports
        )
        validate_host_observation(
            local,
            config.local_host,
            reserved_ports=config.reserved_ports,
            require_active=False,
        )
        validate_host_observation(
            remote,
            config.remote_host,
            reserved_ports=config.reserved_ports,
            require_active=False,
        )
        preflight = {
            config.local_host.name: cast(JsonValue, local.model_dump(mode="json")),
            config.remote_host.name: cast(JsonValue, remote.model_dump(mode="json")),
        }
        for port in config.local_host.hca.ports:
            handle = effects.start_opensm(port, owner_token)
            open_sm_handles.append(handle)
            owned.append(handle.receipt)
            results.write_json(
                RUNTIME_METADATA_FILENAME,
                _runtime_metadata(config, owner_token, owned),
                replace=True,
            )
        if len(open_sm_handles) != 2:
            raise BaselineError("exactly two owned OpenSM processes were not started")
        active_local, active_remote = _wait_for_active_rails(
            config,
            effects,
            [handle.receipt for handle in open_sm_handles],
            signal_latch,
        )
        final_observations["active"] = {
            config.local_host.name: cast(
                JsonValue, active_local.model_dump(mode="json")
            ),
            config.remote_host.name: cast(
                JsonValue, active_remote.model_dump(mode="json")
            ),
        }
        specifications = (
            ("single-port-1", 1, config.benchmark.single_port_1_control_port, False),
            ("single-port-2", 2, config.benchmark.single_port_2_control_port, False),
            ("native-dual-port", 1, config.benchmark.dual_port_control_port, True),
        )
        for name, port, control_port, dual_port in specifications:
            cases.append(
                _case_result(
                    config,
                    effects,
                    results,
                    owner_token,
                    owned,
                    cleanups,
                    tuple(handle.receipt for handle in open_sm_handles),
                    signal_latch,
                    name=name,
                    port=port,
                    control_port=control_port,
                    dual_port=dual_port,
                )
            )
        completed = True
    except ProcessConflictError as error:
        unsafe_conflict = True
        caught = error
    except BaseException as error:
        caught = error
    finally:
        signal_latch.cleanup_started = True
        for handle in reversed(open_sm_handles):
            cleanups.append(effects.stop_local(handle, config.timeouts.cleanup_seconds))
        try:
            local_final = effects.probe_local(config.local_host, reserved_ports=())
            remote_final = effects.probe_remote(config.remote_host, reserved_ports=())
            validate_host_observation(
                local_final,
                config.local_host,
                reserved_ports=(),
                require_active=False,
            )
            validate_host_observation(
                remote_final,
                config.remote_host,
                reserved_ports=(),
                require_active=False,
            )
            final_observations["after_cleanup"] = {
                config.local_host.name: cast(
                    JsonValue, local_final.model_dump(mode="json")
                ),
                config.remote_host.name: cast(
                    JsonValue, remote_final.model_dump(mode="json")
                ),
            }
        except ProcessConflictError as error:
            unsafe_conflict = True
            if caught is None:
                caught = error
        except BaseException as error:
            if caught is None:
                caught = error
    cleanup_succeeded = (
        not unsafe_conflict
        and len(cleanups) == len(owned)
        and all(item.ownership_verified and item.terminated for item in cleanups)
    )
    status = (
        "completed"
        if completed and caught is None and cleanup_succeeded
        else ("cleanup_failed" if not cleanup_succeeded else "benchmark_failed")
    )
    result: JsonObject = {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "status": status,
        "reportable": status == "completed",
        "cleanup_succeeded": cleanup_succeeded,
        "completed_normally": completed and caught is None,
        "started_at_unix_seconds": started,
        "finished_at_unix_seconds": time.time(),
        "interrupted_signal": signal_latch.signal_number,
        "error": None if caught is None else f"{type(caught).__name__}: {caught}",
        "topology": {
            host.name: cast(JsonValue, host.hca.model_dump(mode="json"))
            for host in config.hosts
        },
        "preflight": preflight,
        "observations": final_observations,
        "cases": [cast(JsonValue, case) for case in cases],
        "reserved_ports": list(config.reserved_ports),
        "owned_processes": [cast(JsonValue, asdict(process)) for process in owned],
        "process_cleanup": [cast(JsonValue, asdict(cleanup)) for cleanup in cleanups],
    }
    results.write_json(
        BENCHMARK_RESULT_FILENAME, cast(Mapping[str, object], result), replace=False
    )
    return result


def _terminate_owned_popen(
    receipt: OwnedProcess, process: subprocess.Popen[bytes], timeout_seconds: float
) -> CleanupReceipt:
    forced = False
    try:
        process.poll()
        ownership, members = _group_ownership(receipt)
        if not ownership:
            return CleanupReceipt(
                receipt.host_name,
                receipt.kind,
                False,
                False,
                False,
                "process ownership changed",
            )
        if members:
            os.killpg(receipt.process_group_id, signal.SIGTERM)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                process.poll()
                ownership, members = _group_ownership(receipt)
                if not ownership:
                    return CleanupReceipt(
                        receipt.host_name,
                        receipt.kind,
                        False,
                        False,
                        False,
                        "ownership changed after SIGTERM",
                    )
                if not members:
                    break
                time.sleep(0.05)
            if members:
                forced = True
                os.killpg(receipt.process_group_id, signal.SIGKILL)
                deadline = time.monotonic() + min(timeout_seconds, 5.0)
                while time.monotonic() < deadline:
                    process.poll()
                    ownership, members = _group_ownership(receipt)
                    if not ownership:
                        return CleanupReceipt(
                            receipt.host_name,
                            receipt.kind,
                            False,
                            False,
                            forced,
                            "ownership changed after SIGKILL",
                        )
                    if not members:
                        break
                    time.sleep(0.05)
        process.poll()
        if process.poll() is not None:
            process.wait(timeout=1.0)
        terminated = process.poll() is not None and not _group_ownership(receipt)[1]
        return CleanupReceipt(
            receipt.host_name,
            receipt.kind,
            True,
            terminated,
            forced,
            None if terminated else "owned process survived cleanup",
        )
    except (OSError, subprocess.TimeoutExpired, BaselineError) as error:
        return CleanupReceipt(
            receipt.host_name,
            receipt.kind,
            False,
            False,
            forced,
            f"{type(error).__name__}: {error}",
        )


def parse_remote_request(
    request: JsonObject,
) -> tuple[HostConfig, tuple[str, ...], str, str, float, tuple[int, ...]]:
    if request.get("schema_version") != 1:
        raise BaselineError("remote request schema version must be exactly 1")
    try:
        host_value = request.get("host")
        host = HostConfig.model_validate_json(
            json.dumps(host_value, sort_keys=True, separators=(",", ":"))
        )
    except ValidationError as error:
        raise BaselineError(f"invalid remote host config: {error}") from error
    if host.transport != "ssh":
        raise BaselineError("remote helper requires an SSH host config")
    raw_command = request.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or not all(isinstance(value, str) for value in raw_command)
    ):
        raise BaselineError("remote command must be a nonempty string array")
    command = tuple(cast(list[str], raw_command))
    _validate_remote_server_command(command, host)
    raw_ports = request.get("reserved_ports")
    if (
        not isinstance(raw_ports, list)
        or not raw_ports
        or not all(type(value) is int for value in raw_ports)
    ):
        raise BaselineError("remote reserved ports must be a nonempty integer array")
    reserved_ports = tuple(cast(list[int], raw_ports))
    if (
        tuple(sorted(set(reserved_ports))) != reserved_ports
        or reserved_ports[0] < 1
        or reserved_ports[-1] > 65535
    ):
        raise BaselineError("remote reserved ports must be sorted, unique, and valid")
    _reserved_control_port_from_command(command, reserved_ports)
    owner_token = _string(request.get("owner_token"), "owner token")
    namespace = _string(request.get("namespace"), "namespace")
    timeout_value = request.get("timeout_seconds")
    if (
        not isinstance(timeout_value, int | float)
        or isinstance(timeout_value, bool)
        or not math.isfinite(timeout_value)
        or timeout_value <= 0
    ):
        raise BaselineError("remote timeout must be finite and positive")
    if not owner_token or _SAFE_IDENTIFIER.fullmatch(namespace) is None:
        raise BaselineError("remote ownership values are invalid")
    return (
        host,
        command,
        owner_token,
        namespace,
        float(timeout_value),
        reserved_ports,
    )


def remote_supervise_main() -> int:
    child: subprocess.Popen[bytes] | None = None
    temporary: IO[bytes] | None = None
    receipt: OwnedProcess | None = None
    cleanup = CleanupReceipt(
        "unknown", "remote-server", False, False, False, "server did not start"
    )
    output = ""
    return_code = 70
    try:
        first_line = sys.stdin.readline()
        if not first_line:
            raise BaselineError("remote supervisor received no request")
        request = _parse_json_object(first_line, "remote supervisor request")
        (
            host,
            command,
            owner_token,
            namespace,
            timeout_seconds,
            reserved_ports,
        ) = parse_remote_request(request)
        conflicts = process_conflicts(ignored_pids=(os.getpid(), os.getppid()))
        if conflicts:
            raise ProcessConflictError(
                f"remote host has unowned benchmark/SM processes: {'; '.join(conflicts)}"
            )
        control_port = _reserved_control_port_from_command(command, reserved_ports)
        probe_reserved_ports_unused((control_port,))
        temporary = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        environment = {
            "PATH": os.defpath,
            "LC_ALL": "C",
            "LANG": "C",
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "EXO_BENCHMARK_NAMESPACE": namespace,
        }
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=temporary,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        process_group_id, start_ticks = process_identity(child.pid)
        receipt = OwnedProcess(
            host.name,
            "remote-server",
            child.pid,
            process_group_id,
            start_ticks,
            os.getpid(),
            namespace,
            owner_token,
            "/dev/null",
        )
        ownership, members = _group_ownership(receipt)
        if process_group_id != child.pid or not ownership or child.pid not in members:
            raise BaselineError("remote child ownership could not be verified")
        print(
            "IB_BASELINE_OWNER "
            + json.dumps(
                {
                    "pid": child.pid,
                    "process_group_id": process_group_id,
                    "start_time_ticks": start_ticks,
                    "owner_token": owner_token,
                    "namespace": namespace,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        deadline = time.monotonic() + timeout_seconds
        stop_requested = False
        while child.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [sys.stdin], [], [], min(0.1, max(0.0, deadline - time.monotonic()))
            )
            if ready:
                control = sys.stdin.readline()
                if not control or control.strip() == "STOP":
                    stop_requested = True
                    break
                raise BaselineError(
                    "remote supervisor received an invalid control message"
                )
        if child.poll() is None:
            cleanup = _terminate_owned_popen(receipt, child, min(10.0, timeout_seconds))
        else:
            child.wait(timeout=1.0)
            ownership, members = _group_ownership(receipt)
            cleanup = CleanupReceipt(
                host.name,
                "remote-server",
                ownership,
                ownership and not members,
                False,
                None
                if ownership and not members
                else "remote child group survived exit",
            )
        return_code = child.returncode if child.returncode is not None else 124
        if stop_requested and return_code < 0:
            return_code = 0
    except BaseException as error:
        if child is not None and receipt is not None and child.poll() is None:
            cleanup = _terminate_owned_popen(receipt, child, 10.0)
        if cleanup.error is None:
            cleanup = CleanupReceipt(
                cleanup.host_name,
                cleanup.kind,
                cleanup.ownership_verified,
                cleanup.terminated,
                cleanup.forced,
                f"{type(error).__name__}: {error}",
            )
        else:
            cleanup = CleanupReceipt(
                cleanup.host_name,
                cleanup.kind,
                cleanup.ownership_verified,
                cleanup.terminated,
                cleanup.forced,
                f"{type(error).__name__}: {error}; {cleanup.error}",
            )
    finally:
        if temporary is not None:
            try:
                temporary.flush()
                temporary.seek(0)
                raw = temporary.read(MAX_CAPTURE_BYTES + 1)
                if len(raw) > MAX_CAPTURE_BYTES:
                    output = (
                        raw[:MAX_CAPTURE_BYTES].decode(errors="replace")
                        + "\n[output truncated]"
                    )
                else:
                    output = raw.decode(errors="replace")
            finally:
                temporary.close()
    print(
        "IB_BASELINE_FINAL "
        + json.dumps(
            {
                "ownership_verified": cleanup.ownership_verified,
                "terminated": cleanup.terminated,
                "forced": cleanup.forced,
                "error": cleanup.error,
                "returncode": return_code,
                "output": output,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if cleanup.ownership_verified and cleanup.terminated else 70


def remote_cleanup_main() -> int:
    try:
        request = _parse_json_object(sys.stdin.read(), "remote cleanup request")
        receipt_value = request.get("receipt")
        if not isinstance(receipt_value, dict):
            raise BaselineError("remote cleanup receipt must be an object")
        receipt_mapping = cast(Mapping[str, object], receipt_value)
        receipt = OwnedProcess(
            host_name=_string(receipt_mapping.get("host_name"), "receipt host"),
            kind=_string(receipt_mapping.get("kind"), "receipt kind"),
            pid=_positive_int(receipt_mapping.get("pid"), "receipt pid"),
            process_group_id=_positive_int(
                receipt_mapping.get("process_group_id"), "receipt process group"
            ),
            start_time_ticks=_positive_int(
                receipt_mapping.get("start_time_ticks"), "receipt start time"
            ),
            transport_pid=_positive_int(
                receipt_mapping.get("transport_pid"), "receipt transport pid"
            ),
            namespace=_string(receipt_mapping.get("namespace"), "receipt namespace"),
            owner_token=_string(
                receipt_mapping.get("owner_token"), "receipt owner token"
            ),
            log_path=_string(receipt_mapping.get("log_path"), "receipt log path"),
        )
        if (
            receipt.host_name != socket.gethostname()
            or receipt.pid != receipt.process_group_id
        ):
            raise BaselineError(
                "remote cleanup receipt does not match this host/process-group shape"
            )
        timeout_value = request.get("timeout_seconds")
        if (
            not isinstance(timeout_value, int | float)
            or isinstance(timeout_value, bool)
            or timeout_value <= 0
        ):
            raise BaselineError("remote cleanup timeout is invalid")
        ownership, members = _group_ownership(receipt)
        if not ownership:
            cleanup = CleanupReceipt(
                receipt.host_name,
                receipt.kind,
                False,
                False,
                False,
                "remote ownership changed",
            )
        elif not members:
            cleanup = CleanupReceipt(receipt.host_name, receipt.kind, True, True, False)
        else:
            # A synthetic Popen cannot safely reap the original child. Kill only the
            # verified owned group and confirm disappearance.
            os.killpg(receipt.process_group_id, signal.SIGTERM)
            deadline = time.monotonic() + float(timeout_value)
            forced = False
            while time.monotonic() < deadline:
                ownership, members = _group_ownership(receipt)
                if not ownership or not members:
                    break
                time.sleep(0.05)
            if ownership and members:
                forced = True
                os.killpg(receipt.process_group_id, signal.SIGKILL)
                time.sleep(min(0.5, float(timeout_value)))
                ownership, members = _group_ownership(receipt)
            cleanup = CleanupReceipt(
                receipt.host_name,
                receipt.kind,
                ownership,
                ownership and not members,
                forced,
                None
                if ownership and not members
                else "remote group cleanup unconfirmed",
            )
        print(json.dumps(asdict(cleanup), sort_keys=True, separators=(",", ":")))
        return 0 if cleanup.ownership_verified and cleanup.terminated else 70
    except (BaselineError, OSError, TypeError) as error:
        print(
            json.dumps(
                {
                    "ownership_verified": False,
                    "terminated": False,
                    "forced": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 70


def host_probe_main() -> int:
    try:
        request = HostProbeRequest.model_validate_json(sys.stdin.read())
        observation = collect_host_observation(
            request.host, reserved_ports=request.reserved_ports
        )
    except (ValidationError, BaselineError, OSError, ValueError) as error:
        print(f"host probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(observation.model_dump_json())
    return 0


def _read_json_regular(path: Path, description: str) -> JsonObject:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_CAPTURE_BYTES:
            raise BaselineError(f"{description} must be a bounded regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as input_file:
            return _parse_json_object(
                input_file.read(MAX_CAPTURE_BYTES + 1), description
            )
    except OSError as error:
        raise BaselineError(f"cannot read {description} {path}: {error}") from error


def _command_option(command: Sequence[str], name: str) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(command):
        if argument == name and index + 1 < len(command):
            values.append(command[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) > 1:
        raise BaselineError(f"active lease command repeats {name}")
    return values[0] if values else None


def validate_active_lease(
    config: BaselineConfig,
    *,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int | None = None,
) -> JsonObject:
    for path in (config_path, lease_path, lock_path):
        if not path.is_absolute():
            raise BaselineError("config and lease paths must be absolute")
    try:
        lock_descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(lock_descriptor, "rb", closefd=True) as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                raise BaselineError("benchmark coordination lock is not held")
    except OSError as error:
        raise BaselineError(f"cannot inspect benchmark lock: {error}") from error
    expected_pid = os.getpid() if process_id is None else process_id
    deadline = time.monotonic() + 5.0
    record: JsonObject
    while True:
        record = _read_json_regular(lease_path, "active lease")
        child_pid = record.get("child_pid")
        if child_pid == expected_pid:
            break
        if child_pid is not None or time.monotonic() >= deadline:
            raise BaselineError("active lease does not belong to this child")
        time.sleep(0.05)
    raw_command = record.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or not all(isinstance(value, str) for value in raw_command)
    ):
        raise BaselineError("active lease command is invalid")
    command = tuple(cast(list[str], raw_command))
    if not any(
        Path(argument).resolve() == Path(__file__).resolve()
        for argument in command
        if argument.startswith("/")
    ):
        raise BaselineError("active lease command does not name this harness")
    if _command_option(command, "--config") != str(config_path):
        raise BaselineError("active lease command uses a different config")
    expected_values: dict[str, object] = {
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "wrapper_pid": os.getppid(),
        "child_cleanup_confirmation_required": True,
    }
    for name, expected in expected_values.items():
        if record.get(name) != expected:
            raise BaselineError(f"active lease {name} differs from the strict config")
    grace = record.get("cleanup_grace_seconds")
    if (
        not isinstance(grace, int | float)
        or isinstance(grace, bool)
        or grace < minimum_cleanup_grace_seconds(config)
    ):
        raise BaselineError(
            "active lease cleanup grace is shorter than the cleanup bound"
        )
    metadata = _json_object(record.get("metadata"), "active lease metadata")
    expected_metadata = build_static_metadata(
        config, command, _config_sha256(config_path)
    )
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise BaselineError(
                f"active lease metadata.{name} differs from the strict config"
            )
    try:
        heartbeat = datetime.fromisoformat(
            _string(record.get("heartbeat"), "lease heartbeat").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except ValueError as error:
        raise BaselineError("active lease heartbeat is invalid") from error
    now = datetime.now(timezone.utc)
    if heartbeat < now - timedelta(minutes=2) or heartbeat > now + timedelta(minutes=1):
        raise BaselineError("active lease heartbeat is stale or in the future")
    return record


class RunArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path | None


class PreparationArguments(argparse.Namespace):
    config: Path
    metadata_output: Path
    wrapper_python: Path
    child_python: Path
    benchmark_lease_script: Path
    harness_script: Path
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
        description="Prepare strict benchmark-lease metadata for the IB baseline"
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
    parser.add_argument("--cleanup-grace-seconds", required=True, type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", required=True, type=Path)
    return parser.parse_args(arguments, namespace=PreparationArguments())


def preparation_main(arguments: Sequence[str]) -> int:
    args = parse_preparation_arguments(arguments)
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
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        lease_path=args.lease_path,
        lock_path=args.lock_path,
        result_root=args.result_root,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "metadata_output": str(args.metadata_output),
                "child_argv": list(prepared.child_argv),
                "benchmark_lease_argv": list(prepared.benchmark_lease_argv),
                "minimum_cleanup_grace_seconds": prepared.minimum_cleanup_grace_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def run_main(arguments: Sequence[str]) -> int:
    args = parse_run_arguments(arguments)
    if (
        not args.config.is_absolute()
        or not args.lease_path.is_absolute()
        or not args.lock_path.is_absolute()
    ):
        raise BaselineError("config, lease, and lock paths must be absolute")
    config = load_config(args.config)
    if args.result_dir is not None and str(args.result_dir) != config.result_directory:
        raise BaselineError("--result-dir differs from the strict config")
    results = ResultDirectory.inherited(Path(config.result_directory))
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    latch = SignalLatch()
    try:
        validate_active_lease(
            config,
            config_path=args.config,
            lease_path=args.lease_path,
            lock_path=args.lock_path,
        )
        for managed in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[managed] = signal.getsignal(managed)
            signal.signal(managed, latch.handle)
        outcome = run_harness(config, SystemEffects(config, results), results, latch)
        return 0 if outcome.get("status") == "completed" else 2
    finally:
        for managed, previous in previous_handlers.items():
            signal.signal(managed, previous)
        results.close()


def main(arguments: Sequence[str] | None = None) -> int:
    normalized = list(sys.argv[1:] if arguments is None else arguments)
    if normalized == ["host-probe"]:
        return host_probe_main()
    if normalized == ["remote-supervise"]:
        return remote_supervise_main()
    if normalized == ["remote-cleanup"]:
        return remote_cleanup_main()
    if normalized == ["current-cx3-profile"]:
        print(json.dumps(CURRENT_CX3_IDENTITIES, sort_keys=True, indent=2))
        return 0
    if normalized and normalized[0] == "prepare-lease":
        return preparation_main(normalized[1:])
    return run_main(normalized)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BaselineError, ValidationError, OSError, ValueError) as error:
        print(f"IB baseline failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
