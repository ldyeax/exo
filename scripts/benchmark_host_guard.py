#!/usr/bin/env python3
"""Fail-closed guard for an idle, nonparticipant benchmark peer.

The coordinator invokes this file on the peer through a pinned SSH transport.
The request travels over stdin and is bound to a fresh nonce and canonical
request digest.  The remote response is phase-neutral; callers wrap it in an
explicit preflight or postflight snapshot only after verifying the wire receipt.

This module intentionally uses only the standard library and Pydantic so the
same source can run under the host's system Python without importing Exo.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import selectors
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
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
ProcessClass = Literal["benchmark", "model_server", "profiler", "storage"]

HOST_GUARD_SCHEMA_VERSION = 1
HOST_GUARD_PROTOCOL = "exo-benchmark-host-guard-v1"
MAXIMUM_WIRE_BYTES = 1024 * 1024
MAXIMUM_COMMAND_STDERR_BYTES = 256 * 1024
MAXIMUM_REQUEST_LIFETIME_NS = 120 * 1_000_000_000
DEFAULT_REQUEST_LIFETIME_NS = 30 * 1_000_000_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,254}$")
_SAFE_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]{0,254}\.service$")
_SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./+:-]+$")
_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{16,64}$")
_PCI_BDF = re.compile(r"^(?:[0-9a-f]{4}|[0-9a-f]{8}):[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GUID = re.compile(r"^(?:[0-9a-f]{4}:){3}[0-9a-f]{4}$")
_OPENSM_GUID = re.compile(r"^0x[0-9a-f]{16}$")
_BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_KERNEL_MODULE = re.compile(r"^[A-Za-z0-9_]+$")

DATA_COUNTER_NAMES = (
    "port_rcv_data",
    "port_rcv_packets",
    "port_xmit_data",
    "port_xmit_packets",
)
HEALTH_COUNTER_NAMES = (
    "VL15_dropped",
    "excessive_buffer_overrun_errors",
    "link_downed",
    "link_error_recovery",
    "local_link_integrity_errors",
    "port_rcv_constraint_errors",
    "port_rcv_errors",
    "port_rcv_remote_physical_errors",
    "port_rcv_switch_relay_errors",
    "port_xmit_constraint_errors",
    "port_xmit_discards",
    "symbol_error",
)
KNOWN_COUNTER_NAMES = frozenset((*DATA_COUNTER_NAMES, *HEALTH_COUNTER_NAMES))

_BENCHMARK_EXECUTABLES = frozenset(
    {
        "all_gather_perf",
        "all_reduce_perf",
        "broadcast_perf",
        "ib_read_bw",
        "ib_read_lat",
        "ib_send_bw",
        "ib_send_lat",
        "ib_write_bw",
        "ib_write_lat",
        "nccl-tests",
        "nccl_test",
        "reduce_scatter_perf",
        "sendrecv_perf",
    }
)
_MODEL_SERVER_EXECUTABLES = frozenset(
    {
        "exo",
        "llama-server",
        "ollama",
        "sglang",
        "text-generation-launcher",
        "tritonserver",
        "uvicorn",
        "vllm",
    }
)
_ALWAYS_CONFLICTING_STORAGE_EXECUTABLES = frozenset(
    {
        "aria2c",
        "fio",
        "rclone",
        "rsync",
        "scp",
        "sftp",
    }
)
_PROFILER_EXECUTABLES = frozenset(
    {
        "amplxe-cl",
        "likwid-perfctr",
        "ncu",
        "nsys",
        "pax",
        "perf",
        "py-spy",
        "sep",
        "sep5",
        "vtune",
    }
)
_MODEL_MODULE_PREFIXES = (
    "exo",
    "ktransformers.server",
    "mlx_lm.server",
    "sglang.launch_server",
    "text_generation_server",
    "vllm.entrypoints",
)
WORKFLOW_SCRIPT_CLASSES: dict[str, ProcessClass] = {
    "benchmark_lease.py": "benchmark",
    "build_sglang_kt_runtime.py": "storage",
    "create_sglang_kt_glm47_validation_process_spec.py": "benchmark",
    "create_sglang_kt_model_contract.py": "storage",
    "download_model_to_cluster.py": "storage",
    "install_sglang_kt_runtime.py": "storage",
    "mlx_nccl_smoke.py": "benchmark",
    "prepare_sglang_kt_source.py": "storage",
    "run_sglang_kt_glm47_moe_tuning.py": "benchmark",
    "run_sglang_kt_glm47_serving_benchmark.py": "benchmark",
    "run_sglang_kt_glm47_validation.py": "benchmark",
    "sglang_kt_glm47_backend.py": "model_server",
    "sglang_kt_glm47_live.py": "model_server",
    "sglang_kt_glm47_serving_client.py": "benchmark",
    "sglang_kt_glm47_trace.py": "benchmark",
    "tp1_oracle_capture.py": "benchmark",
    "tune_sglang_kt_glm47_moe.py": "benchmark",
    "two_host_ib_baseline.py": "benchmark",
    "two_host_mlx_nccl_poc.py": "benchmark",
    "two_host_model_stage.py": "storage",
    "validate_sglang_kt_glm47_model.py": "storage",
    "validate_sglang_kt_runtime.py": "storage",
}
_CONDITIONAL_STORAGE_EXECUTABLES = frozenset(
    {
        "b3sum",
        "cp",
        "curl",
        "dd",
        "gzip",
        "md5sum",
        "mv",
        "pigz",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "tar",
        "wget",
    }
)
_MODEL_STORAGE_MARKERS = (
    "/.cache/huggingface/",
    "/huggingface/",
    "/mnt/sanic/",
    "/var/lib/exo/",
    ".gguf",
    ".safetensors",
    "hf.co/",
    "huggingface.co/",
)
_WORKFLOW_LAUNCHERS = frozenset(
    {
        "bash",
        "env",
        "flock",
        "numactl",
        "python",
        "python3",
        "sh",
        "stdbuf",
        "taskset",
        "timeout",
        "torchrun",
        "uv",
        "uvx",
        "zsh",
    }
)
_INACTIVE_RAID_ACTIONS = frozenset({"frozen", "idle"})


def _extract_opensm_guid(arguments: Sequence[str]) -> str:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--guid" and index + 1 < len(arguments):
            values.append(arguments[index + 1].lower())
        elif argument.startswith("--guid="):
            values.append(argument.partition("=")[2].lower())
    if len(values) != 1 or _OPENSM_GUID.fullmatch(values[0]) is None:
        raise ValueError("OpenSM argv must contain exactly one canonical --guid")
    return values[0]


class HostGuardError(RuntimeError):
    """The peer could not be proven idle and stable."""


class HostGuardTransportError(HostGuardError):
    """The pinned SSH transport failed or returned invalid evidence."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return cast(JsonValue, value.model_dump(mode="json"))
    return cast(JsonValue, value)


def canonical_host_guard_json(value: object) -> bytes:
    """Return the only accepted host-guard JSON representation."""

    try:
        return (
            json.dumps(
                _json_value(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise HostGuardError("host-guard value is not canonical JSON") from error


def calculate_host_guard_sha256(value: object) -> str:
    """Hash a value using the canonical wire representation."""

    return hashlib.sha256(canonical_host_guard_json(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HostGuardError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise HostGuardError(f"JSON contains non-finite number {value}")


def parse_bounded_canonical_json(
    contents: bytes,
    *,
    maximum_bytes: int = MAXIMUM_WIRE_BYTES,
) -> JsonObject:
    """Parse one bounded canonical JSON object, including its final newline."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if not contents or len(contents) > maximum_bytes:
        raise HostGuardError("host-guard JSON is empty or exceeds its size limit")
    try:
        text = contents.decode("ascii")
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HostGuardError("host-guard payload is not strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise HostGuardError("host-guard payload must be a JSON object")
    result = cast(JsonObject, value)
    if contents != canonical_host_guard_json(result):
        raise HostGuardError("host-guard payload is not canonically encoded")
    return result


def _validate_absolute_path(value: str, description: str) -> str:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)) or "\0" in value:
        raise ValueError(f"{description} must be absolute, normalized, and NUL-free")
    return value


class FileIdentityBinding(StrictModel):
    path: str
    resolved_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", "resolved_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_absolute_path(value, "file identity path")


class FileIdentityObservation(StrictModel):
    path: str
    resolved_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", "resolved_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _validate_absolute_path(value, "observed file path")


class SshTransportBinding(StrictModel):
    executable: FileIdentityBinding
    target: str
    user: str
    port: int = Field(ge=1, le=65535)
    known_hosts_file: FileIdentityBinding
    identity_file: FileIdentityBinding
    connect_timeout_seconds: int = Field(ge=1, le=60)
    server_alive_interval_seconds: int = Field(ge=1, le=60)
    server_alive_count_max: int = Field(ge=1, le=10)

    @field_validator("target", "user")
    @classmethod
    def validate_names(cls, value: str) -> str:
        if _SAFE_NAME.fullmatch(value) is None or value.startswith("-"):
            raise ValueError("SSH target and user must be safe fixed identifiers")
        return value

    def command(
        self,
        probe: "RemoteProbeIdentityBinding",
        *,
        executable_path: str | None = None,
        known_hosts_path: str | None = None,
        identity_path: str | None = None,
    ) -> tuple[str, ...]:
        remote_python = probe.python.resolved_path
        remote_script = probe.script.resolved_path
        for value in (remote_python, remote_script):
            if _SAFE_REMOTE_PATH.fullmatch(value) is None:
                raise HostGuardError("remote command path is not a safe fixed token")
        ssh_executable = executable_path or self.executable.resolved_path
        ssh_known_hosts = known_hosts_path or self.known_hosts_file.resolved_path
        ssh_identity = identity_path or self.identity_file.resolved_path
        return (
            ssh_executable,
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RequestTTY=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={ssh_known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"IdentityFile={ssh_identity}",
            "-o",
            f"ConnectTimeout={self.connect_timeout_seconds}",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            f"ServerAliveInterval={self.server_alive_interval_seconds}",
            "-o",
            f"ServerAliveCountMax={self.server_alive_count_max}",
            "-p",
            str(self.port),
            "--",
            f"{self.user}@{self.target}",
            remote_python,
            remote_script,
            "remote-probe",
        )


class RemoteProbeIdentityBinding(StrictModel):
    python: FileIdentityBinding
    script: FileIdentityBinding


class HostToolBindings(StrictModel):
    nvidia_smi: FileIdentityBinding
    systemctl: FileIdentityBinding


class GpuBinding(StrictModel):
    uuid: str
    pci_bus_id: str
    name: str = Field(min_length=1, max_length=200)
    memory_total_bytes: int = Field(gt=0)

    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        if _GPU_UUID.fullmatch(value) is None:
            raise ValueError("GPU UUID is not canonical")
        return value

    @field_validator("pci_bus_id")
    @classmethod
    def validate_bdf(cls, value: str) -> str:
        normalized = value.lower()
        if _PCI_BDF.fullmatch(normalized) is None:
            raise ValueError("GPU PCI bus ID is not canonical")
        return normalized


class CpuMemoryBinding(StrictModel):
    minimum_online_cpu_count: int = Field(gt=0)
    numa_cpu_sets: dict[str, tuple[int, ...]]
    minimum_total_memory_bytes: int = Field(gt=0)
    required_cpu_flags: tuple[str, ...] = ("amx_bf16", "amx_int8", "amx_tile")

    @model_validator(mode="after")
    def validate_topology(self) -> "CpuMemoryBinding":
        if not self.numa_cpu_sets:
            raise ValueError("at least one NUMA node must be bound")
        all_cpus: set[int] = set()
        for node, cpus in self.numa_cpu_sets.items():
            if not node.isdigit() or not cpus or tuple(sorted(set(cpus))) != cpus:
                raise ValueError("NUMA CPU sets must be sorted, unique, and nonempty")
            if cpus[0] < 0 or all_cpus.intersection(cpus):
                raise ValueError("NUMA CPU sets must contain disjoint nonnegative CPUs")
            all_cpus.update(cpus)
        if len(all_cpus) < self.minimum_online_cpu_count:
            raise ValueError("bound NUMA topology has too few CPUs")
        if (
            not self.required_cpu_flags
            or tuple(sorted(set(self.required_cpu_flags))) != self.required_cpu_flags
        ):
            raise ValueError("required CPU flags must be sorted and unique")
        return self


class HcaPortBinding(StrictModel):
    port: int = Field(gt=0)
    gid_index: int = Field(ge=0)
    gid: str
    expected_rate: str = Field(min_length=1, max_length=100)
    health_counter_maximums: dict[str, int]
    idle_data_counter_maximum_deltas: dict[str, int]

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.IPv6Address(value)
        if address.is_unspecified:
            raise ValueError("HCA GID must not be unspecified")
        return address.exploded

    @field_validator("health_counter_maximums")
    @classmethod
    def validate_health_counters(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or not set(value) <= set(HEALTH_COUNTER_NAMES):
            raise ValueError("HCA health counter bounds must use known health counters")
        if any(limit < 0 for limit in value.values()):
            raise ValueError("HCA health counter bounds must be nonnegative")
        return value

    @field_validator("idle_data_counter_maximum_deltas")
    @classmethod
    def validate_idle_data_counter_deltas(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != set(DATA_COUNTER_NAMES):
            raise ValueError(
                "idle HCA data counter limits must bind every data counter exactly"
            )
        if any(limit < 0 for limit in value.values()):
            raise ValueError("idle HCA data counter limits must be nonnegative")
        return value


class HcaBinding(StrictModel):
    device: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    node_guid: str
    ports: tuple[HcaPortBinding, HcaPortBinding]

    @field_validator("node_guid")
    @classmethod
    def validate_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _GUID.fullmatch(normalized) is None:
            raise ValueError("HCA node GUID must contain four lowercase groups")
        return normalized

    @model_validator(mode="after")
    def validate_ports(self) -> "HcaBinding":
        if tuple(port.port for port in self.ports) != (1, 2):
            raise ValueError("HCA ports must be exactly ordered ports 1 and 2")
        if self.ports[0].gid == self.ports[1].gid:
            raise ValueError("HCA port GIDs must be unique")
        return self


class OpenSmUnitBinding(StrictModel):
    unit: str
    port: Literal[1, 2]
    guid: str
    executable: FileIdentityBinding
    argv: tuple[str, ...]
    version: str = Field(min_length=1, max_length=500)

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if _SAFE_UNIT.fullmatch(value) is None:
            raise ValueError("OpenSM unit name is not safe or canonical")
        return value

    @field_validator("guid")
    @classmethod
    def validate_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _OPENSM_GUID.fullmatch(normalized) is None:
            raise ValueError("OpenSM GUID must be 0x plus sixteen lowercase hex digits")
        return normalized

    @model_validator(mode="after")
    def validate_command(self) -> "OpenSmUnitBinding":
        if not self.argv or self.argv[0] != self.executable.path:
            raise ValueError("OpenSM argv must start with the bound executable")
        if any("\0" in argument for argument in self.argv):
            raise ValueError("OpenSM argv must be NUL-free")
        if _extract_opensm_guid(self.argv) != self.guid:
            raise ValueError("OpenSM argv does not contain its exact bound GUID")
        return self


class IdlePeerPolicy(StrictModel):
    maximum_load_1m_per_online_cpu: float = Field(ge=0.0)
    minimum_available_memory_bytes: int = Field(ge=0)
    maximum_gpu_memory_used_bytes: int = Field(ge=0)
    maximum_gpu_utilization_percent: int = Field(ge=0, le=100)
    maximum_gpu_memory_utilization_percent: int = Field(ge=0, le=100)
    maximum_gpu_temperature_celsius: int = Field(ge=0, le=150)
    maximum_clock_skew_ns: int = Field(ge=0, le=30_000_000_000)
    unsafe_profiler_kernel_modules: tuple[str, ...] = ("pax", "sep5")

    @field_validator("maximum_load_1m_per_online_cpu")
    @classmethod
    def validate_load(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("maximum normalized load must be finite")
        return value

    @field_validator("unsafe_profiler_kernel_modules")
    @classmethod
    def validate_profiler_modules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or tuple(sorted(set(value))) != value
            or any(_KERNEL_MODULE.fullmatch(module) is None for module in value)
        ):
            raise ValueError(
                "unsafe profiler modules must be sorted, unique safe names"
            )
        return value


class CoordinationPeerBinding(StrictModel):
    """Independent expectations for the idle nonparticipant peer only."""

    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    hostname: str
    ssh: SshTransportBinding
    remote_probe: RemoteProbeIdentityBinding
    tools: HostToolBindings
    cpu_memory: CpuMemoryBinding
    gpus: tuple[GpuBinding, ...]
    hca: HcaBinding
    opensm_units: tuple[OpenSmUnitBinding, OpenSmUnitBinding]
    reserved_ports: tuple[int, ...]
    policy: IdlePeerPolicy

    @field_validator("hostname")
    @classmethod
    def validate_hostname(cls, value: str) -> str:
        if _SAFE_NAME.fullmatch(value) is None or value.startswith("-"):
            raise ValueError("peer hostname is not a safe fixed identifier")
        return value

    @model_validator(mode="after")
    def validate_peer(self) -> "CoordinationPeerBinding":
        if not self.gpus or len({gpu.uuid for gpu in self.gpus}) != len(self.gpus):
            raise ValueError("GPU bindings must be nonempty with unique UUIDs")
        if tuple(sorted(self.gpus, key=lambda gpu: gpu.uuid)) != self.gpus:
            raise ValueError("GPU bindings must be sorted by UUID")
        if tuple(unit.port for unit in self.opensm_units) != (1, 2):
            raise ValueError(
                "exactly one ordered persistent OpenSM unit is required per port"
            )
        if len({unit.unit for unit in self.opensm_units}) != 2:
            raise ValueError("persistent OpenSM unit names must be unique")
        if (
            not self.reserved_ports
            or tuple(sorted(set(self.reserved_ports))) != self.reserved_ports
            or self.reserved_ports[0] < 1
            or self.reserved_ports[-1] > 65535
        ):
            raise ValueError("reserved ports must be sorted, unique, and valid")
        return self


class GpuTelemetryObservation(StrictModel):
    uuid: str
    pci_bus_id: str
    name: str
    memory_total_bytes: int = Field(gt=0)
    memory_used_bytes: int = Field(ge=0)
    gpu_utilization_percent: int = Field(ge=0, le=100)
    memory_utilization_percent: int = Field(ge=0, le=100)
    temperature_celsius: int = Field(ge=0, le=150)
    power_draw_watts: float | None

    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        if _GPU_UUID.fullmatch(value) is None:
            raise ValueError("observed GPU UUID is not canonical")
        return value

    @field_validator("pci_bus_id")
    @classmethod
    def validate_bdf(cls, value: str) -> str:
        normalized = value.lower()
        if _PCI_BDF.fullmatch(normalized) is None:
            raise ValueError("observed GPU PCI bus ID is not canonical")
        return normalized

    @field_validator("power_draw_watts")
    @classmethod
    def validate_power(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("GPU power draw must be finite and nonnegative")
        return value


class GpuProcessObservation(StrictModel):
    gpu_uuid: str
    pid: int = Field(gt=0)
    process_name: str = Field(min_length=1, max_length=1000)
    used_memory_bytes: int = Field(ge=0)


class ConflictingProcessObservation(StrictModel):
    pid: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)
    classes: tuple[ProcessClass, ...]
    argv: tuple[str, ...]

    @model_validator(mode="after")
    def validate_conflict(self) -> "ConflictingProcessObservation":
        if not self.argv or not self.classes:
            raise ValueError(
                "conflicting process evidence must include argv and a class"
            )
        if tuple(sorted(set(self.classes))) != self.classes:
            raise ValueError("conflicting process classes must be sorted and unique")
        return self


class CpuMemoryObservation(StrictModel):
    online_cpus: tuple[int, ...]
    numa_cpu_sets: dict[str, tuple[int, ...]]
    load_average_1m: float = Field(ge=0.0)
    load_average_5m: float = Field(ge=0.0)
    load_average_15m: float = Field(ge=0.0)
    memory_total_bytes: int = Field(gt=0)
    memory_available_bytes: int = Field(ge=0)
    memory_free_bytes: int = Field(ge=0)
    common_cpu_flags: tuple[str, ...]

    @model_validator(mode="after")
    def validate_values(self) -> "CpuMemoryObservation":
        if (
            not self.online_cpus
            or tuple(sorted(set(self.online_cpus))) != self.online_cpus
        ):
            raise ValueError("online CPUs must be sorted, unique, and nonempty")
        if self.online_cpus[0] < 0:
            raise ValueError("online CPU indexes must be nonnegative")
        if any(
            not math.isfinite(value)
            for value in (
                self.load_average_1m,
                self.load_average_5m,
                self.load_average_15m,
            )
        ):
            raise ValueError("load averages must be finite")
        if (
            max(self.memory_available_bytes, self.memory_free_bytes)
            > self.memory_total_bytes
        ):
            raise ValueError("memory observation is inconsistent")
        if tuple(sorted(set(self.common_cpu_flags))) != self.common_cpu_flags:
            raise ValueError("CPU flags must be sorted and unique")
        return self


class HcaPortObservation(StrictModel):
    port: int = Field(gt=0)
    gid: str
    state: str = Field(min_length=1, max_length=100)
    physical_state: str = Field(min_length=1, max_length=100)
    rate: str = Field(min_length=1, max_length=100)
    lid: int = Field(ge=0)
    sm_lid: int = Field(ge=0)
    counter_device: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    counter_port: int = Field(gt=0)
    counters: dict[str, int]

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.IPv6Address(value)
        if address.is_unspecified:
            raise ValueError("observed HCA GID must not be unspecified")
        return address.exploded

    @field_validator("counters")
    @classmethod
    def validate_counters(cls, value: dict[str, int]) -> dict[str, int]:
        if (
            not set(DATA_COUNTER_NAMES) <= set(value)
            or not set(value) <= KNOWN_COUNTER_NAMES
        ):
            raise ValueError(
                "HCA counters must include data counters and only known counters"
            )
        if any(counter < 0 for counter in value.values()):
            raise ValueError("HCA counters must be nonnegative")
        return value


class HcaObservation(StrictModel):
    device: str
    node_guid: str
    ports: tuple[HcaPortObservation, HcaPortObservation]

    @model_validator(mode="after")
    def validate_ports(self) -> "HcaObservation":
        if tuple(port.port for port in self.ports) != (1, 2):
            raise ValueError("observed HCA ports must be ordered ports 1 and 2")
        return self


class OpenSmUnitObservation(StrictModel):
    unit: str
    port: Literal[1, 2]
    guid: str
    load_state: str
    active_state: str
    sub_state: str
    main_pid: int = Field(gt=0)
    systemd_start_monotonic_us: int = Field(gt=0)
    process_start_time_ticks: int = Field(gt=0)
    argv: tuple[str, ...]
    executable: FileIdentityObservation
    version: str

    @field_validator("guid")
    @classmethod
    def validate_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _OPENSM_GUID.fullmatch(normalized) is None:
            raise ValueError("observed OpenSM GUID is not canonical")
        return normalized


class KernelModuleObservation(StrictModel):
    name: str
    size_bytes: int = Field(gt=0)
    reference_count: int = Field(ge=0)
    dependencies: tuple[str, ...]
    state: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if _KERNEL_MODULE.fullmatch(value) is None:
            raise ValueError("kernel module name is not canonical")
        return value

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value or any(
            _KERNEL_MODULE.fullmatch(module) is None for module in value
        ):
            raise ValueError("kernel module dependencies must be sorted unique names")
        return value


class HostObservation(StrictModel):
    hostname: str
    boot_id: str
    probe_python: FileIdentityObservation
    probe_script: FileIdentityObservation
    nvidia_smi: FileIdentityObservation
    systemctl: FileIdentityObservation
    cpu_memory: CpuMemoryObservation
    gpus: tuple[GpuTelemetryObservation, ...]
    gpu_processes: tuple[GpuProcessObservation, ...]
    conflicting_processes: tuple[ConflictingProcessObservation, ...]
    unsafe_profiler_modules: tuple[KernelModuleObservation, ...]
    raid_sync_conflicts: tuple[str, ...]
    unused_reserved_ports: tuple[int, ...]
    hca: HcaObservation
    opensm_units: tuple[OpenSmUnitObservation, OpenSmUnitObservation]

    @field_validator("boot_id")
    @classmethod
    def validate_boot_id(cls, value: str) -> str:
        normalized = value.lower()
        if _BOOT_ID.fullmatch(normalized) is None:
            raise ValueError("boot ID is not canonical")
        return normalized

    @model_validator(mode="after")
    def validate_ordering(self) -> "HostObservation":
        if tuple(sorted(self.gpus, key=lambda gpu: gpu.uuid)) != self.gpus:
            raise ValueError("GPU observations must be sorted by UUID")
        if tuple(
            sorted(self.gpu_processes, key=lambda item: (item.gpu_uuid, item.pid))
        ) != (self.gpu_processes):
            raise ValueError("GPU process observations must be canonically sorted")
        if tuple(sorted(self.conflicting_processes, key=lambda item: item.pid)) != (
            self.conflicting_processes
        ):
            raise ValueError("conflicting processes must be sorted by PID")
        if tuple(module.name for module in self.unsafe_profiler_modules) != tuple(
            sorted({module.name for module in self.unsafe_profiler_modules})
        ):
            raise ValueError(
                "unsafe profiler module observations must be sorted and unique"
            )
        return self


class RemoteProbeRequest(StrictModel):
    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    protocol: Literal["exo-benchmark-host-guard-v1"] = HOST_GUARD_PROTOCOL
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at_unix_ns: int = Field(gt=0)
    expires_at_unix_ns: int = Field(gt=0)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    peer_binding: CoordinationPeerBinding
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_request(self) -> "RemoteProbeRequest":
        lifetime = self.expires_at_unix_ns - self.issued_at_unix_ns
        if lifetime <= 0 or lifetime > MAXIMUM_REQUEST_LIFETIME_NS:
            raise ValueError("remote probe request lifetime is invalid")
        if self.binding_sha256 != calculate_coordination_peer_binding_sha256(
            self.peer_binding
        ):
            raise ValueError("remote probe binding digest is invalid")
        if self.request_sha256 != calculate_remote_probe_request_sha256(self):
            raise ValueError("remote probe request digest is invalid")
        return self


class RemoteProbeReceipt(StrictModel):
    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    protocol: Literal["exo-benchmark-host-guard-v1"] = HOST_GUARD_PROTOCOL
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    peer_binding: CoordinationPeerBinding
    request_issued_at_unix_ns: int = Field(gt=0)
    request_expires_at_unix_ns: int = Field(gt=0)
    observed_at_unix_ns: int = Field(gt=0)
    observation: HostObservation
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_receipt_digest(self) -> "RemoteProbeReceipt":
        if self.binding_sha256 != calculate_coordination_peer_binding_sha256(
            self.peer_binding
        ):
            raise ValueError("remote receipt peer binding digest is invalid")
        if self.receipt_sha256 != calculate_remote_probe_receipt_sha256(self):
            raise ValueError("remote probe receipt digest is invalid")
        return self


class HostGuardSnapshot(StrictModel):
    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    phase: Literal["preflight", "postflight"]
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    collected_at_unix_ns: int = Field(gt=0)
    remote_receipt: RemoteProbeReceipt
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_snapshot_digest(self) -> "HostGuardSnapshot":
        if self.binding_sha256 != self.remote_receipt.binding_sha256:
            raise ValueError("snapshot and remote receipt binding digests differ")
        if self.config_sha256 != self.remote_receipt.config_sha256:
            raise ValueError("snapshot and remote receipt config digests differ")
        if self.snapshot_sha256 != calculate_host_guard_snapshot_sha256(self):
            raise ValueError("host guard snapshot digest is invalid")
        return self

    @property
    def observation(self) -> HostObservation:
        return self.remote_receipt.observation


class HostGuardComparison(StrictModel):
    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    preflight_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    postflight_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_counter_deltas: dict[str, int]
    data_counter_maximum_deltas: dict[str, int]
    health_counter_deltas: dict[str, int]
    stable: bool
    failures: tuple[str, ...]

    @model_validator(mode="after")
    def validate_result(self) -> "HostGuardComparison":
        if self.stable == bool(self.failures):
            raise ValueError("comparison stable flag and failures disagree")
        if any(
            value < 0
            for counters in (
                self.data_counter_deltas,
                self.data_counter_maximum_deltas,
                self.health_counter_deltas,
            )
            for value in counters.values()
        ):
            raise ValueError("counter deltas and limits must be nonnegative")
        if set(self.data_counter_deltas) != set(self.data_counter_maximum_deltas):
            raise ValueError("data counter deltas and limits must cover the same keys")
        return self


class FabricRailBinding(StrictModel):
    local_port: Literal[1, 2]
    remote_port: Literal[1, 2]
    local_gid: str
    remote_gid: str
    rate: str
    subnet_manager_host: Literal["local", "remote"] = "local"
    subnet_manager_unit: str
    subnet_manager_guid: str
    subnet_manager_argv: tuple[str, ...]

    @field_validator("local_gid", "remote_gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.IPv6Address(value)
        if address.is_unspecified:
            raise ValueError("fabric GID must not be unspecified")
        return address.exploded

    @field_validator("subnet_manager_unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if _SAFE_UNIT.fullmatch(value) is None:
            raise ValueError("fabric subnet manager unit is not canonical")
        return value

    @field_validator("subnet_manager_guid")
    @classmethod
    def validate_manager_guid(cls, value: str) -> str:
        normalized = value.lower()
        if _OPENSM_GUID.fullmatch(normalized) is None:
            raise ValueError("fabric subnet manager GUID is not canonical")
        return normalized

    @model_validator(mode="after")
    def validate_endpoints(self) -> "FabricRailBinding":
        if self.local_gid == self.remote_gid:
            raise ValueError("fabric rail endpoint GIDs must differ")
        if not self.subnet_manager_argv or any(
            "\0" in argument for argument in self.subnet_manager_argv
        ):
            raise ValueError("fabric subnet manager argv must be nonempty and NUL-free")
        if _extract_opensm_guid(self.subnet_manager_argv) != self.subnet_manager_guid:
            raise ValueError("fabric subnet manager argv/GUID binding differs")
        return self


class CrossHostFabricBinding(StrictModel):
    rails: tuple[FabricRailBinding, FabricRailBinding]

    @model_validator(mode="after")
    def validate_rails(self) -> "CrossHostFabricBinding":
        if tuple(rail.local_port for rail in self.rails) != (1, 2):
            raise ValueError("fabric rails must be ordered by local ports 1 and 2")
        if len({rail.remote_port for rail in self.rails}) != 2:
            raise ValueError("fabric rails must bind each remote port exactly once")
        gids = {gid for rail in self.rails for gid in (rail.local_gid, rail.remote_gid)}
        if len(gids) != 4:
            raise ValueError("fabric endpoint GIDs must be globally unique")
        return self


class HostGuardConfig(StrictModel):
    """Complete contract for guarding one idle, nonparticipating peer."""

    schema_version: Literal[1] = HOST_GUARD_SCHEMA_VERSION
    peer_role: Literal["idle_nonparticipant"] = "idle_nonparticipant"
    peer: CoordinationPeerBinding
    cross_host_fabric: CrossHostFabricBinding | None = None

    @model_validator(mode="after")
    def validate_remote_fabric_endpoints(self) -> "HostGuardConfig":
        if self.cross_host_fabric is None:
            return self
        remote_ports = {port.port: port for port in self.peer.hca.ports}
        remote_managers = {unit.port: unit for unit in self.peer.opensm_units}
        for rail in self.cross_host_fabric.rails:
            remote = remote_ports[rail.remote_port]
            if (rail.remote_gid, rail.rate) != (remote.gid, remote.expected_rate):
                raise ValueError(
                    "cross-host fabric remote endpoint differs from idle peer binding"
                )
            if rail.subnet_manager_host == "remote":
                manager = remote_managers[rail.remote_port]
                if (
                    rail.subnet_manager_unit,
                    rail.subnet_manager_guid,
                    rail.subnet_manager_argv,
                ) != (manager.unit, manager.guid, manager.argv):
                    raise ValueError(
                        "cross-host fabric remote manager differs from idle peer binding"
                    )
        return self


class CommandResult(StrictModel):
    return_code: int
    stdout: bytes
    stderr: bytes


class CommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> CommandResult: ...


def calculate_coordination_peer_binding_sha256(
    binding: CoordinationPeerBinding,
) -> str:
    return calculate_host_guard_sha256(binding)


def calculate_host_guard_config_sha256(config: HostGuardConfig) -> str:
    return calculate_host_guard_sha256(config)


def _model_payload_without(model: BaseModel, field_name: str) -> JsonObject:
    return cast(
        JsonObject,
        model.model_dump(mode="json", exclude={field_name}),
    )


def calculate_remote_probe_request_sha256(request: RemoteProbeRequest) -> str:
    return calculate_host_guard_sha256(
        _model_payload_without(request, "request_sha256")
    )


def calculate_remote_probe_receipt_sha256(receipt: RemoteProbeReceipt) -> str:
    return calculate_host_guard_sha256(
        _model_payload_without(receipt, "receipt_sha256")
    )


def calculate_host_guard_snapshot_sha256(snapshot: HostGuardSnapshot) -> str:
    return calculate_host_guard_sha256(
        _model_payload_without(snapshot, "snapshot_sha256")
    )


def _stable_file_stat(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_uid,
        file_stat.st_gid,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _open_bound_file(binding: FileIdentityBinding) -> tuple[int, Path]:
    path = Path(binding.path)
    try:
        resolved = path.resolve(strict=True)
        if str(resolved) != binding.resolved_path:
            raise HostGuardError(f"resolved path changed for {path}")
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise HostGuardError(f"cannot open bound file {path}: {error}") from error
    return descriptor, resolved


def _observe_open_file(
    descriptor: int,
    binding: FileIdentityBinding,
    resolved: Path,
) -> FileIdentityObservation:
    path = Path(binding.path)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HostGuardError(f"bound file is not regular: {path}")
        digest = hashlib.sha256()
        bytes_read = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            bytes_read += len(block)
        after = os.fstat(descriptor)
        if (
            _stable_file_stat(before) != _stable_file_stat(after)
            or bytes_read != before.st_size
        ):
            raise HostGuardError(f"bound file changed while hashing: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = FileIdentityObservation(
            path=binding.path,
            resolved_path=str(resolved),
            size_bytes=bytes_read,
            sha256=digest.hexdigest(),
        )
    except OSError as error:
        raise HostGuardError(f"cannot inspect bound file {path}: {error}") from error
    if observed.sha256 != binding.sha256:
        raise HostGuardError(f"bound file digest changed for {path}")
    return observed


def observe_file_identity(binding: FileIdentityBinding) -> FileIdentityObservation:
    """Verify one path, its resolved target, and a stable SHA-256 digest."""

    descriptor, resolved = _open_bound_file(binding)
    try:
        return _observe_open_file(descriptor, binding, resolved)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _pinned_bound_file(
    binding: FileIdentityBinding,
) -> Iterator[tuple[int, FileIdentityObservation]]:
    """Keep a verified inode open and prove its path was not replaced in use."""

    descriptor, resolved = _open_bound_file(binding)
    try:
        before = _observe_open_file(descriptor, binding, resolved)
        pinned_stat = os.fstat(descriptor)
        try:
            yield descriptor, before
        finally:
            after = _observe_open_file(descriptor, binding, resolved)
            if after != before:
                raise HostGuardError(f"bound file changed while in use: {binding.path}")
            current_descriptor, current_resolved = _open_bound_file(binding)
            try:
                current_stat = os.fstat(current_descriptor)
                if (current_stat.st_dev, current_stat.st_ino) != (
                    pinned_stat.st_dev,
                    pinned_stat.st_ino,
                ):
                    raise HostGuardError(
                        f"bound file path was replaced in use: {binding.path}"
                    )
                if (
                    _observe_open_file(current_descriptor, binding, current_resolved)
                    != before
                ):
                    raise HostGuardError(
                        f"bound file path identity changed in use: {binding.path}"
                    )
            finally:
                os.close(current_descriptor)
    finally:
        os.close(descriptor)


def run_bound_executable(
    executable: FileIdentityBinding,
    arguments: tuple[str, ...],
    *,
    runner: CommandRunner,
    input_bytes: bytes = b"",
    timeout_seconds: float = 15.0,
    maximum_stdout_bytes: int = MAXIMUM_WIRE_BYTES,
    maximum_stderr_bytes: int = MAXIMUM_COMMAND_STDERR_BYTES,
) -> CommandResult:
    with _pinned_bound_file(executable) as (descriptor, _observation):
        return runner(
            (f"/proc/self/fd/{descriptor}", *arguments),
            input_bytes=input_bytes,
            timeout_seconds=timeout_seconds,
            maximum_stdout_bytes=maximum_stdout_bytes,
            maximum_stderr_bytes=maximum_stderr_bytes,
            environment=sanitized_command_environment(),
            pass_fds=(descriptor,),
        )


def sanitized_command_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.01)
    return not _process_group_exists(process_group_id)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    process_group_id = process.pid
    if _process_group_exists(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
        _wait_for_process_group_exit(process_group_id, 0.25)
    if _process_group_exists(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as error:
        raise HostGuardTransportError("command leader resisted SIGKILL") from error
    if process.poll() is None:
        raise HostGuardTransportError("command leader was not reaped")
    if not _wait_for_process_group_exit(process_group_id, 2.0):
        raise HostGuardTransportError("command process group cleanup was not confirmed")


def run_bounded_command(
    command: tuple[str, ...],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    environment: Mapping[str, str],
    pass_fds: tuple[int, ...] = (),
) -> CommandResult:
    """Run an argv directly while bounding stdin, stdout, stderr, and time."""

    if (
        not command
        or len(input_bytes) > MAXIMUM_WIRE_BYTES
        or timeout_seconds <= 0
        or maximum_stdout_bytes <= 0
        or maximum_stderr_bytes <= 0
        or tuple(sorted(set(pass_fds))) != pass_fds
        or any(descriptor < 3 for descriptor in pass_fds)
    ):
        raise HostGuardTransportError("invalid bounded command request")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            close_fds=True,
            pass_fds=pass_fds,
            start_new_session=True,
        )
    except OSError as error:
        raise HostGuardTransportError(f"cannot start command: {error}") from error
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    streams: dict[int, tuple[Literal["stdin", "stdout", "stderr"], IO[bytes]]] = {
        process.stdin.fileno(): ("stdin", process.stdin),
        process.stdout.fileno(): ("stdout", process.stdout),
        process.stderr.fileno(): ("stderr", process.stderr),
    }
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    input_offset = 0
    try:
        for descriptor, (kind, _stream) in streams.items():
            os.set_blocking(descriptor, False)
            events = selectors.EVENT_WRITE if kind == "stdin" else selectors.EVENT_READ
            selector.register(descriptor, events, kind)
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HostGuardTransportError("command timed out")
            for key, _events in selector.select(remaining):
                descriptor = key.fd
                kind = cast(Literal["stdin", "stdout", "stderr"], key.data)
                if kind == "stdin":
                    if input_offset == len(input_bytes):
                        selector.unregister(descriptor)
                        streams[descriptor][1].close()
                        continue
                    try:
                        written = os.write(descriptor, input_bytes[input_offset:])
                    except BrokenPipeError:
                        written = 0
                        input_offset = len(input_bytes)
                    input_offset += written
                    continue
                try:
                    block = os.read(descriptor, 65536)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(descriptor)
                    streams[descriptor][1].close()
                    continue
                output = stdout if kind == "stdout" else stderr
                limit = (
                    maximum_stdout_bytes if kind == "stdout" else maximum_stderr_bytes
                )
                output.extend(block)
                if len(output) > limit:
                    raise HostGuardTransportError(
                        f"command {kind} exceeded its size limit"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HostGuardTransportError("command timed out")
        return_code = process.wait(timeout=remaining)
        return CommandResult(
            return_code=return_code,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HostGuardTransportError(f"bounded command failed: {error}") from error
    finally:
        selector.close()
        try:
            _terminate_process(process)
        finally:
            for _kind, stream in streams.values():
                if not stream.closed:
                    stream.close()


def build_remote_probe_request(
    config: HostGuardConfig | CoordinationPeerBinding,
    *,
    now_unix_ns: int | None = None,
    lifetime_ns: int = DEFAULT_REQUEST_LIFETIME_NS,
    nonce: str | None = None,
) -> RemoteProbeRequest:
    complete_config = (
        config if isinstance(config, HostGuardConfig) else HostGuardConfig(peer=config)
    )
    binding = complete_config.peer
    issued = time.time_ns() if now_unix_ns is None else now_unix_ns
    request_nonce = secrets.token_hex(32) if nonce is None else nonce
    unsigned: JsonObject = {
        "binding_sha256": calculate_coordination_peer_binding_sha256(binding),
        "config_sha256": calculate_host_guard_config_sha256(complete_config),
        "expires_at_unix_ns": issued + lifetime_ns,
        "issued_at_unix_ns": issued,
        "nonce": request_nonce,
        "peer_binding": cast(JsonValue, binding.model_dump(mode="json")),
        "protocol": HOST_GUARD_PROTOCOL,
        "schema_version": HOST_GUARD_SCHEMA_VERSION,
    }
    payload = {
        **unsigned,
        "request_sha256": calculate_host_guard_sha256(unsigned),
    }
    try:
        return RemoteProbeRequest.model_validate_json(
            canonical_host_guard_json(payload)
        )
    except ValidationError as error:
        raise HostGuardError(f"cannot build remote probe request: {error}") from error


def _validate_request_time(request: RemoteProbeRequest, now_unix_ns: int) -> None:
    skew = request.peer_binding.policy.maximum_clock_skew_ns
    if now_unix_ns < request.issued_at_unix_ns - skew:
        raise HostGuardError("remote probe request is from the future")
    if now_unix_ns > request.expires_at_unix_ns + skew:
        raise HostGuardError("remote probe request is stale")


def build_remote_probe_receipt(
    request: RemoteProbeRequest,
    observation: HostObservation,
    *,
    observed_at_unix_ns: int | None = None,
) -> RemoteProbeReceipt:
    observed = time.time_ns() if observed_at_unix_ns is None else observed_at_unix_ns
    unsigned: JsonObject = {
        "binding_sha256": request.binding_sha256,
        "config_sha256": request.config_sha256,
        "nonce": request.nonce,
        "observation": cast(JsonValue, observation.model_dump(mode="json")),
        "observed_at_unix_ns": observed,
        "protocol": HOST_GUARD_PROTOCOL,
        "peer_binding": cast(JsonValue, request.peer_binding.model_dump(mode="json")),
        "request_expires_at_unix_ns": request.expires_at_unix_ns,
        "request_issued_at_unix_ns": request.issued_at_unix_ns,
        "request_sha256": request.request_sha256,
        "schema_version": HOST_GUARD_SCHEMA_VERSION,
    }
    payload = {
        **unsigned,
        "receipt_sha256": calculate_host_guard_sha256(unsigned),
    }
    return RemoteProbeReceipt.model_validate_json(canonical_host_guard_json(payload))


def build_host_guard_snapshot(
    phase: Literal["preflight", "postflight"],
    receipt: RemoteProbeReceipt,
    *,
    collected_at_unix_ns: int | None = None,
) -> HostGuardSnapshot:
    collected = time.time_ns() if collected_at_unix_ns is None else collected_at_unix_ns
    unsigned: JsonObject = {
        "binding_sha256": receipt.binding_sha256,
        "config_sha256": receipt.config_sha256,
        "collected_at_unix_ns": collected,
        "phase": phase,
        "remote_receipt": cast(JsonValue, receipt.model_dump(mode="json")),
        "schema_version": HOST_GUARD_SCHEMA_VERSION,
    }
    return HostGuardSnapshot.model_validate_json(
        canonical_host_guard_json(
            {
                **unsigned,
                "snapshot_sha256": calculate_host_guard_sha256(unsigned),
            }
        )
    )


def _same_file_identity(
    observation: FileIdentityObservation,
    binding: FileIdentityBinding,
) -> bool:
    return (
        observation.path,
        observation.resolved_path,
        observation.sha256,
    ) == (binding.path, binding.resolved_path, binding.sha256)


def validate_idle_peer_observation(
    observation: HostObservation,
    binding: CoordinationPeerBinding,
) -> None:
    """Fail unless the peer exactly matches its binding and is currently idle."""

    failures: list[str] = []
    if observation.hostname != binding.hostname:
        failures.append("hostname differs")
    file_pairs = (
        ("probe Python", observation.probe_python, binding.remote_probe.python),
        ("probe script", observation.probe_script, binding.remote_probe.script),
        ("nvidia-smi", observation.nvidia_smi, binding.tools.nvidia_smi),
        ("systemctl", observation.systemctl, binding.tools.systemctl),
    )
    for description, observed, expected in file_pairs:
        if not _same_file_identity(observed, expected):
            failures.append(f"{description} identity differs")

    cpu = observation.cpu_memory
    expected_cpu = binding.cpu_memory
    if len(cpu.online_cpus) < expected_cpu.minimum_online_cpu_count:
        failures.append("online CPU count is below its binding")
    if cpu.numa_cpu_sets != expected_cpu.numa_cpu_sets:
        failures.append("NUMA CPU topology differs")
    if cpu.memory_total_bytes < expected_cpu.minimum_total_memory_bytes:
        failures.append("total memory is below its binding")
    if cpu.memory_available_bytes < binding.policy.minimum_available_memory_bytes:
        failures.append("available memory is below the idle threshold")
    normalized_load = cpu.load_average_1m / len(cpu.online_cpus)
    if normalized_load > binding.policy.maximum_load_1m_per_online_cpu:
        failures.append("one-minute load per CPU exceeds the idle threshold")
    missing_flags = set(expected_cpu.required_cpu_flags) - set(cpu.common_cpu_flags)
    if missing_flags:
        failures.append(f"required CPU/AMX flags are missing: {sorted(missing_flags)}")

    expected_gpus = {gpu.uuid: gpu for gpu in binding.gpus}
    observed_gpus = {gpu.uuid: gpu for gpu in observation.gpus}
    if tuple(sorted(observation.gpus, key=lambda gpu: gpu.uuid)) != observation.gpus:
        failures.append("GPU observations are not canonically ordered")
    if set(observed_gpus) != set(expected_gpus):
        failures.append("GPU inventory UUIDs differ")
    for uuid, expected in expected_gpus.items():
        actual = observed_gpus.get(uuid)
        if actual is None:
            continue
        if (
            actual.pci_bus_id,
            actual.name,
            actual.memory_total_bytes,
        ) != (expected.pci_bus_id, expected.name, expected.memory_total_bytes):
            failures.append(f"GPU {uuid} immutable inventory differs")
        if actual.memory_used_bytes > binding.policy.maximum_gpu_memory_used_bytes:
            failures.append(f"GPU {uuid} memory use exceeds the idle threshold")
        if (
            actual.gpu_utilization_percent
            > binding.policy.maximum_gpu_utilization_percent
        ):
            failures.append(f"GPU {uuid} utilization exceeds the idle threshold")
        if (
            actual.memory_utilization_percent
            > binding.policy.maximum_gpu_memory_utilization_percent
        ):
            failures.append(f"GPU {uuid} memory utilization exceeds the idle threshold")
        if actual.temperature_celsius > binding.policy.maximum_gpu_temperature_celsius:
            failures.append(f"GPU {uuid} temperature exceeds the idle threshold")
    if observation.gpu_processes:
        failures.append("GPU compute processes are present")
    if observation.conflicting_processes:
        failures.append("benchmark, model, storage, or profiler processes are present")
    unsafe_names = set(binding.policy.unsafe_profiler_kernel_modules)
    for module in observation.unsafe_profiler_modules:
        if module.name not in unsafe_names:
            failures.append(
                f"unexpected unsafe profiler module evidence: {module.name}"
            )
        if module.reference_count != 0 or module.state != "Live":
            failures.append(
                f"unsafe profiler module {module.name} is active or unstable"
            )
    if observation.raid_sync_conflicts:
        failures.append("RAID synchronization is active")
    if observation.unused_reserved_ports != binding.reserved_ports:
        failures.append("not every reserved TCP/UDP port was proven unused")

    hca = observation.hca
    if (hca.device, hca.node_guid) != (binding.hca.device, binding.hca.node_guid):
        failures.append("HCA identity differs")
    for actual, expected in zip(hca.ports, binding.hca.ports, strict=True):
        if (actual.port, actual.gid, actual.rate) != (
            expected.port,
            expected.gid,
            expected.expected_rate,
        ):
            failures.append(f"HCA port {expected.port} identity or rate differs")
        if (actual.counter_device, actual.counter_port) != (
            binding.hca.device,
            expected.port,
        ):
            failures.append(f"HCA port {expected.port} counter source differs")
        if "ACTIVE" not in actual.state.upper():
            failures.append(f"HCA port {expected.port} is not ACTIVE")
        if "LINKUP" not in actual.physical_state.upper():
            failures.append(f"HCA port {expected.port} is not physically LinkUp")
        if actual.lid <= 0 or actual.sm_lid <= 0:
            failures.append(f"HCA port {expected.port} has an unassigned LID or SM LID")
        for counter, maximum in expected.health_counter_maximums.items():
            value = actual.counters.get(counter)
            if value is None:
                failures.append(f"HCA port {expected.port} lacks counter {counter}")
            elif value > maximum:
                failures.append(
                    f"HCA port {expected.port} counter {counter} exceeds its bound"
                )

    for actual, expected in zip(
        observation.opensm_units, binding.opensm_units, strict=True
    ):
        if (actual.unit, actual.port, actual.guid) != (
            expected.unit,
            expected.port,
            expected.guid,
        ):
            failures.append(f"OpenSM unit for port {expected.port} differs")
        if (actual.load_state, actual.active_state, actual.sub_state) != (
            "loaded",
            "active",
            "running",
        ):
            failures.append(f"OpenSM unit {expected.unit} is not loaded/active/running")
        if not _same_file_identity(actual.executable, expected.executable):
            failures.append(f"OpenSM unit {expected.unit} executable differs")
        if actual.argv != expected.argv:
            failures.append(f"OpenSM unit {expected.unit} command line differs")
        if actual.version != expected.version:
            failures.append(f"OpenSM unit {expected.unit} version differs")
        if (
            actual.main_pid <= 0
            or min(
                actual.systemd_start_monotonic_us,
                actual.process_start_time_ticks,
            )
            <= 0
        ):
            failures.append(f"OpenSM unit {expected.unit} lacks process identity")

    if failures:
        raise HostGuardError("idle peer validation failed: " + "; ".join(failures))


def verify_remote_probe_receipt(
    receipt: RemoteProbeReceipt,
    request: RemoteProbeRequest,
    config: HostGuardConfig | CoordinationPeerBinding,
    *,
    now_unix_ns: int | None = None,
) -> None:
    complete_config = (
        config if isinstance(config, HostGuardConfig) else HostGuardConfig(peer=config)
    )
    binding = complete_config.peer
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    _validate_request_time(request, now)
    expected_binding_sha256 = calculate_coordination_peer_binding_sha256(binding)
    expected_config_sha256 = calculate_host_guard_config_sha256(complete_config)
    if (
        request.peer_binding != binding
        or request.binding_sha256 != expected_binding_sha256
    ):
        raise HostGuardError(
            "request does not contain the independently supplied peer binding"
        )
    if receipt.nonce != request.nonce:
        raise HostGuardError("remote receipt nonce does not match the request")
    if receipt.request_sha256 != request.request_sha256:
        raise HostGuardError("remote receipt request digest does not match")
    if receipt.binding_sha256 != expected_binding_sha256:
        raise HostGuardError("remote receipt peer binding digest does not match")
    if request.config_sha256 != expected_config_sha256:
        raise HostGuardError("request complete host guard config digest does not match")
    if receipt.config_sha256 != expected_config_sha256:
        raise HostGuardError(
            "remote receipt complete host guard config digest does not match"
        )
    if receipt.peer_binding != binding:
        raise HostGuardError("remote receipt peer binding does not match")
    if (
        receipt.request_issued_at_unix_ns != request.issued_at_unix_ns
        or receipt.request_expires_at_unix_ns != request.expires_at_unix_ns
    ):
        raise HostGuardError("remote receipt request timestamps do not match")
    skew = binding.policy.maximum_clock_skew_ns
    if receipt.observed_at_unix_ns < request.issued_at_unix_ns - skew:
        raise HostGuardError("remote receipt observation is stale")
    if receipt.observed_at_unix_ns > min(now, request.expires_at_unix_ns) + skew:
        raise HostGuardError("remote receipt observation is from the future")
    validate_idle_peer_observation(receipt.observation, binding)


def collect_remote_snapshot(
    config: HostGuardConfig | CoordinationPeerBinding,
    phase: Literal["preflight", "postflight"],
    *,
    runner: CommandRunner = run_bounded_command,
    now_unix_ns: Callable[[], int] = time.time_ns,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(32),
) -> HostGuardSnapshot:
    """Collect and verify one idle-peer snapshot over the exact SSH transport."""

    complete_config = (
        config if isinstance(config, HostGuardConfig) else HostGuardConfig(peer=config)
    )
    binding = complete_config.peer
    issued = now_unix_ns()
    request = build_remote_probe_request(
        complete_config,
        now_unix_ns=issued,
        nonce=nonce_factory(),
    )
    with contextlib.ExitStack() as stack:
        ssh_descriptor, _ = stack.enter_context(
            _pinned_bound_file(binding.ssh.executable)
        )
        _known_hosts_descriptor, known_hosts_observation = stack.enter_context(
            _pinned_bound_file(binding.ssh.known_hosts_file)
        )
        _identity_descriptor, identity_observation = stack.enter_context(
            _pinned_bound_file(binding.ssh.identity_file)
        )
        command = binding.ssh.command(
            binding.remote_probe,
            executable_path=f"/proc/self/fd/{ssh_descriptor}",
            known_hosts_path=known_hosts_observation.resolved_path,
            identity_path=identity_observation.resolved_path,
        )
        result = runner(
            command,
            input_bytes=canonical_host_guard_json(request),
            timeout_seconds=float(
                binding.ssh.connect_timeout_seconds
                + binding.ssh.server_alive_interval_seconds
                * binding.ssh.server_alive_count_max
                + 10
            ),
            maximum_stdout_bytes=MAXIMUM_WIRE_BYTES,
            maximum_stderr_bytes=MAXIMUM_COMMAND_STDERR_BYTES,
            environment=sanitized_command_environment(),
            pass_fds=(ssh_descriptor,),
        )
    if result.return_code != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise HostGuardTransportError(
            f"remote host guard exited {result.return_code}: {stderr}"
        )
    if result.stderr:
        raise HostGuardTransportError("remote host guard emitted unexpected stderr")
    try:
        payload = parse_bounded_canonical_json(result.stdout)
        receipt = RemoteProbeReceipt.model_validate_json(
            canonical_host_guard_json(payload)
        )
    except (HostGuardError, ValidationError) as error:
        raise HostGuardTransportError(
            f"remote host guard output is invalid: {error}"
        ) from error
    collected = now_unix_ns()
    verify_remote_probe_receipt(
        receipt,
        request,
        complete_config,
        now_unix_ns=collected,
    )
    return build_host_guard_snapshot(
        phase,
        receipt,
        collected_at_unix_ns=collected,
    )


def _immutable_observation_identity(observation: HostObservation) -> JsonObject:
    return {
        "boot_id": observation.boot_id,
        "cpu": cast(
            JsonValue,
            {
                "online_cpus": list(observation.cpu_memory.online_cpus),
                "numa_cpu_sets": observation.cpu_memory.numa_cpu_sets,
                "memory_total_bytes": observation.cpu_memory.memory_total_bytes,
                "common_cpu_flags": list(observation.cpu_memory.common_cpu_flags),
            },
        ),
        "files": cast(
            JsonValue,
            {
                "probe_python": observation.probe_python.model_dump(mode="json"),
                "probe_script": observation.probe_script.model_dump(mode="json"),
                "nvidia_smi": observation.nvidia_smi.model_dump(mode="json"),
                "systemctl": observation.systemctl.model_dump(mode="json"),
            },
        ),
        "gpus": cast(
            JsonValue,
            [
                {
                    "uuid": gpu.uuid,
                    "pci_bus_id": gpu.pci_bus_id,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.memory_total_bytes,
                }
                for gpu in observation.gpus
            ],
        ),
        "hca": cast(
            JsonValue,
            {
                "device": observation.hca.device,
                "node_guid": observation.hca.node_guid,
                "ports": [
                    {"port": port.port, "gid": port.gid, "rate": port.rate}
                    for port in observation.hca.ports
                ],
            },
        ),
        "hostname": observation.hostname,
        "unsafe_profiler_modules": cast(
            JsonValue,
            [
                module.model_dump(mode="json")
                for module in observation.unsafe_profiler_modules
            ],
        ),
        "opensm_units": cast(
            JsonValue,
            [unit.model_dump(mode="json") for unit in observation.opensm_units],
        ),
    }


def compare_host_guard_snapshots(
    preflight: HostGuardSnapshot,
    postflight: HostGuardSnapshot,
) -> HostGuardComparison:
    """Compare immutable identities and fail-sensitive HCA counters across a run."""

    failures: list[str] = []
    data_deltas: dict[str, int] = {}
    data_limits: dict[str, int] = {}
    health_deltas: dict[str, int] = {}
    if preflight.phase != "preflight" or postflight.phase != "postflight":
        failures.append("snapshots are not ordered preflight then postflight")
    if postflight.collected_at_unix_ns <= preflight.collected_at_unix_ns:
        failures.append("postflight was not collected after preflight")
    if (
        preflight.remote_receipt.nonce == postflight.remote_receipt.nonce
        or preflight.remote_receipt.request_sha256
        == postflight.remote_receipt.request_sha256
    ):
        failures.append("preflight and postflight reused one remote request")
    if preflight.binding_sha256 != postflight.binding_sha256:
        failures.append("peer binding changed between snapshots")
    if preflight.config_sha256 != postflight.config_sha256:
        failures.append("complete host guard config changed between snapshots")
    if preflight.remote_receipt.peer_binding != postflight.remote_receipt.peer_binding:
        failures.append("peer counter policy changed between snapshots")
    before = preflight.observation
    after = postflight.observation
    if _immutable_observation_identity(before) != _immutable_observation_identity(
        after
    ):
        failures.append("peer hardware, tool, or OpenSM process identity changed")
    expected_ports = {
        port.port: port for port in preflight.remote_receipt.peer_binding.hca.ports
    }
    for before_port, after_port in zip(before.hca.ports, after.hca.ports, strict=True):
        expected_port = expected_ports[before_port.port]
        for counter in DATA_COUNTER_NAMES:
            key = f"port{before_port.port}.{counter}"
            before_value = before_port.counters[counter]
            after_value = after_port.counters[counter]
            maximum = expected_port.idle_data_counter_maximum_deltas[counter]
            data_limits[key] = maximum
            if after_value < before_value:
                data_deltas[key] = 0
                failures.append(f"HCA data counter reset: {key}")
                continue
            delta = after_value - before_value
            data_deltas[key] = delta
            if delta > maximum:
                failures.append(
                    f"HCA data counter exceeded idle OpenSM tolerance: {key} "
                    f"({delta} > {maximum})"
                )
        for counter in HEALTH_COUNTER_NAMES:
            before_value = before_port.counters.get(counter)
            after_value = after_port.counters.get(counter)
            if before_value is None and after_value is None:
                continue
            key = f"port{before_port.port}.{counter}"
            if before_value is None or after_value is None:
                failures.append(f"HCA health counter availability changed: {key}")
                continue
            if after_value < before_value:
                failures.append(f"HCA health counter reset: {key}")
                continue
            delta = after_value - before_value
            health_deltas[key] = delta
            if delta:
                failures.append(f"HCA health counter increased: {key}")
    return HostGuardComparison(
        preflight_snapshot_sha256=preflight.snapshot_sha256,
        postflight_snapshot_sha256=postflight.snapshot_sha256,
        binding_sha256=preflight.binding_sha256,
        config_sha256=preflight.config_sha256,
        data_counter_deltas=data_deltas,
        data_counter_maximum_deltas=data_limits,
        health_counter_deltas=health_deltas,
        stable=not failures,
        failures=tuple(failures),
    )


def compare_snapshots(
    preflight: HostGuardSnapshot,
    postflight: HostGuardSnapshot,
) -> HostGuardComparison:
    """Public short name for the strict preflight/postflight comparison."""

    return compare_host_guard_snapshots(preflight, postflight)


def _host_observation(value: HostGuardSnapshot | HostObservation) -> HostObservation:
    return value.observation if isinstance(value, HostGuardSnapshot) else value


def validate_cross_host_fabric(
    local_snapshot: HostGuardSnapshot | HostObservation,
    remote_snapshot: HostGuardSnapshot | HostObservation,
    binding: CrossHostFabricBinding,
) -> None:
    """Validate exact rail endpoints and subnet-manager agreement per rail."""

    local = _host_observation(local_snapshot)
    remote = _host_observation(remote_snapshot)
    local_ports = {port.port: port for port in local.hca.ports}
    remote_ports = {port.port: port for port in remote.hca.ports}
    local_managers = {unit.port: unit for unit in local.opensm_units}
    remote_managers = {unit.port: unit for unit in remote.opensm_units}
    failures: list[str] = []
    for rail in binding.rails:
        local_port = local_ports.get(rail.local_port)
        remote_port = remote_ports.get(rail.remote_port)
        if local_port is None or remote_port is None:
            failures.append(f"rail {rail.local_port} endpoint is missing")
            continue
        if (local_port.gid, remote_port.gid) != (rail.local_gid, rail.remote_gid):
            failures.append(f"rail {rail.local_port} GID binding differs")
        if (
            local_port.counter_device,
            local_port.counter_port,
            remote_port.counter_device,
            remote_port.counter_port,
        ) != (
            local.hca.device,
            rail.local_port,
            remote.hca.device,
            rail.remote_port,
        ):
            failures.append(
                f"rail {rail.local_port} counter source is not coherent with endpoints"
            )
        if local_port.rate != rail.rate or remote_port.rate != rail.rate:
            failures.append(f"rail {rail.local_port} rate differs")
        if (
            "ACTIVE" not in local_port.state.upper()
            or "ACTIVE" not in remote_port.state.upper()
        ):
            failures.append(f"rail {rail.local_port} is not ACTIVE on both hosts")
        if (
            "LINKUP" not in local_port.physical_state.upper()
            or "LINKUP" not in remote_port.physical_state.upper()
        ):
            failures.append(f"rail {rail.local_port} is not LinkUp on both hosts")
        if (
            min(local_port.lid, remote_port.lid, local_port.sm_lid, remote_port.sm_lid)
            <= 0
        ):
            failures.append(f"rail {rail.local_port} has an unassigned LID")
            continue
        if local_port.sm_lid != remote_port.sm_lid:
            failures.append(f"rail {rail.local_port} endpoints disagree on SM LID")
        manager_lid = (
            local_port.lid if rail.subnet_manager_host == "local" else remote_port.lid
        )
        if local_port.sm_lid != manager_lid:
            failures.append(
                f"rail {rail.local_port} SM LID is not the bound manager LID"
            )
        manager_port = (
            rail.local_port if rail.subnet_manager_host == "local" else rail.remote_port
        )
        manager = (
            local_managers.get(manager_port)
            if rail.subnet_manager_host == "local"
            else remote_managers.get(manager_port)
        )
        if manager is None:
            failures.append(f"rail {rail.local_port} bound OpenSM manager is missing")
            continue
        if (
            manager.unit,
            manager.port,
            manager.guid,
            manager.argv,
        ) != (
            rail.subnet_manager_unit,
            manager_port,
            rail.subnet_manager_guid,
            rail.subnet_manager_argv,
        ):
            failures.append(
                f"rail {rail.local_port} OpenSM unit/port/GUID/argv binding differs"
            )
        if (manager.load_state, manager.active_state, manager.sub_state) != (
            "loaded",
            "active",
            "running",
        ):
            failures.append(f"rail {rail.local_port} OpenSM manager is not running")
    if failures:
        raise HostGuardError(
            "cross-host fabric validation failed: " + "; ".join(failures)
        )


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        raise HostGuardError(f"cannot read {description} at {path}: {error}") from error


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    try:
        for part in value.strip().split(","):
            if not part:
                raise ValueError("empty component")
            if "-" in part:
                first_text, last_text = part.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 0 or last < first:
                    raise ValueError("invalid range")
                cpus.update(range(first, last + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError("negative CPU")
                cpus.add(cpu)
    except ValueError as error:
        raise HostGuardError(f"invalid Linux CPU list {value!r}: {error}") from error
    if not cpus:
        raise HostGuardError("Linux CPU list is empty")
    return tuple(sorted(cpus))


def _load_averages(proc_root: Path) -> tuple[float, float, float]:
    fields = _read_text(proc_root / "loadavg", "load average").split()
    try:
        values = tuple(float(field) for field in fields[:3])
    except ValueError as error:
        raise HostGuardError("cannot parse load average") from error
    if len(values) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in values
    ):
        raise HostGuardError(
            "load average must contain three finite nonnegative values"
        )
    return values


def _memory_bytes(proc_root: Path) -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in _read_text(proc_root / "meminfo", "memory information").splitlines():
        name, separator, remainder = line.partition(":")
        if name not in {"MemTotal", "MemAvailable", "MemFree"}:
            continue
        fields = remainder.split()
        if separator != ":" or len(fields) != 2 or fields[1] != "kB":
            raise HostGuardError(f"cannot parse {name} in memory information")
        try:
            values[name] = int(fields[0]) * 1024
        except ValueError as error:
            raise HostGuardError(
                f"cannot parse {name} in memory information"
            ) from error
    if set(values) != {"MemTotal", "MemAvailable", "MemFree"}:
        raise HostGuardError("memory information is incomplete")
    return values["MemTotal"], values["MemAvailable"], values["MemFree"]


def _common_cpu_flags(proc_root: Path, online_cpus: Sequence[int]) -> tuple[str, ...]:
    records: dict[int, set[str]] = {}
    processor: int | None = None
    flags: set[str] | None = None
    for line in (
        *_read_text(proc_root / "cpuinfo", "CPU information").splitlines(),
        "",
    ):
        if line:
            name, separator, value = line.partition(":")
            if separator != ":":
                continue
            if name.strip() == "processor":
                try:
                    processor = int(value.strip())
                except ValueError as error:
                    raise HostGuardError("cannot parse processor index") from error
            elif name.strip() in {"flags", "Features"}:
                flags = set(value.split())
            continue
        if processor is not None and flags is not None:
            records[processor] = flags
        processor = None
        flags = None
    missing = set(online_cpus) - records.keys()
    if missing:
        raise HostGuardError(f"CPU information is missing CPUs {sorted(missing)}")
    common = set(records[online_cpus[0]])
    for cpu in online_cpus[1:]:
        common.intersection_update(records[cpu])
    return tuple(sorted(common))


def collect_cpu_memory_observation(
    binding: CpuMemoryBinding,
    *,
    proc_root: Path = Path("/proc"),
    sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
    sys_node_root: Path = Path("/sys/devices/system/node"),
) -> CpuMemoryObservation:
    online = _parse_cpu_list(_read_text(sys_cpu_root / "online", "online CPUs"))
    loads = _load_averages(proc_root)
    total, available, free = _memory_bytes(proc_root)
    numa = {
        node: _parse_cpu_list(
            _read_text(
                sys_node_root / f"node{node}" / "cpulist", f"NUMA node {node} CPUs"
            )
        )
        for node in binding.numa_cpu_sets
    }
    return CpuMemoryObservation(
        online_cpus=online,
        numa_cpu_sets=numa,
        load_average_1m=loads[0],
        load_average_5m=loads[1],
        load_average_15m=loads[2],
        memory_total_bytes=total,
        memory_available_bytes=available,
        memory_free_bytes=free,
        common_cpu_flags=_common_cpu_flags(proc_root, online),
    )


def _run_checked_text(
    executable: FileIdentityBinding,
    arguments: tuple[str, ...],
    *,
    runner: CommandRunner,
    maximum_stdout_bytes: int = MAXIMUM_WIRE_BYTES,
) -> str:
    result = run_bound_executable(
        executable,
        arguments,
        runner=runner,
        maximum_stdout_bytes=maximum_stdout_bytes,
    )
    if result.return_code != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise HostGuardError(f"command failed ({result.return_code}): {stderr}")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HostGuardError("command output is not UTF-8") from error


def _csv_rows(
    text: str, expected_columns: int, description: str
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    try:
        for row in csv.reader(text.splitlines(), skipinitialspace=True):
            if not row or all(not field.strip() for field in row):
                continue
            normalized = tuple(field.strip() for field in row)
            if len(normalized) != expected_columns:
                raise HostGuardError(f"{description} has the wrong number of columns")
            rows.append(normalized)
    except csv.Error as error:
        raise HostGuardError(f"cannot parse {description}: {error}") from error
    return tuple(rows)


def collect_gpu_observations(
    executable: FileIdentityBinding,
    *,
    runner: CommandRunner,
) -> tuple[tuple[GpuTelemetryObservation, ...], tuple[GpuProcessObservation, ...]]:
    inventory_text = _run_checked_text(
        executable,
        (
            "--query-gpu=uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ),
        runner=runner,
    )
    gpus: list[GpuTelemetryObservation] = []
    for row in _csv_rows(inventory_text, 9, "GPU inventory"):
        power = None if row[8] in {"N/A", "[N/A]"} else float(row[8])
        gpus.append(
            GpuTelemetryObservation(
                uuid=row[0],
                pci_bus_id=row[1].lower(),
                name=row[2],
                memory_total_bytes=int(float(row[3]) * 1024 * 1024),
                memory_used_bytes=int(float(row[4]) * 1024 * 1024),
                gpu_utilization_percent=int(row[5]),
                memory_utilization_percent=int(row[6]),
                temperature_celsius=int(row[7]),
                power_draw_watts=power,
            )
        )
    process_text = _run_checked_text(
        executable,
        (
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        runner=runner,
    )
    processes = tuple(
        GpuProcessObservation(
            gpu_uuid=row[0],
            pid=int(row[1]),
            process_name=row[2],
            used_memory_bytes=int(float(row[3]) * 1024 * 1024),
        )
        for row in _csv_rows(process_text, 4, "GPU process inventory")
    )
    return tuple(sorted(gpus, key=lambda gpu: gpu.uuid)), tuple(
        sorted(processes, key=lambda process: (process.gpu_uuid, process.pid))
    )


def classify_process(arguments: Sequence[str]) -> tuple[ProcessClass, ...]:
    if not arguments:
        return ()
    executable = Path(arguments[0]).name
    classes: set[ProcessClass] = set()
    if executable in _BENCHMARK_EXECUTABLES or re.fullmatch(
        r"ib_[a-z0-9_]+_(?:bw|lat)", executable
    ):
        classes.add("benchmark")
    if executable in _MODEL_SERVER_EXECUTABLES:
        classes.add("model_server")
    if executable in _ALWAYS_CONFLICTING_STORAGE_EXECUTABLES:
        classes.add("storage")
    if executable in _PROFILER_EXECUTABLES or executable.startswith("vtune"):
        classes.add("profiler")
    if executable in {"hf", "huggingface-cli"} and "download" in arguments[1:]:
        classes.add("storage")
    if executable in {"btrfs", "zpool"} and "scrub" in arguments[1:]:
        classes.add("storage")
    if executable == "mdadm" and any(
        argument in {"--check", "--action=check", "--action=repair"}
        for argument in arguments[1:]
    ):
        classes.add("storage")
    normalized_arguments = tuple(argument.replace("\\", "/") for argument in arguments)
    if executable in _CONDITIONAL_STORAGE_EXECUTABLES and any(
        marker in argument.lower()
        for argument in normalized_arguments[1:]
        for marker in _MODEL_STORAGE_MARKERS
    ):
        classes.add("storage")
    if executable in {"iperf", "iperf3", "qperf", "rdma_bw", "rdma_lat"}:
        classes.add("benchmark")
    if executable.startswith("ibv_") and executable.endswith("_pingpong"):
        classes.add("benchmark")
    if executable in {"mpiexec", "mpirun", "srun", "torchrun"} and any(
        "nccl" in argument.lower() or "sglang" in argument.lower()
        for argument in normalized_arguments[1:]
    ):
        classes.add("benchmark")
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-m" and any(
            arguments[index + 1] == prefix
            or arguments[index + 1].startswith(prefix + ".")
            for prefix in _MODEL_MODULE_PREFIXES
        ):
            classes.add("model_server")
    workflow_arguments = (
        normalized_arguments
        if executable in _WORKFLOW_LAUNCHERS
        or executable.startswith("python")
        or executable in WORKFLOW_SCRIPT_CLASSES
        else normalized_arguments[:1]
    )
    for argument in workflow_arguments:
        workflow_class = WORKFLOW_SCRIPT_CLASSES.get(Path(argument).name)
        if workflow_class is not None:
            classes.add(workflow_class)
    return tuple(sorted(classes))


def _process_start_time_ticks(stat_text: str) -> int:
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19:
        raise HostGuardError("process stat is malformed")
    try:
        value = int(fields[19])
    except ValueError as error:
        raise HostGuardError("process start time is malformed") from error
    if value <= 0:
        raise HostGuardError("process start time is invalid")
    return value


def collect_process_conflicts(
    *,
    ignored_pids: Sequence[int],
    proc_root: Path = Path("/proc"),
) -> tuple[ConflictingProcessObservation, ...]:
    # The probe itself is benign.  Its ancestors are intentionally inspected so
    # a profiler cannot conceal active use by launching the probe as a child.
    ignored = {*ignored_pids, os.getpid()}
    conflicts: list[ConflictingProcessObservation] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise HostGuardError(f"cannot inspect process table: {error}") from error
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) in ignored:
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes()
            stat_text = (entry / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, PermissionError) as error:
            raise HostGuardError(f"cannot inspect process {pid}: {error}") from error
        arguments = tuple(
            part.decode("utf-8", errors="replace")
            for part in command.split(b"\0")
            if part
        )
        classes = classify_process(arguments)
        if classes:
            conflicts.append(
                ConflictingProcessObservation(
                    pid=pid,
                    start_time_ticks=_process_start_time_ticks(stat_text),
                    classes=classes,
                    argv=arguments,
                )
            )
    return tuple(sorted(conflicts, key=lambda conflict: conflict.pid))


def collect_raid_sync_conflicts(
    sys_block_root: Path = Path("/sys/block"),
) -> tuple[str, ...]:
    conflicts: list[str] = []
    try:
        devices = tuple(sys_block_root.glob("md*"))
    except OSError as error:
        raise HostGuardError(f"cannot inspect RAID devices: {error}") from error
    for device in devices:
        action = device / "md" / "sync_action"
        if not action.exists():
            continue
        value = _read_text(action, f"{device.name} RAID sync action")
        if value not in _INACTIVE_RAID_ACTIONS:
            conflicts.append(f"{device.name}:{value}")
    return tuple(sorted(conflicts))


def collect_unsafe_profiler_modules(
    unsafe_module_names: Sequence[str],
    *,
    proc_modules_path: Path = Path("/proc/modules"),
) -> tuple[KernelModuleObservation, ...]:
    """Record loaded unsafe profiler drivers, including their live refcounts."""

    expected = frozenset(unsafe_module_names)
    if (
        not expected
        or tuple(sorted(expected)) != tuple(unsafe_module_names)
        or any(_KERNEL_MODULE.fullmatch(module) is None for module in expected)
    ):
        raise HostGuardError("unsafe profiler module names are not canonical")
    try:
        lines = proc_modules_path.read_text().splitlines()
    except OSError as error:
        raise HostGuardError(
            f"cannot inspect loaded kernel modules: {error}"
        ) from error
    observations: list[KernelModuleObservation] = []
    seen: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 6:
            raise HostGuardError("/proc/modules contains a malformed line")
        name = fields[0]
        if name not in expected:
            continue
        if name in seen:
            raise HostGuardError(f"/proc/modules repeats unsafe module {name}")
        try:
            size_bytes = int(fields[1])
            reference_count = int(fields[2])
        except ValueError as error:
            raise HostGuardError("unsafe module size/refcount is malformed") from error
        dependency_text = fields[3]
        dependencies = (
            ()
            if dependency_text == "-"
            else tuple(sorted(part for part in dependency_text.split(",") if part))
        )
        observations.append(
            KernelModuleObservation(
                name=name,
                size_bytes=size_bytes,
                reference_count=reference_count,
                dependencies=dependencies,
                state=fields[4],
            )
        )
        seen.add(name)
    return tuple(sorted(observations, key=lambda module: module.name))


def probe_reserved_ports_unused(ports: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(ports)
    if (
        not normalized
        or tuple(sorted(set(normalized))) != normalized
        or normalized[0] < 1
        or normalized[-1] > 65535
    ):
        raise HostGuardError("reserved ports must be sorted, unique, and valid")
    probes: list[socket.socket] = []
    try:
        for port in normalized:
            for family, socket_type, address in (
                (socket.AF_INET, socket.SOCK_STREAM, ("0.0.0.0", port)),
                (socket.AF_INET, socket.SOCK_DGRAM, ("0.0.0.0", port)),
                (socket.AF_INET6, socket.SOCK_STREAM, ("::", port)),
                (socket.AF_INET6, socket.SOCK_DGRAM, ("::", port)),
            ):
                probe = socket.socket(family, socket_type)
                probes.append(probe)
                if family == socket.AF_INET6:
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind(address)
                if socket_type == socket.SOCK_STREAM:
                    probe.listen(1)
    except OSError as error:
        raise HostGuardError(
            f"reserved TCP/UDP port is unavailable: {error}"
        ) from error
    finally:
        for probe in reversed(probes):
            probe.close()
    return normalized


def _read_counter(roots: Sequence[Path], name: str, *, required: bool) -> int | None:
    for root in roots:
        path = root / name
        try:
            text = path.read_text().strip()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise HostGuardError(f"cannot read HCA counter {path}: {error}") from error
        try:
            value = int(text)
        except ValueError as error:
            raise HostGuardError(f"HCA counter {path} is not an integer") from error
        if value < 0:
            raise HostGuardError(f"HCA counter {path} is negative")
        return value
    if required:
        raise HostGuardError(f"required HCA counter {name} is unavailable")
    return None


def collect_hca_observation(
    binding: HcaBinding,
    *,
    sys_infiniband_root: Path = Path("/sys/class/infiniband"),
) -> HcaObservation:
    root = sys_infiniband_root / binding.device
    ports: list[HcaPortObservation] = []
    for expected in binding.ports:
        port_root = root / "ports" / str(expected.port)
        counters: dict[str, int] = {}
        counter_roots = (port_root / "counters", port_root / "hw_counters")
        for name in DATA_COUNTER_NAMES:
            value = _read_counter(counter_roots, name, required=True)
            assert value is not None
            counters[name] = value
        for name in HEALTH_COUNTER_NAMES:
            value = _read_counter(
                counter_roots,
                name,
                required=name in expected.health_counter_maximums,
            )
            if value is not None:
                counters[name] = value
        try:
            lid = int(_read_text(port_root / "lid", "HCA port LID"), 0)
            sm_lid = int(_read_text(port_root / "sm_lid", "HCA port SM LID"), 0)
        except ValueError as error:
            raise HostGuardError("HCA LID is not an integer") from error
        ports.append(
            HcaPortObservation(
                port=expected.port,
                gid=ipaddress.IPv6Address(
                    _read_text(
                        port_root / "gids" / str(expected.gid_index), "HCA port GID"
                    )
                ).exploded,
                state=_read_text(port_root / "state", "HCA port state"),
                physical_state=_read_text(
                    port_root / "phys_state", "HCA physical state"
                ),
                rate=_read_text(port_root / "rate", "HCA port rate"),
                lid=lid,
                sm_lid=sm_lid,
                counter_device=binding.device,
                counter_port=expected.port,
                counters=counters,
            )
        )
    return HcaObservation(
        device=binding.device,
        node_guid=_read_text(root / "node_guid", "HCA node GUID").lower(),
        ports=(ports[0], ports[1]),
    )


def _parse_systemctl_show(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or not key or key in values:
            raise HostGuardError("systemctl show output is malformed or repeats a key")
        values[key] = value
    expected = {
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainStartTimestampMonotonic",
    }
    if set(values) != expected:
        raise HostGuardError(
            "systemctl show output has missing or unexpected properties"
        )
    return values


def collect_opensm_observations(
    bindings: tuple[OpenSmUnitBinding, OpenSmUnitBinding],
    systemctl: FileIdentityBinding,
    *,
    runner: CommandRunner,
    proc_root: Path = Path("/proc"),
) -> tuple[OpenSmUnitObservation, OpenSmUnitObservation]:
    observations: list[OpenSmUnitObservation] = []
    for binding in bindings:
        text = _run_checked_text(
            systemctl,
            (
                "show",
                binding.unit,
                "--property=Id",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=ExecMainStartTimestampMonotonic",
                "--no-pager",
            ),
            runner=runner,
        )
        values = _parse_systemctl_show(text)
        try:
            pid = int(values["MainPID"])
            systemd_start = int(values["ExecMainStartTimestampMonotonic"])
        except ValueError as error:
            raise HostGuardError(
                "OpenSM systemd PID/start identity is invalid"
            ) from error
        if pid <= 0 or systemd_start <= 0:
            raise HostGuardError("OpenSM systemd unit has no live process identity")
        process_root = proc_root / str(pid)
        try:
            stat_text_before = (process_root / "stat").read_text()
            argv = tuple(
                part.decode("utf-8", errors="strict")
                for part in (process_root / "cmdline").read_bytes().split(b"\0")
                if part
            )
            executable_resolved = str((process_root / "exe").resolve(strict=True))
            stat_text_after = (process_root / "stat").read_text()
        except (OSError, UnicodeDecodeError) as error:
            raise HostGuardError(
                f"cannot inspect OpenSM process {pid}: {error}"
            ) from error
        process_start_time = _process_start_time_ticks(stat_text_before)
        if process_start_time != _process_start_time_ticks(stat_text_after):
            raise HostGuardError(
                "OpenSM process identity changed while it was inspected"
            )
        try:
            observed_guid = _extract_opensm_guid(argv)
        except ValueError as error:
            raise HostGuardError(f"OpenSM process GUID is invalid: {error}") from error
        executable = observe_file_identity(binding.executable)
        if executable_resolved != executable.resolved_path:
            raise HostGuardError("OpenSM /proc executable does not match its binding")
        version_result = run_bound_executable(
            binding.executable,
            ("--version",),
            runner=runner,
            timeout_seconds=10.0,
            maximum_stdout_bytes=64 * 1024,
            maximum_stderr_bytes=64 * 1024,
        )
        if version_result.return_code != 0:
            raise HostGuardError("OpenSM version command failed")
        try:
            version = (
                (version_result.stdout + version_result.stderr).decode("utf-8").strip()
            )
        except UnicodeDecodeError as error:
            raise HostGuardError("OpenSM version is not UTF-8") from error
        confirmation_text = _run_checked_text(
            systemctl,
            (
                "show",
                binding.unit,
                "--property=Id",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=ExecMainStartTimestampMonotonic",
                "--no-pager",
            ),
            runner=runner,
        )
        if _parse_systemctl_show(confirmation_text) != values:
            raise HostGuardError(
                "OpenSM systemd identity changed while it was inspected"
            )
        observations.append(
            OpenSmUnitObservation(
                unit=values["Id"],
                port=binding.port,
                guid=observed_guid,
                load_state=values["LoadState"],
                active_state=values["ActiveState"],
                sub_state=values["SubState"],
                main_pid=pid,
                systemd_start_monotonic_us=systemd_start,
                process_start_time_ticks=process_start_time,
                argv=argv,
                executable=executable,
                version=version,
            )
        )
    return observations[0], observations[1]


def collect_host_observation(
    binding: CoordinationPeerBinding,
    *,
    runner: CommandRunner = run_bounded_command,
    proc_root: Path = Path("/proc"),
    sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
    sys_node_root: Path = Path("/sys/devices/system/node"),
    sys_block_root: Path = Path("/sys/block"),
    sys_infiniband_root: Path = Path("/sys/class/infiniband"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    proc_modules_path: Path = Path("/proc/modules"),
    hostname: Callable[[], str] = socket.gethostname,
) -> HostObservation:
    probe_python = observe_file_identity(binding.remote_probe.python)
    probe_script = observe_file_identity(binding.remote_probe.script)
    nvidia_smi = observe_file_identity(binding.tools.nvidia_smi)
    systemctl = observe_file_identity(binding.tools.systemctl)
    opensm = collect_opensm_observations(
        binding.opensm_units,
        binding.tools.systemctl,
        runner=runner,
        proc_root=proc_root,
    )
    gpus, gpu_processes = collect_gpu_observations(
        binding.tools.nvidia_smi,
        runner=runner,
    )
    return HostObservation(
        hostname=hostname(),
        boot_id=_read_text(boot_id_path, "kernel boot ID").lower(),
        probe_python=probe_python,
        probe_script=probe_script,
        nvidia_smi=nvidia_smi,
        systemctl=systemctl,
        cpu_memory=collect_cpu_memory_observation(
            binding.cpu_memory,
            proc_root=proc_root,
            sys_cpu_root=sys_cpu_root,
            sys_node_root=sys_node_root,
        ),
        gpus=gpus,
        gpu_processes=gpu_processes,
        conflicting_processes=collect_process_conflicts(
            ignored_pids=tuple(unit.main_pid for unit in opensm),
            proc_root=proc_root,
        ),
        unsafe_profiler_modules=collect_unsafe_profiler_modules(
            binding.policy.unsafe_profiler_kernel_modules,
            proc_modules_path=proc_modules_path,
        ),
        raid_sync_conflicts=collect_raid_sync_conflicts(sys_block_root),
        unused_reserved_ports=probe_reserved_ports_unused(binding.reserved_ports),
        hca=collect_hca_observation(
            binding.hca,
            sys_infiniband_root=sys_infiniband_root,
        ),
        opensm_units=opensm,
    )


def _read_bounded_stdin() -> bytes:
    contents = sys.stdin.buffer.read(MAXIMUM_WIRE_BYTES + 1)
    if len(contents) > MAXIMUM_WIRE_BYTES:
        raise HostGuardError("remote probe request exceeds its size limit")
    return contents


def remote_probe_main() -> int:
    try:
        payload = parse_bounded_canonical_json(_read_bounded_stdin())
        request = RemoteProbeRequest.model_validate_json(
            canonical_host_guard_json(payload)
        )
        now = time.time_ns()
        _validate_request_time(request, now)
        binding = request.peer_binding
        expected_script = Path(binding.remote_probe.script.resolved_path)
        invoked_script = Path(sys.argv[0]).resolve(strict=True)
        if invoked_script != expected_script:
            raise HostGuardError(
                "remote probe was not invoked through the bound script path"
            )
        if Path(sys.executable).resolve(strict=True) != Path(
            binding.remote_probe.python.resolved_path
        ):
            raise HostGuardError(
                "remote probe was not invoked through the bound Python path"
            )
        observation = collect_host_observation(binding)
        validate_idle_peer_observation(observation, binding)
        receipt = build_remote_probe_receipt(
            request,
            observation,
            observed_at_unix_ns=time.time_ns(),
        )
        sys.stdout.buffer.write(canonical_host_guard_json(receipt))
        sys.stdout.buffer.flush()
        return 0
    except (HostGuardError, ValidationError, OSError, ValueError) as error:
        print(
            f"remote host guard failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "remote-probe",
        help="read one canonical request from stdin and emit a phase-neutral receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = cast(str, arguments.command)
    if command == "remote-probe":
        return remote_probe_main()
    raise AssertionError(f"unhandled command {command}")


if __name__ == "__main__":
    raise SystemExit(main())
