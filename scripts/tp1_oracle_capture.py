#!/usr/bin/env python3
"""Capture a deterministic single-GPU model oracle under a benchmark lease.

This is a benchmark-lease child, not a lease acquirer. It runs one local Exo
node, captures only hashes and token metadata from repeated completions, and
stops only the process group whose ownership receipt it created.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import http.client
import importlib.util
import ipaddress
import json
import math
import os
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

MODEL_ID = "mlx-community/SmolLM2-135M-Instruct-8bit"
MODEL_REVISION = "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
MODEL_WEIGHT_BYTES = 142_955_136
LLAMA31_8B_MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-4bit"
LLAMA31_8B_MODEL_REVISION = "90215b22ec18e72f623dde2ea7af4097025160e2"
LLAMA31_8B_MODEL_WEIGHT_BYTES = 4_517_404_672
LLAMA32_3B_MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
LLAMA32_3B_MODEL_REVISION = "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
LLAMA32_3B_MODEL_WEIGHT_BYTES = 1_807_423_488
MODEL_SNAPSHOT_CONTRACTS = {
    MODEL_ID: (MODEL_REVISION, MODEL_WEIGHT_BYTES),
    LLAMA31_8B_MODEL_ID: (
        LLAMA31_8B_MODEL_REVISION,
        LLAMA31_8B_MODEL_WEIGHT_BYTES,
    ),
    LLAMA32_3B_MODEL_ID: (
        LLAMA32_3B_MODEL_REVISION,
        LLAMA32_3B_MODEL_WEIGHT_BYTES,
    ),
}
ORACLE_PROMPT = "Reply with exactly: NCCL proof complete."
ORACLE_MAX_TOKENS = 32
ORACLE_SEED = 42
ORACLE_CHAT_TEMPLATE_DATE = "18 Jul 2026"
DWAGON_HCA_PORT_IDENTITIES = (
    ("mlx4_0", 1, "fe80::10:e000:166:3a19"),
    ("mlx4_0", 2, "fe80::10:e000:166:3a1a"),
)
RUNTIME_METADATA_FILENAME = "runtime-metadata.json"
BENCHMARK_RESULT_FILENAME = "benchmark-result.json"
RESULT_DIRECTORY_FD_ENVIRONMENT = "EXO_BENCHMARK_RESULT_DIRECTORY_FD"
DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
MINIMUM_CLEANUP_GRACE_SECONDS = 300.0
LEASE_HEARTBEAT_MAX_AGE = timedelta(minutes=2)
LEASE_MAX_FUTURE_SKEW = timedelta(minutes=1)
LEASE_METADATA_MAX_AGE = timedelta(minutes=15)
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]+")
_PCI_ADDRESS = re.compile(r"(?:[0-9A-Fa-f]{8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]")


class OracleError(RuntimeError):
    """Expected fail-closed oracle harness error."""


class HttpResponseError(OracleError):
    def __init__(self, status: int, reason: str, body: str) -> None:
        super().__init__(f"HTTP {status} {reason}: {body[:300]}")
        self.status = status


class ManagedSignalError(OracleError):
    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received managed signal {signal_number}")
        self.signal_number = signal_number


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceIdentity(StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_file_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_clean_source(self) -> "SourceIdentity":
        if self.dirty_file_hashes:
            raise ValueError("TP1 oracle source must be clean")
        return self


class GpuIdentity(StrictModel):
    device_uuid: str
    pci_bus_id: str
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "GpuIdentity":
        if _GPU_UUID.fullmatch(self.device_uuid) is None:
            raise ValueError("GPU UUID must be a complete NVIDIA device UUID")
        if _PCI_ADDRESS.fullmatch(self.pci_bus_id) is None:
            raise ValueError("GPU PCI address must be a complete bus address")
        return self

    @property
    def resource_id(self) -> str:
        return f"nvidia-gpu:{self.device_uuid}"


class HcaPort(StrictModel):
    device: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    port: int = Field(ge=1)
    gid: str

    @field_validator("gid")
    @classmethod
    def validate_gid(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if address.version != 6 or address.is_unspecified:
            raise ValueError("HCA GID must be a non-unspecified IPv6 address")
        return str(address)


class ModelSnapshot(StrictModel):
    model_id: str
    revision: str
    local_path: str
    expected_weight_bytes: int = Field(gt=0)
    expected_sha256_manifest: dict[str, str]

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ModelSnapshot":
        expected_contract = MODEL_SNAPSHOT_CONTRACTS.get(self.model_id)
        if expected_contract is None:
            approved_models = ", ".join(sorted(MODEL_SNAPSHOT_CONTRACTS))
            raise ValueError(f"TP1 oracle model_id must be one of: {approved_models}")
        expected_revision, expected_weight_bytes = expected_contract
        if self.revision != expected_revision:
            raise ValueError(
                f"TP1 oracle revision for {self.model_id} must be {expected_revision}"
            )
        if self.expected_weight_bytes != expected_weight_bytes:
            raise ValueError(
                f"TP1 oracle expected_weight_bytes for {self.model_id} must be "
                f"{expected_weight_bytes}"
            )
        path = Path(self.local_path)
        expected_name = f"{self.model_id.replace('/', '--')}--{self.revision}"
        if not path.is_absolute() or path.name != expected_name:
            raise ValueError(
                "model local_path must be an absolute revision-suffixed snapshot"
            )
        if not self.expected_sha256_manifest:
            raise ValueError("expected model SHA-256 manifest must not be empty")
        for relative_name, digest in self.expected_sha256_manifest.items():
            relative = PurePosixPath(relative_name)
            if (
                not relative_name
                or relative.is_absolute()
                or ".." in relative.parts
                or _SHA256.fullmatch(digest) is None
            ):
                raise ValueError("model SHA-256 manifest contains an unsafe entry")
        if ".exo-huggingface-revision.json" not in self.expected_sha256_manifest:
            raise ValueError("model manifest must cover the Exo revision receipt")
        return self


class CpuBinding(StrictModel):
    cpu_set: tuple[int, ...]
    numa_nodes: tuple[int, ...]

    @model_validator(mode="after")
    def validate_binding(self) -> "CpuBinding":
        if (
            not self.cpu_set
            or min(self.cpu_set) < 0
            or tuple(sorted(set(self.cpu_set))) != self.cpu_set
        ):
            raise ValueError(
                "cpu_set must be nonempty, sorted, unique, and nonnegative"
            )
        if (
            not self.numa_nodes
            or min(self.numa_nodes) < 0
            or tuple(sorted(set(self.numa_nodes))) != self.numa_nodes
        ):
            raise ValueError(
                "numa_nodes must be nonempty, sorted, unique, and nonnegative"
            )
        return self


class ServicePorts(StrictModel):
    api: int = Field(ge=1, le=65535)
    zenoh: int = Field(ge=1, le=65535)
    discovery: int = Field(ge=1, le=65535)
    ring: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def require_unique_ports(self) -> "ServicePorts":
        if len(set(self.values)) != len(self.values):
            raise ValueError("API, Zenoh, and discovery ports must be unique")
        return self

    @property
    def values(self) -> tuple[int, int, int, int]:
        return (self.api, self.zenoh, self.discovery, self.ring)


class DeterministicRequest(StrictModel):
    prompt: Literal["Reply with exactly: NCCL proof complete."]
    repetitions: int = Field(ge=3)
    max_tokens: Literal[32]
    seed: Literal[42]
    temperature: float
    stream: Literal[False]
    use_prefix_cache: Literal[False]
    logprobs: Literal[False] = False

    @field_validator("temperature")
    @classmethod
    def require_zero_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("oracle temperature must be exactly zero")
        return value


class Timeouts(StrictModel):
    preflight_seconds: float = Field(gt=0)
    process_start_seconds: float = Field(gt=0)
    api_start_seconds: float = Field(gt=0)
    cluster_seconds: float = Field(gt=0)
    runner_ready_seconds: float = Field(gt=0)
    request_seconds: float = Field(gt=0)
    cleanup_seconds: float = Field(gt=0)
    poll_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def require_finite_timeouts(self) -> "Timeouts":
        values = (
            self.preflight_seconds,
            self.process_start_seconds,
            self.api_start_seconds,
            self.cluster_seconds,
            self.runner_ready_seconds,
            self.request_seconds,
            self.cleanup_seconds,
            self.poll_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("timeouts must be finite")
        return self


class RuntimeRequirements(StrictModel):
    cuda_major: Literal[13]
    minimum_nvidia_driver_version: str
    exo_rs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("minimum_nvidia_driver_version")
    @classmethod
    def validate_driver_version(cls, value: str) -> str:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value) is None:
            raise ValueError("minimum NVIDIA driver must be dotted numeric")
        return value


class OracleConfig(StrictModel):
    schema_version: Literal[1]
    run_id: str
    namespace: str
    result_directory: str
    host_name: Literal["dwagon"]
    source_directory: str
    python_executable: str
    source: SourceIdentity
    model: ModelSnapshot
    gpu: GpuIdentity
    hca_ports: tuple[HcaPort, ...]
    cpu: CpuBinding
    ports: ServicePorts
    environment: dict[str, str]
    launch_argv: tuple[str, ...]
    request: DeterministicRequest
    timeouts: Timeouts
    runtime: RuntimeRequirements

    @model_validator(mode="after")
    def validate_contract(self) -> "OracleConfig":
        if _SAFE_IDENTIFIER.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a safe identifier")
        if _SAFE_IDENTIFIER.fullmatch(self.namespace) is None or self.run_id not in (
            self.namespace
        ):
            raise ValueError("namespace must be safe and contain run_id")
        for raw_path, description in (
            (self.result_directory, "result_directory"),
            (self.source_directory, "source_directory"),
            (self.python_executable, "python_executable"),
        ):
            if not Path(raw_path).is_absolute():
                raise ValueError(f"{description} must be absolute")
        if Path(self.result_directory).name != self.run_id:
            raise ValueError("result_directory must end with the exact run_id")
        if not self.hca_ports or len(
            {(port.device, port.port) for port in self.hca_ports}
        ) != len(self.hca_ports):
            raise ValueError("hca_ports must be nonempty and unique by device and port")
        observed_hca_ports = tuple(
            (port.device, port.port, port.gid) for port in self.hca_ports
        )
        if observed_hca_ports != DWAGON_HCA_PORT_IDENTITIES:
            raise ValueError(
                "hca_ports must exactly match both current dwagon mlx4_0 port GIDs"
            )
        self._validate_environment()
        self._validate_launch()
        return self

    def _validate_environment(self) -> None:
        for name, value in self.environment.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None or "\0" in value:
                raise ValueError(f"invalid environment entry {name!r}")
        required = {
            "CUDA_VISIBLE_DEVICES": self.gpu.device_uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "EXO_MODELS_READ_ONLY_DIRS": str(Path(self.model.local_path).parent),
            "EXO_OFFLINE": "true",
            "EXO_MAX_CONCURRENT_REQUESTS": "1",
            "ENABLE_DISAGGREGATION": "false",
            "EXO_MLX_VISION_LOADING": "disabled",
            "PYTHONHASHSEED": str(self.request.seed),
        }
        if self.model.model_id in {LLAMA31_8B_MODEL_ID, LLAMA32_3B_MODEL_ID}:
            required["EXO_CHAT_TEMPLATE_DATE"] = ORACLE_CHAT_TEMPLATE_DATE
        for name, expected in required.items():
            if self.environment.get(name) != expected:
                raise ValueError(f"environment must set {name}={expected}")
        exo_home = Path(self.environment.get("EXO_HOME", ""))
        if not exo_home.is_absolute() or self.run_id not in str(exo_home):
            raise ValueError("EXO_HOME must be absolute and contain run_id")
        for name in ("PATH", "HOME"):
            value = self.environment.get(name)
            if not value:
                raise ValueError(f"environment must explicitly set {name}")
        if any(
            not Path(entry).is_absolute()
            for entry in self.environment["PATH"].split(":")
            if entry
        ):
            raise ValueError("PATH entries must be absolute")
        if not Path(self.environment["HOME"]).is_absolute():
            raise ValueError("HOME must be absolute")
        library_path = self.environment.get("LD_LIBRARY_PATH")
        if library_path is None or any(
            entry and not Path(entry).is_absolute() for entry in library_path.split(":")
        ):
            raise ValueError("LD_LIBRARY_PATH must be explicit and absolute")

    def _validate_launch(self) -> None:
        expected_prefix = (
            f"--physcpubind={','.join(str(cpu) for cpu in self.cpu.cpu_set)}",
            f"--membind={','.join(str(node) for node in self.cpu.numa_nodes)}",
        )
        arguments = self.launch_argv
        if (
            len(arguments) < 6
            or not Path(arguments[0]).is_absolute()
            or Path(arguments[0]).name != "numactl"
            or tuple(arguments[1:3]) != expected_prefix
            or tuple(arguments[3:6]) != (self.python_executable, "-m", "exo")
        ):
            raise ValueError(
                "launch_argv must use exact numactl bindings and Python -m exo"
            )
        values = {
            flag: _argument_value(arguments, flag)
            for flag in (
                "--namespace",
                "--api-port",
                "--zenoh-port",
                "--discovery-port",
            )
        }
        expected_values = {
            "--namespace": self.namespace,
            "--api-port": str(self.ports.api),
            "--zenoh-port": str(self.ports.zenoh),
            "--discovery-port": str(self.ports.discovery),
        }
        if values != expected_values:
            raise ValueError("launch_argv service identity differs from config")
        for flag in ("--offline", "--no-downloads", "--force-master", "--no-batch"):
            if _boolean_argument_count(arguments, flag) != 1:
                raise ValueError(f"launch_argv must contain exactly one {flag}")
        for forbidden in ("--no-api", "--no-worker", "--legacy-daemon"):
            if _boolean_argument_count(arguments, forbidden):
                raise ValueError(f"launch_argv must not contain {forbidden}")

    @property
    def reserved_ports(self) -> tuple[int, int, int, int]:
        return self.ports.values


class ModelVerification(StrictModel):
    receipt: dict[str, str]
    indexed_weight_bytes: int
    physical_weight_bytes: int
    weight_files: int
    sha256_manifest: dict[str, str]


class RuntimeFacts(StrictModel):
    python_version: str
    exo_version: str
    mlx_version: str
    mlx_cuda_13_version: str
    exo_import_origin: str
    exo_rs_origin: str
    exo_rs_sha256: str
    nvidia_driver_version: str
    cuda_driver_major: int
    nvidia_smi_banner: str


class PreflightReport(StrictModel):
    schema_version: Literal[1]
    run_id: str
    host_name: str
    passed: bool
    conflicts: tuple[str, ...]
    source: SourceIdentity
    full_gpu_inventory: tuple[GpuIdentity, ...]
    selected_gpu: GpuIdentity
    gpu_compute_processes: tuple[str, ...]
    busy_tcp_ports: tuple[int, ...]
    busy_udp_ports: tuple[int, ...]
    online_cpu_ids: tuple[int, ...]
    numa_node_ids: tuple[int, ...]
    storage_conflicts: tuple[str, ...]
    process_conflicts: tuple[str, ...]
    gpu_telemetry_csv: str
    model: ModelVerification
    runtime: RuntimeFacts


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


class StartNodeError(OracleError):
    def __init__(
        self,
        cause: BaseException,
        cleanup: ProcessCleanup,
        receipt: OwnedProcess | None,
    ) -> None:
        super().__init__(
            f"node start failed: {type(cause).__name__}: {cause}; cleanup={cleanup}"
        )
        self.cleanup = cleanup
        self.receipt = receipt


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


@dataclass(frozen=True)
class LeasePreparation:
    metadata: dict[str, object]
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


class LeaseMetadataValidator(Protocol):
    def validate_run_metadata(
        self, metadata: Mapping[str, object], *, now: datetime | None = None
    ) -> dict[str, object]: ...


def _argument_value(arguments: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"argument {flag} has no value")
            values.append(arguments[index + 1])
        elif argument.startswith(flag + "="):
            values.append(argument.partition("=")[2])
    if len(values) > 1:
        raise ValueError(f"arguments repeat {flag}")
    return values[0] if values else None


def _boolean_argument_count(arguments: Sequence[str], flag: str) -> int:
    valued = [argument for argument in arguments if argument.startswith(flag + "=")]
    if valued:
        raise ValueError(f"boolean argument {flag} must not have a value")
    return sum(argument == flag for argument in arguments)


def load_config(path: Path) -> OracleConfig:
    try:
        return OracleConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OracleError(f"cannot read config {path}: {error}") from error
    except ValidationError as error:
        raise OracleError(f"invalid strict TP1 oracle config: {error}") from error


def _object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OracleError(f"{description} must be a JSON object")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise OracleError(f"{description} must have string keys")
    return {cast(str, key): item for key, item in raw.items()}


def _array(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise OracleError(f"{description} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{description} must be a nonempty string")
    return value


def _positive_integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OracleError(f"{description} must be a positive integer")
    return value


def _parse_json_value(text: str, description: str) -> JsonValue:
    try:
        raw = cast(object, json.loads(text))
    except json.JSONDecodeError as error:
        raise OracleError(f"{description} is not valid JSON: {error}") from error
    return _validated_json_value(raw, description)


def _validated_json_value(value: object, description: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OracleError(f"{description} contains a non-finite number")
        return value
    if isinstance(value, list):
        raw_items = cast(list[object], value)
        return [
            _validated_json_value(item, f"{description}[{index}]")
            for index, item in enumerate(raw_items)
        ]
    if isinstance(value, dict):
        raw_items = cast(dict[object, object], value)
        result: JsonObject = {}
        for key, item in raw_items.items():
            if not isinstance(key, str):
                raise OracleError(f"{description} contains a non-string key")
            result[key] = _validated_json_value(item, f"{description}.{key}")
        return result
    raise OracleError(f"{description} contains unsupported JSON data")


def _parse_timestamp(value: object, description: str) -> datetime:
    text = _string(value, description)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise OracleError(f"{description} must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OracleError(f"{description} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def minimum_cleanup_grace_seconds(config: OracleConfig) -> float:
    maximum_checkpoint_delay = max(
        config.timeouts.process_start_seconds,
        config.timeouts.api_start_seconds,
        config.timeouts.cluster_seconds,
        config.timeouts.runner_ready_seconds,
        config.timeouts.request_seconds,
        config.timeouts.poll_seconds,
    )
    delete_bound = 2 * config.timeouts.request_seconds + config.timeouts.cleanup_seconds
    process_stop_bound = 2 * config.timeouts.cleanup_seconds + 10.0
    return max(
        MINIMUM_CLEANUP_GRACE_SECONDS,
        maximum_checkpoint_delay + delete_bound + process_stop_bound + 30.0,
    )


def _canonical_json_sha256(value: object, description: str) -> str:
    validated = _validated_json_value(value, description)
    encoded = json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_config_sha256(config: OracleConfig) -> str:
    return _canonical_json_sha256(
        cast(object, config.model_dump(mode="json")), "full TP1 oracle config"
    )


def expected_manifest_sha256(config: OracleConfig) -> str:
    return _canonical_json_sha256(
        config.model.expected_sha256_manifest, "expected model manifest"
    )


def deterministic_request_contract(config: OracleConfig) -> JsonObject:
    return {
        "body": deterministic_request(config),
        "repetitions": config.request.repetitions,
    }


def deterministic_request_contract_sha256(config: OracleConfig) -> str:
    return _canonical_json_sha256(
        deterministic_request_contract(config),
        "full deterministic TP1 request contract",
    )


def lease_static_metadata(
    config: OracleConfig, command: Sequence[str]
) -> dict[str, object]:
    prompt_sha256 = hashlib.sha256(config.request.prompt.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "command": list(command),
        "git": {
            "commit": config.source.commit,
            "dirty": False,
            "dirty_file_hashes": {},
        },
        "hosts": [config.host_name],
        "models": [
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "paths": {config.host_name: config.model.local_path},
            }
        ],
        "correctness_oracle": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "prompt_sha256": prompt_sha256,
            "capture_mode": "repetition_consensus",
            "expected_content_sha256": None,
        },
        "tp1_oracle_contract": {
            "config_sha256": canonical_config_sha256(config),
            "request": deterministic_request_contract(config),
            "request_sha256": deterministic_request_contract_sha256(config),
            "expected_model_manifest_sha256": expected_manifest_sha256(config),
        },
        "gpu_bindings": {
            config.host_name: [
                {
                    "uuid": config.gpu.device_uuid,
                    "pci_address": config.gpu.pci_bus_id,
                }
            ]
        },
        "cpu_bindings": {
            config.host_name: {
                "cpu_set": ",".join(str(cpu) for cpu in config.cpu.cpu_set),
                "numa_nodes": list(config.cpu.numa_nodes),
                "memory_policy": "bind:"
                + ",".join(str(node) for node in config.cpu.numa_nodes),
            }
        },
        "hca_bindings": {
            config.host_name: [
                {
                    "device": port.device,
                    "port": port.port,
                    "gid": port.gid,
                }
                for port in config.hca_ports
            ]
        },
        "source_deployments": {
            config.host_name: {
                "path": config.source_directory,
                "commit": config.source.commit,
                "dirty_file_hashes": {},
            }
        },
        "execution_contracts": {
            config.host_name: {
                "launch_argv": list(config.launch_argv),
                "probe_environment": {
                    name: value for name, value in sorted(config.environment.items())
                },
            }
        },
        "owner_pids": {config.host_name: []},
    }


def validate_lease_record(
    config: OracleConfig,
    record_value: object,
    *,
    command: Sequence[str],
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    process_id: int,
    parent_process_id: int,
    now: datetime,
    required_lease_id: str | None = None,
) -> str:
    if not all(path.is_absolute() for path in (config_path, lease_path, lock_path)):
        raise OracleError("config, lease, and lock paths must be absolute")
    record = _object(record_value, "active lease record")
    lease_id = _string(record.get("lease_id"), "active lease lease_id")
    if required_lease_id is not None and lease_id != required_lease_id:
        raise OracleError("active lease identity changed")
    if record.get("child_pid") != process_id:
        raise OracleError("active lease belongs to a different child process")
    if record.get("wrapper_pid") != parent_process_id:
        raise OracleError("active lease wrapper is not this process's parent")
    command_values = _array(record.get("command"), "active lease command")
    if not all(isinstance(argument, str) for argument in command_values):
        raise OracleError("active lease command must contain only strings")
    active_command = tuple(cast(str, argument) for argument in command_values)
    if active_command != tuple(command):
        raise OracleError("active lease command differs from this exact child command")
    resolved_harness = Path(__file__).resolve()
    if not any(
        Path(argument).is_absolute() and Path(argument).resolve() == resolved_harness
        for argument in active_command
    ):
        raise OracleError("active lease command does not name this exact harness")
    if _argument_value(active_command, "--config") != str(config_path):
        raise OracleError("active lease command uses a different config path")
    command_lease = _argument_value(active_command, "--lease-path")
    if command_lease is not None and command_lease != str(lease_path):
        raise OracleError("active lease command uses a different lease path")
    if command_lease is None and lease_path != DEFAULT_LEASE_PATH:
        raise OracleError("nonstandard lease path must be explicit in the command")
    command_lock = _argument_value(active_command, "--lock-path")
    if command_lock is not None and command_lock != str(lock_path):
        raise OracleError("active lease command uses a different lock path")
    if command_lock is None and lock_path != DEFAULT_LOCK_PATH:
        raise OracleError("nonstandard lock path must be explicit in the command")
    command_result = _argument_value(active_command, "--result-dir")
    if command_result is not None and command_result != config.result_directory:
        raise OracleError("active lease command uses a different result directory")

    expected_record_values: dict[str, object] = {
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "ports": list(config.reserved_ports),
        "result_directory": config.result_directory,
        "child_cleanup_confirmation_required": True,
    }
    for field_name, expected in expected_record_values.items():
        if record.get(field_name) != expected:
            raise OracleError(f"active lease {field_name} differs from config")
    cleanup_grace = record.get("cleanup_grace_seconds")
    if (
        not isinstance(cleanup_grace, int | float)
        or isinstance(cleanup_grace, bool)
        or not math.isfinite(cleanup_grace)
        or cleanup_grace < minimum_cleanup_grace_seconds(config)
    ):
        raise OracleError("active lease cleanup grace is shorter than cleanup bound")
    heartbeat = _parse_timestamp(record.get("heartbeat"), "active lease heartbeat")
    if heartbeat < now - LEASE_HEARTBEAT_MAX_AGE or heartbeat > (
        now + LEASE_MAX_FUTURE_SKEW
    ):
        raise OracleError("active lease heartbeat is stale or in the future")
    metadata = _object(record.get("metadata"), "active lease metadata")
    generated_at = _parse_timestamp(
        metadata.get("generated_at"), "active lease metadata.generated_at"
    )
    if generated_at < now - LEASE_METADATA_MAX_AGE or generated_at > (
        now + LEASE_MAX_FUTURE_SKEW
    ):
        raise OracleError("active lease metadata is stale or in the future")
    expected_metadata = lease_static_metadata(config, active_command)
    for field_name, expected in expected_metadata.items():
        if metadata.get(field_name) != expected:
            raise OracleError(f"active lease metadata.{field_name} differs from config")
    if record.get("fragment_errors"):
        raise OracleError("active lease reports invalid child result fragments")
    if record.get("manual_clearance_required") is True:
        raise OracleError("active lease already requires manual clearance")
    return lease_id


def read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
        return _object(raw, str(path))
    except json.JSONDecodeError as error:
        raise OracleError(f"invalid JSON in {path}: {error}") from error


def assert_lock_is_held(lock_path: Path) -> None:
    try:
        with lock_path.open("r", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        raise OracleError(
            f"cannot inspect benchmark lock {lock_path}: {error}"
        ) from error
    raise OracleError("benchmark coordination lock is not held")


class LeaseGuard:
    def __init__(
        self,
        config: OracleConfig,
        *,
        command: Sequence[str],
        config_path: Path,
        lease_path: Path,
        lock_path: Path,
        process_id: int,
        parent_process_id: int,
        reader: Callable[[Path], dict[str, object]] = read_json_object,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._command = tuple(command)
        self._config_path = config_path
        self._lease_path = lease_path
        self._lock_path = lock_path
        self._process_id = process_id
        self._parent_process_id = parent_process_id
        self._reader = reader
        self._now = now
        self._monotonic = monotonic
        self._sleep = sleep
        self._lease_id: str | None = None

    def bind(self) -> str:
        assert_lock_is_held(self._lock_path)
        deadline = self._monotonic() + min(
            self._config.timeouts.process_start_seconds, 10.0
        )
        while True:
            record = self._reader(self._lease_path)
            if record.get("child_pid") is not None:
                self._lease_id = self._validate(record=record)
                return self._lease_id
            if self._monotonic() >= deadline:
                raise OracleError("active lease did not publish this child PID")
            self._sleep(0.05)

    def checkpoint(self) -> None:
        if self._lease_id is None:
            raise OracleError("lease guard has not been bound")
        _ = self._validate(required_lease_id=self._lease_id)

    def _validate(
        self,
        required_lease_id: str | None = None,
        *,
        record: dict[str, object] | None = None,
    ) -> str:
        return validate_lease_record(
            self._config,
            self._reader(self._lease_path) if record is None else record,
            command=self._command,
            config_path=self._config_path,
            lease_path=self._lease_path,
            lock_path=self._lock_path,
            process_id=self._process_id,
            parent_process_id=self._parent_process_id,
            now=self._now(),
            required_lease_id=required_lease_id,
        )


def _tagged(value: object, expected_tag: str, description: str) -> dict[str, object]:
    outer = _object(value, description)
    if set(outer) != {expected_tag}:
        raise OracleError(f"{description} must contain only tag {expected_tag}")
    return _object(outer[expected_tag], f"{description}.{expected_tag}")


def _numeric_version(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is None:
        raise OracleError(f"invalid dotted numeric version {value!r}")
    return tuple(int(component) for component in value.split("."))


def _cuda_driver_major_from_nvidia_smi(banner: str) -> int:
    match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+)(?:\.[0-9]+)?", banner)
    if match is None:
        raise OracleError("nvidia-smi did not report a CUDA driver version")
    return int(match.group(1))


def validate_preflight(report: PreflightReport, config: OracleConfig) -> None:
    if report.run_id != config.run_id or report.host_name != config.host_name:
        raise OracleError("preflight identity differs from config")
    if report.source != config.source:
        raise OracleError("preflight source identity differs from config")
    if report.selected_gpu != config.gpu:
        raise OracleError("preflight selected GPU differs from config")
    if sum(gpu == config.gpu for gpu in report.full_gpu_inventory) != 1:
        raise OracleError(
            "full GPU inventory does not contain selected GPU exactly once"
        )
    if set(report.busy_tcp_ports) or set(report.busy_udp_ports):
        raise OracleError("preflight found one or more reserved ports in use")
    if set(report.busy_tcp_ports) - set(config.reserved_ports) or set(
        report.busy_udp_ports
    ) - set(config.reserved_ports):
        raise OracleError("preflight reported an unreserved port")
    if not set(config.cpu.cpu_set).issubset(report.online_cpu_ids):
        raise OracleError("configured CPUs are not all online")
    if not set(config.cpu.numa_nodes).issubset(report.numa_node_ids):
        raise OracleError("configured NUMA nodes are unavailable")
    if (
        report.gpu_compute_processes
        or report.process_conflicts
        or report.storage_conflicts
    ):
        raise OracleError("preflight found GPU, process, or storage conflicts")
    if not report.passed or report.conflicts:
        details = "; ".join(report.conflicts) or "unspecified failure"
        raise OracleError(f"preflight failed: {details}")
    if report.model.receipt != {
        "repo_id": config.model.model_id,
        "revision": config.model.revision,
    }:
        raise OracleError("model revision receipt differs from config")
    if (
        report.model.indexed_weight_bytes != config.model.expected_weight_bytes
        or report.model.physical_weight_bytes <= 0
        or report.model.weight_files < 1
        or report.model.sha256_manifest != config.model.expected_sha256_manifest
    ):
        raise OracleError("exact model manifest verification failed")
    runtime = report.runtime
    if runtime.cuda_driver_major != config.runtime.cuda_major:
        raise OracleError("NVIDIA driver does not advertise required CUDA major")
    if _numeric_version(runtime.nvidia_driver_version) < _numeric_version(
        config.runtime.minimum_nvidia_driver_version
    ):
        raise OracleError("NVIDIA driver is older than configured minimum")
    if runtime.exo_rs_sha256 != config.runtime.exo_rs_sha256:
        raise OracleError("exo_rs runtime artifact differs from config")
    if not runtime.mlx_cuda_13_version:
        raise OracleError("mlx-cuda-13 runtime is not installed")
    source_directory = Path(config.source_directory).resolve()
    import_origin = Path(runtime.exo_import_origin).resolve()
    if not import_origin.is_relative_to(source_directory):
        raise OracleError("Exo import origin is outside configured clean source")


def validate_cluster_inventory(
    resources_value: JsonValue,
    backends_value: JsonValue,
    config: OracleConfig,
) -> str:
    resources_by_node = _object(resources_value, "node compute resources")
    backends_by_node = _object(backends_value, "node backends")
    if len(resources_by_node) != 1 or set(backends_by_node) != set(resources_by_node):
        raise OracleError("TP1 cluster must contain exactly one inventoried node")
    node_id = next(iter(resources_by_node))
    raw_resources = _array(
        resources_by_node[node_id], f"compute resources for node {node_id}"
    )
    if not raw_resources:
        raise OracleError("TP1 Exo node must expose at least one GPU resource")
    observed_gpus: list[GpuIdentity] = []
    for index, raw_resource in enumerate(raw_resources):
        resource = _tagged(
            raw_resource,
            "NvidiaGpuComputeResource",
            f"TP1 compute resource {index}",
        )
        observed_gpus.append(
            GpuIdentity(
                device_uuid=_string(resource.get("deviceUuid"), "GPU UUID"),
                pci_bus_id=_string(resource.get("pciBusId"), "GPU PCI address"),
                model_name=_string(resource.get("modelName"), "GPU model name"),
            )
        )
    observed_resource_ids = [gpu.resource_id for gpu in observed_gpus]
    if len(observed_resource_ids) != len(set(observed_resource_ids)):
        raise OracleError("TP1 live GPU resource IDs must be unique")
    if sum(gpu == config.gpu for gpu in observed_gpus) != 1:
        raise OracleError(
            "TP1 live GPU inventory does not contain the configured selection exactly once"
        )
    node_backends = _array(backends_by_node[node_id], "TP1 node backends")
    if "MlxCuda" not in node_backends:
        raise OracleError("TP1 node does not advertise MlxCuda")
    return node_id


@dataclass(frozen=True)
class PlacementSelection:
    instance: JsonObject
    instance_id: str
    runner_id: str
    runtime_node_id: str
    binding_mode: Literal["explicit_resource"]


def validate_tp1_placement(
    placement_value: JsonValue,
    config: OracleConfig,
    runtime_node_id: str,
    *,
    allow_unbound_template: bool = False,
) -> PlacementSelection:
    placement = _object(placement_value, "TP1 placement")
    instance = _tagged(placement, "MlxRingInstance", "TP1 placement")
    instance_id = _string(instance.get("instanceId"), "TP1 instance ID")
    assignments = _object(instance.get("shardAssignments"), "TP1 shard assignments")
    if assignments.get("modelId") != config.model.model_id:
        raise OracleError("TP1 placement uses the wrong model ID")
    runner_to_shard = _object(assignments.get("runnerToShard"), "runnerToShard")
    node_to_runner = _object(assignments.get("nodeToRunner"), "nodeToRunner")
    if len(runner_to_shard) != 1 or set(node_to_runner) != {runtime_node_id}:
        raise OracleError("TP1 placement must contain one runner on the API node")
    runner_id = next(iter(runner_to_shard))
    if node_to_runner[runtime_node_id] != runner_id:
        raise OracleError("TP1 node representative differs from its only runner")
    shard = _tagged(
        runner_to_shard[runner_id], "PipelineShardMetadata", "TP1 runner shard"
    )
    card = _object(shard.get("modelCard"), "TP1 model card")
    if (
        card.get("modelId") != config.model.model_id
        or card.get("revision") != config.model.revision
    ):
        raise OracleError("TP1 runner does not carry the exact model revision")
    if (
        shard.get("deviceRank") != 0
        or shard.get("worldSize") != 1
        or shard.get("startLayer") != 0
        or shard.get("endLayer") != shard.get("nLayers")
    ):
        raise OracleError("TP1 runner is not a complete one-rank pipeline shard")
    resource_to_runner = _object(
        assignments.get("computeResourceToRunner"), "computeResourceToRunner"
    )
    resource_to_node = _object(
        assignments.get("computeResourceToNode"), "computeResourceToNode"
    )
    expected_resource_to_runner = {config.gpu.resource_id: runner_id}
    expected_resource_to_node = {config.gpu.resource_id: runtime_node_id}
    binding_is_explicit = (
        resource_to_runner == expected_resource_to_runner
        and resource_to_node == expected_resource_to_node
    )
    binding_is_unbound = not resource_to_runner and not resource_to_node
    if not binding_is_explicit and not (allow_unbound_template and binding_is_unbound):
        raise OracleError("TP1 placement contains an unexpected GPU resource binding")

    ephemeral_port = instance.get("ephemeralPort")
    if (
        not isinstance(ephemeral_port, int)
        or isinstance(ephemeral_port, bool)
        or ephemeral_port < 1
        or ephemeral_port > 65535
    ):
        raise OracleError("TP1 placement ephemeralPort must be a valid port")
    hosts_by_node = _object(instance.get("hostsByNode"), "TP1 hostsByNode")
    if set(hosts_by_node) != {runtime_node_id}:
        raise OracleError("TP1 hostsByNode must contain only the runtime node")
    runtime_hosts = _array(
        hosts_by_node[runtime_node_id], "TP1 runtime node ring hosts"
    )
    if len(runtime_hosts) != 1:
        raise OracleError("TP1 runtime node must have exactly one ring host")
    runtime_host = _object(runtime_hosts[0], "TP1 runtime ring host")
    if set(runtime_host) != {"ip", "port"} or runtime_host.get("ip") != "0.0.0.0":
        raise OracleError("TP1 runtime ring host has an unexpected shape")
    if runtime_host.get("port") != ephemeral_port:
        raise OracleError("TP1 runtime ring host port differs from ephemeralPort")

    patched_instance = copy.deepcopy(instance)
    patched_assignments = copy.deepcopy(assignments)
    patched_assignments["computeResourceToRunner"] = expected_resource_to_runner
    patched_assignments["computeResourceToNode"] = expected_resource_to_node
    patched_instance["shardAssignments"] = patched_assignments
    patched_instance["ephemeralPort"] = config.ports.ring
    patched_runtime_host = copy.deepcopy(runtime_host)
    patched_runtime_host["port"] = config.ports.ring
    patched_instance["hostsByNode"] = {runtime_node_id: [patched_runtime_host]}
    patched = {"MlxRingInstance": patched_instance}
    return PlacementSelection(
        instance=cast(
            JsonObject,
            _validated_json_value(patched, "patched TP1 placement"),
        ),
        instance_id=instance_id,
        runner_id=runner_id,
        runtime_node_id=runtime_node_id,
        binding_mode="explicit_resource",
    )


def deterministic_request(config: OracleConfig) -> JsonObject:
    return {
        "model": config.model.model_id,
        "messages": [{"role": "user", "content": config.request.prompt}],
        "max_tokens": config.request.max_tokens,
        "temperature": config.request.temperature,
        "seed": config.request.seed,
        "stream": config.request.stream,
        "use_prefix_cache": config.request.use_prefix_cache,
        "logprobs": config.request.logprobs,
    }


@dataclass(frozen=True)
class CompletionObservation:
    content: str
    content_sha256: str
    content_utf8_bytes: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    response_id: str

    @property
    def consensus_key(self) -> tuple[str, int, int, str]:
        return (
            self.content,
            self.prompt_tokens,
            self.completion_tokens,
            self.finish_reason,
        )

    def result_json(self, iteration: int) -> JsonObject:
        return {
            "iteration": iteration,
            "response_id": self.response_id,
            "content_sha256": self.content_sha256,
            "content_utf8_bytes": self.content_utf8_bytes,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }


def parse_completion(
    response_value: JsonValue, config: OracleConfig
) -> CompletionObservation:
    response = _object(response_value, "oracle completion")
    if response.get("object") != "chat.completion":
        raise OracleError("oracle completion returned the wrong object type")
    if response.get("model") != config.model.model_id:
        raise OracleError("oracle completion returned the wrong model ID")
    response_id = _string(response.get("id"), "oracle completion response ID")
    choices = _array(response.get("choices"), "oracle completion choices")
    if len(choices) != 1:
        raise OracleError("oracle completion must contain exactly one choice")
    choice = _object(choices[0], "oracle completion choice")
    finish_reason = choice.get("finish_reason", choice.get("finishReason"))
    if finish_reason not in {"stop", "length", "tool_calls"}:
        raise OracleError("oracle completion has an invalid finish reason")
    message = _object(choice.get("message"), "oracle completion message")
    content = _string(message.get("content"), "oracle completion content")
    statistics_value = response.get("generation_stats", response.get("generationStats"))
    statistics = _object(statistics_value, "oracle generation statistics")
    prompt_tokens = _positive_integer(
        statistics.get("prompt_tokens", statistics.get("promptTokens")),
        "oracle prompt token count",
    )
    completion_tokens = _positive_integer(
        statistics.get("generation_tokens", statistics.get("generationTokens")),
        "oracle completion token count",
    )
    prefix_cache_hit = statistics.get(
        "prefix_cache_hit", statistics.get("prefixCacheHit", "none")
    )
    if prefix_cache_hit != "none":
        raise OracleError("oracle request unexpectedly used the prefix cache")
    usage_value = response.get("usage")
    if usage_value is not None:
        usage = _object(usage_value, "oracle usage")
        if usage.get("prompt_tokens", usage.get("promptTokens")) != prompt_tokens or (
            usage.get("completion_tokens", usage.get("completionTokens"))
            != completion_tokens
        ):
            raise OracleError("oracle usage and generation statistics disagree")
    encoded = content.encode("utf-8")
    return CompletionObservation(
        content=content,
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        content_utf8_bytes=len(encoded),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=cast(str, finish_reason),
        response_id=response_id,
    )


def cleanup_state_is_clear(
    state_value: JsonValue,
    instance_id: str,
    runner_id: str,
    resource_id: str,
) -> bool:
    state = _object(state_value, "TP1 cleanup state")
    instances = _object(state.get("instances"), "cleanup instances")
    runners = _object(state.get("runners"), "cleanup runners")
    retiring = _object(
        state.get("retiringComputeResources"), "cleanup retiring resources"
    )
    prefill_ports = _object(state.get("prefillServerPorts"), "cleanup prefill ports")
    return (
        not instances
        and instance_id not in instances
        and runner_id not in runners
        and runner_id not in prefill_ports
        and resource_id not in retiring
        and runner_id not in retiring.values()
    )


class OracleEffects(Protocol):
    def checkpoint_lease(self) -> None: ...

    def run_preflight(self, config: OracleConfig) -> PreflightReport: ...

    def start_node(self, config: OracleConfig, owner_token: str) -> OwnedProcess: ...

    def process_alive(self, process: OwnedProcess) -> bool: ...

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup: ...

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


def _checkpoint(
    effects: OracleEffects,
    latch: SignalLatch,
    process: OwnedProcess | None = None,
) -> None:
    effects.checkpoint_lease()
    latch.checkpoint()
    if process is not None and not effects.process_alive(process):
        raise OracleError("owned Exo process exited unexpectedly")


def _wait_until(
    effects: OracleEffects,
    latch: SignalLatch,
    process: OwnedProcess,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    description: str,
    check: Callable[[], bool],
) -> None:
    deadline = effects.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while effects.monotonic() < deadline:
        _checkpoint(effects, latch, process)
        try:
            if check():
                return
            last_error = None
        except (HttpResponseError, ConnectionError, OSError) as error:
            last_error = error
        effects.sleep(poll_seconds)
    suffix = f": last error was {last_error}" if last_error is not None else ""
    raise OracleError(f"timed out waiting for {description}{suffix}")


def wait_for_api(
    effects: OracleEffects,
    config: OracleConfig,
    latch: SignalLatch,
    process: OwnedProcess,
) -> str:
    node_id: str | None = None

    def ready() -> bool:
        nonlocal node_id
        value = effects.request_json("GET", "/node_id")
        if isinstance(value, str) and value:
            node_id = value
            return True
        return False

    _wait_until(
        effects,
        latch,
        process,
        timeout_seconds=config.timeouts.api_start_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="owned Exo API",
        check=ready,
    )
    assert node_id is not None
    return node_id


def wait_for_cluster(
    effects: OracleEffects,
    config: OracleConfig,
    latch: SignalLatch,
    process: OwnedProcess,
) -> str:
    node_id: str | None = None

    def ready() -> bool:
        nonlocal node_id
        resources = effects.request_json("GET", "/state/nodeComputeResources")
        backends = effects.request_json("GET", "/state/nodeBackends")
        try:
            node_id = validate_cluster_inventory(resources, backends, config)
        except OracleError:
            return False
        return True

    _wait_until(
        effects,
        latch,
        process,
        timeout_seconds=config.timeouts.cluster_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="one-node one-GPU MlxCuda cluster",
        check=ready,
    )
    assert node_id is not None
    return node_id


def _get_optional_state(effects: OracleEffects, path: str) -> JsonValue | None:
    try:
        return effects.request_json("GET", path)
    except HttpResponseError as error:
        if error.status == 404:
            return None
        raise


def wait_for_runner_ready(
    effects: OracleEffects,
    config: OracleConfig,
    latch: SignalLatch,
    process: OwnedProcess,
    selection: PlacementSelection,
) -> None:
    def ready() -> bool:
        instance_value = _get_optional_state(
            effects, f"/state/instances/{selection.instance_id}"
        )
        if instance_value is None:
            return False
        observed_selection = validate_tp1_placement(
            instance_value, config, selection.runtime_node_id
        )
        if observed_selection != selection:
            raise OracleError("owned TP1 placement changed after submission")
        status_value = _get_optional_state(
            effects, f"/state/runners/{selection.runner_id}"
        )
        if status_value is None:
            return False
        status = _object(status_value, "owned TP1 runner status")
        if set(status) == {"RunnerFailed"}:
            failure = _object(status["RunnerFailed"], "owned runner failure")
            raise OracleError(f"owned TP1 runner failed: {failure.get('errorMessage')}")
        return set(status) == {"RunnerReady"}

    _wait_until(
        effects,
        latch,
        process,
        timeout_seconds=config.timeouts.runner_ready_seconds,
        poll_seconds=config.timeouts.poll_seconds,
        description="owned TP1 runner to reach RunnerReady",
        check=ready,
    )


def delete_and_verify_instance(
    effects: OracleEffects,
    config: OracleConfig,
    process: OwnedProcess,
    selection: PlacementSelection,
) -> None:
    try:
        effects.request_json("DELETE", f"/instance/{selection.instance_id}")
    except HttpResponseError as error:
        if error.status != 404:
            raise
    deadline = effects.monotonic() + config.timeouts.cleanup_seconds
    while effects.monotonic() < deadline:
        if not effects.process_alive(process):
            raise OracleError(
                "owned Exo process exited before API cleanup confirmation"
            )
        state = effects.request_json("GET", "/state")
        if cleanup_state_is_clear(
            state,
            selection.instance_id,
            selection.runner_id,
            config.gpu.resource_id,
        ):
            return
        effects.sleep(config.timeouts.poll_seconds)
    raise OracleError("timed out verifying TP1 runner and resource lease cleanup")


def _owned_process_json(process: OwnedProcess) -> JsonObject:
    value = _validated_json_value(asdict(process), "owned process receipt")
    return cast(JsonObject, value)


def _runtime_metadata(
    config: OracleConfig,
    owner_token: str,
    processes: Sequence[OwnedProcess],
) -> JsonObject:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "namespace": config.namespace,
        "owner_token": owner_token,
        "owned_processes": [_owned_process_json(process) for process in processes],
    }


def run_oracle_capture(
    config: OracleConfig,
    effects: OracleEffects,
    signal_latch: SignalLatch | None = None,
) -> JsonObject:
    """Run one capture and always attempt an ownership-bound result fragment."""
    latch = signal_latch or SignalLatch()
    owner_token = f"{config.run_id}:{uuid.uuid4().hex}"
    process: OwnedProcess | None = None
    all_receipts: list[OwnedProcess] = []
    runtime_metadata_persisted = False
    preflight: PreflightReport | None = None
    preflight_complete = False
    selection: PlacementSelection | None = None
    placement_submitted = False
    instance_cleanup_succeeded = True
    process_cleanups: list[ProcessCleanup] = []
    expected_process_cleanup_count = 0
    completion_summaries: list[JsonObject] = []
    oracle: JsonObject | None = None
    capture_complete = False
    caught_error: BaseException | None = None
    started_at = time.time()

    try:
        _checkpoint(effects, latch)
        preflight = effects.run_preflight(config)
        validate_preflight(preflight, config)
        preflight_complete = True
        _checkpoint(effects, latch)

        process = effects.start_node(config, owner_token)
        expected_process_cleanup_count += 1
        all_receipts.append(process)
        effects.write_result_json(
            RUNTIME_METADATA_FILENAME,
            _runtime_metadata(config, owner_token, all_receipts),
        )
        runtime_metadata_persisted = True
        _checkpoint(effects, latch, process)

        api_node_id = wait_for_api(effects, config, latch, process)
        runtime_node_id = wait_for_cluster(effects, config, latch, process)
        if api_node_id != runtime_node_id:
            raise OracleError("API node differs from the exact GPU inventory node")

        placement_value = effects.request_json(
            "GET",
            "/instance/placement",
            params={
                "model_id": config.model.model_id,
                "sharding": "Pipeline",
                "instance_meta": "MlxRing",
                "min_nodes": "1",
                "use_all_compute_resources": "false",
            },
        )
        selection = validate_tp1_placement(
            placement_value,
            config,
            runtime_node_id,
            allow_unbound_template=True,
        )
        if (
            _get_optional_state(effects, f"/state/instances/{selection.instance_id}")
            is not None
        ):
            raise OracleError("proposed TP1 instance ID is already in use")
        placement_submitted = True
        effects.request_json(
            "POST",
            "/instance",
            body={"instance": cast(JsonValue, selection.instance)},
        )
        wait_for_runner_ready(effects, config, latch, process, selection)

        request = deterministic_request(config)
        observations: list[CompletionObservation] = []
        for iteration in range(config.request.repetitions):
            _checkpoint(effects, latch, process)
            request_started = effects.monotonic()
            response = effects.request_json(
                "POST", "/bench/chat/completions", body=request
            )
            elapsed_seconds = effects.monotonic() - request_started
            if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
                raise OracleError("oracle completion elapsed time must be positive")
            observation = parse_completion(response, config)
            observations.append(observation)
            summary = observation.result_json(iteration)
            summary["elapsed_seconds"] = elapsed_seconds
            completion_summaries.append(summary)
            _checkpoint(effects, latch, process)
        first = observations[0]
        if any(
            observation.consensus_key != first.consensus_key
            for observation in observations[1:]
        ):
            raise OracleError(
                "repeated deterministic completions differ in content, token counts, "
                "or finish reason"
            )
        oracle = {
            "schema_version": 1,
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "prompt_sha256": hashlib.sha256(
                config.request.prompt.encode("utf-8")
            ).hexdigest(),
            "content_sha256": first.content_sha256,
            "content_utf8_bytes": first.content_utf8_bytes,
            "prompt_tokens": first.prompt_tokens,
            "completion_tokens": first.completion_tokens,
            "finish_reason": first.finish_reason,
            "repetitions": config.request.repetitions,
        }
        capture_complete = True
    except StartNodeError as error:
        caught_error = error
        expected_process_cleanup_count += 1
        process_cleanups.append(error.cleanup)
        if error.receipt is not None:
            all_receipts.append(error.receipt)
            try:
                effects.write_result_json(
                    RUNTIME_METADATA_FILENAME,
                    _runtime_metadata(config, owner_token, all_receipts),
                )
                runtime_metadata_persisted = True
            except BaseException as metadata_error:
                caught_error = OracleError(
                    "start failed and its ownership receipt could not be persisted: "
                    f"{type(metadata_error).__name__}: {metadata_error}"
                )
    except BaseException as error:
        caught_error = error
    finally:
        latch.begin_cleanup()
        if placement_submitted and selection is not None and process is not None:
            try:
                delete_and_verify_instance(effects, config, process, selection)
            except BaseException as error:
                instance_cleanup_succeeded = False
                if caught_error is None:
                    caught_error = error
        if process is not None:
            try:
                process_cleanups.append(
                    effects.stop_node(process, config.timeouts.cleanup_seconds)
                )
            except BaseException as error:
                process_cleanups.append(
                    ProcessCleanup(
                        host_name=config.host_name,
                        ownership_verified=False,
                        terminated=False,
                        forced=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if caught_error is None:
                    caught_error = error

    process_cleanup_succeeded = (
        len(process_cleanups) == expected_process_cleanup_count
        and all(
            cleanup.ownership_verified and cleanup.terminated
            for cleanup in process_cleanups
        )
        and (not all_receipts or runtime_metadata_persisted)
    )
    cleanup_succeeded = instance_cleanup_succeeded and process_cleanup_succeeded
    if not cleanup_succeeded and caught_error is None:
        caught_error = OracleError("owned TP1 cleanup was not fully confirmed")
    if not preflight_complete:
        status = "preflight_failed"
    elif not cleanup_succeeded:
        status = "cleanup_failed"
    elif not capture_complete or caught_error is not None:
        status = "capture_failed"
    else:
        status = "completed"

    result_value = _validated_json_value(
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "namespace": config.namespace,
            "status": status,
            "reportable": status == "completed",
            "result_scope": "deterministic_tp1_oracle_capture",
            "performance_comparable": False,
            "completed_normally": capture_complete and caught_error is None,
            "cleanup_succeeded": cleanup_succeeded,
            "interrupted_signal": latch.signal_number,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": time.time(),
            "error": (
                None
                if caught_error is None
                else f"{type(caught_error).__name__}: {caught_error}"
            ),
            "model": config.model.model_dump(mode="json"),
            "source": config.source.model_dump(mode="json"),
            "model_path": config.model.local_path,
            "selected_gpu": config.gpu.model_dump(mode="json"),
            "cpu_binding": config.cpu.model_dump(mode="json"),
            "reserved_ports": list(config.reserved_ports),
            "launch_argv": list(config.launch_argv),
            "environment": {
                **config.environment,
                "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            },
            "preflight": (
                None if preflight is None else preflight.model_dump(mode="json")
            ),
            "request": deterministic_request(config),
            "tp1_oracle_contract": {
                "config_sha256": canonical_config_sha256(config),
                "request_sha256": deterministic_request_contract_sha256(config),
                "expected_model_manifest_sha256": expected_manifest_sha256(config),
            },
            "oracle": oracle,
            "completion_summaries": completion_summaries,
            "placement": None if selection is None else selection.instance,
            "owned_instance_id": (None if selection is None else selection.instance_id),
            "owned_runner_ids": ([] if selection is None else [selection.runner_id]),
            "owned_compute_resource_ids": [config.gpu.resource_id],
            "resource_binding_mode": (
                None if selection is None else selection.binding_mode
            ),
            "api_explicit_resource_binding": (
                selection is not None and selection.binding_mode == "explicit_resource"
            ),
            "owned_processes": [
                _owned_process_json(receipt) for receipt in all_receipts
            ],
            "instance_cleanup_succeeded": instance_cleanup_succeeded,
            "process_cleanup": [asdict(cleanup) for cleanup in process_cleanups],
        },
        "TP1 benchmark result",
    )
    result = cast(JsonObject, result_value)
    effects.write_result_json(BENCHMARK_RESULT_FILENAME, result)
    return result


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
                raise OracleError(f"invalid descending ID range {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return tuple(sorted(result))


def validate_result_directory_descriptor(path: Path, descriptor: int) -> None:
    try:
        expected = os.fstat(descriptor)
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise OracleError(
            f"cannot validate trusted result directory: {error}"
        ) from error
    if not stat.S_ISDIR(expected.st_mode):
        raise OracleError("trusted result descriptor is not a directory")
    if not stat.S_ISDIR(observed.st_mode) or (
        observed.st_dev,
        observed.st_ino,
    ) != (expected.st_dev, expected.st_ino):
        raise OracleError("trusted result directory path identity changed")


def inherited_result_directory_descriptor(path: Path) -> int:
    raw_descriptor = os.environ.get(RESULT_DIRECTORY_FD_ENVIRONMENT)
    if (
        raw_descriptor is None
        or not raw_descriptor.isascii()
        or not raw_descriptor.isdigit()
    ):
        raise OracleError(
            f"lease wrapper did not pass {RESULT_DIRECTORY_FD_ENVIRONMENT}"
        )
    descriptor = int(raw_descriptor)
    validate_result_directory_descriptor(path, descriptor)
    return descriptor


_RUNTIME_PROBE_PROGRAM = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import exo
import mlx

exo_spec = importlib.util.find_spec("exo")
exo_rs_spec = importlib.util.find_spec("exo_rs")
if exo_spec is None or exo_spec.origin is None:
    raise RuntimeError("cannot resolve exo import origin")
if exo_rs_spec is None or exo_rs_spec.origin is None:
    raise RuntimeError("cannot resolve exo_rs import origin")
exo_rs_path = Path(exo_rs_spec.origin).resolve()
digest = hashlib.sha256()
with exo_rs_path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)

def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""

print(json.dumps({
    "python_version": sys.version,
    "exo_version": str(getattr(exo, "__version__", distribution_version("exo"))),
    "mlx_version": str(getattr(mlx, "__version__", distribution_version("mlx"))),
    "mlx_cuda_13_version": distribution_version("mlx-cuda-13"),
    "exo_import_origin": str(Path(exo_spec.origin).resolve()),
    "exo_rs_origin": str(exo_rs_path),
    "exo_rs_sha256": digest.hexdigest(),
}, sort_keys=True))
"""


@dataclass
class _RunningHandle:
    process: subprocess.Popen[bytes]
    log_file: IO[bytes]


class SystemEffects:
    """Local Linux effects. Every mutation is scoped to the leased run."""

    def __init__(
        self,
        config: OracleConfig,
        guard: LeaseGuard,
        result_directory_descriptor: int,
    ) -> None:
        self._config = config
        self._guard = guard
        self._result_directory = Path(config.result_directory)
        self._result_directory_descriptor = result_directory_descriptor
        validate_result_directory_descriptor(
            self._result_directory, self._result_directory_descriptor
        )
        self._running: dict[int, _RunningHandle] = {}

    def checkpoint_lease(self) -> None:
        self._guard.checkpoint()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _command(
        self,
        arguments: Sequence[str],
        *,
        working_directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        completed = subprocess.run(
            arguments,
            cwd=working_directory,
            env=self._config.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout or self._config.timeouts.preflight_seconds,
        )
        if completed.returncode != 0:
            raise OracleError(
                f"probe command {arguments[0]} failed with {completed.returncode}: "
                f"{completed.stderr[-1000:]}"
            )
        return completed.stdout.strip()

    def _source_identity(self) -> SourceIdentity:
        source = self._config.source_directory
        commit = self._command(("git", "-C", source, "rev-parse", "HEAD"))
        tracked = self._command(
            ("git", "-C", source, "diff", "--name-only", "-z", "HEAD")
        )
        untracked = self._command(
            (
                "git",
                "-C",
                source,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
        )
        paths = sorted(
            path for path in set((*tracked.split("\0"), *untracked.split("\0"))) if path
        )
        root = Path(source)
        deleted_digest = hashlib.sha256(b"<deleted>").hexdigest()
        hashes = {
            relative_path: (
                self._sha256_file(root / relative_path)
                if (root / relative_path).is_file()
                else deleted_digest
            )
            for relative_path in paths
        }
        try:
            return SourceIdentity(commit=commit, dirty_file_hashes=hashes)
        except ValidationError as error:
            raise OracleError(
                f"source identity is not clean and exact: {error}"
            ) from error

    def _gpu_inventory(self) -> tuple[GpuIdentity, ...]:
        output = self._command(
            (
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name",
                "--format=csv,noheader",
            )
        )
        rows = [
            row
            for row in csv.reader(output.splitlines())
            if any(field.strip() for field in row)
        ]
        if not rows:
            raise OracleError("nvidia-smi returned an empty GPU inventory")
        malformed_rows = [
            index for index, row in enumerate(rows, start=1) if len(row) != 3
        ]
        if malformed_rows:
            raise OracleError(
                "nvidia-smi returned malformed GPU inventory rows: "
                + ", ".join(str(index) for index in malformed_rows)
            )
        try:
            return tuple(
                GpuIdentity(
                    device_uuid=row[0].strip(),
                    pci_bus_id=row[1].strip(),
                    model_name=row[2].strip(),
                )
                for row in rows
            )
        except ValidationError as error:
            raise OracleError(
                f"nvidia-smi returned invalid GPU inventory: {error}"
            ) from error

    def _gpu_compute_processes(self) -> tuple[str, ...]:
        output = self._command(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ),
        )
        return tuple(
            line.strip()
            for line in output.splitlines()
            if line.strip() and "No running processes" not in line
        )

    @staticmethod
    def _busy_ports(
        ports: Sequence[int], socket_type: Literal["tcp", "udp"]
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

    @staticmethod
    def _ancestor_pids() -> set[int]:
        ancestors: set[int] = set()
        process_id = os.getpid()
        while process_id > 1 and process_id not in ancestors:
            ancestors.add(process_id)
            try:
                fields = (
                    Path(f"/proc/{process_id}/stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()
                )
                process_id = int(fields[1])
            except (OSError, IndexError, ValueError):
                break
        return ancestors

    def _process_conflicts(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        ignored = self._ancestor_pids()
        inference_conflicts: list[str] = []
        storage_conflicts: list[str] = []
        inference_names = {
            "exo",
            "uvicorn",
            "all_reduce_perf",
            "ib_read_bw",
            "ib_write_bw",
            "ib_send_bw",
            "ib_read_lat",
            "ib_write_lat",
            "ib_send_lat",
        }
        storage_names = {
            "aria2c",
            "b3sum",
            "hf",
            "huggingface-cli",
            "md5sum",
            "rsync",
            "sha256sum",
        }
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) in ignored:
                continue
            try:
                arguments = [
                    part.decode("utf-8", errors="replace")
                    for part in (entry / "cmdline").read_bytes().split(b"\0")
                    if part
                ]
            except OSError:
                continue
            if not arguments:
                continue
            executable = Path(arguments[0]).name
            command_line = " ".join(arguments)
            python_exo = any(
                argument == "-m"
                and index + 1 < len(arguments)
                and arguments[index + 1].split(".", 1)[0] == "exo"
                for index, argument in enumerate(arguments)
            )
            if (
                executable in inference_names
                or python_exo
                or "sglang.launch_server" in command_line
                or "vllm.entrypoints" in command_line
                or "mlx_nccl_smoke" in command_line
                or "two_host_mlx_nccl_poc" in command_line
            ):
                inference_conflicts.append(
                    f"pid={entry.name} command={command_line[:500]}"
                )
            if executable in storage_names or "hf download" in command_line:
                storage_conflicts.append(
                    f"pid={entry.name} command={command_line[:500]}"
                )
        for path in Path("/sys/block").glob("md*/md/sync_action"):
            try:
                action = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if action not in {"idle", "frozen"}:
                storage_conflicts.append(f"{path}: {action}")
        return tuple(sorted(inference_conflicts)), tuple(sorted(storage_conflicts))

    def _verify_model(self) -> ModelVerification:
        path = Path(self._config.model.local_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or path.resolve() != path
            or not path.is_dir()
        ):
            raise OracleError("model snapshot path is not a canonical directory")
        entries = sorted(path.rglob("*"))
        for entry in entries:
            if entry.is_symlink():
                raise OracleError("model snapshot contains a symlink")
            if not entry.is_file() and not entry.is_dir():
                raise OracleError("model snapshot contains a special filesystem entry")
        receipt_path = path / ".exo-huggingface-revision.json"
        raw_receipt = cast(object, json.loads(receipt_path.read_text(encoding="utf-8")))
        receipt = _object(
            raw_receipt,
            "model revision receipt",
        )
        if set(receipt) != {"repo_id", "revision"}:
            raise OracleError("model revision receipt has an invalid schema")
        expected_receipt = {
            "repo_id": self._config.model.model_id,
            "revision": self._config.model.revision,
        }
        if receipt != expected_receipt:
            raise OracleError("model revision receipt differs from config")
        index_path = path / "model.safetensors.index.json"
        raw_index = cast(object, json.loads(index_path.read_text(encoding="utf-8")))
        index = _object(
            raw_index,
            "safetensors index",
        )
        metadata = _object(index.get("metadata"), "safetensors index metadata")
        weight_map = _object(index.get("weight_map"), "safetensors weight map")
        indexed_weight_bytes = _positive_integer(
            metadata.get("total_size"), "indexed model weight bytes"
        )
        raw_shard_names = list(weight_map.values())
        if not raw_shard_names or not all(
            isinstance(name, str) for name in raw_shard_names
        ):
            raise OracleError("safetensors index has no valid shard targets")
        shard_names = sorted({cast(str, name) for name in raw_shard_names})
        weight_paths: list[Path] = []
        for name in shard_names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise OracleError(f"unsafe safetensors shard path {name}")
            shard_path = path.joinpath(*relative.parts)
            if not shard_path.is_file() or shard_path.is_symlink():
                raise OracleError(f"referenced safetensors shard is missing: {name}")
            weight_paths.append(shard_path)
        indexed_paths = {weight.resolve() for weight in weight_paths}
        unexpected_weights = [
            str(entry.relative_to(path))
            for entry in entries
            if entry.is_file()
            and entry.suffix == ".safetensors"
            and entry.resolve() not in indexed_paths
        ]
        if unexpected_weights:
            raise OracleError(
                f"unindexed safetensors files are present: {unexpected_weights}"
            )
        files = [entry for entry in entries if entry.is_file()]
        manifest = {
            entry.relative_to(path).as_posix(): self._sha256_file(entry)
            for entry in files
        }
        return ModelVerification(
            receipt=cast(dict[str, str], receipt),
            indexed_weight_bytes=indexed_weight_bytes,
            physical_weight_bytes=sum(weight.stat().st_size for weight in weight_paths),
            weight_files=len(weight_paths),
            sha256_manifest=manifest,
        )

    def _runtime_facts(self) -> RuntimeFacts:
        raw_runtime = self._command(
            (self._config.python_executable, "-c", _RUNTIME_PROBE_PROGRAM),
            working_directory=self._config.source_directory,
        )
        parsed_runtime = cast(object, json.loads(raw_runtime))
        runtime = _object(parsed_runtime, "configured Python runtime probe")
        banner = self._command(("nvidia-smi",))
        cuda_driver_major = _cuda_driver_major_from_nvidia_smi(banner)
        driver_rows = self._command(
            (
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader",
            )
        )
        selected_driver: str | None = None
        for row in csv.reader(driver_rows.splitlines()):
            if len(row) == 2 and row[0].strip() == self._config.gpu.device_uuid:
                selected_driver = row[1].strip()
        if selected_driver is None:
            raise OracleError("selected GPU has no NVIDIA driver observation")
        return RuntimeFacts(
            python_version=_string(runtime.get("python_version"), "Python version"),
            exo_version=_string(runtime.get("exo_version"), "Exo version"),
            mlx_version=_string(runtime.get("mlx_version"), "MLX version"),
            mlx_cuda_13_version=_string(
                runtime.get("mlx_cuda_13_version"), "mlx-cuda-13 version"
            ),
            exo_import_origin=_string(
                runtime.get("exo_import_origin"), "Exo import origin"
            ),
            exo_rs_origin=_string(runtime.get("exo_rs_origin"), "exo_rs origin"),
            exo_rs_sha256=_string(runtime.get("exo_rs_sha256"), "exo_rs SHA-256"),
            nvidia_driver_version=selected_driver,
            cuda_driver_major=cuda_driver_major,
            nvidia_smi_banner=banner,
        )

    def run_preflight(self, config: OracleConfig) -> PreflightReport:
        if config != self._config:
            raise OracleError("preflight config identity changed")
        started = time.monotonic()
        source = self._source_identity()
        inventory = self._gpu_inventory()
        compute_processes = self._gpu_compute_processes()
        busy_tcp = self._busy_ports(config.reserved_ports, "tcp")
        busy_udp = self._busy_ports(config.reserved_ports, "udp")
        process_conflicts, storage_conflicts = self._process_conflicts()
        online_cpu_ids = _parse_id_ranges(
            Path("/sys/devices/system/cpu/online").read_text(encoding="utf-8")
        )
        numa_node_ids = tuple(
            sorted(
                int(path.name.removeprefix("node"))
                for path in Path("/sys/devices/system/node").glob("node[0-9]*")
            )
        )
        telemetry = self._command(
            (
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used,"
                "memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw,"
                "power.limit,driver_version",
                "--format=csv,noheader,nounits",
            )
        )
        model = self._verify_model()
        runtime = self._runtime_facts()
        conflicts: list[str] = []
        if source != config.source:
            conflicts.append("source identity differs from clean config")
        if sum(gpu == config.gpu for gpu in inventory) != 1:
            conflicts.append("selected GPU identity is absent or ambiguous")
        if compute_processes:
            conflicts.append("one or more GPU compute processes are active")
        if busy_tcp or busy_udp:
            conflicts.append("one or more reserved ports are busy")
        if process_conflicts:
            conflicts.append("an unowned inference or benchmark process is active")
        if storage_conflicts:
            conflicts.append("model I/O or storage maintenance is active")
        if not set(config.cpu.cpu_set).issubset(online_cpu_ids):
            conflicts.append("configured CPUs are not all online")
        if not set(config.cpu.numa_nodes).issubset(numa_node_ids):
            conflicts.append("configured NUMA nodes are unavailable")
        if time.monotonic() - started > config.timeouts.preflight_seconds:
            conflicts.append("preflight exceeded its configured timeout")
        return PreflightReport(
            schema_version=1,
            run_id=config.run_id,
            host_name=config.host_name,
            passed=not conflicts,
            conflicts=tuple(conflicts),
            source=source,
            full_gpu_inventory=inventory,
            selected_gpu=config.gpu,
            gpu_compute_processes=compute_processes,
            busy_tcp_ports=busy_tcp,
            busy_udp_ports=busy_udp,
            online_cpu_ids=online_cpu_ids,
            numa_node_ids=numa_node_ids,
            storage_conflicts=storage_conflicts,
            process_conflicts=process_conflicts,
            gpu_telemetry_csv=telemetry,
            model=model,
            runtime=runtime,
        )

    @staticmethod
    def _read_process_identity(process_id: int) -> tuple[int, int]:
        fields = (
            Path(f"/proc/{process_id}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
        return int(fields[2]), int(fields[19])

    @staticmethod
    def _group_ownership(process: OwnedProcess) -> tuple[bool, tuple[int, ...]]:
        members: list[int] = []
        owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={process.owner_token}".encode()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (
                    (entry / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()
                )
                state = fields[0]
                process_group_id = int(fields[2])
                start_ticks = int(fields[19])
            except (OSError, IndexError, ValueError):
                continue
            if process_group_id != process.process_group_id:
                continue
            member_id = int(entry.name)
            members.append(member_id)
            if member_id == process.pid and start_ticks != process.start_time_ticks:
                return False, tuple(sorted(members))
            if state == "Z":
                continue
            try:
                environment = (entry / "environ").read_bytes().split(b"\0")
            except OSError:
                return False, tuple(sorted(members))
            if owner_entry not in environment:
                return False, tuple(sorted(members))
            if member_id == process.pid:
                try:
                    command_line = (entry / "cmdline").read_bytes()
                except OSError:
                    return False, tuple(sorted(members))
                if process.namespace.encode() not in command_line:
                    return False, tuple(sorted(members))
        return True, tuple(sorted(members))

    def start_node(self, config: OracleConfig, owner_token: str) -> OwnedProcess:
        if config != self._config or self._running:
            raise OracleError("local Exo node is already started or config changed")
        log_path = self._result_directory / "exo-tp1.log"
        log_descriptor = os.open(
            log_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._result_directory_descriptor,
        )
        log_file = os.fdopen(log_descriptor, mode="wb")
        environment = dict(config.environment)
        environment["EXO_BENCHMARK_OWNER_TOKEN"] = owner_token
        process: subprocess.Popen[bytes] | None = None
        receipt: OwnedProcess | None = None
        try:
            launch_record = (
                "EXO_TP1_LAUNCH "
                + json.dumps(
                    {
                        "environment": environment,
                        "argv": list(config.launch_argv),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log_file.write(launch_record.encode("utf-8"))
            log_file.flush()
            process = subprocess.Popen(
                config.launch_argv,
                cwd=config.source_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._running[process.pid] = _RunningHandle(process, log_file)
            process_group_id, start_ticks = self._read_process_identity(process.pid)
            if process_group_id != process.pid:
                raise OracleError("owned Exo node is not its process-group leader")
            receipt = OwnedProcess(
                host_name=config.host_name,
                pid=process.pid,
                process_group_id=process_group_id,
                start_time_ticks=start_ticks,
                owner_token=owner_token,
                namespace=config.namespace,
                transport_pid=process.pid,
                log_path=str(log_path),
            )
            if process.poll() is not None:
                raise OracleError("owned Exo node exited during process start")
            ownership_matches, members = self._group_ownership(receipt)
            if not ownership_matches or process.pid not in members:
                raise OracleError("owned Exo process-group receipt cannot be verified")
            return receipt
        except BaseException as error:
            if receipt is None and process is not None and process.poll() is None:
                with contextlib.suppress(OSError, IndexError, ValueError):
                    process_group_id, start_ticks = self._read_process_identity(
                        process.pid
                    )
                    if process_group_id == process.pid:
                        receipt = OwnedProcess(
                            host_name=config.host_name,
                            pid=process.pid,
                            process_group_id=process_group_id,
                            start_time_ticks=start_ticks,
                            owner_token=owner_token,
                            namespace=config.namespace,
                            transport_pid=process.pid,
                            log_path=str(log_path),
                        )
            if receipt is not None:
                cleanup = self.stop_node(receipt, config.timeouts.cleanup_seconds)
            elif process is not None and process.poll() is not None:
                process.wait(timeout=1.0)
                self._running.pop(process.pid, None)
                log_file.close()
                cleanup = ProcessCleanup(config.host_name, True, True, False)
            else:
                cleanup = ProcessCleanup(
                    config.host_name,
                    False,
                    False,
                    False,
                    "process started without a verifiable ownership receipt",
                )
            raise StartNodeError(error, cleanup, receipt) from error

    def process_alive(self, process: OwnedProcess) -> bool:
        handle = self._running.get(process.pid)
        if handle is None or handle.process.poll() is not None:
            return False
        ownership_matches, members = self._group_ownership(process)
        return ownership_matches and bool(members)

    def _wait_group_gone(
        self,
        process: OwnedProcess,
        handle: _RunningHandle,
        timeout_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            _ = handle.process.poll()
            ownership_matches, members = self._group_ownership(process)
            if not members:
                remaining = max(0.0, deadline - time.monotonic())
                leader_wait_timed_out = False
                try:
                    _ = handle.process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    leader_wait_timed_out = True
                if not leader_wait_timed_out and handle.process.poll() is not None:
                    _, confirmed_members = self._group_ownership(process)
                    return not confirmed_members
            if not ownership_matches:
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        ownership_matches, members = self._group_ownership(process)
        if not ownership_matches or members or handle.process.poll() is None:
            return False
        try:
            _ = handle.process.wait(timeout=0.0)
        except subprocess.TimeoutExpired:
            return False
        return handle.process.poll() is not None

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        handle = self._running.get(process.pid)
        if handle is None:
            return ProcessCleanup(
                process.host_name,
                False,
                False,
                False,
                "owned process handle is missing",
            )
        forced = False
        try:
            ownership_matches, members = self._group_ownership(process)
            if not ownership_matches:
                return ProcessCleanup(
                    process.host_name,
                    False,
                    False,
                    False,
                    "process-group ownership no longer matches receipt",
                )
            if members:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.process_group_id, signal.SIGTERM)
                terminated = self._wait_group_gone(process, handle, timeout_seconds)
                if not terminated:
                    ownership_matches, _ = self._group_ownership(process)
                    if not ownership_matches:
                        return ProcessCleanup(
                            process.host_name,
                            False,
                            False,
                            False,
                            "ownership changed before forced cleanup",
                        )
                    forced = True
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.process_group_id, signal.SIGKILL)
                    terminated = self._wait_group_gone(
                        process, handle, min(timeout_seconds, 5.0)
                    )
            else:
                terminated = self._wait_group_gone(process, handle, timeout_seconds)
            if terminated:
                _ = handle.process.wait(timeout=0.0)
                if handle.process.poll() is None:
                    terminated = False
            if terminated:
                handle.log_file.close()
                _ = self._running.pop(process.pid, None)
            return ProcessCleanup(
                process.host_name,
                True,
                terminated,
                forced,
                (
                    None
                    if terminated
                    else "owned process group or Popen leader survived cleanup"
                ),
            )
        except BaseException as error:
            return ProcessCleanup(
                process.host_name,
                False,
                False,
                forced,
                f"{type(error).__name__}: {error}",
            )

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue:
        normalized_path = path if path.startswith("/") else "/" + path
        if params:
            normalized_path += "?" + urlencode(params)
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self._config.ports.api,
            timeout=self._config.timeouts.request_seconds,
        )
        payload: str | None = None
        headers = {"Accept": "application/json"}
        if body is not None:
            payload = json.dumps(body, sort_keys=True)
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, normalized_path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise HttpResponseError(response.status, response.reason, raw)
            if not raw:
                return None
            return _parse_json_value(raw, f"HTTP {method} {normalized_path}")
        finally:
            connection.close()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def write_result_json(self, filename: str, value: JsonObject) -> None:
        if filename not in {RUNTIME_METADATA_FILENAME, BENCHMARK_RESULT_FILENAME}:
            raise OracleError("result fragment filename is not allowed")
        temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
        temporary_created = False
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._result_directory_descriptor,
            )
            temporary_created = True
            with os.fdopen(
                descriptor, mode="w", encoding="utf-8", closefd=False
            ) as output:
                json.dump(value, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fchmod(descriptor, 0o644)
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(
                    temporary_name,
                    filename,
                    src_dir_fd=self._result_directory_descriptor,
                    dst_dir_fd=self._result_directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise OracleError(
                    f"refusing to replace existing result fragment {filename}"
                ) from error
            os.fsync(self._result_directory_descriptor)
        except OracleError:
            raise
        except OSError as error:
            raise OracleError(
                f"cannot create result fragment {filename}: {error}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_created:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(
                        temporary_name,
                        dir_fd=self._result_directory_descriptor,
                    )


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


def parse_lease_preparation_args(
    arguments: Sequence[str],
) -> LeasePreparationCliArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete strict TP1 config and create canonical, fresh "
            "benchmark-lease metadata. This does not generate hardware config."
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
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-root", required=True, type=Path)
    return parser.parse_args(arguments, namespace=LeasePreparationCliArguments())


def _require_absolute_path(path: Path, description: str) -> None:
    if not path.is_absolute():
        raise OracleError(f"{description} must be an absolute path")
    if "\0" in str(path):
        raise OracleError(f"{description} must not contain NUL")


def _require_canonical_regular_file(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise OracleError(f"cannot resolve {description} {path}: {error}") from error
    if resolved != path:
        raise OracleError(f"{description} must be canonical and must not use symlinks")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise OracleError(f"cannot inspect {description} {path}: {error}") from error
    if not stat.S_ISREG(mode):
        raise OracleError(f"{description} must be a regular file")


def _require_canonical_directory(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise OracleError(f"cannot resolve {description} {path}: {error}") from error
    if resolved != path:
        raise OracleError(f"{description} must be canonical and must not use symlinks")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise OracleError(f"cannot inspect {description} {path}: {error}") from error
    if not stat.S_ISDIR(mode):
        raise OracleError(f"{description} must be a directory")


def _require_absolute_executable(path: Path, description: str) -> None:
    _require_absolute_path(path, description)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OracleError(f"{description} must be an executable file")


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
        raise OracleError(
            f"cannot validate configured Python interpreter: {error}"
        ) from error
    if completed.returncode != 0:
        raise OracleError(
            "configured Python interpreter probe failed with return code "
            f"{completed.returncode}: {completed.stderr[-300:]}"
        )
    try:
        identity = _object(
            _parse_json_value(completed.stdout, "configured Python identity probe"),
            "configured Python identity probe",
        )
    except OracleError as error:
        raise OracleError(
            "configured Python interpreter did not return the identity probe"
        ) from error
    expected_identity: dict[str, object] = {
        "marker": marker,
        "implementation": "cpython",
        "version": [3, 13],
    }
    if any(
        identity.get(name) != expected for name, expected in expected_identity.items()
    ):
        raise OracleError(
            "configured Python interpreter returned an incompatible identity"
        )
    executable = identity.get("executable")
    if not isinstance(executable, str):
        raise OracleError("configured Python interpreter omitted sys.executable")
    try:
        same_executable = Path(executable).samefile(path)
    except OSError as error:
        raise OracleError(
            "cannot bind configured Python interpreter to sys.executable"
        ) from error
    if not same_executable:
        raise OracleError(
            "configured Python interpreter does not match its sys.executable"
        )


def _canonical_prospective_path(path: Path, description: str) -> Path:
    _require_absolute_path(path, description)
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise OracleError(f"cannot resolve {description} {path}: {error}") from error
    if not resolved.is_absolute():
        raise OracleError(f"resolved {description} must remain absolute")
    return resolved


def _require_outside_directory(path: Path, directory: Path, description: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError:
        return
    raise OracleError(f"{description} must be outside the source deployment")


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise OracleError(f"cannot inspect path {path}: {error}") from error
    return True


def _format_cli_number(value: float, description: str) -> str:
    if not math.isfinite(value) or value <= 0:
        raise OracleError(f"{description} must be finite and positive")
    return str(value)


def _open_directory_without_symlinks(path: Path) -> int:
    """Open an absolute directory without following any ancestor symlink."""
    _require_absolute_path(path, "directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            if component in ("", ".", "..") or "\0" in component:
                raise OracleError(f"directory path is not canonical: {path}")
            child_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except OracleError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        raise OracleError(
            f"cannot open directory without symlinks {path}: {error}"
        ) from error


def _atomic_write_new_json(path: Path, value: Mapping[str, object]) -> None:
    """Install metadata atomically without following or replacing symlinks."""
    _require_absolute_path(path, "metadata output")
    if not path.name:
        raise OracleError("metadata output must name a file")
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
            raise OracleError(f"metadata output already exists: {path}")
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
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
            raise OracleError(f"metadata output already exists: {path}") from error
        os.fsync(directory_descriptor)
    except OracleError:
        raise
    except OSError as error:
        raise OracleError(
            f"cannot atomically create metadata {path}: {error}"
        ) from error
    finally:
        if temporary_created:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _source_command(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OracleError(f"cannot inspect source identity: {error}") from error
    if completed.returncode != 0:
        raise OracleError(
            f"source identity command failed with {completed.returncode}: "
            f"{completed.stderr[-500:]}"
        )
    return completed.stdout.strip()


def _read_source_identity(source_directory: str) -> SourceIdentity:
    commit = _source_command(("git", "-C", source_directory, "rev-parse", "HEAD"))
    tracked = _source_command(
        ("git", "-C", source_directory, "diff", "--name-only", "-z", "HEAD")
    )
    untracked = _source_command(
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
        path for path in {*tracked.split("\0"), *untracked.split("\0")} if path
    )
    if dirty_paths:
        raise OracleError(
            "TP1 preparation source must be clean; dirty paths: "
            + ", ".join(dirty_paths[:20])
        )
    try:
        return SourceIdentity(commit=commit, dirty_file_hashes={})
    except ValidationError as error:
        raise OracleError(f"invalid source identity: {error}") from error


def _validate_metadata_with_benchmark_lease(
    benchmark_lease_script: Path,
    metadata: dict[str, object],
    generated_at: datetime,
) -> None:
    specification = importlib.util.spec_from_file_location(
        "_exo_tp1_benchmark_lease_validation", benchmark_lease_script
    )
    if specification is None or specification.loader is None:
        raise OracleError("cannot import the exact benchmark lease wrapper")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
        validator = cast(LeaseMetadataValidator, cast(object, module))
        validated = validator.validate_run_metadata(metadata, now=generated_at)
    except Exception as error:
        raise OracleError(
            f"benchmark lease rejected generated metadata: {error}"
        ) from error
    if validated != metadata:
        raise OracleError("benchmark lease metadata validation changed the document")


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
        raise OracleError(f"metadata output already exists: {metadata_output}")

    config = load_config(config_path)
    source_directory = Path(config.source_directory)
    _require_canonical_directory(source_directory, "source directory")
    expected_harness_script = source_directory / "scripts" / Path(__file__).name
    expected_wrapper_script = source_directory / "scripts" / "benchmark_lease.py"
    if (
        harness_script != expected_harness_script
        or harness_script != Path(__file__).resolve()
    ):
        raise OracleError("harness script does not match the config source deployment")
    if benchmark_lease_script != expected_wrapper_script:
        raise OracleError(
            "benchmark lease script does not match the config source deployment"
        )
    if child_python != Path(config.python_executable):
        raise OracleError("child Python does not match config.python_executable")
    if wrapper_python != child_python:
        raise OracleError(
            "wrapper Python must match the configured child Python interpreter"
        )
    _require_python_interpreter(wrapper_python)

    result_directory = Path(config.result_directory)
    if result_directory != result_root / config.run_id:
        raise OracleError(
            "config result_directory must exactly equal result_root/run_id"
        )
    generated_paths = (metadata_output, result_directory, lease_path, lock_path)
    if len(set(generated_paths)) != len(generated_paths):
        raise OracleError(
            "metadata output, result directory, lease path, and lock path must be "
            "pairwise distinct"
        )
    if _path_exists_without_following(result_directory):
        raise OracleError(f"result directory already exists: {result_directory}")
    for generated_path, description in (
        (metadata_output, "metadata output"),
        (result_directory, "result directory"),
        (lease_path, "lease path"),
        (lock_path, "lock path"),
    ):
        _require_outside_directory(generated_path, source_directory, description)

    if not owner.strip() or "\0" in owner:
        raise OracleError("owner must be nonempty and must not contain NUL")
    if not purpose.strip() or "\0" in purpose:
        raise OracleError("purpose must be nonempty and must not contain NUL")
    expected_duration = _format_cli_number(
        expected_duration_seconds, "expected duration"
    )
    heartbeat = _format_cli_number(heartbeat_seconds, "heartbeat interval")
    cleanup_grace = _format_cli_number(cleanup_grace_seconds, "cleanup grace")
    minimum_grace = minimum_cleanup_grace_seconds(config)
    if cleanup_grace_seconds < minimum_grace:
        raise OracleError(
            "cleanup grace is shorter than the TP1 cleanup bound "
            f"({minimum_grace} seconds)"
        )

    identity_reader = source_identity or _read_source_identity
    observed_source = identity_reader(config.source_directory)
    if observed_source != config.source:
        raise OracleError("strict config source identity is stale")
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
        raise OracleError("metadata clock must return an offset-aware timestamp")
    generated_at_utc = generated_at_value.astimezone(timezone.utc)
    generated_at = generated_at_utc.isoformat(timespec="seconds")
    metadata = lease_static_metadata(config, child_argv)
    metadata["generated_at"] = generated_at
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
    if identity_reader(config.source_directory) != observed_source:
        raise OracleError("source identity changed while preparing lease metadata")
    _atomic_write_new_json(metadata_output, metadata)
    return LeasePreparation(
        metadata=metadata,
        child_argv=child_argv,
        benchmark_lease_argv=benchmark_lease_argv,
        generated_at=generated_at,
        minimum_cleanup_grace_seconds=minimum_grace,
    )


def prepare_lease_main(arguments: Sequence[str]) -> int:
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


class CliArguments(argparse.Namespace):
    config: Path
    lease_path: Path
    lock_path: Path
    result_dir: Path | None


def parse_args(arguments: Sequence[str]) -> CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args(arguments, namespace=CliArguments())


def current_process_command() -> tuple[str, ...]:
    try:
        arguments = tuple(
            part.decode("utf-8")
            for part in Path("/proc/self/cmdline").read_bytes().split(b"\0")
            if part
        )
    except (OSError, UnicodeDecodeError):
        arguments = (sys.executable, *sys.argv)
    if not arguments:
        raise OracleError("cannot determine current child command")
    return arguments


def main(arguments: Sequence[str] | None = None) -> int:
    normalized_arguments = tuple(sys.argv[1:] if arguments is None else arguments)
    if normalized_arguments and normalized_arguments[0] == "prepare-lease":
        return prepare_lease_main(normalized_arguments[1:])
    args = parse_args(normalized_arguments)
    if not args.config.is_absolute():
        raise OracleError("--config must be absolute")
    if not args.lease_path.is_absolute() or not args.lock_path.is_absolute():
        raise OracleError("--lease-path and --lock-path must be absolute")
    config = load_config(args.config)
    if args.result_dir is not None and str(args.result_dir) != config.result_directory:
        raise OracleError("--result-dir must exactly match config.result_directory")
    command = (
        current_process_command()
        if arguments is None
        else (sys.executable, str(Path(__file__).resolve()), *normalized_arguments)
    )
    guard = LeaseGuard(
        config,
        command=command,
        config_path=args.config,
        lease_path=args.lease_path,
        lock_path=args.lock_path,
        process_id=os.getpid(),
        parent_process_id=os.getppid(),
    )
    guard.bind()
    result_directory_descriptor = inherited_result_directory_descriptor(
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
        result = run_oracle_capture(
            config,
            SystemEffects(config, guard, result_directory_descriptor),
            latch,
        )
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
    if latch.signal_number is not None:
        return 128 + latch.signal_number
    if result.get("status") != "completed":
        print(result.get("error") or result.get("status"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
