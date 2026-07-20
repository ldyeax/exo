#!/usr/bin/env python3
"""Run the native three-node GLM-4.7 Flash TP3/EP benchmark.

The controller reuses the PP3 diagnostic's process-group ownership supervisor,
remote stop protocol, log handling, and HCA probes.  It launches fwuff rank 2,
then dwagon rank 1, and rank 0 last because SGLang's DP-attention ``PortArgs``
checks shared coordinator ports before rank 0 binds them.  Only rank 0 serves
inference; nonzero ranks expose dummy health endpoints after engine startup.

Each run performs fast immutable model admission against the previously
full-hashed packaged contract.  It rehashes the small runtime files, config,
and index and checks the exact shard set and sizes, but deliberately does not
rehash 62.4 GB of weight payload on each host before every model load.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import IO, Final, Literal, Protocol, cast, final

from pydantic import Field, ValidationError, model_validator

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import httpx  # noqa: E402

from exo.shared.types.common import Host, NodeId  # noqa: E402
from exo.shared.types.worker.sglang_kt import (  # noqa: E402
    AbsoluteRuntimePath,
    ResourceIndex,
    SglangKtTargetProfile,
)
from exo.utils.pydantic_ext import FrozenModel  # noqa: E402
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
)
from exo.worker.sglang_kt.model_contract import (  # noqa: E402
    load_sglang_kt_model_contract,
)
from exo.worker.sglang_kt.native_glm47_parallelism import (  # noqa: E402
    DWAGON_NODE_ID,
    FWUFF_NODE_ID,
    GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS,
    Glm47NativeParallelism,
    Glm47NativeProcessSpec,
    Glm47NativeSourceSupport,
    Glm47NativeTpEpPlan,
    build_dwagon_fwuff_native_glm47_plan,
    build_native_glm47_benchmark_protocol,
    build_native_glm47_process_specs,
    calculate_native_glm47_process_spec_sha256,
    inspect_native_glm47_source_support,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (  # noqa: E402
    SglangKtServingInvocationEvidence,
    SglangKtServingWorkloadEvidence,
)
from scripts import run_sglang_kt_glm47_pp3_diagnostic as lifecycle  # noqa: E402
from scripts.sglang_kt_glm47_serving_client import (  # noqa: E402
    Glm47NativeServingClient,
    Glm47ServingClientError,
    PreparedServingWorkload,
    prepare_glm47_serving_workload,
    run_glm47_serving_invocation,
    run_glm47_serving_sanity,
)
from scripts.validate_sglang_kt_glm47_model import (  # noqa: E402
    require_immutable_model_snapshot,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type SignalHandler = (
    signal.Handlers | Callable[[int, FrameType | None], object] | int | None
)

DEFAULT_DWAGON_RUNTIME: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/"
    "14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455/"
    "venv/bin/python"
)
DEFAULT_DWAGON_MODEL: Final = (
    "/var/lib/exo/models/"
    "zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
DEFAULT_FWUFF_MODEL: Final = (
    "/mnt/sanic/exo/models/"
    "zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
DEFAULT_DWAGON_SGLANG_SOURCE: Final = (
    "/var/lib/exo/sources/ktransformers-glm47-f9ca696/third_party/sglang"
)
DEFAULT_DWAGON_REPOSITORY: Final = "/root/exo"
DEFAULT_DWAGON_IP: Final = "192.168.40.24"
DEFAULT_FWUFF_IP: Final = "192.168.40.248"
DEFAULT_DWAGON_INTERFACE: Final = "ens13f0np0"
DEFAULT_FWUFF_INTERFACE: Final = "ens17f0"
DEFAULT_HCA_DEVICES: Final = ("mlx4_0:1", "mlx4_0:2")
# Keep fixed ports below Linux's current ephemeral range (32768-60999).
DEFAULT_DISTRIBUTED_PORT: Final = 30_000
DEFAULT_RANK_PORTS: Final = (30_100, 30_101, 30_102)
DEFAULT_LOCK_PATH: Final = Path("/var/lock/exo-glm47-native-tp-ep.lock")
CONTROLLER_FILENAME: Final = "run_sglang_kt_glm47_native_tp_ep_benchmark.py"
RESULT_FILENAME: Final = "native-tp-ep-benchmark-result.json"
WARMUP_COUNT: Final = 2
SAMPLE_COUNT: Final = 3
MINIMUM_HCA_PAYLOAD_BYTES_PER_HOST_RAIL: Final = 1024 * 1024
_REMOTE_COMMAND_TIMEOUT_SECONDS: Final = 300.0
_HEALTH_COUNTER_NAMES: Final = (
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


class NativeTpEpBenchmarkError(RuntimeError):
    """Raised when a native TP/EP run cannot produce admitted evidence."""


@final
class FastImmutableModelEvidence(FrozenModel):
    """Live metadata bound to the previously full-hashed model contract."""

    model_path: AbsoluteRuntimePath
    contract_path: AbsoluteRuntimePath
    contract_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_files_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_size_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_map_entries: int = Field(gt=0)
    shard_count: int = Field(gt=0)
    physical_weight_bytes: int = Field(gt=0)
    immutable_root_owned: Literal[True]
    weight_payload_rehashed: Literal[False]
    verification_basis: Literal[
        "pinned_full_hash_contract_plus_live_immutable_metadata"
    ]


@final
class NativeHostInspection(FrozenModel):
    node_id: NodeId
    controller_source_path: AbsoluteRuntimePath
    controller_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_contract_source_path: AbsoluteRuntimePath
    native_contract_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_support: Glm47NativeSourceSupport
    model: FastImmutableModelEvidence

    @model_validator(mode="after")
    def validate_node_binding(self) -> "NativeHostInspection":
        if self.source_support.node_id != self.node_id:
            raise ValueError("host inspection source support has the wrong node")
        return self


@final
class PortCheckEvidence(FrozenModel):
    host_name: Literal["dwagon", "fwuff"]
    bind_ip: str
    ports: tuple[int, ...]
    checked_at_utc: str
    all_clear: Literal[True]


@final
class OwnershipProbe(FrozenModel):
    ownership_matches: bool
    members: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NativeTpEpConfig:
    run_id: str
    mode: Glm47NativeParallelism
    result_directory: Path
    lock_path: Path
    dwagon_runtime_python: str
    fwuff_runtime_python: str
    dwagon_model_path: str
    fwuff_model_path: str
    dwagon_sglang_source_directory: str
    fwuff_sglang_source_directory: str
    dwagon_repository_directory: str
    fwuff_repository_directory: str
    dwagon_model_contract: str
    fwuff_model_contract: str
    ssh_target: str
    dwagon_ip: str
    fwuff_ip: str
    dwagon_socket_interface: str
    fwuff_socket_interface: str
    distributed_port: int
    rank_ports: tuple[int, int, int]
    hca_devices: tuple[str, str]
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float


class _RunArguments(Protocol):
    run_id: str
    mode: str
    result_directory: Path
    lock_path: Path
    dwagon_runtime_python: str
    fwuff_runtime_python: str
    dwagon_model_path: str
    fwuff_model_path: str
    dwagon_sglang_source_directory: str
    fwuff_sglang_source_directory: str
    dwagon_repository_directory: str
    fwuff_repository_directory: str
    dwagon_model_contract: str | None
    fwuff_model_contract: str | None
    ssh_target: str
    dwagon_ip: str
    fwuff_ip: str
    dwagon_socket_interface: str
    fwuff_socket_interface: str
    distributed_port: int
    rank_zero_port: int
    rank_one_port: int
    rank_two_port: int
    hca_devices: tuple[str, str]
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class _LifecycleStageSpec:
    """Structural adapter for the established PP3 ownership lifecycle."""

    native: Glm47NativeProcessSpec

    @property
    def pipeline_rank(self) -> ResourceIndex:
        return self.native.world_rank

    @property
    def node_id(self) -> NodeId:
        return self.native.rank.node_id

    @property
    def command(self) -> tuple[str, ...]:
        return self.native.command

    @property
    def arguments(self) -> tuple[str, ...]:
        return self.native.arguments

    @property
    def cpu_cores(self) -> tuple[ResourceIndex, ...]:
        return self.native.rank.cpu_cores

    @property
    def memory_nodes(self) -> tuple[ResourceIndex, ...]:
        return self.native.rank.memory_nodes

    @property
    def service_endpoint(self) -> Host:
        return self.native.rank.service_endpoint

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        return self.native.environment

    @property
    def unset_environment_variables(self) -> tuple[str, ...]:
        return self.native.unset_environment_variables

    @property
    def unset_environment_variable_prefixes(self) -> tuple[str, ...]:
        return self.native.unset_environment_variable_prefixes

    @property
    def target_profile(self) -> SglangKtTargetProfile:
        # PP3 uses this only to select dual-rail NCCL environment handling.
        return GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE


class _CancellationLatch:
    def __init__(self) -> None:
        self.reason: str | None = None
        self._prior: dict[int, object] = {}

    def __enter__(self) -> "_CancellationLatch":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._prior[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_arguments: object) -> None:
        for signum, handler in self._prior.items():
            signal.signal(
                signum,
                cast(SignalHandler, handler),
            )

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        self.reason = signal.Signals(signum).name
        raise NativeTpEpBenchmarkError(f"benchmark cancelled by {self.reason}")

    def checkpoint(self) -> None:
        if self.reason is not None:
            raise NativeTpEpBenchmarkError(f"benchmark cancelled by {self.reason}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_sglang_kt_json(value)).hexdigest()


def _object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NativeTpEpBenchmarkError(f"{description} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise NativeTpEpBenchmarkError(f"{description} must be a JSON object")
    return cast(dict[str, object], mapping)


def _exact_weight_map(index_contents: bytes) -> dict[str, str]:
    root = _object(parse_sglang_kt_strict_json(index_contents), "model index")
    if set(root) != {"metadata", "weight_map"}:
        raise NativeTpEpBenchmarkError("model index has unexpected top-level fields")
    raw_weight_map = _object(root["weight_map"], "model index weight_map")
    if not all(
        bool(tensor) and isinstance(shard, str) and bool(shard)
        for tensor, shard in raw_weight_map.items()
    ):
        raise NativeTpEpBenchmarkError("model index weight_map is not string-to-string")
    return cast(dict[str, str], raw_weight_map)


def inspect_fast_immutable_model(
    model_path: Path,
    contract_path: Path,
) -> FastImmutableModelEvidence:
    """Bind a live immutable snapshot without repeating the full shard rehash."""

    if not model_path.is_absolute() or not contract_path.is_absolute():
        raise NativeTpEpBenchmarkError("model and contract paths must be absolute")
    require_immutable_model_snapshot(model_path)
    loaded = load_sglang_kt_model_contract(
        contract_path,
        expected_contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    )
    contract = loaded.contract
    if (
        contract.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
        or contract.revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or contract.ktransformers_method != "BF16"
    ):
        raise NativeTpEpBenchmarkError("model contract identity is not pinned")

    files_by_path = {file.path: file for file in contract.files}
    config_contract = files_by_path.get("config.json")
    index_contract = files_by_path.get("model.safetensors.index.json")
    if config_contract is None or index_contract is None:
        raise NativeTpEpBenchmarkError("model contract lacks config or index")
    config = read_sglang_kt_bound_file(
        model_path / "config.json",
        maximum_bytes=config_contract.size_bytes,
    )
    index = read_sglang_kt_bound_file(
        model_path / "model.safetensors.index.json",
        maximum_bytes=index_contract.size_bytes,
    )
    if (
        config.sha256 != config_contract.sha256
        or len(config.contents) != config_contract.size_bytes
        or index.sha256 != index_contract.sha256
        or len(index.contents) != index_contract.size_bytes
    ):
        raise NativeTpEpBenchmarkError("live model config or index changed")

    weight_map = _exact_weight_map(index.contents)
    shard_contracts = tuple(
        sorted(
            (file for file in contract.files if file.role == "weight_shard"),
            key=lambda file: file.path,
        )
    )
    expected_shards = tuple(file.path for file in shard_contracts)
    if (
        len(weight_map) != contract.weight_map_entries
        or tuple(sorted(set(weight_map.values()))) != expected_shards
    ):
        raise NativeTpEpBenchmarkError("live model index shard mapping changed")
    try:
        observed_shards = tuple(
            sorted(
                entry.name
                for entry in os.scandir(model_path)
                if entry.name.endswith(".safetensors")
                and entry.is_file(follow_symlinks=False)
                and not entry.is_symlink()
            )
        )
    except OSError as error:
        raise NativeTpEpBenchmarkError("cannot enumerate model shards") from error
    if observed_shards != expected_shards:
        raise NativeTpEpBenchmarkError("live safetensors shard set changed")

    shard_sizes: list[dict[str, object]] = []
    for contract_file in shard_contracts:
        path = model_path / contract_file.path
        observed = path.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != contract_file.size_bytes
            or observed.st_nlink != 1
        ):
            raise NativeTpEpBenchmarkError(
                f"live model shard metadata changed: {contract_file.path}"
            )
        shard_sizes.append({"path": contract_file.path, "size_bytes": observed.st_size})

    runtime_files: list[dict[str, object]] = []
    for contract_file in sorted(contract.files, key=lambda file: file.path):
        if contract_file.role == "weight_shard":
            continue
        artifact = read_sglang_kt_bound_file(
            model_path / contract_file.path,
            maximum_bytes=contract_file.size_bytes,
        )
        if (
            artifact.sha256 != contract_file.sha256
            or len(artifact.contents) != contract_file.size_bytes
        ):
            raise NativeTpEpBenchmarkError(
                f"live model runtime file changed: {contract_file.path}"
            )
        runtime_files.append(
            {
                "path": contract_file.path,
                "size_bytes": contract_file.size_bytes,
                "sha256": artifact.sha256,
            }
        )

    return FastImmutableModelEvidence(
        model_path=str(model_path),
        contract_path=loaded.path,
        contract_receipt_sha256=loaded.receipt_sha256,
        contract_sha256=loaded.contract_sha256,
        config_sha256=config.sha256,
        index_sha256=index.sha256,
        runtime_files_sha256=_canonical_sha256(runtime_files),
        shard_size_manifest_sha256=_canonical_sha256(shard_sizes),
        weight_map_entries=contract.weight_map_entries,
        shard_count=len(shard_contracts),
        physical_weight_bytes=sum(file.size_bytes for file in shard_contracts),
        immutable_root_owned=True,
        weight_payload_rehashed=False,
        verification_basis="pinned_full_hash_contract_plus_live_immutable_metadata",
    )


def inspect_host(
    *,
    node_id: NodeId,
    repository_directory: Path,
    sglang_source_directory: Path,
    runtime_executable: Path,
    model_path: Path,
    model_contract: Path,
) -> NativeHostInspection:
    controller_path = repository_directory / "scripts" / CONTROLLER_FILENAME
    native_source_path = (
        repository_directory / "src/exo/worker/sglang_kt/native_glm47_parallelism.py"
    )
    for path in (controller_path, native_source_path):
        if not path.is_file() or path.is_symlink():
            raise NativeTpEpBenchmarkError(f"controller source is not regular: {path}")
    support = inspect_native_glm47_source_support(
        node_id=node_id,
        source_directory=sglang_source_directory,
        model_config_path=model_path / "config.json",
        runtime_executable=runtime_executable,
    )
    return NativeHostInspection(
        node_id=node_id,
        controller_source_path=str(controller_path),
        controller_source_sha256=_sha256_file(controller_path),
        native_contract_source_path=str(native_source_path),
        native_contract_source_sha256=_sha256_file(native_source_path),
        source_support=support,
        model=inspect_fast_immutable_model(model_path, model_contract),
    )


def _remote_controller_command(
    config: NativeTpEpConfig,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    repository = Path(config.fwuff_repository_directory)
    python_path = f"{repository / 'src'}:{repository}"
    return (
        "/usr/bin/env",
        "-i",
        "HOME=/root",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "PYTHONSAFEPATH=1",
        f"PYTHONPATH={python_path}",
        config.fwuff_runtime_python,
        "-P",
        str(repository / "scripts" / CONTROLLER_FILENAME),
        *arguments,
    )


def _run_remote_json(
    config: NativeTpEpConfig,
    arguments: Sequence[str],
    *,
    timeout_seconds: float = _REMOTE_COMMAND_TIMEOUT_SECONDS,
) -> JsonObject:
    completed = subprocess.run(
        lifecycle._ssh_argv(  # pyright: ignore[reportPrivateUsage]
            config.ssh_target,
            _remote_controller_command(config, arguments),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise NativeTpEpBenchmarkError(
            "fwuff controller command failed: "
            f"{completed.stderr[-2000:] or completed.stdout[-2000:]}"
        )
    try:
        parsed = parse_sglang_kt_strict_json(completed.stdout.encode())
        return cast(
            JsonObject,
            _object(parsed, "fwuff controller response"),
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise NativeTpEpBenchmarkError(
            "fwuff controller response was not singular JSON"
        ) from error


def _strict_model_json(payload: JsonObject) -> str:
    """Preserve JSON container semantics for strict tuple-backed receipt models."""

    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def inspect_remote_host(config: NativeTpEpConfig) -> NativeHostInspection:
    payload = _run_remote_json(
        config,
        (
            "inspect-host",
            "--node-id",
            str(FWUFF_NODE_ID),
            "--repository-directory",
            config.fwuff_repository_directory,
            "--sglang-source-directory",
            config.fwuff_sglang_source_directory,
            "--runtime-python",
            config.fwuff_runtime_python,
            "--model-path",
            config.fwuff_model_path,
            "--model-contract",
            config.fwuff_model_contract,
        ),
    )
    return NativeHostInspection.model_validate_json(_strict_model_json(payload))


def _check_ports_clear(
    host_name: Literal["dwagon", "fwuff"],
    bind_ip: str,
    ports: Sequence[int],
) -> PortCheckEvidence:
    if not ports or len(set(ports)) != len(ports):
        raise NativeTpEpBenchmarkError("port reservation must be nonempty and unique")
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(candidate)
            candidate.bind((bind_ip, port))
    except OSError as error:
        raise NativeTpEpBenchmarkError(
            f"{host_name} reserved port is unavailable: {error}"
        ) from error
    finally:
        for candidate in sockets:
            candidate.close()
    return PortCheckEvidence(
        host_name=host_name,
        bind_ip=bind_ip,
        ports=tuple(ports),
        checked_at_utc=_utc_now(),
        all_clear=True,
    )


def _reserved_ports(
    plan: Glm47NativeTpEpPlan,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    coordinator = tuple(
        plan.distributed_coordinator.port + offset
        for offset in GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS
    )
    dwagon = tuple(
        sorted(
            (
                *coordinator,
                plan.ranks[0].service_endpoint.port,
                plan.ranks[1].service_endpoint.port,
            )
        )
    )
    fwuff = tuple(sorted((*coordinator, plan.ranks[2].service_endpoint.port)))
    return dwagon, fwuff


def check_remote_ports(
    config: NativeTpEpConfig,
    ports: tuple[int, ...],
) -> PortCheckEvidence:
    payload = _run_remote_json(
        config,
        (
            "check-ports",
            "--host-name",
            "fwuff",
            "--bind-ip",
            "0.0.0.0",
            "--ports",
            ",".join(str(port) for port in ports),
        ),
        timeout_seconds=30.0,
    )
    return PortCheckEvidence.model_validate_json(_strict_model_json(payload))


def _lifecycle_spec(
    spec: Glm47NativeProcessSpec,
) -> SglangKtProcessLaunchSpec:
    # PP3 predates a protocol for its stage object.  This adapter provides every
    # field its lifecycle reads; focused tests guard that structural contract.
    return cast(SglangKtProcessLaunchSpec, cast(object, _LifecycleStageSpec(spec)))


def _lifecycle_config(config: NativeTpEpConfig) -> lifecycle.Pp3DiagnosticConfig:
    return lifecycle.Pp3DiagnosticConfig(
        run_id=config.run_id,
        result_directory=config.result_directory,
        dwagon_runtime_python=config.dwagon_runtime_python,
        fwuff_runtime_python=config.fwuff_runtime_python,
        dwagon_model_path=config.dwagon_model_path,
        fwuff_model_path=config.fwuff_model_path,
        local_source_directory=config.dwagon_repository_directory,
        remote_source_directory=config.fwuff_repository_directory,
        ssh_target=config.ssh_target,
        dwagon_ip=config.dwagon_ip,
        fwuff_ip=config.fwuff_ip,
        dwagon_socket_interface=config.dwagon_socket_interface,
        fwuff_socket_interface=config.fwuff_socket_interface,
        distributed_port=config.distributed_port,
        stage_ports=config.rank_ports,
        hca_devices=config.hca_devices,
        pipeline_layer_partition=(16, 16, 15),
        dwagon_stage_placement=lifecycle.DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER,
        resident_gpu_experts=1,
        readiness_timeout_seconds=config.readiness_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        cleanup_timeout_seconds=config.cleanup_timeout_seconds,
        warmup_count=WARMUP_COUNT,
        sample_count=SAMPLE_COUNT,
    )


def _probe_local_owner(running: lifecycle.RunningStage) -> OwnershipProbe:
    matches, members = lifecycle._local_group_ownership(  # pyright: ignore[reportPrivateUsage]
        running.owned
    )
    return OwnershipProbe(ownership_matches=matches, members=members)


def _probe_remote_owner(
    config: NativeTpEpConfig,
    running: lifecycle.RunningStage,
) -> OwnershipProbe:
    receipt = asdict(running.owned)
    payload = _run_remote_json(
        config,
        ("probe-owner", "--receipt-json", json.dumps(receipt, sort_keys=True)),
        timeout_seconds=30.0,
    )
    return OwnershipProbe.model_validate_json(_strict_model_json(payload))


def require_stages_owned(
    config: NativeTpEpConfig,
    running_stages: Sequence[lifecycle.RunningStage],
) -> None:
    ranks = tuple(stage.owned.rank for stage in running_stages)
    if not ranks or len(set(ranks)) != len(ranks):
        raise NativeTpEpBenchmarkError("owned native ranks must be nonempty and unique")
    for running in running_stages:
        if running.process.poll() is not None:
            raise NativeTpEpBenchmarkError(
                f"rank {running.owned.rank} transport exited unexpectedly"
            )
        probe = (
            _probe_remote_owner(config, running)
            if running.owned.remote
            else _probe_local_owner(running)
        )
        if (
            not probe.ownership_matches
            or running.owned.pid not in probe.members
            or not probe.members
        ):
            raise NativeTpEpBenchmarkError(
                f"rank {running.owned.rank} ownership no longer matches its receipt"
            )


def require_all_stages_owned(
    config: NativeTpEpConfig,
    running_stages: Sequence[lifecycle.RunningStage],
) -> None:
    if tuple(sorted(stage.owned.rank for stage in running_stages)) != (0, 1, 2):
        raise NativeTpEpBenchmarkError("all three native ranks must be owned")
    require_stages_owned(config, running_stages)


def wait_for_nonzero_port_admission(
    config: NativeTpEpConfig,
    running: lifecycle.RunningStage,
    all_started: Sequence[lifecycle.RunningStage],
    latch: _CancellationLatch,
) -> None:
    """Wait until a nonzero rank has passed ``PortArgs.init_new`` exactly."""

    if running.owned.rank not in {1, 2}:
        raise NativeTpEpBenchmarkError("port-admission gate requires a nonzero rank")
    marker = "server_args=ServerArgs("
    deadline = time.monotonic() + config.readiness_timeout_seconds
    path = Path(running.owned.log_path)
    while time.monotonic() < deadline:
        latch.checkpoint()
        require_stages_owned(config, all_started)
        try:
            if path.stat().st_size > lifecycle._LOG_MAXIMUM_BYTES:  # pyright: ignore[reportPrivateUsage]
                raise NativeTpEpBenchmarkError(
                    f"rank {running.owned.rank} log exceeded its size bound"
                )
            contents = path.read_text(encoding="utf-8")
        except OSError as error:
            raise NativeTpEpBenchmarkError(
                f"cannot inspect rank {running.owned.rank} startup log"
            ) from error
        if marker in contents:
            require_stages_owned(config, all_started)
            return
        time.sleep(0.1)
    raise NativeTpEpBenchmarkError(
        f"rank {running.owned.rank} did not pass PortArgs.init_new before timeout"
    )


def start_native_stages(
    specs: tuple[
        Glm47NativeProcessSpec,
        Glm47NativeProcessSpec,
        Glm47NativeProcessSpec,
    ],
    config: NativeTpEpConfig,
    pp3_config: lifecycle.Pp3DiagnosticConfig,
    owner_token: str,
    latch: _CancellationLatch,
    running: list[lifecycle.RunningStage],
) -> None:
    """Start rank 2, gate it, start rank 1, gate it, then start rank 0."""

    if running:
        raise NativeTpEpBenchmarkError("native stage output list must start empty")
    for rank in (2, 1):
        nonzero = lifecycle.start_stage(
            _lifecycle_spec(specs[rank]), pp3_config, owner_token
        )
        running.append(nonzero)
        wait_for_nonzero_port_admission(config, nonzero, running, latch)
    running.append(
        lifecycle.start_stage(_lifecycle_spec(specs[0]), pp3_config, owner_token)
    )
    running.sort(key=lambda stage: stage.owned.rank)


def wait_for_rank_zero(
    config: NativeTpEpConfig,
    plan: Glm47NativeTpEpPlan,
    running_stages: Sequence[lifecycle.RunningStage],
    latch: _CancellationLatch,
) -> JsonObject:
    deadline = time.monotonic() + config.readiness_timeout_seconds
    endpoint = plan.ranks[0].service_endpoint
    while time.monotonic() < deadline:
        latch.checkpoint()
        require_all_stages_owned(config, running_stages)
        try:
            with Glm47NativeServingClient(
                f"http://{endpoint}", timeout_seconds=2.0
            ) as client:
                health = client.health_generate()
        except (Glm47ServingClientError, httpx.HTTPError, OSError):
            time.sleep(0.25)
            continue
        return cast(JsonObject, health.model_dump(mode="json"))
    raise NativeTpEpBenchmarkError("rank-zero inference endpoint readiness timed out")


def validate_server_info(
    response: Mapping[str, object],
    plan: Glm47NativeTpEpPlan,
) -> None:
    expected: dict[str, object] = {
        "model_path": str(plan.ranks[0].model_path),
        "tp_size": 3,
        "pp_size": 1,
        "dp_size": 3,
        "enable_dp_attention": True,
        "moe_dense_tp_size": 1,
        "ep_size": plan.expert_parallel_size,
        "ep_num_redundant_experts": plan.redundant_expert_count,
        "nnodes": 3,
        "node_rank": 0,
        "dist_init_addr": str(plan.distributed_coordinator),
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "disable_shared_experts_fusion": True,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "max_running_requests": plan.max_running_requests,
        "max_total_tokens": plan.max_total_tokens,
        "mem_fraction_static": plan.static_memory_fraction,
        "context_length": plan.context_length,
        "chunked_prefill_size": plan.chunked_prefill_size // 3,
    }
    mismatches = {
        name: {"expected": value, "observed": response.get(name)}
        for name, value in expected.items()
        if response.get(name) != value
    }
    if mismatches:
        raise NativeTpEpBenchmarkError(
            f"rank-zero server_info does not match native plan: {mismatches}"
        )


def collect_owned_workload(
    client: Glm47NativeServingClient,
    workload: PreparedServingWorkload,
    config: NativeTpEpConfig,
    running: Sequence[lifecycle.RunningStage],
    latch: _CancellationLatch,
) -> SglangKtServingWorkloadEvidence:
    warmups: list[SglangKtServingInvocationEvidence] = []
    samples: list[SglangKtServingInvocationEvidence] = []
    for destination, count in ((warmups, WARMUP_COUNT), (samples, SAMPLE_COUNT)):
        for ordinal in range(1, count + 1):
            latch.checkpoint()
            require_all_stages_owned(config, running)
            destination.append(run_glm47_serving_invocation(client, workload, ordinal))
            require_all_stages_owned(config, running)
    return SglangKtServingWorkloadEvidence(
        request=workload.receipt_request,
        warmups=tuple(warmups),
        samples=tuple(samples),
    )


def validate_hca_evidence(deltas: Mapping[str, object]) -> JsonObject:
    failures: list[str] = []
    rail_payload: dict[str, JsonValue] = {}
    for host_name in ("dwagon", "fwuff"):
        host = _object(deltas.get(host_name), f"{host_name} HCA deltas")
        if set(host) != {"rail-1", "rail-2"}:
            failures.append(f"{host_name} does not contain exactly two HCA rails")
            continue
        for rail_id in ("rail-1", "rail-2"):
            rail = _object(host[rail_id], f"{host_name}/{rail_id}")
            counters = _object(
                rail.get("counter_deltas"), f"{host_name}/{rail_id} counters"
            )
            dirty = {
                name: counters.get(name)
                for name in _HEALTH_COUNTER_NAMES
                if counters.get(name) != 0
            }
            if dirty:
                failures.append(
                    f"{host_name}/{rail_id} health counters changed: {dirty}"
                )
            received = rail.get("received_payload_bytes")
            transmitted = rail.get("transmitted_payload_bytes")
            if not isinstance(received, int) or not isinstance(transmitted, int):
                failures.append(f"{host_name}/{rail_id} payload counters are invalid")
                continue
            rail_payload[f"{host_name}/{rail_id}"] = {
                "received_payload_bytes": received,
                "transmitted_payload_bytes": transmitted,
            }
            if received + transmitted < MINIMUM_HCA_PAYLOAD_BYTES_PER_HOST_RAIL:
                failures.append(
                    f"{host_name}/{rail_id} carried less than "
                    f"{MINIMUM_HCA_PAYLOAD_BYTES_PER_HOST_RAIL} token-workload bytes"
                )
    evidence: JsonObject = {
        "host_staged_by_environment": True,
        "dual_rail_payload_observed": not failures,
        "minimum_payload_bytes_per_host_rail": (
            MINIMUM_HCA_PAYLOAD_BYTES_PER_HOST_RAIL
        ),
        "rail_payload": rail_payload,
        "failures": cast(list[JsonValue], list(failures)),
    }
    if failures:
        raise NativeTpEpBenchmarkError(
            "HCA evidence did not prove clean dual-rail payload: " + "; ".join(failures)
        )
    return evidence


def validate_nccl_log_transport(config: NativeTpEpConfig) -> JsonObject:
    """Require every rank to prove merged dual-rail IB without Socket fallback."""

    required_markers = (
        "NCCL_IB_HCA set to =mlx4_0:1,mlx4_0:2",
        "NET/IB : Using [0]mlx4_0:1/IB [1]mlx4_0:2/IB",
        "Made virtual device [2] name=mlx4_0+mlx4_0 speed=80000 ndevs=2",
        "GPU Direct RDMA Disabled for HCA 0",
        "GPU Direct RDMA Disabled for HCA 1",
        "via NET/IB/2",
    )
    ranks: list[JsonValue] = []
    for rank in range(3):
        path = config.result_directory / f"rank-{rank}.log"
        try:
            status = path.stat()
            if status.st_size > lifecycle._LOG_MAXIMUM_BYTES:  # pyright: ignore[reportPrivateUsage]
                raise NativeTpEpBenchmarkError(
                    f"rank {rank} log exceeded its size bound"
                )
            contents = path.read_text(encoding="utf-8")
        except OSError as error:
            raise NativeTpEpBenchmarkError(
                f"cannot inspect rank {rank} NCCL log"
            ) from error
        missing = tuple(marker for marker in required_markers if marker not in contents)
        socket_fallback = "NET/Socket" in contents
        if missing or socket_fallback:
            raise NativeTpEpBenchmarkError(
                f"rank {rank} did not prove host-staged dual-rail NCCL/IB: "
                f"missing={missing}, socket_fallback={socket_fallback}"
            )
        ranks.append(
            {
                "rank": rank,
                "log_path": str(path),
                "log_size_bytes": status.st_size,
                "required_markers_present": True,
                "socket_fallback_absent": True,
            }
        )
    return {
        "transport": "NCCL/IB",
        "host_staged": True,
        "merged_dual_rail": True,
        "socket_fallback_absent": True,
        "ranks": ranks,
    }


def performance_evidence_is_comparable(
    *,
    failure: BaseException | None,
    cleanup_complete: bool,
    sanity: JsonObject | None,
    workloads: Sequence[JsonValue],
    hca_validation: JsonObject | None,
    nccl_transport: JsonObject | None,
) -> bool:
    return (
        failure is None
        and cleanup_complete
        and sanity is not None
        and len(workloads) == 2
        and hca_validation is not None
        and hca_validation.get("dual_rail_payload_observed") is True
        and nccl_transport is not None
        and nccl_transport.get("transport") == "NCCL/IB"
        and nccl_transport.get("socket_fallback_absent") is True
    )


def _inspection_bindings(
    plan: Glm47NativeTpEpPlan,
    local: NativeHostInspection,
    remote: NativeHostInspection,
) -> None:
    if (
        local.node_id != DWAGON_NODE_ID
        or remote.node_id != FWUFF_NODE_ID
        or local.controller_source_sha256 != remote.controller_source_sha256
        or local.native_contract_source_sha256 != remote.native_contract_source_sha256
    ):
        raise NativeTpEpBenchmarkError(
            "dwagon and fwuff controller source identities do not match"
        )
    for evidence, expected_model, expected_source, expected_runtime in (
        (
            local,
            plan.ranks[0].model_path,
            plan.ranks[0].sglang_source_directory,
            plan.ranks[0].executable,
        ),
        (
            remote,
            plan.ranks[2].model_path,
            plan.ranks[2].sglang_source_directory,
            plan.ranks[2].executable,
        ),
    ):
        evidence.source_support.require_mode(plan.mode)
        if (
            evidence.model.model_path != expected_model
            or evidence.source_support.source_directory != expected_source
            or evidence.source_support.runtime_executable != expected_runtime
            or evidence.model.contract_sha256
            != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
        ):
            raise NativeTpEpBenchmarkError(
                f"{evidence.node_id} inspection does not bind its launch inputs"
            )


def _log_receipts(config: NativeTpEpConfig) -> list[JsonValue]:
    receipts: list[JsonValue] = []
    for rank in range(3):
        path = config.result_directory / f"rank-{rank}.log"
        if path.is_file():
            receipts.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return receipts


def _write_result(config: NativeTpEpConfig, payload: JsonObject) -> None:
    path = config.result_directory / RESULT_FILENAME
    temporary = config.result_directory / f".{RESULT_FILENAME}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _acquire_lock(path: Path) -> IO[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        raise NativeTpEpBenchmarkError(
            f"native GLM benchmark lock is held: {path}"
        ) from error
    return lock


def _process_receipt(stage: lifecycle.RunningStage) -> JsonObject:
    receipt = asdict(stage.owned)
    receipt["owner_token"] = hashlib.sha256(
        stage.owned.owner_token.encode()
    ).hexdigest()
    return cast(JsonObject, receipt)


def run_benchmark(config: NativeTpEpConfig) -> JsonObject:
    """Admit, launch, measure, clean, and publish one native TP/EP run."""

    with _acquire_lock(config.lock_path):
        config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        plan = build_dwagon_fwuff_native_glm47_plan(
            config.mode,
            dwagon_runtime_python=config.dwagon_runtime_python,
            fwuff_runtime_python=config.fwuff_runtime_python,
            dwagon_sglang_source_directory=config.dwagon_sglang_source_directory,
            fwuff_sglang_source_directory=config.fwuff_sglang_source_directory,
            dwagon_model_path=config.dwagon_model_path,
            fwuff_model_path=config.fwuff_model_path,
            dwagon_ip=config.dwagon_ip,
            fwuff_ip=config.fwuff_ip,
            distributed_port=config.distributed_port,
            rank_ports=config.rank_ports,
            hca_devices=config.hca_devices,
            dwagon_socket_interface=config.dwagon_socket_interface,
            fwuff_socket_interface=config.fwuff_socket_interface,
        )
        specs = build_native_glm47_process_specs(plan)
        protocol = build_native_glm47_benchmark_protocol()
        pp3_config = _lifecycle_config(config)
        running: list[lifecycle.RunningStage] = []
        inspections: list[NativeHostInspection] = []
        port_checks: list[PortCheckEvidence] = []
        readiness: JsonObject | None = None
        server_info: JsonObject | None = None
        sanity: JsonObject | None = None
        workloads: list[JsonValue] = []
        summaries: list[JsonValue] = []
        hca_before: dict[str, lifecycle.HcaCounterSnapshot] = {}
        hca_after: dict[str, lifecycle.HcaCounterSnapshot] = {}
        hca_deltas: JsonObject | None = None
        hca_validation: JsonObject | None = None
        nccl_transport: JsonObject | None = None
        cleanup: list[JsonValue] = []
        failure: BaseException | None = None
        cancellation_reason: str | None = None
        owner_token = uuid.uuid4().hex
        latch = _CancellationLatch()
        try:
            with latch:
                local_inspection = inspect_host(
                    node_id=DWAGON_NODE_ID,
                    repository_directory=Path(config.dwagon_repository_directory),
                    sglang_source_directory=Path(config.dwagon_sglang_source_directory),
                    runtime_executable=Path(config.dwagon_runtime_python),
                    model_path=Path(config.dwagon_model_path),
                    model_contract=Path(config.dwagon_model_contract),
                )
                remote_inspection = inspect_remote_host(config)
                inspections.extend((local_inspection, remote_inspection))
                _inspection_bindings(plan, local_inspection, remote_inspection)

                local_ports, remote_ports = _reserved_ports(plan)
                port_checks.append(_check_ports_clear("dwagon", "0.0.0.0", local_ports))
                port_checks.append(check_remote_ports(config, remote_ports))

                start_native_stages(
                    specs,
                    config,
                    pp3_config,
                    owner_token,
                    latch,
                    running,
                )
                require_all_stages_owned(config, running)
                readiness = wait_for_rank_zero(config, plan, running, latch)
                rank_zero = plan.ranks[0]
                with Glm47NativeServingClient(
                    f"http://{rank_zero.service_endpoint}",
                    timeout_seconds=config.request_timeout_seconds,
                ) as client:
                    raw_server_info = client.server_info()
                    validate_server_info(raw_server_info.response, plan)
                    server_info = cast(
                        JsonObject, raw_server_info.model_dump(mode="json")
                    )
                    require_all_stages_owned(config, running)
                    sanity_evidence = run_glm47_serving_sanity(
                        client, config.dwagon_model_path
                    )
                    sanity = cast(JsonObject, sanity_evidence.model_dump(mode="json"))
                    require_all_stages_owned(config, running)
                    hca_before = lifecycle.capture_cluster_hca_counters(pp3_config)
                    for kind in ("prefill", "decode"):
                        workload = collect_owned_workload(
                            client,
                            prepare_glm47_serving_workload(kind),
                            config,
                            running,
                            latch,
                        )
                        workloads.append(
                            cast(JsonObject, workload.model_dump(mode="json"))
                        )
                        summaries.append(lifecycle.summarize_workload(workload))
                    hca_after = lifecycle.capture_cluster_hca_counters(pp3_config)
                    hca_deltas = lifecycle.calculate_hca_deltas(hca_before, hca_after)
                    hca_validation = validate_hca_evidence(hca_deltas)
                    nccl_transport = validate_nccl_log_transport(config)
                    require_all_stages_owned(config, running)
                cancellation_reason = latch.reason
        except BaseException as error:
            failure = error
            cancellation_reason = latch.reason
            if hca_before and not hca_after:
                with contextlib.suppress(BaseException):
                    hca_after = lifecycle.capture_cluster_hca_counters(pp3_config)
                    hca_deltas = lifecycle.calculate_hca_deltas(hca_before, hca_after)
        finally:
            for stage in reversed(running):
                receipt = lifecycle.stop_stage(stage, pp3_config)
                cleanup.append(
                    {"rank": stage.owned.rank, **receipt.model_dump(mode="json")}
                )
            cleanup.sort(
                key=lambda item: cast(int, cast(dict[str, object], item)["rank"])
            )

        cleanup_complete = len(cleanup) == len(running) and all(
            cast(dict[str, object], item)["ownership_verified"] is True
            and cast(dict[str, object], item)["terminated"] is True
            for item in cleanup
        )
        performance_comparable = performance_evidence_is_comparable(
            failure=failure,
            cleanup_complete=cleanup_complete,
            sanity=sanity,
            workloads=workloads,
            hca_validation=hca_validation,
            nccl_transport=nccl_transport,
        )
        payload: JsonObject = {
            "schema_version": 1,
            "kind": "glm47_flash_native_tp3_ep_engineering_benchmark",
            "status": "passed" if failure is None and cleanup_complete else "failed",
            "performance_comparable": performance_comparable,
            "profiler": "none",
            "instrumentation": "nccl_info_logging_and_hca_counters",
            "run_id": config.run_id,
            "mode": config.mode,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "launch_order": [2, 1, 0],
            "rank_zero_only_inference": True,
            "runtime_normalizations": {
                "chunked_prefill_size_launch_global": plan.chunked_prefill_size,
                "chunked_prefill_size_effective_per_dp_rank": (
                    plan.chunked_prefill_size // 3
                ),
                "max_running_requests_launch_global": plan.max_running_requests,
                "max_running_requests_effective_per_dp_rank": (
                    plan.max_running_requests // 3
                ),
            },
            "plan": cast(JsonObject, plan.model_dump(mode="json")),
            "process_specs": [
                {
                    "spec": cast(JsonObject, spec.model_dump(mode="json")),
                    "sha256": calculate_native_glm47_process_spec_sha256(spec),
                }
                for spec in specs
            ],
            "benchmark_protocol": cast(JsonObject, protocol.model_dump(mode="json")),
            "benchmark_protocol_sha256": _canonical_sha256(
                protocol.model_dump(mode="json")
            ),
            "host_inspections": [
                cast(JsonObject, inspection.model_dump(mode="json"))
                for inspection in inspections
            ],
            "port_checks": [
                cast(JsonObject, check.model_dump(mode="json")) for check in port_checks
            ],
            "processes": [_process_receipt(stage) for stage in running],
            "readiness": readiness,
            "server_info": server_info,
            "sanity": sanity,
            "workloads": workloads,
            "benchmark_summary": summaries,
            "hca_counters": {
                "scope": "canonical_token_workloads_after_exact_semantic_sanity",
                "data_counter_unit_bytes": 4,
                "before": {
                    host: cast(JsonObject, snapshot.model_dump(mode="json"))
                    for host, snapshot in hca_before.items()
                },
                "after": {
                    host: cast(JsonObject, snapshot.model_dump(mode="json"))
                    for host, snapshot in hca_after.items()
                },
                "deltas": hca_deltas,
                "validation": hca_validation,
            },
            "nccl_transport": nccl_transport,
            "cleanup": cleanup,
            "cleanup_complete": cleanup_complete,
            "cancellation_reason": cancellation_reason,
            "logs": _log_receipts(config),
            "failure": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        }
        payload["receipt_content_sha256"] = _canonical_sha256(payload)
        _write_result(config, payload)
        if failure is not None:
            raise NativeTpEpBenchmarkError(
                f"native TP/EP benchmark failed; evidence is in "
                f"{config.result_directory}: {type(failure).__name__}: {failure}"
            ) from failure
        if not cleanup_complete:
            raise NativeTpEpBenchmarkError(
                f"native TP/EP cleanup incomplete; evidence is in "
                f"{config.result_directory}"
            )
        return payload


def _safe_identifier(raw: str) -> str:
    if not raw or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in raw
    ):
        raise argparse.ArgumentTypeError("value must be a safe identifier")
    return raw


def _absolute_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _port(raw: str) -> int:
    value = int(raw)
    if not 1024 <= value <= 49_151:
        raise argparse.ArgumentTypeError("fixed port must be in 1024..49151")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _ports(raw: str) -> tuple[int, ...]:
    values = tuple(_port(item) for item in raw.split(","))
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("ports must be nonempty and unique")
    return values


def _hca_devices(raw: str) -> tuple[str, str]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(values) != 2 or len(set(values)) != 2:
        raise argparse.ArgumentTypeError("exactly two distinct HCA rails are required")
    for value in values:
        device, separator, port = value.rpartition(":")
        if not separator or not device or not port.isdigit() or int(port) <= 0:
            raise argparse.ArgumentTypeError(f"invalid HCA selection: {value}")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one owned native TP/EP benchmark")
    run.add_argument("--run-id", type=_safe_identifier, required=True)
    run.add_argument("--mode", choices=("tp3_ep1", "tp3_ep3"), required=True)
    run.add_argument("--result-directory", type=_absolute_path, required=True)
    run.add_argument("--lock-path", type=_absolute_path, default=DEFAULT_LOCK_PATH)
    run.add_argument("--dwagon-runtime-python", default=DEFAULT_DWAGON_RUNTIME)
    run.add_argument("--fwuff-runtime-python", required=True)
    run.add_argument("--dwagon-model-path", default=DEFAULT_DWAGON_MODEL)
    run.add_argument("--fwuff-model-path", default=DEFAULT_FWUFF_MODEL)
    run.add_argument(
        "--dwagon-sglang-source-directory", default=DEFAULT_DWAGON_SGLANG_SOURCE
    )
    run.add_argument("--fwuff-sglang-source-directory", required=True)
    run.add_argument("--dwagon-repository-directory", default=DEFAULT_DWAGON_REPOSITORY)
    run.add_argument("--fwuff-repository-directory", required=True)
    run.add_argument("--dwagon-model-contract")
    run.add_argument("--fwuff-model-contract")
    run.add_argument("--ssh-target", default="fwuff")
    run.add_argument("--dwagon-ip", default=DEFAULT_DWAGON_IP)
    run.add_argument("--fwuff-ip", default=DEFAULT_FWUFF_IP)
    run.add_argument("--dwagon-socket-interface", default=DEFAULT_DWAGON_INTERFACE)
    run.add_argument("--fwuff-socket-interface", default=DEFAULT_FWUFF_INTERFACE)
    run.add_argument("--distributed-port", type=_port, default=DEFAULT_DISTRIBUTED_PORT)
    run.add_argument("--rank-zero-port", type=_port, default=DEFAULT_RANK_PORTS[0])
    run.add_argument("--rank-one-port", type=_port, default=DEFAULT_RANK_PORTS[1])
    run.add_argument("--rank-two-port", type=_port, default=DEFAULT_RANK_PORTS[2])
    run.add_argument("--hca-devices", type=_hca_devices, default=DEFAULT_HCA_DEVICES)
    run.add_argument(
        "--readiness-timeout-seconds", type=_positive_float, default=1800.0
    )
    run.add_argument("--request-timeout-seconds", type=_positive_float, default=900.0)
    run.add_argument("--cleanup-timeout-seconds", type=_positive_float, default=60.0)

    inspect = commands.add_parser("inspect-host", help=argparse.SUPPRESS)
    inspect.add_argument("--node-id", choices=("dwagon", "fwuff"), required=True)
    inspect.add_argument("--repository-directory", type=_absolute_path, required=True)
    inspect.add_argument(
        "--sglang-source-directory", type=_absolute_path, required=True
    )
    inspect.add_argument("--runtime-python", type=_absolute_path, required=True)
    inspect.add_argument("--model-path", type=_absolute_path, required=True)
    inspect.add_argument("--model-contract", type=_absolute_path, required=True)

    check_ports = commands.add_parser("check-ports", help=argparse.SUPPRESS)
    check_ports.add_argument("--host-name", choices=("dwagon", "fwuff"), required=True)
    check_ports.add_argument("--bind-ip", required=True)
    check_ports.add_argument("--ports", type=_ports, required=True)

    probe_owner = commands.add_parser("probe-owner", help=argparse.SUPPRESS)
    probe_owner.add_argument("--receipt-json", required=True)
    return parser


def _contract_path(repository: str) -> str:
    return str(
        Path(repository)
        / "src/exo/worker/sglang_kt/manifests"
        / GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME
    )


def _run_config(arguments: _RunArguments) -> NativeTpEpConfig:
    dwagon_repository = arguments.dwagon_repository_directory
    fwuff_repository = arguments.fwuff_repository_directory
    distributed_port = arguments.distributed_port
    rank_ports = (
        arguments.rank_zero_port,
        arguments.rank_one_port,
        arguments.rank_two_port,
    )
    coordinator_ports = {
        distributed_port + offset
        for offset in GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS
    }
    if max(coordinator_ports) > 49_151:
        raise NativeTpEpBenchmarkError("derived coordinator port exceeds 49151")
    if coordinator_ports.intersection(rank_ports) or len(set(rank_ports)) != 3:
        raise NativeTpEpBenchmarkError(
            "service ports must be unique and outside coordinator ports"
        )
    return NativeTpEpConfig(
        run_id=arguments.run_id,
        mode=cast(Glm47NativeParallelism, arguments.mode),
        result_directory=arguments.result_directory,
        lock_path=arguments.lock_path,
        dwagon_runtime_python=arguments.dwagon_runtime_python,
        fwuff_runtime_python=arguments.fwuff_runtime_python,
        dwagon_model_path=arguments.dwagon_model_path,
        fwuff_model_path=arguments.fwuff_model_path,
        dwagon_sglang_source_directory=arguments.dwagon_sglang_source_directory,
        fwuff_sglang_source_directory=arguments.fwuff_sglang_source_directory,
        dwagon_repository_directory=dwagon_repository,
        fwuff_repository_directory=fwuff_repository,
        dwagon_model_contract=(
            arguments.dwagon_model_contract
            if arguments.dwagon_model_contract is not None
            else _contract_path(dwagon_repository)
        ),
        fwuff_model_contract=(
            arguments.fwuff_model_contract
            if arguments.fwuff_model_contract is not None
            else _contract_path(fwuff_repository)
        ),
        ssh_target=arguments.ssh_target,
        dwagon_ip=arguments.dwagon_ip,
        fwuff_ip=arguments.fwuff_ip,
        dwagon_socket_interface=arguments.dwagon_socket_interface,
        fwuff_socket_interface=arguments.fwuff_socket_interface,
        distributed_port=distributed_port,
        rank_ports=rank_ports,
        hca_devices=arguments.hca_devices,
        readiness_timeout_seconds=arguments.readiness_timeout_seconds,
        request_timeout_seconds=arguments.request_timeout_seconds,
        cleanup_timeout_seconds=arguments.cleanup_timeout_seconds,
    )


def _owned_from_json(raw: str) -> lifecycle.OwnedStageProcess:
    try:
        payload = _object(
            parse_sglang_kt_strict_json(raw.encode()), "owned process receipt"
        )
        expected_fields = {
            "rank",
            "host_name",
            "pid",
            "process_group_id",
            "start_time_ticks",
            "owner_token",
            "ownership_namespace",
            "remote",
            "transport_pid",
            "log_path",
        }
        if set(payload) != expected_fields:
            raise NativeTpEpBenchmarkError(
                "owned process receipt has unexpected fields"
            )
        integer_fields = (
            "rank",
            "pid",
            "process_group_id",
            "start_time_ticks",
            "transport_pid",
        )
        string_fields = (
            "host_name",
            "owner_token",
            "ownership_namespace",
            "log_path",
        )
        if any(type(payload[name]) is not int for name in integer_fields) or any(
            not isinstance(payload[name], str) or not payload[name]
            for name in string_fields
        ):
            raise NativeTpEpBenchmarkError(
                "owned process receipt field types are invalid"
            )
        if type(payload["remote"]) is not bool:
            raise NativeTpEpBenchmarkError("owned process remote flag is invalid")
        return lifecycle.OwnedStageProcess(
            rank=cast(int, payload["rank"]),
            host_name=cast(str, payload["host_name"]),
            pid=cast(int, payload["pid"]),
            process_group_id=cast(int, payload["process_group_id"]),
            start_time_ticks=cast(int, payload["start_time_ticks"]),
            owner_token=cast(str, payload["owner_token"]),
            ownership_namespace=cast(str, payload["ownership_namespace"]),
            remote=payload["remote"],
            transport_pid=cast(int, payload["transport_pid"]),
            log_path=cast(str, payload["log_path"]),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise NativeTpEpBenchmarkError("invalid owned process receipt") from error


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        parsed = _parser().parse_args(arguments)
        command = cast(str, parsed.command)
        if command == "run":
            config = _run_config(cast(_RunArguments, cast(object, parsed)))
            result = run_benchmark(config)
            print(json.dumps(result["benchmark_summary"], indent=2, sort_keys=True))
            print(config.result_directory / RESULT_FILENAME)
        elif command == "inspect-host":
            inspection = inspect_host(
                node_id=NodeId(cast(str, parsed.node_id)),
                repository_directory=cast(Path, parsed.repository_directory),
                sglang_source_directory=cast(Path, parsed.sglang_source_directory),
                runtime_executable=cast(Path, parsed.runtime_python),
                model_path=cast(Path, parsed.model_path),
                model_contract=cast(Path, parsed.model_contract),
            )
            print(inspection.model_dump_json())
        elif command == "check-ports":
            evidence = _check_ports_clear(
                cast(Literal["dwagon", "fwuff"], parsed.host_name),
                cast(str, parsed.bind_ip),
                cast(tuple[int, ...], parsed.ports),
            )
            print(evidence.model_dump_json())
        elif command == "probe-owner":
            owned = _owned_from_json(cast(str, parsed.receipt_json))
            matches, members = lifecycle._local_group_ownership(  # pyright: ignore[reportPrivateUsage]
                owned
            )
            print(
                OwnershipProbe(
                    ownership_matches=matches, members=members
                ).model_dump_json()
            )
        else:
            raise NativeTpEpBenchmarkError(f"unsupported command {command}")
    except (
        NativeTpEpBenchmarkError,
        OSError,
        subprocess.SubprocessError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"native TP/EP benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
