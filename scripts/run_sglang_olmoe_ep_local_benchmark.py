#!/usr/bin/env python3
"""Run the pinned dwagon-local native SGLang OLMoE EP comparison.

This harness intentionally does not use KTransformers. It launches one native
SGLang server with PP1/TP2 and either EP1 (the non-EP control) or EP2 (native
expert-ID sharding). Both ranks are ordered by GPU UUID, communicate over the
dwagon NV4 link, and run under one process group bound to physical CPUs 0-111
whose two ranks are independently CPU- and memory-bound to NUMA nodes 0 and 1.

Admission is fail closed. The caller must provide:

* a canonical SGLang runtime install receipt whose schema-2 build provenance,
  exact patch chain, and all three installed distribution RECORDs still verify;
* a trusted stage contract produced from the pinned Hugging Face snapshot by
  matching TP2/EP1 and TP2/EP2 servers. Admission full-rehashes every snapshot
  file, revalidates Hugging Face revision metadata and local tokenization, and
  requires the exact deterministic EP1/EP2 output equivalence.

An optional external Triton MoE configuration root is admitted as two exact,
hashed RTX 3090 JSON files for the selected EP shape. No caller-provided MoE
configuration environment variable is inherited.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from types import FrameType
from typing import IO, Final, Literal, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    SglangKtReceiptFileError,
    hash_sglang_kt_bound_file,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from scripts.cpu_performance_policy import (  # noqa: E402
    CpuPerformancePolicySession,
    cpu_performance_policy,
)
from scripts.create_sglang_olmoe_stage_contract import (  # noqa: E402
    OLMOE_MODEL_PATH,
    LoadedStageContract,
    ManagedLaunchEvidence,
    SanityCapture,
    collect_sanity_capture,
    load_stage_contract,
    publish_sanity_capture,
    verify_pinned_snapshot,
)
from scripts.install_sglang_kt_runtime import (  # noqa: E402
    EXPECTED_DISTRIBUTIONS,
    RuntimeInstallError,
    plan_runtime_install,
)
from scripts.prepare_sglang_kt_source import (  # noqa: E402
    SglangKtSourcePlan,
)
from scripts.sglang_olmoe_serving_client import (  # noqa: E402
    GenerateObservation,
    LogitParityObservation,
    OlmoeLogitParityRequest,
    OlmoeNativeAsyncServingClient,
    OlmoeNativeGenerateRequest,
    OlmoeNativeServingClient,
    OlmoeSamplingParameters,
    token_ids_sha256,
)
from scripts.validate_sglang_kt_runtime import (  # noqa: E402
    RuntimeValidationError,
    observe_build_provenance,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ExpertParallelSize = Literal[1, 2]
type NumaMappingClass = Literal[
    "model_weight",
    "private_anonymous",
    "regular_file",
    "regular_file_private",
    "special_or_shared",
]

OLMOE_MODEL_ID: Final = "allenai/OLMoE-1B-7B-0924"
OLMOE_MODEL_REVISION: Final = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
OLMOE_SGLANG_REVISION: Final = GLM_4_7_FLASH_SGLANG_REVISION
OLMOE_VOCABULARY_SIZE: Final = 50_304
OLMOE_EXPERT_COUNT: Final = 64
OLMOE_EXPERTS_PER_TOKEN: Final = 8
OLMOE_HIDDEN_SIZE: Final = 2_048
OLMOE_CONTEXT_LENGTH: Final = 4_096
OLMOE_MAX_TOTAL_TOKENS: Final = 4_096
OLMOE_CONCURRENCY_MAX_TOTAL_TOKENS: Final = 9_216
OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS: Final = 8
OLMOE_CONCURRENCY_CUDA_GRAPH_MAX_BATCH_SIZE: Final = 8
DWAGON_GPU_UUIDS: Final = (
    "GPU-63a7760a-6164-0758-9228-03dbf35d721c",
    "GPU-a442b72e-6727-6322-ba5d-5a9512b79886",
)
DWAGON_PHYSICAL_CPUS: Final = tuple(range(112))
DWAGON_NUMA_NODES: Final = (0, 1)
DWAGON_NUMA_LOGICAL_CPUS: Final = (
    frozenset((*range(56), *range(112, 168))),
    frozenset((*range(56, 112), *range(168, 224))),
)
CANONICAL_WARMUP_COUNT: Final = 2
CANONICAL_SAMPLE_COUNT: Final = 3
CANONICAL_SAMPLING_SEED: Final = 2_026_0720
CANONICAL_WORKLOADS: Final = (
    ("prefill", 1_024, 32),
    ("decode", 128, 128),
)
CONCURRENCY_LEVELS: Final = (1, 2, 4, 8)
LOGIT_PARITY_CANONICAL_INPUT_TOKENS: Final = 128
LOGIT_PARITY_COMMON_PREFIX: Final = (431, 3_056, 209)
LOGIT_PARITY_CANDIDATE_TOKEN_IDS: Final = (139, 1_769)
LOGIT_PARITY_TOP_LOGPROBS: Final = 8
DEFAULT_PORT: Final = 62_610
DEFAULT_STATIC_MEMORY_FRACTION: Final = 0.9
_RECEIPT_MAXIMUM_BYTES: Final = 4 * 1024 * 1024
_MOE_CONFIG_MAXIMUM_BYTES: Final = 64 * 1024
_MOE_CONFIG_MAXIMUM_ENTRIES: Final = 256
_IDENTITY_FILE_MAXIMUM_BYTES: Final = 64 * 1024 * 1024
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024
_PROC_MAPS_MAXIMUM_BYTES: Final = 16 * 1024 * 1024
_PROC_SMAPS_MAXIMUM_BYTES: Final = 64 * 1024 * 1024
_OWNERSHIP_JOURNAL_FILENAME: Final = "olmoe-ep-ownership-journal.json"
_RESULT_FILENAME: Final = "olmoe-ep-local-benchmark-result.json"
_SERVER_LOG_FILENAME: Final = "native-sglang-server.log"
_MOE_CONFIG_SNAPSHOT_DIRECTORY_NAME: Final = "effective-moe-config"
_MOE_CONFIG_SNAPSHOT_DIRECTORY_MODE: Final = 0o500
_MOE_CONFIG_SNAPSHOT_FILE_MODE: Final = 0o400
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)
_SAFE_RUN_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MOE_CONFIG_BATCH_SIZE_PATTERN: Final = re.compile(r"[1-9][0-9]{0,4}", re.ASCII)
_MOE_CONFIG_TRITON_VERSION: Final = "3.5.1"
_MOE_CONFIG_VERSION_DIRECTORY: Final = "triton_3_5_1"
_MOE_CONFIG_DEVICE_NAME: Final = "NVIDIA_GeForce_RTX_3090"
_MOE_CONFIG_KEYS: Final = frozenset(
    {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
)
_MOE_CONFIG_ALLOWED_VALUES: Final[dict[str, frozenset[int]]] = {
    "BLOCK_SIZE_M": frozenset({16, 32, 64, 128, 256}),
    "BLOCK_SIZE_N": frozenset({16, 32, 64, 128, 256}),
    "BLOCK_SIZE_K": frozenset({32, 64, 128, 256}),
    "GROUP_SIZE_M": frozenset({1, 2, 4, 8, 16, 32, 64, 128}),
    "num_warps": frozenset({1, 2, 4, 8}),
    "num_stages": frozenset({1, 2, 3, 4, 5}),
}
_KV_CACHE_ALLOCATION_PATTERN: Final = re.compile(
    r"\[(?:[^\]\r\n]* )?TP(?P<tp_rank>[01])"
    r"(?: EP(?P<ep_rank>[01]))?\] KV Cache is allocated\. "
    r"#tokens: (?P<token_count>4096|9216)(?:,|$)",
    re.ASCII,
)
_DISABLED_CUSTOM_ALL_REDUCE_SERVER_ARGS_PATTERN: Final = re.compile(
    r"\bdisable_custom_all_reduce=True\b", re.ASCII
)
_NUMA_MAP_ADDRESS_PATTERN: Final = re.compile(r"[0-9a-f]+", re.ASCII)
_NUMA_MAP_NODE_PAGES_PATTERN: Final = re.compile(
    r"N(?P<node>[0-9]+)=(?P<pages>[0-9]+)", re.ASCII
)
_MAPS_HEADER_PATTERN: Final = re.compile(
    r"(?P<start>[0-9a-f]+)-(?P<end>[0-9a-f]+) "
    r"(?P<permissions>[rwxps-]{4}) (?P<offset>[0-9a-f]+) "
    r"(?P<device>[0-9a-f]+:[0-9a-f]+) (?P<inode>[0-9]+)"
    r"(?: +(?P<path>.*))?",
    re.ASCII,
)
_MODEL_WEIGHT_SUFFIXES: Final = (".safetensors", ".bin", ".gguf", ".pt", ".pth")
_SPECIAL_NUMA_FILE_PREFIXES: Final = (
    "/dev/",
    "/SYSV",
    "SYSV",
    "/memfd:",
    "memfd:",
)
_CUDA_VMM_RESERVATION_VM_FLAGS: Final = frozenset(
    {"rd", "wr", "mr", "mw", "me", "ac", "sd"}
)
_SCHEDULER_TITLE_PATTERN: Final = re.compile(
    r"sglang::scheduler_TP(?P<tp_rank>[01])(?:_EP(?P<ep_rank>[01]))?", re.ASCII
)
_INSTALL_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "install_id",
        "installer_sha256",
        "build",
        "base_runtime",
        "layout",
        "environment",
        "commands",
        "installed_distributions",
        "completed_at_utc",
    }
)
_OWNER_TOKEN_ENVIRONMENT: Final = "EXO_OLMOE_EP_OWNER_TOKEN"
_OWNERSHIP_NAMESPACE_ENVIRONMENT: Final = "EXO_OLMOE_EP_NAMESPACE"
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_CPU_SYSFS_ROOT: Final = Path("/sys/devices/system")
_HARNESS_PATH: Final = Path(__file__).resolve()
_SERVING_CLIENT_PATH: Final = _HARNESS_PATH.parent / "sglang_olmoe_serving_client.py"
_STAGE_CONTRACT_PRODUCER_PATH: Final = (
    _HARNESS_PATH.parent / "create_sglang_olmoe_stage_contract.py"
)
_CPU_POLICY_HELPER_PATH: Final = _HARNESS_PATH.parent / "cpu_performance_policy.py"


class OlmoeEpBenchmarkError(RuntimeError):
    """Raised when admission or benchmark evidence is incomplete."""


class OlmoeEpManagedSignalError(OlmoeEpBenchmarkError):
    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received managed signal {signal_number}")
        self.signal_number = signal_number


@dataclass(frozen=True, slots=True)
class InstalledDistributionAdmission:
    distribution: str
    version: str
    record_sha256: str
    verified_record_file_count: int


@dataclass(frozen=True, slots=True)
class RuntimeAdmission:
    install_receipt_path: str
    install_receipt_sha256: str
    install_receipt_size_bytes: int
    install_id: str
    install_root: str
    runtime_python: str
    resolved_python: str
    resolved_python_sha256: str
    build_receipt_path: str
    build_receipt_sha256: str
    sglang_revision: str
    build_provenance_verified: bool
    patch_stack_sha256: tuple[str, ...]
    installed_distributions: tuple[InstalledDistributionAdmission, ...]
    verified_record_file_count: int


@dataclass(frozen=True, slots=True)
class MoeConfigFileAdmission:
    relative_path: str
    size_bytes: int
    sha256: str
    canonical_sha256: str
    entry_count: int
    batch_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MoeConfigAdmission:
    root: str
    triton_version: str
    version_directory: str
    device_name: str
    expert_parallel_size: ExpertParallelSize
    local_expert_count: int
    moe_intermediate_size: int
    files: tuple[MoeConfigFileAdmission, ...]
    file_set_sha256: str


@dataclass(frozen=True, slots=True)
class MoeConfigSnapshotAdmission:
    source: MoeConfigAdmission
    effective: MoeConfigAdmission
    directory_paths: tuple[str, ...]
    directory_mode: int
    file_mode: int
    exact_tree_verified: bool
    byte_for_byte_copy_verified: bool


@dataclass(frozen=True, slots=True)
class OlmoeEpBenchmarkConfig:
    run_id: str
    result_directory: Path
    runtime_python: str
    runtime_install_receipt: Path
    runtime_install_receipt_sha256: str
    model_path: str
    stage_contract: Path | None
    stage_capture_output: Path | None
    expert_parallel_size: ExpertParallelSize
    host: str
    port: int
    static_memory_fraction: float
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float
    numactl_executable: str
    nvidia_smi_executable: str
    logit_parity_probe: bool = False
    moe_config_root: Path | None = None
    aggregate_concurrency: bool = False


@dataclass(frozen=True, slots=True)
class OwnedServerProcess:
    pid: int
    process_group_id: int
    start_time_ticks: int
    owner_token: str
    ownership_namespace: str
    command: tuple[str, ...]
    launch_environment: tuple[tuple[str, str], ...]
    log_path: str


@dataclass(slots=True)
class RunningServerProcess:
    owned: OwnedServerProcess
    process: subprocess.Popen[bytes]
    log_file: IO[bytes]


@dataclass(frozen=True, slots=True)
class ProcessStatIdentity:
    parent_pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int
    state: str


@dataclass(slots=True)
class _ManagedSignalState:
    signal_number: int | None = None
    cleanup_started: bool = False

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number
        if not self.cleanup_started:
            raise OlmoeEpManagedSignalError(self.signal_number)

    def checkpoint(self) -> None:
        if self.signal_number is not None and not self.cleanup_started:
            raise OlmoeEpManagedSignalError(self.signal_number)

    def begin_cleanup(self) -> None:
        self.cleanup_started = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OlmoeEpBenchmarkError(f"{description} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise OlmoeEpBenchmarkError(f"{description} has a non-string key")
    return cast(dict[str, object], mapping)


def _strict_keys(
    value: Mapping[str, object], expected: frozenset[str], description: str
) -> None:
    if frozenset(value) != expected:
        raise OlmoeEpBenchmarkError(f"{description} does not match its exact schema")


def _required_string(values: Mapping[str, object], key: str, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value or len(value) > 16_384:
        raise OlmoeEpBenchmarkError(f"{description}.{key} must be a bounded string")
    return value


def _required_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise OlmoeEpBenchmarkError(f"{description} must be lowercase SHA-256")
    return value


def _moe_config_shape(
    expert_parallel_size: ExpertParallelSize,
) -> tuple[int, int]:
    if expert_parallel_size == 1:
        return 64, 512
    return 32, 1_024


def _moe_config_relative_paths(
    expert_parallel_size: ExpertParallelSize,
) -> tuple[Path, Path]:
    local_expert_count, moe_intermediate_size = _moe_config_shape(expert_parallel_size)
    stem = (
        f"E={local_expert_count},N={moe_intermediate_size},"
        f"device_name={_MOE_CONFIG_DEVICE_NAME}"
    )
    directory = Path("configs") / _MOE_CONFIG_VERSION_DIRECTORY
    return directory / f"{stem}.json", directory / f"{stem}_down.json"


def _read_moe_config_file(
    root: Path,
    relative_path: Path,
    moe_intermediate_size: int,
) -> tuple[MoeConfigFileAdmission, dict[int, dict[str, int]]]:
    path = root / relative_path
    description = f"Triton MoE config {relative_path}"
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=_MOE_CONFIG_MAXIMUM_BYTES)
        parsed = parse_sglang_kt_strict_json(bound.contents)
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError(f"cannot admit {description}: {error}") from error
    document = _json_object(parsed, description)
    if not 1 <= len(document) <= _MOE_CONFIG_MAXIMUM_ENTRIES:
        raise OlmoeEpBenchmarkError(
            f"{description} must contain 1-{_MOE_CONFIG_MAXIMUM_ENTRIES} entries"
        )

    configurations: dict[int, dict[str, int]] = {}
    for raw_batch_size, raw_configuration in document.items():
        if _MOE_CONFIG_BATCH_SIZE_PATTERN.fullmatch(raw_batch_size) is None:
            raise OlmoeEpBenchmarkError(
                f"{description} batch-size key is not canonical"
            )
        batch_size = int(raw_batch_size)
        if batch_size > OLMOE_MAX_TOTAL_TOKENS:
            raise OlmoeEpBenchmarkError(
                f"{description} batch-size key exceeds the token pool"
            )
        configuration = _json_object(
            raw_configuration, f"{description}[{raw_batch_size}]"
        )
        _strict_keys(
            configuration,
            _MOE_CONFIG_KEYS,
            f"{description}[{raw_batch_size}]",
        )
        admitted: dict[str, int] = {}
        for name, allowed_values in _MOE_CONFIG_ALLOWED_VALUES.items():
            value = configuration.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise OlmoeEpBenchmarkError(
                    f"{description}[{raw_batch_size}].{name} must be an integer"
                )
            if value not in allowed_values:
                raise OlmoeEpBenchmarkError(
                    f"{description}[{raw_batch_size}].{name} is not admitted"
                )
            admitted[name] = value
        if moe_intermediate_size % admitted["BLOCK_SIZE_N"] != 0:
            raise OlmoeEpBenchmarkError(
                f"{description}[{raw_batch_size}].BLOCK_SIZE_N does not tile N"
            )
        if OLMOE_HIDDEN_SIZE % admitted["BLOCK_SIZE_K"] != 0:
            raise OlmoeEpBenchmarkError(
                f"{description}[{raw_batch_size}].BLOCK_SIZE_K does not tile K"
            )
        configurations[batch_size] = admitted

    batch_sizes = tuple(sorted(configurations))
    return (
        MoeConfigFileAdmission(
            relative_path=relative_path.as_posix(),
            size_bytes=len(bound.contents),
            sha256=bound.sha256,
            canonical_sha256=_canonical_sha256(cast(JsonValue, document)),
            entry_count=len(configurations),
            batch_sizes=batch_sizes,
        ),
        configurations,
    )


def _verify_moe_config_root(
    root: Path, expert_parallel_size: ExpertParallelSize
) -> MoeConfigAdmission:
    if (
        not root.is_absolute()
        or ".." in root.parts
        or "\0" in str(root)
        or root != Path(os.path.normpath(root))
    ):
        raise OlmoeEpBenchmarkError(
            "MoE config root must be an absolute lexical directory"
        )

    local_expert_count, moe_intermediate_size = _moe_config_shape(expert_parallel_size)
    relative_paths = _moe_config_relative_paths(expert_parallel_size)
    files: list[MoeConfigFileAdmission] = []
    maps: list[dict[int, dict[str, int]]] = []
    for relative_path in relative_paths:
        file_admission, configurations = _read_moe_config_file(
            root, relative_path, moe_intermediate_size
        )
        files.append(file_admission)
        maps.append(configurations)

    normal_configurations, down_configurations = maps
    if normal_configurations.keys() != down_configurations.keys():
        raise OlmoeEpBenchmarkError(
            "normal and down Triton MoE configs must have the same batch-size grid"
        )
    for batch_size in normal_configurations:
        if (
            normal_configurations[batch_size]["BLOCK_SIZE_M"]
            != down_configurations[batch_size]["BLOCK_SIZE_M"]
        ):
            raise OlmoeEpBenchmarkError(
                "normal and down Triton MoE configs must match BLOCK_SIZE_M"
            )

    file_set_payload = cast(
        JsonObject,
        {
            "expert_parallel_size": expert_parallel_size,
            "local_expert_count": local_expert_count,
            "moe_intermediate_size": moe_intermediate_size,
            "files": [
                {
                    "relative_path": item.relative_path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "canonical_sha256": item.canonical_sha256,
                    "entry_count": item.entry_count,
                    "batch_sizes": list(item.batch_sizes),
                }
                for item in files
            ],
        },
    )
    return MoeConfigAdmission(
        root=str(root),
        triton_version=_MOE_CONFIG_TRITON_VERSION,
        version_directory=_MOE_CONFIG_VERSION_DIRECTORY,
        device_name=_MOE_CONFIG_DEVICE_NAME,
        expert_parallel_size=expert_parallel_size,
        local_expert_count=local_expert_count,
        moe_intermediate_size=moe_intermediate_size,
        files=tuple(files),
        file_set_sha256=_canonical_sha256(file_set_payload),
    )


def verify_moe_config(config: OlmoeEpBenchmarkConfig) -> MoeConfigAdmission | None:
    """Admit the exact external RTX 3090 config pair selected for this run."""

    if config.moe_config_root is None:
        return None
    return _verify_moe_config_root(config.moe_config_root, config.expert_parallel_size)


def _write_moe_config_snapshot_file(path: Path, contents: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot create private MoE config snapshot file: {error}"
        ) from error
    try:
        view = memoryview(contents)
        written = 0
        while written < len(contents):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OlmoeEpBenchmarkError(
                    "private MoE config snapshot write made no progress"
                )
            written += count
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != os.geteuid()
            or observed.st_size != len(contents)
        ):
            raise OlmoeEpBenchmarkError(
                "private MoE config snapshot file identity is invalid"
            )
        os.fchmod(descriptor, _MOE_CONFIG_SNAPSHOT_FILE_MODE)
        os.fsync(descriptor)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot publish private MoE config snapshot file: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _moe_config_snapshot_directories(root: Path) -> tuple[Path, Path, Path]:
    configs = root / "configs"
    version = configs / _MOE_CONFIG_VERSION_DIRECTORY
    return root, configs, version


def _verify_moe_config_snapshot_tree(
    root: Path, expert_parallel_size: ExpertParallelSize
) -> tuple[str, ...]:
    directories = _moe_config_snapshot_directories(root)
    expected_entries = (
        frozenset({"configs"}),
        frozenset({_MOE_CONFIG_VERSION_DIRECTORY}),
        frozenset(
            path.name for path in _moe_config_relative_paths(expert_parallel_size)
        ),
    )
    for directory, expected_names in zip(directories, expected_entries, strict=True):
        try:
            observed = directory.lstat()
            names = frozenset(entry.name for entry in os.scandir(directory))
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot verify private MoE config snapshot tree: {error}"
            ) from error
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != _MOE_CONFIG_SNAPSHOT_DIRECTORY_MODE
            or observed.st_uid != os.geteuid()
            or names != expected_names
        ):
            raise OlmoeEpBenchmarkError(
                "private MoE config snapshot directory is not exact and read-only"
            )

    for relative_path in _moe_config_relative_paths(expert_parallel_size):
        path = root / relative_path
        try:
            observed = path.lstat()
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot verify private MoE config snapshot file: {error}"
            ) from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != _MOE_CONFIG_SNAPSHOT_FILE_MODE
            or observed.st_uid != os.geteuid()
        ):
            raise OlmoeEpBenchmarkError(
                "private MoE config snapshot file is not exact and read-only"
            )
    return tuple(str(path) for path in directories)


def create_moe_config_snapshot(
    config: OlmoeEpBenchmarkConfig,
    source_admission: MoeConfigAdmission | None,
) -> MoeConfigSnapshotAdmission | None:
    """Copy the admitted pair into one immutable, harness-owned launch root."""

    if source_admission is None:
        if config.moe_config_root is not None:
            raise OlmoeEpBenchmarkError("external MoE config was not admitted")
        return None
    if config.moe_config_root is None:
        raise OlmoeEpBenchmarkError("MoE config admission has no source root")
    freshly_admitted = verify_moe_config(config)
    if freshly_admitted != source_admission:
        raise OlmoeEpBenchmarkError("external MoE config changed before snapshot")

    source_contents: dict[str, bytes] = {}
    for file_admission in source_admission.files:
        source_path = Path(source_admission.root) / file_admission.relative_path
        try:
            bound = read_sglang_kt_bound_file(
                source_path, maximum_bytes=_MOE_CONFIG_MAXIMUM_BYTES
            )
        except SglangKtReceiptFileError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot reread admitted MoE config source: {error}"
            ) from error
        if (
            len(bound.contents) != file_admission.size_bytes
            or bound.sha256 != file_admission.sha256
        ):
            raise OlmoeEpBenchmarkError("external MoE config changed before snapshot")
        source_contents[file_admission.relative_path] = bound.contents
    if verify_moe_config(config) != source_admission:
        raise OlmoeEpBenchmarkError("external MoE config changed during snapshot read")

    snapshot_root = config.result_directory / _MOE_CONFIG_SNAPSHOT_DIRECTORY_NAME
    root, configs_directory, version_directory = _moe_config_snapshot_directories(
        snapshot_root
    )
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        configs_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        version_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot create private MoE config snapshot tree: {error}"
        ) from error

    for file_admission in source_admission.files:
        _write_moe_config_snapshot_file(
            snapshot_root / file_admission.relative_path,
            source_contents[file_admission.relative_path],
        )
    for directory in reversed((root, configs_directory, version_directory)):
        _fsync_directory(directory)
    try:
        for directory in (version_directory, configs_directory, root):
            os.chmod(
                directory,
                _MOE_CONFIG_SNAPSHOT_DIRECTORY_MODE,
                follow_symlinks=False,
            )
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot make private MoE config snapshot read-only: {error}"
        ) from error
    for directory in (version_directory, configs_directory, root):
        _fsync_directory(directory)
    _fsync_directory(config.result_directory)

    effective_admission = _verify_moe_config_root(
        snapshot_root, config.expert_parallel_size
    )
    if effective_admission.file_set_sha256 != source_admission.file_set_sha256:
        raise OlmoeEpBenchmarkError(
            "private MoE config snapshot differs from admitted source"
        )
    directory_paths = _verify_moe_config_snapshot_tree(
        snapshot_root, config.expert_parallel_size
    )
    return MoeConfigSnapshotAdmission(
        source=source_admission,
        effective=effective_admission,
        directory_paths=directory_paths,
        directory_mode=_MOE_CONFIG_SNAPSHOT_DIRECTORY_MODE,
        file_mode=_MOE_CONFIG_SNAPSHOT_FILE_MODE,
        exact_tree_verified=True,
        byte_for_byte_copy_verified=True,
    )


def verify_moe_config_snapshot(
    snapshot: MoeConfigSnapshotAdmission,
) -> MoeConfigSnapshotAdmission:
    root = Path(snapshot.effective.root)
    effective = _verify_moe_config_root(root, snapshot.effective.expert_parallel_size)
    directory_paths = _verify_moe_config_snapshot_tree(
        root, snapshot.effective.expert_parallel_size
    )
    observed = MoeConfigSnapshotAdmission(
        source=snapshot.source,
        effective=effective,
        directory_paths=directory_paths,
        directory_mode=_MOE_CONFIG_SNAPSHOT_DIRECTORY_MODE,
        file_mode=_MOE_CONFIG_SNAPSHOT_FILE_MODE,
        exact_tree_verified=True,
        byte_for_byte_copy_verified=(
            effective.file_set_sha256 == snapshot.source.file_set_sha256
        ),
    )
    if observed != snapshot or not observed.byte_for_byte_copy_verified:
        raise OlmoeEpBenchmarkError("private MoE config snapshot changed")
    return observed


def _load_bound_json(
    path: Path, expected_sha256: str, description: str
) -> tuple[dict[str, object], int]:
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES)
        parsed = parse_sglang_kt_strict_json(bound.contents)
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError(f"cannot bind {description}: {error}") from error
    if bound.sha256 != expected_sha256:
        raise OlmoeEpBenchmarkError(f"{description} SHA-256 changed")
    return _json_object(parsed, description), len(bound.contents)


def verify_stage_contract(config: OlmoeEpBenchmarkConfig) -> LoadedStageContract:
    if config.model_path != OLMOE_MODEL_PATH:
        raise OlmoeEpBenchmarkError("model path is not the pinned OLMoE snapshot")
    if config.stage_contract is None:
        raise OlmoeEpBenchmarkError("benchmark mode requires a stage contract")
    try:
        loaded = load_stage_contract(
            config.stage_contract, snapshot_path=Path(config.model_path)
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise OlmoeEpBenchmarkError(
            f"cannot admit the trusted OLMoE stage contract: {error}"
        ) from error
    capture = (
        loaded.contract.ep1.capture
        if config.expert_parallel_size == 1
        else loaded.contract.ep2.capture
    )
    if (
        capture.runtime_install_receipt_sha256 != config.runtime_install_receipt_sha256
        or capture.static_memory_fraction != config.static_memory_fraction
        or capture.server_host != config.host
        or capture.server_port != config.port
    ):
        raise OlmoeEpBenchmarkError(
            "stage contract does not bind this runtime and memory configuration"
        )
    return loaded


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _verify_distribution_record(
    *, install_root: Path, site_packages: Path, distribution: Mapping[str, object]
) -> tuple[str, str, str, int]:
    _strict_keys(
        distribution,
        frozenset({"distribution", "version", "metadata_path", "record_sha256"}),
        "installed distribution",
    )
    name = _normalized_distribution(
        _required_string(distribution, "distribution", "installed distribution")
    )
    version = _required_string(distribution, "version", "installed distribution")
    metadata_path = Path(
        _required_string(distribution, "metadata_path", "installed distribution")
    )
    record_sha256 = _required_sha256(
        distribution.get("record_sha256"), "installed distribution RECORD digest"
    )
    try:
        resolved_install_root = install_root.resolve(strict=True)
        resolved_site_packages = site_packages.resolve(strict=True)
        resolved_metadata = metadata_path.resolve(strict=True)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"installed distribution path is unavailable: {error}"
        ) from error
    if (
        resolved_install_root != install_root
        or resolved_site_packages != site_packages
        or resolved_metadata != metadata_path
        or metadata_path.name != "METADATA"
        or not metadata_path.is_relative_to(site_packages)
        or not metadata_path.is_relative_to(install_root)
    ):
        raise OlmoeEpBenchmarkError("installed distribution escapes install root")
    try:
        metadata = read_sglang_kt_bound_file(
            metadata_path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES
        )
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError(
            "installed distribution metadata changed"
        ) from error
    parsed_metadata = BytesParser().parsebytes(metadata.contents)
    if (
        _normalized_distribution(str(parsed_metadata["Name"] or "")) != name
        or str(parsed_metadata["Version"] or "") != version
    ):
        raise OlmoeEpBenchmarkError("installed distribution metadata changed")
    record_path = metadata_path.parent / "RECORD"
    try:
        record = read_sglang_kt_bound_file(
            record_path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES
        )
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError("installed distribution RECORD changed") from error
    if record.sha256 != record_sha256:
        raise OlmoeEpBenchmarkError("installed distribution RECORD changed")
    try:
        rows = tuple(csv.reader(record.contents.decode("utf-8").splitlines()))
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError(
            "installed distribution RECORD is not UTF-8"
        ) from error
    verified: set[Path] = set()
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise OlmoeEpBenchmarkError("installed distribution RECORD row is invalid")
        recorded_path, recorded_hash, recorded_size = row
        relative = Path(recorded_path)
        if relative.is_absolute():
            raise OlmoeEpBenchmarkError("installed distribution RECORD path escapes")
        if relative.parts[:3] == ("..", "..", "bin"):
            lexical_candidate = site_packages / "bin" / Path(*relative.parts[3:])
        else:
            if ".." in relative.parts:
                raise OlmoeEpBenchmarkError(
                    "installed distribution RECORD path escapes"
                )
            lexical_candidate = site_packages / relative
        candidate = lexical_candidate.resolve(strict=True)
        if (
            not candidate.is_relative_to(install_root)
            or not candidate.is_file()
            or candidate in verified
        ):
            raise OlmoeEpBenchmarkError("installed distribution RECORD path is invalid")
        verified.add(candidate)
        expected_size: int | None = None
        if recorded_size:
            try:
                expected_size = int(recorded_size)
            except ValueError as error:
                raise OlmoeEpBenchmarkError(
                    "installed RECORD size is invalid"
                ) from error
            if expected_size < 0:
                raise OlmoeEpBenchmarkError("installed RECORD size is negative")
        try:
            observed = hash_sglang_kt_bound_file(
                candidate, expected_size_bytes=expected_size
            )
        except SglangKtReceiptFileError as error:
            raise OlmoeEpBenchmarkError(
                f"installed runtime file changed: {recorded_path}: {error}"
            ) from error
        if recorded_hash:
            algorithm, separator, encoded = recorded_hash.partition("=")
            if algorithm != "sha256" or not separator:
                raise OlmoeEpBenchmarkError("installed RECORD hash is unsupported")
            try:
                expected_digest = base64.b64decode(
                    encoded + "=" * (-len(encoded) % 4),
                    altchars=b"-_",
                    validate=True,
                ).hex()
            except (TypeError, ValueError) as error:
                raise OlmoeEpBenchmarkError(
                    "installed RECORD hash is invalid"
                ) from error
            if observed.sha256 != expected_digest:
                raise OlmoeEpBenchmarkError(
                    f"installed runtime file changed: {recorded_path}"
                )
        elif candidate != record_path:
            raise OlmoeEpBenchmarkError("installed RECORD omits a file digest")
    if metadata_path not in verified or record_path not in verified:
        raise OlmoeEpBenchmarkError("installed distribution RECORD is incomplete")
    return name, version, record.sha256, len(verified)


def _json_compatible(value: object) -> object:
    return cast(object, json.loads(json.dumps(value, allow_nan=False, sort_keys=True)))


def _verify_patch_stack(
    ktransformers_revision: str, sglang_revision: str
) -> tuple[str, ...]:
    plan = SglangKtSourcePlan.exo_default()
    patches = (
        plan.sglang_patch,
        plan.ktransformers_patch,
        *plan.sglang_followup_patches,
    )
    if (
        plan.ktransformers_result_revision != ktransformers_revision
        or plan.sglang_result_revision != sglang_revision
        or plan.sglang_patch.base_revision != plan.sglang_base_revision
        or plan.sglang_patch.result_revision != plan.sglang_gitlink_revision
        or len(plan.sglang_followup_patches) != 2
        or plan.sglang_followup_patches[0].base_revision != plan.sglang_gitlink_revision
        or plan.sglang_followup_patches[0].result_revision
        != plan.sglang_followup_patches[1].base_revision
        or plan.sglang_followup_patches[1].result_revision
        != plan.sglang_result_revision
    ):
        raise OlmoeEpBenchmarkError("runtime patch chain is not the pinned chain")
    observed_hashes: list[str] = []
    for patch in patches:
        try:
            observed = hash_sglang_kt_bound_file(patch.path)
        except SglangKtReceiptFileError as error:
            raise OlmoeEpBenchmarkError(
                f"runtime patch is unavailable: {patch.path}: {error}"
            ) from error
        if observed.sha256 != patch.sha256:
            raise OlmoeEpBenchmarkError(f"runtime patch changed: {patch.path}")
        observed_hashes.append(observed.sha256)
    return tuple(observed_hashes)


def verify_runtime_install(config: OlmoeEpBenchmarkConfig) -> RuntimeAdmission:
    document, receipt_size = _load_bound_json(
        config.runtime_install_receipt,
        config.runtime_install_receipt_sha256,
        "SGLang runtime install receipt",
    )
    _strict_keys(document, _INSTALL_RECEIPT_KEYS, "SGLang runtime install receipt")
    if (
        document.get("schema_version") != 1
        or document.get("status") != "install_complete"
    ):
        raise OlmoeEpBenchmarkError(
            "runtime receipt is not a completed schema-1 install"
        )
    completed_at = document.get("completed_at_utc")
    if not isinstance(completed_at, str):
        raise OlmoeEpBenchmarkError("runtime receipt lacks a completion timestamp")
    try:
        completion_time = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise OlmoeEpBenchmarkError(
            "runtime receipt completion timestamp is invalid"
        ) from error
    if completion_time.utcoffset() is None:
        raise OlmoeEpBenchmarkError(
            "runtime receipt completion timestamp lacks a timezone"
        )

    install_id = _required_sha256(document.get("install_id"), "runtime install ID")
    build = _json_object(document.get("build"), "runtime install build")
    layout = _json_object(document.get("layout"), "runtime install layout")
    base_runtime = _json_object(document.get("base_runtime"), "runtime base")
    revision = _required_string(build, "sglang_revision", "runtime install build")
    if revision != OLMOE_SGLANG_REVISION:
        raise OlmoeEpBenchmarkError("runtime does not contain the pinned OLMoE TP fix")

    receipt_path = Path(_required_string(layout, "receipt", "runtime install layout"))
    install_root = Path(
        _required_string(layout, "install_root", "runtime install layout")
    )
    runtime_python = Path(_required_string(layout, "python", "runtime install layout"))
    site_packages = Path(
        _required_string(layout, "site_packages", "runtime install layout")
    )
    output_root = Path(
        _required_string(layout, "output_root", "runtime install layout")
    )
    base_python = Path(_required_string(base_runtime, "python_path", "runtime base"))
    base_site_packages = Path(
        _required_string(base_runtime, "site_packages", "runtime base")
    )
    if (
        receipt_path != config.runtime_install_receipt
        or receipt_path.parent != install_root
        or install_root.name != install_id
        or runtime_python != Path(config.runtime_python)
        or not site_packages.is_relative_to(install_root)
    ):
        raise OlmoeEpBenchmarkError("runtime install layout does not bind this launch")

    build_receipt_path = Path(
        _required_string(build, "receipt_path", "runtime install build")
    )
    expected_build_sha256 = _required_sha256(
        build.get("receipt_sha256"), "runtime build receipt digest"
    )
    try:
        bound_build_receipt = read_sglang_kt_bound_file(
            build_receipt_path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES
        )
        if bound_build_receipt.sha256 != expected_build_sha256:
            raise OlmoeEpBenchmarkError("runtime build receipt changed")
        provenance = observe_build_provenance(build_receipt_path)
        plan = plan_runtime_install(
            build_receipt_path, base_python, base_site_packages, output_root
        )
        rebound_build_receipt = read_sglang_kt_bound_file(
            build_receipt_path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES
        )
    except (
        OSError,
        RuntimeInstallError,
        RuntimeValidationError,
        SglangKtReceiptFileError,
    ) as error:
        raise OlmoeEpBenchmarkError(
            f"runtime provenance validation failed: {error}"
        ) from error
    if (
        provenance.receipt_sha256 != expected_build_sha256
        or provenance.sglang_revision != OLMOE_SGLANG_REVISION
        or not provenance.verified
        or rebound_build_receipt.sha256 != bound_build_receipt.sha256
    ):
        raise OlmoeEpBenchmarkError("runtime build provenance is not admitted")
    planned = _json_object(_json_compatible(asdict(plan)), "runtime install plan")
    for key in (
        "install_id",
        "installer_sha256",
        "build",
        "base_runtime",
        "layout",
        "environment",
        "commands",
    ):
        if document.get(key) != planned.get(key):
            raise OlmoeEpBenchmarkError(
                f"runtime receipt disagrees with canonical install plan: {key}"
            )

    try:
        resolved_python = runtime_python.resolve(strict=True)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"runtime Python is unavailable: {error}"
        ) from error
    expected_python = Path(
        _required_string(base_runtime, "resolved_python_path", "runtime base")
    )
    expected_python_sha256 = _required_sha256(
        base_runtime.get("python_sha256"), "runtime base Python digest"
    )
    try:
        observed_python = hash_sglang_kt_bound_file(resolved_python)
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError("runtime Python target changed") from error
    if (
        not os.access(runtime_python, os.X_OK)
        or resolved_python != expected_python
        or observed_python.sha256 != expected_python_sha256
    ):
        raise OlmoeEpBenchmarkError("runtime Python target changed")
    pth_path = Path(
        _required_string(layout, "base_runtime_pth", "runtime install layout")
    )
    expected_pth_sha256 = _required_sha256(
        layout.get("base_runtime_pth_sha256"), "runtime base .pth digest"
    )
    try:
        pth = read_sglang_kt_bound_file(pth_path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES)
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError("runtime base .pth changed") from error
    if (
        pth.sha256 != expected_pth_sha256
        or pth.contents != f"{base_site_packages}\n".encode("ascii")
    ):
        raise OlmoeEpBenchmarkError("runtime base .pth changed")

    raw_distributions = document.get("installed_distributions")
    if not isinstance(raw_distributions, list):
        raise OlmoeEpBenchmarkError("runtime installed_distributions must be an array")
    admissions: list[InstalledDistributionAdmission] = []
    for value in cast(list[object], raw_distributions):
        distribution = _json_object(value, "installed distribution")
        name, version, record_sha256, file_count = _verify_distribution_record(
            install_root=install_root,
            site_packages=site_packages,
            distribution=distribution,
        )
        admissions.append(
            InstalledDistributionAdmission(
                distribution=name,
                version=version,
                record_sha256=record_sha256,
                verified_record_file_count=file_count,
            )
        )
    if len(admissions) != len(EXPECTED_DISTRIBUTIONS) or frozenset(
        item.distribution for item in admissions
    ) != frozenset(cast(str, item) for item in EXPECTED_DISTRIBUTIONS):
        raise OlmoeEpBenchmarkError(
            "runtime must contain exactly the three pinned distributions"
        )
    expected_versions = {
        _normalized_distribution(wheel.distribution): wheel.version
        for wheel in provenance.wheels
    }
    observed_versions = {
        admission.distribution: admission.version for admission in admissions
    }
    if observed_versions != expected_versions:
        raise OlmoeEpBenchmarkError(
            "installed distribution versions disagree with the admitted wheels"
        )
    patch_hashes = _verify_patch_stack(
        provenance.ktransformers_revision, provenance.sglang_revision
    )
    return RuntimeAdmission(
        install_receipt_path=str(config.runtime_install_receipt),
        install_receipt_sha256=config.runtime_install_receipt_sha256,
        install_receipt_size_bytes=receipt_size,
        install_id=install_id,
        install_root=str(install_root),
        runtime_python=str(runtime_python),
        resolved_python=str(resolved_python),
        resolved_python_sha256=observed_python.sha256,
        build_receipt_path=str(build_receipt_path),
        build_receipt_sha256=provenance.receipt_sha256,
        sglang_revision=revision,
        build_provenance_verified=provenance.verified,
        patch_stack_sha256=patch_hashes,
        installed_distributions=tuple(admissions),
        verified_record_file_count=sum(
            item.verified_record_file_count for item in admissions
        ),
    )


def _server_capacity(config: OlmoeEpBenchmarkConfig) -> tuple[int, int, int | None]:
    if config.aggregate_concurrency:
        return (
            OLMOE_CONCURRENCY_MAX_TOTAL_TOKENS,
            OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS,
            OLMOE_CONCURRENCY_CUDA_GRAPH_MAX_BATCH_SIZE,
        )
    return OLMOE_MAX_TOTAL_TOKENS, 1, None


def build_server_command(config: OlmoeEpBenchmarkConfig) -> tuple[str, ...]:
    """Build the exact native PP1/TP2 server command."""

    max_total_tokens, max_running_requests, cuda_graph_max_batch_size = (
        _server_capacity(config)
    )
    command = (
        config.numactl_executable,
        "--physcpubind",
        "0-111",
        "--interleave",
        "0,1",
        config.runtime_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tp-size",
        "2",
        "--pp-size",
        "1",
        "--ep-size",
        str(config.expert_parallel_size),
        "--numa-node",
        "0",
        "1",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--dtype",
        "bfloat16",
        "--context-length",
        str(OLMOE_CONTEXT_LENGTH),
        "--max-total-tokens",
        str(max_total_tokens),
        "--mem-fraction-static",
        format(config.static_memory_fraction, ".17g"),
        "--max-running-requests",
        str(max_running_requests),
        "--random-seed",
        str(CANONICAL_SAMPLING_SEED),
        "--disable-radix-cache",
        "--disable-custom-all-reduce",
    )
    if cuda_graph_max_batch_size is not None:
        command += ("--cuda-graph-max-bs", str(cuda_graph_max_batch_size))
    return command


def build_server_environment(
    config: OlmoeEpBenchmarkConfig,
    owner_token: str,
    ownership_namespace: str,
    parent_environment: Mapping[str, str] | None = None,
    moe_config_snapshot: MoeConfigSnapshotAdmission | None = None,
) -> dict[str, str]:
    parent = os.environ if parent_environment is None else parent_environment
    environment = {
        "PATH": parent.get(
            "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "HOME": parent.get("HOME", "/root"),
        "LANG": parent.get("LANG", "C.UTF-8"),
        "LC_ALL": parent.get("LC_ALL", "C.UTF-8"),
        "TMPDIR": parent.get("TMPDIR", "/tmp"),
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(DWAGON_GPU_UUIDS),
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_NET_GDR_LEVEL": "LOC",
            "NCCL_DEBUG": "INFO",
            "OMP_NUM_THREADS": "56",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "SGLANG_NUMA_BIND_V2": "1",
            "TOKENIZERS_PARALLELISM": "false",
            _OWNER_TOKEN_ENVIRONMENT: owner_token,
            _OWNERSHIP_NAMESPACE_ENVIRONMENT: ownership_namespace,
        }
    )
    if config.moe_config_root is None:
        if moe_config_snapshot is not None:
            raise OlmoeEpBenchmarkError(
                "MoE config snapshot is invalid without an external source"
            )
    else:
        if (
            moe_config_snapshot is None
            or moe_config_snapshot.source.root != str(config.moe_config_root)
            or moe_config_snapshot.source.expert_parallel_size
            != config.expert_parallel_size
        ):
            raise OlmoeEpBenchmarkError(
                "external MoE config requires its private per-run snapshot"
            )
        environment["SGLANG_MOE_CONFIG_DIR"] = moe_config_snapshot.effective.root
    return environment


def _run_command(
    arguments: tuple[str, ...], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def verify_dwagon_nv4_topology(
    config: OlmoeEpBenchmarkConfig,
    runner: Callable[
        [tuple[str, ...], float], subprocess.CompletedProcess[str]
    ] = _run_command,
) -> JsonObject:
    inventory_command = (
        config.nvidia_smi_executable,
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    topology_command = (config.nvidia_smi_executable, "topo", "-m")
    try:
        inventory = runner(inventory_command, 10.0)
        topology = runner(topology_command, 10.0)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OlmoeEpBenchmarkError(
            f"cannot verify dwagon NV4 topology: {error}"
        ) from error
    if inventory.returncode != 0 or topology.returncode != 0:
        raise OlmoeEpBenchmarkError("nvidia-smi topology admission failed")
    by_uuid: dict[str, tuple[int, str]] = {}
    for row in csv.reader(inventory.stdout.splitlines()):
        normalized = tuple(value.strip() for value in row)
        if len(normalized) != 3:
            raise OlmoeEpBenchmarkError("nvidia-smi inventory row is malformed")
        try:
            index = int(normalized[0])
        except ValueError as error:
            raise OlmoeEpBenchmarkError("nvidia-smi GPU index is invalid") from error
        by_uuid[normalized[1]] = (index, normalized[2].lower())
    expected_gpu_uuids: set[str] = set(DWAGON_GPU_UUIDS)
    if expected_gpu_uuids - set(by_uuid):
        raise OlmoeEpBenchmarkError("ordered dwagon GPU UUIDs are not both present")
    first_index = by_uuid[DWAGON_GPU_UUIDS[0]][0]
    second_index = by_uuid[DWAGON_GPU_UUIDS[1]][0]
    rows = {
        fields[0]: fields[1:]
        for line in topology.stdout.splitlines()
        if (fields := line.split()) and fields[0].startswith("GPU")
    }
    first_row = rows.get(f"GPU{first_index}")
    second_row = rows.get(f"GPU{second_index}")
    maximum_index = max(first_index, second_index)
    if (
        first_row is None
        or second_row is None
        or len(first_row) <= maximum_index
        or len(second_row) <= maximum_index
        or first_row[second_index] != "NV4"
        or second_row[first_index] != "NV4"
    ):
        raise OlmoeEpBenchmarkError(
            "dwagon GPU pair is not an active symmetric NV4 link"
        )
    return {
        "ordered_gpu_uuids": list(DWAGON_GPU_UUIDS),
        "inventory": [
            {
                "launch_ordinal": ordinal,
                "uuid": gpu_uuid,
                "physical_index": by_uuid[gpu_uuid][0],
                "pci_bus_id": by_uuid[gpu_uuid][1],
            }
            for ordinal, gpu_uuid in enumerate(DWAGON_GPU_UUIDS)
        ],
        "required_link": "NV4",
        "observed_forward_link": first_row[second_index],
        "observed_reverse_link": second_row[first_index],
        "inventory_command": list(inventory_command),
        "inventory_stdout_sha256": hashlib.sha256(
            inventory.stdout.encode()
        ).hexdigest(),
        "topology_command": list(topology_command),
        "topology_stdout_sha256": hashlib.sha256(topology.stdout.encode()).hexdigest(),
    }


def _parse_cpu_list(raw: str, description: str) -> frozenset[int]:
    values: set[int] = set()
    for element in raw.strip().split(","):
        if not element:
            raise OlmoeEpBenchmarkError(f"{description} is malformed")
        start_raw, separator, end_raw = element.partition("-")
        try:
            start = int(start_raw)
            end = int(end_raw) if separator else start
        except ValueError as error:
            raise OlmoeEpBenchmarkError(f"{description} is malformed") from error
        if start < 0 or end < start:
            raise OlmoeEpBenchmarkError(f"{description} is malformed")
        values.update(range(start, end + 1))
    return frozenset(values)


def verify_dwagon_cpu_topology(
    sysfs_root: Path = _CPU_SYSFS_ROOT,
) -> JsonObject:
    """Prove that 0-111 selects one physical thread per core on both NUMA nodes."""

    selected = frozenset(DWAGON_PHYSICAL_CPUS)
    try:
        online_raw = (sysfs_root / "cpu/online").read_text()
        node_raw = tuple(
            (sysfs_root / f"node/node{node}/cpulist").read_text()
            for node in DWAGON_NUMA_NODES
        )
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read dwagon CPU topology: {error}"
        ) from error
    online = _parse_cpu_list(online_raw, "online CPU list")
    if not selected.issubset(online):
        raise OlmoeEpBenchmarkError("not all pinned dwagon physical CPUs are online")
    expected_by_node = (frozenset(range(56)), frozenset(range(56, 112)))
    observed_by_node = tuple(
        _parse_cpu_list(raw, f"NUMA node {node} CPU list") & selected
        for node, raw in zip(DWAGON_NUMA_NODES, node_raw, strict=True)
    )
    if observed_by_node != expected_by_node:
        raise OlmoeEpBenchmarkError("dwagon physical CPU to NUMA mapping changed")
    sibling_evidence: list[JsonValue] = []
    for cpu in DWAGON_PHYSICAL_CPUS:
        path = sysfs_root / f"cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            raw = path.read_text()
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot read dwagon CPU {cpu} thread siblings: {error}"
            ) from error
        siblings = _parse_cpu_list(raw, f"CPU {cpu} thread siblings")
        selected_siblings = siblings & selected
        if selected_siblings != frozenset((cpu,)):
            raise OlmoeEpBenchmarkError(
                "pinned dwagon CPU set contains two threads from one physical core"
            )
        sibling_evidence.append(
            cast(
                JsonObject,
                {
                    "cpu": cpu,
                    "thread_siblings": sorted(siblings),
                },
            )
        )
    return cast(
        JsonObject,
        {
            "selected_cpu_count": len(selected),
            "selected_cpus": list(DWAGON_PHYSICAL_CPUS),
            "online_cpus_sha256": hashlib.sha256(online_raw.encode()).hexdigest(),
            "numa_nodes": [
                {
                    "node": node,
                    "selected_cpus": sorted(observed),
                    "cpulist_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                }
                for node, observed, raw in zip(
                    DWAGON_NUMA_NODES, observed_by_node, node_raw, strict=True
                )
            ],
            "simultaneous_multithreading_policy": (
                "one_online_thread_per_physical_core"
            ),
            "thread_siblings": sibling_evidence,
        },
    )


def _read_process_identity_at(pid: int, proc_root: Path) -> ProcessStatIdentity:
    try:
        contents = (proc_root / str(pid) / "stat").read_text()
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read process {pid} identity: {error}"
        ) from error
    closing = contents.rfind(")")
    if closing < 0:
        raise OlmoeEpBenchmarkError(f"process {pid} stat is malformed")
    fields = contents[closing + 2 :].split()
    if len(fields) < 20:
        raise OlmoeEpBenchmarkError(f"process {pid} stat is incomplete")
    try:
        return ProcessStatIdentity(
            parent_pid=int(fields[1]),
            process_group_id=int(fields[2]),
            session_id=int(fields[3]),
            start_time_ticks=int(fields[19]),
            state=fields[0],
        )
    except ValueError as error:
        raise OlmoeEpBenchmarkError(f"process {pid} stat is invalid") from error


def _read_process_stat_at(pid: int, proc_root: Path) -> tuple[int, int, str]:
    identity = _read_process_identity_at(pid, proc_root)
    return identity.process_group_id, identity.start_time_ticks, identity.state


def _read_process_stat(pid: int) -> tuple[int, int, str]:
    return _read_process_stat_at(pid, Path("/proc"))


def _process_group_members(process_group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            observed_group, _start, state = _read_process_stat(int(entry.name))
        except (OlmoeEpBenchmarkError, ValueError):
            continue
        if observed_group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _process_environment(pid: int) -> dict[str, str]:
    try:
        contents = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read process {pid} environment: {error}"
        ) from error
    environment: dict[str, str] = {}
    for entry in contents.split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        if not separator:
            continue
        environment[name.decode(errors="strict")] = value.decode(errors="strict")
    return environment


def _validate_owned_member_identity(
    owned: OwnedServerProcess, pid: int, identity: ProcessStatIdentity
) -> None:
    if (
        identity.process_group_id != owned.process_group_id
        or identity.session_id != owned.pid
        or identity.start_time_ticks < owned.start_time_ticks
        or (pid == owned.pid and identity.start_time_ticks != owned.start_time_ticks)
    ):
        raise OlmoeEpBenchmarkError(
            f"process group {owned.process_group_id} contains an unowned process"
        )


def _owned_group_members(owned: OwnedServerProcess) -> tuple[int, ...]:
    members = _process_group_members(owned.process_group_id)
    for pid in members:
        identity = _read_process_identity_at(pid, Path("/proc"))
        _validate_owned_member_identity(owned, pid, identity)
        if pid == owned.pid:
            environment = _process_environment(pid)
            if (
                environment.get(_OWNER_TOKEN_ENVIRONMENT) != owned.owner_token
                or environment.get(_OWNERSHIP_NAMESPACE_ENVIRONMENT)
                != owned.ownership_namespace
            ):
                raise OlmoeEpBenchmarkError(
                    "native SGLang session leader identity changed"
                )
    return members


def verify_port_vacant(host: str, port: int) -> JsonObject:
    address = "127.0.0.1" if host == "localhost" else host
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((address, port))
        probe.listen(1)
        bound_address, bound_port = cast(tuple[str, int], probe.getsockname())
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"native SGLang port is already occupied: {address}:{port}: {error}"
        ) from error
    finally:
        probe.close()
    return {
        "status": "vacant",
        "host": str(bound_address),
        "port": int(bound_port),
        "checked_at_utc": _utc_now(),
    }


def _listening_socket_inodes(
    port: int, proc_root: Path = Path("/proc")
) -> frozenset[int]:
    inodes: set[int] = set()
    for relative in ("net/tcp", "net/tcp6"):
        try:
            rows = (proc_root / relative).read_text().splitlines()[1:]
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot inspect listening sockets: {relative}: {error}"
            ) from error
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            _address, separator, encoded_port = fields[1].rpartition(":")
            try:
                observed_port = int(encoded_port, 16)
                inode = int(fields[9])
            except ValueError as error:
                raise OlmoeEpBenchmarkError(
                    f"malformed listening socket row in {relative}"
                ) from error
            if separator and observed_port == port and inode > 0:
                inodes.add(inode)
    return frozenset(inodes)


def _socket_inode_owners(
    inodes: frozenset[int], proc_root: Path = Path("/proc")
) -> dict[int, tuple[int, ...]]:
    owners: dict[int, list[int]] = {inode: [] for inode in inodes}
    targets = {f"socket:[{inode}]": inode for inode in inodes}
    for process_path in proc_root.iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            descriptors = tuple((process_path / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            inode = targets.get(target)
            if inode is not None:
                owners[inode].append(int(process_path.name))
    return {inode: tuple(sorted(set(pids))) for inode, pids in owners.items()}


def assert_server_alive(running: RunningServerProcess, phase: str) -> None:
    return_code = running.process.poll()
    if return_code is not None:
        raise OlmoeEpBenchmarkError(
            f"native SGLang exited during {phase} with code {return_code}"
        )
    owned = running.owned
    try:
        process_group, start_time, state = _read_process_stat(owned.pid)
    except OlmoeEpBenchmarkError as error:
        raise OlmoeEpBenchmarkError(
            f"native SGLang identity vanished during {phase}"
        ) from error
    if (
        process_group != owned.process_group_id
        or start_time != owned.start_time_ticks
        or state == "Z"
        or owned.pid not in _owned_group_members(owned)
    ):
        raise OlmoeEpBenchmarkError(f"native SGLang identity changed during {phase}")


def verify_listener_owned(
    running: RunningServerProcess,
    port: int,
    proc_root: Path = Path("/proc"),
) -> JsonObject:
    assert_server_alive(running, "listener ownership verification")
    inodes = _listening_socket_inodes(port, proc_root)
    if not inodes:
        raise OlmoeEpBenchmarkError("native SGLang has no observable listener")
    inode_owners = _socket_inode_owners(inodes, proc_root)
    owned_members = frozenset(_owned_group_members(running.owned))
    observed_owners = frozenset(pid for pids in inode_owners.values() for pid in pids)
    if (
        any(not pids for pids in inode_owners.values())
        or not observed_owners
        or not observed_owners.issubset(owned_members)
    ):
        raise OlmoeEpBenchmarkError(
            "listener is not exclusively owned by the launched process group"
        )
    for pid in observed_owners:
        environment = _process_environment(pid)
        if (
            environment.get(_OWNER_TOKEN_ENVIRONMENT) != running.owned.owner_token
            or environment.get(_OWNERSHIP_NAMESPACE_ENVIRONMENT)
            != running.owned.ownership_namespace
        ):
            raise OlmoeEpBenchmarkError("listener owner nonce does not match launch")
    return cast(
        JsonObject,
        {
            "status": "owned",
            "port": port,
            "socket_inodes": sorted(inodes),
            "owner_pids": sorted(observed_owners),
            "process_group_id": running.owned.process_group_id,
            "owner_token_sha256": hashlib.sha256(
                running.owned.owner_token.encode()
            ).hexdigest(),
        },
    )


def _read_bounded_proc_file(path: Path, maximum_bytes: int) -> bytes:
    try:
        with path.open("rb") as source:
            contents = source.read(maximum_bytes + 1)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read process evidence {path}: {error}"
        ) from error
    if len(contents) > maximum_bytes:
        raise OlmoeEpBenchmarkError(f"process evidence is oversized: {path}")
    return contents


def _task_status_cpu_affinity(path: Path) -> frozenset[int] | None:
    """Read one live task affinity, returning None only if the task exited."""

    try:
        with path.open("rb") as source:
            raw_contents = source.read(256 * 1024 + 1)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read scheduler task status {path}: {error}"
        ) from error
    if len(raw_contents) > 256 * 1024:
        raise OlmoeEpBenchmarkError(f"scheduler task status is oversized: {path}")
    try:
        contents = raw_contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError(
            f"scheduler task status is not ASCII: {path}"
        ) from error
    for line in contents.splitlines():
        name, separator, value = line.partition(":")
        if separator and name == "Cpus_allowed_list":
            return _parse_cpu_list(value, f"CPU affinity in {path}")
    raise OlmoeEpBenchmarkError(f"scheduler task status lacks CPU affinity: {path}")


def _scheduler_process_title(pid: int, proc_root: Path) -> str:
    contents = _read_bounded_proc_file(proc_root / str(pid) / "cmdline", 64 * 1024)
    first, _separator, _remaining = contents.partition(b"\0")
    try:
        title = first.decode("ascii")
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError("scheduler process title is not ASCII") from error
    return title


@dataclass(frozen=True, slots=True)
class _VmaIdentity:
    start_address: str
    end_address: str
    permissions: str
    offset: str
    device: str
    inode: int
    path: str | None


@dataclass(frozen=True, slots=True)
class _SmapsVmaObservation:
    identity: _VmaIdentity
    rss_kibibytes: int
    pss_kibibytes: int
    anonymous_kibibytes: int
    swap_kibibytes: int
    vm_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StableVmaSnapshot:
    maps: Mapping[str, _VmaIdentity]
    smaps: Mapping[str, _SmapsVmaObservation]
    numa_map_addresses: frozenset[str]
    maps_contents: bytes
    smaps_contents: bytes
    numa_maps_contents: bytes
    vsyscall_omissions: tuple[str, ...]
    stable_observation_attempt: int


def _vma_identity(line: str, description: str) -> _VmaIdentity:
    match = _MAPS_HEADER_PATTERN.fullmatch(line)
    if match is None:
        raise OlmoeEpBenchmarkError(f"{description} VMA header is malformed")
    start_address = match.group("start")
    end_address = match.group("end")
    if int(end_address, 16) <= int(start_address, 16):
        raise OlmoeEpBenchmarkError(f"{description} VMA range is invalid")
    raw_path = match.group("path")
    return _VmaIdentity(
        start_address=start_address,
        end_address=end_address,
        permissions=match.group("permissions"),
        offset=match.group("offset"),
        device=match.group("device"),
        inode=int(match.group("inode")),
        path=None if raw_path is None or not raw_path else raw_path,
    )


def _parse_maps(contents: bytes) -> dict[str, _VmaIdentity]:
    try:
        lines = contents.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError("scheduler maps is not ASCII") from error
    identities: dict[str, _VmaIdentity] = {}
    for line in lines:
        if not line:
            continue
        identity = _vma_identity(line, "scheduler maps")
        if identity.start_address in identities:
            raise OlmoeEpBenchmarkError("scheduler maps repeats a VMA start")
        identities[identity.start_address] = identity
    if not identities:
        raise OlmoeEpBenchmarkError("scheduler maps is empty")
    return identities


def _smaps_kibibytes(value: str, field_name: str) -> int:
    fields = value.strip().split()
    if len(fields) != 2 or fields[1] != "kB":
        raise OlmoeEpBenchmarkError(f"scheduler smaps {field_name} is malformed")
    try:
        amount = int(fields[0])
    except ValueError as error:
        raise OlmoeEpBenchmarkError(
            f"scheduler smaps {field_name} is malformed"
        ) from error
    if amount < 0:
        raise OlmoeEpBenchmarkError(f"scheduler smaps {field_name} is negative")
    return amount


def _parse_smaps(contents: bytes) -> dict[str, _SmapsVmaObservation]:
    try:
        lines = contents.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError("scheduler smaps is not ASCII") from error
    observations: dict[str, _SmapsVmaObservation] = {}
    current: _VmaIdentity | None = None
    metrics: dict[str, int] = {}
    vm_flags: tuple[str, ...] | None = None
    required_metrics = frozenset({"Rss", "Pss", "Anonymous", "Swap"})

    def finish_current() -> None:
        if current is None:
            return
        if frozenset(metrics) != required_metrics or vm_flags is None:
            raise OlmoeEpBenchmarkError(
                f"scheduler smaps VMA {current.start_address} lacks required evidence"
            )
        if current.start_address in observations:
            raise OlmoeEpBenchmarkError("scheduler smaps repeats a VMA start")
        observations[current.start_address] = _SmapsVmaObservation(
            identity=current,
            rss_kibibytes=metrics["Rss"],
            pss_kibibytes=metrics["Pss"],
            anonymous_kibibytes=metrics["Anonymous"],
            swap_kibibytes=metrics["Swap"],
            vm_flags=vm_flags,
        )

    for line in lines:
        if _MAPS_HEADER_PATTERN.fullmatch(line) is not None:
            finish_current()
            current = _vma_identity(line, "scheduler smaps")
            metrics = {}
            vm_flags = None
            continue
        if not line:
            continue
        if current is None:
            raise OlmoeEpBenchmarkError("scheduler smaps data precedes its VMA header")
        name, separator, value = line.partition(":")
        if not separator:
            raise OlmoeEpBenchmarkError("scheduler smaps row is malformed")
        if name in required_metrics:
            if name in metrics:
                raise OlmoeEpBenchmarkError(
                    f"scheduler smaps VMA {current.start_address} repeats {name}"
                )
            metrics[name] = _smaps_kibibytes(value, name)
        elif name == "VmFlags":
            if vm_flags is not None:
                raise OlmoeEpBenchmarkError(
                    f"scheduler smaps VMA {current.start_address} repeats VmFlags"
                )
            raw_flags = tuple(value.strip().split())
            if not raw_flags or len(set(raw_flags)) != len(raw_flags):
                raise OlmoeEpBenchmarkError(
                    f"scheduler smaps VMA {current.start_address} has invalid VmFlags"
                )
            vm_flags = raw_flags
    finish_current()
    if not observations:
        raise OlmoeEpBenchmarkError("scheduler smaps is empty")
    return observations


def _numa_map_addresses(contents: bytes) -> frozenset[str]:
    try:
        lines = contents.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError("scheduler numa_maps is not ASCII") from error
    addresses: set[str] = set()
    for line in lines:
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or _NUMA_MAP_ADDRESS_PATTERN.fullmatch(fields[0]) is None:
            raise OlmoeEpBenchmarkError("scheduler numa_maps row is malformed")
        if fields[0] in addresses:
            raise OlmoeEpBenchmarkError("scheduler numa_maps repeats a VMA start")
        addresses.add(fields[0])
    if not addresses:
        raise OlmoeEpBenchmarkError("scheduler numa_maps is empty")
    return frozenset(addresses)


def _read_stable_vma_snapshot(
    *,
    pid: int,
    expected_start_time_ticks: int,
    proc_root: Path,
) -> _StableVmaSnapshot:
    last_instability = "not observed"
    process_root = proc_root / str(pid)
    for attempt in range(3):
        before_group, before_start, before_state = _read_process_stat_at(pid, proc_root)
        if before_start != expected_start_time_ticks or before_state == "Z":
            raise OlmoeEpBenchmarkError("scheduler identity changed before VMA capture")
        maps_contents = _read_bounded_proc_file(
            process_root / "maps", _PROC_MAPS_MAXIMUM_BYTES
        )
        smaps_contents = _read_bounded_proc_file(
            process_root / "smaps", _PROC_SMAPS_MAXIMUM_BYTES
        )
        numa_maps_contents = _read_bounded_proc_file(
            process_root / "numa_maps", _PROC_MAPS_MAXIMUM_BYTES
        )
        final_maps_contents = _read_bounded_proc_file(
            process_root / "maps", _PROC_MAPS_MAXIMUM_BYTES
        )
        after_group, after_start, after_state = _read_process_stat_at(pid, proc_root)
        if (
            after_group != before_group
            or after_start != expected_start_time_ticks
            or after_state == "Z"
        ):
            raise OlmoeEpBenchmarkError("scheduler identity changed during VMA capture")
        if maps_contents != final_maps_contents:
            last_instability = "maps changed during capture"
            time.sleep(0.05)
            continue
        try:
            maps = _parse_maps(maps_contents)
            smaps = _parse_smaps(smaps_contents)
            numa_map_addresses = _numa_map_addresses(numa_maps_contents)
        except OlmoeEpBenchmarkError as error:
            last_instability = str(error)
            time.sleep(0.05)
            continue
        if maps != {address: item.identity for address, item in smaps.items()}:
            last_instability = "maps and smaps VMA headers differ"
            time.sleep(0.05)
            continue
        map_addresses = frozenset(maps)
        extra_numa_addresses = numa_map_addresses - map_addresses
        missing_numa_addresses = map_addresses - numa_map_addresses
        if extra_numa_addresses or any(
            maps[address].path != "[vsyscall]"
            or maps[address].permissions != "--xp"
            or int(maps[address].offset, 16) != 0
            or maps[address].device != "00:00"
            or maps[address].inode != 0
            for address in missing_numa_addresses
        ):
            last_instability = "maps, smaps, and numa_maps VMA starts differ"
            time.sleep(0.05)
            continue
        return _StableVmaSnapshot(
            maps=maps,
            smaps=smaps,
            numa_map_addresses=numa_map_addresses,
            maps_contents=maps_contents,
            smaps_contents=smaps_contents,
            numa_maps_contents=numa_maps_contents,
            vsyscall_omissions=tuple(sorted(missing_numa_addresses)),
            stable_observation_attempt=attempt + 1,
        )
    raise OlmoeEpBenchmarkError(
        f"scheduler VMA evidence did not stabilize: {last_instability}"
    )


def _numa_map_numeric_field(fields: tuple[str, ...], name: str, address: str) -> int:
    prefix = f"{name}="
    values = [
        field.removeprefix(prefix) for field in fields if field.startswith(prefix)
    ]
    if len(values) > 1:
        raise OlmoeEpBenchmarkError(f"scheduler numa_maps row {address} repeats {name}")
    if not values:
        return 0
    try:
        value = int(values[0])
    except ValueError as error:
        raise OlmoeEpBenchmarkError(
            f"scheduler numa_maps row {address} has invalid {name}"
        ) from error
    if value < 0:
        raise OlmoeEpBenchmarkError(
            f"scheduler numa_maps row {address} has negative {name}"
        )
    return value


def _numa_mapping_class(
    *,
    file_path: str | None,
    permissions: str,
    flags: frozenset[str],
    anonymous_pages: int,
) -> NumaMappingClass:
    special_file = file_path is not None and file_path.startswith(
        _SPECIAL_NUMA_FILE_PREFIXES
    )
    named_shared_anonymous = file_path is not None and file_path.startswith(
        "[anon_shmem:"
    )
    if (
        "shmem" in flags
        or special_file
        or named_shared_anonymous
        or permissions.endswith("s")
    ):
        return "special_or_shared"
    if file_path is not None and (
        file_path.startswith(f"{OLMOE_MODEL_PATH.rstrip('/')}/")
        and file_path.endswith(_MODEL_WEIGHT_SUFFIXES)
    ):
        return "model_weight"
    private_anonymous_path = file_path is None or file_path in {"[heap]", "[stack]"}
    if file_path is not None and file_path.startswith("[anon:"):
        private_anonymous_path = True
    if private_anonymous_path:
        return "private_anonymous"
    if file_path is not None:
        return "regular_file_private" if anonymous_pages else "regular_file"
    return "private_anonymous"


def _observe_numa_memory_placement(
    pid: int,
    expected_node: int,
    expected_start_time_ticks: int,
    proc_root: Path,
) -> JsonObject:
    snapshot = _read_stable_vma_snapshot(
        pid=pid,
        expected_start_time_ticks=expected_start_time_ticks,
        proc_root=proc_root,
    )
    raw_contents = snapshot.numa_maps_contents
    try:
        lines = raw_contents.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError("scheduler numa_maps is not ASCII") from error
    expected_policy = f"bind:{expected_node}"
    policy_counts: dict[str, int] = {}
    classification_counts: dict[NumaMappingClass, int] = {}
    classification_resident_kibibytes: dict[NumaMappingClass, int] = {}
    classification_policy_counts: dict[tuple[NumaMappingClass, str], int] = {}
    classification_node_kibibytes: dict[tuple[NumaMappingClass, int], int] = {}
    resident_kibibytes_by_node: dict[int, int] = {}
    placement_sensitive_vma_kibibytes_by_node: dict[int, int] = {}
    classified_rows: list[JsonValue] = []
    violation_counts: dict[str, int] = {}
    violation_examples: list[JsonValue] = []
    local_reservation_exceptions: list[JsonValue] = []
    placement_sensitive_mapping_count = 0
    placement_sensitive_resident_kibibytes = 0
    provable_sensitive_off_node_kibibytes = 0

    def record_violation(
        kind: str,
        *,
        address: str,
        policy: str,
        classification: NumaMappingClass,
        file_path: str | None,
        identity: _VmaIdentity | None = None,
        smaps: _SmapsVmaObservation | None = None,
        off_node_kibibytes: int = 0,
    ) -> None:
        violation_counts[kind] = violation_counts.get(kind, 0) + 1
        if len(violation_examples) < 8:
            violation_examples.append(
                {
                    "kind": kind,
                    "address": address,
                    "policy": policy,
                    "classification": classification,
                    "file_path": file_path,
                    "off_node_kibibytes": off_node_kibibytes,
                    "permissions": None if identity is None else identity.permissions,
                    "device": None if identity is None else identity.device,
                    "inode": None if identity is None else identity.inode,
                    "rss_kibibytes": None if smaps is None else smaps.rss_kibibytes,
                    "pss_kibibytes": None if smaps is None else smaps.pss_kibibytes,
                    "anonymous_kibibytes": (
                        None if smaps is None else smaps.anonymous_kibibytes
                    ),
                    "swap_kibibytes": None if smaps is None else smaps.swap_kibibytes,
                    "vm_flags": None if smaps is None else list(smaps.vm_flags),
                }
            )

    for line in lines:
        if not line:
            continue
        fields = tuple(line.split())
        if len(fields) < 2:
            raise OlmoeEpBenchmarkError("scheduler numa_maps row is malformed")
        address, policy, *details = fields
        if _NUMA_MAP_ADDRESS_PATTERN.fullmatch(address) is None:
            raise OlmoeEpBenchmarkError("scheduler numa_maps address is malformed")
        identity = snapshot.maps[address]
        smaps = snapshot.smaps[address]
        numa_file_paths = [
            field.removeprefix("file=")
            for field in details
            if field.startswith("file=")
        ]
        if len(numa_file_paths) > 1 or (numa_file_paths and not numa_file_paths[0]):
            raise OlmoeEpBenchmarkError(
                f"scheduler numa_maps row {address} has invalid file identity"
            )
        numa_file_path = numa_file_paths[0] if numa_file_paths else None
        file_path = identity.path
        flags = frozenset(field for field in details if "=" not in field)
        anonymous_pages = _numa_map_numeric_field(fields, "anon", address)
        mapped_pages = _numa_map_numeric_field(fields, "mapped", address)
        kernel_page_size_kibibytes = _numa_map_numeric_field(
            fields, "kernelpagesize_kB", address
        )
        node_pages: dict[int, int] = {}
        for field in details:
            match = _NUMA_MAP_NODE_PAGES_PATTERN.fullmatch(field)
            if match is None:
                continue
            node = int(match.group("node"))
            if node in node_pages:
                raise OlmoeEpBenchmarkError(
                    f"scheduler numa_maps row {address} repeats node {node}"
                )
            node_pages[node] = int(match.group("pages"))
        resident_page_count = sum(node_pages.values())
        if resident_page_count and kernel_page_size_kibibytes == 0:
            raise OlmoeEpBenchmarkError(
                f"scheduler numa_maps row {address} lacks kernel page size"
            )
        node_kibibytes = {
            node: pages * kernel_page_size_kibibytes
            for node, pages in node_pages.items()
        }
        resident_kibibytes = sum(node_kibibytes.values())
        classification = _numa_mapping_class(
            file_path=file_path,
            permissions=identity.permissions,
            flags=flags,
            anonymous_pages=anonymous_pages,
        )

        policy_counts[policy] = policy_counts.get(policy, 0) + 1
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
        classification_resident_kibibytes[classification] = (
            classification_resident_kibibytes.get(classification, 0)
            + resident_kibibytes
        )
        policy_key = (classification, policy)
        classification_policy_counts[policy_key] = (
            classification_policy_counts.get(policy_key, 0) + 1
        )
        for node, kibibytes in node_kibibytes.items():
            resident_kibibytes_by_node[node] = (
                resident_kibibytes_by_node.get(node, 0) + kibibytes
            )
            class_node_key = (classification, node)
            classification_node_kibibytes[class_node_key] = (
                classification_node_kibibytes.get(class_node_key, 0) + kibibytes
            )

        local_reservation_exception = (
            policy == "local"
            and resident_page_count == 0
            and not details
            and identity.permissions == "rw-p"
            and int(identity.offset, 16) == 0
            and identity.device == "00:00"
            and identity.inode == 0
            and identity.path is None
            and smaps.rss_kibibytes == 0
            and smaps.pss_kibibytes == 0
            and smaps.anonymous_kibibytes == 0
            and smaps.swap_kibibytes == 0
            and frozenset(smaps.vm_flags) == _CUDA_VMM_RESERVATION_VM_FLAGS
        )
        if policy == expected_policy:
            pass
        elif local_reservation_exception:
            local_reservation_exceptions.append(
                {
                    "address": address,
                    "virtual_size_kibibytes": (
                        int(identity.end_address, 16) - int(identity.start_address, 16)
                    )
                    // 1024,
                    "permissions": identity.permissions,
                    "offset": identity.offset,
                    "device": identity.device,
                    "inode": identity.inode,
                    "path": identity.path,
                    "rss_kibibytes": smaps.rss_kibibytes,
                    "pss_kibibytes": smaps.pss_kibibytes,
                    "anonymous_kibibytes": smaps.anonymous_kibibytes,
                    "swap_kibibytes": smaps.swap_kibibytes,
                    "vm_flags": list(smaps.vm_flags),
                }
            )
        else:
            if policy.startswith("bind:"):
                violation_kind = "wrong_bind_policy"
            elif policy == "local":
                violation_kind = "invalid_local_reservation_exception"
            else:
                violation_kind = "unsupported_memory_policy"
            record_violation(
                violation_kind,
                address=address,
                policy=policy,
                classification=classification,
                file_path=file_path,
                identity=identity,
                smaps=smaps,
            )

        off_node_kibibytes = sum(
            kibibytes
            for node, kibibytes in node_kibibytes.items()
            if node != expected_node
        )
        provable_off_node_kibibytes = 0
        if classification in {"model_weight", "private_anonymous"}:
            placement_sensitive_mapping_count += 1
            placement_sensitive_resident_kibibytes += resident_kibibytes
            provable_off_node_kibibytes = off_node_kibibytes
        elif classification == "regular_file_private":
            placement_sensitive_mapping_count += 1
            resident_anonymous_pages = min(anonymous_pages, resident_page_count)
            placement_sensitive_resident_kibibytes += (
                resident_anonymous_pages * kernel_page_size_kibibytes
            )
            # numa_maps does not assign each node page to the file or COW subset.
            # Subtract every possible resident file page to retain a proven floor.
            resident_file_pages = resident_page_count - resident_anonymous_pages
            off_node_pages = sum(
                pages for node, pages in node_pages.items() if node != expected_node
            )
            provable_off_node_kibibytes = (
                max(0, off_node_pages - resident_file_pages)
                * kernel_page_size_kibibytes
            )
        if classification in {
            "model_weight",
            "private_anonymous",
            "regular_file_private",
        }:
            for node, kibibytes in node_kibibytes.items():
                placement_sensitive_vma_kibibytes_by_node[node] = (
                    placement_sensitive_vma_kibibytes_by_node.get(node, 0) + kibibytes
                )
        if provable_off_node_kibibytes:
            provable_sensitive_off_node_kibibytes += provable_off_node_kibibytes
            record_violation(
                "placement_sensitive_memory_off_node",
                address=address,
                policy=policy,
                classification=classification,
                file_path=file_path,
                identity=identity,
                smaps=smaps,
                off_node_kibibytes=provable_off_node_kibibytes,
            )

        classified_rows.append(
            cast(
                JsonValue,
                {
                    "address": address,
                    "policy": policy,
                    "classification": classification,
                    "maps_identity": {
                        "end_address": identity.end_address,
                        "permissions": identity.permissions,
                        "offset": identity.offset,
                        "device": identity.device,
                        "inode": identity.inode,
                        "path": identity.path,
                    },
                    "numa_file_path": numa_file_path,
                    "numa_flags": sorted(flags),
                    "anonymous_pages": anonymous_pages,
                    "mapped_pages": mapped_pages,
                    "kernel_page_size_kibibytes": kernel_page_size_kibibytes,
                    "resident_page_counts_by_node": {
                        str(node): pages for node, pages in sorted(node_pages.items())
                    },
                    "resident_kibibytes_by_node": {
                        str(node): kibibytes
                        for node, kibibytes in sorted(node_kibibytes.items())
                    },
                    "smaps": {
                        "rss_kibibytes": smaps.rss_kibibytes,
                        "pss_kibibytes": smaps.pss_kibibytes,
                        "anonymous_kibibytes": smaps.anonymous_kibibytes,
                        "swap_kibibytes": smaps.swap_kibibytes,
                        "vm_flags": list(smaps.vm_flags),
                    },
                },
            ),
        )

    line_count = len(classified_rows)
    if line_count == 0:
        raise OlmoeEpBenchmarkError("scheduler numa_maps is empty")
    if (
        placement_sensitive_mapping_count == 0
        or placement_sensitive_resident_kibibytes == 0
    ):
        record_violation(
            "no_resident_placement_sensitive_memory",
            address="none",
            policy="none",
            classification="private_anonymous",
            file_path=None,
        )

    classifications: list[JsonValue] = []
    for classification in sorted(classification_counts):
        classifications.append(
            {
                "classification": classification,
                "mapping_count": classification_counts[classification],
                "resident_kibibytes": classification_resident_kibibytes[classification],
                "policy_counts": {
                    policy: count
                    for (observed_class, policy), count in sorted(
                        classification_policy_counts.items()
                    )
                    if observed_class == classification
                },
                "resident_kibibytes_by_node": {
                    str(node): kibibytes
                    for (observed_class, node), kibibytes in sorted(
                        classification_node_kibibytes.items()
                    )
                    if observed_class == classification
                },
            }
        )
    evidence = cast(
        JsonObject,
        {
            "status": "verified" if not violation_counts else "rejected",
            "expected_policy": expected_policy,
            "policy_contract": (
                "all_vmas_exact_bind_except_proven_zero_resident_"
                "private_cuda_local_reservation"
            ),
            "mapping_count": line_count,
            "policy_counts": policy_counts,
            "classifications": classifications,
            "resident_kibibytes": sum(resident_kibibytes_by_node.values()),
            "resident_kibibytes_by_node": {
                str(node): kibibytes
                for node, kibibytes in sorted(resident_kibibytes_by_node.items())
            },
            "placement_sensitive_mapping_count": placement_sensitive_mapping_count,
            "placement_sensitive_resident_kibibytes": (
                placement_sensitive_resident_kibibytes
            ),
            "placement_sensitive_vma_kibibytes_by_node": {
                str(node): kibibytes
                for node, kibibytes in sorted(
                    placement_sensitive_vma_kibibytes_by_node.items()
                )
            },
            "provable_sensitive_off_node_kibibytes": (
                provable_sensitive_off_node_kibibytes
            ),
            "local_reservation_exception_count": len(local_reservation_exceptions),
            "local_reservation_exceptions": local_reservation_exceptions,
            "violation_counts": violation_counts,
            "violation_examples": violation_examples,
            "vma_join": {
                "maps_vma_count": len(snapshot.maps),
                "smaps_vma_count": len(snapshot.smaps),
                "numa_maps_vma_count": len(snapshot.numa_map_addresses),
                "vsyscall_omissions": list(snapshot.vsyscall_omissions),
                "stable_observation_attempt": snapshot.stable_observation_attempt,
            },
            "maps_sha256": hashlib.sha256(snapshot.maps_contents).hexdigest(),
            "smaps_sha256": hashlib.sha256(snapshot.smaps_contents).hexdigest(),
            "numa_maps_sha256": hashlib.sha256(raw_contents).hexdigest(),
            "classified_rows_sha256": _canonical_sha256(classified_rows),
        },
    )
    if violation_counts:
        counts = ",".join(
            f"{kind}={count}" for kind, count in sorted(violation_counts.items())
        )
        examples = _canonical_json(violation_examples).decode("ascii")
        raise OlmoeEpBenchmarkError(
            f"scheduler NUMA memory placement failed ({counts}); examples={examples}"
        )
    return evidence


def _observe_scheduler_numa(
    *,
    pid: int,
    process_group_id: int,
    tp_rank: int,
    ep_rank: int | None,
    process_title: str,
    expected_node: int,
    expected_cpus: frozenset[int],
    proc_root: Path,
) -> JsonObject:
    for attempt in range(3):
        before_group, before_start, before_state = _read_process_stat_at(pid, proc_root)
        if before_group != process_group_id or before_state == "Z":
            raise OlmoeEpBenchmarkError("scheduler process identity is not owned")
        try:
            first_tasks = tuple(
                sorted(
                    int(path.name)
                    for path in (proc_root / str(pid) / "task").iterdir()
                    if path.name.isdigit()
                )
            )
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot enumerate scheduler {pid} threads: {error}"
            ) from error
        if not first_tasks:
            raise OlmoeEpBenchmarkError("scheduler has no observable threads")
        affinity_counts: dict[tuple[int, ...], int] = {}
        task_affinities: dict[int, frozenset[int]] = {}
        task_exit_observed = False
        for task_id in first_tasks:
            affinity = _task_status_cpu_affinity(
                proc_root / str(pid) / "task" / str(task_id) / "status"
            )
            if affinity is None:
                task_exit_observed = True
                break
            if not affinity or not affinity.issubset(expected_cpus):
                raise OlmoeEpBenchmarkError(
                    f"TP rank {tp_rank} has a thread outside NUMA node {expected_node}"
                )
            task_affinities[task_id] = affinity
            key = tuple(sorted(affinity))
            affinity_counts[key] = affinity_counts.get(key, 0) + 1
        if task_exit_observed:
            after_group, after_start, after_state = _read_process_stat_at(
                pid, proc_root
            )
            if (
                after_group != before_group
                or after_start != before_start
                or after_state == "Z"
            ):
                raise OlmoeEpBenchmarkError("scheduler process identity changed")
            time.sleep(0.05)
            continue
        try:
            memory_placement = _observe_numa_memory_placement(
                pid, expected_node, before_start, proc_root
            )
        except OlmoeEpBenchmarkError as error:
            raise OlmoeEpBenchmarkError(
                f"TP rank {tp_rank} memory placement is invalid: {error}"
            ) from error
        try:
            second_tasks = tuple(
                sorted(
                    int(path.name)
                    for path in (proc_root / str(pid) / "task").iterdir()
                    if path.name.isdigit()
                )
            )
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot re-enumerate scheduler {pid} threads: {error}"
            ) from error
        after_group, after_start, after_state = _read_process_stat_at(pid, proc_root)
        if (
            after_group != before_group
            or after_start != before_start
            or after_state == "Z"
        ):
            raise OlmoeEpBenchmarkError("scheduler process identity changed")
        if first_tasks == second_tasks:
            task_affinity_union: set[int] = set()
            for affinity in task_affinities.values():
                task_affinity_union.update(affinity)
            task_affinity_sets = cast(
                list[JsonValue],
                [
                    {
                        "task_id": task_id,
                        "cpus": sorted(task_affinities[task_id]),
                    }
                    for task_id in first_tasks
                ],
            )
            return cast(
                JsonObject,
                {
                    "tp_rank": tp_rank,
                    "ep_rank": ep_rank,
                    "pid": pid,
                    "start_time_ticks": before_start,
                    "process_title": process_title,
                    "expected_node": expected_node,
                    "expected_node_cpus": sorted(expected_cpus),
                    "task_count": len(first_tasks),
                    "task_affinity_sets": task_affinity_sets,
                    "task_affinity_union": sorted(task_affinity_union),
                    "task_affinity_sha256": _canonical_sha256(task_affinity_sets),
                    "task_affinity_set_counts": [
                        {"cpus": list(cpus), "thread_count": count}
                        for cpus, count in sorted(affinity_counts.items())
                    ],
                    "numa_memory_placement": memory_placement,
                    "stable_observation_attempt": attempt + 1,
                },
            )
        time.sleep(0.05)
    raise OlmoeEpBenchmarkError("scheduler thread set did not stabilize")


def observe_rank_local_numa(
    *,
    process_ids: tuple[int, ...],
    process_group_id: int,
    expert_parallel_size: ExpertParallelSize,
    proc_root: Path = Path("/proc"),
    sysfs_root: Path = _CPU_SYSFS_ROOT,
) -> JsonObject:
    expected_node_cpus: list[frozenset[int]] = []
    for node, pinned in zip(DWAGON_NUMA_NODES, DWAGON_NUMA_LOGICAL_CPUS, strict=True):
        try:
            raw = (sysfs_root / f"node/node{node}/cpulist").read_text()
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"cannot read NUMA node {node} CPU set: {error}"
            ) from error
        observed = _parse_cpu_list(raw, f"NUMA node {node} CPU list")
        if observed != pinned:
            raise OlmoeEpBenchmarkError("dwagon logical CPU to NUMA mapping changed")
        expected_node_cpus.append(observed)

    scheduler_by_rank: dict[int, tuple[int, int | None, str]] = {}
    for pid in process_ids:
        title = _scheduler_process_title(pid, proc_root)
        match = _SCHEDULER_TITLE_PATTERN.fullmatch(title)
        if match is None:
            continue
        tp_rank = int(match.group("tp_rank"))
        raw_ep_rank = match.group("ep_rank")
        ep_rank = None if raw_ep_rank is None else int(raw_ep_rank)
        expected_ep_rank = None if expert_parallel_size == 1 else tp_rank
        if ep_rank != expected_ep_rank or tp_rank in scheduler_by_rank:
            raise OlmoeEpBenchmarkError("scheduler rank titles are not canonical")
        scheduler_by_rank[tp_rank] = (pid, ep_rank, title)
    if set(scheduler_by_rank) != {0, 1}:
        raise OlmoeEpBenchmarkError("exactly one scheduler per TP rank is required")

    ranks = [
        _observe_scheduler_numa(
            pid=scheduler_by_rank[rank][0],
            process_group_id=process_group_id,
            tp_rank=rank,
            ep_rank=scheduler_by_rank[rank][1],
            process_title=scheduler_by_rank[rank][2],
            expected_node=rank,
            expected_cpus=expected_node_cpus[rank],
            proc_root=proc_root,
        )
        for rank in (0, 1)
    ]
    return cast(
        JsonObject,
        {
            "status": "rank_local_numa_verified",
            "expert_parallel_size": expert_parallel_size,
            "process_group_id": process_group_id,
            "ranks": ranks,
            "observed_at_utc": _utc_now(),
        },
    )


def verify_rank_local_numa(running: RunningServerProcess) -> JsonObject:
    assert_server_alive(running, "rank-local NUMA verification")
    members = _owned_group_members(running.owned)
    evidence = observe_rank_local_numa(
        process_ids=members,
        process_group_id=running.owned.process_group_id,
        expert_parallel_size=cast(
            ExpertParallelSize,
            int(running.owned.command[running.owned.command.index("--ep-size") + 1]),
        ),
    )
    assert_server_alive(running, "rank-local NUMA verification")
    return evidence


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, payload: JsonObject) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _write_ownership_journal(
    config: OlmoeEpBenchmarkConfig, owned: OwnedServerProcess
) -> None:
    _write_private_json(
        config.result_directory / _OWNERSHIP_JOURNAL_FILENAME,
        cast(
            JsonObject,
            {
                "schema_version": 1,
                "status": "active",
                "run_id": config.run_id,
                "updated_at_utc": _utc_now(),
                "process": asdict(owned),
            },
        ),
    )


def start_server(
    config: OlmoeEpBenchmarkConfig,
    owner_token: str,
    ownership_namespace: str,
    moe_config_snapshot: MoeConfigSnapshotAdmission | None = None,
) -> RunningServerProcess:
    source_admission = verify_moe_config(config)
    if source_admission is None:
        if moe_config_snapshot is not None:
            raise OlmoeEpBenchmarkError("unexpected MoE config snapshot")
    else:
        if (
            moe_config_snapshot is None
            or source_admission != moe_config_snapshot.source
        ):
            raise OlmoeEpBenchmarkError(
                "external MoE config changed before server launch"
            )
        verify_moe_config_snapshot(moe_config_snapshot)
    verify_port_vacant(config.host, config.port)
    command = build_server_command(config)
    environment = build_server_environment(
        config,
        owner_token,
        ownership_namespace,
        moe_config_snapshot=moe_config_snapshot,
    )
    log_path = config.result_directory / _SERVER_LOG_FILENAME
    log_file = log_path.open("xb", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    try:
        process_group_id, start_time_ticks, _state = _read_process_stat(process.pid)
        if process_group_id != process.pid:
            raise OlmoeEpBenchmarkError("native SGLang did not create an owned session")
        owned = OwnedServerProcess(
            pid=process.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=ownership_namespace,
            command=command,
            launch_environment=tuple(
                sorted(
                    (name, value)
                    for name, value in environment.items()
                    if name != _OWNER_TOKEN_ENVIRONMENT
                )
            ),
            log_path=str(log_path),
        )
        _write_ownership_journal(config, owned)
        return RunningServerProcess(owned=owned, process=process, log_file=log_file)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)
        log_file.close()
        raise


def stop_server(running: RunningServerProcess, timeout_seconds: float) -> JsonObject:
    owned = running.owned
    started = time.monotonic()
    term_sent = False
    kill_sent = False
    failure: str | None = None
    try:
        members_before = _owned_group_members(owned)
        if members_before:
            if owned.pid in members_before:
                observed_group, observed_start, _state = _read_process_stat(owned.pid)
                if (
                    observed_group != owned.process_group_id
                    or observed_start != owned.start_time_ticks
                ):
                    raise OlmoeEpBenchmarkError("native SGLang leader identity changed")
            os.killpg(owned.process_group_id, signal.SIGTERM)
            term_sent = True
            deadline = time.monotonic() + timeout_seconds
            while (
                _process_group_members(owned.process_group_id)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            if _process_group_members(owned.process_group_id):
                _owned_group_members(owned)
                os.killpg(owned.process_group_id, signal.SIGKILL)
                kill_sent = True
                deadline = time.monotonic() + min(timeout_seconds, 5.0)
                while (
                    _process_group_members(owned.process_group_id)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
        group_empty = not _process_group_members(owned.process_group_id)
        if not group_empty:
            failure = "owned process group remained after SIGKILL"
    except BaseException as error:
        members_before = _process_group_members(owned.process_group_id)
        group_empty = not members_before
        failure = f"{type(error).__name__}: {error}"
    try:
        return_code = running.process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return_code = None
    if return_code is not None and not term_sent and failure is None:
        failure = f"native SGLang exited before managed cleanup with code {return_code}"
    running.log_file.close()
    return {
        "pid": owned.pid,
        "process_group_id": owned.process_group_id,
        "owner_token_sha256": hashlib.sha256(owned.owner_token.encode()).hexdigest(),
        "ownership_namespace": owned.ownership_namespace,
        "members_before": list(members_before),
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "return_code": return_code,
        "group_empty": group_empty,
        "cleanup_complete": group_empty and failure is None,
        "failure": failure,
        "elapsed_seconds": time.monotonic() - started,
    }


def _wait_for_readiness(
    running: RunningServerProcess,
    client: OlmoeNativeServingClient,
    timeout_seconds: float,
) -> JsonObject:
    started = time.monotonic()
    attempts = 0
    last_error = "not attempted"
    while time.monotonic() - started < timeout_seconds:
        attempts += 1
        assert_server_alive(running, "readiness")
        try:
            health = client.health_generate()
            if health.status_code == 200:
                assert_server_alive(running, "readiness health response")
                return {
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                    "status_code": health.status_code,
                    "response_sha256": health.response_sha256,
                }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    raise OlmoeEpBenchmarkError(
        f"native SGLang readiness timed out after {attempts} attempts: {last_error}"
    )


def verify_server_log_contract(
    log_path: Path,
    expert_parallel_size: ExpertParallelSize,
    moe_config_snapshot: MoeConfigSnapshotAdmission | None = None,
    *,
    max_total_tokens: int = OLMOE_MAX_TOTAL_TOKENS,
    max_running_requests: int = 1,
    cuda_graph_max_batch_size: int | None = None,
) -> JsonObject:
    """Bind the effective KV-pool and collective policy from startup logs."""

    try:
        with log_path.open("rb") as source:
            contents = source.read(_LOG_MAXIMUM_BYTES + 1)
    except OSError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read native SGLang startup log: {error}"
        ) from error
    if len(contents) > _LOG_MAXIMUM_BYTES:
        raise OlmoeEpBenchmarkError("native SGLang startup log exceeds size bound")
    try:
        lines = contents.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise OlmoeEpBenchmarkError(
            "native SGLang startup log is not valid UTF-8"
        ) from error

    server_args_lines = [line for line in lines if "server_args=ServerArgs(" in line]
    matching_server_args_lines = [
        line
        for line in server_args_lines
        if re.search(rf"\bmax_total_tokens={max_total_tokens}\b", line) is not None
        and (
            max_running_requests == 1
            or re.search(rf"\bmax_running_requests={max_running_requests}\b", line)
            is not None
        )
        and (
            cuda_graph_max_batch_size is None
            or re.search(rf"\bcuda_graph_max_bs={cuda_graph_max_batch_size}\b", line)
            is not None
        )
        and _DISABLED_CUSTOM_ALL_REDUCE_SERVER_ARGS_PATTERN.search(line) is not None
    ]
    kv_cache_allocation_lines = [
        line for line in lines if "KV Cache is allocated." in line
    ]
    matching_kv_cache_lines: list[tuple[str, re.Match[str]]] = []
    for line in kv_cache_allocation_lines:
        if match := _KV_CACHE_ALLOCATION_PATTERN.search(line):
            matching_kv_cache_lines.append((line, match))
    matching_rank_bindings = {
        (match.group("tp_rank"), match.group("ep_rank"))
        for _, match in matching_kv_cache_lines
        if int(match.group("token_count")) == max_total_tokens
    }
    matching_kv_cache_lines = [
        (line, match)
        for line, match in matching_kv_cache_lines
        if int(match.group("token_count")) == max_total_tokens
    ]
    expected_rank_bindings = (
        {("0", None), ("1", None)}
        if expert_parallel_size == 1
        else {("0", "0"), ("1", "1")}
    )
    custom_all_reduce_failures = [
        line for line in lines if "Setup Custom allreduce failed" in line
    ]
    if len(server_args_lines) != 1 or len(matching_server_args_lines) != 1:
        raise OlmoeEpBenchmarkError(
            "native SGLang log does not verify the exact server argument contract"
        )
    if (
        len(kv_cache_allocation_lines) != 2
        or len(matching_kv_cache_lines) != 2
        or matching_rank_bindings != expected_rank_bindings
    ):
        raise OlmoeEpBenchmarkError(
            "native SGLang log does not verify the expected TP/EP rank-bound "
            f"{max_total_tokens}-token KV allocations"
        )
    if custom_all_reduce_failures:
        raise OlmoeEpBenchmarkError(
            "native SGLang log reports a custom all-reduce setup failure"
        )
    moe_config_loads: list[JsonValue] = []
    if moe_config_snapshot is not None:
        effective_admission = moe_config_snapshot.effective
        all_config_load_lines = [
            line for line in lines if "Using MoE kernel config from " in line
        ]
        rejected_config_warning_lines = [
            line
            for line in lines
            if any(
                marker in line
                for marker in (
                    "Using default MoE kernel config",
                    "Using MoE kernel config with down_moe=False",
                    "Config file not found at ",
                    "Fallback to triton version",
                )
            )
        ]
        if len(all_config_load_lines) != 4 or rejected_config_warning_lines:
            raise OlmoeEpBenchmarkError(
                "native SGLang log contains an extra MoE config load, default, "
                "or fallback warning"
            )
        for file_admission in effective_admission.files:
            absolute_path = (
                Path(effective_admission.root) / file_admission.relative_path
            )
            marker = f"Using MoE kernel config from {absolute_path}."
            matches_by_rank = [
                [
                    line
                    for line in lines
                    if marker in line
                    and (
                        f"TP{rank}] {marker}" in line
                        if expert_parallel_size == 1
                        else f"TP{rank} EP{rank}] {marker}" in line
                    )
                ]
                for rank in range(2)
            ]
            if any(len(rank_matches) != 1 for rank_matches in matches_by_rank):
                raise OlmoeEpBenchmarkError(
                    "native SGLang log does not verify both TP ranks loaded every "
                    "admitted MoE config"
                )
            matches = [rank_matches[0] for rank_matches in matches_by_rank]
            moe_config_loads.append(
                {
                    "relative_path": file_admission.relative_path,
                    "sha256": file_admission.sha256,
                    "rank_load_count": len(matches),
                    "load_line_sha256": [
                        hashlib.sha256(line.encode("utf-8")).hexdigest()
                        for line in matches
                    ],
                }
            )

    receipt: JsonObject = {
        "status": "verified",
        "observed_bytes": len(contents),
        "observed_sha256": hashlib.sha256(contents).hexdigest(),
        "server_args_line_sha256": hashlib.sha256(
            matching_server_args_lines[0].encode("utf-8")
        ).hexdigest(),
        "server_args_line_count": len(server_args_lines),
        "kv_cache_allocation_line_count": len(matching_kv_cache_lines),
        "kv_cache_allocation_rank_bindings": [
            {
                "tp_rank": int(tp_rank),
                "ep_rank": None if ep_rank is None else int(ep_rank),
            }
            for tp_rank, ep_rank in sorted(
                matching_rank_bindings,
                key=lambda binding: binding[0],
            )
        ],
        "expert_parallel_size": expert_parallel_size,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": max_running_requests,
        "cuda_graph_max_batch_size": cuda_graph_max_batch_size,
        "custom_all_reduce_disabled": True,
        "custom_all_reduce_failure_line_count": len(custom_all_reduce_failures),
    }
    if moe_config_snapshot is not None:
        receipt["moe_kernel_config"] = {
            "source_file_set_sha256": moe_config_snapshot.source.file_set_sha256,
            "effective_file_set_sha256": (
                moe_config_snapshot.effective.file_set_sha256
            ),
            "effective_root": moe_config_snapshot.effective.root,
            "loads": moe_config_loads,
        }
    return receipt


def _verify_server_info(
    response: Mapping[str, object], config: OlmoeEpBenchmarkConfig
) -> JsonObject:
    max_total_tokens, max_running_requests, cuda_graph_max_batch_size = (
        _server_capacity(config)
    )
    expected: dict[str, object] = {
        "version": "0.0.0.dev0",
        "model_path": config.model_path,
        "host": config.host,
        "port": config.port,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": config.expert_parallel_size,
        "nnodes": 1,
        "node_rank": 0,
        "dtype": "bfloat16",
        "context_length": OLMOE_CONTEXT_LENGTH,
        "max_total_tokens": max_total_tokens,
        "mem_fraction_static": config.static_memory_fraction,
        "max_running_requests": max_running_requests,
        "random_seed": CANONICAL_SAMPLING_SEED,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "numa_node": [0, 1],
    }
    if cuda_graph_max_batch_size is not None:
        expected["cuda_graph_max_bs"] = cuda_graph_max_batch_size
    mismatches = {
        key: {"expected": value, "actual": response.get(key)}
        for key, value in expected.items()
        if response.get(key) != value
    }
    if mismatches:
        raise OlmoeEpBenchmarkError(
            f"native SGLang server_info differs from launch: {mismatches}"
        )
    return {
        "canonical_response_sha256": _canonical_sha256(
            cast(JsonObject, dict(response))
        ),
        "verified_fields": cast(JsonObject, expected),
        "version": cast(JsonValue, response.get("version")),
    }


def _run_exact_sanity(
    client: OlmoeNativeServingClient, oracle: SanityCapture
) -> JsonObject:
    request = OlmoeNativeGenerateRequest(
        input_ids=oracle.input_ids,
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=oracle.max_new_tokens,
            temperature=0.0,
            ignore_eos=False,
            sampling_seed=oracle.sampling_seed,
        ),
        stream=False,
        return_logprob=False,
        log_metrics=False,
    )
    response = client.generate_sanity(request)
    output_ids = tuple(response.output_ids)
    output_text_sha256 = hashlib.sha256(response.text.encode()).hexdigest()
    if (
        output_ids != oracle.output_ids
        or response.text != oracle.output_text
        or output_text_sha256 != oracle.output_text_sha256
        or response.prompt_tokens != len(oracle.input_ids)
        or response.completion_tokens != len(output_ids)
        or response.cached_tokens != 0
    ):
        raise OlmoeEpBenchmarkError("OLMoE deterministic sanity oracle mismatch")
    flush = client.flush_cache()
    return {
        "status": "exact_match",
        "prompt_text": oracle.prompt_text,
        "input_token_count": len(oracle.input_ids),
        "input_ids_sha256": _canonical_sha256(list(oracle.input_ids)),
        "expected_output_ids": list(oracle.output_ids),
        "output_ids_sha256": _canonical_sha256(list(output_ids)),
        "output_text_sha256": output_text_sha256,
        "expected_output_text": oracle.output_text,
        "server_output_text": response.text,
        "max_new_tokens": oracle.max_new_tokens,
        "sampling_seed": oracle.sampling_seed,
        "temperature": 0.0,
        "ignore_eos": False,
        "finish_reason": response.finish_reason,
        "client_seconds": response.total_client_seconds,
        "post_sanity_flush_status_code": flush.status_code,
        "post_sanity_flush_response_sha256": flush.response_sha256,
    }


def build_deterministic_input_ids(kind: str, token_count: int) -> tuple[int, ...]:
    modulus = OLMOE_VOCABULARY_SIZE - 100
    token_ids: list[int] = []
    counter = 0
    while len(token_ids) < token_count:
        digest = hashlib.sha256(f"exo-olmoe-ep-v1:{kind}:{counter}".encode()).digest()
        for offset in range(0, len(digest), 4):
            value = int.from_bytes(digest[offset : offset + 4], "big")
            token_ids.append(100 + value % modulus)
            if len(token_ids) == token_count:
                break
        counter += 1
    return tuple(token_ids)


def _run_logit_parity_probe(client: OlmoeNativeServingClient) -> JsonObject:
    canonical_input_ids = build_deterministic_input_ids(
        "decode", LOGIT_PARITY_CANONICAL_INPUT_TOKENS
    )
    context_ids = canonical_input_ids + LOGIT_PARITY_COMMON_PREFIX
    request = OlmoeLogitParityRequest(
        input_ids=context_ids,
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=1,
            temperature=0.0,
            ignore_eos=True,
            sampling_seed=CANONICAL_SAMPLING_SEED,
        ),
        candidate_token_ids=LOGIT_PARITY_CANDIDATE_TOKEN_IDS,
        top_logprobs_num=LOGIT_PARITY_TOP_LOGPROBS,
    )
    observation: LogitParityObservation = client.generate_logit_parity(request)
    flush = client.flush_cache()
    return {
        "schema_version": 1,
        "status": "captured",
        "context": {
            "canonical_workload_kind": "decode",
            "canonical_input_token_count": len(canonical_input_ids),
            "canonical_input_ids_sha256": _canonical_sha256(list(canonical_input_ids)),
            "common_generated_prefix_token_ids": list(LOGIT_PARITY_COMMON_PREFIX),
            "common_generated_prefix_sha256": _canonical_sha256(
                list(LOGIT_PARITY_COMMON_PREFIX)
            ),
            "context_token_count": len(context_ids),
            "context_token_ids_sha256": observation.input_ids_sha256,
        },
        "request": {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "sampling_seed": CANONICAL_SAMPLING_SEED,
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": -1,
            "candidate_token_ids": list(LOGIT_PARITY_CANDIDATE_TOKEN_IDS),
            "top_logprobs_num": LOGIT_PARITY_TOP_LOGPROBS,
        },
        "response": {
            "raw_response_sha256": observation.response_sha256,
            "generated_token_id": observation.generated_token_id,
            "generated_token_logprob": observation.generated_token_logprob,
            "candidate_logprobs": [
                {
                    "token_id": entry.token_id,
                    "logprob": entry.logprob,
                }
                for entry in observation.candidate_logprobs
            ],
            "top_logprobs": [
                {
                    "token_id": entry.token_id,
                    "logprob": entry.logprob,
                }
                for entry in observation.top_logprobs
            ],
            "prompt_tokens": observation.prompt_tokens,
            "completion_tokens": observation.completion_tokens,
            "cached_tokens": observation.cached_tokens,
            "finish_reason": observation.finish_reason,
            "client_seconds": observation.total_client_seconds,
        },
        "post_probe_flush_status_code": flush.status_code,
        "post_probe_flush_response_sha256": flush.response_sha256,
    }


def _invocation_receipt(
    ordinal: int,
    flush_status_code: int,
    flush_sha256: str,
    observation: GenerateObservation,
) -> JsonObject:
    return {
        "ordinal": ordinal,
        "cache_flush_status_code": flush_status_code,
        "cache_flush_response_sha256": flush_sha256,
        "input_ids_sha256": observation.input_ids_sha256,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "cached_tokens": observation.cached_tokens,
        "output_ids_sha256": observation.output_ids_sha256,
        "output_ids": list(observation.output_ids),
        "finish_reason_sha256": observation.finish_reason_sha256,
        "stream_line_count": observation.stream_line_count,
        "stream_event_count": observation.stream_event_count,
        "output_bearing_event_count": observation.output_bearing_event_count,
        "maximum_stream_line_bytes": observation.maximum_stream_line_bytes,
        "first_stream_event_output_tokens": (
            observation.first_stream_event_output_tokens
        ),
        "total_client_seconds": observation.total_client_seconds,
        "client_observed_ttft_seconds": observation.client_observed_ttft_seconds,
        "client_observed_generation_window_seconds": (
            observation.client_observed_generation_window_seconds
        ),
        "client_observed_decode_tokens_per_second": (
            observation.client_observed_decode_tokens_per_second
        ),
    }


def _run_invocation(
    client: OlmoeNativeServingClient,
    request: OlmoeNativeGenerateRequest,
    ordinal: int,
    phase: str,
    assert_after_phase: Callable[[str], None],
) -> JsonObject:
    flush = client.flush_cache()
    assert_after_phase(f"{phase} cache flush")
    observation = client.generate(request)
    assert_after_phase(phase)
    return _invocation_receipt(
        ordinal,
        flush.status_code,
        flush.response_sha256,
        observation,
    )


def run_canonical_workload(
    client: OlmoeNativeServingClient,
    kind: Literal["prefill", "decode"],
    input_tokens: int,
    output_tokens: int,
    assert_after_phase: Callable[[str], None],
) -> JsonObject:
    input_ids = build_deterministic_input_ids(kind, input_tokens)
    request = OlmoeNativeGenerateRequest(
        input_ids=input_ids,
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=output_tokens,
            temperature=0.0,
            ignore_eos=True,
            sampling_seed=CANONICAL_SAMPLING_SEED,
        ),
        stream=True,
        return_logprob=False,
        log_metrics=True,
    )
    warmups = [
        _run_invocation(
            client,
            request,
            ordinal,
            f"{kind} warmup {ordinal}",
            assert_after_phase,
        )
        for ordinal in range(1, CANONICAL_WARMUP_COUNT + 1)
    ]
    samples = [
        _run_invocation(
            client,
            request,
            ordinal,
            f"{kind} sample {ordinal}",
            assert_after_phase,
        )
        for ordinal in range(1, CANONICAL_SAMPLE_COUNT + 1)
    ]
    decode_rates = [
        cast(float, sample["client_observed_decode_tokens_per_second"])
        for sample in samples
    ]
    total_rates = [
        output_tokens / cast(float, sample["total_client_seconds"])
        for sample in samples
    ]
    ttfts = [cast(float, sample["client_observed_ttft_seconds"]) for sample in samples]
    return cast(
        JsonObject,
        {
            "kind": kind,
            "request": {
                "input_token_count": input_tokens,
                "input_ids_sha256": _canonical_sha256(list(input_ids)),
                "output_token_count": output_tokens,
                "sampling_seed": CANONICAL_SAMPLING_SEED,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
            },
            "warmup_count": CANONICAL_WARMUP_COUNT,
            "sample_count": CANONICAL_SAMPLE_COUNT,
            "warmups": warmups,
            "samples": samples,
            "summary": {
                "median_client_decode_tokens_per_second": statistics.median(
                    decode_rates
                ),
                "median_end_to_end_output_tokens_per_second": statistics.median(
                    total_rates
                ),
                "median_client_ttft_seconds": statistics.median(ttfts),
            },
        },
    )


def build_concurrency_input_ids(
    kind: Literal["prefill", "decode"], token_count: int, lane: int
) -> tuple[int, ...]:
    if lane < 0 or lane >= OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS:
        raise ValueError("concurrency lane is outside the admitted range")
    return build_deterministic_input_ids(f"{kind}:concurrency-lane-{lane}", token_count)


def _aggregate_concurrency_metrics(
    observations: tuple[GenerateObservation, ...], release_monotonic_ns: int
) -> JsonObject:
    if not observations or release_monotonic_ns < 0:
        raise OlmoeEpBenchmarkError("aggregate concurrency timing is empty or invalid")
    for observation in observations:
        if not (
            release_monotonic_ns
            <= observation.request_started_monotonic_ns
            < observation.first_output_monotonic_ns
            < observation.last_output_monotonic_ns
            <= observation.request_completed_monotonic_ns
        ):
            raise OlmoeEpBenchmarkError(
                "aggregate concurrency lane timing is not monotonic from release"
            )
        if (
            observation.completion_tokens
            <= observation.first_stream_event_output_tokens
        ):
            raise OlmoeEpBenchmarkError(
                "aggregate concurrency lane has no post-first-event decode tokens"
            )

    latest_completion_ns = max(
        observation.request_completed_monotonic_ns for observation in observations
    )
    earliest_first_output_ns = min(
        observation.first_output_monotonic_ns for observation in observations
    )
    latest_last_output_ns = max(
        observation.last_output_monotonic_ns for observation in observations
    )
    group_wall_ns = latest_completion_ns - release_monotonic_ns
    decode_envelope_ns = latest_last_output_ns - earliest_first_output_ns
    if group_wall_ns <= 0 or decode_envelope_ns <= 0:
        raise OlmoeEpBenchmarkError("aggregate concurrency denominator is not positive")
    total_completion_tokens = sum(
        observation.completion_tokens for observation in observations
    )
    total_decode_tokens = sum(
        observation.completion_tokens - observation.first_stream_event_output_tokens
        for observation in observations
    )
    group_wall_seconds = group_wall_ns / 1_000_000_000
    decode_envelope_seconds = decode_envelope_ns / 1_000_000_000
    return {
        "total_completion_tokens": total_completion_tokens,
        "group_wall_seconds": group_wall_seconds,
        "aggregate_end_to_end_output_tokens_per_second": (
            total_completion_tokens / group_wall_seconds
        ),
        "total_tokens_after_first_events": total_decode_tokens,
        "decode_envelope_seconds": decode_envelope_seconds,
        "aggregate_decode_tokens_per_second": (
            total_decode_tokens / decode_envelope_seconds
        ),
        "earliest_first_output_monotonic_ns": earliest_first_output_ns,
        "latest_last_output_monotonic_ns": latest_last_output_ns,
        "latest_request_completion_monotonic_ns": latest_completion_ns,
        "formulas": {
            "end_to_end": (
                "sum(lane completion tokens) / "
                "(latest lane request completion - common barrier release)"
            ),
            "decode": (
                "sum(lane completion tokens - lane first-event tokens) / "
                "(latest lane last-token event - earliest lane first-token event)"
            ),
            "per_lane_rates_summed": False,
        },
    }


def _concurrency_lane_receipt(
    lane: int,
    observation: GenerateObservation,
    release_monotonic_ns: int,
) -> JsonObject:
    return {
        "lane": lane,
        "input_ids_sha256": observation.input_ids_sha256,
        "prompt_tokens": observation.prompt_tokens,
        "completion_tokens": observation.completion_tokens,
        "cached_tokens": observation.cached_tokens,
        "output_ids": list(observation.output_ids),
        "output_ids_sha256": observation.output_ids_sha256,
        "finish_reason_sha256": observation.finish_reason_sha256,
        "stream_line_count": observation.stream_line_count,
        "stream_event_count": observation.stream_event_count,
        "output_bearing_event_count": observation.output_bearing_event_count,
        "maximum_stream_line_bytes": observation.maximum_stream_line_bytes,
        "first_stream_event_output_tokens": (
            observation.first_stream_event_output_tokens
        ),
        "timing": {
            "request_started_monotonic_ns": observation.request_started_monotonic_ns,
            "first_output_monotonic_ns": observation.first_output_monotonic_ns,
            "last_output_monotonic_ns": observation.last_output_monotonic_ns,
            "request_completed_monotonic_ns": (
                observation.request_completed_monotonic_ns
            ),
            "request_start_after_release_seconds": (
                observation.request_started_monotonic_ns - release_monotonic_ns
            )
            / 1_000_000_000,
            "total_client_seconds": observation.total_client_seconds,
            "client_observed_ttft_seconds": observation.client_observed_ttft_seconds,
            "client_observed_generation_window_seconds": (
                observation.client_observed_generation_window_seconds
            ),
            "client_observed_decode_tokens_per_second": (
                observation.client_observed_decode_tokens_per_second
            ),
        },
    }


async def _run_concurrency_group(
    administrative_client: OlmoeNativeAsyncServingClient,
    lane_clients: tuple[OlmoeNativeAsyncServingClient, ...],
    requests: tuple[OlmoeNativeGenerateRequest, ...],
    *,
    kind: Literal["prefill", "decode"],
    phase: Literal["warmup", "sample"],
    ordinal: int,
    barrier_timeout_seconds: float,
    assert_after_phase: Callable[[str], None],
    cancellation_checkpoint: Callable[[], None],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[JsonObject, tuple[GenerateObservation, ...]]:
    concurrency = len(requests)
    if concurrency not in CONCURRENCY_LEVELS:
        raise OlmoeEpBenchmarkError("concurrency group size is not admitted")
    if len(lane_clients) < concurrency:
        raise OlmoeEpBenchmarkError("concurrency lane client pool is undersized")

    async def monitor_cancellation() -> None:
        while True:
            await asyncio.sleep(0.05)
            cancellation_checkpoint()

    cancellation_checkpoint()
    flush_task = asyncio.create_task(
        administrative_client.flush_cache(),
        name=f"olmoe-{kind}-c{concurrency}-cache-flush",
    )
    flush_monitor = asyncio.create_task(
        monitor_cancellation(),
        name=f"olmoe-{kind}-c{concurrency}-flush-signal-checkpoint",
    )
    try:
        flush_done, _flush_pending = await asyncio.wait(
            (flush_task, flush_monitor), return_when=asyncio.FIRST_COMPLETED
        )
        if flush_monitor in flush_done:
            exception = flush_monitor.exception()
            if exception is None:
                raise OlmoeEpBenchmarkError(
                    "cache-flush signal checkpoint stopped unexpectedly"
                )
            raise exception
        flush = flush_task.result()
    finally:
        flush_task.cancel()
        flush_monitor.cancel()
        await asyncio.gather(flush_task, flush_monitor, return_exceptions=True)
    cancellation_checkpoint()
    assert_after_phase(f"{kind} C{concurrency} {phase} {ordinal} cache flush")

    release = asyncio.Event()
    all_lanes_ready = asyncio.Event()
    ready_lanes: set[int] = set()

    async def run_lane(lane: int) -> GenerateObservation:
        ready_lanes.add(lane)
        if len(ready_lanes) == concurrency:
            all_lanes_ready.set()
        await asyncio.wait_for(release.wait(), timeout=barrier_timeout_seconds)
        return await lane_clients[lane].generate(requests[lane])

    lane_tasks = tuple(
        asyncio.create_task(
            run_lane(lane), name=f"olmoe-{kind}-c{concurrency}-lane-{lane}"
        )
        for lane in range(concurrency)
    )
    task_lanes = {task: lane for lane, task in enumerate(lane_tasks)}
    monitor_task = asyncio.create_task(
        monitor_cancellation(), name=f"olmoe-{kind}-c{concurrency}-signal-checkpoint"
    )
    ready_task = asyncio.create_task(
        all_lanes_ready.wait(), name=f"olmoe-{kind}-c{concurrency}-ready-rendezvous"
    )
    release_ns: int
    ready_lanes_at_release: tuple[int, ...]
    observations_by_lane: dict[int, GenerateObservation] = {}
    pending_lanes = set(lane_tasks)
    try:
        ready_done, _ready_pending = await asyncio.wait(
            (ready_task, monitor_task),
            timeout=barrier_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if monitor_task in ready_done:
            exception = monitor_task.exception()
            if exception is None:
                raise OlmoeEpBenchmarkError(
                    "concurrency signal checkpoint stopped unexpectedly"
                )
            raise exception
        if ready_task not in ready_done or ready_lanes != set(range(concurrency)):
            raise OlmoeEpBenchmarkError(
                f"{kind} C{concurrency} lane-ready rendezvous timed out"
            )
        ready_lanes_at_release = tuple(sorted(ready_lanes))
        release_ns = clock_ns()
        release.set()
        while pending_lanes:
            done, _pending = await asyncio.wait(
                (*pending_lanes, monitor_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor_task in done:
                exception = monitor_task.exception()
                if exception is None:
                    raise OlmoeEpBenchmarkError(
                        "concurrency signal checkpoint stopped unexpectedly"
                    )
                raise exception
            completed_lane_tasks = sorted(
                (
                    cast(asyncio.Task[GenerateObservation], raw_task)
                    for raw_task in done
                    if raw_task is not monitor_task
                ),
                key=task_lanes.__getitem__,
            )
            for task in completed_lane_tasks:
                lane = task_lanes[task]
                pending_lanes.remove(task)
                try:
                    observations_by_lane[lane] = task.result()
                except OlmoeEpManagedSignalError:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise OlmoeEpBenchmarkError(
                        f"{kind} C{concurrency} {phase} {ordinal} lane {lane} failed"
                    ) from error
    finally:
        monitor_task.cancel()
        ready_task.cancel()
        for task in pending_lanes:
            task.cancel()
        await asyncio.gather(
            monitor_task, ready_task, *pending_lanes, return_exceptions=True
        )

    observed = tuple(observations_by_lane[lane] for lane in range(concurrency))
    metrics = _aggregate_concurrency_metrics(observed, release_ns)
    assert_after_phase(f"{kind} C{concurrency} {phase} {ordinal}")
    receipt = cast(
        JsonObject,
        {
            "phase": phase,
            "ordinal": ordinal,
            "concurrency": concurrency,
            "cache_flush_status_code": flush.status_code,
            "cache_flush_response_sha256": flush.response_sha256,
            "barrier": {
                "participant_count": concurrency + 1,
                "lane_count": concurrency,
                "lane_ready_count_at_release": len(ready_lanes_at_release),
                "lane_ready_ordinals_at_release": list(ready_lanes_at_release),
                "mechanism": "asyncio_lane_ready_rendezvous_and_controller_release",
                "common_release_monotonic_ns": release_ns,
                "all_lane_requests_started_at_or_after_release": all(
                    observation.request_started_monotonic_ns >= release_ns
                    for observation in observed
                ),
            },
            "lanes": [
                _concurrency_lane_receipt(lane, observation, release_ns)
                for lane, observation in enumerate(observed)
            ],
            "aggregate": metrics,
        },
    )
    return receipt, observed


def _observe_concurrency_output_stability(
    *,
    kind: Literal["prefill", "decode"],
    requests: tuple[OlmoeNativeGenerateRequest, ...],
    observations: tuple[GenerateObservation, ...],
    references: dict[tuple[Literal["prefill", "decode"], int], tuple[int, ...]],
    stability_observations: dict[
        tuple[Literal["prefill", "decode"], int], list[tuple[str, bool]]
    ],
) -> JsonObject:
    if len(requests) != len(observations):
        raise OlmoeEpBenchmarkError("concurrency request/observation count differs")
    lane_evidence: list[JsonValue] = []
    for lane, (request, observation) in enumerate(
        zip(requests, observations, strict=True)
    ):
        expected_input_hash = token_ids_sha256(request.input_ids)
        expected_output_count = request.sampling_params.max_new_tokens
        observed_output_hash = token_ids_sha256(observation.output_ids)
        expected_finish_reason_hash = _canonical_sha256(
            {"type": "length", "length": expected_output_count}
        )
        if (
            observation.input_ids_sha256 != expected_input_hash
            or observation.prompt_tokens != len(request.input_ids)
            or observation.completion_tokens != expected_output_count
            or len(observation.output_ids) != expected_output_count
            or observation.cached_tokens != 0
            or observation.output_ids_sha256 != observed_output_hash
            or observation.finish_reason_sha256 != expected_finish_reason_hash
        ):
            raise OlmoeEpBenchmarkError(
                f"{kind} concurrency lane {lane} output is not coherent"
            )
        key = (kind, lane)
        reference = references.setdefault(key, observation.output_ids)
        reference_hash = token_ids_sha256(reference)
        exact_match = observation.output_ids == reference
        stability_observations.setdefault(key, []).append(
            (observation.output_ids_sha256, exact_match)
        )
        differences = [
            {
                "output_index": index,
                "reference_token_id": reference_token_id,
                "observed_token_id": observed_token_id,
            }
            for index, (reference_token_id, observed_token_id) in enumerate(
                zip(reference, observation.output_ids, strict=True)
            )
            if reference_token_id != observed_token_id
        ]
        known_token_4_tie_observed = (
            kind == "decode"
            and lane == 0
            and len(differences) == 1
            and differences[0]["output_index"] == 3
            and {
                differences[0]["reference_token_id"],
                differences[0]["observed_token_id"],
            }
            == set(LOGIT_PARITY_CANDIDATE_TOKEN_IDS)
        )
        lane_evidence.append(
            cast(
                JsonObject,
                {
                    "lane": lane,
                    "input_ids_sha256": expected_input_hash,
                    "reference_output_ids_sha256": reference_hash,
                    "observed_output_ids_sha256": observation.output_ids_sha256,
                    "finish_reason_sha256": observation.finish_reason_sha256,
                    "structural_coherence_status": "verified",
                    "exact_reference_match": exact_match,
                    "status": "exact_match" if exact_match else "coherent_variation",
                    "variation": (
                        None
                        if exact_match
                        else {
                            "differing_token_count": len(differences),
                            "differences": differences,
                            "known_decode_lane_0_token_4_tie_observed": (
                                known_token_4_tie_observed
                            ),
                        }
                    ),
                },
            )
        )
    all_exact = all(
        cast(bool, cast(dict[str, object], lane)["exact_reference_match"])
        for lane in lane_evidence
    )
    return {
        "status": "exact_match" if all_exact else "coherent_variation_observed",
        "policy": {
            "structural_coherence": "required",
            "exact_output_sequence_match": "observational_not_required",
            "reference": "first_coherent_observation_for_each_kind_and_lane",
        },
        "lanes": lane_evidence,
        "known_token_4_tie": {
            "candidate_token_ids": list(LOGIT_PARITY_CANDIDATE_TOKEN_IDS),
            "decode_lane": 0,
            "output_index": 3,
            "role": "observational_classification_only",
        },
    }


def _concurrency_output_stability_summary(
    kind: Literal["prefill", "decode"],
    references: Mapping[tuple[Literal["prefill", "decode"], int], tuple[int, ...]],
    stability_observations: Mapping[
        tuple[Literal["prefill", "decode"], int], list[tuple[str, bool]]
    ],
) -> JsonObject:
    lanes: list[JsonValue] = []
    for lane in range(OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS):
        key = (kind, lane)
        if key not in references or key not in stability_observations:
            continue
        observations = stability_observations[key]
        distinct_hashes = sorted({output_hash for output_hash, _exact in observations})
        exact_count = sum(1 for _output_hash, exact in observations if exact)
        lanes.append(
            cast(
                JsonObject,
                {
                    "lane": lane,
                    "reference_output_ids_sha256": token_ids_sha256(references[key]),
                    "observation_count": len(observations),
                    "exact_reference_match_observation_count": exact_count,
                    "variation_observation_count": len(observations) - exact_count,
                    "distinct_output_ids_sha256_count": len(distinct_hashes),
                    "distinct_output_ids_sha256": distinct_hashes,
                },
            )
        )
    variation_count = sum(
        cast(int, cast(dict[str, object], lane)["variation_observation_count"])
        for lane in lanes
    )
    return cast(
        JsonObject,
        {
            "status": (
                "coherent_variation_observed" if variation_count else "exact_match"
            ),
            "structural_coherence_required": True,
            "exact_output_sequence_match_required": False,
            "exact_output_sequence_match_role": "observational",
            "variation_observation_count": variation_count,
            "lanes": lanes,
        },
    )


async def _run_aggregate_concurrency_benchmark(
    administrative_client: OlmoeNativeAsyncServingClient,
    lane_clients: tuple[OlmoeNativeAsyncServingClient, ...],
    request_timeout_seconds: float,
    assert_after_phase: Callable[[str], None],
    cancellation_checkpoint: Callable[[], None],
) -> list[JsonValue]:
    if len(lane_clients) != OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS:
        raise OlmoeEpBenchmarkError(
            "aggregate concurrency requires one persistent client per admitted lane"
        )
    references: dict[tuple[Literal["prefill", "decode"], int], tuple[int, ...]] = {}
    stability_observations: dict[
        tuple[Literal["prefill", "decode"], int], list[tuple[str, bool]]
    ] = {}
    workloads: list[JsonValue] = []
    for raw_kind, input_tokens, output_tokens in CANONICAL_WORKLOADS:
        kind = cast(Literal["prefill", "decode"], raw_kind)
        concurrency_results: list[JsonValue] = []
        for concurrency in CONCURRENCY_LEVELS:
            requests = tuple(
                OlmoeNativeGenerateRequest(
                    input_ids=build_concurrency_input_ids(kind, input_tokens, lane),
                    sampling_params=OlmoeSamplingParameters(
                        max_new_tokens=output_tokens,
                        temperature=0.0,
                        ignore_eos=True,
                        sampling_seed=CANONICAL_SAMPLING_SEED,
                    ),
                    stream=True,
                    return_logprob=False,
                    log_metrics=True,
                )
                for lane in range(concurrency)
            )
            if len({request.input_ids for request in requests}) != concurrency:
                raise OlmoeEpBenchmarkError(
                    f"{kind} C{concurrency} prompts are not distinct"
                )
            warmups: list[JsonValue] = []
            samples: list[JsonValue] = []
            for phase, count, destination in (
                ("warmup", CANONICAL_WARMUP_COUNT, warmups),
                ("sample", CANONICAL_SAMPLE_COUNT, samples),
            ):
                for ordinal in range(1, count + 1):
                    group, observations = await _run_concurrency_group(
                        administrative_client,
                        lane_clients,
                        requests,
                        kind=kind,
                        phase=cast(Literal["warmup", "sample"], phase),
                        ordinal=ordinal,
                        barrier_timeout_seconds=request_timeout_seconds,
                        assert_after_phase=assert_after_phase,
                        cancellation_checkpoint=cancellation_checkpoint,
                    )
                    group["output_stability"] = _observe_concurrency_output_stability(
                        kind=kind,
                        requests=requests,
                        observations=observations,
                        references=references,
                        stability_observations=stability_observations,
                    )
                    destination.append(group)
            sample_aggregates = [
                cast(dict[str, object], cast(dict[str, object], sample)["aggregate"])
                for sample in samples
            ]
            concurrency_results.append(
                {
                    "concurrency": concurrency,
                    "request": {
                        "input_token_count_per_lane": input_tokens,
                        "input_ids_sha256_by_lane": [
                            token_ids_sha256(request.input_ids) for request in requests
                        ],
                        "distinct_input_count": len(
                            {request.input_ids for request in requests}
                        ),
                        "output_token_count_per_lane": output_tokens,
                        "sampling_seed": CANONICAL_SAMPLING_SEED,
                        "temperature": 0.0,
                        "ignore_eos": True,
                        "stream": True,
                    },
                    "warmup_count": CANONICAL_WARMUP_COUNT,
                    "sample_count": CANONICAL_SAMPLE_COUNT,
                    "warmups": warmups,
                    "samples": samples,
                    "summary": {
                        "median_aggregate_decode_tokens_per_second": (
                            statistics.median(
                                cast(
                                    float,
                                    aggregate["aggregate_decode_tokens_per_second"],
                                )
                                for aggregate in sample_aggregates
                            )
                        ),
                        "median_aggregate_end_to_end_output_tokens_per_second": (
                            statistics.median(
                                cast(
                                    float,
                                    aggregate[
                                        "aggregate_end_to_end_output_tokens_per_second"
                                    ],
                                )
                                for aggregate in sample_aggregates
                            )
                        ),
                        "median_group_wall_seconds": statistics.median(
                            cast(float, aggregate["group_wall_seconds"])
                            for aggregate in sample_aggregates
                        ),
                        "median_decode_envelope_seconds": statistics.median(
                            cast(float, aggregate["decode_envelope_seconds"])
                            for aggregate in sample_aggregates
                        ),
                    },
                }
            )
        workloads.append(
            {
                "kind": kind,
                "concurrency_levels": list(CONCURRENCY_LEVELS),
                "lane_client_pool": {
                    "size": len(lane_clients),
                    "connection_policy": (
                        "one_persistent_independent_http_client_per_lane"
                    ),
                    "execution_model": "cancellable_asyncio_tasks_without_executor_threads",
                },
                "output_stability_summary": _concurrency_output_stability_summary(
                    kind, references, stability_observations
                ),
                "results": concurrency_results,
            }
        )
    return workloads


async def _run_aggregate_concurrency_session(
    base_url: str,
    request_timeout_seconds: float,
    assert_after_phase: Callable[[str], None],
    cancellation_checkpoint: Callable[[], None],
) -> list[JsonValue]:
    async with contextlib.AsyncExitStack() as client_stack:
        administrative_client = await client_stack.enter_async_context(
            OlmoeNativeAsyncServingClient(
                base_url, timeout_seconds=request_timeout_seconds
            )
        )
        lane_clients = tuple(
            [
                await client_stack.enter_async_context(
                    OlmoeNativeAsyncServingClient(
                        base_url, timeout_seconds=request_timeout_seconds
                    )
                )
                for _lane in range(OLMOE_CONCURRENCY_MAX_RUNNING_REQUESTS)
            ]
        )
        return await _run_aggregate_concurrency_benchmark(
            administrative_client,
            lane_clients,
            request_timeout_seconds,
            assert_after_phase,
            cancellation_checkpoint,
        )


def run_aggregate_concurrency_benchmark(
    base_url: str,
    request_timeout_seconds: float,
    assert_after_phase: Callable[[str], None],
    cancellation_checkpoint: Callable[[], None],
) -> list[JsonValue]:
    return asyncio.run(
        _run_aggregate_concurrency_session(
            base_url,
            request_timeout_seconds,
            assert_after_phase,
            cancellation_checkpoint,
        )
    )


def expert_parallel_semantics(expert_parallel_size: ExpertParallelSize) -> JsonObject:
    true_ep = expert_parallel_size == 2
    return {
        "classification": (
            "native_sglang_expert_id_sharding"
            if true_ep
            else "tensor_parallel_non_ep_control"
        ),
        "true_expert_parallel": true_ep,
        "expert_parallel_size": expert_parallel_size,
        "global_expert_count": OLMOE_EXPERT_COUNT,
        "expert_ids_owned_per_ep_rank": (
            OLMOE_EXPERT_COUNT // expert_parallel_size if true_ep else None
        ),
        "expert_id_partitioning": true_ep,
        "token_dispatch_semantics": (
            "replicate_tokens_mask_nonlocal_experts_then_sum_all_reduce"
            if true_ep
            else "standard_tensor_parallel_moe_no_expert_id_partition"
        ),
        "token_all_to_all": False,
        "moe_output_collective": (
            "expert_partition_partial_outputs_sum_all_reduce"
            if true_ep
            else "tensor_parallel_partial_outputs_sum_all_reduce"
        ),
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "ktransformers_enabled": False,
        "cpu_amx_expert_offload": False,
        "comparison_scope": "EP2 true weight sharding versus matched EP1 control",
    }


def _moe_config_receipt(admission: MoeConfigAdmission) -> JsonObject:
    return {
        "status": "admitted",
        "root": admission.root,
        "triton_version": admission.triton_version,
        "version_directory": admission.version_directory,
        "device_name": admission.device_name,
        "expert_parallel_size": admission.expert_parallel_size,
        "local_expert_count": admission.local_expert_count,
        "moe_intermediate_size": admission.moe_intermediate_size,
        "files": [
            {
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "canonical_sha256": item.canonical_sha256,
                "entry_count": item.entry_count,
                "batch_sizes": list(item.batch_sizes),
            }
            for item in admission.files
        ],
        "file_set_sha256": admission.file_set_sha256,
    }


def _moe_config_snapshot_receipt(
    snapshot: MoeConfigSnapshotAdmission,
) -> JsonObject:
    return {
        "status": "snapshotted",
        "source": _moe_config_receipt(snapshot.source),
        "effective_snapshot": {
            **_moe_config_receipt(snapshot.effective),
            "directory_paths": list(snapshot.directory_paths),
            "directory_mode_octal": format(snapshot.directory_mode, "04o"),
            "file_mode_octal": format(snapshot.file_mode, "04o"),
            "exact_tree_verified": snapshot.exact_tree_verified,
            "byte_for_byte_copy_verified": snapshot.byte_for_byte_copy_verified,
        },
    }


def _configuration_receipt(
    config: OlmoeEpBenchmarkConfig,
    moe_config_snapshot: MoeConfigSnapshotAdmission | None = None,
) -> JsonObject:
    max_total_tokens, max_running_requests, cuda_graph_max_batch_size = (
        _server_capacity(config)
    )
    receipt: JsonObject = {
        "host_name": "dwagon",
        "scope": "single_host",
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 2,
        "expert_parallel_size": config.expert_parallel_size,
        "gpu_uuid_order": list(DWAGON_GPU_UUIDS),
        "local_interconnect_requirement": "NV4",
        "physical_cpu_cores": list(DWAGON_PHYSICAL_CPUS),
        "numa_memory_policy": {
            "launcher_parent_mode": "interleave",
            "nodes": list(DWAGON_NUMA_NODES),
            "rank_local_mode": "numactl_cpunodebind_and_membind_v2",
            "rank_process_cpu_bindings": [
                "0-55,112-167",
                "56-111,168-223",
            ],
            "openmp_threads_per_rank": 56,
            "openmp_places": "cores",
        },
        "static_memory_fraction": config.static_memory_fraction,
        "token_pool": {
            "context_length": OLMOE_CONTEXT_LENGTH,
            "max_total_tokens": max_total_tokens,
            "max_running_requests": max_running_requests,
            **(
                {"cuda_graph_max_batch_size": cuda_graph_max_batch_size}
                if cuda_graph_max_batch_size is not None
                else {}
            ),
        },
        "tensor_parallel_collective": {
            "custom_all_reduce": False,
            "fallback": "NCCL",
            "cuda_visible_devices_format": "GPU_UUID",
        },
        "warmup_count": CANONICAL_WARMUP_COUNT,
        "sample_count": CANONICAL_SAMPLE_COUNT,
        **(
            {"benchmark_mode": "aggregate_concurrency"}
            if config.aggregate_concurrency
            else {}
        ),
        "expert_parallel_semantics": expert_parallel_semantics(
            config.expert_parallel_size
        ),
        **({"logit_parity_probe_enabled": True} if config.logit_parity_probe else {}),
    }
    if config.moe_config_root is not None:
        receipt["moe_kernel_config"] = (
            {
                "status": "not_snapshotted",
                "source_root": str(config.moe_config_root),
            }
            if moe_config_snapshot is None
            else _moe_config_snapshot_receipt(moe_config_snapshot)
        )
    return receipt


def _source_identity() -> tuple[JsonObject, ...]:
    files: list[JsonObject] = []
    repository_root = _HARNESS_PATH.parents[1]
    source_plan = SglangKtSourcePlan.exo_default()
    paths = (
        _HARNESS_PATH,
        _SERVING_CLIENT_PATH,
        _STAGE_CONTRACT_PRODUCER_PATH,
        _CPU_POLICY_HELPER_PATH,
        _HARNESS_PATH.parent / "install_sglang_kt_runtime.py",
        _HARNESS_PATH.parent / "validate_sglang_kt_runtime.py",
        _HARNESS_PATH.parent / "prepare_sglang_kt_source.py",
        source_plan.sglang_patch.path,
        source_plan.ktransformers_patch.path,
        *(patch.path for patch in source_plan.sglang_followup_patches),
    )
    for path in paths:
        try:
            status = path.stat()
        except OSError as error:
            raise OlmoeEpBenchmarkError(
                f"benchmark source is unavailable: {error}"
            ) from error
        if not path.is_file() or status.st_size > _IDENTITY_FILE_MAXIMUM_BYTES:
            raise OlmoeEpBenchmarkError("benchmark source identity is invalid")
        files.append(
            {
                "path": str(path.relative_to(repository_root)),
                "size_bytes": status.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(files)


def _managed_launch_evidence(
    *,
    running: RunningServerProcess,
    runtime: RuntimeAdmission,
    listener: JsonObject,
    rank_local_numa: JsonObject,
) -> ManagedLaunchEvidence:
    raw_inodes = listener.get("socket_inodes")
    raw_owner_pids = listener.get("owner_pids")
    if (
        not isinstance(raw_inodes, list)
        or not raw_inodes
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_inodes
        )
        or not isinstance(raw_owner_pids, list)
        or not raw_owner_pids
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_owner_pids
        )
    ):
        raise OlmoeEpBenchmarkError("listener ownership evidence is malformed")
    runtime_payload = cast(JsonValue, _json_compatible(asdict(runtime)))
    return ManagedLaunchEvidence(
        schema_version=1,
        status="owned_runtime_listener_verified",
        harness_sha256=_sha256_file(_HARNESS_PATH),
        runtime_admission_sha256=_canonical_sha256(runtime_payload),
        pid=running.owned.pid,
        process_group_id=running.owned.process_group_id,
        start_time_ticks=running.owned.start_time_ticks,
        owner_token_sha256=hashlib.sha256(
            running.owned.owner_token.encode()
        ).hexdigest(),
        ownership_namespace=running.owned.ownership_namespace,
        command=running.owned.command,
        launch_environment=running.owned.launch_environment,
        listener_socket_inodes=tuple(cast(list[int], raw_inodes)),
        listener_owner_pids=tuple(cast(list[int], raw_owner_pids)),
        rank_local_numa_observation_sha256=_canonical_sha256(rank_local_numa),
        verified_at_utc=_utc_now(),
    )


def _run_benchmark_under_cpu_policy(config: OlmoeEpBenchmarkConfig) -> JsonObject:
    """Admit, launch, sanity-check, measure, clean up, and publish one run."""

    port_preflight = verify_port_vacant(config.host, config.port)
    runtime = verify_runtime_install(config)
    model = verify_stage_contract(config)
    moe_config_source = verify_moe_config(config)
    source_identity = _source_identity()
    topology = {
        "gpu_nv4": verify_dwagon_nv4_topology(config),
        "cpu_numa": verify_dwagon_cpu_topology(),
        "port_preflight": port_preflight,
    }
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    moe_config_snapshot = create_moe_config_snapshot(config, moe_config_source)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    owner_token = uuid.uuid4().hex
    namespace = f"exo-olmoe-ep-{config.run_id}-{uuid.uuid4().hex}"
    running: RunningServerProcess | None = None
    readiness: JsonObject | None = None
    server_log_contract: JsonObject | None = None
    listener_ownership: list[JsonValue] = []
    rank_local_numa: list[JsonValue] = []
    server_info: JsonObject | None = None
    sanity: JsonObject | None = None
    workloads: list[JsonValue] = []
    cleanup: JsonObject = {
        "cleanup_complete": True,
        "group_empty": True,
        "status": "not_started",
    }
    failure: BaseException | None = None
    signal_state = _ManagedSignalState()
    previous_handlers: dict[signal.Signals, object] = {}
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, signal_state.handle)
        try:
            running = start_server(config, owner_token, namespace, moe_config_snapshot)
            signal_state.checkpoint()
            base_url = f"http://{config.host}:{config.port}"
            with OlmoeNativeServingClient(
                base_url,
                timeout_seconds=config.request_timeout_seconds,
            ) as client:
                readiness = _wait_for_readiness(
                    running, client, config.readiness_timeout_seconds
                )
                max_total_tokens, max_running_requests, cuda_graph_max_batch_size = (
                    _server_capacity(config)
                )
                server_log_contract = verify_server_log_contract(
                    Path(running.owned.log_path),
                    config.expert_parallel_size,
                    moe_config_snapshot,
                    max_total_tokens=max_total_tokens,
                    max_running_requests=max_running_requests,
                    cuda_graph_max_batch_size=cuda_graph_max_batch_size,
                )
                listener_ownership.append(verify_listener_owned(running, config.port))
                signal_state.checkpoint()
                observation = client.server_info()
                assert_server_alive(running, "server_info")
                server_info = _verify_server_info(observation.response, config)
                listener_ownership.append(verify_listener_owned(running, config.port))
                rank_local_numa.append(verify_rank_local_numa(running))
                oracle = (
                    model.contract.ep1.capture
                    if config.expert_parallel_size == 1
                    else model.contract.ep2.capture
                )
                sanity = _run_exact_sanity(client, oracle)
                assert_server_alive(running, "deterministic sanity")
                signal_state.checkpoint()
                if config.aggregate_concurrency:
                    workloads.extend(
                        run_aggregate_concurrency_benchmark(
                            base_url,
                            config.request_timeout_seconds,
                            lambda phase: assert_server_alive(running, phase),
                            signal_state.checkpoint,
                        )
                    )
                    signal_state.checkpoint()
                else:
                    for raw_kind, input_tokens, output_tokens in CANONICAL_WORKLOADS:
                        kind = cast(Literal["prefill", "decode"], raw_kind)
                        workloads.append(
                            run_canonical_workload(
                                client,
                                kind,
                                input_tokens,
                                output_tokens,
                                lambda phase: assert_server_alive(running, phase),
                            )
                        )
                        signal_state.checkpoint()
                rank_local_numa.append(verify_rank_local_numa(running))
                if moe_config_snapshot is not None:
                    server_log_contract = verify_server_log_contract(
                        Path(running.owned.log_path),
                        config.expert_parallel_size,
                        moe_config_snapshot,
                        max_total_tokens=max_total_tokens,
                        max_running_requests=max_running_requests,
                        cuda_graph_max_batch_size=cuda_graph_max_batch_size,
                    )
        except BaseException as error:
            failure = error
        finally:
            if running is not None and failure is None:
                try:
                    assert_server_alive(running, "pre-cleanup admission")
                except BaseException as error:
                    failure = error
            signal_state.begin_cleanup()
            if running is not None:
                cleanup = stop_server(running, config.cleanup_timeout_seconds)
                if cleanup.get("cleanup_complete") is True:
                    journal = config.result_directory / _OWNERSHIP_JOURNAL_FILENAME
                    journal.unlink()
                    _fsync_directory(config.result_directory)
                elif failure is None:
                    failure = OlmoeEpBenchmarkError("native SGLang cleanup incomplete")
    finally:
        for managed_signal, previous in previous_handlers.items():
            signal.signal(managed_signal, cast(signal.Handlers, previous))

    admission_reverification: JsonObject
    try:
        final_runtime = verify_runtime_install(config)
        final_model = verify_stage_contract(config)
        final_moe_config_source = verify_moe_config(config)
        final_moe_config_snapshot = (
            None
            if moe_config_snapshot is None
            else verify_moe_config_snapshot(moe_config_snapshot)
        )
        final_source_identity = _source_identity()
        if (
            final_runtime != runtime
            or final_model != model
            or final_moe_config_source != moe_config_source
            or final_moe_config_snapshot != moe_config_snapshot
            or final_source_identity != source_identity
        ):
            raise OlmoeEpBenchmarkError("admitted artifacts changed during benchmark")
        admission_reverification = {
            "status": "matched",
            "completed_at_utc": _utc_now(),
        }
    except BaseException as error:
        admission_reverification = {
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "failure": f"{type(error).__name__}: {error}",
        }
        if failure is None:
            failure = error

    log_path = config.result_directory / _SERVER_LOG_FILENAME
    logs: list[JsonValue] = []
    if log_path.is_file():
        size = log_path.stat().st_size
        if size > _LOG_MAXIMUM_BYTES and failure is None:
            failure = OlmoeEpBenchmarkError("native SGLang log exceeds size bound")
        logs.append(
            {
                "path": str(log_path),
                "size_bytes": size,
                "sha256": _sha256_file(log_path),
            }
        )
    completed_at = _utc_now()
    passed = (
        failure is None
        and sanity is not None
        and len(workloads) == len(CANONICAL_WORKLOADS)
        and cleanup.get("cleanup_complete") is True
    )
    payload = cast(
        JsonObject,
        {
            "schema_version": 1,
            "status": "policy_restore_pending" if passed else "failed",
            "run_id": config.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "configuration": _configuration_receipt(config, moe_config_snapshot),
            "source_identity": list(source_identity),
            "admission": {
                "runtime": cast(JsonObject, asdict(runtime)),
                "model_stage": {
                    "path": model.path,
                    "receipt_sha256": model.receipt_sha256,
                    "contract": model.contract.model_dump(mode="json"),
                },
                **(
                    {
                        "moe_kernel_config": _moe_config_snapshot_receipt(
                            moe_config_snapshot
                        )
                    }
                    if moe_config_snapshot is not None
                    else {}
                ),
                "topology": topology,
                "post_run_reverification": admission_reverification,
            },
            "launch": {
                "command": list(build_server_command(config)),
                "environment": {
                    key: value
                    for key, value in build_server_environment(
                        config,
                        "redacted",
                        namespace,
                        {},
                        moe_config_snapshot,
                    ).items()
                    if key not in {_OWNER_TOKEN_ENVIRONMENT}
                },
                "process": (
                    None
                    if running is None
                    else {
                        **asdict(running.owned),
                        "owner_token": hashlib.sha256(owner_token.encode()).hexdigest(),
                    }
                ),
                **(
                    {
                        "moe_kernel_config": _moe_config_snapshot_receipt(
                            moe_config_snapshot
                        )
                    }
                    if moe_config_snapshot is not None
                    else {}
                ),
            },
            "readiness": readiness,
            "server_log_contract": server_log_contract,
            "listener_ownership": listener_ownership,
            "rank_local_numa": rank_local_numa,
            "server_info": server_info,
            "sanity": sanity,
            "workloads": workloads,
            "cleanup": cleanup,
            "logs": logs,
            "failure": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        },
    )
    _write_private_json(config.result_directory / _RESULT_FILENAME, payload)
    if failure is not None:
        raise OlmoeEpBenchmarkError(str(failure)) from failure
    return payload


def _run_stage_capture_under_cpu_policy(
    config: OlmoeEpBenchmarkConfig,
) -> JsonObject:
    """Create one managed EP1 or EP2 capture without running timed workloads."""

    if config.stage_contract is not None or config.stage_capture_output is None:
        raise OlmoeEpBenchmarkError("stage capture mode is not configured exactly")
    port_preflight = verify_port_vacant(config.host, config.port)
    runtime = verify_runtime_install(config)
    moe_config_source = verify_moe_config(config)
    source_identity = _source_identity()
    topology = {
        "gpu_nv4": verify_dwagon_nv4_topology(config),
        "cpu_numa": verify_dwagon_cpu_topology(),
        "port_preflight": port_preflight,
    }
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    moe_config_snapshot = create_moe_config_snapshot(config, moe_config_source)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    owner_token = uuid.uuid4().hex
    namespace = f"exo-olmoe-ep-{config.run_id}-{uuid.uuid4().hex}"
    running: RunningServerProcess | None = None
    readiness: JsonObject | None = None
    server_log_contract: JsonObject | None = None
    listener_ownership: list[JsonValue] = []
    rank_local_numa: list[JsonValue] = []
    server_info: JsonObject | None = None
    capture: SanityCapture | None = None
    logit_parity_probe: JsonObject | None = None
    capture_receipt_sha256: str | None = None
    cleanup: JsonObject = {
        "cleanup_complete": True,
        "group_empty": True,
        "status": "not_started",
    }
    failure: BaseException | None = None
    signal_state = _ManagedSignalState()
    previous_handlers: dict[signal.Signals, object] = {}
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, signal_state.handle)
        try:
            running = start_server(config, owner_token, namespace, moe_config_snapshot)
            signal_state.checkpoint()
            with OlmoeNativeServingClient(
                f"http://{config.host}:{config.port}",
                timeout_seconds=config.request_timeout_seconds,
            ) as client:
                readiness = _wait_for_readiness(
                    running, client, config.readiness_timeout_seconds
                )
                server_log_contract = verify_server_log_contract(
                    Path(running.owned.log_path),
                    config.expert_parallel_size,
                    moe_config_snapshot,
                )
                first_listener = verify_listener_owned(running, config.port)
                listener_ownership.append(first_listener)
                observation = client.server_info()
                assert_server_alive(running, "stage capture server_info")
                server_info = _verify_server_info(observation.response, config)
                first_numa = verify_rank_local_numa(running)
                rank_local_numa.append(first_numa)
                managed_launch = _managed_launch_evidence(
                    running=running,
                    runtime=runtime,
                    listener=first_listener,
                    rank_local_numa=first_numa,
                )
                try:
                    capture = collect_sanity_capture(
                        snapshot_path=Path(config.model_path),
                        expert_parallel_size=config.expert_parallel_size,
                        base_url=f"http://{config.host}:{config.port}",
                        runtime_install_receipt_sha256=(
                            config.runtime_install_receipt_sha256
                        ),
                        static_memory_fraction=config.static_memory_fraction,
                        managed_launch=managed_launch,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    raise OlmoeEpBenchmarkError(
                        f"managed sanity capture failed: {error}"
                    ) from error
                assert_server_alive(running, "managed sanity capture")
                if config.logit_parity_probe:
                    logit_parity_probe = _run_logit_parity_probe(client)
                    assert_server_alive(running, "logit parity probe")
                listener_ownership.append(verify_listener_owned(running, config.port))
                rank_local_numa.append(verify_rank_local_numa(running))
                if moe_config_snapshot is not None:
                    server_log_contract = verify_server_log_contract(
                        Path(running.owned.log_path),
                        config.expert_parallel_size,
                        moe_config_snapshot,
                    )
                signal_state.checkpoint()
        except BaseException as error:
            failure = error
        finally:
            if running is not None and failure is None:
                try:
                    assert_server_alive(running, "pre-cleanup stage capture")
                except BaseException as error:
                    failure = error
            signal_state.begin_cleanup()
            if running is not None:
                cleanup = stop_server(running, config.cleanup_timeout_seconds)
                if cleanup.get("cleanup_complete") is True:
                    journal = config.result_directory / _OWNERSHIP_JOURNAL_FILENAME
                    journal.unlink()
                    _fsync_directory(config.result_directory)
                elif failure is None:
                    failure = OlmoeEpBenchmarkError("native SGLang cleanup incomplete")
    finally:
        for managed_signal, previous in previous_handlers.items():
            signal.signal(managed_signal, cast(signal.Handlers, previous))

    admission_reverification: JsonObject
    try:
        final_runtime = verify_runtime_install(config)
        final_moe_config_source = verify_moe_config(config)
        final_moe_config_snapshot = (
            None
            if moe_config_snapshot is None
            else verify_moe_config_snapshot(moe_config_snapshot)
        )
        final_snapshot = verify_pinned_snapshot(Path(config.model_path))
        final_source_identity = _source_identity()
        if (
            final_runtime != runtime
            or final_moe_config_source != moe_config_source
            or final_moe_config_snapshot != moe_config_snapshot
            or final_source_identity != source_identity
            or capture is None
            or final_snapshot.canonical_sha256 != capture.snapshot_canonical_sha256
        ):
            raise OlmoeEpBenchmarkError("admitted artifacts changed during capture")
        admission_reverification = {
            "status": "matched",
            "completed_at_utc": _utc_now(),
        }
    except BaseException as error:
        admission_reverification = {
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "failure": f"{type(error).__name__}: {error}",
        }
        if failure is None:
            failure = error

    log_path = config.result_directory / _SERVER_LOG_FILENAME
    logs: list[JsonValue] = []
    if log_path.is_file():
        size = log_path.stat().st_size
        if size > _LOG_MAXIMUM_BYTES and failure is None:
            failure = OlmoeEpBenchmarkError("native SGLang log exceeds size bound")
        logs.append(
            {
                "path": str(log_path),
                "size_bytes": size,
                "sha256": _sha256_file(log_path),
            }
        )
    passed = (
        failure is None
        and capture is not None
        and (not config.logit_parity_probe or logit_parity_probe is not None)
        and cleanup.get("cleanup_complete") is True
    )
    payload = cast(
        JsonObject,
        {
            "schema_version": 1,
            "status": "policy_restore_pending" if passed else "failed",
            "mode": (
                "managed_stage_capture_with_logit_parity_probe"
                if config.logit_parity_probe
                else "managed_stage_capture"
            ),
            "run_id": config.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "configuration": _configuration_receipt(config, moe_config_snapshot),
            "source_identity": list(source_identity),
            "admission": {
                "runtime": cast(JsonObject, asdict(runtime)),
                **(
                    {
                        "moe_kernel_config": _moe_config_snapshot_receipt(
                            moe_config_snapshot
                        )
                    }
                    if moe_config_snapshot is not None
                    else {}
                ),
                "topology": topology,
                "post_run_reverification": admission_reverification,
            },
            "launch": (
                None
                if running is None
                else {
                    **asdict(running.owned),
                    "owner_token": hashlib.sha256(owner_token.encode()).hexdigest(),
                    **(
                        {
                            "moe_kernel_config": _moe_config_snapshot_receipt(
                                moe_config_snapshot
                            )
                        }
                        if moe_config_snapshot is not None
                        else {}
                    ),
                }
            ),
            "readiness": readiness,
            "server_log_contract": server_log_contract,
            "listener_ownership": listener_ownership,
            "rank_local_numa": rank_local_numa,
            "server_info": server_info,
            "capture": None if capture is None else capture.model_dump(mode="json"),
            "capture_output": str(config.stage_capture_output),
            "capture_receipt_sha256": capture_receipt_sha256,
            **(
                {"logit_parity_probe": logit_parity_probe}
                if config.logit_parity_probe
                else {}
            ),
            "cleanup": cleanup,
            "logs": logs,
            "failure": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        },
    )
    _write_private_json(config.result_directory / _RESULT_FILENAME, payload)
    if failure is not None:
        raise OlmoeEpBenchmarkError(str(failure)) from failure
    return payload


type CpuPerformancePolicyFactory = Callable[[], CpuPerformancePolicySession]
type _ManagedRun = Callable[[OlmoeEpBenchmarkConfig], JsonObject]


def _cpu_performance_policy_receipt(
    session: CpuPerformancePolicySession,
) -> JsonObject:
    evidence = _json_object(
        _json_compatible(session.evidence), "CPU performance-policy evidence"
    )
    serialization = _json_object(
        evidence.get("serialization"), "CPU performance-policy serialization"
    )
    snapshots = _json_object(
        evidence.get("snapshots"), "CPU performance-policy snapshots"
    )
    failures = evidence.get("failures")
    policies = evidence.get("policies")
    if (
        evidence.get("lifecycle") != "restored"
        or evidence.get("application_verified") is not True
        or evidence.get("restoration_verified") is not True
        or not isinstance(failures, list)
        or failures
        or not isinstance(policies, list)
        or not policies
        or serialization.get("journal_published") is not False
        or serialization.get("journal_removed") is not True
        or serialization.get("lock_released") is not True
        or serialization.get("lock_acquired") is not False
        or not {"before", "active", "performance_after", "restored"}.issubset(snapshots)
    ):
        raise OlmoeEpBenchmarkError(
            "CPU performance-policy transaction did not verify application and restore"
        )
    normalized_evidence = cast(JsonObject, evidence)
    return {
        "status": "performance_policy_applied_and_restored",
        "transaction_scope": "one_outer_harness_transaction",
        "nested_or_rank_transactions": False,
        "helper_path": str(_CPU_POLICY_HELPER_PATH),
        "helper_sha256": _sha256_file(_CPU_POLICY_HELPER_PATH),
        "evidence_sha256": _canonical_sha256(normalized_evidence),
        "evidence": normalized_evidence,
    }


def _load_provisional_result(config: OlmoeEpBenchmarkConfig) -> JsonObject | None:
    path = config.result_directory / _RESULT_FILENAME
    if not path.is_file():
        return None
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=_RECEIPT_MAXIMUM_BYTES)
        parsed = parse_sglang_kt_strict_json(bound.contents)
    except SglangKtReceiptFileError as error:
        raise OlmoeEpBenchmarkError(
            f"cannot read provisional benchmark receipt: {error}"
        ) from error
    return cast(JsonObject, _json_object(parsed, "provisional benchmark receipt"))


def _minimal_failed_result(
    config: OlmoeEpBenchmarkConfig, started_at_utc: str
) -> JsonObject:
    return {
        "schema_version": 1,
        "status": "failed",
        "run_id": config.run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "configuration": _configuration_receipt(config),
    }


def _run_with_cpu_performance_policy(
    config: OlmoeEpBenchmarkConfig,
    managed_run: _ManagedRun,
    *,
    stage_capture: bool,
    policy_factory: CpuPerformancePolicyFactory,
) -> JsonObject:
    started_at = _utc_now()
    session: CpuPerformancePolicySession | None = None
    payload: JsonObject | None = None
    workload_failure: BaseException | None = None
    policy_failure: BaseException | None = None
    try:
        session = policy_factory()
        with session:
            try:
                payload = managed_run(config)
            except BaseException as error:
                workload_failure = error
    except BaseException as error:
        policy_failure = error

    try:
        if payload is None:
            payload = _load_provisional_result(config)
    except BaseException as error:
        if workload_failure is None:
            workload_failure = error
    if payload is None:
        payload = _minimal_failed_result(config, started_at)

    cpu_receipt: JsonObject | None = None
    if session is not None:
        try:
            cpu_receipt = _cpu_performance_policy_receipt(session)
        except BaseException as error:
            if policy_failure is None:
                policy_failure = error
    if cpu_receipt is None:
        cpu_receipt = {
            "status": "failed",
            "transaction_scope": "one_outer_harness_transaction",
            "nested_or_rank_transactions": False,
            "helper_path": str(_CPU_POLICY_HELPER_PATH),
            "helper_sha256": _sha256_file(_CPU_POLICY_HELPER_PATH),
            "failure": (
                "CPU performance-policy evidence is unavailable"
                if policy_failure is None
                else f"{type(policy_failure).__name__}: {policy_failure}"
            ),
            "evidence": (
                None
                if session is None
                else cast(JsonObject, _json_compatible(session.evidence))
            ),
        }
    payload["cpu_performance_policy"] = cpu_receipt

    if workload_failure is None and payload.get("status") != "policy_restore_pending":
        workload_failure = OlmoeEpBenchmarkError(
            "managed run did not produce a policy-restore-pending receipt"
        )

    if (
        stage_capture
        and workload_failure is None
        and policy_failure is None
        and payload.get("status") == "policy_restore_pending"
    ):
        try:
            raw_capture = payload.get("capture")
            capture = SanityCapture.model_validate_json(_canonical_json(raw_capture))
            if config.stage_capture_output is None:
                raise OlmoeEpBenchmarkError("stage capture output is unavailable")
            payload["capture_receipt_sha256"] = publish_sanity_capture(
                capture=capture, output=config.stage_capture_output
            )
        except BaseException as error:
            workload_failure = error

    failures = tuple(
        error for error in (workload_failure, policy_failure) if error is not None
    )
    if failures:
        payload["status"] = "failed"
        payload["failure"] = "; ".join(
            f"{type(error).__name__}: {error}" for error in failures
        )
    else:
        payload["status"] = "passed"
        payload["failure"] = None
    payload["completed_at_utc"] = _utc_now()

    result_path = config.result_directory / _RESULT_FILENAME
    if not config.result_directory.exists():
        config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_private_json(result_path, payload)
    if failures:
        raise OlmoeEpBenchmarkError(cast(str, payload["failure"])) from failures[0]
    return payload


def run_benchmark(
    config: OlmoeEpBenchmarkConfig,
    policy_factory: CpuPerformancePolicyFactory = cpu_performance_policy,
) -> JsonObject:
    if config.logit_parity_probe:
        raise OlmoeEpBenchmarkError("logit parity probe requires managed stage capture")
    return _run_with_cpu_performance_policy(
        config,
        _run_benchmark_under_cpu_policy,
        stage_capture=False,
        policy_factory=policy_factory,
    )


def run_stage_capture(
    config: OlmoeEpBenchmarkConfig,
    policy_factory: CpuPerformancePolicyFactory = cpu_performance_policy,
) -> JsonObject:
    return _run_with_cpu_performance_policy(
        config,
        _run_stage_capture_under_cpu_policy,
        stage_capture=True,
        policy_factory=policy_factory,
    )


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _memory_fraction(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.5 <= value <= 0.95:
        raise argparse.ArgumentTypeError(
            "static memory fraction must be in [0.5, 0.95]"
        )
    return value


def _sha256_argument(raw: str) -> str:
    if _SHA256_PATTERN.fullmatch(raw) is None:
        raise argparse.ArgumentTypeError("value must be lowercase SHA-256")
    return raw


def _ep_size(raw: str) -> ExpertParallelSize:
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    raise argparse.ArgumentTypeError("EP size must be 1 or 2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--runtime-install-receipt", required=True, type=Path)
    parser.add_argument(
        "--runtime-install-receipt-sha256", required=True, type=_sha256_argument
    )
    parser.add_argument("--model-path", default=OLMOE_MODEL_PATH)
    stage_mode = parser.add_mutually_exclusive_group(required=True)
    stage_mode.add_argument("--stage-contract", type=Path)
    stage_mode.add_argument("--stage-capture-output", type=Path)
    parser.add_argument("--ep-size", required=True, type=_ep_size)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--static-memory-fraction",
        type=_memory_fraction,
        default=DEFAULT_STATIC_MEMORY_FRACTION,
    )
    parser.add_argument(
        "--readiness-timeout-seconds", type=_positive_float, default=900.0
    )
    parser.add_argument(
        "--request-timeout-seconds", type=_positive_float, default=900.0
    )
    parser.add_argument("--cleanup-timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--numactl-executable", default="/usr/bin/numactl")
    parser.add_argument("--nvidia-smi-executable", default="/usr/bin/nvidia-smi")
    parser.add_argument(
        "--moe-config-root",
        type=Path,
        help="absolute root containing the exact admitted Triton config pair",
    )
    parser.add_argument(
        "--logit-parity-probe",
        action="store_true",
        help="capture one receipt-backed logit probe without timed workloads",
    )
    parser.add_argument(
        "--aggregate-concurrency",
        action="store_true",
        help=(
            "run synchronized C=1,2,4,8 aggregate-throughput groups with an "
            "isolated 9216-token/8-request server capacity"
        ),
    )
    return parser


def _absolute_lexical_path(raw: str, description: str) -> str:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or "\0" in raw:
        raise OlmoeEpBenchmarkError(f"{description} must be an absolute lexical path")
    return raw


def config_from_arguments(arguments: argparse.Namespace) -> OlmoeEpBenchmarkConfig:
    run_id = cast(str, arguments.run_id)
    if _SAFE_RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise OlmoeEpBenchmarkError("run ID is invalid")
    result_directory = cast(Path, arguments.result_directory)
    if not result_directory.is_absolute() or ".." in result_directory.parts:
        raise OlmoeEpBenchmarkError("result directory must be an absolute lexical path")
    port = cast(int, arguments.port)
    if isinstance(port, bool) or not 1 <= port <= 65_535:
        raise OlmoeEpBenchmarkError("port must be a valid TCP port")
    host = cast(str, arguments.host)
    if host not in {"127.0.0.1", "localhost"}:
        raise OlmoeEpBenchmarkError("local EP benchmark host must be loopback")
    stage_contract = cast(Path | None, arguments.stage_contract)
    stage_capture_output = cast(Path | None, arguments.stage_capture_output)
    moe_config_root = cast(Path | None, arguments.moe_config_root)
    logit_parity_probe = cast(bool, arguments.logit_parity_probe)
    aggregate_concurrency = cast(bool, arguments.aggregate_concurrency)
    if logit_parity_probe and stage_capture_output is None:
        raise OlmoeEpBenchmarkError("logit parity probe requires managed stage capture")
    if aggregate_concurrency and stage_capture_output is not None:
        raise OlmoeEpBenchmarkError(
            "aggregate concurrency is available only with a trusted stage contract"
        )
    if stage_contract is not None and (
        not stage_contract.is_absolute() or ".." in stage_contract.parts
    ):
        raise OlmoeEpBenchmarkError("stage contract must be an absolute lexical path")
    if stage_capture_output is not None and (
        not stage_capture_output.is_absolute() or ".." in stage_capture_output.parts
    ):
        raise OlmoeEpBenchmarkError(
            "stage capture output must be an absolute lexical path"
        )
    if moe_config_root is not None and (
        not moe_config_root.is_absolute()
        or ".." in moe_config_root.parts
        or "\0" in str(moe_config_root)
        or moe_config_root != Path(os.path.normpath(moe_config_root))
    ):
        raise OlmoeEpBenchmarkError(
            "MoE config root must be an absolute lexical directory"
        )
    if stage_capture_output is not None:
        try:
            resolved_result_directory = result_directory.resolve(strict=False)
            resolved_capture_output = stage_capture_output.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise OlmoeEpBenchmarkError(
                f"cannot normalize stage capture paths: {error}"
            ) from error
        if stage_capture_output.is_relative_to(
            result_directory
        ) or resolved_capture_output.is_relative_to(resolved_result_directory):
            raise OlmoeEpBenchmarkError(
                "stage capture output must be outside the per-run result directory"
            )
    return OlmoeEpBenchmarkConfig(
        run_id=run_id,
        result_directory=result_directory,
        runtime_python=_absolute_lexical_path(
            cast(str, arguments.runtime_python), "runtime Python"
        ),
        runtime_install_receipt=cast(Path, arguments.runtime_install_receipt),
        runtime_install_receipt_sha256=cast(
            str, arguments.runtime_install_receipt_sha256
        ),
        model_path=_absolute_lexical_path(
            cast(str, arguments.model_path), "model path"
        ),
        stage_contract=stage_contract,
        stage_capture_output=stage_capture_output,
        expert_parallel_size=cast(ExpertParallelSize, arguments.ep_size),
        host=host,
        port=port,
        static_memory_fraction=cast(float, arguments.static_memory_fraction),
        readiness_timeout_seconds=cast(float, arguments.readiness_timeout_seconds),
        request_timeout_seconds=cast(float, arguments.request_timeout_seconds),
        cleanup_timeout_seconds=cast(float, arguments.cleanup_timeout_seconds),
        numactl_executable=_absolute_lexical_path(
            cast(str, arguments.numactl_executable), "numactl executable"
        ),
        nvidia_smi_executable=_absolute_lexical_path(
            cast(str, arguments.nvidia_smi_executable), "nvidia-smi executable"
        ),
        logit_parity_probe=logit_parity_probe,
        moe_config_root=moe_config_root,
        aggregate_concurrency=aggregate_concurrency,
    )


def main() -> int:
    try:
        config = config_from_arguments(_parser().parse_args())
        payload = (
            run_stage_capture(config)
            if config.stage_capture_output is not None
            else run_benchmark(config)
        )
    except OlmoeEpBenchmarkError as error:
        print(f"OLMoE EP benchmark failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
