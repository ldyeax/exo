#!/usr/bin/env python3
"""Run the pinned dwagon-local GLM-4.7 Flash PP1/TP2 diagnostic.

One SGLang parent owns both ordered RTX 3090 devices. SGLang creates two local
tensor-parallel GPU workers while KTransformers receives every physical CPU,
both NUMA nodes, and two CPUInfer pools. The run requires semantic sanity before
measuring the canonical 1024/32 and 128/128 workloads.

This harness-local contract intentionally precedes a first-class Exo TP schema.
It cannot be substituted for the existing TP1 launch profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import IO, Final, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.shared.types.common import Host  # noqa: E402
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE,
    GLM_4_7_FLASH_CONTEXT_LENGTH,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_LAYER_COUNT,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    canonical_sglang_kt_json,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (  # noqa: E402
    GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION,
)
from scripts import run_sglang_kt_glm47_pp2_local_diagnostic as pp2  # noqa: E402
from scripts import run_sglang_kt_glm47_pp3_diagnostic as pipeline  # noqa: E402
from scripts import run_sglang_kt_glm47_validation as validation  # noqa: E402
from scripts.sglang_kt_glm47_serving_client import (  # noqa: E402
    Glm47NativeServingClient,
    ServerInfoObservation,
    prepare_glm47_serving_workload,
    run_glm47_serving_sanity,
    run_glm47_serving_workload,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type StageCpuCoreBinding = tuple[int, tuple[int, ...]]

TARGET_PROFILE: Final = "glm47_flash_bf16_sm86_pp1_tp2_local_diagnostic_v1"
DWAGON_GPU_UUIDS: Final = (
    pipeline.DWAGON_STAGE_ZERO_GPU,
    pipeline.DWAGON_STAGE_ONE_GPU,
)
DWAGON_PHYSICAL_CPUS: Final = tuple(range(112))
DWAGON_NUMA_NODES: Final = (0, 1)
DWAGON_CPU_INFER_THREADS: Final = 112
DWAGON_THREADPOOL_COUNT: Final = 2
DEFAULT_RESIDENT_GPU_EXPERTS: Final = 40
MAXIMUM_RESIDENT_GPU_EXPERTS: Final = 44
DEFAULT_DWAGON_IP: Final = "192.168.40.24"
DEFAULT_DISTRIBUTED_PORT: Final = 62600
DEFAULT_SERVICE_PORT: Final = 62610
DEFAULT_STATIC_MEMORY_FRACTION: Final = 0.9
_OWNERSHIP_JOURNAL_NAME: Final = "tp2-local-ownership-journal.json"
_RESULT_FILENAME: Final = "tp2-local-diagnostic-result.json"
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_RETAINED_ENVIRONMENT_NAMES: Final = (
    "CUDA_HOME",
    "CUDA_PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TZ",
)


class Tp2LocalDiagnosticError(RuntimeError):
    """Raised when a local TP2 run cannot produce complete evidence."""


class Tp2ParentLaunchError(Tp2LocalDiagnosticError):
    """Carries evidence for a launch that failed before Popen returned."""

    def __init__(
        self,
        message: str,
        *,
        journal_created: bool,
        journal_cleared: bool,
        launch_evidence: JsonObject,
        popen_attempted: bool,
        journal_error: str | None,
    ) -> None:
        super().__init__(message)
        self.journal_created = journal_created
        self.journal_cleared = journal_cleared
        self.launch_evidence = launch_evidence
        self.popen_attempted = popen_attempted
        self.journal_error = journal_error


class Tp2PartialParentStartError(Tp2LocalDiagnosticError):
    """Carries fail-closed evidence for a parent created but not registered."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_evidence: JsonObject,
        cleanup_verified: bool,
        journal_created: bool,
        launch_evidence: JsonObject,
        process_evidence: JsonObject,
    ) -> None:
        super().__init__(message)
        self.cleanup_evidence = cleanup_evidence
        self.cleanup_verified = cleanup_verified
        self.journal_created = journal_created
        self.launch_evidence = launch_evidence
        self.process_evidence = process_evidence


@dataclass(frozen=True, slots=True)
class Tp2LocalDiagnosticConfig:
    run_id: str
    result_directory: Path
    dwagon_runtime_python: str
    dwagon_runtime_install_receipt: Path
    dwagon_runtime_install_receipt_sha256: str
    dwagon_model_path: str
    local_source_directory: str
    dwagon_ip: str
    dwagon_socket_interface: str
    distributed_port: int
    service_port: int
    static_memory_fraction: float
    resident_gpu_experts: int
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float
    warmup_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class Tp2LocalProcessSpec:
    """Exact, inert launch description for one local SGLang TP2 parent."""

    executable: str
    model_path: str
    service_endpoint: Host
    distributed_coordinator: Host
    static_memory_fraction: float
    target_profile: str = TARGET_PROFILE
    pipeline_rank: int = 0
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 2
    node_count: int = 1
    node_rank: int = 0
    ordered_gpu_uuids: tuple[str, str] = DWAGON_GPU_UUIDS
    cpu_cores: tuple[int, ...] = DWAGON_PHYSICAL_CPUS
    memory_nodes: tuple[int, int] = DWAGON_NUMA_NODES
    cpu_infer_threads: int = DWAGON_CPU_INFER_THREADS
    threadpool_count: int = DWAGON_THREADPOOL_COUNT
    resident_gpu_experts: int = DEFAULT_RESIDENT_GPU_EXPERTS

    def __post_init__(self) -> None:
        exact_topology = (
            self.target_profile == TARGET_PROFILE
            and self.pipeline_rank == 0
            and self.pipeline_parallel_size == 1
            and self.tensor_parallel_size == 2
            and self.node_count == 1
            and self.node_rank == 0
            and self.ordered_gpu_uuids == DWAGON_GPU_UUIDS
            and self.cpu_cores == DWAGON_PHYSICAL_CPUS
            and self.memory_nodes == DWAGON_NUMA_NODES
            and self.cpu_infer_threads == DWAGON_CPU_INFER_THREADS
            and self.threadpool_count == DWAGON_THREADPOOL_COUNT
        )
        if not exact_topology:
            raise ValueError("TP2 process spec differs from the pinned dwagon topology")
        if self.model_path != pipeline.DEFAULT_DWAGON_MODEL_PATH:
            raise ValueError("TP2 process spec requires the pinned GLM-4.7 model path")
        if not Path(self.executable).is_absolute():
            raise ValueError("TP2 runtime executable must be an absolute path")
        if not 0.8 <= self.static_memory_fraction <= 0.95:
            raise ValueError("TP2 static memory fraction must be between 0.8 and 0.95")
        if not 1 <= self.resident_gpu_experts <= MAXIMUM_RESIDENT_GPU_EXPERTS:
            raise ValueError(
                "TP2 resident GPU experts must be between 1 and "
                f"{MAXIMUM_RESIDENT_GPU_EXPERTS}"
            )
        if self.service_endpoint == self.distributed_coordinator:
            raise ValueError("TP2 service and distributed endpoints must be distinct")

    @property
    def command(self) -> tuple[str, ...]:
        return (
            self.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--kt-weight-path",
            self.model_path,
            "--kt-cpuinfer",
            str(self.cpu_infer_threads),
            "--kt-threadpool-count",
            str(self.threadpool_count),
            "--kt-numa-nodes",
            *(str(memory_node) for memory_node in self.memory_nodes),
            "--kt-num-gpu-experts",
            str(self.resident_gpu_experts),
            "--kt-method",
            "BF16",
            "--kt-max-deferred-experts-per-token",
            "0",
            "--kt-expert-placement-strategy",
            "uniform",
            "--pp-size",
            str(self.pipeline_parallel_size),
            "--tp-size",
            str(self.tensor_parallel_size),
            "--nnodes",
            str(self.node_count),
            "--node-rank",
            str(self.node_rank),
            "--dist-init-addr",
            str(self.distributed_coordinator),
            "--host",
            self.service_endpoint.ip,
            "--port",
            str(self.service_endpoint.port),
            "--context-length",
            str(GLM_4_7_FLASH_CONTEXT_LENGTH),
            "--max-total-tokens",
            str(GLM_4_7_FLASH_MAX_TOTAL_TOKENS),
            "--mem-fraction-static",
            str(self.static_memory_fraction),
            "--max-running-requests",
            "1",
            "--chunked-prefill-size",
            str(GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE),
            "--disable-cuda-graph",
            "--attention-backend",
            "flashinfer",
            "--kv-cache-dtype",
            "bfloat16",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--served-model-name",
            "GLM-4.7-Flash",
            "--trust-remote-code",
            "--disable-radix-cache",
        )

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        return (
            ("CUDA_VISIBLE_DEVICES", ",".join(self.ordered_gpu_uuids)),
            ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        )

    def receipt(self) -> JsonObject:
        return {
            "schema_version": 1,
            "target_profile": self.target_profile,
            "model": {
                "model_id": GLM_4_7_FLASH_BF16_MODEL_ID,
                "model_revision": GLM_4_7_FLASH_BF16_MODEL_REVISION,
                "model_contract_sha256": GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
                "model_path": self.model_path,
                "ktransformers_weight_path": self.model_path,
                "layer_range": [0, GLM_4_7_FLASH_LAYER_COUNT],
            },
            "runtime": {
                "executable": self.executable,
                "sglang_revision": GLM_4_7_FLASH_SGLANG_REVISION,
                "ktransformers_revision": GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
            },
            "parallelism": {
                "parent_process_count": 1,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "tensor_parallel_size": self.tensor_parallel_size,
                "node_count": self.node_count,
                "node_rank": self.node_rank,
            },
            "gpu_workers": [
                {
                    "tensor_parallel_rank": rank,
                    "gpu_uuid": gpu_uuid,
                    "matching_numa_node": rank,
                    "numa_local_physical_cpu_ids": list(
                        range(rank * 56, (rank + 1) * 56)
                    ),
                }
                for rank, gpu_uuid in enumerate(self.ordered_gpu_uuids)
            ],
            "cpu": {
                "physical_cpu_ids": list(self.cpu_cores),
                "numa_nodes": list(self.memory_nodes),
                "cpu_infer_threads": self.cpu_infer_threads,
                "threadpool_count": self.threadpool_count,
            },
            "experts": {
                "resident_gpu_experts": self.resident_gpu_experts,
                "count_semantics": "per_layer_global_logical_expert_count",
                "placement_strategy": "uniform",
                "max_deferred_experts_per_token": 0,
                "ktransformers_method": "BF16",
            },
            "hybrid_execution_launch_intent": {
                "evidence_scope": "launch_intent_not_observed_execution",
                "cpuinfer_owner_tensor_parallel_rank": 0,
                "cpuinfer_result_scope": "full_cpu_expert_result",
                "cpuinfer_result_contribution_count": 1,
                "cpuinfer_merge_order": (
                    "contributed_exactly_once_before_tensor_parallel_all_reduce"
                ),
                "resident_expert_gpu_weight_partition": (
                    "tensor_parallel_half_shard_per_gpu_for_each_resident_expert"
                ),
            },
            "service_endpoint": self.service_endpoint.model_dump(mode="json"),
            "distributed_coordinator": self.distributed_coordinator.model_dump(
                mode="json"
            ),
            "static_memory_fraction": self.static_memory_fraction,
            "argv": list(self.command),
            "environment": {name: value for name, value in self.environment},
        }


@dataclass(slots=True)
class RunningParent:
    owned: pipeline.OwnedStageProcess
    process: subprocess.Popen[str]
    log_file: IO[str]
    launch_evidence: JsonObject
    pump_thread: None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_tp2_local_process_spec(
    config: Tp2LocalDiagnosticConfig,
) -> Tp2LocalProcessSpec:
    """Build the only topology admitted by this proof harness."""

    return Tp2LocalProcessSpec(
        executable=config.dwagon_runtime_python,
        model_path=config.dwagon_model_path,
        service_endpoint=Host(ip=config.dwagon_ip, port=config.service_port),
        distributed_coordinator=Host(
            ip=config.dwagon_ip,
            port=config.distributed_port,
        ),
        static_memory_fraction=config.static_memory_fraction,
        resident_gpu_experts=config.resident_gpu_experts,
    )


def validate_tp2_server_info(
    observation: ServerInfoObservation,
    spec: Tp2LocalProcessSpec,
) -> JsonObject:
    """Require the raw rank-zero response to describe the exact TP2 launch."""

    expected: JsonObject = {
        "version": GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION,
        "model_path": spec.model_path,
        "kt_weight_path": spec.model_path,
        "tp_size": 2,
        "pp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
        "dist_init_addr": str(spec.distributed_coordinator),
        "kt_method": "BF16",
        "kt_cpuinfer": 112,
        "kt_threadpool_count": 2,
        "kt_numa_nodes": [0, 1],
        "kt_num_gpu_experts": spec.resident_gpu_experts,
        "kt_max_deferred_experts_per_token": 0,
        "kt_expert_placement_strategy": "uniform",
        "mem_fraction_static": spec.static_memory_fraction,
        "attention_backend": "flashinfer",
        "kv_cache_dtype": "bfloat16",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_shared_experts_fusion": True,
        "chunked_prefill_size": GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE,
        "context_length": GLM_4_7_FLASH_CONTEXT_LENGTH,
        "max_total_tokens": GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
        "max_running_requests": 1,
        "served_model_name": "GLM-4.7-Flash",
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "trust_remote_code": True,
    }
    canonical_response_sha256 = hashlib.sha256(
        canonical_sglang_kt_json(observation.response)
    ).hexdigest()
    if observation.call.status_code != 200:
        raise Tp2LocalDiagnosticError("server_info did not return HTTP 200")
    if observation.canonical_response_sha256 != canonical_response_sha256:
        raise Tp2LocalDiagnosticError("server_info raw response hash is not bound")
    for field_name, expected_value in expected.items():
        observed_value = observation.response.get(field_name)
        if (
            type(observed_value) is not type(expected_value)
            or observed_value != expected_value
        ):
            raise Tp2LocalDiagnosticError(
                f"server_info does not match the pinned local TP2 launch: {field_name}"
            )
    return {
        "validation": "required_raw_rank_zero_fields_exact",
        "validated_identity": {
            **expected,
            "canonical_response_sha256": canonical_response_sha256,
        },
        "raw_observation": observation.model_dump(mode="json"),
    }


def build_parent_environment(
    spec: Tp2LocalProcessSpec,
    owner_token: str,
    socket_interface: str,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a clean local-NVLink environment for the TP2 SGLang parent."""

    source = os.environ if parent_environment is None else parent_environment
    environment = {
        name: source[name] for name in _RETAINED_ENVIRONMENT_NAMES if source.get(name)
    }
    environment.setdefault("HOME", "/root")
    environment.setdefault("LANG", "C.UTF-8")
    environment.setdefault("LC_ALL", "C.UTF-8")
    environment.setdefault(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/cuda/bin",
    )
    environment.update(spec.environment)
    environment.update(
        {
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "GLOO_SOCKET_IFNAME": socket_interface,
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET,ENV",
            "NCCL_SOCKET_IFNAME": socket_interface,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    validation.require_no_profiler_state(environment, spec.command, "")
    return environment


def build_parent_command(spec: Tp2LocalProcessSpec) -> tuple[str, ...]:
    cpu_list = ",".join(str(cpu_core) for cpu_core in spec.cpu_cores)
    memory_nodes = ",".join(str(memory_node) for memory_node in spec.memory_nodes)
    return (
        "/usr/bin/numactl",
        "--physcpubind",
        cpu_list,
        "--membind",
        memory_nodes,
        *spec.command,
    )


def _selected_environment(environment: Mapping[str, str]) -> JsonObject:
    selected: JsonObject = {
        name: value
        for name, value in sorted(environment.items())
        if name.startswith(("CUDA_", "NCCL_", "SGLANG_", "PYTORCH_"))
        or name
        in {
            "EXO_BENCHMARK_OWNER_TOKEN",
            "GLOO_SOCKET_IFNAME",
            "PYTHONHASHSEED",
            "TOKENIZERS_PARALLELISM",
        }
    }
    selected["EXO_BENCHMARK_OWNER_TOKEN"] = "<redacted>"
    return selected


def _launch_evidence(
    command: tuple[str, ...],
    environment: Mapping[str, str],
    working_directory: str,
) -> JsonObject:
    exact_environment = dict(sorted(environment.items()))
    redacted_environment = dict(exact_environment)
    owner_token = redacted_environment.get("EXO_BENCHMARK_OWNER_TOKEN")
    if owner_token is None:
        raise Tp2LocalDiagnosticError("launch environment lacks its ownership token")
    redacted_environment["EXO_BENCHMARK_OWNER_TOKEN"] = "<redacted>"
    return {
        "argv": list(command),
        "working_directory": working_directory,
        "environment": redacted_environment,
        "exact_environment_sha256": _canonical_sha256(exact_environment),
        "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
        "evidence_origin": "captured_immediately_before_popen",
    }


def require_launch_ports_available(spec: Tp2LocalProcessSpec) -> None:
    """Reject a stale service or distributed listener before creating a child."""

    for endpoint_name, endpoint in (
        ("service", spec.service_endpoint),
        ("distributed", spec.distributed_coordinator),
    ):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((endpoint.ip, endpoint.port))
        except OSError as error:
            raise Tp2LocalDiagnosticError(
                f"TP2 {endpoint_name} endpoint is occupied or unavailable: {endpoint}"
            ) from error
        finally:
            probe.close()


def _ipv4_proc_listener_token(endpoint: Host) -> str:
    packed_address = socket.inet_aton(endpoint.ip)
    little_endian_address = int.from_bytes(packed_address, "little")
    return f"{little_endian_address:08X}:{endpoint.port:04X}"


def _find_ipv4_listener_inodes(
    endpoint: Host,
    *,
    proc_root: Path,
) -> set[str]:
    expected_listener = _ipv4_proc_listener_token(endpoint)
    listener_inodes: set[str] = set()
    lines = (proc_root / "net/tcp").read_text(encoding="ascii").splitlines()[1:]
    for line in lines:
        fields = line.split()
        if len(fields) >= 10 and fields[1] == expected_listener and fields[3] == "0A":
            listener_inodes.add(fields[9])
    return listener_inodes


def _verify_listener_ownership(
    endpoint: Host,
    running: RunningParent,
    listener_inodes: set[str],
    *,
    proc_root: Path,
    endpoint_role: str,
) -> JsonObject:
    listener_pids: set[int] = set()
    expected_owner_entry = (
        f"EXO_BENCHMARK_OWNER_TOKEN={running.owned.owner_token}".encode()
    )
    try:
        process_directories = tuple(
            entry for entry in proc_root.iterdir() if entry.name.isdigit()
        )
        for process_directory in process_directories:
            descriptor_directory = process_directory / "fd"
            try:
                descriptors = tuple(descriptor_directory.iterdir())
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            owns_listener = False
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if target.startswith("socket:[") and target[8:-1] in listener_inodes:
                    owns_listener = True
                    break
            if not owns_listener:
                continue
            pid = int(process_directory.name)
            stat_fields = (
                (process_directory / "stat").read_text().rsplit(")", 1)[1].split()
            )
            process_group_id = int(stat_fields[2])
            session_id = int(stat_fields[3])
            environment = (process_directory / "environ").read_bytes().split(b"\0")
            if (
                process_group_id != running.owned.process_group_id
                or session_id != running.owned.pid
                or expected_owner_entry not in environment
            ):
                raise Tp2LocalDiagnosticError(
                    f"TP2 {endpoint_role} listener is not owned by this benchmark run"
                )
            listener_pids.add(pid)
    except (OSError, UnicodeError, IndexError, ValueError) as error:
        raise Tp2LocalDiagnosticError(
            f"cannot prove ownership of the TP2 {endpoint_role} listener"
        ) from error
    if not listener_pids:
        raise Tp2LocalDiagnosticError(
            f"TP2 {endpoint_role} listener has no observable owning process"
        )
    return {
        "endpoint": str(endpoint),
        "endpoint_role": endpoint_role,
        "listener_inodes": sorted(listener_inodes),
        "listener_pids": sorted(listener_pids),
        "owned_process_group_id": running.owned.process_group_id,
        "owned_session_id": running.owned.pid,
        "owner_token_sha256": hashlib.sha256(
            running.owned.owner_token.encode()
        ).hexdigest(),
        "verification": "proc_listener_fd_process_group_session_and_owner_token",
    }


def verify_owned_service_listener(
    spec: Tp2LocalProcessSpec,
    running: RunningParent,
    *,
    proc_root: Path = Path("/proc"),
) -> JsonObject:
    """Bind the ready HTTP listener to this run's owned process group and token."""

    try:
        listener_inodes = _find_ipv4_listener_inodes(
            spec.service_endpoint,
            proc_root=proc_root,
        )
    except (OSError, UnicodeError, IndexError) as error:
        raise Tp2LocalDiagnosticError(
            "cannot inspect the TP2 service listener"
        ) from error
    if not listener_inodes:
        raise Tp2LocalDiagnosticError(
            "ready TP2 endpoint has no matching kernel listener"
        )
    return _verify_listener_ownership(
        spec.service_endpoint,
        running,
        listener_inodes,
        proc_root=proc_root,
        endpoint_role="service",
    )


def observe_distributed_coordinator_listener_ownership(
    spec: Tp2LocalProcessSpec,
    running: RunningParent,
    *,
    proc_root: Path = Path("/proc"),
) -> JsonObject:
    """Verify a surviving TP rendezvous listener, without requiring persistence."""

    lifecycle_note = (
        "SGLang does not expose a readiness contract requiring dist_init_addr "
        "to remain in LISTEN after the local TP ranks initialize"
    )
    try:
        listener_inodes = _find_ipv4_listener_inodes(
            spec.distributed_coordinator,
            proc_root=proc_root,
        )
    except (OSError, UnicodeError, IndexError) as error:
        raise Tp2LocalDiagnosticError(
            "cannot inspect the TP2 distributed coordinator listener"
        ) from error
    if not listener_inodes:
        return {
            "endpoint": str(spec.distributed_coordinator),
            "endpoint_role": "distributed_coordinator",
            "status": "not_listening_after_tp_initialization",
            "verification": "launch_preflight_and_exact_server_info_dist_init_addr",
            "persistent_listener_required": False,
            "lifecycle_note": lifecycle_note,
        }
    evidence = _verify_listener_ownership(
        spec.distributed_coordinator,
        running,
        listener_inodes,
        proc_root=proc_root,
        endpoint_role="distributed_coordinator",
    )
    evidence.update(
        {
            "status": "listening_and_owned",
            "persistent_listener_required": False,
            "lifecycle_note": lifecycle_note,
        }
    )
    return evidence


def _read_process_identity(pid: int) -> tuple[int, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[2]), int(fields[19])


def _close_log_file(log_file: IO[str]) -> str | None:
    try:
        log_file.close()
    except BaseException as error:
        return f"{type(error).__name__}: {error}"
    return None


def start_local_parent(
    spec: Tp2LocalProcessSpec,
    config: Tp2LocalDiagnosticConfig,
    owner_token: str,
) -> RunningParent:
    """Start and register the one local TP2 parent process group."""

    require_launch_ports_available(spec)
    log_path = config.result_directory / "rank-0.log"
    log_file = log_path.open("x", encoding="utf-8")
    command = build_parent_command(spec)
    environment = build_parent_environment(
        spec,
        owner_token,
        config.dwagon_socket_interface,
    )
    launch_evidence = _launch_evidence(
        command,
        environment,
        config.local_source_directory,
    )
    log_file.write(
        "EXO_TP2_LAUNCH "
        + json.dumps(
            {
                "rank": 0,
                "argv": list(command),
                "environment": _selected_environment(environment),
            },
            sort_keys=True,
        )
        + "\n"
    )
    log_file.flush()
    try:
        _write_launch_pending_ownership_journal(
            config,
            spec=spec,
            owner_token=owner_token,
            log_path=str(log_path),
            launch_evidence=launch_evidence,
        )
    except BaseException as journal_error:
        log_close_error = _close_log_file(log_file)
        launch_evidence["popen_attempted"] = False
        launch_evidence["popen_returned"] = False
        close_note = (
            "" if log_close_error is None else f"; log close failed: {log_close_error}"
        )
        raise Tp2ParentLaunchError(
            "refusing to call Popen without a durable launch-pending journal: "
            f"{type(journal_error).__name__}: {journal_error}{close_note}",
            journal_created=False,
            journal_cleared=False,
            launch_evidence=launch_evidence,
            popen_attempted=False,
            journal_error=f"{type(journal_error).__name__}: {journal_error}",
        ) from journal_error

    try:
        process = subprocess.Popen(
            command,
            cwd=config.local_source_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException as popen_error:
        journal_cleared = False
        journal_errors: list[str] = []
        try:
            _write_failed_launch_ownership_journal(
                config,
                spec=spec,
                owner_token=owner_token,
                log_path=str(log_path),
                launch_evidence=launch_evidence,
                failure=f"{type(popen_error).__name__}: {popen_error}",
            )
        except BaseException as update_error:
            journal_errors.append(
                "failed to publish launch-failed state: "
                f"{type(update_error).__name__}: {update_error}"
            )
        try:
            _clear_ownership_journal(config)
            journal_cleared = True
        except BaseException as clear_error:
            journal_errors.append(
                f"failed to clear launch journal: {type(clear_error).__name__}: "
                f"{clear_error}"
            )
        journal_error = "; ".join(journal_errors) or None
        log_close_error = _close_log_file(log_file)
        launch_evidence["popen_attempted"] = True
        launch_evidence["popen_returned"] = False
        retained_note = (
            "launch-pending journal was cleared"
            if journal_cleared
            else f"launch-pending journal is retained: {journal_error}"
        )
        if log_close_error is not None:
            retained_note += f"; log close failed: {log_close_error}"
        raise Tp2ParentLaunchError(
            "Popen failed before a parent process handle was returned; "
            f"{retained_note}: {type(popen_error).__name__}: {popen_error}",
            journal_created=True,
            journal_cleared=journal_cleared,
            launch_evidence=launch_evidence,
            popen_attempted=True,
            journal_error=journal_error,
        ) from popen_error

    launch_evidence["popen_attempted"] = True
    launch_evidence["popen_returned"] = True
    journal_created = True
    observed_process_group_id: int | None = None
    observed_start_time_ticks: int | None = None
    journal_publication_errors: list[str] = []
    try:
        _write_partial_start_ownership_journal(
            config,
            process=process,
            owner_token=owner_token,
            ownership_namespace=str(spec.service_endpoint.port),
            log_path=str(log_path),
            observed_process_group_id=None,
            observed_start_time_ticks=None,
            cleanup_evidence=None,
            failure=None,
        )
        process_group_id, start_time_ticks = _read_process_identity(process.pid)
        observed_process_group_id = process_group_id
        observed_start_time_ticks = start_time_ticks
        if process_group_id != process.pid:
            raise Tp2LocalDiagnosticError(
                "local TP2 parent is not its process-group leader"
            )
        owned = pipeline.OwnedStageProcess(
            rank=0,
            host_name="dwagon",
            pid=process.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=str(spec.service_endpoint.port),
            remote=False,
            transport_pid=process.pid,
            log_path=str(log_path),
        )
        return RunningParent(
            owned=owned,
            process=process,
            log_file=log_file,
            launch_evidence=launch_evidence,
        )
    except BaseException as registration_error:
        try:
            cleanup_receipt = pipeline._terminate_unregistered_local_process(
                process,
                config.cleanup_timeout_seconds,
            )
            cleanup_evidence = cast(
                JsonObject,
                cleanup_receipt.model_dump(mode="json"),
            )
        except BaseException as cleanup_error:
            cleanup_evidence = {
                "host_name": "dwagon",
                "ownership_verified": False,
                "terminated": False,
                "forced": False,
                "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
            }
        cleanup_verified = (
            cleanup_evidence.get("ownership_verified") is True
            and cleanup_evidence.get("terminated") is True
        )
        try:
            _write_partial_start_ownership_journal(
                config,
                process=process,
                owner_token=owner_token,
                ownership_namespace=str(spec.service_endpoint.port),
                log_path=str(log_path),
                observed_process_group_id=observed_process_group_id,
                observed_start_time_ticks=observed_start_time_ticks,
                cleanup_evidence=cleanup_evidence,
                failure=f"{type(registration_error).__name__}: {registration_error}",
            )
            journal_created = True
        except BaseException as journal_error:
            journal_publication_errors.append(
                f"{type(journal_error).__name__}: {journal_error}"
            )
        try:
            log_file.close()
        except BaseException as log_error:
            journal_publication_errors.append(
                f"log close failed: {type(log_error).__name__}: {log_error}"
            )

        process_evidence: JsonObject = {
            "rank": 0,
            "host_name": "dwagon",
            "pid": process.pid,
            "assumed_process_group_id": process.pid,
            "observed_process_group_id": observed_process_group_id,
            "observed_start_time_ticks": observed_start_time_ticks,
            "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
            "ownership_namespace": str(spec.service_endpoint.port),
            "identity_registered": False,
            "journal_publication_errors": journal_publication_errors,
        }
        raise Tp2PartialParentStartError(
            "TP2 parent process was created but could not be registered; "
            f"cleanup_verified={cleanup_verified}: "
            f"{type(registration_error).__name__}: {registration_error}",
            cleanup_evidence=cleanup_evidence,
            cleanup_verified=cleanup_verified,
            journal_created=journal_created,
            launch_evidence=launch_evidence,
            process_evidence=process_evidence,
        ) from registration_error


def stop_local_parent(
    running: RunningParent,
    cleanup_timeout_seconds: float,
) -> pipeline.ProcessCleanupReceipt:
    return pipeline.stop_local_stage(
        cast(pipeline.RunningStage, running),
        cleanup_timeout_seconds,
    )


def wait_for_parent_readiness(
    spec: Tp2LocalProcessSpec,
    running: RunningParent,
    timeout_seconds: float,
) -> tuple[JsonObject, ...]:
    return pipeline.wait_for_all_stages(
        cast(Sequence[pipeline.SglangKtProcessLaunchSpec], (spec,)),
        cast(Sequence[pipeline.RunningStage], (running,)),
        timeout_seconds,
    )


def _verify_runtime_and_model_contract(
    config: Tp2LocalDiagnosticConfig,
) -> JsonObject:
    try:
        return pp2._verify_runtime_and_model_contract(
            cast(pp2.Pp2LocalDiagnosticConfig, config)
        )
    except pp2.Pp2LocalDiagnosticError as error:
        raise Tp2LocalDiagnosticError(str(error)) from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ownership_journal_path(config: Tp2LocalDiagnosticConfig) -> Path:
    return config.result_directory / _OWNERSHIP_JOURNAL_NAME


def _publish_ownership_journal(
    config: Tp2LocalDiagnosticConfig,
    payload: JsonObject,
) -> None:
    destination = _ownership_journal_path(config)
    temporary = config.result_directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("ownership journal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
        _fsync_directory(config.result_directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _write_launch_pending_ownership_journal(
    config: Tp2LocalDiagnosticConfig,
    *,
    spec: Tp2LocalProcessSpec,
    owner_token: str,
    log_path: str,
    launch_evidence: JsonObject,
) -> None:
    """Publish recovery intent before the process-creating syscall is allowed."""

    _publish_ownership_journal(
        config,
        {
            "schema_version": 1,
            "status": "launch_pending_before_popen",
            "run_id": config.run_id,
            "updated_at_utc": _utc_now(),
            "planned_parent_process_count": 1,
            "started_parent_process_count": 0,
            "processes": [],
            "launch_intent": {
                "host_name": "dwagon",
                "owner_token": owner_token,
                "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
                "ownership_namespace": str(spec.service_endpoint.port),
                "expected_new_session_and_process_group_leader": True,
                "service_endpoint": str(spec.service_endpoint),
                "distributed_coordinator": str(spec.distributed_coordinator),
                "argv": launch_evidence["argv"],
                "working_directory": launch_evidence["working_directory"],
                "exact_environment_sha256": launch_evidence["exact_environment_sha256"],
                "log_path": log_path,
            },
            "recovery": {
                "process_discovery": (
                    "scan /proc/[0-9]*/environ for the exact raw owner_token"
                ),
                "termination_guard": (
                    "require the exact owner_token before terminating a discovered "
                    "new-session process group"
                ),
                "pid_may_be_absent_reason": (
                    "this record is durably published before Popen returns a PID"
                ),
            },
        },
    )


def _write_failed_launch_ownership_journal(
    config: Tp2LocalDiagnosticConfig,
    *,
    spec: Tp2LocalProcessSpec,
    owner_token: str,
    log_path: str,
    launch_evidence: JsonObject,
    failure: str,
) -> None:
    """Record a Popen exception before attempting to remove recovery intent."""

    _publish_ownership_journal(
        config,
        {
            "schema_version": 1,
            "status": "launch_failed_before_process_handle",
            "run_id": config.run_id,
            "updated_at_utc": _utc_now(),
            "planned_parent_process_count": 1,
            "started_parent_process_count": 0,
            "processes": [],
            "launch_intent": {
                "host_name": "dwagon",
                "owner_token": owner_token,
                "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
                "ownership_namespace": str(spec.service_endpoint.port),
                "expected_new_session_and_process_group_leader": True,
                "service_endpoint": str(spec.service_endpoint),
                "distributed_coordinator": str(spec.distributed_coordinator),
                "argv": launch_evidence["argv"],
                "working_directory": launch_evidence["working_directory"],
                "exact_environment_sha256": launch_evidence["exact_environment_sha256"],
                "log_path": log_path,
            },
            "recovery": {
                "process_discovery": (
                    "scan /proc/[0-9]*/environ for the exact raw owner_token"
                ),
                "termination_guard": (
                    "require the exact owner_token before terminating any process "
                    "found by the defensive scan"
                ),
                "process_handle_state": (
                    "Popen raised before returning a process handle; scan defensively"
                ),
            },
            "failure": failure,
        },
    )


def _write_ownership_journal(
    config: Tp2LocalDiagnosticConfig,
    running: RunningParent,
) -> None:
    _publish_ownership_journal(
        config,
        {
            "schema_version": 1,
            "status": "active_registered_parent",
            "run_id": config.run_id,
            "updated_at_utc": _utc_now(),
            "started_parent_process_count": 1,
            "processes": [asdict(running.owned)],
        },
    )


def _write_partial_start_ownership_journal(
    config: Tp2LocalDiagnosticConfig,
    *,
    process: subprocess.Popen[str],
    owner_token: str,
    ownership_namespace: str,
    log_path: str,
    observed_process_group_id: int | None,
    observed_start_time_ticks: int | None,
    cleanup_evidence: JsonObject | None,
    failure: str | None,
) -> None:
    cleanup_verified = cleanup_evidence is not None and (
        cleanup_evidence.get("ownership_verified") is True
        and cleanup_evidence.get("terminated") is True
    )
    _publish_ownership_journal(
        config,
        {
            "schema_version": 1,
            "status": (
                "partial_start_cleanup_verified"
                if cleanup_verified
                else "partial_start_recovery_required"
                if cleanup_evidence is not None
                else "partial_start_registration_pending"
            ),
            "run_id": config.run_id,
            "updated_at_utc": _utc_now(),
            "started_parent_process_count": 1,
            "processes": [
                {
                    "rank": 0,
                    "host_name": "dwagon",
                    "pid": process.pid,
                    "assumed_process_group_id": process.pid,
                    "observed_process_group_id": observed_process_group_id,
                    "observed_start_time_ticks": observed_start_time_ticks,
                    "owner_token": owner_token,
                    "ownership_namespace": ownership_namespace,
                    "remote": False,
                    "transport_pid": process.pid,
                    "log_path": log_path,
                    "identity_registered": False,
                }
            ],
            "cleanup": cleanup_evidence,
            "failure": failure,
        },
    )


def _clear_ownership_journal(config: Tp2LocalDiagnosticConfig) -> None:
    _ownership_journal_path(config).unlink()
    _fsync_directory(config.result_directory)


def _process_receipt(running: RunningParent) -> JsonObject:
    owned = asdict(running.owned)
    owned["owner_token"] = hashlib.sha256(
        running.owned.owner_token.encode()
    ).hexdigest()
    return cast(JsonObject, owned)


def _log_receipt(config: Tp2LocalDiagnosticConfig) -> JsonObject | None:
    path = config.result_directory / "rank-0.log"
    if not path.is_file():
        return None
    status = path.stat()
    if status.st_size > _LOG_MAXIMUM_BYTES:
        raise Tp2LocalDiagnosticError("TP2 parent log exceeds diagnostic size bound")
    return {
        "rank": 0,
        "path": str(path),
        "size_bytes": status.st_size,
        "sha256": _sha256_file(path),
    }


def _write_receipt(config: Tp2LocalDiagnosticConfig, payload: JsonObject) -> None:
    path = config.result_directory / _RESULT_FILENAME
    temporary = config.result_directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _fsync_directory(config.result_directory)


def _configuration_receipt(
    config: Tp2LocalDiagnosticConfig,
    spec: Tp2LocalProcessSpec,
) -> JsonObject:
    return {
        "host_name": "dwagon",
        "scope": "single_host",
        "parent_process_count": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 2,
        "ordered_gpu_uuids": list(spec.ordered_gpu_uuids),
        "physical_cpu_ids": list(spec.cpu_cores),
        "memory_nodes": list(spec.memory_nodes),
        "cpu_infer_threads": spec.cpu_infer_threads,
        "threadpool_count": spec.threadpool_count,
        "resident_gpu_experts": spec.resident_gpu_experts,
        "resident_gpu_expert_count_semantics": (
            "per_layer_global_logical_expert_count"
        ),
        "hybrid_execution_launch_intent": {
            "evidence_scope": "launch_intent_not_observed_execution",
            "cpuinfer_owner_tensor_parallel_rank": 0,
            "cpuinfer_result_scope": "full_cpu_expert_result",
            "cpuinfer_result_contribution_count": 1,
            "cpuinfer_merge_order": (
                "contributed_exactly_once_before_tensor_parallel_all_reduce"
            ),
            "resident_expert_gpu_weight_partition": (
                "tensor_parallel_half_shard_per_gpu_for_each_resident_expert"
            ),
        },
        "static_memory_fraction": config.static_memory_fraction,
        "nccl_transport_policy": "automatic_local_p2p_nvlink_allowed",
    }


def run_diagnostic(config: Tp2LocalDiagnosticConfig) -> JsonObject:
    """Launch, measure, clean up, and publish one local PP1/TP2 receipt."""

    runtime_contract = _verify_runtime_and_model_contract(config)
    spec = build_tp2_local_process_spec(config)
    process_spec = spec.receipt()
    process_spec_sha256 = _canonical_sha256(process_spec)
    stage_cpu_bindings: tuple[StageCpuCoreBinding, ...] = (
        (0, tuple(range(56))),
        (1, tuple(range(56, 112))),
    )
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    owner_token = uuid.uuid4().hex
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    running: RunningParent | None = None
    readiness: tuple[JsonObject, ...] = ()
    readiness_ownership: JsonObject | None = None
    distributed_coordinator_ownership: JsonObject | None = None
    server_info: JsonObject | None = None
    sanity: JsonObject | None = None
    workloads: list[JsonValue] = []
    summaries: list[JsonValue] = []
    telemetry_snapshots: list[pp2._HostTelemetrySnapshot] = []
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []
    cleanup_complete = False
    journal_created = False
    journal_cleared = False
    journal_clear_reason: str | None = None
    parent_launch_failure: JsonObject | None = None
    partial_start: JsonObject | None = None
    partial_start_cleanup_verified: bool | None = None
    launch_evidence: JsonObject | None = None
    signal_state = pp2._ManagedSignalState()
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, signal_state.handle)
        try:
            telemetry_snapshots.append(
                pp2._safe_collect_host_telemetry_snapshot(
                    "before_stage_launch",
                    stage_cpu_bindings,
                    spec.ordered_gpu_uuids,
                )
            )
            signal_state.checkpoint()
            with signal_state.defer():
                running = start_local_parent(spec, config, owner_token)
                launch_evidence = running.launch_evidence
                journal_created = True
                _write_ownership_journal(config, running)
            readiness = wait_for_parent_readiness(
                spec,
                running,
                config.readiness_timeout_seconds,
            )
            readiness_ownership = verify_owned_service_listener(spec, running)
            telemetry_snapshots.append(
                pp2._safe_collect_host_telemetry_snapshot(
                    "after_readiness",
                    stage_cpu_bindings,
                    spec.ordered_gpu_uuids,
                )
            )
            signal_state.checkpoint()
            with Glm47NativeServingClient(
                f"http://{spec.service_endpoint}",
                timeout_seconds=config.request_timeout_seconds,
            ) as client:
                server_info = validate_tp2_server_info(
                    client.server_info(),
                    spec,
                )
                distributed_coordinator_ownership = (
                    observe_distributed_coordinator_listener_ownership(spec, running)
                )
                sanity_evidence = run_glm47_serving_sanity(client, spec.model_path)
                sanity = cast(JsonObject, sanity_evidence.model_dump(mode="json"))
                telemetry_snapshots.append(
                    pp2._safe_collect_host_telemetry_snapshot(
                        "after_sanity",
                        stage_cpu_bindings,
                        spec.ordered_gpu_uuids,
                    )
                )
                if running.process.poll() is not None:
                    raise Tp2LocalDiagnosticError(
                        "the TP2 parent exited during semantic sanity"
                    )
                signal_state.checkpoint()

                def build_workload_phase_observer(
                    workload_kind: str,
                ) -> Callable[[str], None]:
                    def record_workload_phase(boundary: str) -> None:
                        suffix = (
                            "warmups" if boundary == "warmups_complete" else "samples"
                        )
                        telemetry_snapshots.append(
                            pp2._safe_collect_host_telemetry_snapshot(
                                f"after_{workload_kind}_{suffix}",
                                stage_cpu_bindings,
                                spec.ordered_gpu_uuids,
                            )
                        )
                        signal_state.checkpoint()

                    return record_workload_phase

                for kind in ("prefill", "decode"):
                    workload = run_glm47_serving_workload(
                        client,
                        prepare_glm47_serving_workload(kind),
                        warmup_count=config.warmup_count,
                        sample_count=config.sample_count,
                        phase_observer=build_workload_phase_observer(kind),
                    )
                    if running.process.poll() is not None:
                        raise Tp2LocalDiagnosticError(
                            f"the TP2 parent exited during the {kind} workload"
                        )
                    signal_state.checkpoint()
                    workloads.append(cast(JsonObject, workload.model_dump(mode="json")))
                    summaries.append(pipeline.summarize_workload(workload))
        except BaseException as error:
            if isinstance(error, Tp2ParentLaunchError):
                launch_evidence = error.launch_evidence
                journal_created = journal_created or error.journal_created
                journal_cleared = journal_cleared or error.journal_cleared
                if error.journal_cleared:
                    journal_clear_reason = "popen_failed_before_process_creation"
                parent_launch_failure = {
                    "popen_attempted": error.popen_attempted,
                    "popen_returned": False,
                    "process_created": False,
                    "journal_error": error.journal_error,
                }
            elif isinstance(error, Tp2PartialParentStartError):
                partial_start = error.process_evidence
                partial_start_cleanup_verified = error.cleanup_verified
                launch_evidence = error.launch_evidence
                journal_created = journal_created or error.journal_created
                cleanup.append(
                    {
                        "rank": 0,
                        "partial_start": True,
                        **error.cleanup_evidence,
                    }
                )
            failure = (
                Tp2LocalDiagnosticError(str(error))
                if isinstance(
                    error, (pipeline.Pp3DiagnosticError, pp2.Pp2LocalDiagnosticError)
                )
                else error
            )
        finally:
            signal_state.begin_cleanup()
            if running is not None:
                try:
                    receipt = stop_local_parent(
                        running,
                        config.cleanup_timeout_seconds,
                    )
                    cleanup.append(
                        {
                            "rank": 0,
                            **receipt.model_dump(mode="json"),
                        }
                    )
                except BaseException as error:
                    cleanup.append(
                        {
                            "rank": 0,
                            "host_name": running.owned.host_name,
                            "ownership_verified": False,
                            "terminated": False,
                            "forced": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if failure is None:
                        failure = error
            if running is not None:
                cleanup_complete = (
                    len(cleanup) == 1
                    and cast(dict[str, object], cleanup[0]).get("ownership_verified")
                    is True
                    and cast(dict[str, object], cleanup[0]).get("terminated") is True
                )
            elif partial_start is not None:
                cleanup_complete = partial_start_cleanup_verified is True
            else:
                cleanup_complete = True
            if journal_created and not journal_cleared and cleanup_complete:
                try:
                    _clear_ownership_journal(config)
                    journal_cleared = True
                    journal_clear_reason = (
                        "verified_registered_parent_cleanup"
                        if running is not None
                        else "verified_partial_parent_cleanup"
                        if partial_start is not None
                        else "popen_failed_before_process_creation"
                    )
                except BaseException as error:
                    if failure is None:
                        failure = error
            telemetry_snapshots.append(
                pp2._safe_collect_host_telemetry_snapshot(
                    "after_cleanup",
                    stage_cpu_bindings,
                    spec.ordered_gpu_uuids,
                )
            )
    finally:
        for managed_signal, previous_handler in previous_handlers.items():
            signal.signal(managed_signal, previous_handler)

    journal_path = _ownership_journal_path(config)
    journal_retained = journal_path.exists()
    parent_registered = running is not None
    parent_process_created = parent_registered or partial_start is not None
    log_receipt = _log_receipt(config)
    host_telemetry = pp2._host_telemetry_receipt(telemetry_snapshots)
    host_telemetry["cpu_binding_label_semantics"] = {
        "source_field_name": "pipeline_rank",
        "meaning_in_this_tp2_receipt": "cpuinfer_threadpool_index",
        "bindings": [
            {
                "cpuinfer_threadpool_index": threadpool_index,
                "physical_cpu_ids": list(cpu_cores),
            }
            for threadpool_index, cpu_cores in stage_cpu_bindings
        ],
    }
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm47_flash_pp1_tp2_local_engineering_diagnostic",
        "status": (
            "passed"
            if failure is None
            and parent_registered
            and cleanup_complete
            and not journal_retained
            else "failed"
        ),
        "performance_comparable": False,
        "profiler": "none",
        "instrumentation": "nccl_info_logging_and_phase_boundary_host_telemetry",
        "run_id": config.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "topology": {
            "scope": "dwagon_local",
            "host_count": 1,
            "parent_process_count": 1,
            "gpu_worker_count": 2,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 2,
            "cross_host_transport": False,
            "nvlink_p2p_allowed": True,
        },
        "configuration": _configuration_receipt(config, spec),
        "runtime_contract": runtime_contract,
        "process_spec": process_spec,
        "process_spec_sha256": process_spec_sha256,
        "launch": (
            launch_evidence
            if launch_evidence is not None
            else {
                "process_created": False,
                "evidence_origin": "no_successful_popen",
            }
        ),
        "processes": [] if running is None else [_process_receipt(running)],
        "partial_start": partial_start,
        "parent_launch_failure": parent_launch_failure,
        "readiness": list(readiness),
        "readiness_ownership": readiness_ownership,
        "distributed_coordinator_ownership": distributed_coordinator_ownership,
        "server_info": server_info,
        "sanity": sanity,
        "workloads": workloads,
        "benchmark_summary": summaries,
        "host_telemetry": host_telemetry,
        "planned_parent_process_count": 1,
        "started_parent_process_count": int(parent_process_created),
        "registered_parent_process_count": int(parent_registered),
        "all_planned_processes_started": parent_registered,
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "managed_signal": signal_state.signal_number,
        "ownership_journal": {
            "path": str(journal_path),
            "created": journal_created,
            "cleared_before_receipt": journal_cleared,
            "clear_reason": journal_clear_reason,
            "retained": journal_retained,
        },
        "logs": [] if log_receipt is None else [log_receipt],
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    _write_receipt(config, payload)
    if failure is not None:
        raise Tp2LocalDiagnosticError(
            "local TP2 diagnostic failed; evidence is in "
            f"{config.result_directory}: {type(failure).__name__}: {failure}"
        ) from failure
    if not cleanup_complete:
        raise Tp2LocalDiagnosticError(
            "local TP2 cleanup was incomplete; evidence is in "
            f"{config.result_directory}"
        )
    return payload


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _memory_fraction(raw: str) -> float:
    value = float(raw)
    if not 0.8 <= value <= 0.95 or not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            "value must be finite and between 0.8 and 0.95"
        )
    return value


def _resident_gpu_experts(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= MAXIMUM_RESIDENT_GPU_EXPERTS:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAXIMUM_RESIDENT_GPU_EXPERTS}"
        )
    return value


def _sha256_argument(raw: str) -> str:
    if validation.SHA256_PATTERN.fullmatch(raw) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--dwagon-runtime-python", required=True)
    parser.add_argument(
        "--dwagon-runtime-install-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dwagon-runtime-install-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--dwagon-model-path",
        default=pipeline.DEFAULT_DWAGON_MODEL_PATH,
    )
    parser.add_argument(
        "--local-source-directory",
        default=pipeline.DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--dwagon-ip", default=DEFAULT_DWAGON_IP)
    parser.add_argument(
        "--dwagon-socket-interface",
        default=pipeline.DEFAULT_DWAGON_SOCKET_INTERFACE,
    )
    parser.add_argument(
        "--distributed-port",
        type=_positive_int,
        default=DEFAULT_DISTRIBUTED_PORT,
    )
    parser.add_argument(
        "--service-port",
        type=_positive_int,
        default=DEFAULT_SERVICE_PORT,
    )
    parser.add_argument(
        "--static-memory-fraction",
        type=_memory_fraction,
        default=DEFAULT_STATIC_MEMORY_FRACTION,
    )
    parser.add_argument(
        "--resident-gpu-experts",
        type=_resident_gpu_experts,
        default=DEFAULT_RESIDENT_GPU_EXPERTS,
    )
    parser.add_argument(
        "--readiness-timeout-seconds",
        type=_positive_float,
        default=1800.0,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=900.0,
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--samples", type=_positive_int, default=3)
    return parser


def _config_from_arguments(
    arguments: argparse.Namespace,
) -> Tp2LocalDiagnosticConfig:
    run_id = cast(str, arguments.run_id)
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise Tp2LocalDiagnosticError(
            "run_id must contain only safe identifier characters"
        )
    distributed_port = cast(int, arguments.distributed_port)
    service_port = cast(int, arguments.service_port)
    if (
        distributed_port == service_port
        or distributed_port > 65535
        or service_port > 65535
    ):
        raise Tp2LocalDiagnosticError(
            "distributed and service ports must be distinct TCP ports"
        )
    runtime_install_receipt = cast(Path, arguments.dwagon_runtime_install_receipt)
    if not runtime_install_receipt.is_absolute() or runtime_install_receipt != Path(
        os.path.normpath(runtime_install_receipt)
    ):
        raise Tp2LocalDiagnosticError(
            "dwagon_runtime_install_receipt must be an absolute normalized path"
        )
    runtime_python = cast(str, arguments.dwagon_runtime_python)
    source_directory = cast(str, arguments.local_source_directory)
    if (
        not Path(runtime_python).is_absolute()
        or not Path(source_directory).is_absolute()
    ):
        raise Tp2LocalDiagnosticError(
            "runtime Python and local source directory must be absolute paths"
        )
    return Tp2LocalDiagnosticConfig(
        run_id=run_id,
        result_directory=cast(Path, arguments.result_directory).resolve(),
        dwagon_runtime_python=runtime_python,
        dwagon_runtime_install_receipt=runtime_install_receipt,
        dwagon_runtime_install_receipt_sha256=cast(
            str,
            arguments.dwagon_runtime_install_receipt_sha256,
        ),
        dwagon_model_path=cast(str, arguments.dwagon_model_path),
        local_source_directory=source_directory,
        dwagon_ip=cast(str, arguments.dwagon_ip),
        dwagon_socket_interface=cast(str, arguments.dwagon_socket_interface),
        distributed_port=distributed_port,
        service_port=service_port,
        static_memory_fraction=cast(float, arguments.static_memory_fraction),
        resident_gpu_experts=cast(int, arguments.resident_gpu_experts),
        readiness_timeout_seconds=cast(float, arguments.readiness_timeout_seconds),
        request_timeout_seconds=cast(float, arguments.request_timeout_seconds),
        cleanup_timeout_seconds=cast(float, arguments.cleanup_timeout_seconds),
        warmup_count=cast(int, arguments.warmups),
        sample_count=cast(int, arguments.samples),
    )


def main() -> int:
    try:
        config = _config_from_arguments(_parser().parse_args())
        payload = run_diagnostic(config)
    except (Tp2LocalDiagnosticError, OSError, ValueError) as error:
        print(f"local TP2 diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload["benchmark_summary"], indent=2, sort_keys=True))
    print(config.result_directory / _RESULT_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
