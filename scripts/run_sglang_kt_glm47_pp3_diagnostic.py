#!/usr/bin/env python3
"""Run the pinned three-stage GLM-4.7 Flash SGLang-KT diagnostic.

This is an engineering benchmark, not a schema-v3 comparable performance
receipt. It launches two NUMA-bound ranks on dwagon and one rank on fwuff,
requires all three HTTP endpoints to become ready, runs the canonical semantic
sanity probe, then measures the canonical 1024/32 and 128/128 workloads.

The fwuff process uses the ownership supervisor from
``two_host_mlx_nccl_poc.py``. The supervisor and an independent stop command
both verify the process group, PID start time, and owner token before signaling.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import select
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Final, Literal, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import httpx  # noqa: E402

from exo.shared.types.common import Host, NodeId  # noqa: E402
from exo.shared.types.worker.sglang_kt import (  # noqa: E402
    SglangKtLaunchPlan,
    SglangKtStageSpec,
)
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CONTEXT_LENGTH,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_LAYER_COUNT,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
    GLM_4_7_FLASH_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_pp3_diagnostic_process_launch_specs,
)
from exo.worker.sglang_kt.process_supervisor import (  # noqa: E402
    build_cpu_bound_sglang_kt_command,
    build_sglang_kt_process_environment,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (  # noqa: E402
    SglangKtServingWorkloadEvidence,
)
from scripts import run_sglang_kt_glm47_validation as validation  # noqa: E402
from scripts.sglang_kt_glm47_serving_client import (  # noqa: E402
    Glm47NativeServingClient,
    Glm47ServingClientError,
    prepare_glm47_serving_workload,
    run_glm47_serving_sanity,
    run_glm47_serving_workload,
)
from scripts.two_host_mlx_nccl_poc import (  # noqa: E402
    REMOTE_PROCESS_LAUNCH_SUPERVISOR_PROGRAM,
    REMOTE_PROCESS_STOP_PROGRAM,
    HcaCounterSnapshot,
    HcaPort,
    HostHcaCounterRequest,
    LinuxHostProbe,
    ProcessCleanupReceipt,
    RemoteOwnerReceipt,
    collect_hca_counter_snapshot,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

DWAGON_NODE_ID: Final = NodeId("dwagon")
FWUFF_NODE_ID: Final = NodeId("fwuff")
DWAGON_STAGE_ZERO_GPU: Final = "GPU-63a7760a-6164-0758-9228-03dbf35d721c"
DWAGON_STAGE_ONE_GPU: Final = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
FWUFF_STAGE_TWO_GPU: Final = "GPU-93e47864-13c3-0211-f3a9-ccee1a00d618"
DWAGON_STAGE_ZERO_CPUS: Final = tuple(range(56))
DWAGON_STAGE_ONE_CPUS: Final = tuple(range(56, 112))
FWUFF_STAGE_TWO_CPUS: Final = tuple(range(60))
PIPELINE_RANGES: Final = ((0, 16), (16, 32), (32, 47))
DEFAULT_DWAGON_MODEL_PATH: Final = (
    "/var/lib/exo/models/"
    "zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
DEFAULT_FWUFF_MODEL_PATH: Final = (
    "/mnt/sanic/exo/models/"
    "zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
DEFAULT_SOURCE_DIRECTORY: Final = "/var/lib/exo/deployments/exo-source-60216e1b"
DEFAULT_HCA_DEVICES: Final = ("mlx4_0:1", "mlx4_0:2")
DEFAULT_DWAGON_SOCKET_INTERFACE: Final = "ens13f0np0"
DEFAULT_FWUFF_SOCKET_INTERFACE: Final = "ens17f0"
DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER: Final = "pipeline-order"
DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL: Final = "cross-host-hca-local"
_REMOTE_START_TIMEOUT_SECONDS: Final = 30.0
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024

type DwagonStagePlacement = Literal[
    "pipeline-order",
    "cross-host-hca-local",
]


class Pp3DiagnosticError(RuntimeError):
    """Raised when the PP3 run cannot produce complete diagnostic evidence."""


@dataclass(frozen=True, slots=True)
class Pp3DiagnosticConfig:
    run_id: str
    result_directory: Path
    dwagon_runtime_python: str
    fwuff_runtime_python: str
    dwagon_model_path: str
    fwuff_model_path: str
    local_source_directory: str
    remote_source_directory: str
    ssh_target: str
    dwagon_ip: str
    fwuff_ip: str
    dwagon_socket_interface: str
    fwuff_socket_interface: str
    distributed_port: int
    stage_ports: tuple[int, int, int]
    hca_devices: tuple[str, ...]
    dwagon_stage_placement: DwagonStagePlacement
    resident_gpu_experts: int
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float
    warmup_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class OwnedStageProcess:
    rank: int
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    owner_token: str
    ownership_namespace: str
    remote: bool
    transport_pid: int
    log_path: str


@dataclass(slots=True)
class _RunningStage:
    owned: OwnedStageProcess
    process: subprocess.Popen[str]
    log_file: IO[str]
    pump_thread: threading.Thread | None


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


def _ssh_argv(ssh_target: str, command: Sequence[str]) -> tuple[str, ...]:
    return (
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
        ssh_target,
        shlex.join(command),
    )


def _read_process_identity(pid: int) -> tuple[int, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[2]), int(fields[19])


def _read_marker(process: subprocess.Popen[str], timeout_seconds: float) -> str:
    if process.stdout is None:
        raise Pp3DiagnosticError("remote supervisor has no ownership channel")
    ready, _, _ = select.select([process.stdout], [], [], timeout_seconds)
    if not ready:
        raise Pp3DiagnosticError("remote supervisor ownership handshake timed out")
    line = process.stdout.readline()
    if not line:
        raise Pp3DiagnosticError("remote supervisor closed during ownership handshake")
    return line.rstrip("\n")


def _pump_output(source: IO[str], destination: IO[str]) -> None:
    for line in source:
        destination.write(line)
        destination.flush()


def stage_ownership_namespace(spec: SglangKtProcessLaunchSpec) -> str:
    # The reused stop protocol verifies this token in /proc/<pid>/cmdline.
    # A stage endpoint is unique within the plan and is an exact launch argument.
    return str(spec.service_endpoint.port)


def build_pp3_plan(config: Pp3DiagnosticConfig) -> SglangKtLaunchPlan:
    """Build the exact dwagon -> dwagon -> fwuff hardware plan."""

    pipeline_order_placements = (
        (DWAGON_STAGE_ZERO_GPU, DWAGON_STAGE_ZERO_CPUS, 0),
        (DWAGON_STAGE_ONE_GPU, DWAGON_STAGE_ONE_CPUS, 1),
    )
    if config.dwagon_stage_placement == DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER:
        dwagon_placements = pipeline_order_placements
    elif config.dwagon_stage_placement == DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL:
        dwagon_placements = tuple(reversed(pipeline_order_placements))
    else:
        raise Pp3DiagnosticError(
            f"unsupported dwagon stage placement {config.dwagon_stage_placement}"
        )

    stage_inputs = (
        (
            DWAGON_NODE_ID,
            dwagon_placements[0][0],
            config.dwagon_ip,
            config.stage_ports[0],
            config.dwagon_model_path,
            dwagon_placements[0][1],
            dwagon_placements[0][2],
        ),
        (
            DWAGON_NODE_ID,
            dwagon_placements[1][0],
            config.dwagon_ip,
            config.stage_ports[1],
            config.dwagon_model_path,
            dwagon_placements[1][1],
            dwagon_placements[1][2],
        ),
        (
            FWUFF_NODE_ID,
            FWUFF_STAGE_TWO_GPU,
            config.fwuff_ip,
            config.stage_ports[2],
            config.fwuff_model_path,
            FWUFF_STAGE_TWO_CPUS,
            0,
        ),
    )
    stages = tuple(
        SglangKtStageSpec(
            pipeline_rank=rank,
            start_layer=PIPELINE_RANGES[rank][0],
            end_layer=PIPELINE_RANGES[rank][1],
            node_id=node_id,
            gpu_uuid=gpu_uuid,
            service_endpoint=Host(ip=service_ip, port=service_port),
            model_path=model_path,
            ktransformers_weight_path=model_path,
            cpu_cores=cpu_cores,
            memory_nodes=(memory_node,),
            cpu_infer_threads=len(cpu_cores),
            threadpool_count=1,
            ktransformers_method="BF16",
            resident_gpu_experts=config.resident_gpu_experts,
            max_deferred_experts_per_token=0,
            hca_devices=config.hca_devices,
        )
        for rank, (
            node_id,
            gpu_uuid,
            service_ip,
            service_port,
            model_path,
            cpu_cores,
            memory_node,
        ) in enumerate(stage_inputs)
    )
    return SglangKtLaunchPlan(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        target_profile=GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
        total_layers=GLM_4_7_FLASH_LAYER_COUNT,
        context_length=GLM_4_7_FLASH_CONTEXT_LENGTH,
        max_total_tokens=GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(
            ip=config.dwagon_ip,
            port=config.distributed_port,
        ),
        rank_zero_endpoint=stages[0].service_endpoint,
        stages=stages,
    )


def build_pp3_process_specs(
    config: Pp3DiagnosticConfig,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    plan = build_pp3_plan(config)
    return build_glm_4_7_flash_bf16_pp3_diagnostic_process_launch_specs(
        plan,
        {
            DWAGON_NODE_ID: config.dwagon_runtime_python,
            FWUFF_NODE_ID: config.fwuff_runtime_python,
        },
    )


def build_stage_environment(
    spec: SglangKtProcessLaunchSpec,
    owner_token: str,
    socket_interface: str,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an instrumentation-free environment with explicit NCCL logging."""

    source = os.environ if parent_environment is None else parent_environment
    retained_names = (
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
    base = {name: source[name] for name in retained_names if source.get(name)}
    base.setdefault("HOME", "/root")
    base.setdefault("LANG", "C.UTF-8")
    base.setdefault("LC_ALL", "C.UTF-8")
    base.setdefault(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/cuda/bin",
    )
    environment = build_sglang_kt_process_environment(spec, base)
    environment.update(
        {
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "GLOO_SOCKET_IFNAME": socket_interface,
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET,ENV",
            "NCCL_IB_MERGE_NICS": "1",
            "NCCL_SOCKET_IFNAME": socket_interface,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    validation.require_no_profiler_state(environment, spec.command, "")
    return environment


def _log_launch(
    log_file: IO[str],
    spec: SglangKtProcessLaunchSpec,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> None:
    selected_environment = {
        name: value
        for name, value in sorted(environment.items())
        if name.startswith(("CUDA_", "NCCL_", "SGLANG_", "PYTORCH_"))
        or name
        in {
            "EXO_BENCHMARK_OWNER_TOKEN",
            "PYTHONHASHSEED",
            "TOKENIZERS_PARALLELISM",
        }
    }
    selected_environment["EXO_BENCHMARK_OWNER_TOKEN"] = "<redacted>"
    log_file.write(
        "EXO_PP3_LAUNCH "
        + json.dumps(
            {
                "rank": spec.pipeline_rank,
                "argv": list(command),
                "environment": selected_environment,
            },
            sort_keys=True,
        )
        + "\n"
    )
    log_file.flush()


def _unregistered_process_group_members(process_group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            state = fields[0]
            observed_group = int(fields[2])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, IndexError, ValueError) as error:
            raise Pp3DiagnosticError(
                "cannot enumerate an unregistered local process group"
            ) from error
        if observed_group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _wait_unregistered_process_group_empty(
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _unregistered_process_group_members(process_group_id):
            return True
        time.sleep(0.1)
    return not _unregistered_process_group_members(process_group_id)


def _terminate_unregistered_local_process(
    process: subprocess.Popen[str],
    timeout_seconds: float,
) -> ProcessCleanupReceipt:
    # Popen returned only after start_new_session=True completed, so its live PID
    # is the process-group ID we created even if the first /proc read fails.
    errors: list[str] = []
    forced = False
    terminated = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as error:
        errors.append(f"SIGTERM failed: {error}")
    try:
        terminated = _wait_unregistered_process_group_empty(
            process.pid,
            timeout_seconds,
        )
    except (OSError, Pp3DiagnosticError) as error:
        errors.append(f"SIGTERM verification failed: {error}")
    if not terminated:
        forced = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"SIGKILL failed: {error}")
        try:
            terminated = _wait_unregistered_process_group_empty(
                process.pid,
                min(timeout_seconds, 5.0),
            )
        except (OSError, Pp3DiagnosticError) as error:
            errors.append(f"SIGKILL verification failed: {error}")
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        errors.append("unregistered local leader could not be reaped")
        terminated = False
    return ProcessCleanupReceipt(
        host_name="dwagon",
        ownership_verified=True,
        terminated=terminated,
        forced=forced,
        error="; ".join(errors) or None,
    )


def _start_local_stage(
    spec: SglangKtProcessLaunchSpec,
    config: Pp3DiagnosticConfig,
    owner_token: str,
    log_file: IO[str],
) -> _RunningStage:
    command = build_cpu_bound_sglang_kt_command(spec)
    environment = build_stage_environment(
        spec,
        owner_token,
        config.dwagon_socket_interface,
    )
    _log_launch(log_file, spec, command, environment)
    process = subprocess.Popen(
        command,
        cwd=config.local_source_directory,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        process_group_id, start_time_ticks = _read_process_identity(process.pid)
        if process_group_id != process.pid:
            raise Pp3DiagnosticError("local stage is not its process-group leader")
        owned = OwnedStageProcess(
            rank=spec.pipeline_rank,
            host_name="dwagon",
            pid=process.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=stage_ownership_namespace(spec),
            remote=False,
            transport_pid=process.pid,
            log_path=log_file.name,
        )
        return _RunningStage(owned, process, log_file, None)
    except BaseException as error:
        cleanup = _terminate_unregistered_local_process(
            process,
            config.cleanup_timeout_seconds,
        )
        if not cleanup.terminated:
            raise Pp3DiagnosticError(
                "local stage start failed and unregistered cleanup was incomplete: "
                f"{cleanup.error}"
            ) from error
        raise


def _close_failed_remote_transport(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(OSError):
        if process.stdin is not None:
            process.stdin.close()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


def _start_remote_stage(
    spec: SglangKtProcessLaunchSpec,
    config: Pp3DiagnosticConfig,
    owner_token: str,
    log_file: IO[str],
) -> _RunningStage:
    command = build_cpu_bound_sglang_kt_command(spec)
    environment = build_stage_environment(
        spec,
        owner_token,
        config.fwuff_socket_interface,
    )
    namespace = stage_ownership_namespace(spec)
    _log_launch(log_file, spec, command, environment)
    remote_command = (
        config.fwuff_runtime_python,
        "-c",
        REMOTE_PROCESS_LAUNCH_SUPERVISOR_PROGRAM,
        owner_token,
        namespace,
        config.remote_source_directory,
        json.dumps(environment, sort_keys=True),
        json.dumps(command),
    )
    process = subprocess.Popen(
        _ssh_argv(config.ssh_target, remote_command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    running: _RunningStage | None = None
    try:
        marker = _read_marker(process, _REMOTE_START_TIMEOUT_SECONDS)
        if marker != "EXO_SUPERVISOR_READY":
            raise Pp3DiagnosticError(f"invalid remote supervisor marker: {marker}")
        if process.stdin is None:
            raise Pp3DiagnosticError("remote supervisor has no acknowledgement channel")
        process.stdin.write("START\n")
        process.stdin.flush()
        marker = _read_marker(process, _REMOTE_START_TIMEOUT_SECONDS)
        if not marker.startswith("EXO_OWNER "):
            raise Pp3DiagnosticError(f"invalid remote ownership marker: {marker}")
        receipt = RemoteOwnerReceipt.model_validate_json(
            marker.removeprefix("EXO_OWNER ")
        )
        if (
            receipt.pid != receipt.process_group_id
            or receipt.owner_token != owner_token
            or receipt.namespace != namespace
        ):
            raise Pp3DiagnosticError("remote ownership receipt does not match launch")
        owned = OwnedStageProcess(
            rank=spec.pipeline_rank,
            host_name="fwuff",
            pid=receipt.pid,
            process_group_id=receipt.process_group_id,
            start_time_ticks=receipt.start_time_ticks,
            owner_token=receipt.owner_token,
            ownership_namespace=receipt.namespace,
            remote=True,
            transport_pid=process.pid,
            log_path=log_file.name,
        )
        running = _RunningStage(owned, process, log_file, None)
        assert process.stdout is not None
        pump = threading.Thread(
            target=_pump_output,
            args=(process.stdout, log_file),
            daemon=True,
            name="exo-pp3-rank2-log",
        )
        pump.start()
        running.pump_thread = pump
        return running
    except BaseException as error:
        if running is None:
            _close_failed_remote_transport(process)
        else:
            cleanup = stop_stage(running, config)
            if not (cleanup.ownership_verified and cleanup.terminated):
                raise Pp3DiagnosticError(
                    "remote stage start failed and registered cleanup was incomplete: "
                    f"{cleanup.error}"
                ) from error
        raise


def start_stage(
    spec: SglangKtProcessLaunchSpec,
    config: Pp3DiagnosticConfig,
    owner_token: str,
) -> _RunningStage:
    log_path = config.result_directory / f"rank-{spec.pipeline_rank}.log"
    log_file = log_path.open("x", encoding="utf-8")
    try:
        if spec.node_id == FWUFF_NODE_ID:
            return _start_remote_stage(spec, config, owner_token, log_file)
        return _start_local_stage(spec, config, owner_token, log_file)
    except BaseException:
        log_file.close()
        raise


def _local_group_ownership(owned: OwnedStageProcess) -> tuple[bool, tuple[int, ...]]:
    members: list[int] = []
    leader_seen = False
    owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={owned.owner_token}".encode()
    namespace = owned.ownership_namespace.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            state = fields[0]
            process_group_id = int(fields[2])
            session_id = int(fields[3])
            start_time_ticks = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return False, (owned.pid,)
        except (IndexError, ValueError):
            continue
        if process_group_id != owned.process_group_id:
            continue
        pid = int(entry.name)
        if session_id != owned.pid:
            return False, (pid,)
        if pid == owned.pid:
            leader_seen = True
            if start_time_ticks != owned.start_time_ticks:
                return False, (pid,)
            if state != "Z":
                try:
                    command_line = (entry / "cmdline").read_bytes()
                    environment = (entry / "environ").read_bytes().split(b"\0")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError:
                    return False, (pid,)
                if namespace not in command_line or owner_entry not in environment:
                    return False, (pid,)
        if state != "Z":
            members.append(pid)
    if leader_seen and owned.process_group_id != owned.pid:
        return False, tuple(sorted(members))
    return True, tuple(sorted(members))


def _wait_local_group_empty(
    running: _RunningStage,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        running.process.poll()
        ownership_matches, members = _local_group_ownership(running.owned)
        if not members:
            return True
        if not ownership_matches:
            return False
        time.sleep(0.1)
    _, members = _local_group_ownership(running.owned)
    return not members


def _stop_local_stage(
    running: _RunningStage,
    timeout_seconds: float,
) -> ProcessCleanupReceipt:
    owned = running.owned
    ownership_matches, members = _local_group_ownership(owned)
    if not members:
        return ProcessCleanupReceipt(
            host_name=owned.host_name,
            ownership_verified=True,
            terminated=True,
            forced=False,
        )
    if not ownership_matches:
        return ProcessCleanupReceipt(
            host_name=owned.host_name,
            ownership_verified=False,
            terminated=False,
            forced=False,
            error="local process ownership no longer matches its receipt",
        )
    with contextlib.suppress(ProcessLookupError):
        os.killpg(owned.process_group_id, signal.SIGTERM)
    if _wait_local_group_empty(running, timeout_seconds):
        return ProcessCleanupReceipt(
            host_name=owned.host_name,
            ownership_verified=True,
            terminated=True,
            forced=False,
        )
    ownership_matches, _ = _local_group_ownership(owned)
    if not ownership_matches:
        return ProcessCleanupReceipt(
            host_name=owned.host_name,
            ownership_verified=False,
            terminated=False,
            forced=False,
            error="local ownership changed before SIGKILL",
        )
    with contextlib.suppress(ProcessLookupError):
        os.killpg(owned.process_group_id, signal.SIGKILL)
    terminated = _wait_local_group_empty(running, min(timeout_seconds, 5.0))
    return ProcessCleanupReceipt(
        host_name=owned.host_name,
        ownership_verified=True,
        terminated=terminated,
        forced=True,
        error=None if terminated else "local process group survived SIGKILL",
    )


def _stop_remote_stage(
    running: _RunningStage,
    config: Pp3DiagnosticConfig,
) -> ProcessCleanupReceipt:
    receipt = asdict(running.owned)
    receipt["namespace"] = receipt.pop("ownership_namespace")
    command = (
        config.fwuff_runtime_python,
        "-c",
        REMOTE_PROCESS_STOP_PROGRAM,
        json.dumps(receipt, sort_keys=True),
        str(config.cleanup_timeout_seconds),
    )
    completed = subprocess.run(
        _ssh_argv(config.ssh_target, command),
        check=False,
        capture_output=True,
        text=True,
        timeout=config.cleanup_timeout_seconds + 15.0,
    )
    if completed.returncode != 0:
        raise Pp3DiagnosticError(
            f"remote cleanup command failed: {completed.stderr[-1000:]}"
        )
    return ProcessCleanupReceipt.model_validate_json(completed.stdout)


def stop_stage(
    running: _RunningStage,
    config: Pp3DiagnosticConfig,
) -> ProcessCleanupReceipt:
    cleanup: ProcessCleanupReceipt
    try:
        cleanup = (
            _stop_remote_stage(running, config)
            if running.owned.remote
            else _stop_local_stage(running, config.cleanup_timeout_seconds)
        )
    except BaseException as error:
        cleanup = ProcessCleanupReceipt(
            host_name=running.owned.host_name,
            ownership_verified=False,
            terminated=False,
            forced=False,
            error=f"{type(error).__name__}: {error}",
        )
    finalization_errors: list[str] = []
    try:
        running.process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        if running.owned.remote:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(running.process.pid, signal.SIGTERM)
            try:
                running.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(running.process.pid, signal.SIGKILL)
                try:
                    running.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    finalization_errors.append("SSH transport could not be reaped")
        else:
            finalization_errors.append("local process could not be reaped")
    if running.pump_thread is not None:
        running.pump_thread.join(timeout=5.0)
        if running.pump_thread.is_alive():
            finalization_errors.append("remote output pump did not finish")
    running.log_file.close()
    if finalization_errors:
        prior_error = cleanup.error
        cleanup = cleanup.model_copy(
            update={
                "terminated": False,
                "error": "; ".join(
                    item for item in (prior_error, *finalization_errors) if item
                ),
            }
        )
    return cleanup


def all_stages_alive(running_stages: Sequence[_RunningStage]) -> bool:
    return all(stage.process.poll() is None for stage in running_stages)


def wait_for_all_stages(
    specs: Sequence[SglangKtProcessLaunchSpec],
    running_stages: Sequence[_RunningStage],
    timeout_seconds: float,
) -> tuple[JsonObject, ...]:
    """Require every logical rank's health endpoint to return HTTP 200."""

    deadline = time.monotonic() + timeout_seconds
    pending = {spec.pipeline_rank: spec for spec in specs}
    observations: dict[int, JsonObject] = {}
    while pending and time.monotonic() < deadline:
        if not all_stages_alive(running_stages):
            return_codes = {
                stage.owned.rank: stage.process.poll() for stage in running_stages
            }
            raise Pp3DiagnosticError(
                f"a pipeline stage exited before readiness: {return_codes}"
            )
        for rank, spec in tuple(pending.items()):
            try:
                with Glm47NativeServingClient(
                    f"http://{spec.service_endpoint}", timeout_seconds=2.0
                ) as client:
                    observation = client.health_generate()
            except (Glm47ServingClientError, httpx.HTTPError, OSError):
                continue
            observations[rank] = cast(JsonObject, observation.model_dump(mode="json"))
            pending.pop(rank)
        if pending:
            time.sleep(0.25)
    if pending:
        raise Pp3DiagnosticError(
            f"pipeline readiness timed out for ranks {tuple(sorted(pending))}"
        )
    return tuple(observations[rank] for rank in range(len(specs)))


def _hca_ports(devices: Sequence[str]) -> tuple[HcaPort, ...]:
    ports: list[HcaPort] = []
    for index, token in enumerate(devices, start=1):
        device, separator, raw_port = token.rpartition(":")
        if not separator:
            raise Pp3DiagnosticError(f"invalid HCA device selection {token}")
        ports.append(
            HcaPort(
                device=device,
                port=int(raw_port),
                gid=f"fe80::{index}",
                rail_id=f"rail-{index}",
            )
        )
    return tuple(ports)


def capture_local_hca_counters(
    hca_devices: Sequence[str],
) -> HcaCounterSnapshot:
    request = HostHcaCounterRequest(
        schema_version=1,
        host_name="dwagon",
        ports=_hca_ports(hca_devices),
    )
    return collect_hca_counter_snapshot(request, LinuxHostProbe())


def capture_remote_hca_counters(
    config: Pp3DiagnosticConfig,
) -> HcaCounterSnapshot:
    request = HostHcaCounterRequest(
        schema_version=1,
        host_name="fwuff",
        ports=_hca_ports(config.hca_devices),
    )
    script = (
        Path(config.remote_source_directory) / "scripts" / "two_host_mlx_nccl_poc.py"
    )
    completed = subprocess.run(
        _ssh_argv(
            config.ssh_target,
            (config.fwuff_runtime_python, str(script), "host-hca-counters"),
        ),
        check=False,
        input=request.model_dump_json(),
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise Pp3DiagnosticError(
            f"fwuff HCA counter capture failed: {completed.stderr[-1000:]}"
        )
    return HcaCounterSnapshot.model_validate_json(completed.stdout)


def capture_cluster_hca_counters(
    config: Pp3DiagnosticConfig,
) -> dict[str, HcaCounterSnapshot]:
    return {
        "dwagon": capture_local_hca_counters(config.hca_devices),
        "fwuff": capture_remote_hca_counters(config),
    }


def calculate_hca_deltas(
    before: Mapping[str, HcaCounterSnapshot],
    after: Mapping[str, HcaCounterSnapshot],
) -> JsonObject:
    result: JsonObject = {}
    if set(before) != set(after):
        raise Pp3DiagnosticError("HCA counter snapshots cover different hosts")
    for host_name in sorted(before):
        before_ports = {port.rail_id: port for port in before[host_name].ports}
        after_ports = {port.rail_id: port for port in after[host_name].ports}
        if set(before_ports) != set(after_ports):
            raise Pp3DiagnosticError(f"HCA rail selection changed on {host_name}")
        host_result: JsonObject = {}
        for rail_id in sorted(before_ports):
            prior = before_ports[rail_id]
            current = after_ports[rail_id]
            if (prior.device, prior.port) != (current.device, current.port):
                raise Pp3DiagnosticError(
                    f"HCA port identity changed on {host_name}/{rail_id}"
                )
            deltas = {
                name: current.counters[name] - prior.counters[name]
                for name in sorted(prior.counters)
            }
            if any(delta < 0 for delta in deltas.values()):
                raise Pp3DiagnosticError(
                    f"HCA counter regressed on {host_name}/{rail_id}"
                )
            rail_result: JsonObject = {
                "device": current.device,
                "port": current.port,
                "counter_deltas": deltas,
                "received_payload_bytes": deltas["port_rcv_data"] * 4,
                "transmitted_payload_bytes": deltas["port_xmit_data"] * 4,
            }
            host_result[rail_id] = rail_result
        result[host_name] = host_result
    return result


def summarize_workload(workload: SglangKtServingWorkloadEvidence) -> JsonObject:
    request = workload.request
    samples = workload.samples
    decode_rates = [
        sample.client_observed_decode_tokens_per_second for sample in samples
    ]
    ttfts = [sample.client_observed_ttft_seconds for sample in samples]
    total_rates = [
        sample.completion_tokens / sample.total_client_seconds for sample in samples
    ]
    if not decode_rates or any(not math.isfinite(value) for value in decode_rates):
        raise Pp3DiagnosticError("workload contains no finite decode samples")
    return {
        "kind": request.kind,
        "input_tokens": request.input_token_count,
        "output_tokens": request.max_new_tokens,
        "sample_count": len(samples),
        "median_client_decode_tokens_per_second": statistics.median(decode_rates),
        "median_client_ttft_seconds": statistics.median(ttfts),
        "median_end_to_end_output_tokens_per_second": statistics.median(total_rates),
    }


def _process_receipt(running: _RunningStage) -> JsonObject:
    owned = asdict(running.owned)
    owned["owner_token"] = hashlib.sha256(
        running.owned.owner_token.encode()
    ).hexdigest()
    return cast(JsonObject, owned)


def _log_receipts(config: Pp3DiagnosticConfig) -> list[JsonValue]:
    receipts: list[JsonValue] = []
    for rank in range(3):
        path = config.result_directory / f"rank-{rank}.log"
        if not path.is_file():
            continue
        status = path.stat()
        if status.st_size > _LOG_MAXIMUM_BYTES:
            raise Pp3DiagnosticError(f"rank {rank} log exceeds diagnostic size bound")
        receipts.append(
            {
                "rank": rank,
                "path": str(path),
                "size_bytes": status.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return receipts


def _write_receipt(config: Pp3DiagnosticConfig, payload: JsonObject) -> None:
    path = config.result_directory / "pp3-diagnostic-result.json"
    temporary = config.result_directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _configuration_receipt(config: Pp3DiagnosticConfig) -> JsonObject:
    return {
        "dwagon_stage_placement": config.dwagon_stage_placement,
    }


def run_diagnostic(config: Pp3DiagnosticConfig) -> JsonObject:
    """Launch, measure, clean up, and publish one PP3 diagnostic receipt."""

    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    specs = build_pp3_process_specs(config)
    owner_token = uuid.uuid4().hex
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    running: list[_RunningStage] = []
    readiness: tuple[JsonObject, ...] = ()
    server_info: JsonObject | None = None
    sanity: JsonObject | None = None
    workloads: list[JsonValue] = []
    summaries: list[JsonValue] = []
    hca_before: dict[str, HcaCounterSnapshot] = {}
    hca_after: dict[str, HcaCounterSnapshot] = {}
    hca_deltas: JsonObject | None = None
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []
    try:
        for spec in specs:
            running.append(start_stage(spec, config, owner_token))
        readiness = wait_for_all_stages(
            specs,
            running,
            config.readiness_timeout_seconds,
        )
        rank_zero = specs[0]
        with Glm47NativeServingClient(
            f"http://{rank_zero.service_endpoint}",
            timeout_seconds=config.request_timeout_seconds,
        ) as client:
            server_info = cast(JsonObject, client.server_info().model_dump(mode="json"))
            sanity_evidence = run_glm47_serving_sanity(
                client,
                config.dwagon_model_path,
            )
            sanity = cast(JsonObject, sanity_evidence.model_dump(mode="json"))
            if not all_stages_alive(running):
                raise Pp3DiagnosticError("a stage exited during semantic sanity")
            hca_before = capture_cluster_hca_counters(config)
            for kind in ("prefill", "decode"):
                workload = run_glm47_serving_workload(
                    client,
                    prepare_glm47_serving_workload(kind),
                    warmup_count=config.warmup_count,
                    sample_count=config.sample_count,
                )
                if not all_stages_alive(running):
                    raise Pp3DiagnosticError(
                        f"a stage exited during the {kind} workload"
                    )
                workloads.append(cast(JsonObject, workload.model_dump(mode="json")))
                summaries.append(summarize_workload(workload))
            hca_after = capture_cluster_hca_counters(config)
            hca_deltas = calculate_hca_deltas(hca_before, hca_after)
    except BaseException as error:
        failure = error
        if hca_before and not hca_after:
            with contextlib.suppress(BaseException):
                hca_after = capture_cluster_hca_counters(config)
                hca_deltas = calculate_hca_deltas(hca_before, hca_after)
    finally:
        for stage in reversed(running):
            receipt = stop_stage(stage, config)
            cleanup.append(
                {
                    "rank": stage.owned.rank,
                    **receipt.model_dump(mode="json"),
                }
            )
        cleanup.sort(key=lambda item: cast(int, cast(dict[str, object], item)["rank"]))

    cleanup_complete = bool(cleanup) and all(
        cast(dict[str, object], item)["ownership_verified"] is True
        and cast(dict[str, object], item)["terminated"] is True
        for item in cleanup
    )
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm47_flash_pp3_engineering_diagnostic",
        "status": "passed" if failure is None and cleanup_complete else "failed",
        "performance_comparable": False,
        "profiler": "none",
        "instrumentation": "nccl_info_logging",
        "run_id": config.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "configuration": _configuration_receipt(config),
        "plan": cast(JsonObject, specs[0].plan.model_dump(mode="json")),
        "process_specs": [
            cast(JsonObject, spec.model_dump(mode="json")) for spec in specs
        ],
        "processes": [_process_receipt(stage) for stage in running],
        "readiness": list(readiness),
        "server_info": server_info,
        "sanity": sanity,
        "workloads": workloads,
        "benchmark_summary": summaries,
        "hca_counters": {
            "scope": "token_workloads_after_semantic_sanity",
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
        },
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "logs": _log_receipts(config),
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    _write_receipt(config, payload)
    if failure is not None:
        raise Pp3DiagnosticError(
            f"PP3 diagnostic failed; evidence is in {config.result_directory}: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    if not cleanup_complete:
        raise Pp3DiagnosticError(
            f"PP3 cleanup was incomplete; evidence is in {config.result_directory}"
        )
    return payload


def _parse_hca_devices(raw: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("at least one HCA device is required")
    try:
        _hca_ports(devices)
    except (Pp3DiagnosticError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return devices


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--dwagon-runtime-python", required=True)
    parser.add_argument("--fwuff-runtime-python", required=True)
    parser.add_argument("--dwagon-model-path", default=DEFAULT_DWAGON_MODEL_PATH)
    parser.add_argument("--fwuff-model-path", default=DEFAULT_FWUFF_MODEL_PATH)
    parser.add_argument("--local-source-directory", default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--remote-source-directory", default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--ssh-target", default="fwuff")
    parser.add_argument("--dwagon-ip", default="192.168.40.24")
    parser.add_argument("--fwuff-ip", default="192.168.40.248")
    parser.add_argument(
        "--dwagon-socket-interface",
        default=DEFAULT_DWAGON_SOCKET_INTERFACE,
    )
    parser.add_argument(
        "--fwuff-socket-interface",
        default=DEFAULT_FWUFF_SOCKET_INTERFACE,
    )
    parser.add_argument("--distributed-port", type=_positive_int, default=62400)
    parser.add_argument("--rank-zero-port", type=_positive_int, default=62410)
    parser.add_argument("--rank-one-port", type=_positive_int, default=62411)
    parser.add_argument("--rank-two-port", type=_positive_int, default=62412)
    parser.add_argument(
        "--hca-devices",
        type=_parse_hca_devices,
        default=DEFAULT_HCA_DEVICES,
    )
    parser.add_argument(
        "--dwagon-stage-placement",
        choices=(
            DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER,
            DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL,
        ),
        default=DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER,
        help=(
            "map dwagon ranks in pipeline order, or put rank 1 on the "
            "NUMA 0/GPU 0 placement nearest the cross-host boundary"
        ),
    )
    parser.add_argument("--resident-gpu-experts", type=_positive_int, default=4)
    parser.add_argument(
        "--readiness-timeout-seconds", type=_positive_float, default=1800.0
    )
    parser.add_argument(
        "--request-timeout-seconds", type=_positive_float, default=900.0
    )
    parser.add_argument("--cleanup-timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--samples", type=_positive_int, default=3)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> Pp3DiagnosticConfig:
    result_directory = cast(Path, arguments.result_directory).resolve()
    ports = (
        cast(int, arguments.distributed_port),
        cast(int, arguments.rank_zero_port),
        cast(int, arguments.rank_one_port),
        cast(int, arguments.rank_two_port),
    )
    if len(set(ports)) != len(ports) or any(port > 65535 for port in ports):
        raise Pp3DiagnosticError(
            "distributed and service ports must be unique TCP ports"
        )
    run_id = cast(str, arguments.run_id)
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise Pp3DiagnosticError("run_id must contain only safe identifier characters")
    resident_gpu_experts = cast(int, arguments.resident_gpu_experts)
    if resident_gpu_experts >= 64:
        raise Pp3DiagnosticError("resident_gpu_experts must be between 1 and 63")
    return Pp3DiagnosticConfig(
        run_id=run_id,
        result_directory=result_directory,
        dwagon_runtime_python=cast(str, arguments.dwagon_runtime_python),
        fwuff_runtime_python=cast(str, arguments.fwuff_runtime_python),
        dwagon_model_path=cast(str, arguments.dwagon_model_path),
        fwuff_model_path=cast(str, arguments.fwuff_model_path),
        local_source_directory=cast(str, arguments.local_source_directory),
        remote_source_directory=cast(str, arguments.remote_source_directory),
        ssh_target=cast(str, arguments.ssh_target),
        dwagon_ip=cast(str, arguments.dwagon_ip),
        fwuff_ip=cast(str, arguments.fwuff_ip),
        dwagon_socket_interface=cast(str, arguments.dwagon_socket_interface),
        fwuff_socket_interface=cast(str, arguments.fwuff_socket_interface),
        distributed_port=ports[0],
        stage_ports=cast(tuple[int, int, int], ports[1:]),
        hca_devices=cast(tuple[str, ...], arguments.hca_devices),
        dwagon_stage_placement=cast(
            DwagonStagePlacement,
            arguments.dwagon_stage_placement,
        ),
        resident_gpu_experts=resident_gpu_experts,
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
    except (Pp3DiagnosticError, OSError, ValueError) as error:
        print(f"PP3 diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload["benchmark_summary"], indent=2, sort_keys=True))
    print(config.result_directory / "pp3-diagnostic-result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
