#!/usr/bin/env python3
"""Fail-closed local interconnect qualification for DSV4 TP2 on two RTX 3090s.

The parent process inventories the GPUs, proves that they are idle, snapshots
the per-link NVLink payload counters, and launches two short CUDA subprocesses:

* a bidirectional CUDA peer-access/copy correctness probe; and
* a two-rank NCCL all-reduce benchmark, including CUDA-graph capture/replay.

The resulting receipt is only admitted when CUDA peer access works in both
directions, NCCL logs select P2P on both rank edges without SHM/NET fallback,
all active NVLinks carry attributed collective traffic, and the representative
small-message latency/bandwidth floors pass.  No model weights are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Sequence

SCHEMA: Final = "dsv4-tp2-interconnect-qualification-v1"
EXPECTED_GPU_NAME: Final = "NVIDIA GeForce RTX 3090"
EXPECTED_COMPUTE_CAPABILITY: Final = "8.6"
EXPECTED_NVLINK_COUNT: Final = 4
DEFAULT_MESSAGE_SIZES: Final = (
    8_192,
    16_384,
    24_576,
    40_960,
    49_152,
    65_536,
    1_048_576,
)
REPRESENTATIVE_MESSAGE_BYTES: Final = 49_152
CONTROL_MESSAGE_BYTES: Final = 1_048_576
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NVLINK_COUNTER = re.compile(
    r"Link\s+(?P<link>\d+):\s+Data\s+(?P<direction>Tx|Rx):\s+"
    r"(?P<value>\d+)\s+KiB"
)
NVLINK_STATUS = re.compile(r"Link\s+(?P<link>\d+):\s+(?P<rate>[0-9.]+)\s+GB/s")
NCCL_ROUTE = re.compile(
    r"(?P<source>\d+)\[[^]]+\]\s*->\s*"
    r"(?P<destination>\d+)\[[^]]+\]\s+via\s+"
    r"(?P<transport>[^\r\n]+)"
)

GateStatus = Literal["pass", "block", "warn"]


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_mib: int
    memory_free_mib: int
    compute_capability: str


@dataclass(frozen=True)
class Gate:
    name: str
    status: GateStatus
    detail: str


@dataclass(frozen=True)
class Configuration:
    devices: tuple[int, int]
    message_sizes_bytes: tuple[int, ...]
    warmup_iterations: int
    iterations: int
    graph_iterations: int
    maximum_48k_latency_us: float
    maximum_graph_48k_latency_us: float
    minimum_1m_bus_bandwidth_gb_s: float
    minimum_counter_delta_kib: int


class QualificationError(RuntimeError):
    """A qualification prerequisite or subprocess failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(
    arguments: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise QualificationError(
            f"failed to execute {' '.join(arguments)}: {error}"
        ) from error


def require_command(
    arguments: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 30,
) -> str:
    result = run_command(arguments, environment=environment, timeout=timeout)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise QualificationError(
            f"{' '.join(arguments)} exited {result.returncode}: {detail}"
        )
    return result.stdout


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE.sub("", value)


def parse_devices(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(not part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("--devices must contain two GPU indices")
    devices = (int(parts[0]), int(parts[1]))
    if devices[0] == devices[1]:
        raise argparse.ArgumentTypeError("--devices must contain distinct GPU indices")
    return devices


def parse_message_sizes(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdecimal() or int(part) <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "--message-sizes must be comma-separated positive byte counts"
        )
    sizes = tuple(int(part) for part in parts)
    for required_size in (REPRESENTATIVE_MESSAGE_BYTES, CONTROL_MESSAGE_BYTES):
        if required_size not in sizes:
            raise argparse.ArgumentTypeError(
                f"--message-sizes must include {required_size} bytes"
            )
    return sizes


def parse_gpu_inventory(value: str) -> tuple[Gpu, ...]:
    gpus: list[Gpu] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise QualificationError(f"unexpected nvidia-smi GPU row: {line!r}")
        try:
            gpus.append(
                Gpu(
                    index=int(fields[0]),
                    name=fields[1],
                    uuid=fields[2],
                    pci_bus_id=fields[3].lower(),
                    memory_total_mib=int(fields[4]),
                    memory_free_mib=int(fields[5]),
                    compute_capability=fields[6],
                )
            )
        except ValueError as error:
            raise QualificationError(f"invalid nvidia-smi GPU row: {line!r}") from error
    return tuple(gpus)


def parse_compute_applications(value: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in value.splitlines()
        if line.strip() and "No running processes found" not in line
    )


def parse_topology_relation(value: str, source: int, destination: int) -> str | None:
    rows = [strip_ansi(line).split() for line in value.splitlines()]
    header = next(
        (
            row
            for row in rows
            if len(row) > 1 and row[0] == "GPU0" and row[1].startswith("GPU")
        ),
        None,
    )
    if header is None:
        return None
    destination_label = f"GPU{destination}"
    try:
        destination_column = header.index(destination_label)
    except ValueError:
        return None
    source_label = f"GPU{source}"
    source_row = next(
        (row for row in rows if row is not header and row and row[0] == source_label),
        None,
    )
    if source_row is None or destination_column + 1 >= len(source_row):
        return None
    # Data rows include their row label before the columns represented by the
    # header, hence the extra one here.
    return source_row[destination_column + 1]


def parse_nvlink_status(value: str) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = {}
    current_gpu: int | None = None
    for line in value.splitlines():
        gpu_match = re.match(r"GPU\s+(\d+):", line)
        if gpu_match is not None:
            current_gpu = int(gpu_match.group(1))
            result.setdefault(current_gpu, {})
            continue
        link_match = NVLINK_STATUS.search(line)
        if current_gpu is not None and link_match is not None:
            result[current_gpu][int(link_match.group("link"))] = float(
                link_match.group("rate")
            )
    return result


def parse_nvlink_counters(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    current_gpu: int | None = None
    for line in value.splitlines():
        gpu_match = re.match(r"GPU\s+(\d+):", line)
        if gpu_match is not None:
            current_gpu = int(gpu_match.group(1))
            continue
        counter_match = NVLINK_COUNTER.search(line)
        if current_gpu is None or counter_match is None:
            continue
        key = (
            f"gpu{current_gpu}.link{int(counter_match.group('link'))}."
            f"{counter_match.group('direction').lower()}_kib"
        )
        result[key] = int(counter_match.group("value"))
    return result


def counter_deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    if before.keys() != after.keys():
        raise QualificationError("NVLink counter keys changed during qualification")
    deltas: dict[str, int] = {}
    for key, before_value in before.items():
        after_value = after[key]
        if after_value < before_value:
            raise QualificationError(f"NVLink counter decreased for {key}")
        deltas[key] = after_value - before_value
    return deltas


def parse_nccl_transports(log_text: str) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    for match in NCCL_ROUTE.finditer(log_text):
        routes.append(
            {
                "source": int(match.group("source")),
                "destination": int(match.group("destination")),
                "transport": match.group("transport").strip(),
            }
        )
    relevant = [
        route for route in routes if {route["source"], route["destination"]} == {0, 1}
    ]
    p2p_directions = {
        (route["source"], route["destination"])
        for route in relevant
        if str(route["transport"]).upper().startswith("P2P/")
    }
    fallback_routes = [
        route
        for route in relevant
        if str(route["transport"]).upper().startswith(("SHM", "NET/"))
    ]
    return {
        "routes": relevant,
        "p2p_directions": [list(direction) for direction in sorted(p2p_directions)],
        "fallback_routes": fallback_routes,
        "init_complete_count": log_text.count("Init COMPLETE"),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_nccl_environment(
    *, devices: tuple[int, int], log_pattern: Path
) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("NCCL_", "TORCH_NCCL_"))
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": f"{devices[0]},{devices[1]}",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P,SHM,NET,ENV",
            "NCCL_DEBUG_FILE": str(log_pattern),
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_GRAPH_MIXING_SUPPORT": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        }
    )
    return environment


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(
            f"cannot read worker result {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise QualificationError(f"worker result is not an object: {path}")
    return value


def run_peer_probe(
    *, python: Path, script: Path, devices: tuple[int, int], result_path: Path
) -> dict[str, Any]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("NCCL_", "TORCH_NCCL_"))
    }
    environment["CUDA_VISIBLE_DEVICES"] = f"{devices[0]},{devices[1]}"
    result = run_command(
        [str(python), str(script), "--peer-worker", str(result_path)],
        environment=environment,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise QualificationError(f"CUDA peer probe failed: {detail}")
    return _load_json(result_path)


def run_nccl_benchmark(
    *,
    python: Path,
    script: Path,
    devices: tuple[int, int],
    configuration: Configuration,
    artifact_directory: Path,
) -> tuple[list[dict[str, Any]], subprocess.CompletedProcess[str], list[Path]]:
    result_directory = artifact_directory / "workers"
    result_directory.mkdir()
    log_pattern = artifact_directory / "nccl-%h-%p.log"
    environment = clean_nccl_environment(devices=devices, log_pattern=log_pattern)
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(script),
        "--worker",
        "--worker-result-directory",
        str(result_directory),
        "--message-sizes",
        ",".join(str(size) for size in configuration.message_sizes_bytes),
        "--warmup-iterations",
        str(configuration.warmup_iterations),
        "--iterations",
        str(configuration.iterations),
        "--graph-iterations",
        str(configuration.graph_iterations),
    ]
    result = run_command(command, environment=environment, timeout=180)
    worker_paths = sorted(result_directory.glob("rank-*.json"))
    workers = [_load_json(path) for path in worker_paths]
    log_paths = sorted(artifact_directory.glob("nccl-*.log"))
    return workers, result, log_paths


def query_compute_apps() -> tuple[str, tuple[str, ...]]:
    raw = require_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return raw, parse_compute_applications(raw)


def metric_for_size(workers: Sequence[dict[str, Any]], size: int) -> dict[str, Any]:
    if not workers:
        raise QualificationError("NCCL benchmark produced no worker receipts")
    metrics = workers[0].get("all_reduce")
    if not isinstance(metrics, list):
        raise QualificationError("worker receipt has no all_reduce metrics")
    for metric in metrics:
        if isinstance(metric, dict) and metric.get("bytes") == size:
            return metric
    raise QualificationError(f"worker receipt has no {size}-byte metric")


def gate_summary(gates: Sequence[Gate]) -> tuple[bool, list[dict[str, Any]]]:
    return (
        not any(gate.status == "block" for gate in gates),
        [asdict(gate) for gate in gates],
    )


def validate_receipt(
    *, receipt_path: Path, devices: tuple[int, int], maximum_age_seconds: int
) -> None:
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != SCHEMA or receipt.get("qualified") is not True:
        raise QualificationError("receipt is not a qualified DSV4 TP2 receipt")
    created_text = receipt.get("created_at_utc")
    if not isinstance(created_text, str):
        raise QualificationError("receipt has no creation timestamp")
    try:
        created = datetime.fromisoformat(created_text)
    except ValueError as error:
        raise QualificationError("receipt creation timestamp is invalid") from error
    if created.tzinfo is None:
        raise QualificationError("receipt creation timestamp is not timezone-aware")
    age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
    if age_seconds < -5 or age_seconds > maximum_age_seconds:
        raise QualificationError(
            f"receipt age {age_seconds:.1f}s is outside 0..{maximum_age_seconds}s"
        )
    script_record = receipt.get("script")
    if not isinstance(script_record, dict) or script_record.get(
        "sha256"
    ) != file_sha256(Path(__file__).resolve()):
        raise QualificationError("receipt was produced by different qualifier code")
    configuration = receipt.get("configuration")
    if (
        not isinstance(configuration, dict)
        or tuple(configuration.get("devices", ())) != devices
    ):
        raise QualificationError("receipt GPU selection does not match the launch")
    required_gates = {
        "gpu-inventory",
        "gpus-idle-before-cuda",
        "nvlink-topology",
        "nvidia-smi-nvlink-p2p-capabilities",
        "active-nvlinks",
        "cuda-peer-access-and-copy",
        "gpus-idle-before-nccl",
        "nccl-benchmark-completed",
        "gpus-idle-after-nccl",
        "nvlink-collective-counter-delta",
        "nccl-selected-transport",
        "nccl-all-reduce-correctness",
        "nccl-48k-latency",
        "nccl-cuda-graph-48k",
        "nccl-1m-bandwidth",
    }
    gates = receipt.get("gates")
    if not isinstance(gates, list):
        raise QualificationError("receipt has no gate evidence")
    passed_gates = {
        gate.get("name")
        for gate in gates
        if isinstance(gate, dict) and gate.get("status") == "pass"
    }
    missing_gates = sorted(required_gates - passed_gates)
    if missing_gates:
        raise QualificationError(
            "receipt is missing passing gates: " + ", ".join(missing_gates)
        )
    nccl = receipt.get("nccl")
    logs = nccl.get("logs") if isinstance(nccl, dict) else None
    if not isinstance(logs, list) or len(logs) < 2:
        raise QualificationError("receipt has fewer than two NCCL log artifacts")
    for log in logs:
        if not isinstance(log, dict):
            raise QualificationError("receipt contains an invalid NCCL log record")
        path_text = log.get("path")
        digest = log.get("sha256")
        if not isinstance(path_text, str) or not isinstance(digest, str):
            raise QualificationError("receipt contains an incomplete NCCL log record")
        path = Path(path_text)
        if not path.is_file() or file_sha256(path) != digest:
            raise QualificationError(f"NCCL log artifact is missing or changed: {path}")
    inventory_raw = require_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    current_gpus = {gpu.index: gpu for gpu in parse_gpu_inventory(inventory_raw)}
    receipt_inventory = receipt.get("inventory")
    if not isinstance(receipt_inventory, list):
        raise QualificationError("receipt has no GPU inventory")
    receipt_uuids = {
        value.get("index"): value.get("uuid")
        for value in receipt_inventory
        if isinstance(value, dict)
    }
    for device in devices:
        gpu = current_gpus.get(device)
        if gpu is None or receipt_uuids.get(device) != gpu.uuid:
            raise QualificationError(
                f"GPU {device} identity changed after qualification"
            )
    _, applications = query_compute_apps()
    if applications:
        raise QualificationError(
            "GPU became busy after qualification: " + "; ".join(applications)
        )


def qualify(
    *, output: Path, python: Path, configuration: Configuration
) -> tuple[int, dict[str, Any]]:
    script = Path(__file__).resolve()
    artifact_directory = output.with_name(f"{output.name}.artifacts")
    if output.exists() or artifact_directory.exists():
        raise QualificationError(
            f"refusing to overwrite qualification evidence at {output}"
        )
    artifact_directory.mkdir(parents=True)
    gates: list[Gate] = []
    inventory_raw = require_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory = parse_gpu_inventory(inventory_raw)
    gpu_by_index = {gpu.index: gpu for gpu in inventory}
    selected = [gpu_by_index.get(index) for index in configuration.devices]
    if any(gpu is None for gpu in selected):
        gates.append(Gate("gpu-inventory", "block", "selected GPU is absent"))
    else:
        selected_gpus = [gpu for gpu in selected if gpu is not None]
        correct_identity = all(
            gpu.name == EXPECTED_GPU_NAME
            and gpu.compute_capability == EXPECTED_COMPUTE_CAPABILITY
            for gpu in selected_gpus
        )
        gates.append(
            Gate(
                "gpu-inventory",
                "pass" if correct_identity else "block",
                ", ".join(
                    f"GPU {gpu.index}: {gpu.name}, SM{gpu.compute_capability}"
                    for gpu in selected_gpus
                ),
            )
        )

    apps_before_raw, apps_before = query_compute_apps()
    gates.append(
        Gate(
            "gpus-idle-before-cuda",
            "pass" if not apps_before else "block",
            "no compute applications" if not apps_before else "; ".join(apps_before),
        )
    )
    topology = require_command(["nvidia-smi", "topo", "-m"])
    p2p_matrices = {
        capability: require_command(["nvidia-smi", "topo", "-p2p", capability])
        for capability in ("n", "r", "w", "p")
    }
    topology_relation = parse_topology_relation(
        topology, configuration.devices[0], configuration.devices[1]
    )
    p2p_relations = {
        capability: parse_topology_relation(
            matrix, configuration.devices[0], configuration.devices[1]
        )
        for capability, matrix in p2p_matrices.items()
    }
    gates.append(
        Gate(
            "nvlink-topology",
            "pass" if topology_relation == "NV4" else "block",
            f"topology relation is {topology_relation or 'unparseable'}",
        )
    )
    gates.append(
        Gate(
            "nvidia-smi-nvlink-p2p-capabilities",
            (
                "pass"
                if all(p2p_relations[capability] == "OK" for capability in "nrw")
                else "block"
            ),
            (
                "NVLink/read/write relations are "
                f"{[p2p_relations[capability] for capability in 'nrw']}"
            ),
        )
    )
    gates.append(
        Gate(
            "nvidia-smi-pcie-only-p2p-matrix",
            "pass" if p2p_relations["p"] == "OK" else "warn",
            (
                f"PCIe-only relation is {p2p_relations['p'] or 'unparseable'}; "
                "this does not negate the separate NVLink capability"
            ),
        )
    )
    nvlink_status_raw = require_command(["nvidia-smi", "nvlink", "-s"])
    nvlink_status = parse_nvlink_status(nvlink_status_raw)
    status_ok = all(
        len(nvlink_status.get(gpu, {})) == EXPECTED_NVLINK_COUNT
        and all(rate > 0 for rate in nvlink_status[gpu].values())
        for gpu in configuration.devices
    )
    gates.append(
        Gate(
            "active-nvlinks",
            "pass" if status_ok else "block",
            f"active link rates: {nvlink_status}",
        )
    )
    preflight_ok, _ = gate_summary(gates)
    if not preflight_ok:
        qualified, serialized_gates = gate_summary(gates)
        report = {
            "schema": SCHEMA,
            "created_at_utc": utc_now(),
            "qualified": qualified,
            "configuration": asdict(configuration),
            "gates": serialized_gates,
            "inventory": [asdict(gpu) for gpu in inventory],
            "compute_applications_before": apps_before_raw,
            "topology": topology,
            "p2p_matrices": p2p_matrices,
            "nvlink_status": nvlink_status_raw,
            "artifact_directory": str(artifact_directory),
        }
        write_json_atomic(output, report)
        return 1, report

    peer_result_path = artifact_directory / "peer-copy.json"
    peer = run_peer_probe(
        python=python,
        script=script,
        devices=configuration.devices,
        result_path=peer_result_path,
    )
    peer_ok = bool(peer.get("qualified"))
    gates.append(
        Gate(
            "cuda-peer-access-and-copy",
            "pass" if peer_ok else "block",
            str(peer.get("detail", "peer worker returned no detail")),
        )
    )
    _, apps_after_peer = query_compute_apps()
    gates.append(
        Gate(
            "gpus-idle-before-nccl",
            "pass" if not apps_after_peer else "block",
            (
                "no compute applications"
                if not apps_after_peer
                else "; ".join(apps_after_peer)
            ),
        )
    )
    if not peer_ok or apps_after_peer:
        qualified, serialized_gates = gate_summary(gates)
        report = {
            "schema": SCHEMA,
            "created_at_utc": utc_now(),
            "qualified": qualified,
            "configuration": asdict(configuration),
            "gates": serialized_gates,
            "inventory": [asdict(gpu) for gpu in inventory],
            "peer": peer,
            "topology": topology,
            "p2p_matrices": p2p_matrices,
            "nvlink_status": nvlink_status_raw,
            "artifact_directory": str(artifact_directory),
        }
        write_json_atomic(output, report)
        return 1, report

    counters_before_raw = require_command(["nvidia-smi", "nvlink", "-gt", "d"])
    counters_before = parse_nvlink_counters(counters_before_raw)
    workers, benchmark_process, nccl_log_paths = run_nccl_benchmark(
        python=python,
        script=script,
        devices=configuration.devices,
        configuration=configuration,
        artifact_directory=artifact_directory,
    )
    counters_after_raw = require_command(["nvidia-smi", "nvlink", "-gt", "d"])
    counters_after = parse_nvlink_counters(counters_after_raw)
    deltas = counter_deltas(counters_before, counters_after)
    _, apps_after = query_compute_apps()
    benchmark_ok = benchmark_process.returncode == 0 and len(workers) == 2
    gates.append(
        Gate(
            "nccl-benchmark-completed",
            "pass" if benchmark_ok else "block",
            (
                "two worker receipts"
                if benchmark_ok
                else (
                    benchmark_process.stderr.strip()
                    or benchmark_process.stdout.strip()
                    or f"return code {benchmark_process.returncode}"
                )
            ),
        )
    )
    gates.append(
        Gate(
            "gpus-idle-after-nccl",
            "pass" if not apps_after else "block",
            "no compute applications" if not apps_after else "; ".join(apps_after),
        )
    )

    expected_counter_keys = {
        f"gpu{gpu}.link{link}.{direction}_kib"
        for gpu in configuration.devices
        for link in range(EXPECTED_NVLINK_COUNT)
        for direction in ("tx", "rx")
    }
    all_links_moved = expected_counter_keys.issubset(deltas) and all(
        deltas[key] > 0 for key in expected_counter_keys
    )
    total_delta_kib = sum(deltas.get(key, 0) for key in expected_counter_keys)
    counters_ok = (
        all_links_moved and total_delta_kib >= configuration.minimum_counter_delta_kib
    )
    gates.append(
        Gate(
            "nvlink-collective-counter-delta",
            "pass" if counters_ok else "block",
            (
                f"all_links_moved={all_links_moved}; "
                f"aggregate delta {total_delta_kib} KiB"
            ),
        )
    )

    combined_log = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in nccl_log_paths
    )
    transport = parse_nccl_transports(combined_log)
    directions = {tuple(value) for value in transport["p2p_directions"]}
    transport_ok = (
        directions == {(0, 1), (1, 0)}
        and not transport["fallback_routes"]
        and transport["init_complete_count"] >= 2
    )
    gates.append(
        Gate(
            "nccl-selected-transport",
            "pass" if transport_ok else "block",
            (
                f"P2P directions={sorted(directions)}, "
                f"fallbacks={len(transport['fallback_routes'])}, "
                f"init_complete={transport['init_complete_count']}"
            ),
        )
    )

    if benchmark_ok:
        metric_48k = metric_for_size(workers, REPRESENTATIVE_MESSAGE_BYTES)
        metric_1m = metric_for_size(workers, CONTROL_MESSAGE_BYTES)
        graph_metric = workers[0].get("cuda_graph")
        graph_ok = isinstance(graph_metric, dict) and bool(
            graph_metric.get("capture_and_replay_ok")
        )
        latency_ok = (
            float(metric_48k["latency_us"]) <= configuration.maximum_48k_latency_us
        )
        graph_latency_ok = graph_ok and (
            float(graph_metric["latency_us"])
            <= configuration.maximum_graph_48k_latency_us
        )
        bandwidth_ok = (
            float(metric_1m["bus_bandwidth_gb_s"])
            >= configuration.minimum_1m_bus_bandwidth_gb_s
        )
        correctness_ok = all(bool(worker.get("correctness_ok")) for worker in workers)
        gates.extend(
            (
                Gate(
                    "nccl-all-reduce-correctness",
                    "pass" if correctness_ok else "block",
                    "all ranks observed the exact sum",
                ),
                Gate(
                    "nccl-48k-latency",
                    "pass" if latency_ok else "block",
                    (
                        f"{float(metric_48k['latency_us']):.3f} us; ceiling "
                        f"{configuration.maximum_48k_latency_us:.3f} us"
                    ),
                ),
                Gate(
                    "nccl-cuda-graph-48k",
                    "pass" if graph_latency_ok else "block",
                    (
                        f"capture/replay={graph_ok}, "
                        f"latency={float(graph_metric.get('latency_us', math.inf)):.3f} us; "
                        f"ceiling {configuration.maximum_graph_48k_latency_us:.3f} us"
                        if isinstance(graph_metric, dict)
                        else "worker omitted CUDA graph evidence"
                    ),
                ),
                Gate(
                    "nccl-1m-bandwidth",
                    "pass" if bandwidth_ok else "block",
                    (
                        f"{float(metric_1m['bus_bandwidth_gb_s']):.3f} GB/s; floor "
                        f"{configuration.minimum_1m_bus_bandwidth_gb_s:.3f} GB/s"
                    ),
                ),
            )
        )

    qualified, serialized_gates = gate_summary(gates)
    report = {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "qualified": qualified,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "script": {"path": str(script), "sha256": file_sha256(script)},
        "configuration": asdict(configuration),
        "gates": serialized_gates,
        "inventory": [asdict(gpu) for gpu in inventory],
        "compute_applications_before": apps_before_raw,
        "compute_applications_after": list(apps_after),
        "topology": topology,
        "topology_relation": topology_relation,
        "p2p_matrices": p2p_matrices,
        "p2p_matrix_relations": p2p_relations,
        "nvlink_status": nvlink_status_raw,
        "peer": peer,
        "nccl": {
            "return_code": benchmark_process.returncode,
            "stdout": benchmark_process.stdout,
            "stderr": benchmark_process.stderr,
            "transport": transport,
            "workers": workers,
            "logs": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in nccl_log_paths
            ],
        },
        "nvlink_counters": {
            "before": counters_before,
            "after": counters_after,
            "delta": deltas,
            "aggregate_delta_kib": total_delta_kib,
        },
        "artifact_directory": str(artifact_directory),
    }
    write_json_atomic(output, report)
    return (0 if qualified else 1), report


def peer_worker(output: Path) -> int:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        write_json_atomic(
            output,
            {
                "qualified": False,
                "detail": (
                    f"CUDA available={torch.cuda.is_available()}, "
                    f"visible device count={torch.cuda.device_count()}"
                ),
            },
        )
        return 1
    access_01 = torch.cuda.can_device_access_peer(0, 1)
    access_10 = torch.cuda.can_device_access_peer(1, 0)
    copies: list[dict[str, Any]] = []
    for source_index, destination_index in ((0, 1), (1, 0)):
        host_pattern = torch.arange(1_048_576, dtype=torch.int32)
        source = host_pattern.to(f"cuda:{source_index}")
        destination = torch.empty_like(source, device=f"cuda:{destination_index}")
        with torch.cuda.device(destination_index):
            destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize(source_index)
        torch.cuda.synchronize(destination_index)
        copies.append(
            {
                "source": source_index,
                "destination": destination_index,
                "bytes": source.numel() * source.element_size(),
                "exact": torch.equal(destination.cpu(), host_pattern),
            }
        )
    qualified = access_01 and access_10 and all(copy["exact"] for copy in copies)
    result = {
        "qualified": qualified,
        "detail": (
            f"cudaDeviceCanAccessPeer 0->1={access_01}, 1->0={access_10}; "
            f"exact peer copies={all(copy['exact'] for copy in copies)}"
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_names": [torch.cuda.get_device_name(index) for index in range(2)],
        "peer_access": {"0_to_1": access_01, "1_to_0": access_10},
        "copies": copies,
    }
    write_json_atomic(output, result)
    return 0 if qualified else 1


def benchmark_collective(
    *,
    tensor: Any,
    torch: Any,
    distributed: Any,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, float | int]:
    tensor.fill_(1)
    for _ in range(warmup_iterations):
        distributed.all_reduce(tensor)
    torch.cuda.synchronize()
    distributed.barrier()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        distributed.all_reduce(tensor)
    stop.record()
    stop.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    local_latency_us = float(start.elapsed_time(stop)) * 1000.0 / iterations
    maximum_latency = torch.tensor(
        [local_latency_us], dtype=torch.float64, device=tensor.device
    )
    distributed.all_reduce(maximum_latency, op=distributed.ReduceOp.MAX)
    latency_us = float(maximum_latency.item())
    byte_count = tensor.numel() * tensor.element_size()
    algorithm_bandwidth = byte_count / (latency_us * 1000.0)
    world_size = distributed.get_world_size()
    bus_bandwidth = algorithm_bandwidth * (2.0 * (world_size - 1) / world_size)
    return {
        "bytes": byte_count,
        "latency_us": latency_us,
        "host_wall_us_per_iteration": wall_elapsed * 1_000_000.0 / iterations,
        "algorithm_bandwidth_gb_s": algorithm_bandwidth,
        "bus_bandwidth_gb_s": bus_bandwidth,
    }


def benchmark_cuda_graph(
    *, tensor: Any, torch: Any, distributed: Any, iterations: int
) -> dict[str, Any]:
    tensor.fill_(1)
    for _ in range(10):
        distributed.all_reduce(tensor)
    torch.cuda.synchronize()
    distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        distributed.all_reduce(tensor)
    distributed.barrier()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    distributed.barrier()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    stop.record()
    stop.synchronize()
    local_latency_us = float(start.elapsed_time(stop)) * 1000.0 / iterations
    maximum_latency = torch.tensor(
        [local_latency_us], dtype=torch.float64, device=tensor.device
    )
    distributed.all_reduce(maximum_latency, op=distributed.ReduceOp.MAX)
    return {
        "bytes": tensor.numel() * tensor.element_size(),
        "capture_and_replay_ok": True,
        "latency_us": float(maximum_latency.item()),
        "iterations": iterations,
    }


def nccl_worker(
    *,
    result_directory: Path,
    message_sizes: tuple[int, ...],
    warmup_iterations: int,
    iterations: int,
    graph_iterations: int,
) -> int:
    import torch
    import torch.distributed as distributed

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    correctness_tensor = torch.full(
        (4096,), rank + 1, dtype=torch.int32, device=f"cuda:{local_rank}"
    )
    distributed.all_reduce(correctness_tensor)
    expected = world_size * (world_size + 1) // 2
    correctness_ok = bool(torch.all(correctness_tensor == expected).item())
    all_reduce: list[dict[str, float | int]] = []
    for size in message_sizes:
        element_count = math.ceil(size / 2)
        tensor = torch.ones(
            element_count, dtype=torch.bfloat16, device=f"cuda:{local_rank}"
        )
        all_reduce.append(
            benchmark_collective(
                tensor=tensor,
                torch=torch,
                distributed=distributed,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
            )
        )
    graph_tensor = torch.ones(
        REPRESENTATIVE_MESSAGE_BYTES // 2,
        dtype=torch.bfloat16,
        device=f"cuda:{local_rank}",
    )
    cuda_graph = benchmark_cuda_graph(
        tensor=graph_tensor,
        torch=torch,
        distributed=distributed,
        iterations=graph_iterations,
    )
    result = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_name": torch.cuda.get_device_name(local_rank),
        "correctness_ok": correctness_ok,
        "all_reduce": all_reduce,
        "cuda_graph": cuda_graph,
    }
    write_json_atomic(result_directory / f"rank-{rank}.json", result)
    distributed.barrier()
    distributed.destroy_process_group()
    return 0 if correctness_ok else 1


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-receipt", type=Path)
    parser.add_argument(
        "--maximum-receipt-age-seconds", type=positive_integer, default=300
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--devices", type=parse_devices, default=(0, 1))
    parser.add_argument(
        "--message-sizes",
        type=parse_message_sizes,
        default=DEFAULT_MESSAGE_SIZES,
    )
    parser.add_argument("--warmup-iterations", type=positive_integer, default=100)
    parser.add_argument("--iterations", type=positive_integer, default=1000)
    parser.add_argument("--graph-iterations", type=positive_integer, default=1000)
    parser.add_argument("--maximum-48k-latency-us", type=positive_float, default=100.0)
    parser.add_argument(
        "--maximum-graph-48k-latency-us", type=positive_float, default=100.0
    )
    parser.add_argument(
        "--minimum-1m-bus-bandwidth-gb-s", type=positive_float, default=10.0
    )
    parser.add_argument(
        "--minimum-counter-delta-kib", type=positive_integer, default=1024
    )
    parser.add_argument("--peer-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-directory", type=Path, help=argparse.SUPPRESS)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.peer_worker is not None:
        return peer_worker(args.peer_worker)
    if args.worker:
        if args.worker_result_directory is None:
            parser.error("--worker requires --worker-result-directory")
        return nccl_worker(
            result_directory=args.worker_result_directory,
            message_sizes=args.message_sizes,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            graph_iterations=args.graph_iterations,
        )
    if args.validate_receipt is not None:
        try:
            validate_receipt(
                receipt_path=args.validate_receipt,
                devices=args.devices,
                maximum_age_seconds=args.maximum_receipt_age_seconds,
            )
        except QualificationError as error:
            print(f"DSV4 TP2 interconnect receipt rejected: {error}", file=sys.stderr)
            return 1
        print(f"DSV4 TP2 interconnect receipt accepted: {args.validate_receipt}")
        return 0
    if args.output is None:
        parser.error("--output is required")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error(f"--python is not executable: {args.python}")
    configuration = Configuration(
        devices=args.devices,
        message_sizes_bytes=args.message_sizes,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        graph_iterations=args.graph_iterations,
        maximum_48k_latency_us=args.maximum_48k_latency_us,
        maximum_graph_48k_latency_us=args.maximum_graph_48k_latency_us,
        minimum_1m_bus_bandwidth_gb_s=args.minimum_1m_bus_bandwidth_gb_s,
        minimum_counter_delta_kib=args.minimum_counter_delta_kib,
    )
    try:
        return_code, report = qualify(
            output=args.output, python=args.python, configuration=configuration
        )
    except QualificationError as error:
        report = {
            "schema": SCHEMA,
            "created_at_utc": utc_now(),
            "qualified": False,
            "fatal_error": str(error),
            "configuration": asdict(configuration),
        }
        if not args.output.exists():
            write_json_atomic(args.output, report)
        print(f"DSV4 TP2 interconnect qualification failed: {error}", file=sys.stderr)
        return 1
    if return_code == 0:
        print(f"DSV4 TP2 interconnect qualified: {args.output}")
    else:
        blocked = [
            gate["name"] for gate in report["gates"] if gate["status"] == "block"
        ]
        print(
            "DSV4 TP2 interconnect qualification blocked: " + ", ".join(blocked),
            file=sys.stderr,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
