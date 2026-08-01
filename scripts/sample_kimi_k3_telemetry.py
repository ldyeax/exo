#!/usr/bin/env python3
"""Sample host, process, GPU, network, and power telemetry as JSONL.

The sampler has no third-party dependencies.  It is intended to run beside a
long-lived llama-server or ggml-rpc-server process and to preserve enough raw
counters to derive request-boundary deltas after a benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Final, cast

NVIDIA_QUERY_FIELDS: Final[tuple[str, ...]] = (
    "index",
    "uuid",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)

stop_requested = False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write 1 Hz Kimi K3 runtime telemetry to a JSONL file.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="PID to sample; the sampler exits after this process disappears.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--interface",
        action="append",
        default=[],
        help="Network interface to sample; repeat for multiple interfaces.",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--numa-every",
        type=int,
        default=30,
        help="Parse the potentially large numa_maps file every N samples; 0 disables it.",
    )
    parser.add_argument(
        "--smaps-every",
        type=int,
        default=1,
        help="Parse smaps_rollup every N samples; 0 disables it.",
    )
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=10,
        help="fsync after this many samples; 0 disables periodic fsync.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Exit cleanly when this path appears.",
    )
    return parser.parse_args()


def handle_signal(_signal_number: int, _frame: object) -> None:
    global stop_requested
    stop_requested = True


def read_key_value_file(path: Path) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                key, separator, raw_value = line.partition(":")
                if not separator:
                    continue
                tokens = raw_value.strip().split()
                if not tokens:
                    values[key] = ""
                    continue
                try:
                    values[key] = int(tokens[0])
                except ValueError:
                    values[key] = raw_value.strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return values


def read_process_stat(pid: int) -> dict[str, int | str]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}

    command_end = raw.rfind(")")
    if command_end < 0:
        return {}
    command_start = raw.find("(")
    fields = raw[command_end + 2 :].split()
    if len(fields) < 20:
        return {}
    return {
        "command": raw[command_start + 1 : command_end],
        "state": fields[0],
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
        "user_jiffies": int(fields[11]),
        "system_jiffies": int(fields[12]),
        "start_jiffies": int(fields[19]),
    }


def read_numa_pages(pid: int) -> dict[str, int]:
    pages: dict[str, int] = {}
    try:
        with Path(f"/proc/{pid}/numa_maps").open(encoding="utf-8") as source:
            for line in source:
                for token in line.split():
                    if not token.startswith("N") or "=" not in token:
                        continue
                    node, raw_pages = token.split("=", 1)
                    if node[1:].isdigit() and raw_pages.isdigit():
                        pages[node] = pages.get(node, 0) + int(raw_pages)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return pages


def read_cpu_stat() -> dict[str, int]:
    try:
        first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, PermissionError, IndexError):
        return {}
    fields = first_line.split()
    names = (
        "user",
        "nice",
        "system",
        "idle",
        "iowait",
        "irq",
        "softirq",
        "steal",
        "guest",
        "guest_nice",
    )
    return {name: int(value) for name, value in zip(names, fields[1:], strict=False)}


def read_vmstat() -> dict[str, int]:
    selected_names = {
        "pgfault",
        "pgmajfault",
        "pgpgin",
        "pgpgout",
        "pswpin",
        "pswpout",
        "numa_hit",
        "numa_miss",
        "numa_foreign",
        "numa_interleave",
        "numa_local",
        "numa_other",
    }
    result: dict[str, int] = {}
    try:
        with Path("/proc/vmstat").open(encoding="utf-8") as source:
            for line in source:
                name, raw_value = line.split()
                if name in selected_names:
                    result[name] = int(raw_value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return result


def read_network_interfaces(interfaces: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    counter_names = (
        "rx_bytes",
        "tx_bytes",
        "rx_packets",
        "tx_packets",
        "rx_errors",
        "tx_errors",
        "rx_dropped",
        "tx_dropped",
    )
    for interface in interfaces:
        counters: dict[str, int] = {}
        base = Path("/sys/class/net") / interface / "statistics"
        for counter_name in counter_names:
            try:
                counters[counter_name] = int(
                    (base / counter_name).read_text(encoding="utf-8").strip(),
                )
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        result[interface] = counters
    return result


def read_infiniband_counters() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    patterns = (
        "/sys/class/infiniband/*/ports/*/counters/port_xmit_data",
        "/sys/class/infiniband/*/ports/*/counters/port_rcv_data",
        "/sys/class/infiniband/*/ports/*/counters/port_xmit_packets",
        "/sys/class/infiniband/*/ports/*/counters/port_rcv_packets",
    )
    for pattern in patterns:
        for raw_path in glob.glob(pattern):
            path = Path(raw_path)
            device = path.parents[3].name
            port = path.parents[1].name
            counter_name = path.name
            key = f"{device}/port{port}"
            try:
                value = int(path.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, PermissionError, ValueError):
                continue
            counters = result.setdefault(key, {})
            counters[counter_name] = value
            if counter_name in {"port_xmit_data", "port_rcv_data"}:
                counters[f"{counter_name}_bytes"] = value * 4
    return result


def read_rapl_energy() -> dict[str, int]:
    result: dict[str, int] = {}
    powercap_root = Path("/sys/class/powercap")
    if not powercap_root.exists():
        return result
    for energy_path in powercap_root.rglob("energy_uj"):
        domain_path = energy_path.parent
        try:
            domain_name = (domain_path / "name").read_text(encoding="utf-8").strip()
            energy = int(energy_path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        result[f"{domain_path.name}:{domain_name}"] = energy
    return result


def parse_nvidia_value(raw_value: str) -> int | float | str | None:
    value = raw_value.strip()
    if value in {"N/A", "[N/A]", "Not Supported", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_gpus() -> tuple[list[dict[str, int | float | str | None]], str | None]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return [], str(error)
    if completed.returncode != 0:
        return [], completed.stderr.strip() or f"exit {completed.returncode}"

    gpus: list[dict[str, int | float | str | None]] = []
    for line in completed.stdout.splitlines():
        values = [parse_nvidia_value(value) for value in line.split(",")]
        if len(values) != len(NVIDIA_QUERY_FIELDS):
            continue
        gpus.append(dict(zip(NVIDIA_QUERY_FIELDS, values, strict=True)))
    return gpus, None


def collect_sample(
    *,
    pid: int,
    label: str,
    interfaces: list[str],
    include_numa: bool,
    include_smaps: bool,
) -> dict[str, Any]:
    monotonic_ns = time.monotonic_ns()
    realtime_ns = time.time_ns()
    gpus, gpu_error = read_gpus()
    sample: dict[str, Any] = {
        "schema": "kimi-k3-telemetry-v1",
        "label": label,
        "host": socket.gethostname(),
        "pid": pid,
        "realtime_ns": realtime_ns,
        "monotonic_ns": monotonic_ns,
        "process_stat": read_process_stat(pid),
        "process_status": read_key_value_file(Path(f"/proc/{pid}/status")),
        "meminfo": read_key_value_file(Path("/proc/meminfo")),
        "cpu": read_cpu_stat(),
        "vmstat": read_vmstat(),
        "network": read_network_interfaces(interfaces),
        "infiniband": read_infiniband_counters(),
        "rapl_energy_uj": read_rapl_energy(),
        "gpus": gpus,
    }
    if gpu_error is not None:
        sample["gpu_error"] = gpu_error
    if include_smaps:
        sample["process_smaps_rollup"] = read_key_value_file(
            Path(f"/proc/{pid}/smaps_rollup"),
        )
    if include_numa:
        sample["process_numa_pages"] = read_numa_pages(pid)
    return sample


def main() -> int:
    arguments = parse_arguments()
    pid = cast(int, arguments.pid)
    output = cast(Path, arguments.output)
    label = cast(str, arguments.label)
    interfaces = cast(list[str], arguments.interface)
    interval = cast(float, arguments.interval)
    numa_every = cast(int, arguments.numa_every)
    smaps_every = cast(int, arguments.smaps_every)
    fsync_every = cast(int, arguments.fsync_every)
    stop_file = cast(Path | None, arguments.stop_file)

    if pid <= 0:
        raise SystemExit("--pid must be positive")
    if interval <= 0:
        raise SystemExit("--interval must be positive")
    if numa_every < 0:
        raise SystemExit("--numa-every must be nonnegative")
    if smaps_every < 0:
        raise SystemExit("--smaps-every must be nonnegative")

    output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sample_index = 0
    next_deadline = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as destination:
        while not stop_requested:
            if stop_file is not None and stop_file.exists():
                break
            if not Path(f"/proc/{pid}").exists():
                break

            sample = collect_sample(
                pid=pid,
                label=label,
                interfaces=interfaces,
                include_numa=(numa_every > 0 and sample_index % numa_every == 0),
                include_smaps=(smaps_every > 0 and sample_index % smaps_every == 0),
            )
            destination.write(
                json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n",
            )
            sample_index += 1
            if fsync_every > 0 and sample_index % fsync_every == 0:
                destination.flush()
                os.fsync(destination.fileno())

            next_deadline += interval
            wait_seconds = next_deadline - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            else:
                next_deadline = time.monotonic()

        destination.flush()
        os.fsync(destination.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
