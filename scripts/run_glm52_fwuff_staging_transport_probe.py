#!/usr/bin/env python3
"""Measure fwuff pinned-CUDA staging and application-level EDR messaging.

This probe intentionally does not launch a model.  It measures two pieces of
the admitted remote GLM-5.2 MTP path:

* preallocated pinned-host H2D/D2H copies and CUDA event waits on fwuff's 3090;
* pyzmq/TCP request-proposal exchanges bound to the mlx5_0 IPoIB addresses.

The transport comparison keeps a one-request REQ/REP control and a two-slot
DEALER/ROUTER pipeline.  Hidden-state requests receive a four-byte proposal
reply, matching the narrow speculative wire contract rather than echoing the
large request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

HIDDEN_SIZE: Final = 6_144
BF16_BYTES: Final = 2
FEATURE_ROW_BYTES: Final = HIDDEN_SIZE * BF16_BYTES
OPEN_PREFIX_ROWS: Final = 512
OPEN_PREFIX_BYTES: Final = OPEN_PREFIX_ROWS * FEATURE_ROW_BYTES

DEFAULT_LOCAL_EDR_IP: Final = "10.44.0.1"
DEFAULT_REMOTE_EDR_IP: Final = "10.44.0.2"
DEFAULT_LOCAL_NETDEV: Final = "ibs5"
DEFAULT_REMOTE_NETDEV: Final = "ibs2"
DEFAULT_SSH_TARGET: Final = "fwuff"
DEFAULT_PORT_BASE: Final = 18_740
DEFAULT_LOCAL_APPLICATION_CPU: Final = 167
DEFAULT_REMOTE_APPLICATION_CPU: Final = 119
DEFAULT_LOCAL_RUNTIME_PYTHON: Final = (
    "/var/lib/exo/runtimes/glm52-osdi26-w8-overlay/dwagon/"
    "b1b05ea2a1b5b893c2ce5d2e3cd907bd100b57e963725936cb743ff5bf3a64e9/"
    "venv/bin/python"
)
DEFAULT_REMOTE_RUNTIME_PYTHON: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/fwuff/"
    "563dba484c323139f05b7853384dc565d6f6f8282f38e365327fa1bb3d9782aa/"
    "venv/bin/python"
)
RESULT_FILENAME: Final = "glm52-fwuff-staging-transport-probe.json"
MANIFEST_FILENAME: Final = "content-manifest.json"
REPLY_BYTES: Final = 4
REMOTE_TIMEOUT_SECONDS: Final = 30.0
SO_BUSY_POLL_NUMBER: Final = 46
BUSY_POLL_MICROSECONDS: Final = 50


class StagingTransportProbeError(RuntimeError):
    """Expected fail-closed probe error."""


@dataclass(frozen=True, slots=True)
class PayloadProbe:
    name: str
    message_kind: Literal["PROPOSAL", "ADVANCE", "OPEN"]
    payload_bytes: int
    staging_iterations: int
    transport_iterations: int
    transport_warmup_iterations: int


PAYLOAD_PROBES: Final = (
    PayloadProbe(
        name="proposal_one_int32_id",
        message_kind="PROPOSAL",
        payload_bytes=4,
        staging_iterations=2_000,
        transport_iterations=2_000,
        transport_warmup_iterations=100,
    ),
    PayloadProbe(
        name="advance_one_bf16_hidden_row",
        message_kind="ADVANCE",
        payload_bytes=FEATURE_ROW_BYTES,
        staging_iterations=2_000,
        transport_iterations=2_000,
        transport_warmup_iterations=100,
    ),
    PayloadProbe(
        name="advance_two_bf16_hidden_rows",
        message_kind="ADVANCE",
        payload_bytes=2 * FEATURE_ROW_BYTES,
        staging_iterations=2_000,
        transport_iterations=2_000,
        transport_warmup_iterations=100,
    ),
    PayloadProbe(
        name="open_512_bf16_hidden_rows",
        message_kind="OPEN",
        payload_bytes=OPEN_PREFIX_BYTES,
        staging_iterations=128,
        transport_iterations=64,
        transport_warmup_iterations=8,
    ),
)

_NETDEV_COUNTERS: Final = (
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
)
_HEALTH_COUNTERS: Final = frozenset(_NETDEV_COUNTERS[4:])


def _nearest_rank(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise StagingTransportProbeError("latency samples are empty")
    if not 0.0 < percentile <= 1.0:
        raise StagingTransportProbeError("percentile must be in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def latency_summary(samples_microseconds: Sequence[float]) -> JsonObject:
    if not samples_microseconds:
        raise StagingTransportProbeError("latency samples are empty")
    return {
        "sample_count": len(samples_microseconds),
        "minimum_microseconds": min(samples_microseconds),
        "mean_microseconds": sum(samples_microseconds) / len(samples_microseconds),
        "p50_microseconds": _nearest_rank(samples_microseconds, 0.50),
        "p95_microseconds": _nearest_rank(samples_microseconds, 0.95),
        "p99_microseconds": _nearest_rank(samples_microseconds, 0.99),
        "maximum_microseconds": max(samples_microseconds),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise StagingTransportProbeError(f"cannot hash {path}") from error
    return digest.hexdigest()


def _read_netdev_counters(netdev: str) -> dict[str, int]:
    root = Path("/sys/class/net") / netdev / "statistics"
    if not root.is_dir():
        raise StagingTransportProbeError(f"network interface is absent: {netdev}")
    counters: dict[str, int] = {}
    for name in _NETDEV_COUNTERS:
        try:
            counters[name] = int((root / name).read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise StagingTransportProbeError(
                f"cannot read {netdev} counter {name}"
            ) from error
    return counters


def _remote_netdev_script(netdev: str) -> str:
    return f"""
import json
from pathlib import Path
netdev = {netdev!r}
names = {_NETDEV_COUNTERS!r}
root = Path("/sys/class/net") / netdev / "statistics"
if not root.is_dir():
    raise RuntimeError("remote EDR network interface is absent")
print(json.dumps({{
    "hostname": Path("/etc/hostname").read_text().strip(),
    "netdev": netdev,
    "counters": {{name: int((root / name).read_text().strip()) for name in names}},
}}, sort_keys=True, separators=(",", ":")))
"""


def _encode_remote_source(source: str) -> str:
    return base64.b64encode(source.encode("utf-8")).decode("ascii")


def remote_python_command(
    runtime_python: str,
    source: str,
    arguments: Sequence[str] = (),
) -> str:
    encoded = _encode_remote_source(source)
    bootstrap = f"import base64;exec(base64.b64decode({encoded!r}))"
    return shlex.join((runtime_python, "-c", bootstrap, *arguments))


def _run_ssh(
    ssh_target: str,
    command: str,
    *,
    timeout_seconds: float = REMOTE_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ("ssh", ssh_target, command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise StagingTransportProbeError(
            f"remote command failed with {result.returncode}: {result.stderr[-2_000:]}"
        )
    return result


def _read_remote_netdev(
    ssh_target: str,
    remote_python: str,
    netdev: str,
) -> JsonObject:
    result = _run_ssh(
        ssh_target,
        remote_python_command(
            remote_python,
            _remote_netdev_script(netdev),
        ),
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise StagingTransportProbeError(
            "remote network observation did not return JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise StagingTransportProbeError("remote network observation is not an object")
    return cast(JsonObject, parsed)


CUDA_WORKER_SOURCE: Final = r"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def nearest_rank(samples, percentile):
    ordered = sorted(samples)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def summary(samples):
    return {
        "sample_count": len(samples),
        "minimum_microseconds": min(samples),
        "mean_microseconds": sum(samples) / len(samples),
        "p50_microseconds": nearest_rank(samples, 0.50),
        "p95_microseconds": nearest_rank(samples, 0.95),
        "p99_microseconds": nearest_rank(samples, 0.99),
        "maximum_microseconds": max(samples),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--payloads-json", required=True)
parser.add_argument("--warmup", type=int, default=50)
args = parser.parse_args()
payloads = json.loads(args.payloads_json)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
torch.cuda.set_device(0)
torch.set_num_threads(1)
torch.cuda.synchronize()

runtime_executable = Path(sys.executable).resolve(strict=True)
nvidia = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=uuid,name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
    timeout=5,
).stdout.strip()

event_stream = torch.cuda.Stream()
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)
for _ in range(args.warmup):
    event_start.record(event_stream)
    event_end.record(event_stream)
    event_end.synchronize()
event_host = []
event_device = []
event_iterations = max(1000, max(int(item["iterations"]) for item in payloads))
for _ in range(event_iterations):
    before = time.perf_counter_ns()
    event_start.record(event_stream)
    event_end.record(event_stream)
    event_end.synchronize()
    event_host.append((time.perf_counter_ns() - before) / 1000.0)
    event_device.append(event_start.elapsed_time(event_end) * 1000.0)

results = []
allocation_addresses = []
for payload in payloads:
    payload_bytes = int(payload["payload_bytes"])
    iterations = int(payload["iterations"])
    host_sources = [
        torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(2)
    ]
    host_destinations = [
        torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(2)
    ]
    devices = [
        torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(2)
    ]
    for host_source in host_sources:
        host_source.fill_(0xA5)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    starts = [
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    ]
    ends = [
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    ]
    allocation_addresses.append({
        "name": payload["name"],
        "source_host_data_ptrs": [item.data_ptr() for item in host_sources],
        "destination_host_data_ptrs": [
            item.data_ptr() for item in host_destinations
        ],
        "device_data_ptrs": [item.data_ptr() for item in devices],
        "sources_are_pinned": all(item.is_pinned() for item in host_sources),
        "destinations_are_pinned": all(
            item.is_pinned() for item in host_destinations
        ),
    })
    for index in range(args.warmup):
        slot = index % 2
        with torch.cuda.stream(streams[slot]):
            devices[slot].copy_(host_sources[slot], non_blocking=True)
            host_destinations[slot].copy_(devices[slot], non_blocking=True)
    torch.cuda.synchronize()

    direction_results = {}
    for direction in ("h2d", "d2h"):
        host_samples = []
        device_samples = []
        stream = streams[0]
        start = starts[0]
        end = ends[0]
        for _ in range(iterations):
            before = time.perf_counter_ns()
            start.record(stream)
            with torch.cuda.stream(stream):
                if direction == "h2d":
                    devices[0].copy_(host_sources[0], non_blocking=True)
                else:
                    host_destinations[0].copy_(devices[0], non_blocking=True)
            end.record(stream)
            end.synchronize()
            host_samples.append((time.perf_counter_ns() - before) / 1000.0)
            device_samples.append(start.elapsed_time(end) * 1000.0)

        two_slot_samples = []
        pair_iterations = max(16, iterations // 2)
        for _ in range(pair_iterations):
            before = time.perf_counter_ns()
            for slot in (0, 1):
                starts[slot].record(streams[slot])
                with torch.cuda.stream(streams[slot]):
                    if direction == "h2d":
                        devices[slot].copy_(
                            host_sources[slot],
                            non_blocking=True,
                        )
                    else:
                        host_destinations[slot].copy_(
                            devices[slot],
                            non_blocking=True,
                        )
                ends[slot].record(streams[slot])
            ends[0].synchronize()
            ends[1].synchronize()
            two_slot_samples.append(
                ((time.perf_counter_ns() - before) / 1000.0) / 2.0
            )
        direction_results[direction] = {
            "single_slot_host_submit_wait": summary(host_samples),
            "cuda_event_elapsed": summary(device_samples),
            "two_slot_host_amortized_per_copy": summary(two_slot_samples),
        }
    results.append({
        "name": payload["name"],
        "payload_bytes": payload_bytes,
        "iterations": iterations,
        "directions": direction_results,
    })
    del devices
    del host_sources
    del host_destinations
torch.cuda.synchronize()

print(json.dumps({
    "schema_version": 1,
    "runtime": {
        "python_executable": str(runtime_executable),
        "python_executable_sha256": sha256_file(runtime_executable),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "zmq_version": __import__("zmq").__version__,
    },
    "gpu": {
        "nvidia_smi_identity": nvidia,
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    },
    "allocation_contract": {
        "preallocated": True,
        "pinned_host": True,
        "non_blocking_cuda_copies": True,
        "allocation_addresses": allocation_addresses,
    },
    "empty_cuda_event_record_and_wait": {
        "host_submit_wait": summary(event_host),
        "cuda_event_elapsed": summary(event_device),
    },
    "payloads": results,
}, sort_keys=True, separators=(",", ":")))
"""


ZMQ_SERVER_SOURCE: Final = r"""
import argparse
import json
import os
import struct
import sys
import time

import zmq

parser = argparse.ArgumentParser()
parser.add_argument("--bind-ip", required=True)
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--mode", choices=("reqrep", "router"), required=True)
parser.add_argument("--payload-bytes", required=True, type=int)
parser.add_argument("--message-count", required=True, type=int)
parser.add_argument("--token", required=True)
parser.add_argument("--application-cpu", required=True, type=int)
args = parser.parse_args()

context = zmq.Context(io_threads=1)
socket_type = zmq.REP if args.mode == "reqrep" else zmq.ROUTER
sock = context.socket(socket_type)
os.sched_setaffinity(0, {args.application_cpu})
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, 30000)
sock.setsockopt(zmq.SNDTIMEO, 30000)
endpoint = f"tcp://{args.bind_ip}:{args.port}"
sock.bind(endpoint)
print(json.dumps({
    "event": "ready",
    "pid": os.getpid(),
    "token": args.token,
    "endpoint": endpoint,
    "mode": args.mode,
    "zmq_version": zmq.__version__,
}, sort_keys=True, separators=(",", ":")), flush=True)

received = 0
started_ns = time.perf_counter_ns()
try:
    while received < args.message_count:
        if args.mode == "reqrep":
            payload = sock.recv(copy=False)
            if len(payload.buffer) != args.payload_bytes:
                raise RuntimeError("REQ payload length mismatch")
            sequence = struct.unpack_from("!I", payload.buffer, 0)[0]
            sock.send(struct.pack("!I", sequence))
        else:
            frames = sock.recv_multipart(copy=False)
            if len(frames) != 2:
                raise RuntimeError("DEALER message must have identity and payload")
            identity, payload = frames
            if len(payload.buffer) != args.payload_bytes:
                raise RuntimeError("DEALER payload length mismatch")
            sequence = struct.unpack_from("!I", payload.buffer, 0)[0]
            sock.send_multipart([identity.buffer, struct.pack("!I", sequence)])
        received += 1
finally:
    elapsed_ns = time.perf_counter_ns() - started_ns
    sock.close(linger=0)
    context.term()
print(json.dumps({
    "event": "complete",
    "pid": os.getpid(),
    "token": args.token,
    "received_messages": received,
    "elapsed_seconds": elapsed_ns / 1e9,
}, sort_keys=True, separators=(",", ":")), flush=True)
"""

RAW_TCP_SERVER_SOURCE: Final = r"""
import argparse
import json
import os
import socket
import struct
import time

parser = argparse.ArgumentParser()
parser.add_argument("--bind-ip", required=True)
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--payload-bytes", required=True, type=int)
parser.add_argument("--message-count", required=True, type=int)
parser.add_argument("--token", required=True)
parser.add_argument("--application-cpu", required=True, type=int)
parser.add_argument("--busy-poll-microseconds", required=True, type=int)
args = parser.parse_args()


def configure_busy_poll(sock):
    if args.busy_poll_microseconds <= 0:
        return {
            "requested_microseconds": args.busy_poll_microseconds,
            "accepted": True,
            "observed_microseconds": 0,
            "error": None,
        }
    try:
        sock.setsockopt(socket.SOL_SOCKET, 46, args.busy_poll_microseconds)
        observed = sock.getsockopt(socket.SOL_SOCKET, 46)
    except OSError as error:
        return {
            "requested_microseconds": args.busy_poll_microseconds,
            "accepted": False,
            "observed_microseconds": None,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "requested_microseconds": args.busy_poll_microseconds,
        "accepted": observed == args.busy_poll_microseconds,
        "observed_microseconds": observed,
        "error": None,
    }


def receive_exact_into(connection, destination):
    view = memoryview(destination)
    offset = 0
    while offset < len(destination):
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise RuntimeError("raw TCP peer closed before one complete request")
        offset += received


if args.application_cpu >= 0:
    os.sched_setaffinity(0, {args.application_cpu})
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
listener.settimeout(30.0)
listener_busy_poll = configure_busy_poll(listener)
listener.bind((args.bind_ip, args.port))
listener.listen(1)
endpoint = f"{args.bind_ip}:{args.port}"
print(json.dumps({
    "event": "ready",
    "pid": os.getpid(),
    "token": args.token,
    "endpoint": endpoint,
    "application_cpu": args.application_cpu,
    "observed_application_affinity": sorted(os.sched_getaffinity(0)),
    "tcp_nodelay": listener.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY),
    "listener_busy_poll": listener_busy_poll,
}, sort_keys=True, separators=(",", ":")), flush=True)

received_messages = 0
accepted_busy_poll = None
request_buffer = bytearray(args.payload_bytes)
reply_buffer = bytearray(4)
started_ns = time.perf_counter_ns()
try:
    connection, peer = listener.accept()
    with connection:
        connection.settimeout(30.0)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        accepted_busy_poll = configure_busy_poll(connection)
        while received_messages < args.message_count:
            receive_exact_into(connection, request_buffer)
            sequence = struct.unpack_from("!I", request_buffer, 0)[0]
            struct.pack_into("!I", reply_buffer, 0, sequence)
            connection.sendall(reply_buffer)
            received_messages += 1
finally:
    elapsed_ns = time.perf_counter_ns() - started_ns
    listener.close()
print(json.dumps({
    "event": "complete",
    "pid": os.getpid(),
    "token": args.token,
    "received_messages": received_messages,
    "elapsed_seconds": elapsed_ns / 1e9,
    "accepted_busy_poll": accepted_busy_poll,
}, sort_keys=True, separators=(",", ":")), flush=True)
"""


def _parse_single_json(output: str, description: str) -> JsonObject:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise StagingTransportProbeError(
            f"{description} returned {len(lines)} non-empty lines, expected one"
        )
    try:
        parsed = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise StagingTransportProbeError(
            f"{description} did not return JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise StagingTransportProbeError(f"{description} result is not an object")
    return cast(JsonObject, parsed)


def run_cuda_staging_probe(
    *,
    ssh_target: str,
    remote_runtime_python: str,
    payloads: Sequence[PayloadProbe],
    timeout_seconds: float,
) -> JsonObject:
    payload_json = json.dumps(
        [
            {
                "name": payload.name,
                "payload_bytes": payload.payload_bytes,
                "iterations": payload.staging_iterations,
            }
            for payload in payloads
        ],
        separators=(",", ":"),
    )
    command = remote_python_command(
        remote_runtime_python,
        CUDA_WORKER_SOURCE,
        ("--payloads-json", payload_json),
    )
    result = _run_ssh(
        ssh_target,
        command,
        timeout_seconds=timeout_seconds,
    )
    return _parse_single_json(result.stdout, "remote CUDA staging worker")


def _read_server_event(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> JsonObject:
    stdout = process.stdout
    if stdout is None:
        raise StagingTransportProbeError("remote server stdout is unavailable")
    readable, _, _ = select.select((stdout,), (), (), timeout_seconds)
    if not readable:
        raise StagingTransportProbeError("timed out waiting for remote server event")
    line = stdout.readline()
    if not line:
        stderr = "" if process.stderr is None else process.stderr.read()
        raise StagingTransportProbeError(
            f"remote server exited before event: {stderr[-2_000:]}"
        )
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise StagingTransportProbeError(
            f"remote server emitted non-JSON event: {line[-500:]}"
        ) from error
    if not isinstance(parsed, dict):
        raise StagingTransportProbeError("remote server event is not an object")
    return cast(JsonObject, parsed)


def _terminate_process(process: subprocess.Popen[str]) -> bool:
    if process.poll() is not None:
        return False
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=3)
        return False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
        return True


def _make_payload(payload_bytes: int) -> bytearray:
    if payload_bytes < 4:
        raise StagingTransportProbeError("payload must hold a uint32 sequence")
    payload = bytearray(payload_bytes)
    if payload_bytes > 4:
        payload[4:] = bytes((0xA5,)) * (payload_bytes - 4)
    return payload


def _put_sequence(payload: bytearray, sequence: int) -> None:
    payload[0:4] = int(sequence).to_bytes(4, byteorder="big", signed=False)


def _import_zmq() -> object:
    try:
        import zmq
    except ImportError as error:
        raise StagingTransportProbeError(
            "the selected local immutable runtime does not provide pyzmq"
        ) from error
    return zmq


def _configure_busy_poll(
    sock: socket.socket,
    busy_poll_microseconds: int,
) -> JsonObject:
    if busy_poll_microseconds <= 0:
        return {
            "requested_microseconds": busy_poll_microseconds,
            "accepted": True,
            "observed_microseconds": 0,
            "error": None,
        }
    try:
        sock.setsockopt(
            socket.SOL_SOCKET,
            SO_BUSY_POLL_NUMBER,
            busy_poll_microseconds,
        )
        observed = sock.getsockopt(socket.SOL_SOCKET, SO_BUSY_POLL_NUMBER)
    except OSError as error:
        return {
            "requested_microseconds": busy_poll_microseconds,
            "accepted": False,
            "observed_microseconds": None,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "requested_microseconds": busy_poll_microseconds,
        "accepted": observed == busy_poll_microseconds,
        "observed_microseconds": observed,
        "error": None,
    }


def _receive_exact_into(sock: socket.socket, destination: bytearray) -> None:
    view = memoryview(destination)
    offset = 0
    while offset < len(destination):
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise StagingTransportProbeError(
                "raw TCP peer closed before one complete response"
            )
        offset += received


def _run_reqrep_client(
    *,
    endpoint: str,
    payload_bytes: int,
    warmup_iterations: int,
    iterations: int,
    application_cpu: int,
) -> JsonObject:
    zmq = _import_zmq()
    context = zmq.Context(io_threads=1)
    sock = context.socket(zmq.REQ)
    os.sched_setaffinity(0, {application_cpu})
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 30_000)
    sock.setsockopt(zmq.SNDTIMEO, 30_000)
    sock.connect(endpoint)
    payload = _make_payload(payload_bytes)
    latencies: list[float] = []
    total = warmup_iterations + iterations
    measured_start_ns = 0
    measured_end_ns = 0
    try:
        for sequence in range(total):
            _put_sequence(payload, sequence)
            if sequence == warmup_iterations:
                measured_start_ns = time.perf_counter_ns()
            start_ns = time.perf_counter_ns()
            tracker = sock.send(
                memoryview(payload),
                copy=False,
                track=True,
            )
            reply = sock.recv(copy=False)
            if int.from_bytes(reply.buffer, byteorder="big", signed=False) != sequence:
                raise StagingTransportProbeError("REQ/REP sequence mismatch")
            tracker.wait()
            if sequence >= warmup_iterations:
                latencies.append((time.perf_counter_ns() - start_ns) / 1000.0)
        measured_end_ns = time.perf_counter_ns()
    finally:
        sock.close(linger=0)
        context.term()
    elapsed_seconds = (measured_end_ns - measured_start_ns) / 1e9
    return {
        "socket_pattern": "REQ_REP",
        "window_slots": 1,
        "request_payload_bytes": payload_bytes,
        "reply_payload_bytes": REPLY_BYTES,
        "send_copy_policy": "zero_copy_tracked_preallocated",
        "application_cpu": application_cpu,
        "zmq_io_thread_affinity": "operating_system_default",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "round_trip_latency": latency_summary(latencies),
        "measured_elapsed_seconds": elapsed_seconds,
        "completed_messages_per_second": iterations / elapsed_seconds,
    }


def _run_two_slot_client(
    *,
    endpoint: str,
    payload_bytes: int,
    warmup_iterations: int,
    iterations: int,
    application_cpu: int,
) -> JsonObject:
    zmq = _import_zmq()
    context = zmq.Context(io_threads=1)
    sock = context.socket(zmq.DEALER)
    os.sched_setaffinity(0, {application_cpu})
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 30_000)
    sock.setsockopt(zmq.SNDTIMEO, 30_000)
    sock.setsockopt(zmq.IDENTITY, f"dwagon-{os.getpid()}".encode("ascii"))
    sock.connect(endpoint)
    slots = (_make_payload(payload_bytes), _make_payload(payload_bytes))
    trackers: list[object | None] = [None, None]
    outstanding: dict[int, tuple[int, int]] = {}
    measured_latencies: list[float] = []
    total = warmup_iterations + iterations
    next_sequence = 0
    completed = 0
    measured_start_ns = 0
    measured_end_ns = 0

    def send_slot(slot: int, sequence: int) -> None:
        previous = trackers[slot]
        if previous is not None:
            previous.wait()
        payload = slots[slot]
        _put_sequence(payload, sequence)
        start_ns = time.perf_counter_ns()
        trackers[slot] = sock.send(
            memoryview(payload),
            copy=False,
            track=True,
        )
        outstanding[sequence] = (slot, start_ns)

    try:
        while next_sequence < min(2, total):
            send_slot(next_sequence, next_sequence)
            next_sequence += 1
        while completed < total:
            reply = sock.recv(copy=False)
            sequence = int.from_bytes(reply.buffer, byteorder="big", signed=False)
            try:
                slot, start_ns = outstanding.pop(sequence)
            except KeyError as error:
                raise StagingTransportProbeError(
                    f"unexpected pipeline reply sequence {sequence}"
                ) from error
            completed += 1
            if sequence >= warmup_iterations:
                if measured_start_ns == 0:
                    measured_start_ns = start_ns
                measured_latencies.append((time.perf_counter_ns() - start_ns) / 1000.0)
            if next_sequence < total:
                send_slot(slot, next_sequence)
                next_sequence += 1
        measured_end_ns = time.perf_counter_ns()
        for tracker in trackers:
            if tracker is not None:
                tracker.wait()
    finally:
        sock.close(linger=0)
        context.term()
    elapsed_seconds = (measured_end_ns - measured_start_ns) / 1e9
    return {
        "socket_pattern": "DEALER_ROUTER",
        "window_slots": 2,
        "request_payload_bytes": payload_bytes,
        "reply_payload_bytes": REPLY_BYTES,
        "send_copy_policy": "zero_copy_tracked_preallocated",
        "application_cpu": application_cpu,
        "zmq_io_thread_affinity": "operating_system_default",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "completion_latency": latency_summary(measured_latencies),
        "measured_elapsed_seconds": elapsed_seconds,
        "completed_messages_per_second": iterations / elapsed_seconds,
    }


def _connect_raw_tcp(
    *,
    local_edr_ip: str,
    remote_edr_ip: str,
    port: int,
    application_cpu: int,
    busy_poll_microseconds: int,
) -> tuple[socket.socket, JsonObject]:
    if application_cpu >= 0:
        os.sched_setaffinity(0, {application_cpu})
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(30.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        busy_poll = _configure_busy_poll(sock, busy_poll_microseconds)
        sock.bind((local_edr_ip, 0))
        sock.connect((remote_edr_ip, port))
    except BaseException:
        sock.close()
        raise
    observation: JsonObject = {
        "local_endpoint": f"{sock.getsockname()[0]}:{sock.getsockname()[1]}",
        "remote_endpoint": f"{sock.getpeername()[0]}:{sock.getpeername()[1]}",
        "tcp_nodelay": sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY),
        "busy_poll": busy_poll,
        "application_cpu": application_cpu,
        "observed_application_affinity": sorted(os.sched_getaffinity(0)),
    }
    return sock, observation


def _run_raw_sync_client(
    *,
    local_edr_ip: str,
    remote_edr_ip: str,
    port: int,
    payload_bytes: int,
    warmup_iterations: int,
    iterations: int,
    application_cpu: int,
    busy_poll_microseconds: int,
) -> JsonObject:
    sock, socket_observation = _connect_raw_tcp(
        local_edr_ip=local_edr_ip,
        remote_edr_ip=remote_edr_ip,
        port=port,
        application_cpu=application_cpu,
        busy_poll_microseconds=busy_poll_microseconds,
    )
    payload = _make_payload(payload_bytes)
    reply = bytearray(REPLY_BYTES)
    latencies: list[float] = []
    total = warmup_iterations + iterations
    measured_start_ns = 0
    measured_end_ns = 0
    try:
        for sequence in range(total):
            _put_sequence(payload, sequence)
            if sequence == warmup_iterations:
                measured_start_ns = time.perf_counter_ns()
            start_ns = time.perf_counter_ns()
            sock.sendall(payload)
            _receive_exact_into(sock, reply)
            if int.from_bytes(reply, byteorder="big", signed=False) != sequence:
                raise StagingTransportProbeError("raw TCP sequence mismatch")
            if sequence >= warmup_iterations:
                latencies.append((time.perf_counter_ns() - start_ns) / 1000.0)
        measured_end_ns = time.perf_counter_ns()
    finally:
        sock.close()
    elapsed_seconds = (measured_end_ns - measured_start_ns) / 1e9
    return {
        "socket_pattern": "TCP_NODELAY_SYNC",
        "window_slots": 1,
        "request_payload_bytes": payload_bytes,
        "reply_payload_bytes": REPLY_BYTES,
        "send_copy_policy": "sendall_from_preallocated_buffer",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "socket": socket_observation,
        "round_trip_latency": latency_summary(latencies),
        "measured_elapsed_seconds": elapsed_seconds,
        "completed_messages_per_second": iterations / elapsed_seconds,
    }


def _run_raw_two_slot_client(
    *,
    local_edr_ip: str,
    remote_edr_ip: str,
    port: int,
    payload_bytes: int,
    warmup_iterations: int,
    iterations: int,
    application_cpu: int,
    busy_poll_microseconds: int,
) -> JsonObject:
    sock, socket_observation = _connect_raw_tcp(
        local_edr_ip=local_edr_ip,
        remote_edr_ip=remote_edr_ip,
        port=port,
        application_cpu=application_cpu,
        busy_poll_microseconds=busy_poll_microseconds,
    )
    slots = (_make_payload(payload_bytes), _make_payload(payload_bytes))
    reply = bytearray(REPLY_BYTES)
    outstanding: dict[int, tuple[int, int]] = {}
    latencies: list[float] = []
    total = warmup_iterations + iterations
    next_sequence = 0
    completed = 0
    measured_start_ns = 0
    measured_end_ns = 0

    def send_slot(slot: int, sequence: int) -> None:
        payload = slots[slot]
        _put_sequence(payload, sequence)
        start_ns = time.perf_counter_ns()
        sock.sendall(payload)
        outstanding[sequence] = (slot, start_ns)

    try:
        while next_sequence < min(2, total):
            send_slot(next_sequence, next_sequence)
            next_sequence += 1
        while completed < total:
            _receive_exact_into(sock, reply)
            sequence = int.from_bytes(reply, byteorder="big", signed=False)
            try:
                slot, start_ns = outstanding.pop(sequence)
            except KeyError as error:
                raise StagingTransportProbeError(
                    f"unexpected raw pipeline reply sequence {sequence}"
                ) from error
            completed += 1
            if sequence >= warmup_iterations:
                if measured_start_ns == 0:
                    measured_start_ns = start_ns
                latencies.append((time.perf_counter_ns() - start_ns) / 1000.0)
            if next_sequence < total:
                send_slot(slot, next_sequence)
                next_sequence += 1
        measured_end_ns = time.perf_counter_ns()
    finally:
        sock.close()
    elapsed_seconds = (measured_end_ns - measured_start_ns) / 1e9
    return {
        "socket_pattern": "TCP_NODELAY_TWO_SLOT",
        "window_slots": 2,
        "request_payload_bytes": payload_bytes,
        "reply_payload_bytes": REPLY_BYTES,
        "send_copy_policy": "sendall_from_two_preallocated_buffers",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "socket": socket_observation,
        "completion_latency": latency_summary(latencies),
        "measured_elapsed_seconds": elapsed_seconds,
        "completed_messages_per_second": iterations / elapsed_seconds,
    }


def _remote_server_command(
    *,
    remote_runtime_python: str,
    remote_edr_ip: str,
    port: int,
    mode: Literal["reqrep", "router"],
    payload_bytes: int,
    message_count: int,
    token: str,
    remote_application_cpu: int,
) -> str:
    return remote_python_command(
        remote_runtime_python,
        ZMQ_SERVER_SOURCE,
        (
            "--bind-ip",
            remote_edr_ip,
            "--port",
            str(port),
            "--mode",
            mode,
            "--payload-bytes",
            str(payload_bytes),
            "--message-count",
            str(message_count),
            "--token",
            token,
            "--application-cpu",
            str(remote_application_cpu),
        ),
    )


def _remote_raw_tcp_server_command(
    *,
    remote_runtime_python: str,
    remote_edr_ip: str,
    port: int,
    payload_bytes: int,
    message_count: int,
    token: str,
    remote_application_cpu: int,
    busy_poll_microseconds: int,
) -> str:
    return remote_python_command(
        remote_runtime_python,
        RAW_TCP_SERVER_SOURCE,
        (
            "--bind-ip",
            remote_edr_ip,
            "--port",
            str(port),
            "--payload-bytes",
            str(payload_bytes),
            "--message-count",
            str(message_count),
            "--token",
            token,
            "--application-cpu",
            str(remote_application_cpu),
            "--busy-poll-microseconds",
            str(busy_poll_microseconds),
        ),
    )


def _owned_remote_processes(
    ssh_target: str,
    token: str,
) -> tuple[int, ...]:
    source = f"""
from pathlib import Path
token = {token!r}.encode()
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdecimal():
        continue
    try:
        command = (entry / "cmdline").read_bytes()
    except OSError:
        continue
    if token in command:
        matches.append(int(entry.name))
print(" ".join(str(pid) for pid in sorted(matches)))
"""
    result = _run_ssh(
        ssh_target,
        remote_python_command("/usr/bin/python3", source),
    )
    return tuple(int(value) for value in result.stdout.split())


def _cleanup_owned_remote_processes(
    ssh_target: str,
    token: str,
) -> tuple[tuple[int, ...], bool]:
    """Terminate only remote processes whose command line contains our token."""
    source = f"""
import os
import signal
import time
from pathlib import Path
token = {token!r}.encode()
self_pid = os.getpid()

def matches():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == self_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if token in command:
            result.append(int(entry.name))
    return sorted(result)

initial = matches()
for pid in initial:
    os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 3.0
while matches() and time.monotonic() < deadline:
    time.sleep(0.05)
survivors = matches()
forced = bool(survivors)
for pid in survivors:
    os.kill(pid, signal.SIGKILL)
print(" ".join(str(pid) for pid in initial) + "|" + ("1" if forced else "0"))
"""
    result = _run_ssh(
        ssh_target,
        remote_python_command("/usr/bin/python3", source),
    )
    initial_text, separator, forced_text = result.stdout.strip().partition("|")
    if not separator or forced_text not in {"0", "1"}:
        raise StagingTransportProbeError(
            "remote owned-process cleanup returned malformed evidence"
        )
    initial = tuple(int(value) for value in initial_text.split())
    return initial, forced_text == "1"


def _wait_remote_server_exit(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> tuple[JsonObject, str]:
    complete = _read_server_event(process, timeout_seconds=timeout_seconds)
    try:
        stdout_tail, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise StagingTransportProbeError(
            "remote transport server did not exit after completing its message count"
        ) from error
    if process.returncode != 0:
        raise StagingTransportProbeError(
            f"remote transport server failed with {process.returncode}: {stderr[-2_000:]}"
        )
    return complete, stdout_tail + stderr


def run_transport_mode(
    *,
    ssh_target: str,
    remote_runtime_python: str,
    remote_edr_ip: str,
    payload: PayloadProbe,
    mode: Literal["reqrep", "router"],
    port: int,
    token: str,
    timeout_seconds: float,
    local_application_cpu: int,
    remote_application_cpu: int,
) -> JsonObject:
    message_count = payload.transport_warmup_iterations + payload.transport_iterations
    command = _remote_server_command(
        remote_runtime_python=remote_runtime_python,
        remote_edr_ip=remote_edr_ip,
        port=port,
        mode=mode,
        payload_bytes=payload.payload_bytes,
        message_count=message_count,
        token=token,
        remote_application_cpu=remote_application_cpu,
    )
    server = subprocess.Popen(
        ("ssh", ssh_target, command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    completed_normally = False
    try:
        ready = _read_server_event(server, timeout_seconds=timeout_seconds)
        if ready.get("event") != "ready" or ready.get("token") != token:
            raise StagingTransportProbeError(
                f"unexpected remote server ready event: {ready}"
            )
        endpoint = f"tcp://{remote_edr_ip}:{port}"
        if mode == "reqrep":
            client = _run_reqrep_client(
                endpoint=endpoint,
                payload_bytes=payload.payload_bytes,
                warmup_iterations=payload.transport_warmup_iterations,
                iterations=payload.transport_iterations,
                application_cpu=local_application_cpu,
            )
        else:
            client = _run_two_slot_client(
                endpoint=endpoint,
                payload_bytes=payload.payload_bytes,
                warmup_iterations=payload.transport_warmup_iterations,
                iterations=payload.transport_iterations,
                application_cpu=local_application_cpu,
            )
        complete, tail = _wait_remote_server_exit(
            server,
            timeout_seconds=timeout_seconds,
        )
        if (
            complete.get("event") != "complete"
            or complete.get("token") != token
            or complete.get("received_messages") != message_count
        ):
            raise StagingTransportProbeError(
                f"unexpected remote server completion event: {complete}"
            )
        survivors = _owned_remote_processes(ssh_target, token)
        if survivors:
            raise StagingTransportProbeError(
                f"owned remote transport processes survived: {survivors}"
            )
        completed_normally = True
        return {
            "mode": mode,
            "port": port,
            "token": token,
            "ready": ready,
            "complete": complete,
            "client": client,
            "forced_cleanup": False,
            "server_output_tail": tail[-1_000:],
            "postflight_owned_remote_processes": [],
        }
    finally:
        local_forced_cleanup = _terminate_process(server)
        cleaned_remote_processes: tuple[int, ...] = ()
        remote_forced_cleanup = False
        if not completed_normally:
            cleaned_remote_processes, remote_forced_cleanup = (
                _cleanup_owned_remote_processes(ssh_target, token)
            )
        if local_forced_cleanup or remote_forced_cleanup:
            raise StagingTransportProbeError(
                "owned transport process required SIGKILL cleanup; "
                f"remote_owned={cleaned_remote_processes}"
            )


def run_raw_tcp_mode(
    *,
    ssh_target: str,
    remote_runtime_python: str,
    local_edr_ip: str,
    remote_edr_ip: str,
    payload: PayloadProbe,
    mode: Literal["sync", "two_slot"],
    port: int,
    token: str,
    timeout_seconds: float,
    local_application_cpu: int,
    remote_application_cpu: int,
    busy_poll_microseconds: int,
) -> JsonObject:
    message_count = payload.transport_warmup_iterations + payload.transport_iterations
    command = _remote_raw_tcp_server_command(
        remote_runtime_python=remote_runtime_python,
        remote_edr_ip=remote_edr_ip,
        port=port,
        payload_bytes=payload.payload_bytes,
        message_count=message_count,
        token=token,
        remote_application_cpu=remote_application_cpu,
        busy_poll_microseconds=busy_poll_microseconds,
    )
    server = subprocess.Popen(
        ("ssh", ssh_target, command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    completed_normally = False
    try:
        ready = _read_server_event(server, timeout_seconds=timeout_seconds)
        if ready.get("event") != "ready" or ready.get("token") != token:
            raise StagingTransportProbeError(
                f"unexpected raw TCP server ready event: {ready}"
            )
        if mode == "sync":
            client = _run_raw_sync_client(
                local_edr_ip=local_edr_ip,
                remote_edr_ip=remote_edr_ip,
                port=port,
                payload_bytes=payload.payload_bytes,
                warmup_iterations=payload.transport_warmup_iterations,
                iterations=payload.transport_iterations,
                application_cpu=local_application_cpu,
                busy_poll_microseconds=busy_poll_microseconds,
            )
        else:
            client = _run_raw_two_slot_client(
                local_edr_ip=local_edr_ip,
                remote_edr_ip=remote_edr_ip,
                port=port,
                payload_bytes=payload.payload_bytes,
                warmup_iterations=payload.transport_warmup_iterations,
                iterations=payload.transport_iterations,
                application_cpu=local_application_cpu,
                busy_poll_microseconds=busy_poll_microseconds,
            )
        complete, tail = _wait_remote_server_exit(
            server,
            timeout_seconds=timeout_seconds,
        )
        if (
            complete.get("event") != "complete"
            or complete.get("token") != token
            or complete.get("received_messages") != message_count
        ):
            raise StagingTransportProbeError(
                f"unexpected raw TCP server completion event: {complete}"
            )
        survivors = _owned_remote_processes(ssh_target, token)
        if survivors:
            raise StagingTransportProbeError(
                f"owned raw TCP server processes survived: {survivors}"
            )
        completed_normally = True
        return {
            "mode": mode,
            "port": port,
            "token": token,
            "busy_poll_microseconds": busy_poll_microseconds,
            "ready": ready,
            "complete": complete,
            "client": client,
            "forced_cleanup": False,
            "server_output_tail": tail[-1_000:],
            "postflight_owned_remote_processes": [],
        }
    finally:
        local_forced_cleanup = _terminate_process(server)
        cleaned_remote_processes: tuple[int, ...] = ()
        remote_forced_cleanup = False
        if not completed_normally:
            cleaned_remote_processes, remote_forced_cleanup = (
                _cleanup_owned_remote_processes(ssh_target, token)
            )
        if local_forced_cleanup or remote_forced_cleanup:
            raise StagingTransportProbeError(
                "owned raw TCP process required SIGKILL cleanup; "
                f"remote_owned={cleaned_remote_processes}"
            )


def select_best_transport(
    reqrep: Mapping[str, object],
    two_slot: Mapping[str, object],
) -> JsonObject:
    req_rate = float(reqrep["completed_messages_per_second"])
    two_rate = float(two_slot["completed_messages_per_second"])
    if req_rate <= 0.0 or two_rate <= 0.0:
        raise StagingTransportProbeError("transport throughput must be positive")
    winner = "two_slot" if two_rate > req_rate else "synchronous"
    return {
        "throughput_winner": winner,
        "synchronous_messages_per_second": req_rate,
        "two_slot_messages_per_second": two_rate,
        "two_slot_throughput_speedup": two_rate / req_rate,
    }


def select_best_application_transport(
    clients: Mapping[str, Mapping[str, object]],
) -> JsonObject:
    if not clients:
        raise StagingTransportProbeError("application transport candidates are empty")
    throughput: dict[str, float] = {}
    p50_latency: dict[str, float] = {}
    for name, client in clients.items():
        throughput[name] = float(client["completed_messages_per_second"])
        latency_value = client.get("round_trip_latency")
        if latency_value is None:
            latency_value = client.get("completion_latency")
        if not isinstance(latency_value, dict):
            raise StagingTransportProbeError(
                f"application transport {name} lacks latency summary"
            )
        p50_latency[name] = float(latency_value["p50_microseconds"])
    return {
        "throughput_winner": max(throughput, key=throughput.__getitem__),
        "latency_winner": min(p50_latency, key=p50_latency.__getitem__),
        "messages_per_second": throughput,
        "p50_microseconds": p50_latency,
    }


def _counter_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    return {name: after[name] - before[name] for name in _NETDEV_COUNTERS}


def _validate_counter_delta(
    *,
    host: str,
    delta: Mapping[str, int],
) -> None:
    if delta["rx_bytes"] <= 0 or delta["tx_bytes"] <= 0:
        raise StagingTransportProbeError(
            f"{host} did not prove bidirectional EDR application traffic"
        )
    unhealthy = {name: delta[name] for name in _HEALTH_COUNTERS if delta[name]}
    if unhealthy:
        raise StagingTransportProbeError(
            f"{host} EDR network health counters changed: {unhealthy}"
        )


def _ensure_local_route(local_edr_ip: str, remote_edr_ip: str) -> JsonObject:
    if socket.gethostbyname(socket.gethostname()) == remote_edr_ip:
        raise StagingTransportProbeError("remote EDR address resolves to local host")
    command = (
        "/usr/sbin/ip",
        "-json",
        "route",
        "get",
        remote_edr_ip,
        "from",
        local_edr_ip,
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise StagingTransportProbeError(
            f"cannot resolve local EDR route: {result.stderr.strip()}"
        )
    try:
        routes = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise StagingTransportProbeError("ip route output is not JSON") from error
    if not isinstance(routes, list) or len(routes) != 1:
        raise StagingTransportProbeError("local EDR route is not singular")
    route = routes[0]
    if not isinstance(route, dict):
        raise StagingTransportProbeError("local EDR route is not an object")
    return cast(JsonObject, route)


def _write_exclusive(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
    except OSError as error:
        raise StagingTransportProbeError(f"cannot create {path}") from error


def _freeze_outputs(result_directory: Path, files: Sequence[Path]) -> None:
    for path in files:
        path.chmod(0o444)
    result_directory.chmod(0o555)


def run_probe(arguments: argparse.Namespace) -> JsonObject:
    initial_local_affinity = tuple(sorted(os.sched_getaffinity(0)))
    result_directory = Path(cast(str, arguments.result_directory)).resolve()
    if result_directory.exists():
        raise StagingTransportProbeError(
            f"result directory already exists: {result_directory}"
        )
    if not 1_024 <= cast(int, arguments.port_base) <= 65_512:
        raise StagingTransportProbeError("port range is outside 1024..65535")
    local_runtime_python = Path(cast(str, arguments.local_runtime_python))
    if Path(sys.executable).resolve() != local_runtime_python.resolve():
        raise StagingTransportProbeError(
            "launch this probe with --local-runtime-python so pyzmq provenance "
            "matches the receipt"
        )

    route = _ensure_local_route(
        cast(str, arguments.local_edr_ip),
        cast(str, arguments.remote_edr_ip),
    )
    if route.get("dev") != cast(str, arguments.local_netdev):
        raise StagingTransportProbeError(
            f"remote EDR route selected {route.get('dev')!r}, expected "
            f"{arguments.local_netdev!r}"
        )
    local_before = _read_netdev_counters(cast(str, arguments.local_netdev))
    remote_before_observation = _read_remote_netdev(
        cast(str, arguments.ssh_target),
        cast(str, arguments.remote_runtime_python),
        cast(str, arguments.remote_netdev),
    )
    remote_before = cast(
        dict[str, int],
        remote_before_observation["counters"],
    )

    cuda_staging = run_cuda_staging_probe(
        ssh_target=cast(str, arguments.ssh_target),
        remote_runtime_python=cast(str, arguments.remote_runtime_python),
        payloads=PAYLOAD_PROBES,
        timeout_seconds=cast(float, arguments.timeout_seconds),
    )

    transport_results: list[JsonValue] = []
    base_token = (
        f"exo-fwuff-staging-{result_directory.name}-{os.getpid()}-{time.time_ns()}"
    )
    for index, payload in enumerate(PAYLOAD_PROBES):
        reqrep = run_transport_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="reqrep",
            port=cast(int, arguments.port_base) + index * 2,
            token=f"{base_token}-{index}-reqrep",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=cast(int, arguments.local_application_cpu),
            remote_application_cpu=cast(int, arguments.remote_application_cpu),
        )
        two_slot = run_transport_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="router",
            port=cast(int, arguments.port_base) + index * 2 + 1,
            token=f"{base_token}-{index}-router",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=cast(int, arguments.local_application_cpu),
            remote_application_cpu=cast(int, arguments.remote_application_cpu),
        )
        raw_port_base = cast(int, arguments.port_base) + 8 + index * 4
        os.sched_setaffinity(0, initial_local_affinity)
        raw_sync = run_raw_tcp_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            local_edr_ip=cast(str, arguments.local_edr_ip),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="sync",
            port=raw_port_base,
            token=f"{base_token}-{index}-raw-sync",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=-1,
            remote_application_cpu=-1,
            busy_poll_microseconds=0,
        )
        os.sched_setaffinity(0, initial_local_affinity)
        raw_two_slot = run_raw_tcp_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            local_edr_ip=cast(str, arguments.local_edr_ip),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="two_slot",
            port=raw_port_base + 1,
            token=f"{base_token}-{index}-raw-two-slot",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=-1,
            remote_application_cpu=-1,
            busy_poll_microseconds=0,
        )
        raw_busy_sync = run_raw_tcp_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            local_edr_ip=cast(str, arguments.local_edr_ip),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="sync",
            port=raw_port_base + 2,
            token=f"{base_token}-{index}-raw-busy-sync",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=cast(int, arguments.local_application_cpu),
            remote_application_cpu=cast(int, arguments.remote_application_cpu),
            busy_poll_microseconds=BUSY_POLL_MICROSECONDS,
        )
        raw_busy_two_slot = run_raw_tcp_mode(
            ssh_target=cast(str, arguments.ssh_target),
            remote_runtime_python=cast(str, arguments.remote_runtime_python),
            local_edr_ip=cast(str, arguments.local_edr_ip),
            remote_edr_ip=cast(str, arguments.remote_edr_ip),
            payload=payload,
            mode="two_slot",
            port=raw_port_base + 3,
            token=f"{base_token}-{index}-raw-busy-two-slot",
            timeout_seconds=cast(float, arguments.timeout_seconds),
            local_application_cpu=cast(int, arguments.local_application_cpu),
            remote_application_cpu=cast(int, arguments.remote_application_cpu),
            busy_poll_microseconds=BUSY_POLL_MICROSECONDS,
        )
        reqrep_client = cast(dict[str, object], reqrep["client"])
        two_slot_client = cast(dict[str, object], two_slot["client"])
        raw_sync_client = cast(dict[str, object], raw_sync["client"])
        raw_two_slot_client = cast(dict[str, object], raw_two_slot["client"])
        raw_busy_sync_client = cast(
            dict[str, object],
            raw_busy_sync["client"],
        )
        raw_busy_two_slot_client = cast(
            dict[str, object],
            raw_busy_two_slot["client"],
        )
        transport_results.append(
            {
                "payload": asdict(payload),
                "pyzmq": {
                    "synchronous": reqrep,
                    "two_slot": two_slot,
                    "comparison": select_best_transport(
                        reqrep_client,
                        two_slot_client,
                    ),
                },
                "raw_tcp_nodelay": {
                    "baseline": {
                        "synchronous": raw_sync,
                        "two_slot": raw_two_slot,
                        "comparison": select_best_transport(
                            raw_sync_client,
                            raw_two_slot_client,
                        ),
                    },
                    "busy_poll_50_microseconds": {
                        "synchronous": raw_busy_sync,
                        "two_slot": raw_busy_two_slot,
                        "comparison": select_best_transport(
                            raw_busy_sync_client,
                            raw_busy_two_slot_client,
                        ),
                    },
                },
                "best_application_transport": select_best_application_transport(
                    {
                        "pyzmq_sync": reqrep_client,
                        "pyzmq_two_slot": two_slot_client,
                        "raw_tcp_sync": raw_sync_client,
                        "raw_tcp_two_slot": raw_two_slot_client,
                        "raw_tcp_busy_sync": raw_busy_sync_client,
                        "raw_tcp_busy_two_slot": raw_busy_two_slot_client,
                    }
                ),
            }
        )

    local_after = _read_netdev_counters(cast(str, arguments.local_netdev))
    remote_after_observation = _read_remote_netdev(
        cast(str, arguments.ssh_target),
        cast(str, arguments.remote_runtime_python),
        cast(str, arguments.remote_netdev),
    )
    remote_after = cast(
        dict[str, int],
        remote_after_observation["counters"],
    )
    local_delta = _counter_delta(local_before, local_after)
    remote_delta = _counter_delta(remote_before, remote_after)
    _validate_counter_delta(host="dwagon", delta=local_delta)
    _validate_counter_delta(host="fwuff", delta=remote_delta)

    source_path = Path(__file__).resolve()
    runtime_path = Path(sys.executable).resolve(strict=True)
    receipt: JsonObject = {
        "schema_version": 1,
        "kind": "glm52_fwuff_pinned_cuda_and_application_edr_probe",
        "status": "passed",
        "run_id": result_directory.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model_launched": False,
            "dwagon_gpu_used": False,
            "fwuff_gpu_used_only_for_cuda_copy_microbenchmarks": True,
            "draft_compute_included": False,
            "target_verification_included": False,
            "transport": (
                "pyzmq plus raw TCP_NODELAY over explicitly routed mlx5_0 IPoIB"
            ),
            "wire_reply": "four-byte proposal sequence",
            "low_latency_policy": {
                "pyzmq_send_policy": "tracked zero-copy from preallocated buffers",
                "raw_tcp_policy": "TCP_NODELAY with preallocated application buffers",
                "busy_poll_variant_microseconds": BUSY_POLL_MICROSECONDS,
                "raw_baseline_local_affinity": list(initial_local_affinity),
                "local_application_cpu": cast(
                    int,
                    arguments.local_application_cpu,
                ),
                "remote_application_cpu": cast(
                    int,
                    arguments.remote_application_cpu,
                ),
                "zmq_io_thread_affinity": "operating_system_default",
            },
        },
        "source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        },
        "local_runtime": {
            "requested_python": str(local_runtime_python),
            "resolved_python": str(runtime_path),
            "resolved_python_sha256": _sha256_file(runtime_path),
            "python_version": sys.version,
            "zmq_version": cast(object, _import_zmq()).__version__,
        },
        "topology": {
            "local_edr_ip": cast(str, arguments.local_edr_ip),
            "remote_edr_ip": cast(str, arguments.remote_edr_ip),
            "local_netdev": cast(str, arguments.local_netdev),
            "remote_netdev": cast(str, arguments.remote_netdev),
            "route": route,
            "local_counters_before": local_before,
            "local_counters_after": local_after,
            "local_counter_delta": local_delta,
            "remote_observation_before": remote_before_observation,
            "remote_observation_after": remote_after_observation,
            "remote_counter_delta": remote_delta,
        },
        "cuda_staging": cuda_staging,
        "transport_payloads": transport_results,
        "lifecycle": {
            "transport_server_count": len(PAYLOAD_PROBES) * 6,
            "forced_cleanup": False,
            "postflight_owned_remote_processes": [],
        },
        "interpretation_guard": (
            "These are isolated staging and application-transport measurements. "
            "They do not yet prove end-to-end remote draft speedup."
        ),
    }

    result_directory.mkdir(parents=True, exist_ok=False)
    receipt_path = result_directory / RESULT_FILENAME
    _write_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    manifest: JsonObject = {
        "schema_version": 1,
        "kind": "immutable_content_manifest",
        "files": [
            {
                "path": RESULT_FILENAME,
                "size_bytes": receipt_path.stat().st_size,
                "sha256": _sha256_file(receipt_path),
            }
        ],
    }
    manifest_path = result_directory / MANIFEST_FILENAME
    _write_exclusive(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _freeze_outputs(result_directory, (receipt_path, manifest_path))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", required=True)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--local-edr-ip", default=DEFAULT_LOCAL_EDR_IP)
    parser.add_argument("--remote-edr-ip", default=DEFAULT_REMOTE_EDR_IP)
    parser.add_argument("--local-netdev", default=DEFAULT_LOCAL_NETDEV)
    parser.add_argument("--remote-netdev", default=DEFAULT_REMOTE_NETDEV)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument(
        "--local-application-cpu",
        type=int,
        default=DEFAULT_LOCAL_APPLICATION_CPU,
    )
    parser.add_argument(
        "--remote-application-cpu",
        type=int,
        default=DEFAULT_REMOTE_APPLICATION_CPU,
    )
    parser.add_argument(
        "--local-runtime-python",
        default=DEFAULT_LOCAL_RUNTIME_PYTHON,
    )
    parser.add_argument(
        "--remote-runtime-python",
        default=DEFAULT_REMOTE_RUNTIME_PYTHON,
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = run_probe(arguments)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "run_id": receipt["run_id"],
                "result_directory": arguments.result_directory,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StagingTransportProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
