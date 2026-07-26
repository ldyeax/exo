#!/usr/bin/env python3
"""Measure the registered-host EDR floor for a remote GLM-5.2 MTP drafter.

This is deliberately not an end-to-end speculative-decoding benchmark.  It
uses the installed ``ib_write_lat`` binaries and their registered host-memory
buffers to measure only the wire-sized messages admitted by the first
synchronous ``fwuff`` design.  It does not launch SGLang, allocate target KV
on ``fwuff``, or claim that perftest includes CUDA D2H/H2D and application
doorbell costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
CANONICAL_C1_INPUT_ROWS: Final = 7_744
CANONICAL_C1_OPEN_BYTES: Final = CANONICAL_C1_INPUT_ROWS * FEATURE_ROW_BYTES
TARGET_LAYER_COUNT: Final = 78
TARGET_KV_BYTES_PER_TOKEN_PER_TP2: Final = 179_712
REMOTE_DRAFT_HEADER_WEIGHT_NAMES: Final = (
    "model.embed_tokens.weight",
    "lm_head.qweight",
    "lm_head.scales",
)
REMOTE_DRAFT_LAYER_WEIGHT_NAMES: Final = (
    "model.layers.78.eh_proj.weight",
    "model.layers.78.enorm.weight",
    "model.layers.78.hnorm.weight",
    "model.layers.78.input_layernorm.weight",
    "model.layers.78.mlp.gate.e_score_correction_bias",
    "model.layers.78.mlp.gate.weight",
    "model.layers.78.mlp.shared_experts.down_proj.qweight",
    "model.layers.78.mlp.shared_experts.down_proj.scales",
    "model.layers.78.mlp.shared_experts.gate_proj.qweight",
    "model.layers.78.mlp.shared_experts.gate_proj.scales",
    "model.layers.78.mlp.shared_experts.up_proj.qweight",
    "model.layers.78.mlp.shared_experts.up_proj.scales",
    "model.layers.78.post_attention_layernorm.weight",
    "model.layers.78.self_attn.indexer.k_norm.bias",
    "model.layers.78.self_attn.indexer.k_norm.weight",
    "model.layers.78.self_attn.indexer.weights_proj.weight",
    "model.layers.78.self_attn.indexer.wk.qweight",
    "model.layers.78.self_attn.indexer.wk.scales",
    "model.layers.78.self_attn.indexer.wq_b.qweight",
    "model.layers.78.self_attn.indexer.wq_b.scales",
    "model.layers.78.self_attn.kv_a_layernorm.weight",
    "model.layers.78.self_attn.kv_a_proj_with_mqa.qweight",
    "model.layers.78.self_attn.kv_a_proj_with_mqa.scales",
    "model.layers.78.self_attn.kv_b_proj.kc_qweight",
    "model.layers.78.self_attn.kv_b_proj.kc_scales",
    "model.layers.78.self_attn.kv_b_proj.vc_qweight",
    "model.layers.78.self_attn.kv_b_proj.vc_scales",
    "model.layers.78.self_attn.o_proj.qweight",
    "model.layers.78.self_attn.o_proj.scales",
    "model.layers.78.self_attn.q_a_layernorm.weight",
    "model.layers.78.self_attn.q_a_proj.qweight",
    "model.layers.78.self_attn.q_a_proj.scales",
    "model.layers.78.self_attn.q_b_proj.qweight",
    "model.layers.78.self_attn.q_b_proj.scales",
    "model.layers.78.shared_head.norm.weight",
)
REMOTE_DRAFT_WEIGHT_NAMES: Final = (
    *REMOTE_DRAFT_HEADER_WEIGHT_NAMES,
    *REMOTE_DRAFT_LAYER_WEIGHT_NAMES,
)
REMOTE_DRAFT_HEADER_SHARD: Final = "model-00001-of-00005.safetensors"
REMOTE_DRAFT_LAYER_SHARD: Final = "model-00005-of-00005.safetensors"
REMOTE_DRAFT_SHARD_IDENTITIES: Final = {
    REMOTE_DRAFT_HEADER_SHARD: {
        "sha256": "6510d686a4433ae4b1c85b336cc6b9d34dd81c045348a435f98dbb2e4b39747a",
        "size_bytes": 4_288_811_144,
    },
    REMOTE_DRAFT_LAYER_SHARD: {
        "sha256": "73dd53133a8bc34b9dec226bc489739006539f9a54f031cb697c1e6b53251ff8",
        "size_bytes": 3_006_152_264,
    },
}

DEFAULT_LOCAL_HCA: Final = "mlx5_0"
DEFAULT_REMOTE_HCA: Final = "mlx5_0"
DEFAULT_LOCAL_EDR_IP: Final = "10.44.0.1"
DEFAULT_REMOTE_EDR_IP: Final = "10.44.0.2"
DEFAULT_SSH_TARGET: Final = "fwuff"
DEFAULT_PERFTEST: Final = "/usr/bin/ib_write_lat"
DEFAULT_CONTROL_PORT_BASE: Final = 18_610
RESULT_FILENAME: Final = "glm52-fwuff-edr-speculative-probe.json"
MAXIMUM_CAPTURE_BYTES: Final = 16 * 1024 * 1024

_LATENCY_ROW = re.compile(
    r"^\s*(?P<bytes>[0-9]+)\s+"
    r"(?P<iterations>[0-9]+)\s+"
    r"(?P<minimum>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<maximum>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<typical>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<average>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<standard_deviation>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<p99>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<p999>[0-9]+(?:\.[0-9]+)?)\s*$"
)
_HISTOGRAM_ROW = re.compile(
    r"^\s*(?P<index>[0-9]+),\s*"
    r"(?P<latency>[0-9]+(?:\.[0-9]+)?)\s*$"
)


class SpeculativeEdrProbeError(RuntimeError):
    """Expected fail-closed probe error."""


@dataclass(frozen=True, slots=True)
class PayloadProbe:
    name: str
    message_kind: Literal["PROPOSAL", "ADVANCE", "OPEN"]
    payload_bytes: int
    row_count: int
    candidate_count: int
    default_iterations: int


PAYLOAD_PROBES: Final = (
    PayloadProbe(
        name="proposal_one_int32_id",
        message_kind="PROPOSAL",
        payload_bytes=4,
        row_count=0,
        candidate_count=1,
        default_iterations=2_000,
    ),
    PayloadProbe(
        name="advance_one_bf16_hidden_row",
        message_kind="ADVANCE",
        payload_bytes=FEATURE_ROW_BYTES,
        row_count=1,
        candidate_count=0,
        default_iterations=2_000,
    ),
    PayloadProbe(
        name="advance_two_bf16_hidden_rows",
        message_kind="ADVANCE",
        payload_bytes=2 * FEATURE_ROW_BYTES,
        row_count=2,
        candidate_count=0,
        default_iterations=2_000,
    ),
    PayloadProbe(
        name="open_512_bf16_hidden_rows",
        message_kind="OPEN",
        payload_bytes=OPEN_PREFIX_BYTES,
        row_count=OPEN_PREFIX_ROWS,
        candidate_count=0,
        default_iterations=128,
    ),
    PayloadProbe(
        name="open_canonical_c1_7744_bf16_hidden_rows",
        message_kind="OPEN",
        payload_bytes=CANONICAL_C1_OPEN_BYTES,
        row_count=CANONICAL_C1_INPUT_ROWS,
        candidate_count=0,
        default_iterations=16,
    ),
)


@dataclass(frozen=True, slots=True)
class PerftestSummary:
    payload_bytes: int
    iterations: int
    minimum_microseconds: float
    maximum_microseconds: float
    typical_microseconds: float
    average_microseconds: float
    standard_deviation_microseconds: float
    reported_p99_microseconds: float
    reported_p999_microseconds: float
    histogram_sample_count: int
    histogram_p50_microseconds: float
    histogram_p95_microseconds: float
    histogram_p99_microseconds: float


@dataclass(frozen=True, slots=True)
class HostObservation:
    hostname: str
    hca: str
    hca_bdf: str
    numa_node: int
    netdevs: tuple[str, ...]
    state: str
    physical_state: str
    rate: str
    lid: str
    perftest_path: str
    perftest_sha256: str
    perftest_version: str
    perftest_process_ids: tuple[int, ...]
    counters: Mapping[str, int]


_COUNTERS: Final = (
    "port_xmit_data",
    "port_rcv_data",
    "port_xmit_packets",
    "port_rcv_packets",
    "symbol_error",
    "link_downed",
    "link_error_recovery",
    "port_rcv_errors",
    "port_xmit_discards",
)
_HEALTH_COUNTERS: Final = frozenset(_COUNTERS[4:])


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise SpeculativeEdrProbeError(f"cannot read required path {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SpeculativeEdrProbeError(f"cannot hash required path {path}") from error
    return digest.hexdigest()


def _perftest_process_ids() -> tuple[int, ...]:
    process_ids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
        except OSError:
            continue
        if executable.name == "ib_write_lat":
            process_ids.append(int(entry.name))
    return tuple(sorted(process_ids))


def observe_host(hca: str, perftest_path: str) -> HostObservation:
    hca_root = Path("/sys/class/infiniband") / hca
    device_root = hca_root / "device"
    port_root = hca_root / "ports" / "1"
    if not hca_root.is_dir():
        raise SpeculativeEdrProbeError(f"required HCA is absent: {hca}")
    try:
        hca_bdf = device_root.resolve(strict=True).name
        netdevs = tuple(sorted(path.name for path in (device_root / "net").iterdir()))
    except OSError as error:
        raise SpeculativeEdrProbeError(
            f"cannot resolve HCA topology for {hca}"
        ) from error
    perftest = Path(perftest_path)
    if not perftest.is_file() or not os.access(perftest, os.X_OK):
        raise SpeculativeEdrProbeError(
            f"perftest executable is absent or not executable: {perftest}"
        )
    version = subprocess.run(
        (str(perftest), "--version"),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    version_text = (version.stdout + version.stderr).strip()
    if not version_text.startswith("Version:"):
        raise SpeculativeEdrProbeError(
            f"cannot identify perftest version: {version_text}"
        )
    counters: dict[str, int] = {}
    for name in _COUNTERS:
        raw_value = _read_text(port_root / "counters" / name)
        try:
            counters[name] = int(raw_value)
        except ValueError as error:
            raise SpeculativeEdrProbeError(
                f"HCA counter {name} is not an integer"
            ) from error
    return HostObservation(
        hostname=socket.gethostname(),
        hca=hca,
        hca_bdf=hca_bdf,
        numa_node=int(_read_text(device_root / "numa_node")),
        netdevs=netdevs,
        state=_read_text(port_root / "state"),
        physical_state=_read_text(port_root / "phys_state"),
        rate=_read_text(port_root / "rate"),
        lid=_read_text(port_root / "lid"),
        perftest_path=str(perftest),
        perftest_sha256=_sha256_file(perftest),
        perftest_version=version_text,
        perftest_process_ids=_perftest_process_ids(),
        counters=counters,
    )


def _remote_observation_script(hca: str, perftest_path: str) -> str:
    """Return a self-contained read-only probe; no remote checkout is required."""
    source = f"""
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

hca = {hca!r}
perftest_path = {perftest_path!r}
hca_root = Path("/sys/class/infiniband") / hca
device_root = hca_root / "device"
port_root = hca_root / "ports" / "1"
perftest = Path(perftest_path)
if not hca_root.is_dir() or not perftest.is_file() or not os.access(perftest, os.X_OK):
    raise RuntimeError("required remote HCA or perftest is absent")
digest = hashlib.sha256()
with perftest.open("rb") as file:
    while chunk := file.read(1024 * 1024):
        digest.update(chunk)
version = subprocess.run(
    (str(perftest), "--version"),
    check=False,
    capture_output=True,
    text=True,
    timeout=5,
)
version_text = (version.stdout + version.stderr).strip()
if not version_text.startswith("Version:"):
    raise RuntimeError("cannot identify remote perftest version")
process_ids = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdecimal():
        continue
    try:
        executable = (entry / "exe").resolve(strict=True)
    except OSError:
        continue
    if executable.name == "ib_write_lat":
        process_ids.append(int(entry.name))
counters = {{
    name: int((port_root / "counters" / name).read_text().strip())
    for name in {tuple(_COUNTERS)!r}
}}
observation = {{
    "hostname": socket.gethostname(),
    "hca": hca,
    "hca_bdf": device_root.resolve(strict=True).name,
    "numa_node": int((device_root / "numa_node").read_text().strip()),
    "netdevs": sorted(path.name for path in (device_root / "net").iterdir()),
    "state": (port_root / "state").read_text().strip(),
    "physical_state": (port_root / "phys_state").read_text().strip(),
    "rate": (port_root / "rate").read_text().strip(),
    "lid": (port_root / "lid").read_text().strip(),
    "perftest_path": str(perftest),
    "perftest_sha256": digest.hexdigest(),
    "perftest_version": version_text,
    "perftest_process_ids": sorted(process_ids),
    "counters": counters,
}}
print(json.dumps(observation, sort_keys=True, separators=(",", ":")))
"""
    return source


def host_observation_json(observation: HostObservation) -> JsonObject:
    value = asdict(observation)
    value["netdevs"] = list(observation.netdevs)
    value["perftest_process_ids"] = list(observation.perftest_process_ids)
    value["counters"] = dict(observation.counters)
    return cast(JsonObject, value)


def observe_remote_host(
    ssh_target: str,
    remote_python: str,
    hca: str,
    perftest_path: str,
    timeout_seconds: float,
) -> HostObservation:
    remote_code = _remote_observation_script(hca, perftest_path)
    command = (
        "ssh",
        ssh_target,
        shlex.join((remote_python, "-c", remote_code)),
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise SpeculativeEdrProbeError(
            f"remote host observation failed: {result.stderr.strip()}"
        )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SpeculativeEdrProbeError(
            "remote host observation did not return JSON"
        ) from error
    if not isinstance(raw, dict):
        raise SpeculativeEdrProbeError("remote host observation is not an object")
    counters_value = raw.get("counters")
    if not isinstance(counters_value, dict):
        raise SpeculativeEdrProbeError("remote HCA counters are absent")
    return HostObservation(
        hostname=str(raw.get("hostname")),
        hca=str(raw.get("hca")),
        hca_bdf=str(raw.get("hca_bdf")),
        numa_node=int(raw.get("numa_node")),
        netdevs=tuple(str(item) for item in cast(list[object], raw.get("netdevs"))),
        state=str(raw.get("state")),
        physical_state=str(raw.get("physical_state")),
        rate=str(raw.get("rate")),
        lid=str(raw.get("lid")),
        perftest_path=str(raw.get("perftest_path")),
        perftest_sha256=str(raw.get("perftest_sha256")),
        perftest_version=str(raw.get("perftest_version")),
        perftest_process_ids=tuple(
            int(item) for item in cast(list[object], raw.get("perftest_process_ids"))
        ),
        counters={str(key): int(value) for key, value in counters_value.items()},
    )


def validate_host_observation(
    observation: HostObservation,
    *,
    expected_hostname: str,
    expected_hca: str,
    expected_netdev: str,
) -> None:
    errors: list[str] = []
    if observation.hostname != expected_hostname:
        errors.append(
            f"hostname={observation.hostname!r}, expected {expected_hostname!r}"
        )
    if observation.hca != expected_hca:
        errors.append(f"hca={observation.hca!r}, expected {expected_hca!r}")
    if expected_netdev not in observation.netdevs:
        errors.append(f"netdevs={observation.netdevs!r}, expected {expected_netdev!r}")
    if observation.state != "4: ACTIVE":
        errors.append(f"state={observation.state!r}")
    if observation.physical_state != "5: LinkUp":
        errors.append(f"physical_state={observation.physical_state!r}")
    if observation.rate != "100 Gb/sec (4X EDR)":
        errors.append(f"rate={observation.rate!r}")
    if observation.numa_node < 0:
        errors.append(f"numa_node={observation.numa_node}")
    if observation.perftest_process_ids:
        errors.append(
            f"unowned ib_write_lat processes={observation.perftest_process_ids!r}"
        )
    if errors:
        raise SpeculativeEdrProbeError(
            f"{expected_hostname} EDR preflight rejected: " + "; ".join(errors)
        )


def perftest_arguments(
    perftest_path: str,
    hca: str,
    payload_bytes: int,
    iterations: int,
    control_port: int,
    *,
    peer_ip: str | None,
) -> tuple[str, ...]:
    if payload_bytes <= 0:
        raise SpeculativeEdrProbeError("payload_bytes must be positive")
    if iterations < 5:
        raise SpeculativeEdrProbeError("perftest requires at least five iterations")
    if not 1_024 <= control_port <= 65_535:
        raise SpeculativeEdrProbeError("control port is outside the admitted range")
    positional = () if peer_ip is None else (peer_ip,)
    return (
        perftest_path,
        *positional,
        "-d",
        hca,
        "-i",
        "1",
        "-c",
        "RC",
        "-s",
        str(payload_bytes),
        "-n",
        str(iterations),
        "-p",
        str(control_port),
        "-F",
        "-H",
    )


def _nearest_rank(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        raise SpeculativeEdrProbeError("latency histogram is empty")
    if not 0.0 < percentile <= 1.0:
        raise SpeculativeEdrProbeError("percentile must be in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def parse_perftest_latency(
    output: str,
    *,
    expected_payload_bytes: int,
    expected_iterations: int,
) -> PerftestSummary:
    encoded = output.encode("utf-8")
    if len(encoded) > MAXIMUM_CAPTURE_BYTES:
        raise SpeculativeEdrProbeError("perftest output exceeds its size bound")
    rows = [
        match
        for line in output.splitlines()
        if (match := _LATENCY_ROW.fullmatch(line)) is not None
        and int(match.group("bytes")) == expected_payload_bytes
        and int(match.group("iterations")) == expected_iterations
    ]
    if len(rows) != 1:
        raise SpeculativeEdrProbeError(
            "perftest output lacks one exact summary row "
            f"for {expected_payload_bytes} bytes/{expected_iterations} iterations"
        )
    samples = [
        float(match.group("latency"))
        for line in output.splitlines()
        if (match := _HISTOGRAM_ROW.fullmatch(line)) is not None
    ]
    if len(samples) < 2:
        raise SpeculativeEdrProbeError("perftest output lacks a usable histogram")
    row = rows[0]
    return PerftestSummary(
        payload_bytes=int(row.group("bytes")),
        iterations=int(row.group("iterations")),
        minimum_microseconds=float(row.group("minimum")),
        maximum_microseconds=float(row.group("maximum")),
        typical_microseconds=float(row.group("typical")),
        average_microseconds=float(row.group("average")),
        standard_deviation_microseconds=float(row.group("standard_deviation")),
        reported_p99_microseconds=float(row.group("p99")),
        reported_p999_microseconds=float(row.group("p999")),
        histogram_sample_count=len(samples),
        histogram_p50_microseconds=_nearest_rank(samples, 0.50),
        histogram_p95_microseconds=_nearest_rank(samples, 0.95),
        histogram_p99_microseconds=_nearest_rank(samples, 0.99),
    )


def _ensure_control_port_available(local_edr_ip: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind((local_edr_ip, port))
    except OSError as error:
        raise SpeculativeEdrProbeError(
            f"local EDR control port is unavailable: {local_edr_ip}:{port}"
        ) from error
    finally:
        probe.close()


def _terminate_owned_process(process: subprocess.Popen[str]) -> bool:
    if process.poll() is not None:
        return False
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
        return True
    return False


def run_latency_pair(
    *,
    payload: PayloadProbe,
    iterations: int,
    control_port: int,
    local_hca: str,
    remote_hca: str,
    local_perftest: str,
    remote_perftest: str,
    local_edr_ip: str,
    ssh_target: str,
    timeout_seconds: float,
) -> tuple[str, str, PerftestSummary, PerftestSummary, int, bool]:
    _ensure_control_port_available(local_edr_ip, control_port)
    server_command = perftest_arguments(
        local_perftest,
        local_hca,
        payload.payload_bytes,
        iterations,
        control_port,
        peer_ip=None,
    )
    client_command = perftest_arguments(
        remote_perftest,
        remote_hca,
        payload.payload_bytes,
        iterations,
        control_port,
        peer_ip=local_edr_ip,
    )
    server = subprocess.Popen(
        server_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    forced_cleanup = False
    client_output = ""
    try:
        time.sleep(0.5)
        if server.poll() is not None:
            server_output, _ = server.communicate(timeout=1)
            raise SpeculativeEdrProbeError(
                f"local perftest server exited before the client: {server_output}"
            )
        remote_timeout = max(5, math.ceil(timeout_seconds))
        remote_command = (
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=3",
            f"{remote_timeout}s",
            *client_command,
        )
        client = subprocess.run(
            ("ssh", ssh_target, shlex.join(remote_command)),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 5,
        )
        client_output = client.stdout + client.stderr
        if client.returncode != 0:
            raise SpeculativeEdrProbeError(
                f"remote perftest client failed with {client.returncode}: "
                f"{client_output[-2_000:]}"
            )
        try:
            server_output, _ = server.communicate(timeout=10)
        except subprocess.TimeoutExpired as error:
            raise SpeculativeEdrProbeError(
                "local perftest server did not exit after the remote client"
            ) from error
        if server.returncode != 0:
            raise SpeculativeEdrProbeError(
                f"local perftest server failed with {server.returncode}: "
                f"{server_output[-2_000:]}"
            )
        server_summary = parse_perftest_latency(
            server_output,
            expected_payload_bytes=payload.payload_bytes,
            expected_iterations=iterations,
        )
        client_summary = parse_perftest_latency(
            client_output,
            expected_payload_bytes=payload.payload_bytes,
            expected_iterations=iterations,
        )
        return (
            server_output,
            client_output,
            server_summary,
            client_summary,
            server.pid,
            forced_cleanup,
        )
    finally:
        forced_cleanup = _terminate_owned_process(server) or forced_cleanup


def _counter_deltas(
    before: HostObservation,
    after: HostObservation,
) -> dict[str, int]:
    return {name: after.counters[name] - before.counters[name] for name in _COUNTERS}


def _validate_counter_deltas(
    local_deltas: Mapping[str, int],
    remote_deltas: Mapping[str, int],
) -> None:
    for host, deltas in (("dwagon", local_deltas), ("fwuff", remote_deltas)):
        if deltas["port_xmit_data"] <= 0 or deltas["port_rcv_data"] <= 0:
            raise SpeculativeEdrProbeError(
                f"{host} did not prove bidirectional mlx5_0 payload movement"
            )
        changed_health = {
            name: deltas[name] for name in _HEALTH_COUNTERS if deltas[name] != 0
        }
        if changed_health:
            raise SpeculativeEdrProbeError(
                f"{host} EDR health counters changed: {changed_health}"
            )


def select_remote_draft_weight_names(
    weight_map: Mapping[str, object],
) -> tuple[frozenset[str], frozenset[str]]:
    """Scaffold the standalone selector without changing pinned SGLang yet."""
    names = frozenset(REMOTE_DRAFT_WEIGHT_NAMES)
    missing = [name for name in REMOTE_DRAFT_WEIGHT_NAMES if name not in weight_map]
    if missing:
        raise SpeculativeEdrProbeError(
            f"standalone remote draft selector lacks exact owned weights: {missing[:3]}"
        )
    invalid = [name for name in names if not isinstance(weight_map[name], str)]
    if invalid:
        raise SpeculativeEdrProbeError(
            f"standalone remote draft selector has non-string shards: {invalid[:3]}"
        )
    shards = frozenset(cast(str, weight_map[name]) for name in names)
    expected_shards = frozenset((REMOTE_DRAFT_HEADER_SHARD, REMOTE_DRAFT_LAYER_SHARD))
    if shards != expected_shards:
        raise SpeculativeEdrProbeError(
            "standalone remote draft selector differs from immutable shard pair: "
            f"{sorted(shards)}"
        )
    return names, shards


def transfer_contract() -> JsonObject:
    return {
        "target": {
            "host": "dwagon",
            "tensor_parallel_size": 2,
            "causal_layer_count": TARGET_LAYER_COUNT,
            "verifier_and_sampling_remain_local": True,
        },
        "draft": {
            "host": "fwuff",
            "tensor_parallel_size": 1,
            "physical_layer_index": 78,
            "draft_kv_and_tentative_state_remain_remote": True,
            "next_admission_contract": {
                "status": "blocked_pending_dedicated_remote_loader",
                "amx_artifact_logical_partition_slots": [0, 1],
                "physical_numa_node_map": [0, 0],
                "kt_threadpool_count": 2,
                "kt_cpuinfer_threads": 60,
                "physical_cores_per_subpool": 30,
                "shared_host_weights": False,
                "standalone_weight_selector": {
                    "existing_local_selector_is_sufficient": False,
                    "selected_tensor_count": len(REMOTE_DRAFT_WEIGHT_NAMES),
                    "exact_names": list(REMOTE_DRAFT_WEIGHT_NAMES),
                    "required_shards": {
                        name: dict(identity)
                        for name, identity in REMOTE_DRAFT_SHARD_IDENTITIES.items()
                    },
                    "loader_behavior": (
                        "construct owned embedding and compact W8 LM head with no "
                        "target shared-module context"
                    ),
                },
                "reason": (
                    "the current local MTP admission requires TP2 and distinct "
                    "physical NUMA nodes; fwuff must retain both immutable AMX "
                    "logical partitions while mapping both disjoint subpools to "
                    "its sole NUMA node"
                ),
            },
        },
        "wire": {
            "activation_dtype": "BF16",
            "hidden_size": HIDDEN_SIZE,
            "feature_row_bytes": FEATURE_ROW_BYTES,
            "proposal_dtype": "int32",
            "allowed_payloads": [
                {
                    "name": probe.name,
                    "message_kind": probe.message_kind,
                    "payload_bytes": probe.payload_bytes,
                    "row_count": probe.row_count,
                    "candidate_count": probe.candidate_count,
                }
                for probe in PAYLOAD_PROBES
            ],
            "forbidden_payloads": [
                "target_kv_cache",
                "draft_kv_cache",
                "full_vocabulary_logits",
                "full_vocabulary_probabilities",
                "target_model_weights",
            ],
            "target_kv_bytes_per_token_per_tp2_negative_control": (
                TARGET_KV_BYTES_PER_TOKEN_PER_TP2
            ),
        },
        "measurement_scope": {
            "included": "registered host-memory RC write latency on mlx5_0",
            "excluded": [
                "CUDA D2H copy",
                "CUDA H2D copy",
                "CUDA event wait",
                "application descriptor validation",
                "draft compute",
                "target verification",
            ],
            "admission_meaning": (
                "network lower bound only; cannot admit synchronous remote drafting"
            ),
        },
    }


def _write_text_exclusive(path: Path, value: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(value)
    except OSError as error:
        raise SpeculativeEdrProbeError(
            f"cannot create immutable output {path}"
        ) from error


def run_probe(arguments: argparse.Namespace) -> JsonObject:
    result_directory = Path(cast(str, arguments.result_directory)).resolve()
    if result_directory.exists():
        raise SpeculativeEdrProbeError(
            f"result directory already exists: {result_directory}"
        )
    ports = tuple(
        cast(int, arguments.control_port_base) + index
        for index in range(len(PAYLOAD_PROBES))
    )
    if ports[-1] > 65_535:
        raise SpeculativeEdrProbeError("control port range exceeds 65535")

    local_before = observe_host(
        cast(str, arguments.local_hca),
        cast(str, arguments.local_perftest),
    )
    remote_before = observe_remote_host(
        cast(str, arguments.ssh_target),
        cast(str, arguments.remote_python),
        cast(str, arguments.remote_hca),
        cast(str, arguments.remote_perftest),
        cast(float, arguments.timeout_seconds),
    )
    validate_host_observation(
        local_before,
        expected_hostname="dwagon",
        expected_hca=cast(str, arguments.local_hca),
        expected_netdev="ibs5",
    )
    validate_host_observation(
        remote_before,
        expected_hostname="fwuff",
        expected_hca=cast(str, arguments.remote_hca),
        expected_netdev="ibs2",
    )

    result_directory.mkdir(parents=True, exist_ok=False)
    probe_receipts: list[JsonValue] = []
    forced_cleanup = False
    for index, payload in enumerate(PAYLOAD_PROBES):
        if payload.name == "open_canonical_c1_7744_bf16_hidden_rows":
            iterations = cast(int, arguments.canonical_prefix_iterations)
        elif payload.message_kind == "OPEN":
            iterations = cast(int, arguments.prefix_iterations)
        else:
            iterations = cast(int, arguments.iterations)
        (
            server_output,
            client_output,
            server_summary,
            client_summary,
            server_pid,
            probe_forced_cleanup,
        ) = run_latency_pair(
            payload=payload,
            iterations=iterations,
            control_port=ports[index],
            local_hca=cast(str, arguments.local_hca),
            remote_hca=cast(str, arguments.remote_hca),
            local_perftest=cast(str, arguments.local_perftest),
            remote_perftest=cast(str, arguments.remote_perftest),
            local_edr_ip=cast(str, arguments.local_edr_ip),
            ssh_target=cast(str, arguments.ssh_target),
            timeout_seconds=cast(float, arguments.timeout_seconds),
        )
        forced_cleanup = forced_cleanup or probe_forced_cleanup
        server_log = result_directory / f"{payload.name}-dwagon-server.log"
        client_log = result_directory / f"{payload.name}-fwuff-client.log"
        _write_text_exclusive(server_log, server_output)
        _write_text_exclusive(client_log, client_output)
        probe_receipts.append(
            {
                "payload": asdict(payload),
                "iterations": iterations,
                "control_port": ports[index],
                "owned_local_server_pid": server_pid,
                "server": asdict(server_summary),
                "client": asdict(client_summary),
                "server_log": {
                    "path": str(server_log),
                    "sha256": _sha256_file(server_log),
                },
                "client_log": {
                    "path": str(client_log),
                    "sha256": _sha256_file(client_log),
                },
            }
        )

    local_after = observe_host(
        cast(str, arguments.local_hca),
        cast(str, arguments.local_perftest),
    )
    remote_after = observe_remote_host(
        cast(str, arguments.ssh_target),
        cast(str, arguments.remote_python),
        cast(str, arguments.remote_hca),
        cast(str, arguments.remote_perftest),
        cast(float, arguments.timeout_seconds),
    )
    local_deltas = _counter_deltas(local_before, local_after)
    remote_deltas = _counter_deltas(remote_before, remote_after)
    validate_host_observation(
        local_after,
        expected_hostname="dwagon",
        expected_hca=cast(str, arguments.local_hca),
        expected_netdev="ibs5",
    )
    validate_host_observation(
        remote_after,
        expected_hostname="fwuff",
        expected_hca=cast(str, arguments.remote_hca),
        expected_netdev="ibs2",
    )
    _validate_counter_deltas(local_deltas, remote_deltas)
    if forced_cleanup:
        raise SpeculativeEdrProbeError(
            "an owned perftest server required forced cleanup"
        )
    source_path = Path(__file__).resolve()
    receipt: JsonObject = {
        "schema_version": 1,
        "kind": "glm52_fwuff_registered_host_edr_speculative_probe",
        "status": "passed",
        "run_id": result_directory.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        },
        "transfer_contract": transfer_contract(),
        "topology": {
            "local_edr_ip": cast(str, arguments.local_edr_ip),
            "remote_edr_ip": cast(str, arguments.remote_edr_ip),
            "local": {
                "before": host_observation_json(local_before),
                "after": host_observation_json(local_after),
                "counter_deltas": local_deltas,
            },
            "remote": {
                "before": host_observation_json(remote_before),
                "after": host_observation_json(remote_after),
                "counter_deltas": remote_deltas,
            },
        },
        "probes": probe_receipts,
        "lifecycle": {
            "owned_server_count": len(PAYLOAD_PROBES),
            "forced_cleanup": False,
            "postflight_perftest_processes": {
                "local": list(local_after.perftest_process_ids),
                "remote": list(remote_after.perftest_process_ids),
            },
        },
        "next_gate": (
            "build a dedicated TP1 remote-draft loader that attests logical AMX "
            "partition slots [0,1] -> physical NUMA map [0,0], then add measured "
            "pinned CUDA D2H/H2D plus event/doorbell costs and compare the total "
            "with isolated fwuff layer-78 draft compute"
        ),
    }
    receipt_path = result_directory / RESULT_FILENAME
    _write_text_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--local-hca", default=DEFAULT_LOCAL_HCA)
    parser.add_argument("--remote-hca", default=DEFAULT_REMOTE_HCA)
    parser.add_argument("--local-edr-ip", default=DEFAULT_LOCAL_EDR_IP)
    parser.add_argument("--remote-edr-ip", default=DEFAULT_REMOTE_EDR_IP)
    parser.add_argument("--local-perftest", default=DEFAULT_PERFTEST)
    parser.add_argument("--remote-perftest", default=DEFAULT_PERFTEST)
    parser.add_argument("--remote-python", default="/usr/bin/python3")
    parser.add_argument(
        "--control-port-base", type=int, default=DEFAULT_CONTROL_PORT_BASE
    )
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--prefix-iterations", type=int, default=128)
    parser.add_argument("--canonical-prefix-iterations", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.print_contract:
        print(json.dumps(transfer_contract(), indent=2, sort_keys=True))
        return 0
    if arguments.result_directory is None:
        raise SpeculativeEdrProbeError(
            "--result-directory is required unless --print-contract is used"
        )
    receipt = run_probe(arguments)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpeculativeEdrProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
