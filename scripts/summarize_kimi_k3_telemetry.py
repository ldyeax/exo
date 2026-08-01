#!/usr/bin/env python3
"""Join Kimi K3 benchmark windows to 1 Hz telemetry and summarize them.

Only Python's standard library is required.  The input JSONL files are treated
as immutable receipts: their exact hashes and line-level join provenance are
preserved, and the output JSON is published with an atomic same-directory
replace.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

SUMMARY_SCHEMA: Final[str] = "kimi-k3-telemetry-summary-v1"
PARTIAL_COVERAGE_SUMMARY_SCHEMA: Final[str] = "kimi-k3-telemetry-summary-v2"
TELEMETRY_SCHEMA: Final[str] = "kimi-k3-telemetry-v1"
EXPECTED_PERFORMANCE_RUNS: Final[int] = 5
NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000
CPU_COUNTER_NAMES: Final[tuple[str, ...]] = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)
PACKAGE_RAPL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^intel-rapl:[0-9]+:[^:]+$",
    re.ASCII,
)

type JsonObject = dict[str, object]
type NumericExtractor = Callable[[JsonObject], float | None]


class SummaryError(Exception):
    """An expected, user-actionable summarization failure."""


@dataclass(frozen=True)
class ParsedLine:
    """One decoded JSONL line with its exact raw hash."""

    line_number: int
    payload: JsonObject
    raw_sha256: str


@dataclass(frozen=True)
class FileProvenance:
    """Stable identity and parsing facts for one input file."""

    path: Path
    byte_count: int
    line_count: int
    sha256: str

    def to_json(self) -> JsonObject:
        return {
            "path": str(self.path),
            "byte_count": self.byte_count,
            "line_count": self.line_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PerformanceWindow:
    """One accepted benchmark performance request."""

    run_index: int
    case: str
    sequence: int | None
    source_line_number: int
    source_line_sha256: str
    started_at_utc: str
    start_realtime_ns: int
    end_realtime_ns: int
    elapsed_seconds: float
    response_summary: JsonObject

    def to_json(self) -> JsonObject:
        return {
            "run_index": self.run_index,
            "case": self.case,
            "benchmark_sequence": self.sequence,
            "benchmark_source_line_number": self.source_line_number,
            "benchmark_source_line_sha256": self.source_line_sha256,
            "started_at_utc": self.started_at_utc,
            "start_realtime_ns": self.start_realtime_ns,
            "end_realtime_ns": self.end_realtime_ns,
            "elapsed_seconds": self.elapsed_seconds,
            "response_summary": self.response_summary,
        }


@dataclass(frozen=True)
class TelemetrySample:
    """One telemetry sample and the line that binds it."""

    line_number: int
    realtime_ns: int
    payload: JsonObject
    raw_sha256: str


@dataclass(frozen=True)
class TelemetrySource:
    """Validated samples and provenance for one CLI-labelled source."""

    label: str
    provenance: FileProvenance
    samples: tuple[TelemetrySample, ...]
    host: str
    pid: int
    embedded_label: str
    process_start_jiffies: int | None

    @property
    def realtime_values(self) -> tuple[int, ...]:
        return tuple(sample.realtime_ns for sample in self.samples)

    def provenance_json(self) -> JsonObject:
        return {
            "cli_label": self.label,
            **self.provenance.to_json(),
            "schema": TELEMETRY_SCHEMA,
            "sample_count": len(self.samples),
            "host": self.host,
            "pid": self.pid,
            "embedded_label": self.embedded_label,
            "process_start_jiffies": self.process_start_jiffies,
            "coverage": {
                "first_realtime_ns": self.samples[0].realtime_ns,
                "last_realtime_ns": self.samples[-1].realtime_ns,
                "duration_seconds": (
                    self.samples[-1].realtime_ns - self.samples[0].realtime_ns
                )
                / NANOSECONDS_PER_SECOND,
            },
        }


@dataclass(frozen=True)
class WindowJoin:
    """Telemetry samples bracketing an exact request window."""

    first_sample_index: int
    last_sample_index: int
    sample_count_inside: int
    before: TelemetrySample
    after: TelemetrySample
    maximum_sample_gap_seconds: float

    def to_json(self) -> JsonObject:
        return {
            "sample_count_inside_window": self.sample_count_inside,
            "bracketing_sample_count": self.last_sample_index
            - self.first_sample_index
            + 1,
            "maximum_bracketing_sample_gap_seconds": (self.maximum_sample_gap_seconds),
            "start_boundary": {
                "line_number": self.before.line_number,
                "raw_line_sha256": self.before.raw_sha256,
                "realtime_ns": self.before.realtime_ns,
            },
            "end_boundary": {
                "line_number": self.after.line_number,
                "raw_line_sha256": self.after.raw_sha256,
                "realtime_ns": self.after.realtime_ns,
            },
        }


@dataclass(frozen=True)
class UnavailableSourceWindow:
    """One source/window pair that must not be interpolated across a sample gap."""

    source: TelemetrySource
    window: PerformanceWindow
    before: TelemetrySample
    after: TelemetrySample

    @property
    def gap_seconds(self) -> float:
        return (
            self.after.realtime_ns - self.before.realtime_ns
        ) / NANOSECONDS_PER_SECOND

    def to_json(self) -> JsonObject:
        return {
            "availability": "unavailable",
            "source": self.source.label,
            "host": self.source.host,
            "pid": self.source.pid,
            "reason_code": "no_sample_inside_window",
            "reason": (
                "no telemetry sample exists inside the exact performance "
                "window; interpolation across this gap is disabled"
            ),
            "metrics_computed": False,
            "interpolation_performed": False,
            "sample_count_inside_window": 0,
            "request": {
                "start_realtime_ns": self.window.start_realtime_ns,
                "end_realtime_ns": self.window.end_realtime_ns,
                "elapsed_seconds": self.window.elapsed_seconds,
            },
            "gap": {
                "duration_seconds": self.gap_seconds,
                "seconds_from_before_sample_to_request_start": (
                    self.window.start_realtime_ns - self.before.realtime_ns
                )
                / NANOSECONDS_PER_SECOND,
                "seconds_from_request_end_to_after_sample": (
                    self.after.realtime_ns - self.window.end_realtime_ns
                )
                / NANOSECONDS_PER_SECOND,
                "before": telemetry_boundary_json(self.before),
                "after": telemetry_boundary_json(self.after),
            },
        }


@dataclass(frozen=True)
class SeriesPoint:
    """One numeric counter or gauge observation."""

    realtime_ns: int
    value: float
    line_number: int


@dataclass(frozen=True)
class ClippedSeries:
    """A numeric series interpolated to exact request boundaries."""

    points: tuple[SeriesPoint, ...]
    raw_span: tuple[SeriesPoint, ...]
    start_interpolated: bool
    end_interpolated: bool
    maximum_gap_seconds: float


@dataclass(frozen=True)
class GaugeStatistics:
    """Time-weighted statistics for an instantaneous gauge."""

    minimum: float
    maximum: float
    average: float
    integral: float
    raw_sample_count: int
    maximum_gap_seconds: float
    start_interpolated: bool
    end_interpolated: bool

    def to_json(self, unit: str) -> JsonObject:
        return {
            "unit": unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "time_weighted_average": self.average,
            "raw_sample_count": self.raw_sample_count,
            "maximum_sample_gap_seconds": self.maximum_gap_seconds,
            "start_interpolated": self.start_interpolated,
            "end_interpolated": self.end_interpolated,
        }


@dataclass(frozen=True)
class CounterDelta:
    """A monotonic counter delta estimated at exact request boundaries."""

    start_value: float
    end_value: float
    delta: float
    raw_sample_count: int
    maximum_gap_seconds: float
    start_interpolated: bool
    end_interpolated: bool

    def to_json(self, unit: str) -> JsonObject:
        return {
            "unit": unit,
            "start_value_estimate": self.start_value,
            "end_value_estimate": self.end_value,
            "delta_estimate": self.delta,
            "raw_sample_count": self.raw_sample_count,
            "maximum_sample_gap_seconds": self.maximum_gap_seconds,
            "start_interpolated": self.start_interpolated,
            "end_interpolated": self.end_interpolated,
        }


@dataclass(frozen=True)
class MemoryStatistics:
    """Memory gauges for one source and request."""

    smaps_rss: GaugeStatistics | None
    status_rss: GaugeStatistics | None
    mem_available: GaugeStatistics | None

    def to_json(self) -> JsonObject:
        selected = self.smaps_rss if self.smaps_rss is not None else self.status_rss
        selected_source = (
            "process_smaps_rollup.Rss"
            if self.smaps_rss is not None
            else "process_status.VmRSS"
            if self.status_rss is not None
            else None
        )
        return {
            "selected_process_peak_rss_kibibytes": (
                selected.maximum if selected is not None else None
            ),
            "selected_process_peak_rss_source": selected_source,
            "smaps_rollup_rss": (
                self.smaps_rss.to_json("KiB") if self.smaps_rss is not None else None
            ),
            "status_vm_rss": (
                self.status_rss.to_json("KiB") if self.status_rss is not None else None
            ),
            "mem_available": (
                self.mem_available.to_json("KiB")
                if self.mem_available is not None
                else None
            ),
            "minimum_mem_available_kibibytes": (
                self.mem_available.minimum if self.mem_available is not None else None
            ),
        }


@dataclass(frozen=True)
class CpuStatistics:
    """Host aggregate CPU utilization from /proc/stat deltas."""

    total_jiffies: float
    busy_jiffies: float
    idle_jiffies: float
    utilization_percent: float

    def to_json(self) -> JsonObject:
        return {
            "total_jiffies_delta_estimate": self.total_jiffies,
            "busy_jiffies_delta_estimate": self.busy_jiffies,
            "idle_and_iowait_jiffies_delta_estimate": self.idle_jiffies,
            "utilization_percent": self.utilization_percent,
            "guest_counters_excluded_to_avoid_double_counting": True,
        }


@dataclass(frozen=True)
class ProcessStatistics:
    """Process CPU and fault counter deltas."""

    user_jiffies: CounterDelta | None
    system_jiffies: CounterDelta | None
    process_major_faults: CounterDelta | None
    host_major_faults: CounterDelta | None
    clock_ticks_per_second: int
    duration_seconds: float

    @property
    def total_jiffies(self) -> float | None:
        if self.user_jiffies is None or self.system_jiffies is None:
            return None
        return self.user_jiffies.delta + self.system_jiffies.delta

    @property
    def average_core_equivalents(self) -> float | None:
        total = self.total_jiffies
        if total is None:
            return None
        return total / (self.clock_ticks_per_second * self.duration_seconds)

    def to_json(self) -> JsonObject:
        return {
            "clock_ticks_per_second": self.clock_ticks_per_second,
            "clock_ticks_source": "summarizer os.sysconf(SC_CLK_TCK)",
            "user_jiffies": (
                self.user_jiffies.to_json("jiffies")
                if self.user_jiffies is not None
                else None
            ),
            "system_jiffies": (
                self.system_jiffies.to_json("jiffies")
                if self.system_jiffies is not None
                else None
            ),
            "total_jiffies_delta_estimate": self.total_jiffies,
            "average_cpu_core_equivalents": self.average_core_equivalents,
            "process_major_faults": (
                self.process_major_faults.to_json("faults")
                if self.process_major_faults is not None
                else None
            ),
            "host_pgmajfault": (
                self.host_major_faults.to_json("faults")
                if self.host_major_faults is not None
                else None
            ),
        }


@dataclass(frozen=True)
class TrafficStatistics:
    """Receive/transmit byte deltas and exact-window throughput."""

    rx: CounterDelta | None
    tx: CounterDelta | None
    duration_seconds: float

    def to_json(self) -> JsonObject:
        rx_delta = self.rx.delta if self.rx is not None else None
        tx_delta = self.tx.delta if self.tx is not None else None
        total_delta = (
            rx_delta + tx_delta
            if rx_delta is not None and tx_delta is not None
            else None
        )
        return {
            "rx": self.rx.to_json("bytes") if self.rx is not None else None,
            "tx": self.tx.to_json("bytes") if self.tx is not None else None,
            "rx_bytes_per_second": (
                rx_delta / self.duration_seconds if rx_delta is not None else None
            ),
            "tx_bytes_per_second": (
                tx_delta / self.duration_seconds if tx_delta is not None else None
            ),
            "total_bytes_delta_estimate": total_delta,
            "total_gigabits_per_second": (
                total_delta * 8 / self.duration_seconds / 1_000_000_000
                if total_delta is not None
                else None
            ),
        }


@dataclass(frozen=True)
class GpuStatistics:
    """Gauge and energy statistics for one stable GPU identity."""

    identity: str
    uuid: str | None
    index: int | None
    memory_used: GaugeStatistics | None
    gpu_utilization: GaugeStatistics | None
    memory_utilization: GaugeStatistics | None
    power_draw: GaugeStatistics | None
    temperature: GaugeStatistics | None

    def to_json(self) -> JsonObject:
        return {
            "identity": self.identity,
            "uuid": self.uuid,
            "index": self.index,
            "memory_used": (
                self.memory_used.to_json("MiB")
                if self.memory_used is not None
                else None
            ),
            "peak_memory_used_mibibytes": (
                self.memory_used.maximum if self.memory_used is not None else None
            ),
            "gpu_utilization": (
                self.gpu_utilization.to_json("percent")
                if self.gpu_utilization is not None
                else None
            ),
            "memory_utilization": (
                self.memory_utilization.to_json("percent")
                if self.memory_utilization is not None
                else None
            ),
            "power_draw": (
                self.power_draw.to_json("watts")
                if self.power_draw is not None
                else None
            ),
            "integrated_gpu_joules": (
                self.power_draw.integral if self.power_draw is not None else None
            ),
            "temperature": (
                self.temperature.to_json("degrees_celsius")
                if self.temperature is not None
                else None
            ),
            "peak_temperature_celsius": (
                self.temperature.maximum if self.temperature is not None else None
            ),
        }


@dataclass(frozen=True)
class SourceWindowStatistics:
    """All supported statistics for one source and request window."""

    source: TelemetrySource
    window: PerformanceWindow
    join: WindowJoin
    memory: MemoryStatistics
    cpu: CpuStatistics | None
    process: ProcessStatistics
    network: Mapping[str, TrafficStatistics]
    infiniband: Mapping[str, TrafficStatistics]
    gpus: Mapping[str, GpuStatistics]
    rapl: Mapping[str, CounterDelta]
    omissions: Mapping[str, str]

    def to_json(self) -> JsonObject:
        package_domains = {
            name: metric.to_json("microjoules")
            for name, metric in sorted(self.rapl.items())
            if PACKAGE_RAPL_PATTERN.fullmatch(name)
        }
        package_joules = (
            sum(
                metric.delta
                for name, metric in self.rapl.items()
                if name in package_domains
            )
            / 1_000_000
            if package_domains
            else None
        )
        return {
            "source": self.source.label,
            "host": self.source.host,
            "pid": self.source.pid,
            "join": self.join.to_json(),
            "memory": self.memory.to_json(),
            "cpu": self.cpu.to_json() if self.cpu is not None else None,
            "process": self.process.to_json(),
            "network": {
                name: metric.to_json() for name, metric in sorted(self.network.items())
            },
            "infiniband": {
                name: metric.to_json()
                for name, metric in sorted(self.infiniband.items())
            },
            "gpus": {
                name: metric.to_json() for name, metric in sorted(self.gpus.items())
            },
            "rapl": {
                "domains": {
                    name: metric.to_json("microjoules")
                    for name, metric in sorted(self.rapl.items())
                },
                "package_domains": package_domains,
                "package_energy_joules": package_joules,
                "wrap_policy": (
                    "omit a domain if its counter decreases because telemetry "
                    "does not record max_energy_range_uj"
                ),
            },
            "omissions": [
                {"metric": metric, "reason": reason}
                for metric, reason in sorted(self.omissions.items())
            ],
        }


def telemetry_boundary_json(sample: TelemetrySample) -> JsonObject:
    """Serialize one exact telemetry receipt boundary."""

    return {
        "line_number": sample.line_number,
        "raw_line_sha256": sample.raw_sha256,
        "realtime_ns": sample.realtime_ns,
    }


def require_object(value: object, context: str) -> JsonObject:
    """Narrow a decoded JSON value to an object with string keys."""

    if not isinstance(value, dict):
        raise SummaryError(f"{context} must be a JSON object")
    unknown = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in unknown):
        raise SummaryError(f"{context} contains a non-string JSON key")
    return cast(JsonObject, value)


def optional_object(value: object) -> JsonObject | None:
    """Return a string-keyed object or None without raising."""

    if not isinstance(value, dict):
        return None
    unknown = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in unknown):
        return None
    return cast(JsonObject, value)


def finite_number(value: object) -> float | None:
    """Return a finite JSON number while excluding booleans."""

    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def integer_value(value: object) -> int | None:
    """Return a JSON integer while excluding booleans."""

    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def nested_number(payload: JsonObject, *keys: str) -> float | None:
    """Read a finite number through a sequence of object keys."""

    current: object = payload
    for key in keys:
        mapping = optional_object(current)
        if mapping is None or key not in mapping:
            return None
        current = mapping[key]
    return finite_number(current)


def parse_jsonl(
    path: Path, description: str
) -> tuple[list[ParsedLine], FileProvenance]:
    """Read and hash a stable JSONL input with line-specific diagnostics."""

    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
    except OSError as error:
        raise SummaryError(f"cannot inspect {description} {path}: {error}") from error
    if not resolved.is_file():
        raise SummaryError(f"{description} is not a regular file: {resolved}")

    digest = hashlib.sha256()
    parsed: list[ParsedLine] = []
    try:
        with resolved.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise SummaryError(
                        f"{description} {resolved}:{line_number} is blank"
                    )
                try:
                    text = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise SummaryError(
                        f"{description} {resolved}:{line_number} is not UTF-8"
                    ) from error
                try:
                    decoded = cast(object, json.loads(text))
                except json.JSONDecodeError as error:
                    raise SummaryError(
                        f"{description} {resolved}:{line_number} is invalid JSON: "
                        f"{error}"
                    ) from error
                parsed.append(
                    ParsedLine(
                        line_number=line_number,
                        payload=require_object(
                            decoded,
                            f"{description} {resolved}:{line_number}",
                        ),
                        raw_sha256=hashlib.sha256(raw_line).hexdigest(),
                    )
                )
    except OSError as error:
        raise SummaryError(f"cannot read {description} {resolved}: {error}") from error

    try:
        after = resolved.stat()
    except OSError as error:
        raise SummaryError(
            f"cannot re-inspect {description} {resolved}: {error}"
        ) from error
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    final_fields = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_fields != final_fields:
        raise SummaryError(f"{description} changed while it was being read: {resolved}")
    if not parsed:
        raise SummaryError(f"{description} is empty: {resolved}")

    return parsed, FileProvenance(
        path=resolved,
        byte_count=after.st_size,
        line_count=len(parsed),
        sha256=digest.hexdigest(),
    )


def utc_timestamp_to_ns(value: str, context: str) -> int:
    """Parse an aware ISO 8601 timestamp without floating-point epoch loss."""

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SummaryError(f"{context} is not a valid ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise SummaryError(f"{context} must include a UTC offset")
    utc_value = parsed.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_value - epoch
    return (
        delta.days * 86_400 + delta.seconds
    ) * NANOSECONDS_PER_SECOND + delta.microseconds * 1_000


def benchmark_response_summary(response: JsonObject) -> JsonObject:
    """Keep timing evidence without copying generated text into this receipt."""

    timings = optional_object(response.get("timings"))
    usage = optional_object(response.get("usage"))
    return {
        "time_to_first_token_seconds": response.get("time_to_first_token_seconds"),
        "time_to_first_content_seconds": response.get("time_to_first_content_seconds"),
        "finish_reason": response.get("finish_reason"),
        "timings": timings if timings is not None else {},
        "usage": usage if usage is not None else {},
    }


def validate_completed_benchmark_campaign(
    lines: Sequence[ParsedLine],
    provenance: FileProvenance,
    expected_performance_runs: int,
) -> None:
    """Bind telemetry admission to one complete benchmark campaign."""

    started = [
        line for line in lines if line.payload.get("event") == "campaign_started"
    ]
    completed = [
        line for line in lines if line.payload.get("event") == "campaign_completed"
    ]
    failed = [line for line in lines if line.payload.get("event") == "campaign_failed"]
    if len(started) != 1:
        raise SummaryError(
            "benchmark must contain exactly one campaign_started event; "
            f"found {len(started)} in {provenance.path}"
        )
    if len(completed) != 1:
        raise SummaryError(
            "benchmark must contain exactly one campaign_completed event; "
            f"found {len(completed)} in {provenance.path}"
        )
    if failed:
        raise SummaryError(
            f"benchmark contains campaign_failed at {provenance.path}:"
            f"{failed[0].line_number}"
        )
    if lines[0] is not started[0]:
        raise SummaryError("campaign_started must be the first benchmark event")
    if lines[-1] is not completed[0]:
        raise SummaryError("campaign_completed must be the final benchmark event")

    configuration = require_object(
        started[0].payload.get("configuration"),
        f"campaign_started configuration at {provenance.path}:{started[0].line_number}",
    )
    configured_run_count = integer_value(configuration.get("run_count"))
    if configured_run_count != expected_performance_runs:
        raise SummaryError(
            "benchmark campaign run_count does not match requested telemetry "
            f"summary count: configured={configured_run_count!r}, "
            f"expected={expected_performance_runs}"
        )
    accepted_pairs = integer_value(completed[0].payload.get("accepted_pairs"))
    if accepted_pairs != expected_performance_runs:
        raise SummaryError(
            "campaign_completed accepted_pairs does not match requested telemetry "
            f"summary count: completed={accepted_pairs!r}, "
            f"expected={expected_performance_runs}"
        )


def read_performance_windows(
    path: Path,
    expected_performance_runs: int = EXPECTED_PERFORMANCE_RUNS,
) -> tuple[tuple[PerformanceWindow, ...], FileProvenance]:
    """Extract the expected unique accepted performance result windows."""

    if isinstance(expected_performance_runs, bool) or expected_performance_runs <= 0:
        raise SummaryError("expected performance run count must be positive")

    lines, provenance = parse_jsonl(path, "benchmark JSONL")
    validate_completed_benchmark_campaign(
        lines,
        provenance,
        expected_performance_runs,
    )
    windows: list[PerformanceWindow] = []
    for line in lines:
        event = line.payload
        if (
            event.get("event") != "performance_result"
            or event.get("accepted") is not True
        ):
            continue
        run_index = integer_value(event.get("run_index"))
        if run_index is None:
            raise SummaryError(
                f"accepted performance event at {provenance.path}:"
                f"{line.line_number} lacks an integer run_index"
            )
        case_value = event.get("case")
        case = case_value if isinstance(case_value, str) else ""
        response = require_object(
            event.get("response"),
            f"accepted performance response at {provenance.path}:{line.line_number}",
        )
        started_value = response.get("started_at_utc")
        if not isinstance(started_value, str) or not started_value:
            raise SummaryError(
                f"accepted performance response at {provenance.path}:"
                f"{line.line_number} lacks started_at_utc"
            )
        elapsed = finite_number(response.get("elapsed_seconds"))
        if elapsed is None or elapsed <= 0:
            raise SummaryError(
                f"accepted performance response at {provenance.path}:"
                f"{line.line_number} has unusable elapsed_seconds"
            )
        start_ns = utc_timestamp_to_ns(
            started_value,
            f"{provenance.path}:{line.line_number} response.started_at_utc",
        )
        duration_ns = round(elapsed * NANOSECONDS_PER_SECOND)
        if duration_ns <= 0:
            raise SummaryError(
                f"accepted performance response at {provenance.path}:"
                f"{line.line_number} has a sub-nanosecond duration"
            )
        sequence = integer_value(event.get("sequence"))
        windows.append(
            PerformanceWindow(
                run_index=run_index,
                case=case,
                sequence=sequence,
                source_line_number=line.line_number,
                source_line_sha256=line.raw_sha256,
                started_at_utc=started_value,
                start_realtime_ns=start_ns,
                end_realtime_ns=start_ns + duration_ns,
                elapsed_seconds=elapsed,
                response_summary=benchmark_response_summary(response),
            )
        )

    if len(windows) != expected_performance_runs:
        raise SummaryError(
            "benchmark must contain exactly "
            f"{expected_performance_runs} accepted performance_result events; "
            f"found {len(windows)} in {provenance.path}"
        )
    windows.sort(key=lambda item: item.run_index)
    indices = [window.run_index for window in windows]
    expected_indices = list(range(1, expected_performance_runs + 1))
    if indices != expected_indices:
        raise SummaryError(
            "accepted performance run indices must be exactly "
            f"1..{expected_performance_runs}; found {indices}"
        )
    chronological = sorted(windows, key=lambda item: item.start_realtime_ns)
    for previous, current in zip(chronological, chronological[1:], strict=False):
        if current.start_realtime_ns < previous.end_realtime_ns:
            raise SummaryError(
                f"performance windows {previous.run_index} and "
                f"{current.run_index} overlap"
            )
    return tuple(windows), provenance


def source_has_useful_payload(payload: JsonObject) -> bool:
    """Whether a telemetry sample contains at least one supported source."""

    direct_paths = (
        ("process_stat", "user_jiffies"),
        ("process_stat", "system_jiffies"),
        ("process_stat", "major_faults"),
        ("process_status", "VmRSS"),
        ("process_smaps_rollup", "Rss"),
        ("meminfo", "MemAvailable"),
        ("vmstat", "pgmajfault"),
    )
    if any(nested_number(payload, *path) is not None for path in direct_paths):
        return True
    if any(
        nested_number(payload, "cpu", counter_name) is not None
        for counter_name in CPU_COUNTER_NAMES
    ):
        return True
    for root_key, counter_names in (
        ("network", ("rx_bytes", "tx_bytes")),
        (
            "infiniband",
            ("port_rcv_data_bytes", "port_xmit_data_bytes"),
        ),
    ):
        root = optional_object(payload.get(root_key))
        if root is None:
            continue
        for raw_child in root.values():
            child = optional_object(raw_child)
            if child is not None and any(
                finite_number(child.get(name)) is not None for name in counter_names
            ):
                return True
    rapl = optional_object(payload.get("rapl_energy_uj"))
    if rapl is not None and any(
        finite_number(value) is not None for value in rapl.values()
    ):
        return True
    raw_gpus = payload.get("gpus")
    if not isinstance(raw_gpus, list):
        return False
    for raw_gpu in cast(list[object], raw_gpus):
        gpu = optional_object(raw_gpu)
        if gpu is not None and any(
            finite_number(gpu.get(field)) is not None
            for field in (
                "memory.used",
                "utilization.gpu",
                "utilization.memory",
                "power.draw",
                "temperature.gpu",
            )
        ):
            return True
    return False


def read_telemetry_source(label: str, path: Path) -> TelemetrySource:
    """Read one stable telemetry stream and validate its identity."""

    lines, provenance = parse_jsonl(path, f"telemetry {label!r}")
    samples: list[TelemetrySample] = []
    hosts: set[str] = set()
    pids: set[int] = set()
    embedded_labels: set[str] = set()
    start_jiffies: set[int] = set()
    useful_samples = 0

    for line in lines:
        payload = line.payload
        if payload.get("schema") != TELEMETRY_SCHEMA:
            raise SummaryError(
                f"telemetry {label!r} {provenance.path}:{line.line_number} "
                f"does not use schema {TELEMETRY_SCHEMA!r}"
            )
        realtime_ns = integer_value(payload.get("realtime_ns"))
        if realtime_ns is None or realtime_ns <= 0:
            raise SummaryError(
                f"telemetry {label!r} {provenance.path}:{line.line_number} "
                "lacks a positive integer realtime_ns"
            )
        if samples and realtime_ns <= samples[-1].realtime_ns:
            raise SummaryError(
                f"telemetry {label!r} timestamps are not strictly increasing "
                f"at {provenance.path}:{line.line_number}"
            )
        host = payload.get("host")
        if isinstance(host, str) and host:
            hosts.add(host)
        pid = integer_value(payload.get("pid"))
        if pid is not None and pid > 0:
            pids.add(pid)
        embedded_label = payload.get("label")
        if isinstance(embedded_label, str):
            embedded_labels.add(embedded_label)
        process_stat = optional_object(payload.get("process_stat"))
        if process_stat is not None:
            start = integer_value(process_stat.get("start_jiffies"))
            if start is not None:
                start_jiffies.add(start)
        if source_has_useful_payload(payload):
            useful_samples += 1
        samples.append(
            TelemetrySample(
                line_number=line.line_number,
                realtime_ns=realtime_ns,
                payload=payload,
                raw_sha256=line.raw_sha256,
            )
        )

    if len(samples) < 2:
        raise SummaryError(f"telemetry {label!r} must contain at least two samples")
    if len(hosts) != 1:
        raise SummaryError(
            f"telemetry {label!r} must identify exactly one host; found {sorted(hosts)}"
        )
    if len(pids) != 1:
        raise SummaryError(
            f"telemetry {label!r} must identify exactly one positive PID; found "
            f"{sorted(pids)}"
        )
    if len(embedded_labels) != 1:
        raise SummaryError(
            f"telemetry {label!r} must contain one stable embedded label; "
            f"found {sorted(embedded_labels)}"
        )
    if len(start_jiffies) > 1:
        raise SummaryError(
            f"telemetry {label!r} spans more than one process start identity"
        )
    if useful_samples == 0:
        raise SummaryError(
            f"telemetry {label!r} contains no supported telemetry fields"
        )

    return TelemetrySource(
        label=label,
        provenance=provenance,
        samples=tuple(samples),
        host=next(iter(hosts)),
        pid=next(iter(pids)),
        embedded_label=next(iter(embedded_labels)),
        process_start_jiffies=(next(iter(start_jiffies)) if start_jiffies else None),
    )


def locate_window_samples(
    source: TelemetrySource,
    window: PerformanceWindow,
) -> tuple[int, int, int]:
    """Locate exact bracketing indices and count samples inside a request."""

    times = source.realtime_values
    before_index = bisect.bisect_right(times, window.start_realtime_ns) - 1
    after_index = bisect.bisect_left(times, window.end_realtime_ns)
    if before_index < 0 or after_index >= len(source.samples):
        raise SummaryError(
            f"telemetry {source.label!r} does not bracket performance run "
            f"{window.run_index}: source coverage is "
            f"{times[0]}..{times[-1]}, request is "
            f"{window.start_realtime_ns}..{window.end_realtime_ns}"
        )
    inside_start = bisect.bisect_left(times, window.start_realtime_ns)
    inside_end = bisect.bisect_right(times, window.end_realtime_ns)
    inside_count = inside_end - inside_start
    return before_index, after_index, inside_count


def unavailable_source_window(
    source: TelemetrySource,
    window: PerformanceWindow,
) -> UnavailableSourceWindow | None:
    """Describe a fully bracketed window with no interior telemetry sample."""

    before_index, after_index, inside_count = locate_window_samples(source, window)
    if inside_count > 0:
        return None
    return UnavailableSourceWindow(
        source=source,
        window=window,
        before=source.samples[before_index],
        after=source.samples[after_index],
    )


def join_window(source: TelemetrySource, window: PerformanceWindow) -> WindowJoin:
    """Find source samples bracketing and falling within a request window."""

    before_index, after_index, inside_count = locate_window_samples(source, window)
    if inside_count == 0:
        raise SummaryError(
            f"telemetry {source.label!r} has no sample inside performance run "
            f"{window.run_index}"
        )
    gaps = [
        (source.samples[index + 1].realtime_ns - source.samples[index].realtime_ns)
        / NANOSECONDS_PER_SECOND
        for index in range(before_index, after_index)
    ]
    return WindowJoin(
        first_sample_index=before_index,
        last_sample_index=after_index,
        sample_count_inside=inside_count,
        before=source.samples[before_index],
        after=source.samples[after_index],
        maximum_sample_gap_seconds=max(gaps, default=0.0),
    )


def build_numeric_series(
    source: TelemetrySource,
    extractor: NumericExtractor,
) -> tuple[SeriesPoint, ...]:
    """Extract all finite observations for one metric."""

    points: list[SeriesPoint] = []
    for sample in source.samples:
        value = extractor(sample.payload)
        if value is None:
            continue
        points.append(
            SeriesPoint(
                realtime_ns=sample.realtime_ns,
                value=value,
                line_number=sample.line_number,
            )
        )
    return tuple(points)


def interpolate_boundary(
    series: Sequence[SeriesPoint],
    times: Sequence[int],
    target_ns: int,
) -> tuple[SeriesPoint, bool]:
    """Linearly interpolate one boundary within a numeric series."""

    position = bisect.bisect_left(times, target_ns)
    if position < len(series) and series[position].realtime_ns == target_ns:
        return series[position], False
    if position == 0 or position >= len(series):
        raise SummaryError("numeric series does not cover request boundary")
    left = series[position - 1]
    right = series[position]
    span = right.realtime_ns - left.realtime_ns
    if span <= 0:
        raise SummaryError("numeric series timestamps are not increasing")
    fraction = (target_ns - left.realtime_ns) / span
    return (
        SeriesPoint(
            realtime_ns=target_ns,
            value=left.value + fraction * (right.value - left.value),
            line_number=left.line_number,
        ),
        True,
    )


def clip_series(
    series: Sequence[SeriesPoint],
    window: PerformanceWindow,
) -> ClippedSeries | None:
    """Clip a numeric series to a request, interpolating exact endpoints."""

    if len(series) < 2:
        return None
    times = tuple(point.realtime_ns for point in series)
    try:
        start, start_interpolated = interpolate_boundary(
            series,
            times,
            window.start_realtime_ns,
        )
        end, end_interpolated = interpolate_boundary(
            series,
            times,
            window.end_realtime_ns,
        )
    except SummaryError:
        return None
    interior_start = bisect.bisect_right(times, window.start_realtime_ns)
    interior_end = bisect.bisect_left(times, window.end_realtime_ns)
    points = (start, *series[interior_start:interior_end], end)

    raw_start = max(0, bisect.bisect_right(times, window.start_realtime_ns) - 1)
    raw_end = min(len(series) - 1, bisect.bisect_left(times, window.end_realtime_ns))
    raw_span = tuple(series[raw_start : raw_end + 1])
    gaps = [
        (right.realtime_ns - left.realtime_ns) / NANOSECONDS_PER_SECOND
        for left, right in zip(raw_span, raw_span[1:], strict=False)
    ]
    return ClippedSeries(
        points=tuple(points),
        raw_span=raw_span,
        start_interpolated=start_interpolated,
        end_interpolated=end_interpolated,
        maximum_gap_seconds=max(gaps, default=0.0),
    )


def gauge_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    extractor: NumericExtractor,
) -> GaugeStatistics | None:
    """Calculate time-weighted gauge statistics over an exact request."""

    clipped = clip_series(build_numeric_series(source, extractor), window)
    if clipped is None:
        return None
    integral = 0.0
    for left, right in zip(clipped.points, clipped.points[1:], strict=False):
        duration = (right.realtime_ns - left.realtime_ns) / NANOSECONDS_PER_SECOND
        integral += (left.value + right.value) * 0.5 * duration
    values = [point.value for point in clipped.points]
    return GaugeStatistics(
        minimum=min(values),
        maximum=max(values),
        average=integral / window.elapsed_seconds,
        integral=integral,
        raw_sample_count=len(clipped.raw_span),
        maximum_gap_seconds=clipped.maximum_gap_seconds,
        start_interpolated=clipped.start_interpolated,
        end_interpolated=clipped.end_interpolated,
    )


def counter_delta(
    source: TelemetrySource,
    window: PerformanceWindow,
    extractor: NumericExtractor,
) -> tuple[CounterDelta | None, str | None]:
    """Calculate a monotonic exact-window counter delta."""

    clipped = clip_series(build_numeric_series(source, extractor), window)
    if clipped is None:
        return None, "counter does not have numeric samples bracketing the window"
    for left, right in zip(clipped.raw_span, clipped.raw_span[1:], strict=False):
        if right.value < left.value:
            return (
                None,
                "counter decreased inside its bracketing samples; reset or wrap "
                "modulus is not recorded",
            )
    start_value = clipped.points[0].value
    end_value = clipped.points[-1].value
    if end_value < start_value:
        return None, "interpolated counter decreased across the request"
    return (
        CounterDelta(
            start_value=start_value,
            end_value=end_value,
            delta=end_value - start_value,
            raw_sample_count=len(clipped.raw_span),
            maximum_gap_seconds=clipped.maximum_gap_seconds,
            start_interpolated=clipped.start_interpolated,
            end_interpolated=clipped.end_interpolated,
        ),
        None,
    )


def cpu_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    omissions: dict[str, str],
) -> CpuStatistics | None:
    """Derive aggregate host CPU utilization from /proc/stat counters."""

    deltas: dict[str, float] = {}
    for counter_name in CPU_COUNTER_NAMES:
        metric, reason = counter_delta(
            source,
            window,
            lambda payload, name=counter_name: nested_number(payload, "cpu", name),
        )
        if metric is None:
            omissions["cpu.utilization"] = f"{counter_name}: {reason or 'unavailable'}"
            return None
        deltas[counter_name] = metric.delta
    total = sum(deltas.values())
    idle = deltas["idle"] + deltas["iowait"]
    busy = total - idle
    if total <= 0 or busy < 0:
        omissions["cpu.utilization"] = "non-positive or inconsistent /proc/stat delta"
        return None
    return CpuStatistics(
        total_jiffies=total,
        busy_jiffies=busy,
        idle_jiffies=idle,
        utilization_percent=100 * busy / total,
    )


def process_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    clock_ticks_per_second: int,
    omissions: dict[str, str],
) -> ProcessStatistics:
    """Derive process CPU and major-fault counters."""

    user, user_reason = counter_delta(
        source,
        window,
        lambda payload: nested_number(payload, "process_stat", "user_jiffies"),
    )
    system, system_reason = counter_delta(
        source,
        window,
        lambda payload: nested_number(payload, "process_stat", "system_jiffies"),
    )
    process_faults, process_fault_reason = counter_delta(
        source,
        window,
        lambda payload: nested_number(payload, "process_stat", "major_faults"),
    )
    host_faults, host_fault_reason = counter_delta(
        source,
        window,
        lambda payload: nested_number(payload, "vmstat", "pgmajfault"),
    )
    if user is None:
        omissions["process.user_jiffies"] = user_reason or "unavailable"
    if system is None:
        omissions["process.system_jiffies"] = system_reason or "unavailable"
    if process_faults is None:
        omissions["process.major_faults"] = process_fault_reason or "unavailable"
    if host_faults is None:
        omissions["host.pgmajfault"] = host_fault_reason or "unavailable"
    return ProcessStatistics(
        user_jiffies=user,
        system_jiffies=system,
        process_major_faults=process_faults,
        host_major_faults=host_faults,
        clock_ticks_per_second=clock_ticks_per_second,
        duration_seconds=window.elapsed_seconds,
    )


def discover_nested_names(
    source: TelemetrySource,
    root_key: str,
) -> tuple[str, ...]:
    """Discover stable child-object names under a telemetry root."""

    names: set[str] = set()
    for sample in source.samples:
        root = optional_object(sample.payload.get(root_key))
        if root is not None:
            names.update(root)
    return tuple(sorted(names))


def traffic_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    root_key: str,
    receive_counter: str,
    transmit_counter: str,
    omissions: dict[str, str],
) -> dict[str, TrafficStatistics]:
    """Summarize every network interface or InfiniBand port."""

    result: dict[str, TrafficStatistics] = {}
    for name in discover_nested_names(source, root_key):
        rx, rx_reason = counter_delta(
            source,
            window,
            lambda payload, child=name: nested_number(
                payload,
                root_key,
                child,
                receive_counter,
            ),
        )
        tx, tx_reason = counter_delta(
            source,
            window,
            lambda payload, child=name: nested_number(
                payload,
                root_key,
                child,
                transmit_counter,
            ),
        )
        if rx is None:
            omissions[f"{root_key}.{name}.{receive_counter}"] = (
                rx_reason or "unavailable"
            )
        if tx is None:
            omissions[f"{root_key}.{name}.{transmit_counter}"] = (
                tx_reason or "unavailable"
            )
        if rx is not None or tx is not None:
            result[name] = TrafficStatistics(
                rx=rx,
                tx=tx,
                duration_seconds=window.elapsed_seconds,
            )
    if not result:
        omissions[root_key] = f"no usable {root_key} byte counters"
    return result


def gpu_object_identity(gpu: JsonObject) -> tuple[str, str | None, int | None] | None:
    """Return a stable GPU identity, preferring UUID over index."""

    uuid_value = gpu.get("uuid")
    uuid = uuid_value if isinstance(uuid_value, str) and uuid_value else None
    index = integer_value(gpu.get("index"))
    if uuid is not None:
        return uuid, uuid, index
    if index is not None:
        return f"index:{index}", None, index
    return None


def gpu_objects(payload: JsonObject) -> tuple[JsonObject, ...]:
    """Narrow the telemetry GPU list to JSON objects."""

    raw_gpus = payload.get("gpus")
    if not isinstance(raw_gpus, list):
        return ()
    result: list[JsonObject] = []
    for raw_gpu in cast(list[object], raw_gpus):
        gpu = optional_object(raw_gpu)
        if gpu is not None:
            result.append(gpu)
    return tuple(result)


def discover_gpus(
    source: TelemetrySource,
) -> dict[str, tuple[str | None, int | None]]:
    """Discover stable GPU identities and reject conflicting metadata."""

    identities: dict[str, tuple[str | None, int | None]] = {}
    for sample in source.samples:
        seen_in_sample: set[str] = set()
        for gpu in gpu_objects(sample.payload):
            identity = gpu_object_identity(gpu)
            if identity is None:
                continue
            name, uuid, index = identity
            if name in seen_in_sample:
                raise SummaryError(
                    f"telemetry {source.label!r} repeats GPU {name!r} at "
                    f"line {sample.line_number}"
                )
            seen_in_sample.add(name)
            existing = identities.get(name)
            metadata = (uuid, index)
            if existing is not None and existing != metadata:
                raise SummaryError(
                    f"telemetry {source.label!r} changes metadata for GPU {name!r}"
                )
            identities[name] = metadata
    return identities


def gpu_metric_extractor(identity: str, field: str) -> NumericExtractor:
    """Build an extractor for one GPU identity and nvidia-smi field."""

    def extract(payload: JsonObject) -> float | None:
        for gpu in gpu_objects(payload):
            gpu_identity = gpu_object_identity(gpu)
            if gpu_identity is not None and gpu_identity[0] == identity:
                return finite_number(gpu.get(field))
        return None

    return extract


def gpu_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    omissions: dict[str, str],
) -> dict[str, GpuStatistics]:
    """Summarize all GPUs observed by one telemetry source."""

    result: dict[str, GpuStatistics] = {}
    identities = discover_gpus(source)
    for identity, (uuid, index) in sorted(identities.items()):
        memory = gauge_statistics(
            source,
            window,
            gpu_metric_extractor(identity, "memory.used"),
        )
        utilization = gauge_statistics(
            source,
            window,
            gpu_metric_extractor(identity, "utilization.gpu"),
        )
        memory_utilization = gauge_statistics(
            source,
            window,
            gpu_metric_extractor(identity, "utilization.memory"),
        )
        power = gauge_statistics(
            source,
            window,
            gpu_metric_extractor(identity, "power.draw"),
        )
        temperature = gauge_statistics(
            source,
            window,
            gpu_metric_extractor(identity, "temperature.gpu"),
        )
        metrics = {
            "memory.used": memory,
            "utilization.gpu": utilization,
            "utilization.memory": memory_utilization,
            "power.draw": power,
            "temperature.gpu": temperature,
        }
        for field, metric in metrics.items():
            if metric is None:
                omissions[f"gpus.{identity}.{field}"] = (
                    "numeric samples do not bracket the request"
                )
        result[identity] = GpuStatistics(
            identity=identity,
            uuid=uuid,
            index=index,
            memory_used=memory,
            gpu_utilization=utilization,
            memory_utilization=memory_utilization,
            power_draw=power,
            temperature=temperature,
        )
    if not result:
        omissions["gpus"] = "no stable GPU identity was observed"
    return result


def rapl_statistics(
    source: TelemetrySource,
    window: PerformanceWindow,
    omissions: dict[str, str],
) -> dict[str, CounterDelta]:
    """Summarize monotonic RAPL domains, omitting unsafe wraps."""

    result: dict[str, CounterDelta] = {}
    for domain in discover_nested_names(source, "rapl_energy_uj"):
        metric, reason = counter_delta(
            source,
            window,
            lambda payload, name=domain: nested_number(
                payload,
                "rapl_energy_uj",
                name,
            ),
        )
        if metric is None:
            omissions[f"rapl_energy_uj.{domain}"] = reason or "unavailable"
        else:
            result[domain] = metric
    if not result:
        omissions["rapl_energy_uj"] = "no wrap-safe RAPL domain spans this request"
    return result


def summarize_source_window(
    source: TelemetrySource,
    window: PerformanceWindow,
    clock_ticks_per_second: int,
) -> SourceWindowStatistics:
    """Compute all supported statistics for one source and request."""

    join = join_window(source, window)
    omissions: dict[str, str] = {}
    smaps_rss = gauge_statistics(
        source,
        window,
        lambda payload: nested_number(
            payload,
            "process_smaps_rollup",
            "Rss",
        ),
    )
    status_rss = gauge_statistics(
        source,
        window,
        lambda payload: nested_number(payload, "process_status", "VmRSS"),
    )
    mem_available = gauge_statistics(
        source,
        window,
        lambda payload: nested_number(payload, "meminfo", "MemAvailable"),
    )
    if smaps_rss is None:
        omissions["process_smaps_rollup.Rss"] = (
            "numeric samples do not bracket the request"
        )
    if status_rss is None:
        omissions["process_status.VmRSS"] = "numeric samples do not bracket the request"
    if mem_available is None:
        omissions["meminfo.MemAvailable"] = "numeric samples do not bracket the request"

    return SourceWindowStatistics(
        source=source,
        window=window,
        join=join,
        memory=MemoryStatistics(
            smaps_rss=smaps_rss,
            status_rss=status_rss,
            mem_available=mem_available,
        ),
        cpu=cpu_statistics(source, window, omissions),
        process=process_statistics(
            source,
            window,
            clock_ticks_per_second,
            omissions,
        ),
        network=traffic_statistics(
            source,
            window,
            "network",
            "rx_bytes",
            "tx_bytes",
            omissions,
        ),
        infiniband=traffic_statistics(
            source,
            window,
            "infiniband",
            "port_rcv_data_bytes",
            "port_xmit_data_bytes",
            omissions,
        ),
        gpus=gpu_statistics(source, window, omissions),
        rapl=rapl_statistics(source, window, omissions),
        omissions=omissions,
    )


def aggregate_gauges(
    values: Iterable[tuple[GaugeStatistics | None, float]],
    unit: str,
    expected_count: int,
    *,
    include_integral: bool = False,
) -> JsonObject | None:
    """Aggregate per-window gauge statistics without counting gaps."""

    present = [(metric, duration) for metric, duration in values if metric is not None]
    if not present:
        return None
    duration = sum(item_duration for _, item_duration in present)
    result: JsonObject = {
        "unit": unit,
        "runs_included": len(present),
        "complete_across_all_runs": len(present) == expected_count,
        "measured_duration_seconds": duration,
        "minimum": min(metric.minimum for metric, _ in present),
        "maximum": max(metric.maximum for metric, _ in present),
        "time_weighted_average": (
            sum(metric.average * item_duration for metric, item_duration in present)
            / duration
        ),
        "maximum_sample_gap_seconds": max(
            metric.maximum_gap_seconds for metric, _ in present
        ),
    }
    if include_integral:
        result["integral"] = sum(metric.integral for metric, _ in present)
    return result


def aggregate_traffic(
    metrics: Sequence[SourceWindowStatistics],
    attribute: str,
    expected_count: int,
) -> JsonObject:
    """Aggregate network or InfiniBand deltas across disjoint windows."""

    by_name: dict[str, list[TrafficStatistics]] = {}
    for metric in metrics:
        mapping = metric.network if attribute == "network" else metric.infiniband
        for name, traffic in mapping.items():
            by_name.setdefault(name, []).append(traffic)
    result: JsonObject = {}
    for name, traffic_values in sorted(by_name.items()):
        duration = sum(value.duration_seconds for value in traffic_values)
        rx_values = [value.rx for value in traffic_values if value.rx is not None]
        tx_values = [value.tx for value in traffic_values if value.tx is not None]
        rx_duration = sum(
            value.duration_seconds for value in traffic_values if value.rx is not None
        )
        tx_duration = sum(
            value.duration_seconds for value in traffic_values if value.tx is not None
        )
        rx_delta = sum(value.delta for value in rx_values)
        tx_delta = sum(value.delta for value in tx_values)
        complete_rx = len(rx_values) == expected_count
        complete_tx = len(tx_values) == expected_count
        total_delta = rx_delta + tx_delta if complete_rx and complete_tx else None
        result[name] = {
            "runs_included": len(traffic_values),
            "complete_across_all_runs": len(traffic_values) == expected_count,
            "measured_duration_seconds": duration,
            "rx_bytes_delta_estimate": rx_delta if rx_values else None,
            "tx_bytes_delta_estimate": tx_delta if tx_values else None,
            "rx_measured_duration_seconds": rx_duration,
            "tx_measured_duration_seconds": tx_duration,
            "rx_complete_across_all_runs": complete_rx,
            "tx_complete_across_all_runs": complete_tx,
            "rx_bytes_per_second": (
                rx_delta / rx_duration if rx_duration > 0 else None
            ),
            "tx_bytes_per_second": (
                tx_delta / tx_duration if tx_duration > 0 else None
            ),
            "total_bytes_delta_estimate": total_delta,
            "total_gigabits_per_second": (
                total_delta * 8 / duration / 1_000_000_000
                if total_delta is not None
                else None
            ),
        }
    return result


def aggregate_gpus(
    metrics: Sequence[SourceWindowStatistics],
    expected_count: int,
) -> JsonObject:
    """Aggregate GPU gauges and trapezoidal energy across all windows."""

    identities = sorted({identity for metric in metrics for identity in metric.gpus})
    result: JsonObject = {}
    for identity in identities:
        entries = [
            (metric.gpus.get(identity), metric.window.elapsed_seconds)
            for metric in metrics
        ]
        first = next((entry for entry, _ in entries if entry is not None), None)
        if first is None:
            continue
        result[identity] = {
            "uuid": first.uuid,
            "index": first.index,
            "memory_used": aggregate_gauges(
                (
                    (
                        entry.memory_used if entry is not None else None,
                        duration,
                    )
                    for entry, duration in entries
                ),
                "MiB",
                expected_count,
            ),
            "gpu_utilization": aggregate_gauges(
                (
                    (
                        entry.gpu_utilization if entry is not None else None,
                        duration,
                    )
                    for entry, duration in entries
                ),
                "percent",
                expected_count,
            ),
            "memory_utilization": aggregate_gauges(
                (
                    (
                        entry.memory_utilization if entry is not None else None,
                        duration,
                    )
                    for entry, duration in entries
                ),
                "percent",
                expected_count,
            ),
            "power_draw": aggregate_gauges(
                (
                    (
                        entry.power_draw if entry is not None else None,
                        duration,
                    )
                    for entry, duration in entries
                ),
                "watts",
                expected_count,
                include_integral=True,
            ),
            "temperature": aggregate_gauges(
                (
                    (
                        entry.temperature if entry is not None else None,
                        duration,
                    )
                    for entry, duration in entries
                ),
                "degrees_celsius",
                expected_count,
            ),
        }
        gpu_result = require_object(result[identity], f"GPU aggregate {identity}")
        power_result = optional_object(gpu_result.get("power_draw"))
        gpu_result["integrated_gpu_joules"] = (
            power_result.get("integral") if power_result is not None else None
        )
    return result


def aggregate_rapl(
    metrics: Sequence[SourceWindowStatistics],
    expected_count: int,
) -> JsonObject:
    """Aggregate only domains that remain wrap-safe in every request."""

    domains = sorted({domain for metric in metrics for domain in metric.rapl})
    result: JsonObject = {}
    for domain in domains:
        values = [metric.rapl.get(domain) for metric in metrics]
        present = [value for value in values if value is not None]
        result[domain] = {
            "runs_included": len(present),
            "complete_across_all_runs": len(present) == expected_count,
            "energy_joules": (
                sum(value.delta for value in present) / 1_000_000 if present else None
            ),
        }
    package_values = [
        cast(JsonObject, value).get("energy_joules")
        for domain, value in result.items()
        if PACKAGE_RAPL_PATTERN.fullmatch(domain)
    ]
    numeric_package_values = [
        number
        for value in package_values
        if (number := finite_number(value)) is not None
    ]
    package_complete = bool(numeric_package_values) and all(
        cast(JsonObject, value).get("complete_across_all_runs") is True
        for domain, value in result.items()
        if PACKAGE_RAPL_PATTERN.fullmatch(domain)
    )
    return {
        "domains": result,
        "package_energy_joules": (
            sum(numeric_package_values) if package_complete else None
        ),
        "package_energy_complete_across_all_runs": package_complete,
        "wrap_policy": (
            "a domain is omitted for any window containing a counter decrease; "
            "max_energy_range_uj is absent from telemetry-v1"
        ),
    }


def aggregate_source(
    source: TelemetrySource,
    metrics: Sequence[SourceWindowStatistics],
    expected_count: int | None = None,
    unavailable: Sequence[UnavailableSourceWindow] = (),
) -> JsonObject:
    """Build the all-performance-windows summary for one source."""

    aggregate_expected_count = (
        len(metrics) if expected_count is None else expected_count
    )
    duration = sum(metric.window.elapsed_seconds for metric in metrics)
    cpu_values = [metric.cpu for metric in metrics if metric.cpu is not None]
    cpu_total = sum(metric.total_jiffies for metric in cpu_values)
    cpu_busy = sum(metric.busy_jiffies for metric in cpu_values)
    process_total_values = [
        metric.process.total_jiffies
        for metric in metrics
        if metric.process.total_jiffies is not None
    ]
    process_total = sum(process_total_values)
    process_fault_values = [
        metric.process.process_major_faults.delta
        for metric in metrics
        if metric.process.process_major_faults is not None
    ]
    host_fault_values = [
        metric.process.host_major_faults.delta
        for metric in metrics
        if metric.process.host_major_faults is not None
    ]
    result: JsonObject = {
        "source": source.label,
        "host": source.host,
        "pid": source.pid,
        "performance_window_count": len(metrics),
        "performance_window_duration_seconds": duration,
        "join_quality": {
            "maximum_sample_gap_seconds": max(
                (metric.join.maximum_sample_gap_seconds for metric in metrics),
                default=None,
            )
        },
        "memory": {
            "smaps_rollup_rss": aggregate_gauges(
                (
                    (metric.memory.smaps_rss, metric.window.elapsed_seconds)
                    for metric in metrics
                ),
                "KiB",
                aggregate_expected_count,
            ),
            "status_vm_rss": aggregate_gauges(
                (
                    (metric.memory.status_rss, metric.window.elapsed_seconds)
                    for metric in metrics
                ),
                "KiB",
                aggregate_expected_count,
            ),
            "mem_available": aggregate_gauges(
                (
                    (
                        metric.memory.mem_available,
                        metric.window.elapsed_seconds,
                    )
                    for metric in metrics
                ),
                "KiB",
                aggregate_expected_count,
            ),
        },
        "cpu": {
            "runs_included": len(cpu_values),
            "complete_across_all_runs": (len(cpu_values) == aggregate_expected_count),
            "total_jiffies_delta_estimate": cpu_total if cpu_values else None,
            "busy_jiffies_delta_estimate": cpu_busy if cpu_values else None,
            "utilization_percent": (
                100 * cpu_busy / cpu_total if cpu_total > 0 else None
            ),
        },
        "process": {
            "clock_ticks_per_second": (
                metrics[0].process.clock_ticks_per_second if metrics else None
            ),
            "runs_with_cpu_jiffies": len(process_total_values),
            "total_jiffies_delta_estimate": (
                process_total if process_total_values else None
            ),
            "average_cpu_core_equivalents": (
                process_total / (metrics[0].process.clock_ticks_per_second * duration)
                if metrics and len(process_total_values) == aggregate_expected_count
                else None
            ),
            "process_major_faults_delta_estimate": (
                sum(process_fault_values) if process_fault_values else None
            ),
            "process_major_faults_complete_across_all_runs": (
                len(process_fault_values) == aggregate_expected_count
            ),
            "host_pgmajfault_delta_estimate": (
                sum(host_fault_values) if host_fault_values else None
            ),
            "host_pgmajfault_complete_across_all_runs": (
                len(host_fault_values) == aggregate_expected_count
            ),
        },
        "network": aggregate_traffic(metrics, "network", aggregate_expected_count),
        "infiniband": aggregate_traffic(
            metrics,
            "infiniband",
            aggregate_expected_count,
        ),
        "gpus": aggregate_gpus(metrics, aggregate_expected_count),
        "rapl": aggregate_rapl(metrics, aggregate_expected_count),
        "omitted_metric_occurrences": sum(len(metric.omissions) for metric in metrics),
    }
    if expected_count is not None:
        covered_run_indices = [metric.window.run_index for metric in metrics]
        unavailable_by_run = {
            str(item.window.run_index): {
                "reason_code": "no_sample_inside_window",
                "gap": {
                    "duration_seconds": item.gap_seconds,
                    "before_realtime_ns": item.before.realtime_ns,
                    "after_realtime_ns": item.after.realtime_ns,
                },
            }
            for item in unavailable
        }
        result["coverage"] = {
            "expected_window_count": expected_count,
            "covered_window_count": len(metrics),
            "complete": len(metrics) == expected_count,
            "covered_run_indices": covered_run_indices,
            "unavailable_run_indices": [item.window.run_index for item in unavailable],
            "unavailable_by_run": unavailable_by_run,
            "aggregation_scope": (
                "covered performance windows only; unavailable windows are "
                "neither interpolated nor included"
            ),
        }
        process_result = require_object(result["process"], "process aggregate")
        process_result["average_cpu_core_equivalents_over_covered_windows"] = (
            process_total / (metrics[0].process.clock_ticks_per_second * duration)
            if metrics and process_total_values and duration > 0
            else None
        )
    return result


def run_cluster_totals(
    metrics: Sequence[SourceWindowStatistics],
    expected_source_labels: Sequence[str] | None = None,
) -> JsonObject:
    """Aggregate only cluster metrics that can be safely added across sources."""

    included_source_labels = sorted({metric.source.label for metric in metrics})
    expected_labels = (
        set(included_source_labels)
        if expected_source_labels is None
        else set(expected_source_labels)
    )
    source_coverage_complete = set(included_source_labels) == expected_labels
    gpu_count = 0
    gpu_energy_values: list[float] = []
    for metric in metrics:
        for gpu in metric.gpus.values():
            gpu_count += 1
            if gpu.power_draw is not None:
                gpu_energy_values.append(gpu.power_draw.integral)

    package_domains = [
        value
        for metric in metrics
        for domain, value in metric.rapl.items()
        if PACKAGE_RAPL_PATTERN.fullmatch(domain)
    ]
    sources_with_package = {
        metric.source.label
        for metric in metrics
        if any(PACKAGE_RAPL_PATTERN.fullmatch(domain) for domain in metric.rapl)
    }
    all_source_labels = {metric.source.label for metric in metrics}
    gpu_energy_for_included_sources_complete = (
        gpu_count > 0 and len(gpu_energy_values) == gpu_count
    )
    rapl_for_included_sources_complete = (
        bool(package_domains) and sources_with_package == all_source_labels
    )
    result: JsonObject = {
        "gpu_count": gpu_count,
        "integrated_gpu_joules": (
            sum(gpu_energy_values) if gpu_energy_for_included_sources_complete else None
        ),
        "gpu_energy_complete": (
            gpu_energy_for_included_sources_complete and source_coverage_complete
        ),
        "rapl_package_energy_joules": (
            sum(value.delta for value in package_domains) / 1_000_000
            if (
                package_domains
                and (
                    expected_source_labels is not None
                    or sources_with_package == all_source_labels
                )
            )
            else None
        ),
        "rapl_package_energy_complete": (
            rapl_for_included_sources_complete and source_coverage_complete
        ),
        "energy_scope_note": (
            "GPU energy is trapezoidal nvidia-smi board power. RAPL package "
            "energy is separate and may overlap non-GPU system energy; they are "
            "not combined into one joule total."
        ),
    }
    if expected_source_labels is not None:
        result["source_coverage"] = {
            "expected_source_count": len(expected_labels),
            "included_source_count": len(included_source_labels),
            "complete": source_coverage_complete,
            "included_sources": included_source_labels,
            "excluded_sources": sorted(expected_labels - set(included_source_labels)),
            "totals_scope": (
                "included sources only; excluded source-windows make completeness "
                "flags false"
            ),
        }
        result["gpu_energy_complete_for_included_sources"] = (
            gpu_energy_for_included_sources_complete
        )
        result["rapl_package_energy_complete_for_included_sources"] = (
            rapl_for_included_sources_complete
        )
    return result


def overall_cluster_totals(
    per_run: Sequence[Sequence[SourceWindowStatistics]],
    expected_source_labels: Sequence[str] | None = None,
) -> JsonObject:
    """Sum complete per-run cluster energy without filling missing windows."""

    run_totals = [
        run_cluster_totals(metrics, expected_source_labels) for metrics in per_run
    ]
    gpu_values = [
        finite_number(item.get("integrated_gpu_joules")) for item in run_totals
    ]
    rapl_values = [
        finite_number(item.get("rapl_package_energy_joules")) for item in run_totals
    ]
    result: JsonObject = {
        "integrated_gpu_joules": (
            sum(cast(float, value) for value in gpu_values)
            if all(value is not None for value in gpu_values)
            else None
        ),
        "gpu_energy_complete_across_all_runs": all(
            item.get("gpu_energy_complete") is True for item in run_totals
        ),
        "rapl_package_energy_joules": (
            sum(cast(float, value) for value in rapl_values)
            if all(value is not None for value in rapl_values)
            else None
        ),
        "rapl_package_energy_complete_across_all_runs": all(
            item.get("rapl_package_energy_complete") is True for item in run_totals
        ),
        "energy_scope_note": (
            "GPU board energy and CPU-package RAPL energy are reported "
            "separately and must not be blindly added."
        ),
    }
    if expected_source_labels is not None:
        expected_count = len(expected_source_labels) * len(per_run)
        covered_count = sum(len(metrics) for metrics in per_run)
        result["source_window_coverage"] = {
            "expected_source_window_count": expected_count,
            "covered_source_window_count": covered_count,
            "complete": covered_count == expected_count,
            "coverage_fraction": (
                covered_count / expected_count if expected_count > 0 else None
            ),
            "aggregation_scope": (
                "covered source-windows only; no value is imputed for an "
                "unavailable source-window"
            ),
        }
    return result


def sha256_file(path: Path) -> str | None:
    """Hash a regular file, returning None for unavailable script provenance."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def generated_at_utc() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def atomic_write_json(destination: Path, payload: JsonObject) -> None:
    """Durably publish JSON using an atomic replace in the target directory."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    except OSError as error:
        raise SummaryError(
            f"cannot create temporary output beside {destination}: {error}"
        ) from error

    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except (OSError, ValueError) as error:
        raise SummaryError(
            f"cannot atomically write summary {destination}: {error}"
        ) from error
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


def parse_telemetry_arguments(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    """Parse repeated LABEL=PATH specifications with unique labels."""

    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise SummaryError(f"--telemetry must be LABEL=PATH, received {value!r}")
        if label in labels:
            raise SummaryError(f"duplicate --telemetry label: {label!r}")
        labels.add(label)
        parsed.append((label, Path(raw_path)))
    if not parsed:
        raise SummaryError("at least one --telemetry LABEL=PATH is required")
    return tuple(parsed)


def clock_ticks_per_second() -> int:
    """Read the local POSIX clock tick rate used for core-equivalent estimates."""

    try:
        value = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError) as error:
        raise SummaryError(f"cannot read SC_CLK_TCK: {error}") from error
    if value <= 0:
        raise SummaryError(f"SC_CLK_TCK is not a positive integer: {value!r}")
    return value


def ensure_distinct_paths(
    benchmark: Path,
    telemetry: Sequence[tuple[str, Path]],
    output: Path,
) -> None:
    """Prevent atomically replacing any input receipt."""

    try:
        output_identity = output.resolve()
    except OSError:
        output_identity = output.absolute()
    inputs = [benchmark, *(path for _, path in telemetry)]
    for input_path in inputs:
        try:
            input_identity = input_path.resolve(strict=True)
        except OSError:
            continue
        if input_identity == output_identity:
            raise SummaryError(f"--output must differ from every input file: {output}")


def build_summary(
    benchmark_path: Path,
    telemetry_specs: Sequence[tuple[str, Path]],
    output_path: Path,
    *,
    allow_uncovered_source_windows: bool = False,
    expected_performance_runs: int = EXPECTED_PERFORMANCE_RUNS,
) -> JsonObject:
    """Read, validate, join, and summarize all requested receipts."""

    windows, benchmark_provenance = read_performance_windows(
        benchmark_path,
        expected_performance_runs,
    )
    sources = tuple(
        read_telemetry_source(label, path) for label, path in telemetry_specs
    )
    clock_ticks = clock_ticks_per_second()

    per_run_metrics: list[tuple[SourceWindowStatistics, ...]] = []
    performance_runs: list[JsonObject] = []
    per_source_metrics: dict[str, list[SourceWindowStatistics]] = {
        source.label: [] for source in sources
    }
    per_source_unavailable: dict[str, list[UnavailableSourceWindow]] = {
        source.label: [] for source in sources
    }
    for window in windows:
        metrics_list: list[SourceWindowStatistics] = []
        rendered_telemetry: JsonObject = {}
        for source in sources:
            unavailable = (
                unavailable_source_window(source, window)
                if allow_uncovered_source_windows
                else None
            )
            if unavailable is not None:
                per_source_unavailable[source.label].append(unavailable)
                rendered_telemetry[source.label] = unavailable.to_json()
                continue
            metric = summarize_source_window(source, window, clock_ticks)
            metrics_list.append(metric)
            per_source_metrics[source.label].append(metric)
            metric_json = metric.to_json()
            if allow_uncovered_source_windows:
                metric_json["availability"] = "available"
            rendered_telemetry[source.label] = metric_json
        metrics = tuple(metrics_list)
        per_run_metrics.append(metrics)
        performance_runs.append(
            {
                "benchmark_window": window.to_json(),
                "telemetry": rendered_telemetry,
                "cluster_totals": run_cluster_totals(
                    metrics,
                    (
                        [source.label for source in sources]
                        if allow_uncovered_source_windows
                        else None
                    ),
                ),
            }
        )

    script_path = Path(__file__).resolve()
    return {
        "schema": (
            PARTIAL_COVERAGE_SUMMARY_SCHEMA
            if allow_uncovered_source_windows
            else SUMMARY_SCHEMA
        ),
        "generated_at_utc": generated_at_utc(),
        "provenance": {
            "benchmark": benchmark_provenance.to_json(),
            "telemetry": [source.provenance_json() for source in sources],
            "summarizer": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
                "python_version": sys.version,
                "clock_ticks_per_second": clock_ticks,
                "output_path": str(output_path),
                "allow_uncovered_source_windows": (allow_uncovered_source_windows),
                "expected_performance_runs": expected_performance_runs,
            },
        },
        "coverage_mode": (
            "allow_uncovered_source_windows"
            if allow_uncovered_source_windows
            else "strict"
        ),
        "join_policy": {
            "clock": "UTC realtime_ns",
            "window_start": "response.started_at_utc",
            "window_end": "start + response.elapsed_seconds",
            "counter_boundary_method": "linear interpolation",
            "gauge_boundary_method": "linear interpolation",
            "gauge_average_method": "trapezoidal time weighting",
            "gpu_energy_method": "trapezoidal integration of power.draw watts",
            "overall_scope": (
                "sum or extrema across the disjoint accepted performance "
                "windows only; semantic requests and gaps are excluded"
            ),
            "counter_reset_policy": (
                "omit a metric if a cumulative counter decreases within the "
                "bracketing samples"
            ),
            "rapl_wrap_policy": (
                "omit the affected domain because telemetry-v1 does not record "
                "max_energy_range_uj"
            ),
            "uncovered_source_window_policy": (
                (
                    "record the source-window as unavailable, perform no "
                    "interpolation, and exclude it from run and overall totals"
                )
                if allow_uncovered_source_windows
                else "fail if any source has no sample inside a performance window"
            ),
        },
        "accepted_performance_run_count": len(windows),
        "performance_runs": performance_runs,
        "overall": {
            "performance_window_count": len(windows),
            "performance_window_duration_seconds": sum(
                window.elapsed_seconds for window in windows
            ),
            "telemetry": {
                source.label: aggregate_source(
                    source,
                    per_source_metrics[source.label],
                    (len(windows) if allow_uncovered_source_windows else None),
                    per_source_unavailable[source.label],
                )
                for source in sources
            },
            "cluster_totals": overall_cluster_totals(
                per_run_metrics,
                (
                    [source.label for source in sources]
                    if allow_uncovered_source_windows
                    else None
                ),
            ),
        },
        "limitations": [
            (
                "1 Hz instantaneous gauges cannot reveal peaks between samples; "
                "reported peaks are sampled/interpolated maxima."
            ),
            (
                "Counter boundaries are linear estimates between bracketing "
                "samples and may include error for bursty traffic."
            ),
            (
                "SC_CLK_TCK comes from the summarizer host because telemetry-v1 "
                "does not record it; core-equivalent estimates assume the "
                "telemetry hosts use the same tick rate."
            ),
            (
                "RAPL domains are not summed indiscriminately because child "
                "domains overlap package energy."
            ),
            ("GPU power.draw is nvidia-smi board power, not whole-host power."),
        ],
    }


def parse_arguments() -> argparse.Namespace:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Join accepted Kimi K3 performance windows to one or more "
            "kimi-k3-telemetry-v1 JSONL streams."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="benchmark_kimi_k3.py event JSONL",
    )
    parser.add_argument(
        "--telemetry",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="telemetry stream; repeat once per host/process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="atomically replaced summary JSON",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_PERFORMANCE_RUNS,
        help=(
            "required accepted performance_result count and contiguous run-index "
            f"range (default: {EXPECTED_PERFORMANCE_RUNS})"
        ),
    )
    parser.add_argument(
        "--allow-uncovered-source-windows",
        action="store_true",
        help=(
            "record fully bracketed source-windows with no interior sample as "
            "unavailable instead of failing; never interpolates those windows"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point with concise expected-failure diagnostics."""

    arguments = parse_arguments()
    benchmark = cast(Path, arguments.benchmark)
    telemetry_values = cast(list[str], arguments.telemetry)
    output = cast(Path, arguments.output)
    allow_uncovered_source_windows = cast(
        bool,
        arguments.allow_uncovered_source_windows,
    )
    expected_performance_runs = cast(int, arguments.expected_runs)
    if expected_performance_runs <= 0:
        print(
            "summarize-kimi-k3-telemetry: error: --expected-runs must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        telemetry_specs = parse_telemetry_arguments(telemetry_values)
        ensure_distinct_paths(benchmark, telemetry_specs, output)
        summary = build_summary(
            benchmark,
            telemetry_specs,
            output,
            allow_uncovered_source_windows=allow_uncovered_source_windows,
            expected_performance_runs=expected_performance_runs,
        )
        atomic_write_json(output, summary)
    except SummaryError as error:
        print(f"summarize-kimi-k3-telemetry: error: {error}", file=sys.stderr)
        return 2
    print(
        f"summarize-kimi-k3-telemetry: wrote {output} "
        f"from {len(telemetry_specs)} telemetry source(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
