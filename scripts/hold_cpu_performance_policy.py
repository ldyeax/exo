#!/usr/bin/env python3
"""Hold the host CPU policy at performance/performance until asked to stop.

This process intentionally launches no workload.  It enters the existing
crash-recoverable CPU policy transaction, waits for SIGINT, SIGTERM, or an
optional stop file, then restores the original policy and publishes an atomic
JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Final, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))

from scripts.cpu_performance_policy import (  # noqa: E402
    CpuPerformancePolicySession,
    CpuPerformancePolicyTransactionConfig,
    cpu_performance_policy,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type SignalHandler = (
    signal.Handlers | int | Callable[[int, FrameType | None], object] | None
)

_SCHEMA_VERSION: Final = 1
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
_DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.25
_HOLDER_PATH: Final = Path(__file__).resolve()
_POLICY_HELPER_PATH: Final = _HOLDER_PATH.with_name("cpu_performance_policy.py")


class CpuPerformancePolicyHolderError(RuntimeError):
    """Raised when holder configuration or evidence is invalid."""


@dataclass(frozen=True, slots=True)
class HolderConfig:
    output_path: Path
    stop_file: Path | None
    poll_interval_seconds: float
    transaction: CpuPerformancePolicyTransactionConfig
    dry_run: bool


@dataclass(slots=True)
class _StopState:
    stop_requested: bool = False
    first_signal_number: int | None = None
    signal_count: int = 0
    accepting_signals: bool = True

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        """Latch scalars only; Python signal handlers must not take locks."""

        if not self.accepting_signals:
            return
        if self.first_signal_number is None:
            self.first_signal_number = signal_number
        self.signal_count += 1
        self.stop_requested = True


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_stat = path.parent.stat()
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise NotADirectoryError(path.parent)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)

    contents = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output_file:
            written = output_file.write(contents)
            if written != len(contents):
                raise OSError(
                    f"short receipt write: wrote {written} of {len(contents)} bytes"
                )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parse_lock_timeout(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 5.0:
        raise argparse.ArgumentTypeError(
            "lock timeout must be finite and between 0 and 5 seconds"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    defaults = CpuPerformancePolicyTransactionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Hold all discovered CPU policies at performance/performance, then "
            "restore them after SIGINT, SIGTERM, or an optional stop file."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="required JSON evidence receipt path",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="stop cleanly when this regular file appears",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
        help=(
            f"stop-file polling interval (default: {_DEFAULT_POLL_INTERVAL_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=defaults.lock_path,
        help=f"shared policy transaction lock (default: {defaults.lock_path})",
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=defaults.journal_path,
        help=f"shared recovery journal (default: {defaults.journal_path})",
    )
    parser.add_argument(
        "--lock-timeout-seconds",
        type=_parse_lock_timeout,
        default=defaults.lock_timeout_seconds,
        help="bounded transaction lock wait in seconds, from 0 to 5 (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and atomic receipt writing without using sysfs",
    )
    return parser


def _config_from_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> HolderConfig:
    output_path = _absolute(cast(Path, arguments.output))
    raw_stop_file = cast(Path | None, arguments.stop_file)
    stop_file = None if raw_stop_file is None else _absolute(raw_stop_file)
    poll_interval_seconds = cast(float, arguments.poll_interval_seconds)
    lock_path = _absolute(cast(Path, arguments.lock_path))
    journal_path = _absolute(cast(Path, arguments.journal_path))
    lock_timeout_seconds = cast(float, arguments.lock_timeout_seconds)

    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0.0:
        parser.error("--poll-interval-seconds must be finite and positive")
    if output_path == output_path.parent:
        parser.error("--output must name a file")
    role_paths = {
        "output": output_path.resolve(strict=False),
        "lock": lock_path.resolve(strict=False),
        "journal": journal_path.resolve(strict=False),
    }
    if stop_file is not None:
        role_paths["stop-file"] = stop_file.resolve(strict=False)
    names_by_path: dict[Path, list[str]] = {}
    for role_name, role_path in role_paths.items():
        names_by_path.setdefault(role_path, []).append(role_name)
    collisions = [
        "/".join(role_names)
        for role_names in names_by_path.values()
        if len(role_names) > 1
    ]
    if collisions:
        parser.error(
            "holder safety paths must be distinct after resolving symlinks: "
            + ", ".join(collisions)
        )

    return HolderConfig(
        output_path=output_path,
        stop_file=stop_file,
        poll_interval_seconds=poll_interval_seconds,
        transaction=CpuPerformancePolicyTransactionConfig(
            lock_path=lock_path,
            journal_path=journal_path,
            lock_timeout_seconds=lock_timeout_seconds,
        ),
        dry_run=cast(bool, arguments.dry_run),
    )


def _configuration_receipt(config: HolderConfig) -> JsonObject:
    return {
        "output_path": str(config.output_path),
        "stop_file": None if config.stop_file is None else str(config.stop_file),
        "poll_interval_seconds": config.poll_interval_seconds,
        "transaction": {
            "lock_path": str(config.transaction.lock_path),
            "journal_path": str(config.transaction.journal_path),
            "lock_timeout_seconds": config.transaction.lock_timeout_seconds,
            "lock_poll_interval_seconds": (
                config.transaction.lock_poll_interval_seconds
            ),
        },
        "dry_run": config.dry_run,
    }


def _stop_file_exists(path: Path) -> bool:
    try:
        observed = path.stat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(observed.st_mode):
        raise CpuPerformancePolicyHolderError(
            f"stop path exists but is not a regular file: {path}"
        )
    return True


def _wait_for_stop(
    stop_state: _StopState,
    stop_file: Path | None,
    poll_interval_seconds: float,
) -> tuple[str, bool, float]:
    while True:
        if stop_state.stop_requested:
            stop_file_observed = stop_file is not None and _stop_file_exists(stop_file)
            return "signal", stop_file_observed, time.monotonic()
        if stop_file is not None and _stop_file_exists(stop_file):
            return "stop_file", True, time.monotonic()
        time.sleep(poll_interval_seconds)


def _signal_receipt(stop_state: _StopState) -> JsonObject:
    signal_number = stop_state.first_signal_number
    signal_name: str | None = None
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = None
    return {
        "first_signal_number": signal_number,
        "first_signal_name": signal_name,
        "count": stop_state.signal_count,
    }


def _policy_evidence_verified(evidence: JsonObject) -> bool:
    failures = evidence.get("failures")
    serialization = evidence.get("serialization")
    snapshots = evidence.get("snapshots")
    return (
        evidence.get("lifecycle") == "restored"
        and evidence.get("application_verified") is True
        and evidence.get("restoration_verified") is True
        and isinstance(failures, list)
        and not failures
        and isinstance(serialization, dict)
        and serialization.get("lock_acquired") is False
        and serialization.get("lock_released") is True
        and serialization.get("journal_published") is False
        and serialization.get("journal_removed") is True
        and isinstance(snapshots, dict)
        and {"before", "active", "performance_after", "restored"}.issubset(snapshots)
    )


def _failure_receipt(error: BaseException) -> JsonObject:
    return {
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _base_receipt(
    config: HolderConfig,
    *,
    started_at_utc: str,
    started_monotonic: float,
) -> JsonObject:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "cpu_performance_policy_holder",
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "configuration": _configuration_receipt(config),
        "holder": {
            "path": str(_HOLDER_PATH),
            "sha256": _sha256_file(_HOLDER_PATH),
        },
        "policy_helper": {
            "path": str(_POLICY_HELPER_PATH),
            "sha256": _sha256_file(_POLICY_HELPER_PATH),
        },
        "stop": None,
        "policy": None,
        "failure": None,
        "_started_monotonic": started_monotonic,
    }


def _complete_receipt(
    receipt: JsonObject,
    *,
    started_monotonic: float,
) -> None:
    receipt["completed_at_utc"] = _utc_now()
    receipt["elapsed_seconds"] = max(0.0, time.monotonic() - started_monotonic)
    receipt.pop("_started_monotonic", None)
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)


def _dry_run_receipt(config: HolderConfig) -> JsonObject:
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    receipt = _base_receipt(
        config,
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
    )
    receipt["status"] = "dry_run_validated"
    receipt["stop"] = {
        "reason": "dry_run",
        "stop_file_observed": False,
        "signals": {
            "first_signal_number": None,
            "first_signal_name": None,
            "count": 0,
        },
    }
    _complete_receipt(receipt, started_monotonic=started_monotonic)
    return receipt


def _run_holder(config: HolderConfig, stop_state: _StopState) -> tuple[JsonObject, int]:
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    receipt = _base_receipt(
        config,
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
    )
    session: CpuPerformancePolicySession | None = None
    active_at_utc: str | None = None
    active_monotonic: float | None = None
    stopped_monotonic: float | None = None
    stop_reason: str | None = None
    stop_file_observed = False
    failure: BaseException | None = None

    try:
        if stop_state.stop_requested:
            stop_reason = "signal_before_entry"
            stopped_monotonic = time.monotonic()
        elif config.stop_file is not None and _stop_file_exists(config.stop_file):
            stop_reason = "stop_file_preexisting"
            stop_file_observed = True
            stopped_monotonic = time.monotonic()
        else:
            session = cpu_performance_policy(transaction=config.transaction)
            with session:
                active_at_utc = _utc_now()
                active_monotonic = time.monotonic()
                stop_reason, stop_file_observed, stopped_monotonic = _wait_for_stop(
                    stop_state,
                    config.stop_file,
                    config.poll_interval_seconds,
                )
    except BaseException as error:
        failure = error
        if stop_reason is None:
            stop_reason = "error"

    policy_receipt: JsonObject | None = None
    policy_verified = False
    if session is not None:
        evidence = session.evidence
        policy_verified = _policy_evidence_verified(evidence)
        policy_receipt = {
            "verified": policy_verified,
            "evidence_sha256": _canonical_sha256(evidence),
            "evidence": evidence,
        }
        if failure is None and not policy_verified:
            failure = CpuPerformancePolicyHolderError(
                "CPU policy transaction did not verify application and restoration"
            )

    hold_seconds = (
        None
        if active_monotonic is None or stopped_monotonic is None
        else max(0.0, stopped_monotonic - active_monotonic)
    )
    stop_state.accepting_signals = False
    receipt["status"] = (
        "restored"
        if failure is None and policy_verified
        else (
            "not_started"
            if failure is None
            and stop_reason in {"signal_before_entry", "stop_file_preexisting"}
            else "failed"
        )
    )
    receipt["stop"] = {
        "reason": stop_reason,
        "active_at_utc": active_at_utc,
        "hold_seconds": hold_seconds,
        "stop_file_observed": stop_file_observed,
        "signals": _signal_receipt(stop_state),
    }
    receipt["policy"] = policy_receipt
    receipt["failure"] = None if failure is None else _failure_receipt(failure)
    _complete_receipt(receipt, started_monotonic=started_monotonic)
    return receipt, 0 if failure is None else 1


def _print_summary(receipt: JsonObject, output_path: Path) -> None:
    stop = receipt.get("stop")
    stop_reason = stop.get("reason") if isinstance(stop, dict) else None
    print(
        json.dumps(
            {
                "status": receipt.get("status"),
                "stop_reason": stop_reason,
                "output_path": str(output_path),
                "receipt_content_sha256": receipt.get("receipt_content_sha256"),
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    parsed = parser.parse_args(arguments)
    config = _config_from_arguments(parser, parsed)
    if config.dry_run:
        receipt = _dry_run_receipt(config)
        try:
            _atomic_write_json(config.output_path, receipt)
        except OSError as error:
            print(
                f"failed to write CPU policy holder receipt: {error}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        _print_summary(receipt, config.output_path)
        return 0

    stop_state = _StopState()
    previous_handlers: dict[signal.Signals, SignalHandler] = {}
    receipt: JsonObject
    exit_code: int
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, stop_state.handle)
        receipt, exit_code = _run_holder(config, stop_state)
        try:
            _atomic_write_json(config.output_path, receipt)
        except OSError as error:
            print(
                f"failed to write CPU policy holder receipt: {error}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        _print_summary(receipt, config.output_path)
        return exit_code
    finally:
        for managed_signal, previous_handler in previous_handlers.items():
            signal.signal(managed_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
