#!/usr/bin/env python3
"""Run a command while holding an ownership-safe benchmark lease."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType, TracebackType
from typing import Self, TextIO, cast
from uuid import uuid4

DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_RESULT_ROOT = Path("/var/lib/exo/benchmarks")
FORCED_CLEANUP_RETURN_CODE = 70
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LeaseError(RuntimeError):
    """Base error for lease acquisition and ownership failures."""


class LeaseBusyError(LeaseError):
    """Raised when another process owns the benchmark lock."""


class StaleLeaseError(LeaseError):
    """Raised when metadata remains without a live file lock."""


class CommandInterruptedError(LeaseError):
    """Raised by a managed signal while the benchmark command is running."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(f"benchmark wrapper received signal {signal_number}")
        self.signal_number = signal_number


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def validate_identifier(value: str, field_name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_', or '-'"
        )
    return value


def validate_ports(ports: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(ports)
    if len(set(normalized)) != len(normalized):
        raise ValueError("reserved ports must be unique")
    if any(port < 1 or port > 65535 for port in normalized):
        raise ValueError("reserved ports must be between 1 and 65535")
    return normalized


def read_json_object(path: Path) -> dict[str, object]:
    raw_value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_value, dict):
        raise LeaseError(f"Expected a JSON object in {path}")
    object_mapping = cast(dict[object, object], raw_value)
    if not all(isinstance(key, str) for key in object_mapping):
        raise LeaseError(f"Expected string keys in {path}")
    return {cast(str, key): item for key, item in object_mapping.items()}


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_owned_process_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 10.0
) -> bool:
    """Terminate only the process group created for this lease's command."""
    process_group_id = process.pid
    if not process_group_exists(process_group_id):
        return True

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + grace_seconds
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
        time.sleep(0.01)

    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return False

    return not process_group_exists(process_group_id)


class BenchmarkLease:
    def __init__(
        self,
        *,
        lock_path: Path,
        lease_path: Path,
        result_directory: Path,
        owner: str,
        purpose: str,
        run_id: str,
        namespace: str,
        ports: Sequence[int],
        command: Sequence[str],
        metadata: Mapping[str, object] | None = None,
        heartbeat_seconds: float = 30.0,
        expected_duration_seconds: float | None = None,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if expected_duration_seconds is not None and expected_duration_seconds <= 0:
            raise ValueError("expected_duration_seconds must be positive")
        if not command:
            raise ValueError("benchmark command must not be empty")

        self.lock_path = lock_path
        self.lease_path = lease_path
        self.result_directory = result_directory
        self.owner = owner
        self.purpose = purpose
        self.run_id = validate_identifier(run_id, "run_id")
        self.namespace = validate_identifier(namespace, "namespace")
        self.ports = validate_ports(ports)
        self.command = tuple(command)
        self.metadata = dict(metadata or {})
        self.heartbeat_seconds = heartbeat_seconds
        self.expected_duration_seconds = expected_duration_seconds

        self.lease_id = uuid4().hex
        self.started_at: datetime | None = None
        self.child_pid: int | None = None
        self._lock_file: TextIO | None = None
        self._metadata_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error_lock = threading.Lock()
        self._heartbeat_error: Exception | None = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.release()

    def _lease_payload(self, heartbeat: datetime) -> dict[str, object]:
        if self.started_at is None:
            raise LeaseError("lease has not been acquired")
        expected_completion = None
        if self.expected_duration_seconds is not None:
            expected_completion = format_timestamp(
                self.started_at + timedelta(seconds=self.expected_duration_seconds)
            )
        return {
            "lease_id": self.lease_id,
            "owner": self.owner,
            "purpose": self.purpose,
            "run_id": self.run_id,
            "wrapper_pid": os.getpid(),
            "child_pid": self.child_pid,
            "command": list(self.command),
            "exo_namespace": self.namespace,
            "ports": list(self.ports),
            "start_time": format_timestamp(self.started_at),
            "expected_completion_time": expected_completion,
            "heartbeat": format_timestamp(heartbeat),
            "result_directory": str(self.result_directory),
            "metadata": self.metadata,
        }

    def acquire(self) -> None:
        if self._lock_file is not None:
            raise LeaseError("lease is already acquired")

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            details = ""
            if self.lease_path.exists():
                details = f": {read_json_object(self.lease_path)}"
            raise LeaseBusyError(f"benchmark lease is already held{details}") from error

        if self.lease_path.exists():
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            raise StaleLeaseError(
                f"stale lease metadata exists at {self.lease_path}; inspect it manually"
            )

        try:
            if self.result_directory.exists() and any(self.result_directory.iterdir()):
                raise LeaseError(
                    f"result directory is not empty: {self.result_directory}; "
                    "use a new run ID"
                )
            self.result_directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            raise

        self._lock_file = lock_file
        self.started_at = utc_now()
        try:
            atomic_write_json(self.lease_path, self._lease_payload(self.started_at))
        except Exception:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            self._lock_file = None
            self.started_at = None
            raise

    def start_heartbeat(self) -> None:
        if self._lock_file is None:
            raise LeaseError("cannot start heartbeat before acquiring lease")
        if self._heartbeat_thread is not None:
            raise LeaseError("heartbeat is already running")

        def heartbeat_loop() -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_seconds):
                try:
                    self.update_heartbeat()
                except Exception as error:
                    with self._heartbeat_error_lock:
                        self._heartbeat_error = error
                    self._heartbeat_stop.set()
                    return

        self._heartbeat_stop.clear()
        with self._heartbeat_error_lock:
            self._heartbeat_error = None
        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"benchmark-lease-{self.run_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def update_heartbeat(self) -> None:
        with self._metadata_lock:
            current = read_json_object(self.lease_path)
            if current.get("lease_id") != self.lease_id:
                raise LeaseError("lease metadata ownership changed")
            atomic_write_json(self.lease_path, self._lease_payload(utc_now()))

    def raise_if_heartbeat_failed(self) -> None:
        with self._heartbeat_error_lock:
            heartbeat_error = self._heartbeat_error
        if heartbeat_error is not None:
            raise LeaseError("benchmark lease heartbeat failed") from heartbeat_error

    def write_manifest(
        self,
        *,
        status: str,
        return_code: int | None,
        cleanup_succeeded: bool,
        cleanup_forced: bool = False,
        command_return_code: int | None = None,
        error_message: str | None = None,
    ) -> Path:
        if self.started_at is None:
            raise LeaseError("cannot write manifest before acquiring lease")
        manifest_path = self.result_directory / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                **self._lease_payload(utc_now()),
                "status": status,
                "return_code": return_code,
                "command_return_code": command_return_code,
                "cleanup_succeeded": cleanup_succeeded,
                "cleanup_forced": cleanup_forced,
                "error_message": error_message,
                "end_time": format_timestamp(utc_now()),
            },
        )
        return manifest_path

    def release(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self.heartbeat_seconds + 1)
            self._heartbeat_thread = None

        lock_file = self._lock_file
        if lock_file is None:
            return
        try:
            with self._metadata_lock:
                if self.lease_path.exists():
                    current = read_json_object(self.lease_path)
                    if current.get("lease_id") == self.lease_id:
                        self.lease_path.unlink()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            self._lock_file = None


def run_managed_command(lease: BenchmarkLease) -> int:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(lease.command, start_new_session=True)
        lease.child_pid = process.pid
        lease.update_heartbeat()
        while True:
            try:
                command_return_code = process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                lease.raise_if_heartbeat_failed()
        lease.raise_if_heartbeat_failed()
    except CommandInterruptedError as error:
        return_code = 128 + error.signal_number
        cleanup_succeeded = process is None or terminate_owned_process_group(process)
        lease.write_manifest(
            status="interrupted",
            return_code=return_code,
            command_return_code=process.returncode if process is not None else None,
            cleanup_succeeded=cleanup_succeeded,
            cleanup_forced=process is not None,
            error_message=str(error),
        )
        return return_code
    except Exception as error:
        cleanup_succeeded = process is None or terminate_owned_process_group(process)
        lease.write_manifest(
            status="wrapper_error",
            return_code=75,
            command_return_code=process.returncode if process is not None else None,
            cleanup_succeeded=cleanup_succeeded,
            cleanup_forced=process is not None,
            error_message=str(error),
        )
        raise

    assert process is not None
    cleanup_forced = process_group_exists(process.pid)
    cleanup_succeeded = not cleanup_forced or terminate_owned_process_group(process)
    if cleanup_forced:
        status = "cleanup_forced" if cleanup_succeeded else "cleanup_failed"
        return_code = FORCED_CLEANUP_RETURN_CODE
    else:
        status = "completed" if command_return_code == 0 else "failed"
        return_code = command_return_code

    lease.write_manifest(
        status=status,
        return_code=return_code,
        command_return_code=command_return_code,
        cleanup_succeeded=cleanup_succeeded,
        cleanup_forced=cleanup_forced,
    )
    return return_code


def run_with_lease(lease: BenchmarkLease) -> int:
    managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_signal_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in managed_signals
    }
    installed_signals: list[signal.Signals] = []

    def handle_signal(signal_number: int, _frame: FrameType | None) -> None:
        raise CommandInterruptedError(signal_number)

    try:
        with lease:
            for signal_number in managed_signals:
                signal.signal(signal_number, handle_signal)
                installed_signals.append(signal_number)
            try:
                lease.start_heartbeat()
                return run_managed_command(lease)
            finally:
                for signal_number in installed_signals:
                    signal.signal(signal_number, signal.SIG_IGN)
    finally:
        for signal_number in installed_signals:
            signal.signal(signal_number, previous_signal_handlers[signal_number])


def parse_ports(values: Sequence[str]) -> tuple[int, ...]:
    ports: list[int] = []
    for value in values:
        ports.extend(int(item) for item in value.split(",") if item)
    return validate_ports(ports)


def load_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return read_json_object(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--port", action="append", default=[])
    parser.add_argument(
        "--metadata-json",
        type=Path,
        required=True,
        help="run metadata required by /ai/coordinate.md",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--expected-duration-seconds", type=float)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    owner = cast(str, parsed.owner)
    purpose = cast(str, parsed.purpose)
    run_id = cast(str, parsed.run_id)
    namespace = cast(str, parsed.namespace)
    port_values = cast(list[str], parsed.port)
    metadata_json = cast(Path | None, parsed.metadata_json)
    heartbeat_seconds = cast(float, parsed.heartbeat_seconds)
    expected_duration_seconds = cast(float | None, parsed.expected_duration_seconds)
    lock_path = cast(Path, parsed.lock_path)
    lease_path = cast(Path, parsed.lease_path)
    result_root = cast(Path, parsed.result_root)
    command = tuple(cast(list[str], parsed.command))
    if command and command[0] == "--":
        command = command[1:]
    ports = parse_ports(port_values)
    lease = BenchmarkLease(
        lock_path=lock_path,
        lease_path=lease_path,
        result_directory=result_root / run_id,
        owner=owner,
        purpose=purpose,
        run_id=run_id,
        namespace=namespace,
        ports=ports,
        command=command,
        metadata=load_metadata(metadata_json),
        heartbeat_seconds=heartbeat_seconds,
        expected_duration_seconds=expected_duration_seconds,
    )
    try:
        return run_with_lease(lease)
    except LeaseError as error:
        print(error, file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
