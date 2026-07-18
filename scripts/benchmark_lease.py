#!/usr/bin/env python3
"""Run a command while holding an ownership-safe benchmark lease.

The child may atomically write ``runtime-metadata.json`` during a run and
``benchmark-result.json`` before exiting. The wrapper preserves those fragments
and incorporates them into its lease record and final ``manifest.json``. The
result fragment must explicitly confirm cleanup before the lease can be
released. Production and CLI invocations always fail closed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType, TracebackType
from typing import Self, TextIO, cast
from uuid import uuid4

DEFAULT_LOCK_PATH = Path("/var/lock/fwuffydwagon-benchmark.lock")
DEFAULT_LEASE_PATH = Path("/var/lib/exo/coordination/benchmark-lease.json")
DEFAULT_RESULT_ROOT = Path("/var/lib/exo/benchmarks")
DEFAULT_CLEANUP_GRACE_SECONDS = 300.0
FORCED_CLEANUP_RETURN_CODE = 70
METADATA_SCHEMA_VERSION = 1
BENCHMARK_RESULT_SCHEMA_VERSION = 1
METADATA_MAX_AGE = timedelta(minutes=15)
METADATA_MAX_FUTURE_SKEW = timedelta(minutes=1)
BENCHMARK_RESULT_FILENAME = "benchmark-result.json"
RUNTIME_METADATA_FILENAME = "runtime-metadata.json"
MANIFEST_FILENAME = "manifest.json"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class LeaseError(RuntimeError):
    """Base error for lease acquisition and ownership failures."""


class LeaseBusyError(LeaseError):
    """Raised when another process owns the benchmark lock."""


class StaleLeaseError(LeaseError):
    """Raised when metadata remains without a live file lock."""


class ManagedSignalState:
    """Latch the first managed signal without raising asynchronously."""

    def __init__(self) -> None:
        self.first_signal_number: int | None = None

    def record(self, signal_number: int) -> None:
        if self.first_signal_number is None:
            self.first_signal_number = signal_number


@contextlib.contextmanager
def block_managed_signals(
    managed_signals: Sequence[signal.Signals],
) -> Iterator[None]:
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, managed_signals)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LeaseError(f"metadata.{field_name} must be a JSON object")
    object_mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in object_mapping):
        raise LeaseError(f"metadata.{field_name} must be a JSON object")
    return cast(Mapping[str, object], object_mapping)


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LeaseError(f"metadata.{field_name} must be a JSON array")
    return cast(Sequence[object], value)


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeaseError(f"metadata.{field_name} must be a nonempty string")
    return value


def _require_commit(value: object, field_name: str) -> str:
    commit = _require_nonempty_string(value, field_name)
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise LeaseError(f"metadata.{field_name} must be an exact commit hash")
    return commit


def _validate_file_hashes(value: object, field_name: str) -> Mapping[str, object]:
    file_hashes = _require_mapping(value, field_name)
    for dirty_path, digest_value in file_hashes.items():
        _require_nonempty_string(dirty_path, f"{field_name} key")
        digest = _require_nonempty_string(digest_value, f"{field_name}.{dirty_path}")
        if not _SHA256_PATTERN.fullmatch(digest):
            raise LeaseError(
                f"metadata.{field_name}.{dirty_path} must be a hexadecimal digest"
            )
    return file_hashes


def _require_absolute_path(value: object, field_name: str) -> str:
    path = _require_nonempty_string(value, field_name)
    if not Path(path).is_absolute():
        raise LeaseError(f"metadata.{field_name} must be an absolute path")
    return path


def _parse_metadata_timestamp(value: object) -> datetime:
    timestamp = _require_nonempty_string(value, "generated_at")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise LeaseError(
            "metadata.generated_at must be an ISO 8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LeaseError("metadata.generated_at must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_run_metadata(
    metadata: Mapping[str, object], *, now: datetime | None = None
) -> dict[str, object]:
    """Validate the reproducibility metadata required by /ai/coordinate.md."""
    validated = dict(metadata)
    schema_version = validated.get("schema_version")
    if type(schema_version) is not int or schema_version != METADATA_SCHEMA_VERSION:
        raise LeaseError(f"metadata.schema_version must be {METADATA_SCHEMA_VERSION}")

    generated_at = _parse_metadata_timestamp(validated.get("generated_at"))
    reference_time = (now or utc_now()).astimezone(timezone.utc)
    if generated_at < reference_time - METADATA_MAX_AGE:
        raise LeaseError(f"metadata.generated_at is older than {METADATA_MAX_AGE}")
    if generated_at > reference_time + METADATA_MAX_FUTURE_SKEW:
        raise LeaseError("metadata.generated_at is unreasonably far in the future")

    run_id = _require_nonempty_string(validated.get("run_id"), "run_id")
    namespace = _require_nonempty_string(validated.get("namespace"), "namespace")
    if not _IDENTIFIER_PATTERN.fullmatch(run_id):
        raise LeaseError("metadata.run_id is not a valid identifier")
    if not _IDENTIFIER_PATTERN.fullmatch(namespace):
        raise LeaseError("metadata.namespace is not a valid identifier")
    reserved_ports = _require_sequence(
        validated.get("reserved_ports"), "reserved_ports"
    )
    if not all(type(port) is int for port in reserved_ports):
        raise LeaseError("metadata.reserved_ports must contain integers")
    try:
        validate_ports(cast(Sequence[int], reserved_ports))
    except ValueError as error:
        raise LeaseError(f"metadata.reserved_ports is invalid: {error}") from error
    _require_absolute_path(validated.get("result_directory"), "result_directory")
    command = _require_sequence(validated.get("command"), "command")
    if not command or not all(isinstance(argument, str) for argument in command):
        raise LeaseError("metadata.command must be a nonempty array of strings")

    git_metadata = _require_mapping(validated.get("git"), "git")
    git_commit = _require_commit(git_metadata.get("commit"), "git.commit")
    dirty = git_metadata.get("dirty")
    if not isinstance(dirty, bool):
        raise LeaseError("metadata.git.dirty must be a boolean")
    dirty_file_hashes = _validate_file_hashes(
        git_metadata.get("dirty_file_hashes"), "git.dirty_file_hashes"
    )
    if dirty != bool(dirty_file_hashes):
        raise LeaseError(
            "metadata.git.dirty must match whether git.dirty_file_hashes is nonempty"
        )
    hosts = _require_sequence(validated.get("hosts"), "hosts")
    host_names = tuple(
        _require_nonempty_string(host, f"hosts[{index}]")
        for index, host in enumerate(hosts)
    )
    if not host_names:
        raise LeaseError("metadata.hosts must not be empty")
    if len(set(host_names)) != len(host_names):
        raise LeaseError("metadata.hosts must be unique")

    models = _require_sequence(validated.get("models"), "models")
    if not models:
        raise LeaseError("metadata.models must not be empty")
    for index, model_value in enumerate(models):
        model = _require_mapping(model_value, f"models[{index}]")
        _require_nonempty_string(model.get("model_id"), f"models[{index}].model_id")
        _require_commit(model.get("revision"), f"models[{index}].revision")
        model_paths = _require_mapping(model.get("paths"), f"models[{index}].paths")
        if set(model_paths) != set(host_names):
            raise LeaseError(
                f"metadata.models[{index}].paths must exactly match metadata.hosts"
            )
        for host_name in host_names:
            _require_absolute_path(
                model_paths.get(host_name), f"models[{index}].paths.{host_name}"
            )

    host_sections = (
        "gpu_bindings",
        "cpu_bindings",
        "hca_bindings",
        "source_deployments",
        "owner_pids",
    )
    for section_name in host_sections:
        section = _require_mapping(validated.get(section_name), section_name)
        if set(section) != set(host_names):
            raise LeaseError(
                f"metadata.{section_name} must exactly match metadata.hosts"
            )

    source_deployments = _require_mapping(
        validated["source_deployments"], "source_deployments"
    )
    gpu_bindings = _require_mapping(validated["gpu_bindings"], "gpu_bindings")
    cpu_bindings = _require_mapping(validated["cpu_bindings"], "cpu_bindings")
    hca_bindings = _require_mapping(validated["hca_bindings"], "hca_bindings")
    owner_pids = _require_mapping(validated["owner_pids"], "owner_pids")
    for host_name in host_names:
        host_gpu_bindings = _require_sequence(
            gpu_bindings[host_name], f"gpu_bindings.{host_name}"
        )
        for index, binding_value in enumerate(host_gpu_bindings):
            binding = _require_mapping(
                binding_value, f"gpu_bindings.{host_name}[{index}]"
            )
            _require_nonempty_string(
                binding.get("uuid"), f"gpu_bindings.{host_name}[{index}].uuid"
            )
            _require_nonempty_string(
                binding.get("pci_address"),
                f"gpu_bindings.{host_name}[{index}].pci_address",
            )

        cpu_binding = _require_mapping(
            cpu_bindings[host_name], f"cpu_bindings.{host_name}"
        )
        _require_nonempty_string(
            cpu_binding.get("cpu_set"), f"cpu_bindings.{host_name}.cpu_set"
        )
        numa_nodes = _require_sequence(
            cpu_binding.get("numa_nodes"), f"cpu_bindings.{host_name}.numa_nodes"
        )
        if not numa_nodes or not all(
            type(numa_node) is int and numa_node >= 0 for numa_node in numa_nodes
        ):
            raise LeaseError(
                f"metadata.cpu_bindings.{host_name}.numa_nodes must contain "
                "nonnegative integers"
            )
        _require_nonempty_string(
            cpu_binding.get("memory_policy"),
            f"cpu_bindings.{host_name}.memory_policy",
        )

        host_hca_bindings = _require_sequence(
            hca_bindings[host_name], f"hca_bindings.{host_name}"
        )
        if not host_hca_bindings:
            raise LeaseError(f"metadata.hca_bindings.{host_name} must not be empty")
        for index, binding_value in enumerate(host_hca_bindings):
            binding = _require_mapping(
                binding_value, f"hca_bindings.{host_name}[{index}]"
            )
            _require_nonempty_string(
                binding.get("device"), f"hca_bindings.{host_name}[{index}].device"
            )
            port = binding.get("port")
            if type(port) is not int or port <= 0:
                raise LeaseError(
                    f"metadata.hca_bindings.{host_name}[{index}].port must be "
                    "a positive integer"
                )
            _require_nonempty_string(
                binding.get("ip_address"),
                f"hca_bindings.{host_name}[{index}].ip_address",
            )

        deployment = _require_mapping(
            source_deployments[host_name], f"source_deployments.{host_name}"
        )
        _require_absolute_path(
            deployment.get("path"), f"source_deployments.{host_name}.path"
        )
        deployment_commit = _require_commit(
            deployment.get("commit"), f"source_deployments.{host_name}.commit"
        )
        if deployment_commit != git_commit:
            raise LeaseError(
                f"metadata.source_deployments.{host_name}.commit must match git.commit"
            )
        deployment_dirty_file_hashes = _validate_file_hashes(
            deployment.get("dirty_file_hashes"),
            f"source_deployments.{host_name}.dirty_file_hashes",
        )
        if dict(deployment_dirty_file_hashes) != dict(dirty_file_hashes):
            raise LeaseError(
                f"metadata.source_deployments.{host_name}.dirty_file_hashes must "
                "match git.dirty_file_hashes"
            )

        host_owner_pids = _require_sequence(
            owner_pids[host_name], f"owner_pids.{host_name}"
        )
        if not all(
            type(process_id) is int and process_id > 0 for process_id in host_owner_pids
        ):
            raise LeaseError(
                f"metadata.owner_pids.{host_name} must contain positive integers"
            )

    return validated


def validate_identifier(value: str, field_name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_', or '-'"
        )
    return value


def validate_ports(ports: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(ports)
    if not normalized:
        raise ValueError("at least one reserved port is required")
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
        cleanup_grace_seconds: float = DEFAULT_CLEANUP_GRACE_SECONDS,
        _allow_unconfirmed_cleanup_for_tests: bool = False,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if not math.isfinite(cleanup_grace_seconds) or cleanup_grace_seconds <= 0:
            raise ValueError("cleanup_grace_seconds must be positive and finite")
        if expected_duration_seconds is not None and expected_duration_seconds <= 0:
            raise ValueError("expected_duration_seconds must be positive")
        if not command:
            raise ValueError("benchmark command must not be empty")
        if not owner.strip():
            raise ValueError("owner must not be empty")
        if not purpose.strip():
            raise ValueError("purpose must not be empty")
        if _allow_unconfirmed_cleanup_for_tests and os.environ.get("EXO_TESTS") != "1":
            raise ValueError(
                "unconfirmed cleanup is available only under the test harness"
            )
        for path, field_name in (
            (lock_path, "lock_path"),
            (lease_path, "lease_path"),
            (result_directory, "result_directory"),
        ):
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute path")

        self.lock_path = lock_path
        self.lease_path = lease_path
        self.result_directory = result_directory
        self.owner = owner
        self.purpose = purpose
        self.run_id = validate_identifier(run_id, "run_id")
        self.namespace = validate_identifier(namespace, "namespace")
        self.ports = validate_ports(ports)
        self.command = tuple(command)
        self.metadata = validate_run_metadata(dict(metadata or {}))
        self._validate_metadata_binding()
        self.heartbeat_seconds = heartbeat_seconds
        self.expected_duration_seconds = expected_duration_seconds
        self.cleanup_grace_seconds = cleanup_grace_seconds

        self.lease_id = uuid4().hex
        self.started_at: datetime | None = None
        self.child_pid: int | None = None
        self._lock_file: TextIO | None = None
        self._metadata_lock = threading.Lock()
        self._fragment_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error_lock = threading.Lock()
        self._heartbeat_error: Exception | None = None
        self._preserve_lease_record = False
        self._cleanup_failure_reason: str | None = None
        self._runtime_metadata_cache: dict[str, object] | None = None
        self._runtime_metadata_error: str | None = None
        self._benchmark_result_cache: dict[str, object] | None = None
        self._benchmark_result_error: str | None = None
        self._child_manifest_error: str | None = None
        self._allow_unconfirmed_cleanup_for_tests = _allow_unconfirmed_cleanup_for_tests

    def _validate_metadata_binding(self) -> None:
        expected_values: tuple[tuple[str, object], ...] = (
            ("run_id", self.run_id),
            ("namespace", self.namespace),
            ("reserved_ports", list(self.ports)),
            ("result_directory", str(self.result_directory)),
            ("command", list(self.command)),
        )
        for field_name, expected_value in expected_values:
            if self.metadata.get(field_name) != expected_value:
                raise LeaseError(
                    f"metadata.{field_name} does not match the benchmark lease"
                )

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
        payload: dict[str, object] = {
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
            "cleanup_grace_seconds": self.cleanup_grace_seconds,
            "child_cleanup_confirmation_required": True,
        }
        runtime_metadata = self._refresh_runtime_metadata()
        if runtime_metadata is not None:
            payload["runtime_metadata"] = runtime_metadata
        fragment_errors = self._fragment_errors()
        if fragment_errors:
            payload["fragment_errors"] = fragment_errors
        if self._preserve_lease_record:
            payload.update(
                {
                    "cleanup_succeeded": False,
                    "manual_clearance_required": True,
                    "cleanup_failure_reason": self._cleanup_failure_reason,
                }
            )
        return payload

    def _fragment_errors(self) -> dict[str, object]:
        with self._fragment_lock:
            return {
                filename: error
                for filename, error in (
                    (RUNTIME_METADATA_FILENAME, self._runtime_metadata_error),
                    (BENCHMARK_RESULT_FILENAME, self._benchmark_result_error),
                    (MANIFEST_FILENAME, self._child_manifest_error),
                )
                if error is not None
            }

    def _record_fragment_error(self, filename: str, error: Exception) -> str:
        return f"failed to read or validate {filename}: {error}"

    def _validate_runtime_metadata(
        self, runtime_metadata: Mapping[str, object]
    ) -> dict[str, object]:
        validated = dict(runtime_metadata)
        runtime_schema_version = validated.get("schema_version")
        if type(runtime_schema_version) is not int or runtime_schema_version != 1:
            raise LeaseError("runtime metadata schema_version must be 1")
        if validated.get("run_id") != self.run_id:
            raise LeaseError("runtime metadata run_id does not match the lease")
        if validated.get("namespace") != self.namespace:
            raise LeaseError("runtime metadata namespace does not match the lease")
        owner_token = _require_nonempty_string(
            validated.get("owner_token"), "runtime.owner_token"
        )
        owned_processes = _require_sequence(
            validated.get("owned_processes"), "runtime.owned_processes"
        )
        host_values = _require_sequence(self.metadata.get("hosts"), "hosts")
        allowed_hosts = {
            _require_nonempty_string(host, "hosts entry") for host in host_values
        }
        process_identities: set[tuple[str, int, int]] = set()
        for index, process_value in enumerate(owned_processes):
            process = _require_mapping(
                process_value, f"runtime.owned_processes[{index}]"
            )
            host_name = _require_nonempty_string(
                process.get("host_name"),
                f"runtime.owned_processes[{index}].host_name",
            )
            if host_name not in allowed_hosts:
                raise LeaseError(
                    f"runtime owned process host {host_name!r} is not in the lease"
                )
            if process.get("namespace") != self.namespace:
                raise LeaseError(
                    f"runtime owned process {index} namespace does not match the lease"
                )
            if process.get("owner_token") != owner_token:
                raise LeaseError(
                    f"runtime owned process {index} owner_token does not match"
                )

            integer_fields = (
                "pid",
                "process_group_id",
                "start_time_ticks",
                "transport_pid",
            )
            integers: dict[str, int] = {}
            for field_name in integer_fields:
                field_value = process.get(field_name)
                if type(field_value) is not int or field_value <= 0:
                    raise LeaseError(
                        f"runtime owned process {index} {field_name} must be a "
                        "positive integer"
                    )
                integers[field_name] = field_value
            _require_absolute_path(
                process.get("log_path"),
                f"runtime.owned_processes[{index}].log_path",
            )
            identity = (
                host_name,
                integers["pid"],
                integers["start_time_ticks"],
            )
            if identity in process_identities:
                raise LeaseError("runtime owned process identities must be unique")
            process_identities.add(identity)
        return validated

    def _refresh_runtime_metadata(self) -> dict[str, object] | None:
        runtime_path = self.result_directory / RUNTIME_METADATA_FILENAME
        with self._fragment_lock:
            if self._runtime_metadata_error is not None:
                return self._runtime_metadata_cache
            try:
                runtime_metadata = self._validate_runtime_metadata(
                    read_json_object(runtime_path)
                )
                previous = self._runtime_metadata_cache
                if previous is not None:
                    if runtime_metadata["owner_token"] != previous["owner_token"]:
                        raise LeaseError(
                            "runtime metadata owner_token changed during the run"
                        )
                    previous_processes = self._runtime_process_map(previous)
                    current_processes = self._runtime_process_map(runtime_metadata)
                    if not previous_processes.keys() <= current_processes.keys():
                        raise LeaseError(
                            "runtime owned process identities must only grow"
                        )
                    for identity, previous_process in previous_processes.items():
                        if current_processes[identity] != previous_process:
                            raise LeaseError(
                                "runtime owned process identity metadata changed"
                            )
            except FileNotFoundError:
                return self._runtime_metadata_cache
            except (LeaseError, OSError, ValueError) as error:
                self._runtime_metadata_error = self._record_fragment_error(
                    RUNTIME_METADATA_FILENAME, error
                )
                return self._runtime_metadata_cache
            self._runtime_metadata_cache = runtime_metadata
            return runtime_metadata

    def _runtime_process_map(
        self, runtime_metadata: Mapping[str, object]
    ) -> dict[tuple[str, int, int], dict[str, object]]:
        process_values = _require_sequence(
            runtime_metadata.get("owned_processes"), "runtime.owned_processes"
        )
        processes: dict[tuple[str, int, int], dict[str, object]] = {}
        for index, process_value in enumerate(process_values):
            process = dict(
                _require_mapping(process_value, f"runtime.owned_processes[{index}]")
            )
            host_name = _require_nonempty_string(
                process.get("host_name"),
                f"runtime.owned_processes[{index}].host_name",
            )
            process_id = process.get("pid")
            start_time_ticks = process.get("start_time_ticks")
            if type(process_id) is not int or process_id <= 0:
                raise LeaseError(
                    f"runtime owned process {index} pid must be a positive integer"
                )
            if type(start_time_ticks) is not int or start_time_ticks <= 0:
                raise LeaseError(
                    "runtime owned process "
                    f"{index} start_time_ticks must be a positive integer"
                )
            identity = (
                host_name,
                process_id,
                start_time_ticks,
            )
            if identity in processes:
                raise LeaseError("runtime owned process identities must be unique")
            processes[identity] = process
        return processes

    def _validate_benchmark_result(
        self, benchmark_result: Mapping[str, object]
    ) -> dict[str, object]:
        validated = dict(benchmark_result)
        schema_version = validated.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != BENCHMARK_RESULT_SCHEMA_VERSION
        ):
            raise LeaseError(
                f"benchmark result schema_version must be "
                f"{BENCHMARK_RESULT_SCHEMA_VERSION}"
            )
        if validated.get("run_id") != self.run_id:
            raise LeaseError("benchmark result run_id does not match the lease")
        if validated.get("namespace") != self.namespace:
            raise LeaseError("benchmark result namespace does not match the lease")
        if not isinstance(validated.get("cleanup_succeeded"), bool):
            raise LeaseError("benchmark result cleanup_succeeded must be a boolean")

        owned_process_values = validated.get("owned_processes", ())
        owned_processes = _require_sequence(
            owned_process_values, "benchmark_result.owned_processes"
        )
        result_process_map = self._runtime_process_map(
            {"owned_processes": owned_processes}
        )
        runtime_metadata = self._runtime_metadata_cache
        if runtime_metadata is None:
            if result_process_map:
                raise LeaseError(
                    "benchmark result owned_processes require runtime metadata"
                )
        else:
            runtime_process_map = self._runtime_process_map(runtime_metadata)
            if result_process_map != runtime_process_map:
                raise LeaseError(
                    "benchmark result owned_processes do not match runtime metadata"
                )
        return validated

    def _refresh_benchmark_result(self) -> dict[str, object] | None:
        result_path = self.result_directory / BENCHMARK_RESULT_FILENAME
        with self._fragment_lock:
            self._refresh_runtime_metadata()
            if self._benchmark_result_error is not None:
                return self._benchmark_result_cache
            try:
                benchmark_result = self._validate_benchmark_result(
                    read_json_object(result_path)
                )
                if (
                    self._benchmark_result_cache is not None
                    and benchmark_result != self._benchmark_result_cache
                ):
                    raise LeaseError("benchmark result changed after validation")
            except FileNotFoundError:
                return self._benchmark_result_cache
            except (LeaseError, OSError, ValueError) as error:
                self._benchmark_result_error = self._record_fragment_error(
                    BENCHMARK_RESULT_FILENAME, error
                )
                return self._benchmark_result_cache
            self._benchmark_result_cache = benchmark_result
            return benchmark_result

    def cleanup_succeeded(self, local_cleanup_succeeded: bool) -> bool:
        if not local_cleanup_succeeded:
            return False
        self._refresh_runtime_metadata()
        benchmark_result = self._refresh_benchmark_result()
        if self._runtime_metadata_error is not None:
            return False
        if self._benchmark_result_error is not None:
            return False
        if benchmark_result is None:
            if self._allow_unconfirmed_cleanup_for_tests:
                return True
            with self._fragment_lock:
                self._benchmark_result_error = (
                    f"{BENCHMARK_RESULT_FILENAME} is required to confirm cleanup"
                )
            return False
        return cast(bool, benchmark_result["cleanup_succeeded"])

    def cleanup_confirmation_error(self) -> str | None:
        with self._fragment_lock:
            return self._runtime_metadata_error or self._benchmark_result_error

    def _ensure_preserved_record_locked(self) -> None:
        if not self._preserve_lease_record:
            raise LeaseError("cannot preserve a lease without a cleanup failure")
        if self.started_at is None:
            raise LeaseError("cannot preserve a lease before acquisition")

        current: dict[str, object] | None = None
        if self.lease_path.exists():
            current = read_json_object(self.lease_path)
        if current is not None and current.get("lease_id") != self.lease_id:
            raise LeaseError("lease metadata ownership changed")

        atomic_write_json(self.lease_path, self._lease_payload(utc_now()))
        verified = read_json_object(self.lease_path)
        if (
            verified.get("lease_id") != self.lease_id
            or verified.get("manual_clearance_required") is not True
            or verified.get("cleanup_succeeded") is not False
        ):
            raise LeaseError("failed to verify preserved lease tombstone")

    def preserve_after_cleanup_failure(self, reason: str) -> None:
        """Keep an exclusion record after owned cleanup cannot be confirmed."""
        self._preserve_lease_record = True
        self._cleanup_failure_reason = reason
        if self._lock_file is None or self.started_at is None:
            return
        with self._metadata_lock:
            self._ensure_preserved_record_locked()

    def acquire(self) -> None:
        if self._lock_file is not None:
            raise LeaseError("lease is already acquired")
        self.metadata = validate_run_metadata(self.metadata)
        self._validate_metadata_binding()

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
            if self._runtime_metadata_error is not None:
                raise LeaseError(self._runtime_metadata_error)

    def raise_if_heartbeat_failed(self) -> None:
        with self._heartbeat_error_lock:
            heartbeat_error = self._heartbeat_error
        if heartbeat_error is not None:
            raise LeaseError("benchmark lease heartbeat failed") from heartbeat_error

    def stop_heartbeat(self) -> Exception | None:
        """Stop and join the heartbeat before final ownership adjudication."""
        self._heartbeat_stop.set()
        heartbeat_thread = self._heartbeat_thread
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=self.heartbeat_seconds + 1.0)
            if heartbeat_thread.is_alive():
                return LeaseError("benchmark lease heartbeat did not stop")
            self._heartbeat_thread = None
        with self._heartbeat_error_lock:
            heartbeat_error = self._heartbeat_error
        if heartbeat_error is not None:
            return LeaseError(f"benchmark lease heartbeat failed: {heartbeat_error}")
        return None

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
        manifest_path = self.result_directory / MANIFEST_FILENAME
        benchmark_result = self._refresh_benchmark_result()
        runtime_metadata = self._refresh_runtime_metadata()
        child_manifest: dict[str, object] | None = None
        existing_benchmark_result: object | None = None
        existing_runtime_metadata: object | None = None
        existing_child_manifest: object | None = None
        try:
            existing_manifest = read_json_object(manifest_path)
        except FileNotFoundError:
            existing_manifest = None
        except (LeaseError, OSError, ValueError) as error:
            if self._child_manifest_error is None:
                self._child_manifest_error = self._record_fragment_error(
                    MANIFEST_FILENAME, error
                )
            existing_manifest = None
        if existing_manifest is not None:
            if (
                existing_manifest.get("manifest_writer") == "benchmark_lease.py"
                and existing_manifest.get("lease_id") == self.lease_id
            ):
                existing_benchmark_result = existing_manifest.get("benchmark_result")
                existing_runtime_metadata = existing_manifest.get("runtime_metadata")
                existing_child_manifest = existing_manifest.get("child_manifest")
            else:
                child_manifest = existing_manifest

        manifest: dict[str, object] = {
            **self._lease_payload(utc_now()),
            "manifest_writer": "benchmark_lease.py",
            "status": status,
            "return_code": return_code,
            "command_return_code": command_return_code,
            "cleanup_succeeded": cleanup_succeeded,
            "cleanup_forced": cleanup_forced,
            "error_message": error_message,
            "end_time": format_timestamp(utc_now()),
        }
        final_benchmark_result = (
            benchmark_result
            if benchmark_result is not None
            else existing_benchmark_result
        )
        final_runtime_metadata = (
            runtime_metadata
            if runtime_metadata is not None
            else existing_runtime_metadata
        )
        final_child_manifest = (
            child_manifest if child_manifest is not None else existing_child_manifest
        )
        if final_benchmark_result is not None:
            manifest["benchmark_result"] = final_benchmark_result
        if final_runtime_metadata is not None:
            manifest["runtime_metadata"] = final_runtime_metadata
        if final_child_manifest is not None:
            manifest["child_manifest"] = final_child_manifest
        atomic_write_json(manifest_path, manifest)
        return manifest_path

    def release(self) -> None:
        heartbeat_error = self.stop_heartbeat()
        if heartbeat_error is not None and not self._preserve_lease_record:
            self.preserve_after_cleanup_failure(str(heartbeat_error))

        lock_file = self._lock_file
        if lock_file is None:
            return
        with self._metadata_lock:
            if self._preserve_lease_record:
                self._ensure_preserved_record_locked()
            elif self.lease_path.exists():
                current = read_json_object(self.lease_path)
                if current.get("lease_id") == self.lease_id:
                    self.lease_path.unlink()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        self._lock_file = None


def confirm_owned_cleanup(
    lease: BenchmarkLease,
    *,
    local_cleanup_succeeded: bool,
    failure_context: str,
    finalization_error: Exception | None = None,
) -> tuple[bool, str | None]:
    failure_reason: str | None = None
    try:
        cleanup_succeeded = lease.cleanup_succeeded(local_cleanup_succeeded)
    except Exception as error:
        cleanup_succeeded = False
        failure_reason = f"cleanup could not be confirmed: {error}"

    if finalization_error is not None:
        cleanup_succeeded = False
        failure_reason = combine_error_messages(
            failure_reason,
            f"benchmark finalization failed: {finalization_error}",
        )

    if not cleanup_succeeded:
        if failure_reason is None:
            if not local_cleanup_succeeded:
                failure_reason = f"{failure_context}: owned local process group remains"
            elif lease.cleanup_confirmation_error() is not None:
                failure_reason = combine_error_messages(
                    failure_context, lease.cleanup_confirmation_error()
                )
            else:
                failure_reason = f"{failure_context}: child reported cleanup failure"
        try:
            assert failure_reason is not None
            lease.preserve_after_cleanup_failure(failure_reason)
        except Exception as error:
            failure_reason = (
                f"{failure_reason}; failed to refresh preserved lease record: {error}"
            )
    return cleanup_succeeded, failure_reason


def combine_error_messages(*messages: str | None) -> str | None:
    present_messages = tuple(message for message in messages if message)
    return "; ".join(present_messages) if present_messages else None


def cleanup_owned_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> tuple[bool, bool, str | None]:
    try:
        cleanup_forced = process_group_exists(process.pid)
    except Exception as error:
        return True, False, f"failed to inspect owned process group: {error}"
    if not cleanup_forced:
        return False, True, None
    try:
        cleanup_succeeded = terminate_owned_process_group(
            process, grace_seconds=grace_seconds
        )
    except Exception as error:
        return True, False, f"failed to terminate owned process group: {error}"
    if not cleanup_succeeded:
        return True, False, "owned process group remained after termination"
    return True, True, None


def run_managed_command(
    lease: BenchmarkLease, signal_state: ManagedSignalState | None = None
) -> int:
    managed_signal_state = signal_state or ManagedSignalState()
    process: subprocess.Popen[bytes] | None = None
    command_return_code: int | None = None
    wrapper_error: Exception | None = None
    try:
        if managed_signal_state.first_signal_number is None:
            process = subprocess.Popen(lease.command, start_new_session=True)
            lease.child_pid = process.pid
            lease.update_heartbeat()
            while managed_signal_state.first_signal_number is None:
                try:
                    command_return_code = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    lease.raise_if_heartbeat_failed()
            if command_return_code is not None:
                lease.raise_if_heartbeat_failed()
    except Exception as error:
        wrapper_error = error

    cleanup_forced = False
    local_cleanup_succeeded = True
    local_cleanup_error: str | None = None
    if process is not None:
        cleanup_forced, local_cleanup_succeeded, local_cleanup_error = (
            cleanup_owned_process_group(
                process, grace_seconds=lease.cleanup_grace_seconds
            )
        )
        command_return_code = process.returncode

    finalization_error = lease.stop_heartbeat()
    cleanup_succeeded, cleanup_failure = confirm_owned_cleanup(
        lease,
        local_cleanup_succeeded=local_cleanup_succeeded,
        failure_context="benchmark cleanup failed",
        finalization_error=finalization_error,
    )

    interrupted_signal = managed_signal_state.first_signal_number
    if interrupted_signal is not None:
        status = "interrupted"
        return_code = 128 + interrupted_signal
        primary_error = f"benchmark wrapper received signal {interrupted_signal}"
    elif wrapper_error is not None:
        status = "wrapper_error"
        return_code = 75
        primary_error = str(wrapper_error)
    elif not cleanup_succeeded:
        status = "cleanup_failed"
        return_code = FORCED_CLEANUP_RETURN_CODE
        primary_error = None
    elif cleanup_forced:
        status = "cleanup_forced"
        return_code = FORCED_CLEANUP_RETURN_CODE
        primary_error = None
    else:
        assert command_return_code is not None
        status = "completed" if command_return_code == 0 else "failed"
        return_code = command_return_code
        primary_error = None

    lease.write_manifest(
        status=status,
        return_code=return_code,
        command_return_code=command_return_code,
        cleanup_succeeded=cleanup_succeeded,
        cleanup_forced=cleanup_forced,
        error_message=combine_error_messages(
            primary_error, local_cleanup_error, cleanup_failure
        ),
    )

    late_signal = managed_signal_state.first_signal_number
    if interrupted_signal is None and late_signal is not None:
        return_code = 128 + late_signal
        lease.write_manifest(
            status="interrupted",
            return_code=return_code,
            command_return_code=command_return_code,
            cleanup_succeeded=cleanup_succeeded,
            cleanup_forced=cleanup_forced,
            error_message=combine_error_messages(
                f"benchmark wrapper received signal {late_signal}",
                primary_error,
                local_cleanup_error,
                cleanup_failure,
            ),
        )

    if wrapper_error is not None and managed_signal_state.first_signal_number is None:
        raise wrapper_error
    return return_code


def run_with_lease(lease: BenchmarkLease) -> int:
    managed_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    managed_signal_state = ManagedSignalState()
    previous_signal_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in managed_signals
    }
    installed_signals: list[signal.Signals] = []

    def handle_signal(signal_number: int, _frame: FrameType | None) -> None:
        managed_signal_state.record(signal_number)

    try:
        with block_managed_signals(managed_signals):
            for signal_number in managed_signals:
                signal.signal(signal_number, handle_signal)
                installed_signals.append(signal_number)
        with lease:
            lease.start_heartbeat()
            return run_managed_command(lease, managed_signal_state)
    finally:
        with block_managed_signals(managed_signals):
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
    parser.add_argument(
        "--cleanup-grace-seconds",
        type=float,
        default=DEFAULT_CLEANUP_GRACE_SECONDS,
        help="seconds to wait for cooperative child cleanup before SIGKILL",
    )
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
    cleanup_grace_seconds = cast(float, parsed.cleanup_grace_seconds)
    lock_path = cast(Path, parsed.lock_path)
    lease_path = cast(Path, parsed.lease_path)
    result_root = cast(Path, parsed.result_root)
    command = tuple(cast(list[str], parsed.command))
    if command and command[0] == "--":
        command = command[1:]
    try:
        ports = parse_ports(port_values)
        metadata = load_metadata(metadata_json)
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
            metadata=metadata,
            heartbeat_seconds=heartbeat_seconds,
            expected_duration_seconds=expected_duration_seconds,
            cleanup_grace_seconds=cleanup_grace_seconds,
        )
        return run_with_lease(lease)
    except (LeaseError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
