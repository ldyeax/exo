from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import stat
import time
import uuid
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_MAXIMUM_SYSFS_VALUE_BYTES: Final = 16 * 1024
_MAXIMUM_JOURNAL_BYTES: Final = 1024 * 1024
_POLICY_NAME_PATTERN: Final = re.compile(r"policy(\d+)", re.ASCII)
_RAPL_PACKAGE_PATTERN: Final = re.compile(r"intel-rapl:(\d+)", re.ASCII)
_RAPL_LIMIT_FIELD_PATTERN: Final = re.compile(
    r"constraint_\d+_(?:max_power_uw|min_power_uw|name|power_limit_uw|time_window_us)",
    re.ASCII,
)
_TEMPERATURE_FIELD_PATTERN: Final = re.compile(
    r"temp\d+_(?:crit|crit_alarm|crit_hyst|emergency|emergency_alarm|label|lcrit|max|max_alarm|max_hyst|min)",
    re.ASCII,
)
_THROTTLE_FIELD_PATTERN: Final = re.compile(
    r"[a-z0-9_]*throttle[a-z0-9_]*",
    re.ASCII,
)


@dataclass(frozen=True)
class CpuPerformancePolicySysfsRoots:
    cpu: Path = Path("/sys/devices/system/cpu")
    powercap: Path = Path("/sys/class/powercap")
    hwmon: Path = Path("/sys/class/hwmon")


_DEFAULT_SYSFS_ROOTS: Final = CpuPerformancePolicySysfsRoots()


@dataclass(frozen=True)
class CpuPerformancePolicyTransactionConfig:
    """Host-global serialization paths and process identity sources.

    Every harness controlling the same host CPUs must share the same lock and
    journal paths. A zero timeout rejects overlap, ``None`` blocks, and a
    positive timeout waits for at most that many seconds.
    """

    lock_path: Path = Path("/run/lock/exo/cpu-performance-policy.lock")
    journal_path: Path = Path("/run/lock/exo/cpu-performance-policy.journal.json")
    proc_root: Path = Path("/proc")
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id")
    lock_timeout_seconds: float | None = 0.0
    lock_poll_interval_seconds: float = 0.05


_DEFAULT_TRANSACTION_CONFIG: Final = CpuPerformancePolicyTransactionConfig()


@dataclass(frozen=True)
class CpuPerformancePolicyFailure:
    phase: str
    operation: str
    path: str | None
    error_type: str
    message: str

    def as_json(self) -> JsonObject:
        return {
            "phase": self.phase,
            "operation": self.operation,
            "path": self.path,
            "error_type": self.error_type,
            "message": self.message,
        }


class CpuPerformancePolicyError(RuntimeError):
    def __init__(
        self,
        message: str,
        failures: tuple[CpuPerformancePolicyFailure, ...],
        evidence: JsonObject,
    ) -> None:
        self.failures = failures
        self.evidence = evidence
        details = "; ".join(
            f"{failure.phase}/{failure.operation}"
            f"{f' [{failure.path}]' if failure.path is not None else ''}: "
            f"{failure.error_type}: {failure.message}"
            for failure in failures
        )
        super().__init__(f"{message}: {details}")


@dataclass(frozen=True)
class _PolicyState:
    name: str
    path: Path
    device: int
    inode: int
    affected_cpu_ids: tuple[int, ...]
    online_cpu_ids: tuple[int, ...]
    representative_cpu_id: int
    scaling_governor: str
    energy_performance_preference: str
    metadata: dict[str, str]

    def as_json(self) -> JsonObject:
        return {
            "name": self.name,
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "affected_cpu_ids": list(self.affected_cpu_ids),
            "online_cpu_ids": list(self.online_cpu_ids),
            "representative_cpu_id": self.representative_cpu_id,
            "original": {
                "scaling_governor": self.scaling_governor,
                "energy_performance_preference": (self.energy_performance_preference),
            },
            "metadata": dict(self.metadata),
        }

    def identity_json(self) -> JsonObject:
        return {
            "name": self.name,
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "affected_cpu_ids": list(self.affected_cpu_ids),
            "online_cpu_ids": list(self.online_cpu_ids),
            "representative_cpu_id": self.representative_cpu_id,
            "metadata": dict(self.metadata),
        }

    def identity_key(self) -> str:
        return _canonical_json(self.identity_json()).decode("ascii")


@dataclass(frozen=True)
class _ProcessOwner:
    pid: int
    start_time_ticks: int
    boot_id: str

    def as_json(self) -> JsonObject:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "boot_id": self.boot_id,
        }


@dataclass(frozen=True)
class _JournalState:
    transaction_id: str
    owner: _ProcessOwner
    recoverable: bool
    online_cpu_ids: tuple[int, ...]
    policies: tuple[_PolicyState, ...]


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _failure(
    phase: str,
    operation: str,
    error: BaseException,
    path: Path | None = None,
) -> CpuPerformancePolicyFailure:
    return CpuPerformancePolicyFailure(
        phase=phase,
        operation=operation,
        path=None if path is None else str(path),
        error_type=type(error).__name__,
        message=str(error),
    )


def _read_sysfs_value(path: Path) -> str:
    with path.open("rb") as input_file:
        contents = input_file.read(_MAXIMUM_SYSFS_VALUE_BYTES + 1)
    if len(contents) > _MAXIMUM_SYSFS_VALUE_BYTES:
        raise ValueError("sysfs value exceeds the size bound")
    value = contents.decode("ascii").strip()
    if not value:
        raise ValueError("sysfs value is empty")
    return value


def _write_sysfs_value(path: Path, value: str) -> None:
    with path.open("w", encoding="ascii", newline="") as output_file:
        written = output_file.write(value)
    if written != len(value):
        raise OSError(f"short sysfs write: wrote {written} of {len(value)} bytes")


def _read_verified_integer(path: Path) -> int:
    raw = _read_sysfs_value(path)
    value = int(raw)
    if value < 0:
        raise ValueError("sysfs counter is negative")
    return value


def _parse_cpu_list(raw: str) -> tuple[int, ...]:
    cpu_ids: set[int] = set()
    for item in raw.split(","):
        if not item or item.strip() != item:
            raise ValueError(f"invalid CPU-list item: {item!r}")
        bounds = item.split("-")
        if len(bounds) == 1:
            first = last = int(bounds[0])
        elif len(bounds) == 2:
            first, last = (int(bound) for bound in bounds)
        else:
            raise ValueError(f"invalid CPU-list range: {item!r}")
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU-list range: {item!r}")
        new_cpu_ids = set(range(first, last + 1))
        if cpu_ids.intersection(new_cpu_ids):
            raise ValueError(f"CPU-list contains duplicate CPUs: {item!r}")
        cpu_ids.update(new_cpu_ids)
    if not cpu_ids:
        raise ValueError("CPU list is empty")
    return tuple(sorted(cpu_ids))


def _optional_directory(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(path)
    return True


def _optional_file(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise OSError(f"expected a regular sysfs file: {path}")
    return True


def _ordered_directory_entries(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(path.iterdir(), key=lambda item: item.name))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_process_start_time(proc_root: Path, pid: int) -> int:
    raw = _read_sysfs_value(proc_root / str(pid) / "stat")
    closing_parenthesis = raw.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process stat lacks a closing command parenthesis")
    fields = raw[closing_parenthesis + 1 :].strip().split()
    start_time_index = 22 - 3
    if len(fields) <= start_time_index:
        raise ValueError("process stat lacks the start-time field")
    start_time_ticks = int(fields[start_time_index])
    if start_time_ticks < 0:
        raise ValueError("process start time is negative")
    return start_time_ticks


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{name} must contain only string values")
        result[key] = item
    return result


def _require_integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty array")
    result = tuple(_require_int(item, name) for item in value)
    if any(item < 0 for item in result) or tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must contain unique sorted nonnegative integers")
    return result


class CpuPerformancePolicySession(
    AbstractContextManager["CpuPerformancePolicySession"]
):
    """Temporarily enforce and verify the Linux CPU performance policy."""

    def __init__(
        self,
        *,
        roots: CpuPerformancePolicySysfsRoots = _DEFAULT_SYSFS_ROOTS,
        transaction: CpuPerformancePolicyTransactionConfig = (
            _DEFAULT_TRANSACTION_CONFIG
        ),
    ) -> None:
        self._roots = roots
        self._transaction = transaction
        self._policies: tuple[_PolicyState, ...] = ()
        self._online_cpu_ids: tuple[int, ...] = ()
        self._failures: list[CpuPerformancePolicyFailure] = []
        self._entered = False
        self._closed = False
        self._lock_descriptor: int | None = None
        self._owner: _ProcessOwner | None = None
        self._journal_transaction_id: str | None = None
        self._evidence: JsonObject = {
            "schema_version": 1,
            "requested_policy": {
                "scaling_governor": "performance",
                "energy_performance_preference": "performance",
            },
            "sysfs_roots": {
                "cpu": str(roots.cpu),
                "powercap": str(roots.powercap),
                "hwmon": str(roots.hwmon),
            },
            "lifecycle": "new",
            "online_cpu_ids": [],
            "policies": [],
            "snapshots": {},
            "application_verified": False,
            "restoration_verified": False,
            "serialization": {
                "lock_path": str(transaction.lock_path),
                "journal_path": str(transaction.journal_path),
                "lock_timeout_seconds": transaction.lock_timeout_seconds,
                "lock_acquired": False,
                "journal_published": False,
                "journal_removed": False,
            },
            "stale_recovery": {
                "journal_found": False,
                "performed": False,
                "restoration_verified": False,
            },
            "failures": [],
        }

    @property
    def evidence(self) -> JsonObject:
        return self._evidence

    def _record_failure(self, failure: CpuPerformancePolicyFailure) -> None:
        self._failures.append(failure)
        failures = self._evidence["failures"]
        assert isinstance(failures, list)
        failures.append(failure.as_json())

    def _raise(self, message: str) -> None:
        self._evidence["lifecycle"] = "failed"
        raise CpuPerformancePolicyError(
            message,
            tuple(self._failures),
            self._evidence,
        )

    def _serialization_evidence(self) -> JsonObject:
        evidence = self._evidence["serialization"]
        assert isinstance(evidence, dict)
        return evidence

    def _recovery_evidence(self) -> JsonObject:
        evidence = self._evidence["stale_recovery"]
        assert isinstance(evidence, dict)
        return evidence

    def _current_process_owner(self) -> _ProcessOwner:
        pid = os.getpid()
        return _ProcessOwner(
            pid=pid,
            start_time_ticks=_read_process_start_time(self._transaction.proc_root, pid),
            boot_id=_read_sysfs_value(self._transaction.boot_id_path),
        )

    def _acquire_transaction_lock(self) -> None:
        phase = "serialization"
        lock_path = self._transaction.lock_path
        journal_path = self._transaction.journal_path
        descriptor: int | None = None
        try:
            if lock_path == journal_path:
                raise ValueError("lock and journal paths must differ")
            timeout = self._transaction.lock_timeout_seconds
            if timeout is not None and (not math.isfinite(timeout) or timeout < 0.0):
                raise ValueError("lock timeout must be finite and nonnegative")
            poll_interval = self._transaction.lock_poll_interval_seconds
            if not math.isfinite(poll_interval) or poll_interval <= 0.0:
                raise ValueError("lock poll interval must be finite and positive")
            lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            journal_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_path, flags, 0o600)
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise OSError("CPU policy lock is not a regular file")

            if timeout is None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            else:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(
                            descriptor,
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "CPU performance-policy transaction lock is held"
                            ) from None
                        time.sleep(
                            min(poll_interval, max(0.0, deadline - time.monotonic()))
                        )
            self._lock_descriptor = descriptor
            descriptor = None
            self._owner = self._current_process_owner()
            serialization = self._serialization_evidence()
            serialization["lock_acquired"] = True
            serialization["owner"] = self._owner.as_json()
        except BaseException as error:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            self._record_failure(
                _failure(phase, "acquire_transaction_lock", error, lock_path)
            )

    def _release_transaction_lock(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is None:
            return
        self._lock_descriptor = None
        failures: list[CpuPerformancePolicyFailure] = []
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except BaseException as error:
            failures.append(
                _failure(
                    "serialization",
                    "unlock_transaction_lock",
                    error,
                    self._transaction.lock_path,
                )
            )
        try:
            os.close(descriptor)
        except BaseException as error:
            failures.append(
                _failure(
                    "serialization",
                    "close_transaction_lock",
                    error,
                    self._transaction.lock_path,
                )
            )
        self._serialization_evidence()["lock_acquired"] = False
        self._serialization_evidence()["lock_released"] = not failures
        for failure in failures:
            self._record_failure(failure)

    def _journal_payload(self, transaction_id: str) -> JsonObject:
        if self._owner is None:
            raise RuntimeError("transaction owner is unavailable")
        return {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "owner": self._owner.as_json(),
            "recoverable": False,
            "roots": {
                "cpu": str(self._roots.cpu),
                "powercap": str(self._roots.powercap),
                "hwmon": str(self._roots.hwmon),
            },
            "online_cpu_ids": list(self._online_cpu_ids),
            "policies": [policy.as_json() for policy in self._policies],
        }

    def _adopt_exact_published_journal(
        self,
        transaction_id: str,
        expected_payload_bytes: bytes,
        *,
        after_interruption: bool,
    ) -> bool:
        verified_payload = self._read_journal_payload()
        if verified_payload is None or not hmac.compare_digest(
            _canonical_json(verified_payload),
            expected_payload_bytes,
        ):
            return False
        verified_journal = self._decode_journal(verified_payload)
        if (
            verified_journal.transaction_id != transaction_id
            or verified_journal.recoverable
        ):
            raise ValueError("published CPU policy journal did not verify")
        self._journal_transaction_id = transaction_id
        serialization = self._serialization_evidence()
        serialization["journal_published"] = True
        serialization["journal_removed"] = False
        serialization["journal_verified"] = True
        if after_interruption:
            serialization["journal_adopted_after_interruption"] = True
        return True

    def _publish_journal(self) -> None:
        journal_path = self._transaction.journal_path
        transaction_id = uuid.uuid4().hex
        payload = self._journal_payload(transaction_id)
        payload_bytes = _canonical_json(payload)
        envelope: JsonObject = {
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        contents = _canonical_json(envelope) + b"\n"
        temporary_path = journal_path.with_name(
            f".{journal_path.name}.{os.getpid()}.{transaction_id}.tmp"
        )
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary_path, flags, 0o600)
            offset = 0
            while offset < len(contents):
                written = os.write(descriptor, contents[offset:])
                if written <= 0:
                    raise OSError("short write while publishing CPU policy journal")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary_path, journal_path, follow_symlinks=False)
            os.unlink(temporary_path)
            _fsync_directory(journal_path.parent)
            if not self._adopt_exact_published_journal(
                transaction_id,
                payload_bytes,
                after_interruption=False,
            ):
                raise ValueError("published CPU policy journal did not verify")
        except BaseException as error:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
            try:
                self._adopt_exact_published_journal(
                    transaction_id,
                    payload_bytes,
                    after_interruption=True,
                )
            except BaseException as adoption_error:
                self._record_failure(
                    _failure(
                        "journal",
                        "adopt_interrupted_journal_publication",
                        adoption_error,
                        journal_path,
                    )
                )
            self._record_failure(
                _failure(
                    "journal",
                    "publish_pre_mutation_journal",
                    error,
                    journal_path,
                )
            )

    def _read_journal_payload(self) -> JsonObject | None:
        journal_path = self._transaction.journal_path
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(journal_path, flags)
        except FileNotFoundError:
            return None
        try:
            journal_stat = os.fstat(descriptor)
            if not stat.S_ISREG(journal_stat.st_mode):
                raise OSError("CPU policy journal is not a regular file")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, _MAXIMUM_JOURNAL_BYTES + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAXIMUM_JOURNAL_BYTES:
                    raise ValueError("CPU policy journal exceeds the size bound")
        finally:
            os.close(descriptor)
        decoded = b"".join(chunks).decode("ascii")
        raw_envelope = cast(object, json.loads(decoded))
        if not isinstance(raw_envelope, dict):
            raise ValueError("CPU policy journal envelope must be an object")
        payload = raw_envelope.get("payload")
        digest = raw_envelope.get("payload_sha256")
        if not isinstance(payload, dict) or not isinstance(digest, str):
            raise ValueError("CPU policy journal envelope is incomplete")
        observed_digest = hashlib.sha256(
            _canonical_json(cast(JsonObject, payload))
        ).hexdigest()
        if not hmac.compare_digest(observed_digest, digest):
            raise ValueError("CPU policy journal integrity check failed")
        return cast(JsonObject, payload)

    def _decode_journal(self, payload: JsonObject) -> _JournalState:
        if _require_int(payload.get("schema_version"), "schema_version") != 1:
            raise ValueError("unsupported CPU policy journal schema")
        transaction_id = _require_string(
            payload.get("transaction_id"), "transaction_id"
        )
        try:
            if uuid.UUID(hex=transaction_id).hex != transaction_id:
                raise ValueError
        except ValueError:
            raise ValueError("invalid CPU policy journal transaction ID") from None
        raw_owner = payload.get("owner")
        if not isinstance(raw_owner, dict):
            raise ValueError("journal owner must be an object")
        owner = _ProcessOwner(
            pid=_require_int(raw_owner.get("pid"), "owner.pid"),
            start_time_ticks=_require_int(
                raw_owner.get("start_time_ticks"), "owner.start_time_ticks"
            ),
            boot_id=_require_string(raw_owner.get("boot_id"), "owner.boot_id"),
        )
        if owner.pid <= 0 or owner.start_time_ticks < 0:
            raise ValueError("journal owner identity is invalid")
        recoverable = payload.get("recoverable")
        if not isinstance(recoverable, bool):
            raise ValueError("journal recoverable state must be boolean")
        raw_roots = payload.get("roots")
        if not isinstance(raw_roots, dict) or raw_roots != {
            "cpu": str(self._roots.cpu),
            "powercap": str(self._roots.powercap),
            "hwmon": str(self._roots.hwmon),
        }:
            raise ValueError("journal sysfs roots do not match this session")
        online_cpu_ids = _require_integer_tuple(
            payload.get("online_cpu_ids"), "online_cpu_ids"
        )
        raw_policies = payload.get("policies")
        if not isinstance(raw_policies, list) or not raw_policies:
            raise ValueError("journal policies must be a nonempty array")
        policies: list[_PolicyState] = []
        for index, raw_policy in enumerate(raw_policies):
            if not isinstance(raw_policy, dict):
                raise ValueError(f"journal policy {index} must be an object")
            raw_original = raw_policy.get("original")
            if not isinstance(raw_original, dict):
                raise ValueError(f"journal policy {index} lacks original state")
            policy = _PolicyState(
                name=_require_string(raw_policy.get("name"), "policy.name"),
                path=Path(_require_string(raw_policy.get("path"), "policy.path")),
                device=_require_int(raw_policy.get("device"), "policy.device"),
                inode=_require_int(raw_policy.get("inode"), "policy.inode"),
                affected_cpu_ids=_require_integer_tuple(
                    raw_policy.get("affected_cpu_ids"), "policy.affected_cpu_ids"
                ),
                online_cpu_ids=_require_integer_tuple(
                    raw_policy.get("online_cpu_ids"), "policy.online_cpu_ids"
                ),
                representative_cpu_id=_require_int(
                    raw_policy.get("representative_cpu_id"),
                    "policy.representative_cpu_id",
                ),
                scaling_governor=_require_string(
                    raw_original.get("scaling_governor"),
                    "policy.original.scaling_governor",
                ),
                energy_performance_preference=_require_string(
                    raw_original.get("energy_performance_preference"),
                    "policy.original.energy_performance_preference",
                ),
                metadata=_require_string_mapping(
                    raw_policy.get("metadata"), "policy.metadata"
                ),
            )
            if policy.device < 0 or policy.inode < 0:
                raise ValueError(f"journal policy {index} has invalid file identity")
            if (
                not policy.path.is_absolute()
                or policy.representative_cpu_id not in policy.online_cpu_ids
                or not set(policy.online_cpu_ids).issubset(policy.affected_cpu_ids)
            ):
                raise ValueError(f"journal policy {index} has invalid CPU identity")
            policies.append(policy)
        claimed: set[int] = set()
        for policy in policies:
            overlap = claimed.intersection(policy.online_cpu_ids)
            if overlap:
                raise ValueError("journal policies have overlapping online CPUs")
            claimed.update(policy.online_cpu_ids)
        if claimed != set(online_cpu_ids):
            raise ValueError("journal policies do not cover all journal online CPUs")
        identities = [policy.identity_key() for policy in policies]
        if len(identities) != len(set(identities)):
            raise ValueError("journal contains duplicate policy identities")
        return _JournalState(
            transaction_id=transaction_id,
            owner=owner,
            recoverable=recoverable,
            online_cpu_ids=online_cpu_ids,
            policies=tuple(policies),
        )

    def _owner_is_live(self, owner: _ProcessOwner) -> bool:
        current_boot_id = _read_sysfs_value(self._transaction.boot_id_path)
        if current_boot_id != owner.boot_id:
            return False
        try:
            observed_start_time = _read_process_start_time(
                self._transaction.proc_root, owner.pid
            )
        except FileNotFoundError:
            return False
        return observed_start_time == owner.start_time_ticks

    def _remove_journal(
        self,
        expected_transaction_id: str,
        expected_online_cpu_ids: tuple[int, ...],
        expected_policies: tuple[_PolicyState, ...],
        expected_owner: _ProcessOwner,
    ) -> None:
        journal_path = self._transaction.journal_path
        try:
            payload = self._read_journal_payload()
            if payload is None:
                raise FileNotFoundError("CPU policy journal disappeared")
            journal = self._decode_journal(payload)
            if (
                journal.transaction_id != expected_transaction_id
                or journal.owner != expected_owner
                or not self._topology_matches(
                    journal.online_cpu_ids,
                    journal.policies,
                    expected_online_cpu_ids,
                    expected_policies,
                )
                or not self._policy_states_match_exactly(
                    journal.policies,
                    expected_policies,
                )
            ):
                raise ValueError("CPU policy journal identity or contents changed")
            os.unlink(journal_path)
            _fsync_directory(journal_path.parent)
            try:
                journal_path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise OSError("CPU policy journal still exists after removal")
            self._serialization_evidence()["journal_removed"] = True
            self._serialization_evidence()["journal_published"] = False
            if self._journal_transaction_id == expected_transaction_id:
                self._journal_transaction_id = None
        except BaseException as error:
            self._record_failure(
                _failure(
                    "journal",
                    "remove_verified_journal",
                    error,
                    journal_path,
                )
            )

    def _mark_journal_recoverable(self, reason: str) -> None:
        transaction_id = self._journal_transaction_id
        if transaction_id is None:
            return
        journal_path = self._transaction.journal_path
        temporary_path = journal_path.with_name(
            f".{journal_path.name}.{os.getpid()}.{transaction_id}.recoverable.tmp"
        )
        descriptor: int | None = None
        try:
            payload = self._read_journal_payload()
            if payload is None:
                raise FileNotFoundError("CPU policy journal disappeared")
            journal = self._decode_journal(payload)
            if journal.transaction_id != transaction_id:
                raise ValueError("CPU policy journal transaction ID changed")
            payload["recoverable"] = True
            payload["abandoned_reason"] = reason
            payload_bytes = _canonical_json(payload)
            envelope: JsonObject = {
                "payload": payload,
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
            contents = _canonical_json(envelope) + b"\n"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary_path, flags, 0o600)
            offset = 0
            while offset < len(contents):
                written = os.write(descriptor, contents[offset:])
                if written <= 0:
                    raise OSError("short write while updating CPU policy journal")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary_path, journal_path)
            _fsync_directory(journal_path.parent)
            verified_payload = self._read_journal_payload()
            if verified_payload is None:
                raise FileNotFoundError("updated CPU policy journal disappeared")
            verified = self._decode_journal(verified_payload)
            if verified.transaction_id != transaction_id or not verified.recoverable:
                raise ValueError("CPU policy journal recoverable update did not verify")
            self._serialization_evidence()["journal_marked_recoverable"] = True
        except BaseException as error:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
            self._record_failure(
                _failure(
                    "journal",
                    "mark_journal_recoverable",
                    error,
                    journal_path,
                )
            )

    def _read_policy_topology(
        self,
        phase: str,
    ) -> tuple[
        tuple[int, ...],
        tuple[_PolicyState, ...],
        list[CpuPerformancePolicyFailure],
    ]:
        failures: list[CpuPerformancePolicyFailure] = []
        online_path = self._roots.cpu / "online"
        try:
            online_cpu_ids = _parse_cpu_list(_read_sysfs_value(online_path))
        except Exception as error:
            failures.append(_failure(phase, "read_online_cpus", error, online_path))
            return (), (), failures

        cpufreq_root = self._roots.cpu / "cpufreq"
        try:
            entries = _ordered_directory_entries(cpufreq_root)
        except Exception as error:
            failures.append(
                _failure(phase, "list_policy_directories", error, cpufreq_root)
            )
            return online_cpu_ids, (), failures

        candidates: list[tuple[int, Path]] = []
        for entry in entries:
            match = _POLICY_NAME_PATTERN.fullmatch(entry.name)
            if match is None:
                continue
            try:
                resolved = entry.resolve(strict=True)
                resolved_stat = resolved.stat()
                if not stat.S_ISDIR(resolved_stat.st_mode):
                    raise NotADirectoryError(resolved)
            except Exception as error:
                failures.append(
                    _failure(phase, "resolve_policy_directory", error, entry)
                )
                continue
            candidates.append((int(match.group(1)), resolved))

        policies: list[_PolicyState] = []
        seen_paths: set[Path] = set()
        claimed_cpu_ids: set[int] = set()
        online_cpu_set = set(online_cpu_ids)
        for _, path in sorted(candidates, key=lambda item: (item[0], str(item[1]))):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                policy_stat = path.stat()
                affected_cpu_ids = _parse_cpu_list(
                    _read_sysfs_value(path / "affected_cpus")
                )
                policy_online_cpu_ids = tuple(
                    cpu_id for cpu_id in affected_cpu_ids if cpu_id in online_cpu_set
                )
                if not policy_online_cpu_ids:
                    continue
                overlap = claimed_cpu_ids.intersection(policy_online_cpu_ids)
                if overlap:
                    raise ValueError(
                        "online CPUs belong to multiple unique policies: "
                        f"{sorted(overlap)}"
                    )
                governor = _read_sysfs_value(path / "scaling_governor")
                preference = _read_sysfs_value(path / "energy_performance_preference")
                metadata: dict[str, str] = {}
                for field in (
                    "scaling_driver",
                    "scaling_available_governors",
                    "energy_performance_available_preferences",
                    "related_cpus",
                ):
                    field_path = path / field
                    if _optional_file(field_path):
                        metadata[field] = _read_sysfs_value(field_path)
                available_governors = metadata.get("scaling_available_governors")
                if (
                    available_governors is not None
                    and "performance" not in available_governors.split()
                ):
                    raise ValueError("performance governor is not available")
                available_preferences = metadata.get(
                    "energy_performance_available_preferences"
                )
                if (
                    available_preferences is not None
                    and "performance" not in available_preferences.split()
                ):
                    raise ValueError("performance EPP is not available")
                policies.append(
                    _PolicyState(
                        name=path.name,
                        path=path,
                        device=policy_stat.st_dev,
                        inode=policy_stat.st_ino,
                        affected_cpu_ids=affected_cpu_ids,
                        online_cpu_ids=policy_online_cpu_ids,
                        representative_cpu_id=min(policy_online_cpu_ids),
                        scaling_governor=governor,
                        energy_performance_preference=preference,
                        metadata=metadata,
                    )
                )
                claimed_cpu_ids.update(policy_online_cpu_ids)
            except Exception as error:
                failures.append(_failure(phase, "snapshot_policy", error, path))

        missing_cpu_ids = online_cpu_set.difference(claimed_cpu_ids)
        if missing_cpu_ids:
            failures.append(
                _failure(
                    phase,
                    "verify_online_cpu_policy_coverage",
                    ValueError(
                        "online CPUs lack a unique cpufreq policy: "
                        f"{sorted(missing_cpu_ids)}"
                    ),
                    cpufreq_root,
                )
            )
        if not policies:
            failures.append(
                _failure(
                    phase,
                    "verify_policy_count",
                    ValueError("no online cpufreq policies were discovered"),
                    cpufreq_root,
                )
            )
        identities = [policy.identity_key() for policy in policies]
        if len(identities) != len(set(identities)):
            failures.append(
                _failure(
                    phase,
                    "verify_unique_policy_identities",
                    ValueError("duplicate policy identities were discovered"),
                    cpufreq_root,
                )
            )
        return online_cpu_ids, tuple(policies), failures

    def _discover_policies(self) -> None:
        online_cpu_ids, policies, failures = self._read_policy_topology("discovery")
        for failure in failures:
            self._record_failure(failure)
        if failures:
            return
        self._online_cpu_ids = online_cpu_ids
        self._policies = policies
        self._evidence["online_cpu_ids"] = list(online_cpu_ids)
        self._evidence["policies"] = [policy.as_json() for policy in policies]

    def _capture_and_validate_topology(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> JsonObject:
        online_cpu_ids, policies, topology_failures = self._read_policy_topology(phase)
        failures.extend(topology_failures)
        expected_identities = tuple(
            sorted(policy.identity_key() for policy in self._policies)
        )
        observed_identities = tuple(
            sorted(policy.identity_key() for policy in policies)
        )
        topology_verified = (
            not topology_failures
            and online_cpu_ids == self._online_cpu_ids
            and observed_identities == expected_identities
        )
        if not topology_failures and not topology_verified:
            failures.append(
                _failure(
                    phase,
                    "verify_policy_topology_identity",
                    ValueError(
                        "online CPUs or cpufreq policy identities changed during "
                        "the managed transaction"
                    ),
                    self._roots.cpu,
                )
            )
        return {
            "verified": topology_verified,
            "online_cpu_ids": list(online_cpu_ids),
            "policies": [policy.identity_json() for policy in policies],
        }

    @staticmethod
    def _topology_matches(
        online_cpu_ids: tuple[int, ...],
        policies: tuple[_PolicyState, ...],
        expected_online_cpu_ids: tuple[int, ...],
        expected_policies: tuple[_PolicyState, ...],
    ) -> bool:
        return online_cpu_ids == expected_online_cpu_ids and tuple(
            sorted(policy.identity_key() for policy in policies)
        ) == tuple(sorted(policy.identity_key() for policy in expected_policies))

    @staticmethod
    def _policy_states_match_exactly(
        observed_policies: tuple[_PolicyState, ...],
        expected_policies: tuple[_PolicyState, ...],
    ) -> bool:
        expected = {
            policy.identity_key(): (
                policy.scaling_governor,
                policy.energy_performance_preference,
            )
            for policy in expected_policies
        }
        observed = {
            policy.identity_key(): (
                policy.scaling_governor,
                policy.energy_performance_preference,
            )
            for policy in observed_policies
        }
        return observed == expected and len(observed_policies) == len(observed)

    def _recover_stale_journal(self) -> None:
        journal_path = self._transaction.journal_path
        try:
            payload = self._read_journal_payload()
            if payload is None:
                return
            recovery = self._recovery_evidence()
            recovery["journal_found"] = True
            journal = self._decode_journal(payload)
            recovery["owner"] = journal.owner.as_json()
            recovery["transaction_id"] = journal.transaction_id
            if not journal.recoverable and self._owner_is_live(journal.owner):
                raise RuntimeError(
                    "refusing to recover a CPU policy journal owned by a live "
                    "process identity"
                )

            online_cpu_ids, policies, failures = self._read_policy_topology(
                "stale_recovery"
            )
            for failure in failures:
                self._record_failure(failure)
            if failures:
                return
            if not self._topology_matches(
                online_cpu_ids,
                policies,
                journal.online_cpu_ids,
                journal.policies,
            ):
                raise ValueError(
                    "current CPU policy topology does not match the stale journal"
                )

            failure_count_before_restore = len(self._failures)
            self._restore_policy_states(
                journal.policies,
                phase="stale_recovery_restoration",
            )
            verified_online, verified_policies, verification_failures = (
                self._read_policy_topology("stale_recovery_verification")
            )
            for failure in verification_failures:
                self._record_failure(failure)
            restoration_verified = (
                len(self._failures) == failure_count_before_restore
                and not verification_failures
                and self._topology_matches(
                    verified_online,
                    verified_policies,
                    journal.online_cpu_ids,
                    journal.policies,
                )
                and self._policy_states_match_exactly(
                    verified_policies,
                    journal.policies,
                )
            )
            recovery["performed"] = True
            recovery["restoration_verified"] = restoration_verified
            if not restoration_verified:
                self._record_validation_failure(
                    "stale_recovery",
                    "verify_recovered_policy_values",
                    "stale journal policy values were not restored exactly",
                )
                return
            self._remove_journal(
                journal.transaction_id,
                journal.online_cpu_ids,
                journal.policies,
                journal.owner,
            )
            recovery["journal_removed"] = self._read_journal_payload() is None
        except BaseException as error:
            self._record_failure(
                _failure(
                    "stale_recovery",
                    "recover_abandoned_transaction",
                    error,
                    journal_path,
                )
            )

    def _capture_platform_state(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> JsonObject:
        values: dict[str, str] = {}
        intel_pstate_root = self._roots.cpu / "intel_pstate"
        try:
            intel_pstate_available = _optional_directory(intel_pstate_root)
            if intel_pstate_available:
                entries = _ordered_directory_entries(intel_pstate_root)
                for entry in entries:
                    mode = entry.stat().st_mode
                    if stat.S_ISDIR(mode):
                        continue
                    if not stat.S_ISREG(mode):
                        raise OSError(f"unexpected intel_pstate entry: {entry}")
                    values[f"intel_pstate/{entry.name}"] = _read_sysfs_value(entry)
                if not values:
                    raise ValueError("intel_pstate directory contains no state files")
            boost_path = self._roots.cpu / "cpufreq" / "boost"
            boost_available = _optional_file(boost_path)
            if boost_available:
                values["cpufreq/boost"] = _read_sysfs_value(boost_path)
        except Exception as error:
            failures.append(_failure(phase, "snapshot_intel_pstate_and_turbo", error))
            intel_pstate_available = False
            boost_available = False
        return {
            "intel_pstate_available": intel_pstate_available,
            "generic_boost_available": boost_available,
            "values": values,
        }

    def _capture_policy_values(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> list[JsonValue]:
        values: list[JsonValue] = []
        for policy in self._policies:
            try:
                governor = _read_sysfs_value(policy.path / "scaling_governor")
                preference = _read_sysfs_value(
                    policy.path / "energy_performance_preference"
                )
                values.append(
                    {
                        "name": policy.name,
                        "path": str(policy.path),
                        "scaling_governor": governor,
                        "energy_performance_preference": preference,
                    }
                )
            except Exception as error:
                failures.append(
                    _failure(phase, "snapshot_policy_values", error, policy.path)
                )
        return values

    def _capture_throttle_counters(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> list[JsonValue]:
        observations: list[JsonValue] = []
        for policy in self._policies:
            throttle_root = (
                self._roots.cpu
                / f"cpu{policy.representative_cpu_id}"
                / "thermal_throttle"
            )
            try:
                if not _optional_directory(throttle_root):
                    observations.append(
                        {
                            "policy_name": policy.name,
                            "representative_cpu_id": policy.representative_cpu_id,
                            "available": False,
                            "source_path": str(throttle_root),
                            "values": {},
                        }
                    )
                    continue
                values: dict[str, JsonValue] = {}
                for entry in _ordered_directory_entries(throttle_root):
                    if _THROTTLE_FIELD_PATTERN.fullmatch(
                        entry.name
                    ) is not None and _optional_file(entry):
                        values[entry.name] = _read_verified_integer(entry)
                observations.append(
                    {
                        "policy_name": policy.name,
                        "representative_cpu_id": policy.representative_cpu_id,
                        "available": bool(values),
                        "source_path": str(throttle_root),
                        "values": values,
                    }
                )
            except Exception as error:
                failures.append(
                    _failure(
                        phase,
                        "snapshot_throttle_counters",
                        error,
                        throttle_root,
                    )
                )
        return observations

    def _capture_rapl_limits(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> JsonObject:
        observations: list[JsonValue] = []
        try:
            available = _optional_directory(self._roots.powercap)
            if available:
                entries = _ordered_directory_entries(self._roots.powercap)
                packages = [
                    entry
                    for entry in entries
                    if _RAPL_PACKAGE_PATTERN.fullmatch(entry.name) is not None
                ]
                for package in packages:
                    values: dict[str, str] = {}
                    for entry in _ordered_directory_entries(package):
                        if (
                            entry.name in {"enabled", "name"}
                            or _RAPL_LIMIT_FIELD_PATTERN.fullmatch(entry.name)
                            is not None
                        ) and _optional_file(entry):
                            values[entry.name] = _read_sysfs_value(entry)
                    observations.append(
                        {
                            "domain": package.name,
                            "source_path": str(package),
                            "values": values,
                        }
                    )
            return {
                "available": available and bool(observations),
                "domains": observations,
            }
        except Exception as error:
            failures.append(
                _failure(
                    phase,
                    "snapshot_rapl_limits",
                    error,
                    self._roots.powercap,
                )
            )
            return {"available": False, "domains": observations}

    def _capture_temperature_thresholds(
        self,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> JsonObject:
        observations: list[JsonValue] = []
        try:
            available = _optional_directory(self._roots.hwmon)
            if available:
                for hwmon in _ordered_directory_entries(self._roots.hwmon):
                    if not _optional_directory(hwmon):
                        continue
                    name_path = hwmon / "name"
                    if not _optional_file(name_path):
                        continue
                    name = _read_sysfs_value(name_path)
                    if name != "coretemp":
                        continue
                    values: dict[str, str] = {"name": name}
                    for entry in _ordered_directory_entries(hwmon):
                        if _TEMPERATURE_FIELD_PATTERN.fullmatch(
                            entry.name
                        ) is not None and _optional_file(entry):
                            values[entry.name] = _read_sysfs_value(entry)
                    observations.append(
                        {
                            "hwmon": hwmon.name,
                            "source_path": str(hwmon),
                            "values": values,
                        }
                    )
            return {
                "available": available and bool(observations),
                "devices": observations,
            }
        except Exception as error:
            failures.append(
                _failure(
                    phase,
                    "snapshot_temperature_thresholds",
                    error,
                    self._roots.hwmon,
                )
            )
            return {"available": False, "devices": observations}

    def _capture_snapshot(self, phase: str) -> JsonObject:
        failures: list[CpuPerformancePolicyFailure] = []
        topology = self._capture_and_validate_topology(phase, failures)
        snapshot: JsonObject = {
            "topology": topology,
            "policy_values": self._capture_policy_values(phase, failures),
            "intel_pstate_and_turbo": self._capture_platform_state(phase, failures),
            "representative_throttle_counters": (
                self._capture_throttle_counters(phase, failures)
            ),
            "rapl_limits": self._capture_rapl_limits(phase, failures),
            "temperature_thresholds": self._capture_temperature_thresholds(
                phase, failures
            ),
        }
        for failure in failures:
            self._record_failure(failure)
        return snapshot

    def _write_and_verify(
        self,
        policy: _PolicyState,
        field: str,
        value: str,
        phase: str,
        failures: list[CpuPerformancePolicyFailure],
    ) -> bool:
        path = policy.path / field
        try:
            _write_sysfs_value(path, value)
            observed = _read_sysfs_value(path)
            if observed != value:
                raise ValueError(
                    f"verification mismatch: requested {value!r}, observed {observed!r}"
                )
        except Exception as error:
            failures.append(_failure(phase, f"write_and_verify_{field}", error, path))
            return False
        return True

    def _apply_performance_policy(self) -> None:
        failures: list[CpuPerformancePolicyFailure] = []
        for policy in self._policies:
            governor_applied = self._write_and_verify(
                policy,
                "scaling_governor",
                "performance",
                "application",
                failures,
            )
            if not governor_applied:
                break
            preference_applied = self._write_and_verify(
                policy,
                "energy_performance_preference",
                "performance",
                "application",
                failures,
            )
            if not preference_applied:
                break
        for failure in failures:
            self._record_failure(failure)

    def _restore_policy_states(
        self,
        policies: tuple[_PolicyState, ...],
        *,
        phase: str = "restoration",
    ) -> None:
        failures: list[CpuPerformancePolicyFailure] = []
        for policy in policies:
            for field, value in (
                ("scaling_governor", policy.scaling_governor),
                (
                    "energy_performance_preference",
                    policy.energy_performance_preference,
                ),
            ):
                try:
                    self._write_and_verify(
                        policy,
                        field,
                        value,
                        phase,
                        failures,
                    )
                except BaseException as error:
                    failures.append(
                        _failure(
                            phase,
                            f"write_and_verify_{field}",
                            error,
                            policy.path / field,
                        )
                    )
        for failure in failures:
            self._record_failure(failure)

    def _restore_original_policy(self) -> None:
        self._restore_policy_states(self._policies)

    @staticmethod
    def _policy_values_match(
        snapshot: JsonObject,
        policies: tuple[_PolicyState, ...],
        *,
        performance: bool,
    ) -> bool:
        raw_values = snapshot["policy_values"]
        if not isinstance(raw_values, list) or len(raw_values) != len(policies):
            return False
        expected_by_name = {
            policy.name: (
                ("performance", "performance")
                if performance
                else (
                    policy.scaling_governor,
                    policy.energy_performance_preference,
                )
            )
            for policy in policies
        }
        observed_names: set[str] = set()
        for raw_value in raw_values:
            if not isinstance(raw_value, dict):
                return False
            name = raw_value.get("name")
            if (
                not isinstance(name, str)
                or name not in expected_by_name
                or name in observed_names
            ):
                return False
            observed_names.add(name)
            governor, preference = expected_by_name[name]
            if (
                raw_value.get("scaling_governor") != governor
                or raw_value.get("energy_performance_preference") != preference
            ):
                return False
        return observed_names == set(expected_by_name)

    def _record_validation_failure(
        self,
        phase: str,
        operation: str,
        message: str,
    ) -> None:
        self._record_failure(_failure(phase, operation, ValueError(message)))

    def _validate_static_state(
        self,
        before: JsonObject,
        after: JsonObject,
        *,
        phase: str,
    ) -> None:
        for field in (
            "intel_pstate_and_turbo",
            "rapl_limits",
            "temperature_thresholds",
        ):
            if before[field] != after[field]:
                self._record_validation_failure(
                    phase,
                    f"verify_stable_{field}",
                    f"{field} changed while the performance policy was active",
                )

    def _snapshot_store(self) -> JsonObject:
        snapshots = self._evidence["snapshots"]
        assert isinstance(snapshots, dict)
        return snapshots

    @staticmethod
    def _snapshot_topology_verified(snapshot: JsonObject) -> bool:
        topology = snapshot.get("topology")
        return isinstance(topology, dict) and topology.get("verified") is True

    def _restore_verify_and_finalize_journal(self, snapshot_name: str) -> None:
        try:
            self._restore_original_policy()
        except BaseException as error:
            self._record_failure(
                _failure(
                    "restoration",
                    "restore_original_policy",
                    error,
                    self._roots.cpu,
                )
            )
        try:
            restored = self._capture_snapshot(snapshot_name)
            self._snapshot_store()[snapshot_name] = restored
            restoration_verified = self._policy_values_match(
                restored,
                self._policies,
                performance=False,
            ) and self._snapshot_topology_verified(restored)
        except BaseException as error:
            self._record_failure(
                _failure(
                    "restoration",
                    "capture_restoration_verification",
                    error,
                    self._roots.cpu,
                )
            )
            restoration_verified = False
        self._evidence["restoration_verified"] = restoration_verified
        if not restoration_verified:
            self._record_validation_failure(
                "restoration",
                "verify_restored_policy_values_and_topology",
                "one or more policies or topology identities do not match their "
                "original state",
            )
            return
        transaction_id = self._journal_transaction_id
        if transaction_id is not None:
            if self._owner is None:
                self._record_validation_failure(
                    "journal",
                    "verify_journal_owner_before_removal",
                    "current transaction owner is unavailable",
                )
                return
            self._remove_journal(
                transaction_id,
                self._online_cpu_ids,
                self._policies,
                self._owner,
            )

    def _fail_entry(self, message: str, *, restore: bool) -> None:
        if restore:
            self._restore_verify_and_finalize_journal("restored_after_failed_entry")
        if self._journal_transaction_id is not None:
            self._mark_journal_recoverable(message)
        self._release_transaction_lock()
        self._closed = True
        self._raise(message)

    def __enter__(self) -> CpuPerformancePolicySession:
        if self._entered:
            raise RuntimeError("CPU performance-policy session cannot be reused")
        self._entered = True
        self._evidence["lifecycle"] = "entering"
        self._acquire_transaction_lock()
        if self._failures:
            self._fail_entry(
                "CPU performance-policy serialization failed",
                restore=False,
            )
        try:
            self._recover_stale_journal()
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy stale recovery failed",
                    restore=False,
                )

            self._discover_policies()
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy discovery failed",
                    restore=False,
                )
            before = self._capture_snapshot("before")
            self._snapshot_store()["before"] = before
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy snapshot failed",
                    restore=False,
                )

            self._publish_journal()
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy journal publication failed",
                    restore=self._journal_transaction_id is not None,
                )

            self._apply_performance_policy()
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy application failed",
                    restore=True,
                )

            active = self._capture_snapshot("active")
            self._snapshot_store()["active"] = active
            application_verified = self._policy_values_match(
                active,
                self._policies,
                performance=True,
            ) and self._snapshot_topology_verified(active)
            self._evidence["application_verified"] = application_verified
            if not application_verified:
                self._record_validation_failure(
                    "application",
                    "verify_performance_policy_values_and_topology",
                    "one or more policies are not in performance/performance mode "
                    "or the topology identity changed",
                )
            self._validate_static_state(before, active, phase="application")
            if self._failures:
                self._fail_entry(
                    "CPU performance-policy verification failed",
                    restore=True,
                )

            self._evidence["lifecycle"] = "active"
            return self
        except CpuPerformancePolicyError:
            raise
        except BaseException as error:
            if self._journal_transaction_id is not None:
                self._restore_verify_and_finalize_journal(
                    "restored_after_interrupted_entry"
                )
            if self._journal_transaction_id is not None:
                self._mark_journal_recoverable(
                    "entry interruption cleanup did not verify"
                )
            self._release_transaction_lock()
            self._closed = True
            if self._failures:
                self._evidence["lifecycle"] = "failed"
                raise CpuPerformancePolicyError(
                    "CPU performance-policy entry was interrupted and cleanup failed",
                    tuple(self._failures),
                    self._evidence,
                ) from error
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, traceback
        if not self._entered or self._closed:
            raise RuntimeError("CPU performance-policy session is not active")
        self._closed = True

        internal_interruption: BaseException | None = None
        try:
            before = self._snapshot_store()["before"]
            assert isinstance(before, dict)
            performance_after = self._capture_snapshot("performance_after")
            self._snapshot_store()["performance_after"] = performance_after
            if not (
                self._policy_values_match(
                    performance_after,
                    self._policies,
                    performance=True,
                )
                and self._snapshot_topology_verified(performance_after)
            ):
                self._record_validation_failure(
                    "performance_after",
                    "verify_performance_policy_values_and_topology",
                    "one or more policies drifted from performance/performance "
                    "mode or the topology identity changed",
                )
            self._validate_static_state(
                before,
                performance_after,
                phase="performance_after",
            )
        except BaseException as error:
            internal_interruption = error
        finally:
            try:
                self._restore_verify_and_finalize_journal("restored")
                if self._journal_transaction_id is not None:
                    self._mark_journal_recoverable(
                        "normal or exceptional exit cleanup did not verify"
                    )
            finally:
                self._release_transaction_lock()

        if self._failures:
            self._evidence["lifecycle"] = "failed"
            message = "CPU performance-policy cleanup or verification failed"
            cause = internal_interruption or exception
            if cause is not None:
                message += " after an interruption or managed workload exception"
            raise CpuPerformancePolicyError(
                message,
                tuple(self._failures),
                self._evidence,
            ) from cause
        if internal_interruption is not None:
            raise internal_interruption
        self._evidence["lifecycle"] = "restored"
        return False


def cpu_performance_policy(
    *,
    roots: CpuPerformancePolicySysfsRoots = _DEFAULT_SYSFS_ROOTS,
    transaction: CpuPerformancePolicyTransactionConfig = _DEFAULT_TRANSACTION_CONFIG,
) -> CpuPerformancePolicySession:
    return CpuPerformancePolicySession(roots=roots, transaction=transaction)
