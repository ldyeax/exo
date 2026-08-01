#!/usr/bin/env python3
"""Cleanup-safe ownership and read-only attachment for resident SGLang clusters.

This module deliberately does not detach rank processes from their launching
controller. The controller keeps every local process handle and remote SSH
supervisor alive, so controller loss retains the existing fail-closed cleanup
behavior. A second benchmark process may attach for HTTP work only after it
proves all of the following:

* it possesses an owner-only explicit receipt;
* its expected launch contract has the exact receipt digest;
* the original controller PID and start time still match; and
* every rank still has the receipt's process-group, PID start time, namespace,
  and secret owner token.

An attachment never receives signaling authority. Cleanup remains exclusively
with the controller that created the lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, Self, cast

sys.dont_write_bytecode = True

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ResidentClusterStatus = Literal["running", "stopped", "cleanup_failed"]

RESIDENT_CLUSTER_RECEIPT_SCHEMA_VERSION: Final = 1
_MAXIMUM_RECEIPT_BYTES: Final = 1024 * 1024
_SAFE_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")
_SAFE_OWNER_TOKEN: Final = re.compile(r"^[A-Za-z0-9_.-]{16,256}$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class ResidentClusterError(RuntimeError):
    """Base error for resident-cluster ownership failures."""


class ResidentClusterAttachmentError(ResidentClusterError):
    """Raised when an explicit resident-cluster attachment cannot be proven."""


class ResidentClusterCleanupError(ResidentClusterError):
    """Raised when the owning controller cannot prove complete cleanup."""


class OwnedStageLike(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def host_name(self) -> str: ...

    @property
    def pid(self) -> int: ...

    @property
    def process_group_id(self) -> int: ...

    @property
    def start_time_ticks(self) -> int: ...

    @property
    def owner_token(self) -> str: ...

    @property
    def ownership_namespace(self) -> str: ...

    @property
    def remote(self) -> bool: ...


class RunningStageLike(Protocol):
    @property
    def owned(self) -> OwnedStageLike: ...


class CleanupReceiptLike(Protocol):
    @property
    def ownership_verified(self) -> bool: ...

    @property
    def terminated(self) -> bool: ...

    @property
    def forced(self) -> bool: ...

    @property
    def error(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ResidentStageOwnership:
    pipeline_rank: int
    node_id: str
    service_endpoint: str
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    owner_token: str
    ownership_namespace: str
    remote: bool

    @classmethod
    def from_owned_stage(
        cls,
        owned: OwnedStageLike,
        *,
        node_id: str,
        service_endpoint: str,
    ) -> Self:
        return cls(
            pipeline_rank=owned.rank,
            node_id=node_id,
            service_endpoint=service_endpoint,
            host_name=owned.host_name,
            pid=owned.pid,
            process_group_id=owned.process_group_id,
            start_time_ticks=owned.start_time_ticks,
            owner_token=owned.owner_token,
            ownership_namespace=owned.ownership_namespace,
            remote=owned.remote,
        )


@dataclass(frozen=True, slots=True)
class ResidentStageVerification:
    pipeline_rank: int
    ownership_verified: bool
    alive: bool
    process_group_members: tuple[int, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ResidentStageCleanupEvidence:
    pipeline_rank: int
    ownership_verified: bool
    terminated: bool
    forced: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class ResidentClusterOwnershipReceipt:
    schema_version: int
    kind: str
    status: ResidentClusterStatus
    run_id: str
    created_at_utc: str
    completed_at_utc: str | None
    controller_pid: int
    controller_start_time_ticks: int
    launch_contract_sha256: str
    owner_token: str
    stages: tuple[ResidentStageOwnership, ...]
    cleanup: tuple[ResidentStageCleanupEvidence, ...]

    def redacted_json(self) -> JsonObject:
        receipt = _receipt_to_json(self)
        receipt["owner_token"] = hashlib.sha256(self.owner_token.encode()).hexdigest()
        raw_stages = cast(list[JsonValue], receipt["stages"])
        for raw_stage in raw_stages:
            stage = cast(JsonObject, raw_stage)
            stage["owner_token"] = hashlib.sha256(
                cast(str, stage["owner_token"]).encode()
            ).hexdigest()
        return receipt


@dataclass(frozen=True, slots=True)
class VerifiedResidentClusterAttachment:
    receipt_path: Path
    receipt: ResidentClusterOwnershipReceipt
    verified_at_utc: str
    stage_verifications: tuple[ResidentStageVerification, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def launch_contract_sha256(launch_contract: JsonObject) -> str:
    """Hash the caller's complete, secret-free launch contract."""

    return hashlib.sha256(_canonical_json_bytes(launch_contract)).hexdigest()


def _read_process_identity(pid: int) -> tuple[int, int, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[2]), int(fields[3]), int(fields[19])


def _validate_stage(stage: ResidentStageOwnership) -> None:
    if isinstance(stage.pipeline_rank, bool) or stage.pipeline_rank < 0:
        raise ResidentClusterError("resident stage rank must be non-negative")
    if not stage.node_id or not stage.host_name or not stage.service_endpoint:
        raise ResidentClusterError("resident stage identity fields cannot be empty")
    if (
        isinstance(stage.pid, bool)
        or isinstance(stage.process_group_id, bool)
        or isinstance(stage.start_time_ticks, bool)
        or stage.pid <= 0
        or stage.process_group_id <= 0
        or stage.start_time_ticks <= 0
    ):
        raise ResidentClusterError("resident stage process identity must be positive")
    if stage.pid != stage.process_group_id:
        raise ResidentClusterError(
            "resident stage must be its owned process-group leader"
        )
    if _SAFE_OWNER_TOKEN.fullmatch(stage.owner_token) is None:
        raise ResidentClusterError("resident stage owner token is unsafe")
    if not stage.ownership_namespace:
        raise ResidentClusterError("resident stage ownership namespace is empty")


def _validate_receipt(receipt: ResidentClusterOwnershipReceipt) -> None:
    if receipt.schema_version != RESIDENT_CLUSTER_RECEIPT_SCHEMA_VERSION:
        raise ResidentClusterError("unsupported resident-cluster receipt schema")
    if receipt.kind != "sglang_kt_resident_cluster_ownership":
        raise ResidentClusterError("invalid resident-cluster receipt kind")
    if receipt.status not in {"running", "stopped", "cleanup_failed"}:
        raise ResidentClusterError("invalid resident-cluster receipt status")
    if _SAFE_IDENTIFIER.fullmatch(receipt.run_id) is None:
        raise ResidentClusterError("resident-cluster run_id is unsafe")
    if (
        isinstance(receipt.controller_pid, bool)
        or isinstance(receipt.controller_start_time_ticks, bool)
        or receipt.controller_pid <= 0
        or receipt.controller_start_time_ticks <= 0
    ):
        raise ResidentClusterError("resident controller identity must be positive")
    if _SHA256.fullmatch(receipt.launch_contract_sha256) is None:
        raise ResidentClusterError("resident launch contract digest is invalid")
    if _SAFE_OWNER_TOKEN.fullmatch(receipt.owner_token) is None:
        raise ResidentClusterError("resident owner token is unsafe")
    if not receipt.stages:
        raise ResidentClusterError("resident-cluster receipt has no stages")
    for stage in receipt.stages:
        _validate_stage(stage)
        if stage.owner_token != receipt.owner_token:
            raise ResidentClusterError(
                "resident stage token does not match cluster owner token"
            )
    ranks = tuple(stage.pipeline_rank for stage in receipt.stages)
    if ranks != tuple(sorted(set(ranks))):
        raise ResidentClusterError(
            "resident-cluster stages must have sorted, unique ranks"
        )
    cleanup_ranks = tuple(item.pipeline_rank for item in receipt.cleanup)
    if cleanup_ranks != tuple(sorted(set(cleanup_ranks))):
        raise ResidentClusterError(
            "resident cleanup evidence must have sorted, unique ranks"
        )
    if receipt.status == "running":
        if receipt.completed_at_utc is not None or receipt.cleanup:
            raise ResidentClusterError(
                "running resident receipt cannot contain cleanup evidence"
            )
    elif receipt.completed_at_utc is None:
        raise ResidentClusterError("terminal resident receipt requires completion time")


def _receipt_to_json(receipt: ResidentClusterOwnershipReceipt) -> JsonObject:
    return {
        "schema_version": receipt.schema_version,
        "kind": receipt.kind,
        "status": receipt.status,
        "run_id": receipt.run_id,
        "created_at_utc": receipt.created_at_utc,
        "completed_at_utc": receipt.completed_at_utc,
        "controller_pid": receipt.controller_pid,
        "controller_start_time_ticks": receipt.controller_start_time_ticks,
        "launch_contract_sha256": receipt.launch_contract_sha256,
        "owner_token": receipt.owner_token,
        "stages": [cast(JsonObject, asdict(stage)) for stage in receipt.stages],
        "cleanup": [cast(JsonObject, asdict(evidence)) for evidence in receipt.cleanup],
    }


def _required(raw: MappingLike, field: str, expected_type: type[object]) -> object:
    value = raw.get(field)
    if not isinstance(value, expected_type):
        raise ResidentClusterError(
            f"resident-cluster receipt field {field} has the wrong type"
        )
    return value


def _required_integer(raw: MappingLike, field: str) -> int:
    value = raw.get(field)
    if type(value) is not int:
        raise ResidentClusterError(
            f"resident-cluster receipt field {field} has the wrong type"
        )
    return cast(int, value)


def _required_boolean(raw: MappingLike, field: str) -> bool:
    value = raw.get(field)
    if type(value) is not bool:
        raise ResidentClusterError(
            f"resident-cluster receipt field {field} has the wrong type"
        )
    return cast(bool, value)


class MappingLike(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def _stage_from_json(raw: object) -> ResidentStageOwnership:
    if not isinstance(raw, dict) or set(raw) != {
        "pipeline_rank",
        "node_id",
        "service_endpoint",
        "host_name",
        "pid",
        "process_group_id",
        "start_time_ticks",
        "owner_token",
        "ownership_namespace",
        "remote",
    }:
        raise ResidentClusterError("resident stage receipt has an invalid schema")
    return ResidentStageOwnership(
        pipeline_rank=_required_integer(raw, "pipeline_rank"),
        node_id=cast(str, _required(raw, "node_id", str)),
        service_endpoint=cast(str, _required(raw, "service_endpoint", str)),
        host_name=cast(str, _required(raw, "host_name", str)),
        pid=_required_integer(raw, "pid"),
        process_group_id=_required_integer(raw, "process_group_id"),
        start_time_ticks=_required_integer(raw, "start_time_ticks"),
        owner_token=cast(str, _required(raw, "owner_token", str)),
        ownership_namespace=cast(
            str,
            _required(raw, "ownership_namespace", str),
        ),
        remote=_required_boolean(raw, "remote"),
    )


def _cleanup_from_json(raw: object) -> ResidentStageCleanupEvidence:
    if not isinstance(raw, dict) or set(raw) != {
        "pipeline_rank",
        "ownership_verified",
        "terminated",
        "forced",
        "error",
    }:
        raise ResidentClusterError("resident cleanup evidence has an invalid schema")
    error = raw.get("error")
    if error is not None and not isinstance(error, str):
        raise ResidentClusterError("resident cleanup error has the wrong type")
    return ResidentStageCleanupEvidence(
        pipeline_rank=_required_integer(raw, "pipeline_rank"),
        ownership_verified=_required_boolean(raw, "ownership_verified"),
        terminated=_required_boolean(raw, "terminated"),
        forced=_required_boolean(raw, "forced"),
        error=error,
    )


def _receipt_from_json(raw: object) -> ResidentClusterOwnershipReceipt:
    expected_fields = {
        "schema_version",
        "kind",
        "status",
        "run_id",
        "created_at_utc",
        "completed_at_utc",
        "controller_pid",
        "controller_start_time_ticks",
        "launch_contract_sha256",
        "owner_token",
        "stages",
        "cleanup",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ResidentClusterError(
            "resident-cluster ownership receipt has an invalid schema"
        )
    completed_at = raw.get("completed_at_utc")
    if completed_at is not None and not isinstance(completed_at, str):
        raise ResidentClusterError("resident completion time has the wrong type")
    raw_stages = _required(raw, "stages", list)
    raw_cleanup = _required(raw, "cleanup", list)
    status = cast(str, _required(raw, "status", str))
    if status not in {"running", "stopped", "cleanup_failed"}:
        raise ResidentClusterError("invalid resident-cluster receipt status")
    receipt = ResidentClusterOwnershipReceipt(
        schema_version=_required_integer(raw, "schema_version"),
        kind=cast(str, _required(raw, "kind", str)),
        status=cast(ResidentClusterStatus, status),
        run_id=cast(str, _required(raw, "run_id", str)),
        created_at_utc=cast(str, _required(raw, "created_at_utc", str)),
        completed_at_utc=completed_at,
        controller_pid=_required_integer(raw, "controller_pid"),
        controller_start_time_ticks=_required_integer(
            raw,
            "controller_start_time_ticks",
        ),
        launch_contract_sha256=cast(
            str,
            _required(raw, "launch_contract_sha256", str),
        ),
        owner_token=cast(str, _required(raw, "owner_token", str)),
        stages=tuple(_stage_from_json(item) for item in cast(list[object], raw_stages)),
        cleanup=tuple(
            _cleanup_from_json(item) for item in cast(list[object], raw_cleanup)
        ),
    )
    _validate_receipt(receipt)
    return receipt


def _write_owner_only_receipt(
    path: Path,
    receipt: ResidentClusterOwnershipReceipt,
) -> None:
    _validate_receipt(receipt)
    if not path.is_absolute():
        raise ResidentClusterError("resident receipt path must be absolute")
    parent = path.parent.resolve(strict=True)
    if parent != path.parent:
        raise ResidentClusterError(
            "resident receipt parent must be canonical and contain no symlinks"
        )
    encoded = (
        json.dumps(
            _receipt_to_json(receipt),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    if len(encoded) > _MAXIMUM_RECEIPT_BYTES:
        raise ResidentClusterError("resident receipt exceeds the size limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def load_resident_cluster_ownership_receipt(
    path: Path,
) -> ResidentClusterOwnershipReceipt:
    """Load a private explicit receipt without following a final symlink."""

    if not path.is_absolute():
        raise ResidentClusterAttachmentError("resident receipt path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ResidentClusterAttachmentError(
            f"cannot open resident receipt {path}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ResidentClusterAttachmentError(
                "resident receipt must be a regular file"
            )
        if file_stat.st_uid != os.getuid() or file_stat.st_mode & 0o077:
            raise ResidentClusterAttachmentError(
                "resident receipt must be owned by this user and mode 0600"
            )
        if file_stat.st_size > _MAXIMUM_RECEIPT_BYTES:
            raise ResidentClusterAttachmentError(
                "resident receipt exceeds the size limit"
            )
        contents = bytearray()
        while chunk := os.read(descriptor, 64 * 1024):
            contents.extend(chunk)
            if len(contents) > _MAXIMUM_RECEIPT_BYTES:
                raise ResidentClusterAttachmentError(
                    "resident receipt exceeds the size limit"
                )
    finally:
        os.close(descriptor)
    try:
        raw = cast(object, json.loads(contents))
        return _receipt_from_json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ResidentClusterError) as error:
        raise ResidentClusterAttachmentError(
            f"invalid resident receipt {path}"
        ) from error


def verify_local_resident_stage(
    stage: ResidentStageOwnership,
) -> ResidentStageVerification:
    """Verify a local stage by exact Linux process identity and secret token."""

    if stage.remote:
        return ResidentStageVerification(
            pipeline_rank=stage.pipeline_rank,
            ownership_verified=False,
            alive=False,
            process_group_members=(),
            error="local verifier refuses a remote stage",
        )
    members: list[int] = []
    leader_seen = False
    try:
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
            except (OSError, IndexError, ValueError) as error:
                raise ResidentClusterAttachmentError(
                    "cannot enumerate local process ownership"
                ) from error
            if process_group_id != stage.process_group_id:
                continue
            pid = int(entry.name)
            if session_id != stage.pid:
                raise ResidentClusterAttachmentError(
                    f"rank {stage.pipeline_rank} process group escaped its session"
                )
            if state == "Z":
                continue
            if pid == stage.pid:
                leader_seen = True
                if start_time_ticks != stage.start_time_ticks:
                    raise ResidentClusterAttachmentError(
                        f"rank {stage.pipeline_rank} PID start time changed"
                    )
                environment = (entry / "environ").read_bytes().split(b"\0")
                owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={stage.owner_token}".encode()
                command_line = (entry / "cmdline").read_bytes()
                if owner_entry not in environment:
                    raise ResidentClusterAttachmentError(
                        f"rank {stage.pipeline_rank} owner token changed"
                    )
                if stage.ownership_namespace.encode() not in command_line:
                    raise ResidentClusterAttachmentError(
                        f"rank {stage.pipeline_rank} namespace changed"
                    )
            members.append(pid)
    except (OSError, ResidentClusterAttachmentError) as error:
        return ResidentStageVerification(
            pipeline_rank=stage.pipeline_rank,
            ownership_verified=False,
            alive=False,
            process_group_members=tuple(sorted(members)),
            error=f"{type(error).__name__}: {error}",
        )
    verified = leader_seen and bool(members)
    return ResidentStageVerification(
        pipeline_rank=stage.pipeline_rank,
        ownership_verified=verified,
        alive=verified,
        process_group_members=tuple(sorted(members)),
        error=None if verified else "owned local process-group leader is not alive",
    )


_REMOTE_VERIFY_PROGRAM: Final = r"""
import json
import sys
from pathlib import Path

stage = json.loads(sys.argv[1])
result = {
    "pipeline_rank": stage["pipeline_rank"],
    "ownership_verified": False,
    "alive": False,
    "process_group_members": [],
    "error": None,
}
try:
    members = []
    leader_seen = False
    owner_entry = (
        "EXO_BENCHMARK_OWNER_TOKEN=" + stage["owner_token"]
    ).encode()
    namespace = stage["ownership_namespace"].encode()
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
        except (OSError, IndexError, ValueError) as error:
            raise RuntimeError("cannot enumerate remote process ownership") from error
        if process_group_id != stage["process_group_id"]:
            continue
        pid = int(entry.name)
        if session_id != stage["pid"]:
            raise RuntimeError("remote process group escaped its owned session")
        if state == "Z":
            continue
        if pid == stage["pid"]:
            leader_seen = True
            if start_time_ticks != stage["start_time_ticks"]:
                raise RuntimeError("remote PID start time changed")
            environment = (entry / "environ").read_bytes().split(b"\0")
            command_line = (entry / "cmdline").read_bytes()
            if owner_entry not in environment:
                raise RuntimeError("remote owner token changed")
            if namespace not in command_line:
                raise RuntimeError("remote ownership namespace changed")
        members.append(pid)
    verified = leader_seen and bool(members)
    result.update({
        "ownership_verified": verified,
        "alive": verified,
        "process_group_members": sorted(members),
        "error": None if verified else "owned remote process-group leader is not alive",
    })
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(result, sort_keys=True))
"""


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
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "--",
        ssh_target,
        shlex.join(command),
    )


@dataclass(frozen=True, slots=True)
class TwoHostResidentStageVerifier:
    ssh_target: str
    remote_python: str
    timeout_seconds: float = 30.0

    def __call__(
        self,
        stage: ResidentStageOwnership,
    ) -> ResidentStageVerification:
        if not stage.remote:
            return verify_local_resident_stage(stage)
        command = (
            self.remote_python,
            "-c",
            _REMOTE_VERIFY_PROGRAM,
            json.dumps(asdict(stage), sort_keys=True),
        )
        try:
            completed = subprocess.run(
                _ssh_argv(self.ssh_target, command),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise ResidentClusterAttachmentError(
                    f"remote ownership verification failed: {completed.stderr[-500:]}"
                )
            raw = cast(object, json.loads(completed.stdout))
            if not isinstance(raw, dict):
                raise ResidentClusterAttachmentError(
                    "remote ownership verification is not an object"
                )
            members = raw.get("process_group_members")
            error = raw.get("error")
            if (
                not isinstance(raw.get("pipeline_rank"), int)
                or not isinstance(raw.get("ownership_verified"), bool)
                or not isinstance(raw.get("alive"), bool)
                or not isinstance(members, list)
                or any(not isinstance(member, int) for member in members)
                or (error is not None and not isinstance(error, str))
            ):
                raise ResidentClusterAttachmentError(
                    "remote ownership verification has an invalid schema"
                )
            verification = ResidentStageVerification(
                pipeline_rank=cast(int, raw["pipeline_rank"]),
                ownership_verified=cast(bool, raw["ownership_verified"]),
                alive=cast(bool, raw["alive"]),
                process_group_members=tuple(cast(list[int], members)),
                error=cast(str | None, error),
            )
            if verification.pipeline_rank != stage.pipeline_rank:
                raise ResidentClusterAttachmentError(
                    "remote verifier returned another pipeline rank"
                )
            return verification
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            ResidentClusterAttachmentError,
        ) as error:
            return ResidentStageVerification(
                pipeline_rank=stage.pipeline_rank,
                ownership_verified=False,
                alive=False,
                process_group_members=(),
                error=f"{type(error).__name__}: {error}",
            )


StageVerifier = Callable[[ResidentStageOwnership], ResidentStageVerification]


def verify_resident_cluster_attachment(
    receipt_path: Path,
    *,
    expected_launch_contract: JsonObject,
    stage_verifier: StageVerifier,
) -> VerifiedResidentClusterAttachment:
    """Verify an explicit receipt and return a read-only attachment."""

    receipt = load_resident_cluster_ownership_receipt(receipt_path)
    if receipt.status != "running":
        raise ResidentClusterAttachmentError(
            f"resident cluster is not running: {receipt.status}"
        )
    expected_digest = launch_contract_sha256(expected_launch_contract)
    if receipt.launch_contract_sha256 != expected_digest:
        raise ResidentClusterAttachmentError(
            "resident launch contract does not match the requested benchmark"
        )
    try:
        _, _, controller_start_time = _read_process_identity(receipt.controller_pid)
    except (OSError, IndexError, ValueError) as error:
        raise ResidentClusterAttachmentError(
            "resident controller is not alive"
        ) from error
    if controller_start_time != receipt.controller_start_time_ticks:
        raise ResidentClusterAttachmentError(
            "resident controller PID start time changed"
        )
    verifications = tuple(stage_verifier(stage) for stage in receipt.stages)
    if any(
        verification.pipeline_rank != stage.pipeline_rank
        or not verification.ownership_verified
        or not verification.alive
        for stage, verification in zip(
            receipt.stages,
            verifications,
            strict=True,
        )
    ):
        failures = "; ".join(
            f"rank {verification.pipeline_rank}: "
            f"{verification.error or 'ownership not verified'}"
            for verification in verifications
            if not verification.ownership_verified or not verification.alive
        )
        raise ResidentClusterAttachmentError(
            f"resident rank ownership verification failed: {failures}"
        )
    return VerifiedResidentClusterAttachment(
        receipt_path=receipt_path,
        receipt=receipt,
        verified_at_utc=_utc_now(),
        stage_verifications=verifications,
    )


type StageStarter[SpecT, RunningT] = Callable[[SpecT, str], RunningT]
type StageStopper[RunningT, CleanupT] = Callable[[RunningT], CleanupT]
type StageOwnershipBuilder[SpecT, RunningT] = Callable[
    [SpecT, RunningT],
    ResidentStageOwnership,
]
type CleanupEvidenceBuilder[RunningT, CleanupT] = Callable[
    [RunningT, CleanupT],
    ResidentStageCleanupEvidence,
]


class ResidentClusterLifecycle[SpecT, RunningT, CleanupT]:
    """Own ranks once so multiple benchmark phases can reuse warm processes."""

    def __init__(
        self,
        *,
        run_id: str,
        receipt_path: Path,
        launch_contract: JsonObject,
        specs: Sequence[SpecT],
        start_stage: StageStarter[SpecT, RunningT],
        stop_stage: StageStopper[RunningT, CleanupT],
        build_stage_ownership: StageOwnershipBuilder[SpecT, RunningT],
        build_cleanup_evidence: CleanupEvidenceBuilder[RunningT, CleanupT],
    ) -> None:
        if _SAFE_IDENTIFIER.fullmatch(run_id) is None:
            raise ValueError("resident-cluster run_id is unsafe")
        if not specs:
            raise ValueError("resident cluster requires at least one stage")
        self.run_id = run_id
        self.receipt_path = receipt_path
        self.launch_contract = launch_contract
        self.specs = tuple(specs)
        self._start_stage = start_stage
        self._stop_stage = stop_stage
        self._build_stage_ownership = build_stage_ownership
        self._build_cleanup_evidence = build_cleanup_evidence
        self._owner_token = uuid.uuid4().hex
        self._running: list[RunningT] = []
        self._created_at_utc: str | None = None
        self._receipt: ResidentClusterOwnershipReceipt | None = None
        self._cleanup: tuple[ResidentStageCleanupEvidence, ...] | None = None
        self._state: Literal["new", "running", "stopped", "cleanup_failed"] = "new"

    @property
    def running(self) -> tuple[RunningT, ...]:
        if self._state != "running":
            raise ResidentClusterError("resident cluster is not running")
        return tuple(self._running)

    @property
    def ownership_receipt(self) -> ResidentClusterOwnershipReceipt:
        if self._receipt is None:
            raise ResidentClusterError(
                "resident ownership receipt has not been published"
            )
        return self._receipt

    @property
    def cleanup(self) -> tuple[ResidentStageCleanupEvidence, ...]:
        if self._cleanup is None:
            raise ResidentClusterError("resident cluster has not been stopped")
        return self._cleanup

    def start(self) -> tuple[RunningT, ...]:
        if self._state != "new":
            raise ResidentClusterError("resident lifecycle can only be started once")
        self._created_at_utc = _utc_now()
        try:
            for spec in self.specs:
                self._running.append(self._start_stage(spec, self._owner_token))
            ownership = tuple(
                sorted(
                    (
                        self._build_stage_ownership(spec, running)
                        for spec, running in zip(
                            self.specs,
                            self._running,
                            strict=True,
                        )
                    ),
                    key=lambda stage: stage.pipeline_rank,
                )
            )
            _, _, controller_start_time = _read_process_identity(os.getpid())
            receipt = ResidentClusterOwnershipReceipt(
                schema_version=RESIDENT_CLUSTER_RECEIPT_SCHEMA_VERSION,
                kind="sglang_kt_resident_cluster_ownership",
                status="running",
                run_id=self.run_id,
                created_at_utc=self._created_at_utc,
                completed_at_utc=None,
                controller_pid=os.getpid(),
                controller_start_time_ticks=controller_start_time,
                launch_contract_sha256=launch_contract_sha256(self.launch_contract),
                owner_token=self._owner_token,
                stages=ownership,
                cleanup=(),
            )
            _write_owner_only_receipt(self.receipt_path, receipt)
            self._receipt = receipt
            self._state = "running"
            return tuple(self._running)
        except BaseException as start_error:
            cleanup_error: BaseException | None = None
            if self._running:
                try:
                    self._stop_started_stages()
                except BaseException as error:
                    cleanup_error = error
            if cleanup_error is not None:
                raise ExceptionGroup(
                    "resident cluster start and partial cleanup failed",
                    [start_error, cleanup_error],
                ) from None
            raise

    def _stop_started_stages(
        self,
    ) -> tuple[ResidentStageCleanupEvidence, ...]:
        evidence: list[ResidentStageCleanupEvidence] = []
        stop_errors: list[BaseException] = []
        for running in reversed(self._running):
            try:
                cleanup = self._stop_stage(running)
                evidence.append(self._build_cleanup_evidence(running, cleanup))
            except BaseException as error:
                stop_errors.append(error)
                rank = cast(RunningStageLike, running).owned.rank
                evidence.append(
                    ResidentStageCleanupEvidence(
                        pipeline_rank=rank,
                        ownership_verified=False,
                        terminated=False,
                        forced=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        ordered = tuple(sorted(evidence, key=lambda item: item.pipeline_rank))
        self._cleanup = ordered
        complete = (
            not stop_errors
            and len(ordered) == len(self._running)
            and all(item.ownership_verified and item.terminated for item in ordered)
        )
        self._state = "stopped" if complete else "cleanup_failed"
        if not complete:
            details = "; ".join(
                f"rank {item.pipeline_rank}: {item.error or 'cleanup incomplete'}"
                for item in ordered
                if not item.ownership_verified or not item.terminated
            )
            raise ResidentClusterCleanupError(
                f"resident cluster cleanup is incomplete: {details}"
            )
        return ordered

    def stop(self) -> tuple[ResidentStageCleanupEvidence, ...]:
        if self._state == "new":
            raise ResidentClusterError("resident cluster was never started")
        if self._state in {"stopped", "cleanup_failed"}:
            assert self._cleanup is not None
            if self._state == "cleanup_failed":
                raise ResidentClusterCleanupError(
                    "resident cluster cleanup previously failed"
                )
            return self._cleanup
        cleanup_error: BaseException | None = None
        try:
            cleanup = self._stop_started_stages()
        except BaseException as error:
            cleanup_error = error
            assert self._cleanup is not None
            cleanup = self._cleanup

        assert self._receipt is not None
        terminal_receipt = replace(
            self._receipt,
            status=cast(ResidentClusterStatus, self._state),
            completed_at_utc=_utc_now(),
            cleanup=cleanup,
        )
        terminal_path = self.receipt_path.with_name(
            f"{self.receipt_path.name}.{self._state}.json"
        )
        terminal_write_error: BaseException | None = None
        try:
            _write_owner_only_receipt(terminal_path, terminal_receipt)
            if self._state == "stopped":
                self.receipt_path.unlink()
        except BaseException as error:
            terminal_write_error = error
        self._receipt = terminal_receipt
        if cleanup_error is not None and terminal_write_error is not None:
            raise ExceptionGroup(
                "resident cleanup and terminal receipt publication failed",
                [cleanup_error, terminal_write_error],
            )
        if cleanup_error is not None:
            raise cleanup_error
        if terminal_write_error is not None:
            raise terminal_write_error
        return cleanup

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        del exception_type, traceback
        try:
            self.stop()
        except BaseException as cleanup_error:
            if exception is not None:
                raise ExceptionGroup(
                    "resident benchmark and cleanup both failed",
                    [exception, cleanup_error],
                ) from None
            raise
        return False


def cleanup_evidence_from_receipt(
    running: RunningStageLike,
    cleanup: CleanupReceiptLike,
) -> ResidentStageCleanupEvidence:
    """Adapt the existing PP lifecycle cleanup receipt."""

    return ResidentStageCleanupEvidence(
        pipeline_rank=running.owned.rank,
        ownership_verified=cleanup.ownership_verified,
        terminated=cleanup.terminated,
        forced=cleanup.forced,
        error=cleanup.error,
    )
