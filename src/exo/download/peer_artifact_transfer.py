from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from fractions import Fraction
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, NewType, Protocol, final

from filelock import AsyncFileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from exo.shared.types.common import Host, NodeId

DEFAULT_ARTIFACT_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
PEER_ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
_PEER_ARTIFACT_MANIFEST_CANONICALIZATION = "exo-peer-artifact-manifest-v1"
_RESUME_JOURNAL_SCHEMA_VERSION = 1
_VERIFIED_ARTIFACT_RECEIPT_SCHEMA_VERSION = 1
_FILE_COPY_BUFFER_BYTES = 8 * 1024 * 1024

PeerArtifactLinkId = NewType("PeerArtifactLinkId", str)
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PeerArtifactSnapshotId = NewType("PeerArtifactSnapshotId", str)
RelativeArtifactPath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=4096),
]
PeerArtifactLinkMedium = Literal["infiniband", "ethernet"]
PeerArtifactStorageKind = Literal["disk_cache", "memory_cache"]


class PeerArtifactTransferError(Exception):
    """Base exception for peer artifact planning and transfer failures."""


class PeerArtifactIntegrityError(PeerArtifactTransferError):
    """Raised when peer bytes disagree with their content-addressed manifest."""


class PeerArtifactProtocolError(PeerArtifactTransferError):
    """Raised when a peer range reader violates the requested range contract."""


class InsufficientPeerArtifactStorageError(PeerArtifactTransferError):
    """Raised when neither configured destination has sufficient observed capacity."""


class PeerArtifactStorageConfigurationError(PeerArtifactTransferError):
    """Raised when a configured memory cache is not on a RAM-backed filesystem."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@final
class PeerArtifactChunk(_StrictModel):
    offset_bytes: NonNegativeInt
    size_bytes: PositiveInt
    sha256: Sha256Digest


def _validate_relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact paths must be normalized relative POSIX paths")
    return value


@final
class PeerArtifactManifest(_StrictModel):
    schema_version: Literal[1]
    canonicalization: Literal["exo-peer-artifact-manifest-v1"]
    artifact_path: RelativeArtifactPath
    size_bytes: NonNegativeInt
    sha256: Sha256Digest
    chunks: tuple[PeerArtifactChunk, ...]

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def validate_chunks(self) -> PeerArtifactManifest:
        expected_offset = 0
        for chunk in self.chunks:
            if chunk.offset_bytes != expected_offset:
                raise ValueError("artifact chunks must be contiguous and ordered")
            expected_offset += chunk.size_bytes
        if expected_offset != self.size_bytes:
            raise ValueError("artifact chunks must cover the declared artifact size")
        if self.size_bytes == 0 and self.chunks:
            raise ValueError("an empty artifact cannot declare chunks")
        if self.size_bytes > 0 and not self.chunks:
            raise ValueError("a nonempty artifact must declare chunks")
        return self


@final
class PeerArtifactLink(_StrictModel):
    link_id: PeerArtifactLinkId
    peer_node_id: NodeId
    medium: PeerArtifactLinkMedium
    local_interface: str
    local_ip_address: str
    peer_endpoint: Host
    estimated_bytes_per_second: PositiveInt
    maximum_concurrent_chunks: PositiveInt = 1

    @field_validator("link_id")
    @classmethod
    def validate_link_id(cls, value: PeerArtifactLinkId) -> PeerArtifactLinkId:
        if not str(value).strip():
            raise ValueError("peer artifact link IDs cannot be empty")
        return value

    @field_validator("local_interface")
    @classmethod
    def validate_local_interface(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("peer artifact local interfaces cannot be empty")
        return value

    @field_validator("local_ip_address")
    @classmethod
    def validate_local_ip_address(cls, value: str) -> str:
        parsed = ip_address(value)
        if parsed.is_unspecified or parsed.is_multicast:
            raise ValueError("peer artifact links require a concrete local IP address")
        return value

    @field_validator("peer_endpoint")
    @classmethod
    def validate_peer_endpoint(cls, value: Host) -> Host:
        parsed = ip_address(value.ip)
        if parsed.is_unspecified or parsed.is_multicast or value.port == 0:
            raise ValueError(
                "peer artifact links require a concrete peer IP address and port"
            )
        return value


@final
class PeerArtifactChunkAssignment(_StrictModel):
    chunk_index: NonNegativeInt
    link_id: PeerArtifactLinkId


@final
class PeerArtifactTransferPlan(_StrictModel):
    manifest: PeerArtifactManifest
    links: tuple[PeerArtifactLink, ...]
    completed_chunk_indexes: tuple[NonNegativeInt, ...]
    assignments: tuple[PeerArtifactChunkAssignment, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> PeerArtifactTransferPlan:
        link_ids = tuple(link.link_id for link in self.links)
        if not link_ids:
            raise ValueError("peer artifact transfer plans require at least one link")
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("peer artifact transfer link IDs must be unique")
        if len({link.peer_node_id for link in self.links}) != 1:
            raise ValueError("all artifact links in a plan must target the same peer")

        chunk_count = len(self.manifest.chunks)
        completed = tuple(int(index) for index in self.completed_chunk_indexes)
        if tuple(sorted(set(completed))) != completed:
            raise ValueError("completed chunk indexes must be sorted and unique")
        if any(index >= chunk_count for index in completed):
            raise ValueError("a completed chunk index is outside the manifest")
        completed_set = set(completed)

        assignment_indexes = tuple(
            int(assignment.chunk_index) for assignment in self.assignments
        )
        expected_assignment_indexes = tuple(
            index for index in range(chunk_count) if index not in completed_set
        )
        if tuple(sorted(assignment_indexes)) != expected_assignment_indexes:
            raise ValueError("transfer assignments must cover every incomplete chunk once")
        if any(assignment.link_id not in set(link_ids) for assignment in self.assignments):
            raise ValueError("a transfer assignment names an unavailable link")
        return self


@final
class PeerArtifactStorage(_StrictModel):
    disk_cache_directory: Path
    memory_cache_directory: Path | None = None
    disk_reserve_bytes: NonNegativeInt = 0
    memory_reserve_bytes: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_directories(self) -> PeerArtifactStorage:
        if not self.disk_cache_directory.is_absolute():
            raise ValueError("peer artifact disk cache paths must be absolute")
        if (
            self.memory_cache_directory is not None
            and self.memory_cache_directory == self.disk_cache_directory
        ):
            raise ValueError("disk and memory artifact cache directories must differ")
        if (
            self.memory_cache_directory is not None
            and not self.memory_cache_directory.is_absolute()
        ):
            raise ValueError("peer artifact memory cache paths must be absolute")
        return self


@final
class PeerArtifactStorageAvailability(_StrictModel):
    disk_available_bytes: NonNegativeInt
    memory_available_bytes: NonNegativeInt = 0


def _ensure_secure_cache_directory(directory: Path) -> None:
    """Create an owner-controlled cache root and reject symlinked/writable roots."""
    if not directory.is_absolute():
        raise PeerArtifactStorageConfigurationError(
            "peer artifact cache roots must be absolute"
        )
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_stat = directory.lstat()
    except OSError as error:
        raise PeerArtifactStorageConfigurationError(
            f"cannot prepare peer artifact cache root {directory}"
        ) from error
    if (
        stat.S_ISLNK(directory_stat.st_mode)
        or not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.geteuid()
        or directory_stat.st_mode & 0o022
    ):
        raise PeerArtifactStorageConfigurationError(
            "peer artifact cache roots must be owner-controlled, non-symlink "
            f"directories without group/world write permission: {directory}"
        )


def _ensure_secure_cache_subdirectory(directory: Path, cache_root: Path) -> None:
    if not directory.is_relative_to(cache_root):
        raise PeerArtifactStorageConfigurationError(
            "peer artifact cache path escaped its configured root"
        )
    relative = directory.relative_to(cache_root)
    current = cache_root
    for component in relative.parts:
        current /= component
        try:
            current.mkdir(mode=0o700, exist_ok=True)
            current_stat = current.lstat()
        except OSError as error:
            raise PeerArtifactStorageConfigurationError(
                f"cannot prepare peer artifact cache directory {current}"
            ) from error
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or current_stat.st_uid != os.geteuid()
            or current_stat.st_mode & 0o022
        ):
            raise PeerArtifactStorageConfigurationError(
                f"unsafe peer artifact cache directory {current}"
            )


def observe_peer_artifact_storage_availability(
    storage: PeerArtifactStorage,
) -> PeerArtifactStorageAvailability:
    """Observe usable disk and verified RAM-backed filesystem capacity.

    A configured memory cache is accepted only when its longest matching Linux
    mount is ``tmpfs`` or ``ramfs``.  Its reported capacity is capped by both
    filesystem free space and the kernel's current ``MemAvailable`` value.
    """
    _ensure_secure_cache_directory(storage.disk_cache_directory)
    disk_available_bytes = shutil.disk_usage(storage.disk_cache_directory).free
    memory_available_bytes = 0
    if storage.memory_cache_directory is not None:
        _ensure_secure_cache_directory(storage.memory_cache_directory)
        filesystem_type = _linux_filesystem_type(storage.memory_cache_directory)
        if filesystem_type not in {"tmpfs", "ramfs"}:
            raise PeerArtifactStorageConfigurationError(
                "peer artifact memory cache must be on tmpfs or ramfs; "
                f"{storage.memory_cache_directory} is on "
                f"{filesystem_type or 'an unidentified filesystem'}"
            )
        filesystem_available = shutil.disk_usage(
            storage.memory_cache_directory
        ).free
        memory_available_bytes = min(
            filesystem_available,
            _linux_available_memory_bytes(),
        )
    return PeerArtifactStorageAvailability(
        disk_available_bytes=disk_available_bytes,
        memory_available_bytes=memory_available_bytes,
    )


def _decode_linux_mount_path(encoded_path: str) -> str:
    result = encoded_path
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        result = result.replace(encoded, decoded)
    return result


def _linux_filesystem_type(path: Path) -> str | None:
    try:
        resolved_path = path.resolve(strict=True)
        mount_lines = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return None
    candidates: list[tuple[int, str]] = []
    for line in mount_lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount_path = Path(_decode_linux_mount_path(left_fields[4]))
        try:
            resolved_mount_path = mount_path.resolve(strict=True)
        except OSError:
            continue
        if resolved_path == resolved_mount_path or resolved_path.is_relative_to(
            resolved_mount_path
        ):
            candidates.append((len(resolved_mount_path.parts), right_fields[0]))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _linux_available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if name == "MemAvailable" and separator:
                fields = value.split()
                if len(fields) == 2 and fields[1] == "kB":
                    return int(fields[0]) * 1024
    except (OSError, ValueError):
        pass
    return 0


@final
class PeerArtifactLinkTransfer(_StrictModel):
    link_id: PeerArtifactLinkId
    transferred_bytes: NonNegativeInt


@final
class PublishedPeerArtifact(_StrictModel):
    path: Path
    storage_kind: PeerArtifactStorageKind
    sha256: Sha256Digest
    size_bytes: NonNegativeInt
    cache_hit: bool
    transferred_bytes: NonNegativeInt
    link_transfers: tuple[PeerArtifactLinkTransfer, ...]
    plan: PeerArtifactTransferPlan


class PeerArtifactRangeReader(Protocol):
    async def read_artifact_range(
        self,
        *,
        link: PeerArtifactLink,
        snapshot_id: PeerArtifactSnapshotId,
        artifact_sha256: Sha256Digest,
        artifact_path: RelativeArtifactPath,
        offset_bytes: int,
        size_bytes: int,
    ) -> bytes:
        """Read exactly one range through the explicitly selected peer link."""
        ...


@final
class _ResumeJournalHeader(_StrictModel):
    schema_version: Literal[1]
    manifest_fingerprint: Sha256Digest


@final
class _VerifiedArtifactReceipt(_StrictModel):
    schema_version: Literal[1]
    sha256: Sha256Digest
    size_bytes: NonNegativeInt
    device: NonNegativeInt
    inode: NonNegativeInt
    modified_time_nanoseconds: NonNegativeInt


@final
class _CapacityReservation(_StrictModel):
    schema_version: Literal[1]
    process_id: PositiveInt
    process_start_identity: str
    created_at_unix_seconds: float
    storage_kind: PeerArtifactStorageKind
    reserved_bytes: NonNegativeInt


@dataclass(frozen=True, slots=True)
class _ArtifactCachePaths:
    root: Path
    published: Path
    verified_receipt: Path
    partial: Path
    resume_journal: Path
    lock: Path


def build_peer_artifact_manifest(
    source_path: Path,
    artifact_path: str,
    *,
    chunk_size_bytes: int = DEFAULT_ARTIFACT_CHUNK_SIZE_BYTES,
) -> PeerArtifactManifest:
    """Hash a stable regular file into a content-addressed range manifest."""
    if chunk_size_bytes <= 0:
        raise ValueError("artifact chunk size must be positive")
    normalized_artifact_path = _validate_relative_artifact_path(artifact_path)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, os.O_RDONLY | no_follow)
    chunks: list[PeerArtifactChunk] = []
    artifact_hasher = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("peer artifacts must be regular files")
        offset = 0
        while True:
            contents = os.read(descriptor, chunk_size_bytes)
            if not contents:
                break
            artifact_hasher.update(contents)
            chunks.append(
                PeerArtifactChunk(
                    offset_bytes=offset,
                    size_bytes=len(contents),
                    sha256=hashlib.sha256(contents).hexdigest(),
                )
            )
            offset += len(contents)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_identity_before != stable_identity_after or offset != after.st_size:
        raise PeerArtifactIntegrityError(
            f"artifact changed while its manifest was built: {source_path}"
        )
    return PeerArtifactManifest(
        schema_version=PEER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        canonicalization=_PEER_ARTIFACT_MANIFEST_CANONICALIZATION,
        artifact_path=normalized_artifact_path,
        size_bytes=offset,
        sha256=artifact_hasher.hexdigest(),
        chunks=tuple(chunks),
    )


def peer_artifact_manifest_fingerprint(
    manifest: PeerArtifactManifest,
) -> Sha256Digest:
    canonical_manifest = json.dumps(
        manifest.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical_manifest).hexdigest()


def plan_peer_artifact_transfer(
    manifest: PeerArtifactManifest,
    links: Sequence[PeerArtifactLink],
    *,
    completed_chunk_indexes: Collection[int] = (),
) -> PeerArtifactTransferPlan:
    """Assign incomplete chunks to every explicit link in bandwidth-weighted stripes."""
    stable_links = tuple(
        sorted(
            links,
            key=lambda link: (
                -int(link.estimated_bytes_per_second),
                str(link.link_id),
            ),
        )
    )
    if not stable_links:
        raise ValueError("at least one peer artifact link is required")
    link_ids = tuple(link.link_id for link in stable_links)
    if len(set(link_ids)) != len(link_ids):
        raise ValueError("peer artifact link IDs must be unique")
    if len({link.peer_node_id for link in stable_links}) != 1:
        raise ValueError("all peer artifact links must target the same peer")

    completed = tuple(sorted(set(completed_chunk_indexes)))
    if any(index < 0 or index >= len(manifest.chunks) for index in completed):
        raise ValueError("a completed chunk index is outside the artifact manifest")
    completed_set = set(completed)
    assigned_bytes = {link.link_id: 0 for link in stable_links}
    assignments: list[PeerArtifactChunkAssignment] = []
    for chunk_index, chunk in enumerate(manifest.chunks):
        if chunk_index in completed_set:
            continue
        selected_link = min(
            stable_links,
            key=lambda link: (
                Fraction(
                    assigned_bytes[link.link_id],
                    int(link.estimated_bytes_per_second),
                ),
                -int(link.estimated_bytes_per_second),
                str(link.link_id),
            ),
        )
        assignments.append(
            PeerArtifactChunkAssignment(
                chunk_index=chunk_index,
                link_id=selected_link.link_id,
            )
        )
        assigned_bytes[selected_link.link_id] += chunk.size_bytes
    return PeerArtifactTransferPlan(
        manifest=manifest,
        links=stable_links,
        completed_chunk_indexes=completed,
        assignments=tuple(assignments),
    )


def select_peer_artifact_storage(
    manifest: PeerArtifactManifest,
    storage: PeerArtifactStorage,
    availability: PeerArtifactStorageAvailability,
    *,
    disk_completed_chunk_indexes: Collection[int] = (),
    memory_completed_chunk_indexes: Collection[int] = (),
) -> PeerArtifactStorageKind:
    """Choose disk first, then configured memory-backed storage, using missing bytes."""

    def missing_bytes(completed_chunk_indexes: Collection[int]) -> int:
        completed = set(completed_chunk_indexes)
        if any(index < 0 or index >= len(manifest.chunks) for index in completed):
            raise ValueError("a completed chunk index is outside the artifact manifest")
        return sum(
            chunk.size_bytes
            for index, chunk in enumerate(manifest.chunks)
            if index not in completed
        )

    required_disk_bytes = missing_bytes(disk_completed_chunk_indexes)
    if (
        required_disk_bytes + storage.disk_reserve_bytes
        <= availability.disk_available_bytes
    ):
        return "disk_cache"

    if storage.memory_cache_directory is not None:
        required_memory_bytes = missing_bytes(memory_completed_chunk_indexes)
        if (
            required_memory_bytes + storage.memory_reserve_bytes
            <= availability.memory_available_bytes
        ):
            return "memory_cache"
    raise InsufficientPeerArtifactStorageError(
        "artifact requires "
        f"{required_disk_bytes} additional disk bytes or a configured memory-backed "
        "cache with sufficient available RAM"
    )


def _missing_manifest_bytes(
    manifest: PeerArtifactManifest,
    completed_chunk_indexes: Collection[int],
) -> int:
    completed = set(completed_chunk_indexes)
    return sum(
        int(chunk.size_bytes)
        for index, chunk in enumerate(manifest.chunks)
        if index not in completed
    )


def _process_start_identity(process_id: int) -> str | None:
    try:
        stat_contents = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, separator, remaining = stat_contents.rpartition(") ")
    if not separator:
        return None
    fields_after_command = remaining.split()
    if len(fields_after_command) <= 19:
        return None
    return fields_after_command[19]


def _reservation_directory(storage: PeerArtifactStorage) -> Path:
    directory = storage.disk_cache_directory / ".capacity-reservations"
    _ensure_secure_cache_subdirectory(directory, storage.disk_cache_directory)
    return directory


def _active_capacity_reservations(
    storage: PeerArtifactStorage,
) -> tuple[_CapacityReservation, ...]:
    directory = _reservation_directory(storage)
    reservations: list[_CapacityReservation] = []
    for path in directory.glob("*.json"):
        try:
            path_stat = path.lstat()
            if (
                stat.S_ISLNK(path_stat.st_mode)
                or not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_size > 4096
            ):
                path.unlink(missing_ok=True)
                continue
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                contents = os.read(descriptor, 4097)
            finally:
                os.close(descriptor)
            reservation = _CapacityReservation.model_validate_json(contents)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            continue
        if (
            _process_start_identity(int(reservation.process_id))
            != reservation.process_start_identity
        ):
            path.unlink(missing_ok=True)
            continue
        reservations.append(reservation)
    return tuple(reservations)


def _availability_after_active_reservations(
    storage: PeerArtifactStorage,
    availability: PeerArtifactStorageAvailability,
) -> PeerArtifactStorageAvailability:
    disk_reserved = 0
    memory_reserved = 0
    for reservation in _active_capacity_reservations(storage):
        if reservation.storage_kind == "disk_cache":
            disk_reserved += int(reservation.reserved_bytes)
        else:
            memory_reserved += int(reservation.reserved_bytes)
    return PeerArtifactStorageAvailability(
        disk_available_bytes=max(
            0, int(availability.disk_available_bytes) - disk_reserved
        ),
        memory_available_bytes=max(
            0, int(availability.memory_available_bytes) - memory_reserved
        ),
    )


def _create_capacity_reservation(
    storage: PeerArtifactStorage,
    storage_kind: PeerArtifactStorageKind,
    reserved_bytes: int,
) -> Path:
    process_id = os.getpid()
    process_start_identity = _process_start_identity(process_id)
    if process_start_identity is None:
        raise PeerArtifactStorageConfigurationError(
            "cannot establish a crash-safe peer artifact capacity reservation"
        )
    reservation = _CapacityReservation(
        schema_version=1,
        process_id=process_id,
        process_start_identity=process_start_identity,
        created_at_unix_seconds=time.time(),
        storage_kind=storage_kind,
        reserved_bytes=reserved_bytes,
    )
    path = _reservation_directory(storage) / f"{uuid.uuid4().hex}.json"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        contents = reservation.model_dump_json(by_alias=True).encode()
        if os.write(descriptor, contents) != len(contents):
            raise OSError("short peer artifact capacity reservation write")
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return path


async def execute_peer_artifact_transfer(
    manifest: PeerArtifactManifest,
    snapshot_id: PeerArtifactSnapshotId,
    links: Sequence[PeerArtifactLink],
    range_reader: PeerArtifactRangeReader,
    storage: PeerArtifactStorage,
    availability: PeerArtifactStorageAvailability,
) -> PublishedPeerArtifact:
    """Transfer, resume, verify, and atomically publish one peer artifact.

    Network reads are injected through ``range_reader``. Sibling reads are cancelled
    after the first failure so no background task can keep mutating the partial file
    after this function reports an error.
    """
    _ensure_secure_cache_directory(storage.disk_cache_directory)
    if storage.memory_cache_directory is not None:
        _ensure_secure_cache_directory(storage.memory_cache_directory)
    base_plan = plan_peer_artifact_transfer(manifest, links)
    disk_paths = _artifact_cache_paths(storage.disk_cache_directory, manifest.sha256)
    memory_paths = (
        _artifact_cache_paths(storage.memory_cache_directory, manifest.sha256)
        if storage.memory_cache_directory is not None
        else None
    )

    coordination_directory = storage.disk_cache_directory / ".coordination"
    _ensure_secure_cache_subdirectory(
        coordination_directory, storage.disk_cache_directory
    )
    coordination_lock = coordination_directory / f"{manifest.sha256}.lock"
    async with AsyncFileLock(coordination_lock, run_in_executor=False):
        disk_hit = await _validated_published_artifact(manifest, disk_paths)
        if disk_hit:
            return _published_result(
                disk_paths.published, "disk_cache", manifest, True, 0, base_plan
            )
        if memory_paths is not None:
            memory_hit = await _validated_published_artifact(manifest, memory_paths)
            if memory_hit:
                return _published_result(
                    memory_paths.published,
                    "memory_cache",
                    manifest,
                    True,
                    0,
                    base_plan,
                )

        disk_completed = await _verified_resume_chunk_indexes(manifest, disk_paths)
        memory_completed = (
            await _verified_resume_chunk_indexes(manifest, memory_paths)
            if memory_paths is not None
            else ()
        )
        capacity_lock = storage.disk_cache_directory / ".capacity.lock"
        reservation_path: Path | None = None
        async with AsyncFileLock(capacity_lock, run_in_executor=False):
            adjusted_availability = _availability_after_active_reservations(
                storage, availability
            )
            storage_kind = select_peer_artifact_storage(
                manifest,
                storage,
                adjusted_availability,
                disk_completed_chunk_indexes=disk_completed,
                memory_completed_chunk_indexes=memory_completed,
            )
            selected_paths = (
                disk_paths if storage_kind == "disk_cache" else memory_paths
            )
            if selected_paths is None:
                raise AssertionError(
                    "memory storage was selected without a configured path"
                )
            completed = (
                disk_completed
                if storage_kind == "disk_cache"
                else memory_completed
            )
            reservation_path = _create_capacity_reservation(
                storage,
                storage_kind,
                _missing_manifest_bytes(manifest, completed),
            )
        try:
            _ensure_secure_cache_subdirectory(
                selected_paths.partial.parent, selected_paths.root
            )
            _ensure_secure_cache_subdirectory(
                selected_paths.published.parent, selected_paths.root
            )
            if await _validated_published_artifact(manifest, selected_paths):
                return _published_result(
                    selected_paths.published,
                    storage_kind,
                    manifest,
                    True,
                    0,
                    base_plan,
                )
            completed = await _verified_resume_chunk_indexes(
                manifest, selected_paths
            )
            plan = plan_peer_artifact_transfer(
                manifest, links, completed_chunk_indexes=completed
            )
            transferred_bytes = await _execute_plan(
                plan,
                snapshot_id,
                range_reader,
                selected_paths,
                completed,
            )
            return _published_result(
                selected_paths.published,
                storage_kind,
                manifest,
                False,
                transferred_bytes,
                plan,
            )
        finally:
            if reservation_path is not None:
                async with AsyncFileLock(capacity_lock, run_in_executor=False):
                    reservation_path.unlink(missing_ok=True)


def _published_result(
    path: Path,
    storage_kind: PeerArtifactStorageKind,
    manifest: PeerArtifactManifest,
    cache_hit: bool,
    transferred_bytes: int,
    plan: PeerArtifactTransferPlan,
) -> PublishedPeerArtifact:
    link_bytes = {link.link_id: 0 for link in plan.links}
    if not cache_hit:
        for assignment in plan.assignments:
            link_bytes[assignment.link_id] += int(
                plan.manifest.chunks[assignment.chunk_index].size_bytes
            )
    return PublishedPeerArtifact(
        path=path,
        storage_kind=storage_kind,
        sha256=manifest.sha256,
        size_bytes=manifest.size_bytes,
        cache_hit=cache_hit,
        transferred_bytes=transferred_bytes,
        link_transfers=tuple(
            PeerArtifactLinkTransfer(
                link_id=link.link_id,
                transferred_bytes=link_bytes[link.link_id],
            )
            for link in plan.links
        ),
        plan=plan,
    )


def _artifact_cache_paths(
    root: Path, artifact_sha256: Sha256Digest
) -> _ArtifactCachePaths:
    published_directory = root / "sha256" / artifact_sha256[:2]
    partial_directory = root / ".partial" / "sha256"
    return _ArtifactCachePaths(
        root=root,
        published=published_directory / artifact_sha256,
        verified_receipt=published_directory / f"{artifact_sha256}.verified.json",
        partial=partial_directory / f"{artifact_sha256}.part",
        resume_journal=partial_directory / f"{artifact_sha256}.resume",
        lock=partial_directory / f"{artifact_sha256}.lock",
    )


async def _execute_plan(
    plan: PeerArtifactTransferPlan,
    snapshot_id: PeerArtifactSnapshotId,
    range_reader: PeerArtifactRangeReader,
    paths: _ArtifactCachePaths,
    completed_chunk_indexes: Collection[int],
) -> int:
    descriptor, journal_descriptor = _prepare_partial_artifact(
        plan.manifest, paths, completed_chunk_indexes
    )
    assignments_by_link: dict[
        PeerArtifactLinkId, list[PeerArtifactChunkAssignment]
    ] = {link.link_id: [] for link in plan.links}
    for assignment in plan.assignments:
        assignments_by_link[assignment.link_id].append(assignment)

    async def transfer_assignments(
        link: PeerArtifactLink,
        assignments: Sequence[PeerArtifactChunkAssignment],
    ) -> None:
        for assignment in assignments:
            chunk = plan.manifest.chunks[assignment.chunk_index]
            contents = await range_reader.read_artifact_range(
                link=link,
                snapshot_id=snapshot_id,
                artifact_sha256=plan.manifest.sha256,
                artifact_path=plan.manifest.artifact_path,
                offset_bytes=chunk.offset_bytes,
                size_bytes=chunk.size_bytes,
            )
            if len(contents) != chunk.size_bytes:
                raise PeerArtifactProtocolError(
                    f"link {link.link_id} returned {len(contents)} bytes for a "
                    f"{chunk.size_bytes}-byte artifact range"
                )
            actual_sha256 = hashlib.sha256(contents).hexdigest()
            if actual_sha256 != chunk.sha256:
                raise PeerArtifactIntegrityError(
                    f"artifact chunk {assignment.chunk_index} from link "
                    f"{link.link_id} has SHA-256 {actual_sha256}, expected {chunk.sha256}"
                )
            await _write_verified_chunk(
                descriptor,
                journal_descriptor,
                chunk.offset_bytes,
                contents,
                int(assignment.chunk_index),
            )

    tasks: list[asyncio.Task[None]] = []
    for link in plan.links:
        assignments = assignments_by_link[link.link_id]
        worker_count = min(link.maximum_concurrent_chunks, len(assignments))
        for worker_index in range(worker_count):
            worker_assignments = assignments[worker_index::worker_count]
            tasks.append(
                asyncio.create_task(transfer_assignments(link, worker_assignments))
            )

    try:
        if tasks:
            await asyncio.gather(*tasks)
        os.fsync(descriptor)
        actual_sha256 = await _sha256_file_descriptor(
            descriptor, plan.manifest.size_bytes
        )
        if actual_sha256 != plan.manifest.sha256:
            raise PeerArtifactIntegrityError(
                f"assembled artifact has SHA-256 {actual_sha256}, expected "
                f"{plan.manifest.sha256}"
            )
    except BaseException:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        os.close(journal_descriptor)
        os.close(descriptor)

    _publish_partial_artifact(plan.manifest, paths)
    return sum(
        plan.manifest.chunks[assignment.chunk_index].size_bytes
        for assignment in plan.assignments
    )


def _prepare_partial_artifact(
    manifest: PeerArtifactManifest,
    paths: _ArtifactCachePaths,
    completed_chunk_indexes: Collection[int],
) -> tuple[int, int]:
    _ensure_secure_cache_subdirectory(paths.partial.parent, paths.root)
    resume_indexes = tuple(sorted(set(completed_chunk_indexes)))
    descriptor = os.open(
        paths.partial,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        current_size = os.fstat(descriptor).st_size
        if current_size != manifest.size_bytes:
            os.ftruncate(descriptor, manifest.size_bytes)
            resume_indexes = ()
        _write_resume_journal(manifest, paths.resume_journal, resume_indexes)
        journal_descriptor = os.open(
            paths.resume_journal,
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, journal_descriptor


async def _write_verified_chunk(
    descriptor: int,
    journal_descriptor: int,
    offset_bytes: int,
    contents: bytes,
    chunk_index: int,
) -> None:
    remaining = memoryview(contents)
    write_offset = offset_bytes
    while remaining:
        next_write = remaining[:_FILE_COPY_BUFFER_BYTES]
        written = os.pwrite(descriptor, next_write, write_offset)
        if written <= 0:
            raise OSError("failed to write a verified peer artifact chunk")
        remaining = remaining[written:]
        write_offset += written
        await asyncio.sleep(0)
    journal_entry = f"{chunk_index}\n".encode()
    written_entry_bytes = os.write(journal_descriptor, journal_entry)
    if written_entry_bytes != len(journal_entry):
        raise OSError("failed to append a peer artifact resume journal entry")


async def _verified_resume_chunk_indexes(
    manifest: PeerArtifactManifest,
    paths: _ArtifactCachePaths,
) -> tuple[int, ...]:
    try:
        partial_stat = paths.partial.lstat()
        journal_stat = paths.resume_journal.lstat()
    except FileNotFoundError:
        return ()
    if (
        not stat.S_ISREG(partial_stat.st_mode)
        or stat.S_ISLNK(partial_stat.st_mode)
        or partial_stat.st_size != manifest.size_bytes
        or not stat.S_ISREG(journal_stat.st_mode)
        or stat.S_ISLNK(journal_stat.st_mode)
    ):
        return ()
    maximum_journal_bytes = 1024 + max(1, len(manifest.chunks)) * 32
    if journal_stat.st_size > maximum_journal_bytes:
        return ()
    try:
        journal_descriptor = os.open(
            paths.resume_journal,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            journal_lines = os.read(
                journal_descriptor, maximum_journal_bytes + 1
            ).splitlines()
        finally:
            os.close(journal_descriptor)
        if not journal_lines:
            return ()
        header = _ResumeJournalHeader.model_validate_json(journal_lines[0])
    except (OSError, ValueError):
        return ()
    if header.manifest_fingerprint != peer_artifact_manifest_fingerprint(manifest):
        return ()

    declared_indexes: set[int] = set()
    try:
        for line in journal_lines[1:]:
            index = int(line)
            if index < 0 or index >= len(manifest.chunks):
                return ()
            declared_indexes.add(index)
    except ValueError:
        return ()

    descriptor = os.open(
        paths.partial, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    verified_indexes: list[int] = []
    try:
        for index in sorted(declared_indexes):
            chunk = manifest.chunks[index]
            actual_sha256 = await _sha256_file_descriptor_range(
                descriptor,
                chunk.offset_bytes,
                chunk.size_bytes,
            )
            if actual_sha256 == chunk.sha256:
                verified_indexes.append(index)
    finally:
        os.close(descriptor)
    return tuple(verified_indexes)


def _write_resume_journal(
    manifest: PeerArtifactManifest,
    journal_path: Path,
    completed_chunk_indexes: Collection[int],
) -> None:
    header = _ResumeJournalHeader(
        schema_version=_RESUME_JOURNAL_SCHEMA_VERSION,
        manifest_fingerprint=peer_artifact_manifest_fingerprint(manifest),
    )
    payload = bytearray(header.model_dump_json(by_alias=True).encode())
    payload.extend(b"\n")
    for index in sorted(set(completed_chunk_indexes)):
        payload.extend(f"{index}\n".encode())
    _atomic_write(journal_path, bytes(payload))


async def _sha256_file_descriptor_range(
    descriptor: int, offset_bytes: int, size_bytes: int
) -> str:
    hasher = hashlib.sha256()
    consumed_bytes = 0
    while consumed_bytes < size_bytes:
        contents = os.pread(
            descriptor,
            min(_FILE_COPY_BUFFER_BYTES, size_bytes - consumed_bytes),
            offset_bytes + consumed_bytes,
        )
        if not contents:
            raise PeerArtifactIntegrityError(
                "partial artifact ended before its declared size"
            )
        hasher.update(contents)
        consumed_bytes += len(contents)
        await asyncio.sleep(0)
    return hasher.hexdigest()


async def _sha256_file_descriptor(descriptor: int, size_bytes: int) -> str:
    return await _sha256_file_descriptor_range(descriptor, 0, size_bytes)


async def _sha256_regular_file(path: Path, expected_size_bytes: int) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != expected_size_bytes:
            raise PeerArtifactIntegrityError(
                f"cached artifact is not a regular {expected_size_bytes}-byte file: "
                f"{path}"
            )
        return await _sha256_file_descriptor(descriptor, expected_size_bytes)
    finally:
        os.close(descriptor)


async def _validated_published_artifact(
    manifest: PeerArtifactManifest,
    paths: _ArtifactCachePaths,
) -> bool:
    try:
        published_stat = paths.published.lstat()
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(published_stat.st_mode)
        or stat.S_ISLNK(published_stat.st_mode)
        or published_stat.st_size != manifest.size_bytes
    ):
        return False

    receipt: _VerifiedArtifactReceipt | None = None
    try:
        receipt_stat = paths.verified_receipt.lstat()
        if (
            stat.S_ISREG(receipt_stat.st_mode)
            and not stat.S_ISLNK(receipt_stat.st_mode)
            and receipt_stat.st_size <= 4096
        ):
            descriptor = os.open(
                paths.verified_receipt,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                receipt_contents = os.read(descriptor, 4097)
            finally:
                os.close(descriptor)
            receipt = _VerifiedArtifactReceipt.model_validate_json(receipt_contents)
    except (OSError, ValueError):
        pass
    if receipt is not None and (
        receipt.sha256 == manifest.sha256
        and receipt.size_bytes == manifest.size_bytes
        and receipt.device == published_stat.st_dev
        and receipt.inode == published_stat.st_ino
        and receipt.modified_time_nanoseconds == published_stat.st_mtime_ns
    ):
        return True

    actual_sha256 = await _sha256_regular_file(
        paths.published, manifest.size_bytes
    )
    if actual_sha256 != manifest.sha256:
        return False
    _write_verified_artifact_receipt(manifest, paths)
    return True


def _publish_partial_artifact(
    manifest: PeerArtifactManifest,
    paths: _ArtifactCachePaths,
) -> None:
    _ensure_secure_cache_subdirectory(paths.published.parent, paths.root)
    os.replace(paths.partial, paths.published)
    os.chmod(paths.published, 0o444)
    _fsync_directory(paths.published.parent)
    _write_verified_artifact_receipt(manifest, paths)
    paths.resume_journal.unlink(missing_ok=True)


def _write_verified_artifact_receipt(
    manifest: PeerArtifactManifest,
    paths: _ArtifactCachePaths,
) -> None:
    published_stat = paths.published.lstat()
    receipt = _VerifiedArtifactReceipt(
        schema_version=_VERIFIED_ARTIFACT_RECEIPT_SCHEMA_VERSION,
        sha256=manifest.sha256,
        size_bytes=manifest.size_bytes,
        device=published_stat.st_dev,
        inode=published_stat.st_ino,
        modified_time_nanoseconds=published_stat.st_mtime_ns,
    )
    _atomic_write(
        paths.verified_receipt,
        receipt.model_dump_json(by_alias=True).encode(),
    )


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
