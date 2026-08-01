from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import stat
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Protocol, final

from anyio import to_thread
from loguru import logger

from exo.download.download_utils import (
    build_model_path,
    is_model_directory_complete,
    resolve_model_dir,
)
from exo.download.peer_artifact_http import (
    PeerArtifactAuthenticationError,
    PeerArtifactDeploymentConfig,
    PeerArtifactHttpClient,
    PeerArtifactLinkUnavailableError,
    PeerArtifactPeerClient,
    PeerArtifactPeerConfig,
    PeerArtifactSnapshotFile,
    PeerArtifactSnapshotManifest,
    PeerArtifactUnavailableError,
    links_by_preference,
)
from exo.download.peer_artifact_transfer import (
    PEER_ARTIFACT_SNAPSHOT_RECEIPT_FILENAME,
    PeerArtifactIntegrityError,
    PeerArtifactLink,
    PeerArtifactManifest,
    PeerArtifactProtocolError,
    PeerArtifactSnapshotId,
    PeerArtifactStorage,
    PublishedPeerArtifact,
    execute_peer_artifact_transfer,
    observe_peer_artifact_storage_availability,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import HuggingFaceRevision
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    RepoDownloadProgress,
    RepoFileDownloadProgress,
)
from exo.shared.types.worker.shards import ShardMetadata

_HASH_READ_SIZE_BYTES = 8 * 1024 * 1024


class PeerArtifactClientFactory(Protocol):
    def __call__(self, peer: PeerArtifactPeerConfig, /) -> PeerArtifactPeerClient: ...


async def _sha256_regular_file(path: Path, expected_size_bytes: int) -> str | None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    hasher = hashlib.sha256()
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != expected_size_bytes
        ):
            return None
        while contents := os.read(descriptor, _HASH_READ_SIZE_BYTES):
            hasher.update(contents)
            await asyncio.sleep(0)
    finally:
        os.close(descriptor)
    return hasher.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _materialize_published_artifact(
    source: Path,
    destination: Path,
    *,
    destination_root: Path,
) -> None:
    """Atomically expose a verified cache file at the worker's model path.

    A hard link is preferred.  Cross-filesystem artifacts (notably tmpfs cache
    entries) are exposed through an absolute symlink so model loaders read the
    RAM-backed file directly without duplicating it onto disk.
    """
    if not destination.is_relative_to(destination_root):
        raise ValueError("peer artifact destination escaped its snapshot root")
    destination_root_stat = destination_root.lstat()
    if (
        stat.S_ISLNK(destination_root_stat.st_mode)
        or not stat.S_ISDIR(destination_root_stat.st_mode)
        or destination_root_stat.st_uid != os.geteuid()
        or destination_root_stat.st_mode & 0o022
    ):
        raise PermissionError(
            f"unsafe peer artifact materialization root {destination_root}"
        )
    current = destination_root
    for component in destination.parent.relative_to(destination_root).parts:
        current /= component
        current.mkdir(mode=0o755, exist_ok=True)
        current_stat = current.lstat()
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or current_stat.st_uid != os.geteuid()
            or current_stat.st_mode & 0o022
        ):
            raise PermissionError(
                f"unsafe peer artifact materialization directory {current}"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.peer-", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        try:
            os.link(source, temporary_path)
        except OSError as error:
            if error.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EOPNOTSUPP,
            }:
                raise
            os.symlink(source.resolve(strict=True), temporary_path)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(
                f"peer artifact target is unexpectedly a directory: {destination}"
            )
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


async def _fetch_snapshot_from_links(
    client: PeerArtifactPeerClient,
    peer: PeerArtifactPeerConfig,
    model_id: ModelId,
    revision: HuggingFaceRevision,
) -> tuple[PeerArtifactSnapshotManifest, tuple[PeerArtifactLink, ...]]:
    active_links = list(links_by_preference(peer.links))
    last_link_error: PeerArtifactLinkUnavailableError | None = None
    for link in tuple(active_links):
        try:
            snapshot = await client.fetch_snapshot_manifest(
                link=link,
                model_id=model_id,
                revision=revision,
            )
        except PeerArtifactLinkUnavailableError as error:
            last_link_error = error
            active_links.remove(link)
            continue
        if snapshot.model_id != model_id or snapshot.revision != revision:
            raise PeerArtifactProtocolError(
                f"peer {peer.peer_node_id} returned a different model snapshot"
            )
        return snapshot, tuple(active_links)
    if last_link_error is not None:
        raise PeerArtifactUnavailableError(
            f"all links to peer {peer.peer_node_id} are unavailable"
        ) from last_link_error
    raise PeerArtifactUnavailableError(
        f"peer {peer.peer_node_id} did not expose the requested snapshot"
    )


async def _fetch_manifest_from_links(
    client: PeerArtifactPeerClient,
    links: Sequence[PeerArtifactLink],
    snapshot_id: PeerArtifactSnapshotId,
    snapshot_file: PeerArtifactSnapshotFile,
) -> tuple[PeerArtifactManifest, tuple[PeerArtifactLink, ...]]:
    active_links = list(links_by_preference(links))
    last_link_error: PeerArtifactLinkUnavailableError | None = None
    for link in tuple(active_links):
        try:
            manifest = await client.fetch_artifact_manifest(
                link=link,
                snapshot_id=snapshot_id,
                expected_manifest_sha256=snapshot_file.manifest_sha256,
                artifact_path=snapshot_file.artifact_path,
            )
        except PeerArtifactLinkUnavailableError as error:
            last_link_error = error
            active_links.remove(link)
            continue
        if (
            manifest.size_bytes != snapshot_file.size_bytes
            or manifest.sha256 != snapshot_file.sha256
        ):
            raise PeerArtifactProtocolError(
                f"peer manifest size changed for {snapshot_file.file_path}"
            )
        return manifest, tuple(active_links)
    if last_link_error is not None:
        raise PeerArtifactUnavailableError(
            "every configured link failed while fetching an artifact manifest"
        ) from last_link_error
    raise PeerArtifactUnavailableError("no configured links remain for the artifact")


async def _execute_with_link_failover(
    manifest: PeerArtifactManifest,
    snapshot_id: PeerArtifactSnapshotId,
    links: Sequence[PeerArtifactLink],
    client: PeerArtifactPeerClient,
    storage: PeerArtifactStorage,
) -> tuple[PublishedPeerArtifact, tuple[PeerArtifactLink, ...]]:
    active_links: list[PeerArtifactLink] = list(links_by_preference(links))
    while active_links:
        availability = await to_thread.run_sync(
            observe_peer_artifact_storage_availability, storage
        )
        try:
            published = await execute_peer_artifact_transfer(
                manifest,
                snapshot_id,
                active_links,
                client,
                storage,
                availability,
            )
        except PeerArtifactLinkUnavailableError as error:
            active_links = [
                link for link in active_links if link.link_id != error.link_id
            ]
            continue
        return published, tuple(active_links)
    raise PeerArtifactUnavailableError(
        "every configured link failed during peer artifact transfer"
    )


def _target_file_size(path: Path) -> int:
    try:
        if not path.is_file():
            return 0
        return path.stat().st_size
    except OSError:
        return 0


def _snapshot_progress(
    shard: ShardMetadata,
    snapshot_files: Sequence[PeerArtifactSnapshotFile],
    target_directory: Path,
    *,
    started_at: float,
    downloaded_this_session: dict[str, int] | None = None,
    force_complete: bool = False,
) -> RepoDownloadProgress:
    session_bytes = downloaded_this_session or {}
    elapsed = max(time.time() - started_at, 0.000_001)
    file_progress: dict[str, RepoFileDownloadProgress] = {}
    completed_files = 0
    downloaded_bytes = 0
    total_bytes = sum(int(file.size_bytes) for file in snapshot_files)
    for snapshot_file in snapshot_files:
        local_size = min(
            _target_file_size(target_directory / snapshot_file.file_path),
            int(snapshot_file.size_bytes),
        )
        complete = force_complete or local_size == snapshot_file.size_bytes
        if complete:
            local_size = int(snapshot_file.size_bytes)
            completed_files += 1
        transferred = session_bytes.get(snapshot_file.file_path, 0)
        downloaded_bytes += local_size
        speed = transferred / elapsed
        file_progress[snapshot_file.file_path] = RepoFileDownloadProgress(
            repo_id=shard.model_card.model_id,
            repo_revision=shard.model_card.revision,
            file_path=snapshot_file.file_path,
            downloaded=Memory.from_bytes(local_size),
            downloaded_this_session=Memory.from_bytes(transferred),
            total=Memory.from_bytes(int(snapshot_file.size_bytes)),
            speed=speed,
            eta=(
                timedelta(
                    seconds=max(int(snapshot_file.size_bytes) - local_size, 0) / speed
                )
                if speed > 0
                else timedelta(0)
            ),
            status="complete" if complete else "not_started",
            start_time=started_at,
        )
    total_session_bytes = sum(session_bytes.values())
    overall_speed = total_session_bytes / elapsed
    return RepoDownloadProgress(
        repo_id=str(shard.model_card.model_id),
        repo_revision=shard.model_card.revision,
        shard=shard,
        completed_files=completed_files,
        total_files=len(snapshot_files),
        downloaded=Memory.from_bytes(downloaded_bytes),
        downloaded_this_session=Memory.from_bytes(total_session_bytes),
        total=Memory.from_bytes(total_bytes),
        overall_speed=overall_speed,
        overall_eta=(
            timedelta(seconds=max(total_bytes - downloaded_bytes, 0) / overall_speed)
            if overall_speed > 0
            else timedelta(0)
        ),
        status="complete"
        if completed_files == len(snapshot_files)
        else ("in_progress" if total_session_bytes else "not_started"),
        file_progress=file_progress,
    )


@final
class PeerArtifactShardDownloader(ShardDownloader):
    """Peer-first worker downloader with explicit origin fallback policy."""

    def __init__(
        self,
        origin_downloader: ShardDownloader,
        config: PeerArtifactDeploymentConfig,
        *,
        origin_offline: bool,
        client_factory: PeerArtifactClientFactory | None = None,
    ) -> None:
        if not config.peers or config.disk_cache_directory is None:
            raise ValueError("peer artifact downloader requires peers and storage")
        self._origin_downloader = origin_downloader
        self._config = config
        self._origin_offline = origin_offline
        self._storage = PeerArtifactStorage(
            disk_cache_directory=config.disk_cache_directory,
            memory_cache_directory=config.memory_cache_directory,
            disk_reserve_bytes=config.disk_reserve_bytes,
            memory_reserve_bytes=config.memory_reserve_bytes,
        )
        self._callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []
        if client_factory is None:

            def http_client_factory(
                _peer: PeerArtifactPeerConfig,
            ) -> PeerArtifactPeerClient:
                return PeerArtifactHttpClient(
                    config.authentication_secret,
                    int(config.request_timeout_seconds),
                )

            self._client_factory = http_client_factory
        else:
            self._client_factory = client_factory

    @property
    def supports_offline_download(self) -> bool:
        return True

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._callbacks.append(callback)
        self._origin_downloader.on_progress(callback)

    async def _emit_progress(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        for callback in self._callbacks:
            await callback(shard, progress)

    async def _ensure_from_peer(
        self,
        peer: PeerArtifactPeerConfig,
        shard: ShardMetadata,
        config_only: bool,
    ) -> Path:
        started_at = time.time()
        model_id = shard.model_card.model_id
        revision = shard.model_card.revision
        client = self._client_factory(peer)
        async with client:
            snapshot, active_links = await _fetch_snapshot_from_links(
                client, peer, model_id, revision
            )
            selected_files = (
                tuple(
                    snapshot_file
                    for snapshot_file in snapshot.files
                    if snapshot_file.file_path == "config.json"
                )
                if config_only
                else snapshot.files
            )
            if not selected_files:
                raise PeerArtifactProtocolError(
                    f"peer {peer.peer_node_id} snapshot contains no requested files"
                )
            target_directory = await resolve_model_dir(model_id, revision)
            session_bytes: dict[str, int] = {}
            await self._emit_progress(
                shard,
                _snapshot_progress(
                    shard,
                    selected_files,
                    target_directory,
                    started_at=started_at,
                ),
            )
            for snapshot_file in selected_files:
                manifest, active_links = await _fetch_manifest_from_links(
                    client,
                    active_links,
                    snapshot.snapshot_id,
                    snapshot_file,
                )
                target_path = target_directory / snapshot_file.file_path
                local_sha256 = await _sha256_regular_file(
                    target_path, int(manifest.size_bytes)
                )
                if local_sha256 != manifest.sha256:
                    published, active_links = await _execute_with_link_failover(
                        manifest,
                        snapshot.snapshot_id,
                        active_links,
                        client,
                        self._storage,
                    )
                    _materialize_published_artifact(
                        published.path,
                        target_path,
                        destination_root=target_directory,
                    )
                    session_bytes[snapshot_file.file_path] = int(
                        published.transferred_bytes
                    )
                    logger.info(
                        "Peer artifact rail bytes for "
                        f"{snapshot_file.file_path}: "
                        + ", ".join(
                            f"{transfer.link_id}={transfer.transferred_bytes}"
                            for transfer in published.link_transfers
                        )
                    )
                await self._emit_progress(
                    shard,
                    _snapshot_progress(
                        shard,
                        selected_files,
                        target_directory,
                        started_at=started_at,
                        downloaded_this_session=session_bytes,
                    ),
                )
            if (
                not config_only
                and not snapshot.allow_partial_snapshot
                and not await to_thread.run_sync(
                    is_model_directory_complete,
                    target_directory,
                    shard.model_card,
                )
            ):
                raise PeerArtifactProtocolError(
                    f"peer {peer.peer_node_id} snapshot is not a complete Exo model"
                )
            final_progress = _snapshot_progress(
                shard,
                selected_files,
                target_directory,
                started_at=started_at,
                downloaded_this_session=session_bytes,
                force_complete=True,
            )
            await self._emit_progress(shard, final_progress)
            return target_directory

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        peer_errors: list[PeerArtifactUnavailableError] = []
        for peer in self._config.peers:
            try:
                return await self._ensure_from_peer(peer, shard, config_only)
            except PeerArtifactUnavailableError as error:
                peer_errors.append(error)
                logger.warning(
                    f"Peer artifact source {peer.peer_node_id} is unavailable for "
                    f"{shard.model_card.model_id}@{shard.model_card.revision}: {error}"
                )
            except (
                PeerArtifactAuthenticationError,
                PeerArtifactIntegrityError,
                PeerArtifactProtocolError,
            ):
                raise
        if not self._config.fallback_to_origin or self._origin_offline:
            if peer_errors:
                raise PeerArtifactUnavailableError(
                    "no configured peer could provide the requested model snapshot"
                ) from peer_errors[-1]
            raise PeerArtifactUnavailableError(
                "no configured peer could provide the requested model snapshot"
            )
        logger.info(
            f"Falling back to the origin downloader for "
            f"{shard.model_card.model_id}@{shard.model_card.revision}"
        )
        return await self._origin_downloader.ensure_shard(shard, config_only)

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, progress in self._origin_downloader.get_shard_download_status():
            yield path, progress

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        target_directory = await to_thread.run_sync(
            build_model_path,
            shard.model_card.model_id,
            shard.model_card.revision,
        )
        for peer in self._config.peers:
            client = self._client_factory(peer)
            try:
                async with client:
                    snapshot, _ = await _fetch_snapshot_from_links(
                        client,
                        peer,
                        shard.model_card.model_id,
                        shard.model_card.revision,
                    )
                return _snapshot_progress(
                    shard,
                    snapshot.files,
                    target_directory,
                    started_at=time.time(),
                )
            except PeerArtifactUnavailableError:
                continue
        return await self._origin_downloader.get_shard_download_status_for_shard(shard)


def _prepare_materialization_directory(destination: Path) -> None:
    if not destination.is_absolute():
        raise ValueError("peer snapshot materialization destinations must be absolute")
    destination.mkdir(mode=0o755, parents=True, exist_ok=True)
    destination_stat = destination.lstat()
    if (
        stat.S_ISLNK(destination_stat.st_mode)
        or not stat.S_ISDIR(destination_stat.st_mode)
        or destination_stat.st_uid != os.geteuid()
        or destination_stat.st_mode & 0o022
    ):
        raise PermissionError(
            "peer snapshot destinations must be owner-controlled, non-symlink "
            "directories without group/world write permission"
        )


def _write_snapshot_receipt(
    destination: Path, snapshot: PeerArtifactSnapshotManifest
) -> None:
    receipt_path = destination / PEER_ARTIFACT_SNAPSHOT_RECEIPT_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.",
        dir=destination,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as receipt_file:
            receipt_file.write(snapshot.model_dump_json(by_alias=True).encode())
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, receipt_path)
        _fsync_directory(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


async def materialize_peer_artifact_snapshot(
    config: PeerArtifactDeploymentConfig,
    *,
    peer_node_id: NodeId,
    model_id: ModelId,
    revision: HuggingFaceRevision,
    destination: Path,
    client_factory: PeerArtifactClientFactory | None = None,
) -> PeerArtifactSnapshotManifest:
    """Fetch one pinned peer snapshot into an explicit SGLang/KT view directory."""
    if config.disk_cache_directory is None:
        raise ValueError("peer snapshot materialization requires a disk cache")
    peer = next(
        (
            configured_peer
            for configured_peer in config.peers
            if configured_peer.peer_node_id == peer_node_id
        ),
        None,
    )
    if peer is None:
        raise ValueError(f"peer {peer_node_id} is not configured")
    _prepare_materialization_directory(destination)
    storage = PeerArtifactStorage(
        disk_cache_directory=config.disk_cache_directory,
        memory_cache_directory=config.memory_cache_directory,
        disk_reserve_bytes=config.disk_reserve_bytes,
        memory_reserve_bytes=config.memory_reserve_bytes,
    )
    client = (
        PeerArtifactHttpClient(
            config.authentication_secret,
            int(config.request_timeout_seconds),
        )
        if client_factory is None
        else client_factory(peer)
    )
    async with client:
        snapshot, active_links = await _fetch_snapshot_from_links(
            client, peer, model_id, revision
        )
        for snapshot_file in snapshot.files:
            manifest, active_links = await _fetch_manifest_from_links(
                client,
                active_links,
                snapshot.snapshot_id,
                snapshot_file,
            )
            destination_path = destination / snapshot_file.file_path
            local_sha256 = await _sha256_regular_file(
                destination_path, int(manifest.size_bytes)
            )
            if local_sha256 == manifest.sha256:
                continue
            published, active_links = await _execute_with_link_failover(
                manifest,
                snapshot.snapshot_id,
                active_links,
                client,
                storage,
            )
            _materialize_published_artifact(
                published.path,
                destination_path,
                destination_root=destination,
            )
            logger.info(
                "Peer snapshot rail bytes for "
                f"{snapshot_file.file_path}: "
                + ", ".join(
                    f"{transfer.link_id}={transfer.transferred_bytes}"
                    for transfer in published.link_transfers
                )
            )
    _write_snapshot_receipt(destination, snapshot)
    return snapshot
