import asyncio
import hashlib
import os
import stat
from pathlib import Path

import pytest

from exo.download.peer_artifact_transfer import (
    InsufficientPeerArtifactStorageError,
    PeerArtifactIntegrityError,
    PeerArtifactLink,
    PeerArtifactLinkId,
    PeerArtifactManifest,
    PeerArtifactRangeReader,
    PeerArtifactSnapshotId,
    PeerArtifactStorage,
    PeerArtifactStorageAvailability,
    RelativeArtifactPath,
    Sha256Digest,
    build_peer_artifact_manifest,
    execute_peer_artifact_transfer,
    plan_peer_artifact_transfer,
)
from exo.shared.types.common import Host, NodeId

_SNAPSHOT_ID = PeerArtifactSnapshotId("a" * 64)


class InMemoryPeerArtifactReader(PeerArtifactRangeReader):
    def __init__(
        self,
        contents: bytes,
        *,
        fail_after_reads: int | None = None,
        corrupt_offset: int | None = None,
    ) -> None:
        self.contents = contents
        self.fail_after_reads = fail_after_reads
        self.corrupt_offset = corrupt_offset
        self.calls: list[tuple[PeerArtifactLinkId, int, int]] = []

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
        del artifact_path, snapshot_id, artifact_sha256
        await asyncio.sleep(0)
        if self.fail_after_reads is not None and len(self.calls) >= self.fail_after_reads:
            raise ConnectionError("injected interrupted peer transfer")
        self.calls.append((link.link_id, offset_bytes, size_bytes))
        contents = self.contents[offset_bytes : offset_bytes + size_bytes]
        if self.corrupt_offset == offset_bytes:
            return b"x" * size_bytes
        return contents


def _manifest(tmp_path: Path, contents: bytes, chunk_size_bytes: int = 4):
    source = tmp_path / "source.bin"
    source.write_bytes(contents)
    return build_peer_artifact_manifest(
        source, "layers/model-00001.safetensors", chunk_size_bytes=chunk_size_bytes
    )


def _links() -> tuple[PeerArtifactLink, ...]:
    peer_node_id = NodeId("fwuff")
    return (
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("ib-edr"),
            peer_node_id=peer_node_id,
            medium="infiniband",
            local_interface="ibp65s0",
            local_ip_address="10.40.0.1",
            peer_endpoint=Host(ip="10.40.0.2", port=52417),
            estimated_bytes_per_second=12_500_000_000,
        ),
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("ib-qdr"),
            peer_node_id=peer_node_id,
            medium="infiniband",
            local_interface="ibp129s0",
            local_ip_address="10.41.0.1",
            peer_endpoint=Host(ip="10.41.0.2", port=52417),
            estimated_bytes_per_second=5_000_000_000,
        ),
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("ethernet-a"),
            peer_node_id=peer_node_id,
            medium="ethernet",
            local_interface="enp1s0f0",
            local_ip_address="10.42.0.1",
            peer_endpoint=Host(ip="10.42.0.2", port=52417),
            estimated_bytes_per_second=1_250_000_000,
        ),
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("ethernet-b"),
            peer_node_id=peer_node_id,
            medium="ethernet",
            local_interface="enp1s0f1",
            local_ip_address="10.43.0.1",
            peer_endpoint=Host(ip="10.43.0.2", port=52417),
            estimated_bytes_per_second=1_250_000_000,
        ),
    )


def test_planner_uses_every_explicit_link_and_prefers_faster_links(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, bytes(range(128)), chunk_size_bytes=4)
    plan = plan_peer_artifact_transfer(manifest, _links())
    assigned_link_ids = [assignment.link_id for assignment in plan.assignments]

    assert set(assigned_link_ids) == {link.link_id for link in _links()}
    assert assigned_link_ids.count(PeerArtifactLinkId("ib-edr")) > assigned_link_ids.count(
        PeerArtifactLinkId("ethernet-a")
    )


async def test_transfer_verifies_and_atomically_publishes_then_reuses_cache(
    tmp_path: Path,
) -> None:
    contents = bytes(range(128))
    manifest = _manifest(tmp_path, contents)
    reader = InMemoryPeerArtifactReader(contents)
    storage = PeerArtifactStorage(disk_cache_directory=tmp_path / "disk-cache")
    availability = PeerArtifactStorageAvailability(
        disk_available_bytes=1024 * 1024
    )

    result = await execute_peer_artifact_transfer(
        manifest, _SNAPSHOT_ID, _links(), reader, storage, availability
    )

    assert result.path.read_bytes() == contents
    assert result.sha256 == hashlib.sha256(contents).hexdigest()
    assert result.storage_kind == "disk_cache"
    assert not result.cache_hit
    assert result.transferred_bytes == len(contents)
    assert not tuple((tmp_path / "disk-cache" / ".partial").rglob("*.part"))
    assert {call[0] for call in reader.calls} == {link.link_id for link in _links()}

    cache_reader = InMemoryPeerArtifactReader(b"must not be read")
    cached = await execute_peer_artifact_transfer(
        manifest, _SNAPSHOT_ID, _links(), cache_reader, storage, availability
    )
    assert cached.path == result.path
    assert cached.cache_hit
    assert cached.transferred_bytes == 0
    assert cache_reader.calls == []


async def test_interrupted_transfer_resumes_only_verified_chunks(
    tmp_path: Path,
) -> None:
    contents = bytes(range(80))
    manifest = _manifest(tmp_path, contents, chunk_size_bytes=4)
    single_link = (_links()[0].model_copy(update={"maximum_concurrent_chunks": 1}),)
    storage = PeerArtifactStorage(disk_cache_directory=tmp_path / "disk-cache")
    availability = PeerArtifactStorageAvailability(
        disk_available_bytes=1024 * 1024
    )
    interrupted_reader = InMemoryPeerArtifactReader(contents, fail_after_reads=5)

    with pytest.raises(ConnectionError, match="injected interrupted"):
        await execute_peer_artifact_transfer(
            manifest,
            _SNAPSHOT_ID,
            single_link,
            interrupted_reader,
            storage,
            availability,
        )

    partial_path = next((tmp_path / "disk-cache" / ".partial").rglob("*.part"))
    with partial_path.open("r+b") as partial_file:
        partial_file.write(b"xxxx")

    resumed_reader = InMemoryPeerArtifactReader(contents)
    result = await execute_peer_artifact_transfer(
        manifest,
        _SNAPSHOT_ID,
        single_link,
        resumed_reader,
        storage,
        availability,
    )

    assert result.path.read_bytes() == contents
    assert result.transferred_bytes == len(contents) - 4 * 4
    resumed_offsets = {offset for _, offset, _ in resumed_reader.calls}
    interrupted_offsets = {offset for _, offset, _ in interrupted_reader.calls}
    assert resumed_offsets.intersection(interrupted_offsets) == {0}


async def test_corrupt_peer_chunk_is_not_published(tmp_path: Path) -> None:
    contents = bytes(range(64))
    manifest = _manifest(tmp_path, contents)
    reader = InMemoryPeerArtifactReader(contents, corrupt_offset=0)
    cache_root = tmp_path / "disk-cache"

    with pytest.raises(PeerArtifactIntegrityError, match="chunk"):
        await execute_peer_artifact_transfer(
            manifest,
            _SNAPSHOT_ID,
            _links(),
            reader,
            PeerArtifactStorage(disk_cache_directory=cache_root),
            PeerArtifactStorageAvailability(disk_available_bytes=1024 * 1024),
        )

    published_path = cache_root / "sha256" / manifest.sha256[:2] / manifest.sha256
    assert not published_path.exists()


async def test_failed_atomic_publish_leaves_writable_resumable_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = bytes(range(64))
    manifest = _manifest(tmp_path, contents)
    cache_root = tmp_path / "disk-cache"
    storage = PeerArtifactStorage(disk_cache_directory=cache_root)
    availability = PeerArtifactStorageAvailability(
        disk_available_bytes=1024 * 1024
    )
    original_replace = os.replace
    publish_attempts = 0

    def fail_first_publish(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        nonlocal publish_attempts
        if Path(source).suffix == ".part" and publish_attempts == 0:
            publish_attempts += 1
            raise OSError("injected atomic publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_first_publish)
    with pytest.raises(OSError, match="injected atomic publish failure"):
        await execute_peer_artifact_transfer(
            manifest,
            _SNAPSHOT_ID,
            _links(),
            InMemoryPeerArtifactReader(contents),
            storage,
            availability,
        )

    partial_path = next((cache_root / ".partial").rglob("*.part"))
    assert partial_path.stat().st_mode & stat.S_IWUSR

    resumed = await execute_peer_artifact_transfer(
        manifest,
        _SNAPSHOT_ID,
        _links(),
        InMemoryPeerArtifactReader(contents),
        storage,
        availability,
    )
    assert resumed.transferred_bytes == 0
    assert resumed.path.read_bytes() == contents
    assert stat.S_IMODE(resumed.path.stat().st_mode) == 0o444


async def test_uses_configured_memory_backed_cache_when_disk_is_full(
    tmp_path: Path,
) -> None:
    contents = bytes(range(64))
    manifest = _manifest(tmp_path, contents)
    memory_cache = tmp_path / "configured-tmpfs"
    result = await execute_peer_artifact_transfer(
        manifest,
        _SNAPSHOT_ID,
        _links(),
        InMemoryPeerArtifactReader(contents),
        PeerArtifactStorage(
            disk_cache_directory=tmp_path / "disk-cache",
            memory_cache_directory=memory_cache,
        ),
        PeerArtifactStorageAvailability(
            disk_available_bytes=0,
            memory_available_bytes=len(contents),
        ),
    )

    assert result.storage_kind == "memory_cache"
    assert result.path.is_relative_to(memory_cache)
    assert result.path.read_bytes() == contents


async def test_rejects_transfer_when_disk_and_memory_are_insufficient(
    tmp_path: Path,
) -> None:
    contents = bytes(range(64))
    manifest = _manifest(tmp_path, contents)

    with pytest.raises(InsufficientPeerArtifactStorageError):
        await execute_peer_artifact_transfer(
            manifest,
            _SNAPSHOT_ID,
            _links(),
            InMemoryPeerArtifactReader(contents),
            PeerArtifactStorage(
                disk_cache_directory=tmp_path / "disk-cache",
                memory_cache_directory=tmp_path / "memory-cache",
            ),
            PeerArtifactStorageAvailability(
                disk_available_bytes=len(contents) - 1,
                memory_available_bytes=len(contents) - 1,
            ),
        )


def test_manifest_rejects_noncontiguous_chunks(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, b"abcdefgh", chunk_size_bytes=4)
    with pytest.raises(ValueError, match="contiguous"):
        PeerArtifactManifest.model_validate(
            {
                **manifest.model_dump(),
                "chunks": (
                    manifest.chunks[0],
                    manifest.chunks[1].model_copy(update={"offset_bytes": 5}),
                ),
            }
        )
