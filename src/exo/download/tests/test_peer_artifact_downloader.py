from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import ParamSpec, Self, TypeVar

import pytest
from pydantic import SecretStr

import exo.download.download_utils as download_utils
from exo.download.peer_artifact_downloader import PeerArtifactShardDownloader
from exo.download.peer_artifact_http import (
    PeerArtifactDeploymentConfig,
    PeerArtifactLinkUnavailableError,
    PeerArtifactPeerConfig,
    PeerArtifactSnapshotFile,
    PeerArtifactSnapshotManifest,
)
from exo.download.peer_artifact_transfer import (
    PeerArtifactLink,
    PeerArtifactLinkId,
    PeerArtifactManifest,
    RelativeArtifactPath,
    build_peer_artifact_manifest,
)
from exo.download.shard_downloader import NOOP_DOWNLOAD_PROGRESS, ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import RepoDownloadProgress
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

_TOKEN = SecretStr("peer-artifact-test-token-that-is-long-enough")
_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


async def _run_sync_inline(
    function: Callable[_Parameters, _Result],
    *args: _Parameters.args,
    **kwargs: _Parameters.kwargs,
) -> _Result:
    return function(*args, **kwargs)


class RecordingOriginDownloader(ShardDownloader):
    def __init__(self, target: Path) -> None:
        self.target = target
        self.ensure_calls = 0
        self.callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        del shard, config_only
        self.ensure_calls += 1
        return self.target

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.callbacks.append(callback)

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:
            yield self.target, NOOP_DOWNLOAD_PROGRESS

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return NOOP_DOWNLOAD_PROGRESS.model_copy(
            update={"shard": shard, "status": "not_started"}
        )


class InMemoryPeerClient:
    def __init__(
        self,
        snapshot: PeerArtifactSnapshotManifest,
        manifests: dict[str, PeerArtifactManifest],
        contents: dict[str, bytes],
        *,
        unavailable: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.manifests = manifests
        self.contents = contents
        self.unavailable = unavailable
        self.range_link_ids: list[PeerArtifactLinkId] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback

    async def fetch_snapshot_manifest(
        self,
        *,
        link: PeerArtifactLink,
        model_id: ModelId,
        revision: str,
    ) -> PeerArtifactSnapshotManifest:
        del model_id, revision
        if self.unavailable:
            raise PeerArtifactLinkUnavailableError(
                link.link_id, "injected unavailable peer"
            )
        return self.snapshot

    async def fetch_artifact_manifest(
        self,
        *,
        link: PeerArtifactLink,
        artifact_path: RelativeArtifactPath,
    ) -> PeerArtifactManifest:
        del link
        return self.manifests[artifact_path]

    async def read_artifact_range(
        self,
        *,
        link: PeerArtifactLink,
        artifact_path: RelativeArtifactPath,
        offset_bytes: int,
        size_bytes: int,
    ) -> bytes:
        self.range_link_ids.append(link.link_id)
        return self.contents[artifact_path][
            offset_bytes : offset_bytes + size_bytes
        ]


def _links() -> tuple[PeerArtifactLink, ...]:
    return (
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("fast"),
            peer_node_id=NodeId("source"),
            medium="infiniband",
            local_interface="ib0",
            local_ip_address="127.0.0.1",
            peer_endpoint=Host(ip="127.0.0.1", port=52415),
            estimated_bytes_per_second=10_000,
        ),
        PeerArtifactLink(
            link_id=PeerArtifactLinkId("slow"),
            peer_node_id=NodeId("source"),
            medium="ethernet",
            local_interface="eth0",
            local_ip_address="127.0.0.1",
            peer_endpoint=Host(ip="127.0.0.1", port=52415),
            estimated_bytes_per_second=1_000,
        ),
    )


def _shard() -> PipelineShardMetadata:
    card = ModelCard(
        model_id=ModelId("example/model"),
        revision="main",
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _peer_fixture(
    tmp_path: Path,
) -> tuple[
    PeerArtifactSnapshotManifest,
    dict[str, PeerArtifactManifest],
    dict[str, bytes],
]:
    contents = {
        "config.json": b'{"model_type":"test"}',
        "model.safetensors.index.json": (
            b'{"metadata":{"total_size":8},"weight_map":{"weight":'
            b'"model-00001-of-00001.safetensors"}}'
        ),
        "model-00001-of-00001.safetensors": b"weights!",
    }
    source = tmp_path / "source"
    source.mkdir()
    manifests: dict[str, PeerArtifactManifest] = {}
    files: list[PeerArtifactSnapshotFile] = []
    artifact_contents: dict[str, bytes] = {}
    for file_path, file_contents in contents.items():
        source_path = source / file_path
        source_path.write_bytes(file_contents)
        artifact_path = f"snapshot/{file_path}"
        manifests[artifact_path] = build_peer_artifact_manifest(
            source_path,
            artifact_path,
            chunk_size_bytes=4,
        )
        artifact_contents[artifact_path] = file_contents
        files.append(
            PeerArtifactSnapshotFile(
                file_path=file_path,
                artifact_path=artifact_path,
                size_bytes=len(file_contents),
            )
        )
    snapshot = PeerArtifactSnapshotManifest(
        schema_version=1,
        model_id=ModelId("example/model"),
        revision="main",
        files=tuple(sorted(files, key=lambda file: file.file_path)),
    )
    return snapshot, manifests, artifact_contents


def _config(tmp_path: Path) -> PeerArtifactDeploymentConfig:
    return PeerArtifactDeploymentConfig(
        schema_version=1,
        bearer_token=_TOKEN,
        peers=(
            PeerArtifactPeerConfig(
                peer_node_id=NodeId("source"),
                links=_links(),
            ),
        ),
        disk_cache_directory=tmp_path / "artifact-cache",
        fallback_to_origin=True,
    )


async def test_peer_downloader_materializes_verified_snapshot_for_worker_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve_model_dir_inline(
        model_id: ModelId, revision: str
    ) -> Path:
        target = download_utils.build_model_path(model_id, revision)
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(
        "exo.download.peer_artifact_downloader.to_thread.run_sync",
        _run_sync_inline,
    )
    monkeypatch.setattr(download_utils.asyncio, "to_thread", _run_sync_inline)
    monkeypatch.setattr(
        "exo.download.peer_artifact_downloader.resolve_model_dir",
        resolve_model_dir_inline,
    )
    models_root = tmp_path / "models"
    models_root.mkdir()
    monkeypatch.setattr(download_utils, "EXO_DEFAULT_MODELS_DIR", models_root)
    monkeypatch.setattr(download_utils, "EXO_MODELS_DIRS", (models_root,))
    monkeypatch.setattr(download_utils, "EXO_MODELS_READ_ONLY_DIRS", ())
    snapshot, manifests, contents = _peer_fixture(tmp_path)
    client = InMemoryPeerClient(snapshot, manifests, contents)
    origin = RecordingOriginDownloader(tmp_path / "origin")
    downloader = PeerArtifactShardDownloader(
        origin,
        _config(tmp_path),
        origin_offline=True,
        client_factory=lambda _peer: client,
    )
    progress: list[RepoDownloadProgress] = []

    async def record_progress(
        _shard: ShardMetadata, observation: RepoDownloadProgress
    ) -> None:
        progress.append(observation)

    downloader.on_progress(record_progress)
    result = await downloader.ensure_shard(_shard())

    assert origin.ensure_calls == 0
    assert downloader.supports_offline_download
    assert (result / "config.json").read_bytes() == contents["snapshot/config.json"]
    assert (
        result / "model-00001-of-00001.safetensors"
    ).read_bytes() == contents["snapshot/model-00001-of-00001.safetensors"]
    assert progress[-1].status == "complete"
    assert progress[-1].completed_files == 3
    assert set(client.range_link_ids) == {
        PeerArtifactLinkId("fast"),
        PeerArtifactLinkId("slow"),
    }
    assert download_utils.resolve_existing_model(
        ModelId("example/model"), _shard().model_card
    ) == result


async def test_unavailable_peer_falls_back_only_when_origin_is_enabled(
    tmp_path: Path,
) -> None:
    snapshot, manifests, contents = _peer_fixture(tmp_path)
    client = InMemoryPeerClient(
        snapshot, manifests, contents, unavailable=True
    )
    origin_target = tmp_path / "origin"
    origin = RecordingOriginDownloader(origin_target)
    downloader = PeerArtifactShardDownloader(
        origin,
        _config(tmp_path),
        origin_offline=False,
        client_factory=lambda _peer: client,
    )

    assert await downloader.ensure_shard(_shard()) == origin_target
    assert origin.ensure_calls == 1
