"""Tests for cancelling (pausing) an active download via CancelDownload command."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock

import anyio
from pytest import MonkeyPatch

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.impl_shard_downloader import SingletonShardDownloader
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    model_snapshot_id,
)
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadPending
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model")
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _make_shard(model_id: ModelId = MODEL_ID) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class SlowShardDownloader(ShardDownloader):
    """Fake downloader that blocks during ensure_shard until cancelled,
    simulating a long-running download."""

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []
        self.download_started = asyncio.Event()

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        # Fire an in-progress callback, then block forever (until cancelled)
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            total=Memory.from_mb(100),
            overall_speed=1024 * 1024,
            overall_eta=timedelta(seconds=50),
            status="in_progress",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        self.download_started.set()
        # Block until cancelled
        await asyncio.Event().wait()
        return (
            Path("/fake/models") / shard.model_card.model_id.normalize()
        )  # pragma: no cover

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


def _setup_coordinator(
    downloader: ShardDownloader,
) -> tuple[
    DownloadCoordinator,
    Sender[ForwarderDownloadCommand],
    Receiver[Event],
]:
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    wrapped = SingletonShardDownloader(downloader)
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=wrapped,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
    )
    return coordinator, cmd_send, event_recv


async def _wait_for_pending(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> DownloadPending | None:
    """Drain events until we see a DownloadPending for the given model, or timeout."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadPending)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event.download_progress
    except TimeoutError:
        return None


async def test_cancel_active_download_transitions_to_pending(
    monkeypatch: MonkeyPatch,
) -> None:
    """Cancelling an in-progress download should emit a DownloadPending event
    and remove the model from active_downloads."""
    slow_downloader = SlowShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(slow_downloader)
    shard = _make_shard()
    snapshot_id = model_snapshot_id(shard.model_card)
    origin = SystemId("test")
    monkeypatch.setattr(
        "exo.download.coordinator.to_thread.run_sync", AsyncMock(return_value=None)
    )

    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Start a download
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )

        # Wait for the download to actually start (blocking in ensure_shard)
        await asyncio.wait_for(slow_downloader.download_started.wait(), timeout=2.0)

        # Drain any events emitted before the cancel (initial DownloadPending, DownloadOngoing)
        while True:
            try:
                async with asyncio.timeout(0.1):
                    await event_recv.receive()
            except TimeoutError:
                break

        # Cancel the download
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )

        # Should receive a DownloadPending event with preserved progress
        pending = await _wait_for_pending(event_recv, MODEL_ID)
        assert pending is not None, "Cancel should emit DownloadPending"
        assert pending.shard_metadata.model_card.model_id == MODEL_ID
        assert pending.total == Memory.from_mb(100), "Should preserve total bytes"

        # Give coordinator time to clean up
        await asyncio.sleep(0.05)

        # Model should no longer be in active_downloads
        assert snapshot_id not in coordinator.active_downloads
        # But should still be in download_status as pending
        assert snapshot_id in coordinator.download_status
        assert isinstance(coordinator.download_status[snapshot_id], DownloadPending)
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_cancel_nonexistent_download_is_noop() -> None:
    """Cancelling a model that isn't being downloaded should be a no-op."""
    slow_downloader = SlowShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(slow_downloader)
    origin = SystemId("test")

    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Cancel a model that was never started
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )

        # Should NOT receive any DownloadPending event
        pending = await _wait_for_pending(event_recv, MODEL_ID, timeout=0.5)
        assert pending is None, "Cancel of non-existent download should not emit events"

        # Coordinator state should be empty
        assert not coordinator.active_downloads
        assert not coordinator.download_status
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_model_wide_cancel_cancels_every_tracked_revision() -> None:
    slow_downloader = SlowShardDownloader()
    coordinator, _cmd_send, _event_recv = _setup_coordinator(slow_downloader)
    main_shard = _make_shard()
    pinned_shard = main_shard.model_copy(
        update={
            "model_card": main_shard.model_card.model_copy(
                update={"revision": REVISION}
            )
        }
    )
    main_snapshot_id = model_snapshot_id(main_shard.model_card)
    pinned_snapshot_id = model_snapshot_id(pinned_shard.model_card)
    main_scope = anyio.CancelScope()
    pinned_scope = anyio.CancelScope()
    coordinator.active_downloads = {
        main_snapshot_id: main_scope,
        pinned_snapshot_id: pinned_scope,
    }
    coordinator.download_status = {
        main_snapshot_id: DownloadPending(
            node_id=NODE_ID,
            shard_metadata=main_shard,
        ),
        pinned_snapshot_id: DownloadPending(
            node_id=NODE_ID,
            shard_metadata=pinned_shard,
        ),
    }

    await coordinator._cancel_download(MODEL_ID)  # pyright: ignore[reportPrivateUsage]

    assert main_scope.cancel_called
    assert pinned_scope.cancel_called
    assert coordinator.download_status[main_snapshot_id].model_directory.endswith(
        MODEL_ID.normalize()
    )
    assert coordinator.download_status[pinned_snapshot_id].model_directory.endswith(
        f"--{REVISION}"
    )


async def test_cancel_then_resume_download(monkeypatch: MonkeyPatch) -> None:
    """After cancelling, re-issuing StartDownload should restart the download."""
    slow_downloader = SlowShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(slow_downloader)
    shard = _make_shard()
    snapshot_id = model_snapshot_id(shard.model_card)
    origin = SystemId("test")
    monkeypatch.setattr(
        "exo.download.coordinator.to_thread.run_sync", AsyncMock(return_value=None)
    )

    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Start download
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        await asyncio.wait_for(slow_downloader.download_started.wait(), timeout=2.0)

        # Cancel
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )
        pending = await _wait_for_pending(event_recv, MODEL_ID)
        assert pending is not None, "Cancel should emit DownloadPending"

        await asyncio.sleep(0.05)

        # Reset the event so we can detect the next download start
        slow_downloader.download_started.clear()

        # Resume by sending StartDownload again
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )

        # The download should restart
        await asyncio.wait_for(slow_downloader.download_started.wait(), timeout=2.0)
        assert snapshot_id in coordinator.active_downloads, (
            "Model should be actively downloading again after resume"
        )
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task
