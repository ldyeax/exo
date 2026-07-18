"""Regression tests for #1918: download status must not revert from completed to pending.

The periodic rescan in _emit_existing_download_progress compares local file
sizes against HuggingFace API sizes.  Text files (README, YAML, jinja) can
have different local vs remote sizes due to encoding changes.  This must NOT
cause a completed download to be downgraded.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import AsyncMock, patch

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
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model")
MODEL_DIR = Path("/fake/models/test-org--test-model")
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


SHARD = _make_shard()
SNAPSHOT_ID = model_snapshot_id(SHARD.model_card)


class FakeShardDownloader(ShardDownloader):
    """Fake downloader that yields a single model with configurable status."""

    def __init__(
        self, status: Literal["not_started", "in_progress", "complete"] = "not_started"
    ) -> None:
        self._status: Literal["not_started", "in_progress", "complete"] = status
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

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
        return MODEL_DIR  # pragma: no cover

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        yield (
            MODEL_DIR,
            RepoDownloadProgress(
                repo_id=str(MODEL_ID),
                repo_revision="main",
                shard=SHARD,
                completed_files=10,
                total_files=13,
                downloaded=Memory.from_mb(95),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_mb(100),
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                status=self._status,
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
            total_files=0,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_bytes(0),
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


async def _collect_events(
    event_recv: Receiver[Event], timeout: float = 1.0
) -> list[Event]:
    """Drain events until timeout."""
    events: list[Event] = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                events.append(await event_recv.receive())
    except TimeoutError:
        pass
    return events


async def test_completed_status_not_downgraded_by_rescan() -> None:
    """A model already marked DownloadCompleted must not revert to
    DownloadPending when the periodic rescan reports a non-complete
    file-size status (regression test for #1918)."""
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    # Pre-seed the coordinator with a completed status for the model
    completed = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=SHARD,
        total=Memory.from_mb(100),
        model_directory=str(MODEL_DIR),
    )
    coordinator.download_status[SNAPSHOT_ID] = completed

    # Run the coordinator (the rescan loop fires immediately)
    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Wait for the rescan to process (it should skip the completed model)
        events = await _collect_events(event_recv, timeout=1.5)

        # The model must still be DownloadCompleted — not downgraded
        assert isinstance(
            coordinator.download_status[SNAPSHOT_ID], DownloadCompleted
        ), (
            f"Expected DownloadCompleted but got {type(coordinator.download_status[SNAPSHOT_ID]).__name__}"
        )

        # No DownloadPending event should have been emitted for this model
        pending_events = [
            e
            for e in events
            if isinstance(e, NodeDownloadProgress)
            and isinstance(e.download_progress, DownloadPending)
            and e.download_progress.shard_metadata.model_card.model_id == MODEL_ID
        ]
        assert len(pending_events) == 0, (
            f"Expected no DownloadPending events for completed model, got {len(pending_events)}"
        )
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_incomplete_model_with_files_present_detected_as_complete() -> None:
    """When the per-file size check says not_started but resolve_existing_model
    confirms the model directory is complete, the model should be marked
    DownloadCompleted (regression test for #1918 — initial scan case)."""
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    # Mock resolve_existing_model to return a valid path (model is on disk)
    with patch(
        "exo.download.coordinator.resolve_existing_model",
        return_value=MODEL_DIR,
    ):
        coordinator_task = asyncio.create_task(coordinator.run())
        try:
            events = await _collect_events(event_recv, timeout=1.5)

            # The model should be DownloadCompleted (resolve_existing_model confirmed it)
            assert isinstance(
                coordinator.download_status.get(SNAPSHOT_ID), DownloadCompleted
            ), (
                f"Expected DownloadCompleted but got "
                f"{type(coordinator.download_status.get(SNAPSHOT_ID)).__name__}"
            )

            # Should have emitted a DownloadCompleted event
            completed_events = [
                e
                for e in events
                if isinstance(e, NodeDownloadProgress)
                and isinstance(e.download_progress, DownloadCompleted)
                and e.download_progress.shard_metadata.model_card.model_id == MODEL_ID
            ]
            assert len(completed_events) > 0, (
                "Expected at least one DownloadCompleted event"
            )
        finally:
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def test_genuinely_incomplete_model_stays_pending() -> None:
    """When the per-file size check says not_started and resolve_existing_model
    returns None (model truly not complete), the model should correctly be
    DownloadPending."""
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, event_recv = _setup_coordinator(downloader)

    # Mock resolve_existing_model to return None (model not on disk)
    with patch(
        "exo.download.coordinator.resolve_existing_model",
        return_value=None,
    ):
        coordinator_task = asyncio.create_task(coordinator.run())
        try:
            events = await _collect_events(event_recv, timeout=1.5)

            # The model should be DownloadPending
            assert isinstance(
                coordinator.download_status.get(SNAPSHOT_ID), DownloadPending
            ), (
                f"Expected DownloadPending but got "
                f"{type(coordinator.download_status.get(SNAPSHOT_ID)).__name__}"
            )

            # Should have emitted a DownloadPending event
            pending_events = [
                e
                for e in events
                if isinstance(e, NodeDownloadProgress)
                and isinstance(e.download_progress, DownloadPending)
                and e.download_progress.shard_metadata.model_card.model_id == MODEL_ID
            ]
            assert len(pending_events) > 0, (
                "Expected at least one DownloadPending event"
            )
        finally:
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def test_wrong_revision_status_does_not_suppress_pinned_download() -> None:
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, _event_recv = _setup_coordinator(downloader)
    coordinator.offline = True
    main_completed = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=SHARD,
        total=Memory.from_mb(100),
        model_directory=str(MODEL_DIR),
    )
    coordinator.download_status[SNAPSHOT_ID] = main_completed
    pinned_shard = SHARD.model_copy(
        update={
            "model_card": SHARD.model_card.model_copy(update={"revision": REVISION})
        }
    )
    pinned_snapshot_id = model_snapshot_id(pinned_shard.model_card)

    with patch(
        "exo.download.coordinator.to_thread.run_sync",
        new=AsyncMock(return_value=None),
    ):
        await coordinator._start_download(pinned_shard)  # pyright: ignore[reportPrivateUsage]

    assert coordinator.download_status[SNAPSHOT_ID] is main_completed
    pinned_status = coordinator.download_status[pinned_snapshot_id]
    assert isinstance(pinned_status, DownloadFailed)
    assert pinned_status.model_directory.endswith(f"--{REVISION}")


async def test_progress_callback_tracks_revisions_independently() -> None:
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, _event_recv = _setup_coordinator(downloader)
    pinned_shard = SHARD.model_copy(
        update={
            "model_card": SHARD.model_card.model_copy(update={"revision": REVISION})
        }
    )
    pinned_snapshot_id = model_snapshot_id(pinned_shard.model_card)
    main_progress = RepoDownloadProgress(
        repo_id=str(MODEL_ID),
        repo_revision="main",
        shard=SHARD,
        completed_files=0,
        total_files=1,
        downloaded=Memory.from_mb(50),
        downloaded_this_session=Memory.from_mb(50),
        total=Memory.from_mb(100),
        overall_speed=1024,
        overall_eta=timedelta(seconds=1),
        status="in_progress",
    )
    pinned_progress = main_progress.model_copy(
        update={"repo_revision": REVISION, "shard": pinned_shard}
    )

    await coordinator._download_progress_callback(  # pyright: ignore[reportPrivateUsage]
        SHARD, main_progress
    )
    await coordinator._download_progress_callback(  # pyright: ignore[reportPrivateUsage]
        pinned_shard, pinned_progress
    )

    assert set(coordinator.download_status) == {SNAPSHOT_ID, pinned_snapshot_id}
    assert set(
        coordinator._last_progress_time  # pyright: ignore[reportPrivateUsage]
    ) == {SNAPSHOT_ID, pinned_snapshot_id}
    main_status = coordinator.download_status[SNAPSHOT_ID]
    pinned_status = coordinator.download_status[pinned_snapshot_id]
    assert isinstance(main_status, DownloadOngoing)
    assert isinstance(pinned_status, DownloadOngoing)
    assert main_status.model_directory.endswith(MODEL_ID.normalize())
    assert pinned_status.model_directory.endswith(f"--{REVISION}")


async def test_model_wide_delete_refuses_if_any_revision_is_read_only() -> None:
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, _event_recv = _setup_coordinator(downloader)
    pinned_shard = SHARD.model_copy(
        update={
            "model_card": SHARD.model_card.model_copy(update={"revision": REVISION})
        }
    )
    pinned_snapshot_id = model_snapshot_id(pinned_shard.model_card)
    coordinator.download_status[SNAPSHOT_ID] = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=SHARD,
        total=Memory.from_mb(100),
        model_directory=str(MODEL_DIR),
        read_only=True,
    )
    coordinator.download_status[pinned_snapshot_id] = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=pinned_shard,
        total=Memory.from_mb(100),
    )

    with patch(
        "exo.download.coordinator.delete_model", new_callable=AsyncMock
    ) as delete:
        await coordinator._delete_download(MODEL_ID)  # pyright: ignore[reportPrivateUsage]

    delete.assert_not_awaited()
    assert set(coordinator.download_status) == {SNAPSHOT_ID, pinned_snapshot_id}


async def test_model_wide_delete_clears_every_tracked_revision() -> None:
    downloader = FakeShardDownloader(status="not_started")
    coordinator, _cmd_send, _event_recv = _setup_coordinator(downloader)
    pinned_shard = SHARD.model_copy(
        update={
            "model_card": SHARD.model_card.model_copy(update={"revision": REVISION})
        }
    )
    pinned_snapshot_id = model_snapshot_id(pinned_shard.model_card)
    coordinator.download_status[SNAPSHOT_ID] = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=SHARD,
        total=Memory.from_mb(100),
    )
    coordinator.download_status[pinned_snapshot_id] = DownloadCompleted(
        node_id=NODE_ID,
        shard_metadata=pinned_shard,
        total=Memory.from_mb(100),
    )

    with patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    ) as delete:
        await coordinator._delete_download(MODEL_ID)  # pyright: ignore[reportPrivateUsage]

    delete.assert_awaited_once_with(MODEL_ID)
    assert not coordinator.download_status
