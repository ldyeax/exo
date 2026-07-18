from dataclasses import dataclass, field
from typing import cast

import anyio
import pytest

import exo.worker.main as worker_main
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerShutdown
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


@dataclass
class _ShutdownRunner:
    started: anyio.Event = field(default_factory=anyio.Event)
    allow_shutdown: anyio.Event = field(default_factory=anyio.Event)
    shutdown_requested: anyio.Event = field(default_factory=anyio.Event)
    allow_stop: anyio.Event = field(default_factory=anyio.Event)
    stop_confirmed: anyio.Event = field(default_factory=anyio.Event)
    completed: set[TaskId] = field(default_factory=set)

    async def start_task(self, _task: Task) -> None:
        self.started.set()

    async def wait_for_shutdown_received(self) -> None:
        await self.allow_shutdown.wait()

    def shutdown(self) -> None:
        self.shutdown_requested.set()

    async def wait_for_stopped(self) -> None:
        await self.allow_stop.wait()
        self.stop_confirmed.set()

    def shutdown_was_forwarded(self) -> bool:
        return False


class _ClosedShutdownRunner(_ShutdownRunner):
    async def start_task(self, _task: Task) -> None:
        self.started.set()
        raise anyio.BrokenResourceError


def _make_worker() -> tuple[Worker, Sender[Event], Receiver[Event]]:
    _, indexed_event_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_command_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NodeId("node-a"),
        event_receiver=indexed_event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_command_sender,
        api_port=52415,
    )
    return worker, event_sender, event_receiver


@pytest.mark.anyio
async def test_closed_runner_channel_still_waits_for_confirmed_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_id = RunnerId("runner-a")
    task = Shutdown(
        task_id=TaskId("shutdown-a"),
        instance_id=InstanceId("instance-a"),
        runner_id=runner_id,
    )
    _return_task_once(monkeypatch, task)
    worker, event_sender, event_receiver = _make_worker()
    runner = _ClosedShutdownRunner()
    worker.runners[runner_id] = cast(RunnerSupervisor, cast(object, runner))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.plan_step)
        await runner.shutdown_requested.wait()
        assert runner_id in worker.runners
        assert not any(
            isinstance(event, RunnerStatusUpdated) for event in event_receiver.collect()
        )

        runner.allow_stop.set()
        await runner.stop_confirmed.wait()
        shutdown = await event_receiver.receive()
        assert isinstance(shutdown, RunnerStatusUpdated)
        assert isinstance(shutdown.runner_status, RunnerShutdown)
        assert runner_id not in worker.runners
        task_group.cancel_scope.cancel()

    event_sender.close()


def _return_task_once(monkeypatch: pytest.MonkeyPatch, task: Task) -> None:
    task_returned = False

    def fake_plan(*_args: object, **_kwargs: object) -> Task | None:
        nonlocal task_returned
        if task_returned:
            return None
        task_returned = True
        return task

    monkeypatch.setattr(worker_main, "plan", fake_plan)


@pytest.mark.anyio
async def test_worker_retains_runner_until_physical_stop_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_id = RunnerId("runner-a")
    task = Shutdown(
        task_id=TaskId("shutdown-a"),
        instance_id=InstanceId("instance-a"),
        runner_id=runner_id,
    )
    _return_task_once(monkeypatch, task)
    worker, event_sender, _ = _make_worker()
    runner = _ShutdownRunner()
    worker.runners[runner_id] = cast(RunnerSupervisor, cast(object, runner))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.plan_step)
        await runner.started.wait()
        await anyio.sleep(0)
        assert not runner.shutdown_requested.is_set()
        assert runner_id in worker.runners

        runner.allow_shutdown.set()
        await runner.shutdown_requested.wait()
        assert not runner.stop_confirmed.is_set()
        assert runner_id in worker.runners

        runner.allow_stop.set()
        await runner.stop_confirmed.wait()
        while runner_id in worker.runners:
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    event_sender.close()


@pytest.mark.anyio
async def test_worker_removes_stale_runner_status_when_shutdown_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_id = RunnerId("runner-a")
    task = Shutdown(
        task_id=TaskId("shutdown-a"),
        instance_id=InstanceId("instance-a"),
        runner_id=runner_id,
    )
    _return_task_once(monkeypatch, task)
    monkeypatch.setattr(worker_main, "RUNNER_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    worker, event_sender, event_receiver = _make_worker()
    runner = _ShutdownRunner()
    worker.runners[runner_id] = cast(RunnerSupervisor, cast(object, runner))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.plan_step)
        await runner.shutdown_requested.wait()

        before_stop = event_receiver.collect()
        assert any(isinstance(event, TaskCreated) for event in before_stop)
        assert any(
            isinstance(event, TaskStatusUpdated)
            and event.task_id == task.task_id
            and event.task_status == TaskStatus.TimedOut
            for event in before_stop
        )
        assert not any(isinstance(event, RunnerStatusUpdated) for event in before_stop)
        assert runner_id in worker.runners

        runner.allow_stop.set()
        await runner.stop_confirmed.wait()
        shutdown = await event_receiver.receive()
        assert isinstance(shutdown, RunnerStatusUpdated)
        assert shutdown.runner_id == runner_id
        assert isinstance(shutdown.runner_status, RunnerShutdown)
        assert runner_id not in worker.runners
        task_group.cancel_scope.cancel()

    event_sender.close()


@pytest.mark.anyio
async def test_deletion_during_shutdown_does_not_ack_before_process_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_id = RunnerId("runner-a")
    instance_id = InstanceId("instance-a")
    task = Shutdown(
        task_id=TaskId("shutdown-a"),
        instance_id=instance_id,
        runner_id=runner_id,
    )
    _return_task_once(monkeypatch, task)
    worker, event_sender, event_receiver = _make_worker()
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=instance_id,
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=runner_id,
        node_id=worker.node_id,
    )
    worker.state = State(instances={instance_id: bound_instance.instance})
    runner = _ShutdownRunner()
    worker.runners[runner_id] = cast(RunnerSupervisor, cast(object, runner))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.plan_step)
        await runner.started.wait()

        await worker._apply_indexed_event(  # pyright: ignore[reportPrivateUsage]
            IndexedEvent(idx=0, event=InstanceDeleted(instance_id=instance_id))
        )

        before_stop = event_receiver.collect()
        assert any(isinstance(event, TaskCreated) for event in before_stop)
        assert not any(isinstance(event, RunnerStatusUpdated) for event in before_stop)
        assert runner_id in worker.runners

        runner.allow_shutdown.set()
        await runner.shutdown_requested.wait()
        assert runner_id in worker.runners
        assert not runner.stop_confirmed.is_set()

        runner.allow_stop.set()
        await runner.stop_confirmed.wait()
        while runner_id in worker.runners:
            await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    assert not any(
        isinstance(event, RunnerStatusUpdated) for event in event_receiver.collect()
    )
    event_sender.close()
