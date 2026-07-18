from dataclasses import dataclass, field
from typing import cast

import anyio
import pytest

import exo.worker.main as worker_main
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerShutdown
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker
from exo.worker.runner.supervisor import RunnerSupervisor


@dataclass
class _ShutdownRunner:
    started: anyio.Event = field(default_factory=anyio.Event)
    allow_shutdown: anyio.Event = field(default_factory=anyio.Event)
    stopped: anyio.Event = field(default_factory=anyio.Event)
    completed: set[TaskId] = field(default_factory=set)

    async def start_task(self, _task: Task) -> None:
        self.started.set()

    async def wait_for_shutdown_forwarded(self) -> None:
        await self.allow_shutdown.wait()

    def shutdown(self) -> None:
        self.stopped.set()


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
async def test_worker_waits_for_forwarded_runner_shutdown_before_stopping_supervisor(
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
        assert not runner.stopped.is_set()

        runner.allow_shutdown.set()
        await runner.stopped.wait()
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
        await runner.stopped.wait()
        task_group.cancel_scope.cancel()

    created = await event_receiver.receive()
    timed_out = await event_receiver.receive()
    shutdown = await event_receiver.receive()
    assert isinstance(created, TaskCreated)
    assert isinstance(timed_out, TaskStatusUpdated)
    assert timed_out.task_id == task.task_id
    assert timed_out.task_status == TaskStatus.TimedOut
    assert isinstance(shutdown, RunnerStatusUpdated)
    assert shutdown.runner_id == runner_id
    assert isinstance(shutdown.runner_status, RunnerShutdown)

    event_sender.close()
