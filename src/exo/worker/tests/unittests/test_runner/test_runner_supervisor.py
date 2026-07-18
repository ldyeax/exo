from types import TracebackType
from typing import Self, cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, Sender, channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _DeadProcess:
    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = -6

    def is_alive(self) -> bool:
        return False


class _RunnerEventReceiver:
    def __init__(self, events: list[Event | RunnerTerminationError]) -> None:
        self.events = events

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        pass

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Event | RunnerTerminationError:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


async def _make_supervisor(
    event_sender: Sender[Event],
    runner_events: _RunnerEventReceiver,
) -> RunnerSupervisor:
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    process = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=process.stdout, stderr_rx=process.stderr
    )
    return RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=cast(
            MpReceiver[Event | RunnerTerminationError], cast(object, runner_events)
        ),
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )


@pytest.mark.anyio
async def test_wait_for_shutdown_forwarded_waits_for_terminal_status() -> None:
    event_sender, event_receiver = channel[Event]()
    runner_events = _RunnerEventReceiver([])
    supervisor = await _make_supervisor(event_sender, runner_events)

    runner_events.events.append(
        RunnerStatusUpdated(
            runner_id=supervisor.bound_instance.bound_runner_id,
            runner_status=RunnerShuttingDown(),
        )
    )
    await supervisor._forward_events()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(await event_receiver.receive(), RunnerStatusUpdated)
    assert not supervisor._shutdown_forwarded.is_set()  # pyright: ignore[reportPrivateUsage]

    runner_events.events.append(
        RunnerStatusUpdated(
            runner_id=supervisor.bound_instance.bound_runner_id,
            runner_status=RunnerShutdown(),
        )
    )
    await supervisor._forward_events()  # pyright: ignore[reportPrivateUsage]
    forwarded = await event_receiver.receive()
    assert isinstance(forwarded, RunnerStatusUpdated)
    assert isinstance(forwarded.runner_status, RunnerShutdown)
    await supervisor.wait_for_shutdown_forwarded()


@pytest.mark.anyio
async def test_forwarded_task_status_is_authoritatively_attributed() -> None:
    event_sender, event_receiver = channel[Event]()
    runner_events = _RunnerEventReceiver(
        [
            TaskStatusUpdated(
                task_id=TaskId("task-a"),
                task_status=TaskStatus.Running,
                runner_id=RunnerId("spoofed-runner"),
            )
        ]
    )
    supervisor = await _make_supervisor(event_sender, runner_events)

    await supervisor._forward_events()  # pyright: ignore[reportPrivateUsage]
    forwarded = await event_receiver.receive()

    assert isinstance(forwarded, TaskStatusUpdated)
    assert forwarded.runner_id == supervisor.bound_instance.bound_runner_id


@pytest.mark.anyio
async def test_task_ack_timeout_cleans_pending_and_late_ack_is_harmless() -> None:
    event_sender, _ = channel[Event]()
    runner_events = _RunnerEventReceiver([])
    supervisor = await _make_supervisor(event_sender, runner_events)
    supervisor.task_ack_timeout = 0.01
    task = TextGeneration(
        task_id=TaskId("timeout-task"),
        instance_id=supervisor.bound_instance.instance.instance_id,
        command_id=CommandId("timeout-command"),
        task_params=TextGenerationTaskParams(
            model=supervisor.shard_metadata.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )

    with pytest.raises(TimeoutError):
        await supervisor.start_task(task)

    assert task.task_id not in supervisor.pending
    runner_events.events.append(TaskAcknowledged(task_id=task.task_id))
    await supervisor._forward_events()  # pyright: ignore[reportPrivateUsage]
    assert supervisor.pending == {}


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_task_status = await event_receiver.receive()
    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_task_status, TaskStatusUpdated)
    assert got_task_status.task_id == task.task_id
    assert got_task_status.task_status == TaskStatus.Failed
    assert got_task_status.runner_id == bound_instance.bound_runner_id

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()
