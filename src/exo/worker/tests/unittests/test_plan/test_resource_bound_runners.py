from collections.abc import Mapping, Sequence
from typing import cast

import anyio
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.compute_resources import (
    ComputeResource,
    NvidiaGpuComputeResource,
)
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ConnectToGroup,
    CreateRunner,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance, InstanceId, MlxNcclInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerReady,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.utils.channels import Receiver, channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import (
    Worker,
    get_local_runner_ids_for_task,
    reset_runner_backoff_for_instance,
    start_local_runner_task,
)
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor
from exo.worker.tests.unittests.conftest import FakeRunnerSupervisor

DWAGON = NodeId("dwagon")
FWUFF = NodeId("fwuff")


class _RecordingRunner:
    def __init__(self, bound_instance: BoundInstance) -> None:
        self.bound_instance = bound_instance
        self.status = RunnerReady()
        self.completed: set[TaskId] = set()
        self.in_progress: set[TaskId] = set()
        self.started: list[Task] = []

    async def start_task(self, task: Task) -> None:
        self.started.append(task)


class _BlockingRunner(_RecordingRunner):
    async def start_task(self, task: Task) -> None:
        self.started.append(task)
        await anyio.sleep_forever()


class _RaisingRunner(_RecordingRunner):
    async def start_task(self, task: Task) -> None:
        self.started.append(task)
        raise RuntimeError("rank start failed")


class _TimeoutRunner(_RecordingRunner):
    async def start_task(self, task: Task) -> None:
        self.started.append(task)
        raise TimeoutError


class _EventApplierTestWorker(Worker):
    def __init__(
        self,
        event_receiver: Receiver[IndexedEvent],
        state: State,
    ) -> None:
        self.node_id = DWAGON
        self.event_receiver = event_receiver
        self.event_sender, self.output_event_receiver = channel[Event]()
        self.state = state
        self.runners = {}
        self._instance_backoff = KeyedBackoff()
        self._runner_backoff = KeyedBackoff()
        self._runner_lifecycle_lock = anyio.Lock()
        self.creation_started = anyio.Event()
        self.allow_creation = anyio.Event()
        self.allow_creation.set()
        self.pause_creation = False
        self.create_supervisor_calls = 0

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        self.create_supervisor_calls += 1
        self.creation_started.set()
        if self.pause_creation:
            await self.allow_creation.wait()
        runner = FakeRunnerSupervisor(
            bound_instance=task.bound_instance,
            status=RunnerIdle(),
        )
        supervisor = cast(RunnerSupervisor, cast(object, runner))
        self.runners[task.bound_instance.bound_runner_id] = supervisor
        return supervisor

    def record_backoff_attempts(
        self,
        instance_id: InstanceId,
        runner_ids: Sequence[RunnerId],
    ) -> None:
        self._instance_backoff.record_attempt(instance_id)
        for runner_id in runner_ids:
            self._runner_backoff.record_attempt(runner_id)

    def instance_backoff_attempts(self, instance_id: InstanceId) -> int:
        return self._instance_backoff.attempts(instance_id)

    def runner_backoff_attempts(self, runner_id: RunnerId) -> int:
        return self._runner_backoff.attempts(runner_id)

    async def apply_events(self) -> None:
        await self._event_applier()


def _gpu_resource(index: int) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=f"GPU-00000000-0000-0000-0000-{index:012d}",
        pci_bus_id=f"00000000:{index + 32:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
    )


def _resource_bound_instance() -> tuple[
    MlxNcclInstance,
    tuple[RunnerId, RunnerId, RunnerId],
    Mapping[NodeId, Sequence[ComputeResource]],
]:
    resources = {
        DWAGON: [_gpu_resource(1), _gpu_resource(2)],
        FWUFF: [_gpu_resource(3)],
    }
    runner_ids = (
        RunnerId("dwagon-rank-0"),
        RunnerId("dwagon-rank-1"),
        RunnerId("fwuff-rank-2"),
    )
    model_card = ModelCard(
        model_id=ModelId("resource-runner-model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    shards = {
        runner_id: TensorShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=3,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        for rank, runner_id in enumerate(runner_ids)
    }
    resource_ids = [
        resource.resource_id
        for node_resources in resources.values()
        for resource in node_resources
    ]
    instance = MlxNcclInstance(
        instance_id=InstanceId("resource-bound-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner={DWAGON: runner_ids[0], FWUFF: runner_ids[2]},
            compute_resource_to_runner=dict(zip(resource_ids, runner_ids, strict=True)),
            compute_resource_to_node=dict(
                zip(resource_ids, (DWAGON, DWAGON, FWUFF), strict=True)
            ),
        ),
        nccl_coordinator=Host(ip="192.0.2.1", port=5000),
    )
    return instance, runner_ids, resources


def _plan_create_runner(
    instance: MlxNcclInstance,
    runners: Mapping[RunnerId, FakeRunnerSupervisor],
    resources: Mapping[NodeId, Sequence[ComputeResource]],
    runner_backoff: KeyedBackoff[RunnerId] | None = None,
) -> Task | None:
    return plan(
        node_id=DWAGON,
        runners=cast(Mapping[RunnerId, RunnerSupervisor], cast(object, runners)),
        global_download_status={},
        instances={instance.instance_id: instance},
        all_runners={},
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        node_compute_resources=resources,
        runner_backoff=runner_backoff,
    )


def test_plan_creates_each_runner_bound_to_local_gpu() -> None:
    instance, runner_ids, resources = _resource_bound_instance()

    first_task = _plan_create_runner(instance, {}, resources)

    assert isinstance(first_task, CreateRunner)
    assert first_task.bound_instance.bound_runner_id == runner_ids[0]
    first_runner = FakeRunnerSupervisor(
        bound_instance=first_task.bound_instance,
        status=RunnerReady(),
    )
    second_task = _plan_create_runner(
        instance,
        {runner_ids[0]: first_runner},
        resources,
    )

    assert isinstance(second_task, CreateRunner)
    assert second_task.bound_instance.bound_runner_id == runner_ids[1]
    assert second_task.bound_instance.bound_compute_resource_ids == (
        resources[DWAGON][1].resource_id,
    )


def test_bound_instance_rejects_resource_owned_by_another_node() -> None:
    instance, runner_ids, _ = _resource_bound_instance()

    with pytest.raises(ValueError, match="owned by dwagon, not bound node fwuff"):
        BoundInstance(
            instance=instance,
            bound_runner_id=runner_ids[0],
            bound_node_id=FWUFF,
        )


def test_plan_does_not_create_runner_for_resource_not_advertised_locally() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    local_resources = {DWAGON: [resources[DWAGON][0]]}
    bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[0],
        bound_node_id=DWAGON,
    )
    runners: dict[RunnerId, FakeRunnerSupervisor] = {
        runner_ids[0]: FakeRunnerSupervisor(
            bound_instance=bound_instance,
            status=RunnerReady(),
        )
    }

    task = _plan_create_runner(instance, runners, local_resources)

    assert task is None


def test_resource_runner_creation_backoff_is_per_runner() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    runner_backoff: KeyedBackoff[RunnerId] = KeyedBackoff(base=60.0, cap=60.0)
    first_task = _plan_create_runner(
        instance, {}, resources, runner_backoff=runner_backoff
    )
    assert isinstance(first_task, CreateRunner)
    runner_backoff.record_attempt(first_task.bound_instance.bound_runner_id)
    first_runner = FakeRunnerSupervisor(
        bound_instance=first_task.bound_instance,
        status=RunnerReady(),
    )

    second_task = _plan_create_runner(
        instance,
        {runner_ids[0]: first_runner},
        resources,
        runner_backoff=runner_backoff,
    )

    assert isinstance(second_task, CreateRunner)
    assert second_task.bound_instance.bound_runner_id == runner_ids[1]


def test_reset_runner_backoff_uses_all_instance_assignments() -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    runner_backoff: KeyedBackoff[RunnerId] = KeyedBackoff()
    for runner_id in runner_ids:
        runner_backoff.record_attempt(runner_id)

    reset_runner_backoff_for_instance(runner_backoff, instance)

    assert all(runner_backoff.attempts(runner_id) == 0 for runner_id in runner_ids)


@pytest.mark.anyio
async def test_instance_deletion_resets_backoff_after_supervisors_are_gone() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    worker = _EventApplierTestWorker(
        indexed_event_receiver,
        State(
            instances={instance.instance_id: instance},
            node_compute_resources=resources,
        ),
    )
    worker.record_backoff_attempts(instance.instance_id, runner_ids)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.apply_events)
        await indexed_event_sender.send(
            IndexedEvent(
                idx=0,
                event=InstanceDeleted(instance_id=instance.instance_id),
            )
        )
        indexed_event_sender.close()

    assert instance.instance_id not in worker.state.instances
    assert worker.instance_backoff_attempts(instance.instance_id) == 0
    assert all(
        worker.runner_backoff_attempts(runner_id) == 0 for runner_id in runner_ids
    )
    shutdown_events = worker.output_event_receiver.collect()
    assert {
        event.runner_id
        for event in shutdown_events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerShutdown)
    } == set(runner_ids[:2])


@pytest.mark.anyio
async def test_stale_create_is_suppressed_after_deletion_ack() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    _indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    worker = _EventApplierTestWorker(
        indexed_event_receiver,
        State(
            instances={instance.instance_id: instance},
            node_compute_resources=resources,
        ),
    )
    stale_task = _plan_create_runner(instance, {}, resources)
    assert isinstance(stale_task, CreateRunner)

    await worker._apply_indexed_event(  # pyright: ignore[reportPrivateUsage]
        IndexedEvent(
            idx=0,
            event=InstanceDeleted(instance_id=instance.instance_id),
        )
    )
    runner_started = await worker._start_planned_runner(  # pyright: ignore[reportPrivateUsage]
        stale_task
    )

    assert not runner_started
    assert worker.create_supervisor_calls == 0
    assert worker.runners == {}
    emitted_events = worker.output_event_receiver.collect()
    assert all(isinstance(event, RunnerStatusUpdated) for event in emitted_events)
    assert {
        event.runner_id
        for event in emitted_events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerShutdown)
    } == set(runner_ids[:2])


@pytest.mark.anyio
async def test_creation_finishes_before_deletion_uses_normal_shutdown_path() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    _indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    worker = _EventApplierTestWorker(
        indexed_event_receiver,
        State(
            instances={instance.instance_id: instance},
            node_compute_resources=resources,
        ),
    )
    worker.pause_creation = True
    worker.allow_creation = anyio.Event()
    create_task = _plan_create_runner(instance, {}, resources)
    assert isinstance(create_task, CreateRunner)
    create_results: list[bool] = []
    deletion_complete = anyio.Event()

    async def start_runner() -> None:
        create_results.append(
            await worker._start_planned_runner(  # pyright: ignore[reportPrivateUsage]
                create_task
            )
        )

    async def delete_instance() -> None:
        await worker._apply_indexed_event(  # pyright: ignore[reportPrivateUsage]
            IndexedEvent(
                idx=0,
                event=InstanceDeleted(instance_id=instance.instance_id),
            )
        )
        deletion_complete.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(start_runner)
        await worker.creation_started.wait()
        task_group.start_soon(delete_instance)
        await anyio.sleep(0)
        assert not deletion_complete.is_set()
        assert instance.instance_id in worker.state.instances
        worker.allow_creation.set()

    assert create_results == [True]
    assert instance.instance_id not in worker.state.instances
    assert runner_ids[0] in worker.runners
    emitted_events = worker.output_event_receiver.collect()
    assert any(isinstance(event, TaskCreated) for event in emitted_events)
    assert {
        event.runner_id
        for event in emitted_events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerShutdown)
    } == {runner_ids[1]}

    shutdown_task = plan(
        node_id=DWAGON,
        runners=worker.runners,
        global_download_status={},
        instances=worker.state.instances,
        all_runners=worker.state.runners,
        tasks=worker.state.tasks,
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        node_compute_resources=resources,
        runner_backoff=KeyedBackoff(),
    )
    assert isinstance(shutdown_task, Shutdown)
    assert shutdown_task.runner_id == runner_ids[0]


@pytest.mark.anyio
async def test_failed_task_created_send_does_not_register_supervisor() -> None:
    instance, _, resources = _resource_bound_instance()
    _indexed_event_sender, indexed_event_receiver = channel[IndexedEvent]()
    worker = _EventApplierTestWorker(
        indexed_event_receiver,
        State(
            instances={instance.instance_id: instance},
            node_compute_resources=resources,
        ),
    )
    create_task = _plan_create_runner(instance, {}, resources)
    assert isinstance(create_task, CreateRunner)
    worker.event_sender.close()

    with pytest.raises(anyio.ClosedResourceError):
        await worker._start_planned_runner(  # pyright: ignore[reportPrivateUsage]
            create_task
        )

    assert worker.create_supervisor_calls == 0
    assert worker.runners == {}


def test_resource_lifecycle_tasks_target_selected_local_rank() -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    idle_runners = {
        runner_id: FakeRunnerSupervisor(
            bound_instance=BoundInstance(
                instance=instance,
                bound_runner_id=runner_id,
                bound_node_id=DWAGON,
            ),
            status=RunnerIdle(),
        )
        for runner_id in runner_ids[:2]
    }
    local_download = DownloadCompleted(
        node_id=DWAGON,
        shard_metadata=instance.shard_assignments.runner_to_shard[runner_ids[0]],
        total=Memory.from_mb(1),
    )
    connect_task = plan(
        node_id=DWAGON,
        runners=cast(Mapping[RunnerId, RunnerSupervisor], cast(object, idle_runners)),
        global_download_status={DWAGON: [local_download]},
        instances={instance.instance_id: instance},
        all_runners={runner_id: RunnerIdle() for runner_id in runner_ids},
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(connect_task, ConnectToGroup)
    assert connect_task.runner_id == runner_ids[0]

    loaded_runners = {
        runner_id: FakeRunnerSupervisor(
            bound_instance=runner.bound_instance,
            status=RunnerLoaded(),
        )
        for runner_id, runner in idle_runners.items()
    }
    warmup_task = plan(
        node_id=DWAGON,
        runners=cast(Mapping[RunnerId, RunnerSupervisor], cast(object, loaded_runners)),
        global_download_status={},
        instances={instance.instance_id: instance},
        all_runners={runner_id: RunnerLoaded() for runner_id in runner_ids},
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(warmup_task, StartWarmup)
    assert warmup_task.runner_id == runner_ids[1]


def test_generation_dispatch_targets_all_eligible_local_ranks() -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    task = TextGeneration(
        task_id=TaskId("generation-task"),
        instance_id=instance.instance_id,
        task_status=TaskStatus.Pending,
        command_id=CommandId("generation-command"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    first_bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[0],
        bound_node_id=DWAGON,
    )
    second_bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[1],
        bound_node_id=DWAGON,
    )
    runners = {
        runner_ids[0]: FakeRunnerSupervisor(
            bound_instance=first_bound_instance,
            status=RunnerReady(),
        ),
        runner_ids[1]: FakeRunnerSupervisor(
            bound_instance=second_bound_instance,
            status=RunnerReady(),
        ),
    }

    target_runner_ids = get_local_runner_ids_for_task(
        task,
        instance,
        DWAGON,
        cast(Mapping[RunnerId, RunnerSupervisor], cast(object, runners)),
    )

    assert target_runner_ids == runner_ids[:2]


@pytest.mark.anyio
async def test_generation_task_is_started_on_both_local_ranks() -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    task = TextGeneration(
        task_id=TaskId("fanout-generation-task"),
        instance_id=instance.instance_id,
        task_status=TaskStatus.Pending,
        command_id=CommandId("fanout-generation-command"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    recording_runners = {
        runner_id: _RecordingRunner(
            BoundInstance(
                instance=instance,
                bound_runner_id=runner_id,
                bound_node_id=DWAGON,
            )
        )
        for runner_id in runner_ids[:2]
    }

    failures = await start_local_runner_task(
        task,
        instance,
        DWAGON,
        cast(Mapping[RunnerId, RunnerSupervisor], cast(object, recording_runners)),
    )

    assert failures == ()
    assert all(runner.started == [task] for runner in recording_runners.values())


@pytest.mark.anyio
async def test_generation_task_start_timeout_is_contained_per_local_rank() -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    task = TextGeneration(
        task_id=TaskId("timeout-generation-task"),
        instance_id=instance.instance_id,
        task_status=TaskStatus.Pending,
        command_id=CommandId("timeout-generation-command"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    first_runner = _RecordingRunner(
        BoundInstance(
            instance=instance,
            bound_runner_id=runner_ids[0],
            bound_node_id=DWAGON,
        )
    )
    second_runner = _BlockingRunner(
        BoundInstance(
            instance=instance,
            bound_runner_id=runner_ids[1],
            bound_node_id=DWAGON,
        )
    )
    recording_runners = {
        runner_ids[0]: first_runner,
        runner_ids[1]: second_runner,
    }

    failures = await start_local_runner_task(
        task,
        instance,
        DWAGON,
        cast(Mapping[RunnerId, RunnerSupervisor], cast(object, recording_runners)),
        timeout_seconds=0.01,
    )

    assert first_runner.started == [task]
    assert second_runner.started == [task]
    assert len(failures) == 1
    assert failures[0].runner_id == runner_ids[1]
    assert failures[0].task_status == TaskStatus.TimedOut
    assert "Timed out after 0.01s" in failures[0].error_message


@pytest.mark.parametrize(
    ("failed_runner_type", "expected_task_status"),
    [
        (_RaisingRunner, TaskStatus.Failed),
        (_TimeoutRunner, TaskStatus.TimedOut),
    ],
)
@pytest.mark.anyio
async def test_one_local_rank_failure_is_attributed_without_cancelling_sibling(
    failed_runner_type: type[_RecordingRunner],
    expected_task_status: TaskStatus,
) -> None:
    instance, runner_ids, _ = _resource_bound_instance()
    task = TextGeneration(
        task_id=TaskId("contained-generation-task"),
        instance_id=instance.instance_id,
        command_id=CommandId("contained-generation-command"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    healthy_runner = _RecordingRunner(
        BoundInstance(
            instance=instance,
            bound_runner_id=runner_ids[0],
            bound_node_id=DWAGON,
        )
    )
    failed_runner = failed_runner_type(
        BoundInstance(
            instance=instance,
            bound_runner_id=runner_ids[1],
            bound_node_id=DWAGON,
        )
    )
    event_sender, event_receiver = channel[Event]()
    worker = object.__new__(Worker)
    worker.node_id = DWAGON
    worker.state = State(instances={instance.instance_id: instance})
    worker.runners = cast(
        dict[RunnerId, RunnerSupervisor],
        cast(
            object,
            {runner_ids[0]: healthy_runner, runner_ids[1]: failed_runner},
        ),
    )
    worker.event_sender = event_sender

    await worker._start_runner_task(task)  # pyright: ignore[reportPrivateUsage]

    task_status = await event_receiver.receive()
    runner_status = await event_receiver.receive()
    assert healthy_runner.started == [task]
    assert isinstance(task_status, TaskStatusUpdated)
    assert task_status.runner_id == runner_ids[1]
    assert task_status.task_status == expected_task_status
    assert isinstance(runner_status, RunnerStatusUpdated)
    assert runner_status.runner_id == runner_ids[1]
    assert isinstance(runner_status.runner_status, RunnerFailed)


def test_planner_keeps_dispatching_running_task_to_unfinished_local_rank() -> None:
    instance, runner_ids, resources = _resource_bound_instance()
    task = TextGeneration(
        task_id=TaskId("partially-complete-generation-task"),
        instance_id=instance.instance_id,
        task_status=TaskStatus.Running,
        command_id=CommandId("partially-complete-generation-command"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    runners = {
        runner_id: FakeRunnerSupervisor(
            bound_instance=BoundInstance(
                instance=instance,
                bound_runner_id=runner_id,
                bound_node_id=DWAGON,
            ),
            status=RunnerReady(),
            completed={task.task_id} if rank == 0 else set(),
        )
        for rank, runner_id in enumerate(runner_ids[:2])
    }
    all_runners = {runner_id: RunnerReady() for runner_id in runner_ids}

    planned = plan(
        node_id=DWAGON,
        runners=cast(Mapping[RunnerId, RunnerSupervisor], cast(object, runners)),
        global_download_status={},
        instances={instance.instance_id: instance},
        all_runners=all_runners,
        tasks={task.task_id: task},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        node_compute_resources=resources,
        runner_backoff=KeyedBackoff(),
    )
    assert planned is task

    runners[runner_ids[1]].completed.add(task.task_id)
    planned = plan(
        node_id=DWAGON,
        runners=cast(Mapping[RunnerId, RunnerSupervisor], cast(object, runners)),
        global_download_status={},
        instances={instance.instance_id: instance},
        all_runners=all_runners,
        tasks={task.task_id: task},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        node_compute_resources=resources,
        runner_backoff=KeyedBackoff(),
    )
    assert planned is None
