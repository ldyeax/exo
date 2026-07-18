from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.compute_resources import (
    ComputeResource,
    NvidiaGpuComputeResource,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import (
    ConnectToGroup,
    CreateRunner,
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
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerReady,
    ShardAssignments,
)
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import get_local_runner_ids_for_task, start_local_runner_task
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

    await start_local_runner_task(
        task,
        instance,
        DWAGON,
        cast(Mapping[RunnerId, RunnerSupervisor], cast(object, recording_runners)),
    )

    assert all(runner.started == [task] for runner in recording_runners.values())
