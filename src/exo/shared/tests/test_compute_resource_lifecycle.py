from exo.shared.apply import (
    apply_instance_deleted,
    apply_node_timed_out,
    apply_runner_status_updated,
    apply_task_status_updated,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.compute_resources import NvidiaGpuComputeResource
from exo.shared.types.events import (
    InstanceDeleted,
    NodeTimedOut,
    RunnerStatusUpdated,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import Shutdown, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerShutdown,
    RunnerShuttingDown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _gpu_resource(index: int) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=f"GPU-00000000-0000-0000-0000-{index:012d}",
        pci_bus_id=f"00000000:{index + 32:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
    )


def _ring_instance(
    *,
    node_id: NodeId,
    runner_id: RunnerId,
    resource: NvidiaGpuComputeResource | None,
) -> MlxRingInstance:
    model_card = ModelCard(
        model_id=ModelId("resource-lifecycle-model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    shard = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    compute_resource_to_runner = (
        {resource.resource_id: runner_id} if resource is not None else {}
    )
    compute_resource_to_node = (
        {resource.resource_id: node_id} if resource is not None else {}
    )
    return MlxRingInstance(
        instance_id=InstanceId("ring-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
            compute_resource_to_runner=compute_resource_to_runner,
            compute_resource_to_node=compute_resource_to_node,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_deleted_instance_keeps_gpu_until_runner_shutdown() -> None:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    resource = _gpu_resource(1)
    instance = _ring_instance(node_id=node_id, runner_id=runner_id, resource=resource)
    shutdown_task_id = TaskId("shutdown-task")
    state = State(
        instances={instance.instance_id: instance},
        runners={runner_id: RunnerIdle()},
        tasks={
            shutdown_task_id: Shutdown(
                task_id=shutdown_task_id,
                instance_id=instance.instance_id,
                runner_id=runner_id,
            )
        },
        node_compute_resources={node_id: [resource]},
    )

    state = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id), state
    )

    assert instance.instance_id not in state.instances
    assert state.retiring_compute_resources == {resource.resource_id: runner_id}

    state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShuttingDown()),
        state,
    )
    state = apply_runner_status_updated(
        RunnerStatusUpdated(
            runner_id=runner_id,
            runner_status=RunnerFailed(error_message="failed", diagnostics=[]),
        ),
        state,
    )
    state = apply_task_status_updated(
        TaskStatusUpdated(
            task_id=shutdown_task_id,
            task_status=TaskStatus.TimedOut,
            runner_id=runner_id,
        ),
        state,
    )

    assert state.retiring_compute_resources == {resource.resource_id: runner_id}

    state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()),
        state,
    )

    assert state.retiring_compute_resources == {}


def test_missing_runner_status_leases_until_shutdown_ack() -> None:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    resource = _gpu_resource(1)
    instance = _ring_instance(node_id=node_id, runner_id=runner_id, resource=resource)

    missing_status = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id),
        State(instances={instance.instance_id: instance}),
    )

    assert missing_status.retiring_compute_resources == {
        resource.resource_id: runner_id
    }

    acknowledged = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()),
        missing_status,
    )

    assert acknowledged.retiring_compute_resources == {}


def test_deleting_instance_with_known_shutdown_runner_does_not_lease_gpu() -> None:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    resource = _gpu_resource(1)
    instance = _ring_instance(node_id=node_id, runner_id=runner_id, resource=resource)
    already_shutdown = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id),
        State(
            instances={instance.instance_id: instance},
            runners={runner_id: RunnerShutdown()},
        ),
    )

    assert already_shutdown.retiring_compute_resources == {}


def test_resource_less_ring_leases_all_node_gpus_until_node_timeout() -> None:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    resources = [_gpu_resource(1), _gpu_resource(2)]
    instance = _ring_instance(node_id=node_id, runner_id=runner_id, resource=None)
    state = State(
        instances={instance.instance_id: instance},
        runners={runner_id: RunnerIdle()},
        node_compute_resources={node_id: resources},
    )

    state = apply_instance_deleted(
        InstanceDeleted(instance_id=instance.instance_id), state
    )

    assert state.retiring_compute_resources == {
        resource.resource_id: runner_id for resource in resources
    }

    state = apply_node_timed_out(NodeTimedOut(node_id=node_id), state)

    assert state.retiring_compute_resources == {}
