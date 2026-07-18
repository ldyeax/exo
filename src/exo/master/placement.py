from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from exo.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_nccl_coordinator,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.shared.models.model_cards import ModelCard, ModelId, model_snapshot_id
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.compute_resources import ComputeResource, ComputeResourceId
from exo.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo, NodeRdmaCtlStatus
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxNcclInstance,
    MlxRingInstance,
    instance_compute_resource_runners,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import Sharding
from exo.utils.ports import random_ephemeral_port

INSTANCE_META_BACKENDS: dict[InstanceMeta, list[Backend]] = {
    InstanceMeta.MlxRing: [Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    InstanceMeta.MlxJaccl: [Backend.MlxMetal],
    InstanceMeta.MlxNccl: [Backend.MlxCuda],
}

GPU_PLACEMENT_MINIMUM_HEADROOM_BYTES = 1024**3
GPU_PLACEMENT_HEADROOM_PERCENT = 10


def estimated_gpu_rank_memory_requirement(
    model_storage: Memory,
    world_size: int,
    gpu_total_memory: Memory,
) -> Memory:
    """Estimate sharded weights plus conservative placement-only GPU headroom.

    This intentionally does not claim to model KV cache or runtime workspace.
    """
    if world_size <= 0:
        raise ValueError("Tensor world size must be positive")
    model_share_bytes = (model_storage.in_bytes + world_size - 1) // world_size
    proportional_headroom_bytes = (
        gpu_total_memory.in_bytes * GPU_PLACEMENT_HEADROOM_PERCENT + 99
    ) // 100
    headroom_bytes = max(
        GPU_PLACEMENT_MINIMUM_HEADROOM_BYTES,
        proportional_headroom_bytes,
    )
    return Memory.from_bytes(model_share_bytes + headroom_bytes)


def _compute_resource_inventory_by_id(
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]],
) -> dict[ComputeResourceId, tuple[NodeId, ComputeResource]]:
    inventory: dict[ComputeResourceId, tuple[NodeId, ComputeResource]] = {}
    for node_id, resources in node_compute_resources.items():
        for resource in resources:
            if resource.resource_id in inventory:
                raise ValueError(
                    f"Compute resource {resource.resource_id} is advertised more than once"
                )
            inventory[resource.resource_id] = (node_id, resource)
    return inventory


def _occupied_compute_resource_ids(
    instances: Mapping[InstanceId, Instance],
    *,
    excluding_instance_id: InstanceId | None = None,
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]] | None = None,
    retiring_compute_resources: Mapping[ComputeResourceId, RunnerId] | None = None,
) -> set[ComputeResourceId]:
    occupied_resource_ids = set(retiring_compute_resources or {})
    occupied_resource_ids.update(
        resource_id
        for instance_id, instance in instances.items()
        if instance_id != excluding_instance_id
        for resource_id in instance_compute_resource_runners(
            instance, node_compute_resources or {}
        )
    )
    return occupied_resource_ids


def validate_instance_compute_resources(
    instance: Instance,
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]],
    current_instances: Mapping[InstanceId, Instance] | None = None,
    retiring_compute_resources: Mapping[ComputeResourceId, RunnerId] | None = None,
) -> None:
    assignments = instance.shard_assignments
    resource_assignments = assignments.compute_resource_to_runner
    if (
        isinstance(instance, MlxNcclInstance)
        and any(node_compute_resources.values())
        and not resource_assignments
    ):
        raise ValueError(
            "New MlxNccl instances require explicit compute resource bindings "
            "when live GPU inventory is available"
        )
    inventory = _compute_resource_inventory_by_id(node_compute_resources)
    requested_resource_ids = set(
        instance_compute_resource_runners(instance, node_compute_resources)
    )
    occupied_resource_ids = _occupied_compute_resource_ids(
        current_instances or {},
        excluding_instance_id=instance.instance_id,
        node_compute_resources=node_compute_resources,
        retiring_compute_resources=retiring_compute_resources,
    )
    conflicting_resource_ids = requested_resource_ids & occupied_resource_ids
    if conflicting_resource_ids:
        raise ValueError(
            "Compute resources are already occupied by another instance: "
            f"{sorted(conflicting_resource_ids)}"
        )
    if not resource_assignments:
        return
    resource_owners = assignments.compute_resource_to_node
    if not resource_owners:
        raise ValueError(
            "Resource-bound instances require explicit compute resource ownership"
        )

    world_size = len(assignments.runner_to_shard)
    for resource_id, runner_id in resource_assignments.items():
        inventory_entry = inventory.get(resource_id)
        if inventory_entry is None:
            raise ValueError(f"Compute resource {resource_id} is not live")
        live_node_id, resource = inventory_entry
        assigned_node_id = resource_owners[resource_id]
        if live_node_id != assigned_node_id:
            raise ValueError(
                f"Compute resource {resource_id} is assigned to {assigned_node_id} "
                f"but advertised by {live_node_id}"
            )
        model_storage = assignments.runner_to_shard[runner_id].model_card.storage_size
        required_memory = estimated_gpu_rank_memory_requirement(
            model_storage,
            world_size,
            resource.total_memory,
        )
        if required_memory > resource.total_memory:
            raise ValueError(
                f"Compute resource {resource_id} has {resource.total_memory} but the "
                f"placement estimate requires {required_memory}: ceil(model storage / "
                "world size) + max(1 GiB, 10% GPU VRAM). KV cache and runtime "
                "workspace are not modeled"
            )


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]] | None = None,
    retiring_compute_resources: Mapping[ComputeResourceId, RunnerId] | None = None,
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    if node_compute_resources is not None:
        validate_instance_compute_resources(
            command.instance,
            node_compute_resources,
            current_instances,
            retiring_compute_resources,
        )

    return {**current_instances, command.instance.instance_id: command.instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_card: ModelCard,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction for an exact model snapshot on a node."""
    target_snapshot_id = model_snapshot_id(model_card)
    for progress in download_status.get(node_id, []):
        if model_snapshot_id(progress.shard_metadata.model_card) != target_snapshot_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_card: ModelCard,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_card, download_status)
        for node_id in cycle
    )


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    node_backends: Mapping[NodeId, list[Backend]],
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] | None = None,
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]] | None = None,
    retiring_compute_resources: Mapping[ComputeResourceId, RunnerId] | None = None,
) -> dict[InstanceId, Instance]:
    if (
        command.instance_meta == InstanceMeta.MlxNccl
        and command.sharding != Sharding.Tensor
    ):
        raise ValueError(
            "MlxNccl requires Tensor sharding because MLX NCCL does not support "
            "point-to-point pipeline communication"
        )

    cycles = topology.get_cycles()
    minimum_nodes = (
        max(command.min_nodes, 2)
        if command.instance_meta == InstanceMeta.MlxNccl
        else command.min_nodes
    )
    candidate_cycles = list(filter(lambda it: len(it) >= minimum_nodes, cycles))

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, node_memory, command.model_card.storage_size
    )
    if len(cycles_with_sufficient_memory) == 0:
        raise ValueError("No cycles found with sufficient memory")

    placement_compute_resources: dict[NodeId, Sequence[ComputeResource]] | None = None
    if command.instance_meta == InstanceMeta.MlxNccl and node_compute_resources:
        _ = _compute_resource_inventory_by_id(node_compute_resources)
        occupied_resource_ids = _occupied_compute_resource_ids(
            current_instances,
            node_compute_resources=node_compute_resources,
            retiring_compute_resources=retiring_compute_resources,
        )
        placement_compute_resources = {
            node_id: [
                resource
                for resource in resources
                if resource.resource_id not in occupied_resource_ids
            ]
            for node_id, resources in node_compute_resources.items()
        }
    if placement_compute_resources is not None:
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if all(placement_compute_resources.get(node_id) for node_id in cycle)
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "No cycles found where every node advertises an available NVIDIA "
                "GPU compute resource"
            )

    def selected_compute_resources(
        cycle: Cycle,
    ) -> dict[NodeId, Sequence[ComputeResource]] | None:
        if placement_compute_resources is None:
            return None
        resources_by_node: dict[NodeId, Sequence[ComputeResource]] = {
            node_id: sorted(
                placement_compute_resources[node_id],
                key=lambda resource: resource.resource_id,
            )
            for node_id in cycle
        }
        if command.use_all_compute_resources:
            world_size = sum(len(resources) for resources in resources_by_node.values())
            if all(
                estimated_gpu_rank_memory_requirement(
                    command.model_card.storage_size,
                    world_size,
                    resource.total_memory,
                )
                <= resource.total_memory
                for resources in resources_by_node.values()
                for resource in resources
            ):
                return resources_by_node
            return None

        world_size = len(cycle)
        selection: dict[NodeId, Sequence[ComputeResource]] = {}
        for node_id, resources in resources_by_node.items():
            capable_resource = next(
                (
                    resource
                    for resource in resources
                    if estimated_gpu_rank_memory_requirement(
                        command.model_card.storage_size,
                        world_size,
                        resource.total_memory,
                    )
                    <= resource.total_memory
                ),
                None,
            )
            if capable_resource is None:
                return None
            selection[node_id] = [capable_resource]
        return selection

    if placement_compute_resources is not None:
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if selected_compute_resources(cycle) is not None
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "No available GPU selection satisfies the conservative placement "
                "estimate: ceil(model storage / world size) + max(1 GiB, 10% GPU "
                "VRAM). KV cache and runtime workspace are not modeled"
            )

    if command.sharding == Sharding.Tensor:
        if not command.model_card.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        # DeepSeek V4 is MQA (num_key_value_heads=1) but its sharding strategy
        # head-parallelises wq_b/wo_a and shards MoE experts instead of splitting
        # KV heads, so the kv-head divisibility check doesn't apply.
        is_deepseek_v4 = command.model_card.base_model.startswith("DeepSeek V4")
        kv_heads = command.model_card.num_key_value_heads

        def tensor_world_size(cycle: Cycle) -> int:
            compute_resources = selected_compute_resources(cycle)
            if compute_resources is None:
                assert placement_compute_resources is None
                return len(cycle)
            return sum(len(compute_resources[node_id]) for node_id in cycle)

        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_card.hidden_size % tensor_world_size(cycle) == 0
            and (
                is_deepseek_v4
                or kv_heads is None
                or kv_heads % tensor_world_size(cycle) == 0
            )
        ]
        if not cycles_with_sufficient_memory:
            resource_policy = (
                " using all advertised compute resources"
                if placement_compute_resources is not None
                and command.use_all_compute_resources
                else ""
            )
            raise ValueError(
                f"No tensor sharding found for model with "
                f"hidden_size={command.model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles{resource_policy}"
            )
    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )
    if (
        command.sharding == Sharding.Pipeline
        and command.model_card.base_model.startswith("Gemma 4")
    ):
        cycles_with_sufficient_memory = [
            cycle for cycle in cycles_with_sufficient_memory if len(cycle) == 1
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead."
            )

    required_backends = set(INSTANCE_META_BACKENDS[command.instance_meta]) & set(
        command.model_card.backends
    )
    if not required_backends:
        raise ValueError(
            f"Model {command.model_card.model_id} backends "
            f"{sorted(b.value for b in command.model_card.backends)} cannot satisfy engine "
            f"{command.instance_meta.value} which requires "
            f"{sorted(b.value for b in INSTANCE_META_BACKENDS[command.instance_meta])}"
        )
    backend_compatible_cycles = [
        cycle
        for cycle in cycles_with_sufficient_memory
        if all(
            set(node_backends.get(node_id, [])) & required_backends for node_id in cycle
        )
    ]
    if not backend_compatible_cycles:
        raise ValueError(
            f"No cycle where every node supports a backend in "
            f"{sorted(b.value for b in required_backends)} for {command.model_card.model_id}"
        )
    smallest_cycles = get_smallest_cycles(backend_compatible_cycles)

    rdma_ctl_status = node_rdma_ctl or {}

    def _all_rdma_ctl_enabled(cycle: Cycle) -> bool:
        return all(
            ((status := rdma_ctl_status.get(node_id)) is not None and status.enabled)
            for node_id in cycle
        )

    smallest_rdma_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.is_rdma_cycle(cycle) and _all_rdma_ctl_enabled(cycle)
    ]

    if command.instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise ValueError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles

    cycles_with_leaf_nodes: list[Cycle] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]

    resolved_download_status = download_status or {}
    candidate_cycles = (
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles
    )

    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
            _cycle_download_score(cycle, command.model_card, resolved_download_status),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        ),
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command = command.model_copy(
            update={
                "instance_meta": InstanceMeta.MlxRing,
                "sharding": Sharding.Pipeline,
            }
        )

    shard_assignments = get_shard_assignments(
        command.model_card,
        selected_cycle,
        command.sharding,
        node_memory,
        node_compute_resources=selected_compute_resources(selected_cycle),
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    def get_rank_zero_node() -> NodeId:
        zero_node_ids = [
            node_id
            for node_id in selected_cycle.node_ids
            if shard_assignments.runner_to_shard[
                shard_assignments.node_to_runner[node_id]
            ].device_rank
            == 0
        ]
        assert len(zero_node_ids) == 1
        return zero_node_ids[0]

    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            coordinator_node_id = get_rank_zero_node()

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                [node_id for node_id in selected_cycle],
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxNccl:
            coordinator_node_id = get_rank_zero_node()
            nccl_coordinator = get_mlx_nccl_coordinator(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxNcclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                nccl_coordinator=nccl_coordinator,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    if node_compute_resources is not None:
        validate_instance_compute_resources(
            target_instances[instance_id],
            node_compute_resources,
            current_instances,
            retiring_compute_resources,
        )

    return target_instances


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    # CancelDownload is model-id-wide, so deduplicate multiple revisions while
    # preserving state order and keep all of them if any matching instance is active.
    currently_downloading = tuple(
        dict.fromkeys(
            (node_id, progress.shard_metadata.model_card.model_id)
            for node_id, progress_by_model in download_status.items()
            for progress in progress_by_model
            if isinstance(progress, DownloadOngoing)
        )
    )
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
