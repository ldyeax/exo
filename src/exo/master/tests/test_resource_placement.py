from collections.abc import Sequence

import pytest

from exo.master.placement import (
    add_instance_to_placements,
    estimated_gpu_rank_memory_requirement,
    place_instance,
    validate_instance_compute_resources,
)
from exo.master.placement_utils import get_shard_assignments_for_tensor_parallel
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import CreateInstance, PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.compute_resources import (
    ComputeResource,
    ComputeResourceId,
    NvidiaGpuComputeResource,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxNcclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import Sharding


def _gpu_resource(index: int, *, total_memory_gb: int = 24) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=f"GPU-00000000-0000-0000-0000-{index:012d}",
        pci_bus_id=f"00000000:{index + 32:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=total_memory_gb * 1024**3,
    )


def _model_card(
    *,
    hidden_size: int = 30,
    storage_size: Memory | None = None,
    num_key_value_heads: int | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("resource-placement-model"),
        storage_size=storage_size or Memory.from_mb(1),
        n_layers=2,
        hidden_size=hidden_size,
        supports_tensor=True,
        num_key_value_heads=num_key_value_heads,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )


def _two_node_topology(
    node_a: NodeId, node_b: NodeId
) -> tuple[Topology, dict[NodeId, NodeNetworkInfo]]:
    topology = Topology()
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )
    node_network = {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address="192.0.2.1",
                    interface_type="ethernet",
                )
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address="192.0.2.2",
                    interface_type="ethernet",
                )
            ]
        ),
    }
    return topology, node_network


def _place_cuda_runtime(
    instance_meta: InstanceMeta,
    *,
    topology: Topology,
    node_network: dict[NodeId, NodeNetworkInfo],
    resources: dict[NodeId, Sequence[ComputeResource]],
    current_instances: dict[InstanceId, Instance] | None = None,
    retiring_compute_resources: dict[ComputeResourceId, RunnerId] | None = None,
) -> dict[InstanceId, Instance]:
    node_memory = {node_id: create_node_memory(2 * 1024**3) for node_id in resources}
    return place_instance(
        PlaceInstance(
            command_id=CommandId(f"place-{instance_meta.value}"),
            model_card=_model_card(hidden_size=2048),
            sharding=(
                Sharding.Tensor
                if instance_meta == InstanceMeta.MlxNccl
                else Sharding.Pipeline
            ),
            instance_meta=instance_meta,
            min_nodes=2,
        ),
        topology,
        current_instances or {},
        node_memory,
        node_network,
        {node_id: [Backend.MlxCuda] for node_id in resources},
        node_compute_resources=resources,
        retiring_compute_resources=retiring_compute_resources,
    )


def test_tensor_assignments_expand_each_gpu_into_a_rank() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, _ = _two_node_topology(node_a, node_b)
    cycle = next(cycle for cycle in topology.get_cycles() if len(cycle) == 2)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }

    assignments = get_shard_assignments_for_tensor_parallel(
        _model_card(), cycle, resources
    )

    assert len(assignments.runner_to_shard) == 3
    assert len(assignments.node_to_runner) == 2
    assert len(assignments.compute_resource_to_runner) == 3
    assert assignments.compute_resource_to_node == {
        resource.resource_id: node_id
        for node_id, node_resources in resources.items()
        for resource in node_resources
    }
    assert {shard.world_size for shard in assignments.runner_to_shard.values()} == {3}
    assert {shard.device_rank for shard in assignments.runner_to_shard.values()} == {
        0,
        1,
        2,
    }
    for node_id in cycle:
        representative = assignments.node_to_runner[node_id]
        node_ranks = [
            assignments.runner_to_shard[
                assignments.compute_resource_to_runner[resource.resource_id]
            ].device_rank
            for resource in resources[node_id]
        ]
        assert assignments.runner_to_shard[representative].device_rank == min(
            node_ranks
        )


def test_resource_bound_nccl_placement_creates_three_rank_instance() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-resource-instance"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        use_all_compute_resources=True,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assert len(instance.shard_assignments.runner_to_shard) == 3
    assert set(instance.shard_assignments.compute_resource_to_runner) == {
        resource.resource_id
        for node_resources in resources.values()
        for resource in node_resources
    }
    assert instance.shard_assignments.compute_resource_to_node == {
        resource.resource_id: node_id
        for node_id, node_resources in resources.items()
        for resource in node_resources
    }


def test_default_nccl_placement_selects_one_gpu_per_node() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(2), _gpu_resource(1)],
        node_b: [_gpu_resource(3)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-default-resource-instance"),
        model_card=_model_card(hidden_size=2048),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assert len(instance.shard_assignments.runner_to_shard) == 2
    assert set(instance.shard_assignments.compute_resource_to_runner) == {
        _gpu_resource(1).resource_id,
        _gpu_resource(3).resource_id,
    }


def test_second_nccl_placement_does_not_reuse_occupied_gpus() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3), _gpu_resource(4)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-two-resource-instances"),
        model_card=_model_card(hidden_size=2048),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )
    node_memory = {
        node_a: create_node_memory(2 * 1024**3),
        node_b: create_node_memory(2 * 1024**3),
    }
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}

    first_placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_compute_resources=resources,
    )
    second_placements = place_instance(
        command,
        topology,
        first_placements,
        node_memory,
        node_network,
        node_backends,
        node_compute_resources=resources,
    )

    first_instance = next(iter(first_placements.values()))
    second_instance = next(
        instance
        for instance_id, instance in second_placements.items()
        if instance_id not in first_placements
    )
    assert set(first_instance.shard_assignments.compute_resource_to_runner) == {
        _gpu_resource(1).resource_id,
        _gpu_resource(3).resource_id,
    }
    assert set(second_instance.shard_assignments.compute_resource_to_runner) == {
        _gpu_resource(2).resource_id,
        _gpu_resource(4).resource_id,
    }
    assert not (
        set(first_instance.shard_assignments.compute_resource_to_runner)
        & set(second_instance.shard_assignments.compute_resource_to_runner)
    )


def test_legacy_nccl_instance_reserves_all_gpus_on_its_nodes() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3), _gpu_resource(4)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-after-legacy-instance"),
        model_card=_model_card(hidden_size=2048),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )
    node_memory = {
        node_a: create_node_memory(2 * 1024**3),
        node_b: create_node_memory(2 * 1024**3),
    }
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}
    legacy_placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_compute_resources={},
    )

    with pytest.raises(ValueError, match="available NVIDIA GPU"):
        place_instance(
            command,
            topology,
            legacy_placements,
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )


def test_resource_less_ring_blocks_nccl_placement() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1)],
        node_b: [_gpu_resource(2)],
    }
    ring_placements = _place_cuda_runtime(
        InstanceMeta.MlxRing,
        topology=topology,
        node_network=node_network,
        resources=resources,
    )
    assert isinstance(next(iter(ring_placements.values())), MlxRingInstance)

    with pytest.raises(ValueError, match="available NVIDIA GPU"):
        _place_cuda_runtime(
            InstanceMeta.MlxNccl,
            topology=topology,
            node_network=node_network,
            resources=resources,
            current_instances=ring_placements,
        )


def test_resource_bound_nccl_blocks_ring_placement() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1)],
        node_b: [_gpu_resource(2)],
    }
    nccl_placements = _place_cuda_runtime(
        InstanceMeta.MlxNccl,
        topology=topology,
        node_network=node_network,
        resources=resources,
    )

    with pytest.raises(ValueError, match="already occupied"):
        _place_cuda_runtime(
            InstanceMeta.MlxRing,
            topology=topology,
            node_network=node_network,
            resources=resources,
            current_instances=nccl_placements,
        )


def test_direct_creation_rejects_ring_nccl_collisions_in_both_orders() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1)],
        node_b: [_gpu_resource(2)],
    }
    ring_placements = _place_cuda_runtime(
        InstanceMeta.MlxRing,
        topology=topology,
        node_network=node_network,
        resources=resources,
    )
    nccl_placements = _place_cuda_runtime(
        InstanceMeta.MlxNccl,
        topology=topology,
        node_network=node_network,
        resources=resources,
    )
    ring_instance = next(iter(ring_placements.values()))
    nccl_instance = next(iter(nccl_placements.values()))

    with pytest.raises(ValueError, match="already occupied"):
        add_instance_to_placements(
            CreateInstance(instance=nccl_instance),
            topology,
            ring_placements,
            resources,
        )
    with pytest.raises(ValueError, match="already occupied"):
        add_instance_to_placements(
            CreateInstance(instance=ring_instance),
            topology,
            nccl_placements,
            resources,
        )


def test_retiring_compute_resources_block_placement_and_direct_creation() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1)],
        node_b: [_gpu_resource(2)],
    }
    ring_instance = next(
        iter(
            _place_cuda_runtime(
                InstanceMeta.MlxRing,
                topology=topology,
                node_network=node_network,
                resources=resources,
            ).values()
        )
    )
    retiring_compute_resources = {
        resource.resource_id: RunnerId(f"retiring-{node_id}")
        for node_id, node_resources in resources.items()
        for resource in node_resources
    }

    with pytest.raises(ValueError, match="available NVIDIA GPU"):
        _place_cuda_runtime(
            InstanceMeta.MlxNccl,
            topology=topology,
            node_network=node_network,
            resources=resources,
            retiring_compute_resources=retiring_compute_resources,
        )
    with pytest.raises(ValueError, match="already occupied"):
        add_instance_to_placements(
            CreateInstance(instance=ring_instance),
            topology,
            {},
            resources,
            retiring_compute_resources,
        )


@pytest.mark.parametrize("backend", [Backend.MlxCpu, Backend.MlxMetal])
def test_resource_less_ring_does_not_collide_without_nvidia_inventory(
    backend: Backend,
) -> None:
    node_id = NodeId(f"{backend.value}-node")
    topology = Topology()
    topology.add_node(node_id)
    model_card = _model_card().model_copy(update={"backends": [backend]})
    command = PlaceInstance(
        command_id=CommandId(f"place-{backend.value}"),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placement_arguments = (
        command,
        topology,
    )
    first_placements = place_instance(
        *placement_arguments,
        {},
        {node_id: create_node_memory(2 * 1024**3)},
        {node_id: NodeNetworkInfo()},
        {node_id: [backend]},
        node_compute_resources={node_id: []},
    )

    second_placements = place_instance(
        *placement_arguments,
        first_placements,
        {node_id: create_node_memory(2 * 1024**3)},
        {node_id: NodeNetworkInfo()},
        {node_id: [backend]},
        node_compute_resources={node_id: []},
    )

    assert len(second_placements) == 2


def test_default_nccl_placement_skips_gpu_that_cannot_fit_rank_estimate() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [
            _gpu_resource(1, total_memory_gb=4),
            _gpu_resource(2, total_memory_gb=16),
        ],
        node_b: [_gpu_resource(3, total_memory_gb=16)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-heterogeneous-resource-instance"),
        model_card=_model_card(
            hidden_size=2048,
            storage_size=Memory.from_bytes(20 * 1024**3),
        ),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(32 * 1024**3),
            node_b: create_node_memory(32 * 1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.compute_resource_to_runner) == {
        _gpu_resource(2).resource_id,
        _gpu_resource(3).resource_id,
    }


def test_nccl_placement_rejects_gpu_selection_too_small_for_rank_estimate() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1, total_memory_gb=12)],
        node_b: [_gpu_resource(2, total_memory_gb=12)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-too-large-resource-instance"),
        model_card=_model_card(
            hidden_size=2048,
            storage_size=Memory.from_bytes(24 * 1024**3),
        ),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="KV cache and runtime workspace"):
        place_instance(
            command,
            topology,
            {},
            {
                node_a: create_node_memory(32 * 1024**3),
                node_b: create_node_memory(32 * 1024**3),
            },
            node_network,
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_compute_resources=resources,
        )


def test_gpu_rank_memory_estimate_uses_semantic_percentage_rounding() -> None:
    gibibyte = 1024**3
    gpu_bytes = 24 * gibibyte

    required = estimated_gpu_rank_memory_requirement(
        Memory.from_bytes(20 * gibibyte),
        world_size=2,
        gpu_total_memory=Memory.from_bytes(gpu_bytes),
    )

    assert required.in_bytes == 10 * gibibyte + (gpu_bytes * 10 + 99) // 100


def test_all_gpu_nccl_placement_rejects_incompatible_world_size() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-incompatible-resource-instance"),
        model_card=_model_card(hidden_size=2048),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        use_all_compute_resources=True,
    )

    with pytest.raises(ValueError, match="all advertised compute resources"):
        place_instance(
            command,
            topology,
            {},
            {
                node_a: create_node_memory(1024**3),
                node_b: create_node_memory(1024**3),
            },
            node_network,
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_compute_resources=resources,
        )


def test_place_instance_resource_policy_roundtrip() -> None:
    command = PlaceInstance(
        command_id=CommandId("place-all-resources"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        use_all_compute_resources=True,
    )

    restored = PlaceInstance.model_validate_json(command.model_dump_json())

    assert restored.use_all_compute_resources
    assert not PlaceInstance(
        command_id=CommandId("place-default-resources"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    ).use_all_compute_resources


def test_place_instance_explicit_resource_policy_roundtrip_and_validation() -> None:
    first_resource_id = _gpu_resource(2).resource_id
    second_resource_id = _gpu_resource(3).resource_id
    command = PlaceInstance(
        command_id=CommandId("place-requested-resources"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        requested_compute_resource_ids=(first_resource_id, second_resource_id),
    )

    restored = PlaceInstance.model_validate_json(command.model_dump_json())

    assert restored.requested_compute_resource_ids == (
        first_resource_id,
        second_resource_id,
    )
    with pytest.raises(ValueError, match="must be unique"):
        PlaceInstance(
            model_card=_model_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(first_resource_id, first_resource_id),
        )
    with pytest.raises(ValueError, match="incompatible"):
        PlaceInstance(
            model_card=_model_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            use_all_compute_resources=True,
            requested_compute_resource_ids=(first_resource_id, second_resource_id),
        )
    with pytest.raises(ValueError, match="requires MlxNccl with Tensor"):
        PlaceInstance(
            model_card=_model_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
            requested_compute_resource_ids=(first_resource_id, second_resource_id),
        )
    with pytest.raises(ValueError, match="not an NVIDIA GPU"):
        PlaceInstance(
            model_card=_model_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(
                ComputeResourceId("other-backend:resource"),
            ),
        )


def test_explicit_selection_uses_exact_gpus_and_request_order_for_ranks() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    first_dwagon_gpu = _gpu_resource(1)
    requested_dwagon_gpu = _gpu_resource(2)
    fwuff_gpu = _gpu_resource(3)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [first_dwagon_gpu, requested_dwagon_gpu],
        node_b: [fwuff_gpu],
    }
    requested_resource_ids = (
        fwuff_gpu.resource_id,
        requested_dwagon_gpu.resource_id,
    )

    placements = place_instance(
        PlaceInstance(
            model_card=_model_card(hidden_size=32, num_key_value_heads=2),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=requested_resource_ids,
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assignments = instance.shard_assignments
    assert tuple(assignments.compute_resource_to_runner) == requested_resource_ids
    assert first_dwagon_gpu.resource_id not in assignments.compute_resource_to_runner
    assert [
        assignments.runner_to_shard[
            assignments.compute_resource_to_runner[resource_id]
        ].device_rank
        for resource_id in requested_resource_ids
    ] == [0, 1]
    assert (
        assignments.node_to_runner[node_b]
        == assignments.compute_resource_to_runner[fwuff_gpu.resource_id]
    )
    assert instance.nccl_coordinator.ip == "169.254.0.2"


def test_explicit_selection_projects_hosts_from_a_larger_common_cycle() -> None:
    node_a = NodeId("dwagon")
    transit_node = NodeId("transit")
    node_b = NodeId("fwuff")
    topology = Topology()
    topology.add_connection(
        Connection(source=node_a, sink=transit_node, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=transit_node, sink=node_b, edge=create_socket_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(1))
    )
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)

    placements = place_instance(
        PlaceInstance(
            model_card=_model_card(hidden_size=32, num_key_value_heads=2),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(
                dwagon_gpu.resource_id,
                fwuff_gpu.resource_id,
            ),
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        {
            node_a: NodeNetworkInfo(),
            node_b: NodeNetworkInfo(),
        },
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources={node_a: [dwagon_gpu], node_b: [fwuff_gpu]},
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assert set(instance.shard_assignments.node_to_runner) == {node_a, node_b}
    assert transit_node not in instance.shard_assignments.node_to_runner
    assert instance.nccl_coordinator.ip == "169.254.0.1"

    with pytest.raises(
        ValueError,
        match="rank-0 coordinator must have one IPv4 address reachable",
    ):
        place_instance(
            PlaceInstance(
                model_card=_model_card(hidden_size=32, num_key_value_heads=2),
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                requested_compute_resource_ids=(
                    fwuff_gpu.resource_id,
                    dwagon_gpu.resource_id,
                ),
            ),
            topology,
            {},
            {
                node_a: create_node_memory(1024**3),
                node_b: create_node_memory(1024**3),
            },
            {
                node_a: NodeNetworkInfo(),
                node_b: NodeNetworkInfo(),
            },
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_compute_resources={node_a: [dwagon_gpu], node_b: [fwuff_gpu]},
        )


def test_explicit_tp2_accepts_two_kv_heads_while_all_resource_tp3_rejects() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    model_card = _model_card(hidden_size=32, num_key_value_heads=2)

    exact_placements = place_instance(
        PlaceInstance(
            model_card=model_card,
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(
                resources[node_a][1].resource_id,
                resources[node_b][0].resource_id,
            ),
        ),
        topology,
        {},
        {node_a: create_node_memory(1024**3), node_b: create_node_memory(1024**3)},
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    assert (
        len(next(iter(exact_placements.values())).shard_assignments.runner_to_shard)
        == 2
    )
    with pytest.raises(ValueError, match="num_key_value_heads=2"):
        place_instance(
            PlaceInstance(
                model_card=model_card,
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                use_all_compute_resources=True,
            ),
            topology,
            {},
            {
                node_a: create_node_memory(1024**3),
                node_b: create_node_memory(1024**3),
            },
            node_network,
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_compute_resources=resources,
        )


def test_explicit_selection_rejects_unknown_occupied_and_retiring_resources() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [dwagon_gpu],
        node_b: [fwuff_gpu],
    }
    node_memory = {
        node_a: create_node_memory(1024**3),
        node_b: create_node_memory(1024**3),
    }
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}

    def requested_command(
        resource_ids: tuple[ComputeResourceId, ...],
    ) -> PlaceInstance:
        return PlaceInstance(
            model_card=_model_card(hidden_size=32, num_key_value_heads=2),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=resource_ids,
        )

    unknown_resource_id = ComputeResourceId.from_nvidia_device_uuid("GPU-unknown")
    with pytest.raises(ValueError, match="not live"):
        place_instance(
            requested_command((dwagon_gpu.resource_id, unknown_resource_id)),
            topology,
            {},
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )

    first_placements = place_instance(
        requested_command((dwagon_gpu.resource_id, fwuff_gpu.resource_id)),
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_compute_resources=resources,
    )
    with pytest.raises(ValueError, match="already occupied or retiring"):
        place_instance(
            requested_command((dwagon_gpu.resource_id, fwuff_gpu.resource_id)),
            topology,
            first_placements,
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )
    with pytest.raises(ValueError, match="already occupied or retiring"):
        place_instance(
            requested_command((dwagon_gpu.resource_id, fwuff_gpu.resource_id)),
            topology,
            {},
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
            retiring_compute_resources={
                dwagon_gpu.resource_id: RunnerId("retiring-runner")
            },
        )


def test_explicit_selection_rejects_low_memory_same_node_and_disconnected_gpus() -> (
    None
):
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    connected_topology, node_network = _two_node_topology(node_a, node_b)
    small_gpu = _gpu_resource(1, total_memory_gb=8)
    second_dwagon_gpu = _gpu_resource(2)
    fwuff_gpu = _gpu_resource(3)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [small_gpu, second_dwagon_gpu],
        node_b: [fwuff_gpu],
    }
    node_memory = {
        node_a: create_node_memory(64 * 1024**3),
        node_b: create_node_memory(64 * 1024**3),
    }
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}

    with pytest.raises(ValueError, match="Requested GPU selection"):
        place_instance(
            PlaceInstance(
                model_card=_model_card(
                    hidden_size=32,
                    storage_size=Memory.from_bytes(16 * 1024**3),
                    num_key_value_heads=2,
                ),
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                requested_compute_resource_ids=(
                    small_gpu.resource_id,
                    fwuff_gpu.resource_id,
                ),
            ),
            connected_topology,
            {},
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )

    with pytest.raises(ValueError, match="at least two nodes"):
        place_instance(
            PlaceInstance(
                model_card=_model_card(hidden_size=32, num_key_value_heads=2),
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                requested_compute_resource_ids=(
                    small_gpu.resource_id,
                    second_dwagon_gpu.resource_id,
                ),
            ),
            connected_topology,
            {},
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )

    disconnected_topology = Topology()
    disconnected_topology.add_node(node_a)
    disconnected_topology.add_node(node_b)
    with pytest.raises(ValueError, match="common connectivity cycle"):
        place_instance(
            PlaceInstance(
                model_card=_model_card(hidden_size=32, num_key_value_heads=2),
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                requested_compute_resource_ids=(
                    second_dwagon_gpu.resource_id,
                    fwuff_gpu.resource_id,
                ),
            ),
            disconnected_topology,
            {},
            node_memory,
            node_network,
            node_backends,
            node_compute_resources=resources,
        )


def test_explicit_selection_rejects_node_without_mlx_cuda_backend() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)

    with pytest.raises(ValueError, match="every node supports a backend"):
        place_instance(
            PlaceInstance(
                model_card=_model_card(hidden_size=32, num_key_value_heads=2),
                sharding=Sharding.Tensor,
                instance_meta=InstanceMeta.MlxNccl,
                min_nodes=2,
                requested_compute_resource_ids=(
                    dwagon_gpu.resource_id,
                    fwuff_gpu.resource_id,
                ),
            ),
            topology,
            {},
            {
                node_a: create_node_memory(1024**3),
                node_b: create_node_memory(1024**3),
            },
            node_network,
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxMetal]},
            node_compute_resources={node_a: [dwagon_gpu], node_b: [fwuff_gpu]},
        )


def test_direct_create_validation_rechecks_connectivity_and_cuda_backends() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [dwagon_gpu],
        node_b: [fwuff_gpu],
    }
    placements = place_instance(
        PlaceInstance(
            model_card=_model_card(hidden_size=32, num_key_value_heads=2),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(
                dwagon_gpu.resource_id,
                fwuff_gpu.resource_id,
            ),
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    disconnected_topology = Topology()
    disconnected_topology.add_node(node_a)
    disconnected_topology.add_node(node_b)

    with pytest.raises(ValueError, match="requires live topology and node network"):
        validate_instance_compute_resources(
            instance,
            resources,
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        )
    with pytest.raises(ValueError, match="do not share a connectivity cycle"):
        validate_instance_compute_resources(
            instance,
            resources,
            topology=disconnected_topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_network=node_network,
        )
    with pytest.raises(ValueError, match="do not advertise MlxCuda"):
        validate_instance_compute_resources(
            instance,
            resources,
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxMetal]},
            node_network=node_network,
        )
    with pytest.raises(ValueError, match="placement estimate requires"):
        validate_instance_compute_resources(
            instance,
            {
                node_a: [_gpu_resource(1, total_memory_gb=1)],
                node_b: [fwuff_gpu],
            },
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_network=node_network,
        )

    assignments = instance.shard_assignments
    runner_by_rank = {
        shard.device_rank: runner_id
        for runner_id, shard in assignments.runner_to_shard.items()
    }
    rank_zero_runner = runner_by_rank[0]
    rank_one_runner = runner_by_rank[1]
    reassigned_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=ShardAssignments(
            model_id=assignments.model_id,
            runner_to_shard=assignments.runner_to_shard,
            node_to_runner={node_a: rank_one_runner, node_b: rank_zero_runner},
            compute_resource_to_runner={
                dwagon_gpu.resource_id: rank_one_runner,
                fwuff_gpu.resource_id: rank_zero_runner,
            },
            compute_resource_to_node=assignments.compute_resource_to_node,
        ),
        nccl_coordinator=instance.nccl_coordinator,
    )
    with pytest.raises(ValueError, match="live rank-zero endpoint"):
        validate_instance_compute_resources(
            reassigned_instance,
            resources,
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_network=node_network,
        )

    tampered_coordinator_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=instance.shard_assignments,
        nccl_coordinator=instance.nccl_coordinator.model_copy(
            update={"ip": "203.0.113.99"}
        ),
    )
    with pytest.raises(ValueError, match="live rank-zero endpoint"):
        validate_instance_compute_resources(
            tampered_coordinator_instance,
            resources,
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_network=node_network,
        )


def test_direct_create_validation_rechecks_tensor_divisibility() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    placements = place_instance(
        PlaceInstance(
            model_card=_model_card(hidden_size=30),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            use_all_compute_resources=True,
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    incompatible_model_card = _model_card(hidden_size=30, num_key_value_heads=2)
    incompatible_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=ShardAssignments(
            model_id=incompatible_model_card.model_id,
            runner_to_shard={
                runner_id: shard.model_copy(
                    update={"model_card": incompatible_model_card}
                )
                for runner_id, shard in instance.shard_assignments.runner_to_shard.items()
            },
            node_to_runner=instance.shard_assignments.node_to_runner,
            compute_resource_to_runner=(
                instance.shard_assignments.compute_resource_to_runner
            ),
            compute_resource_to_node=instance.shard_assignments.compute_resource_to_node,
        ),
        nccl_coordinator=instance.nccl_coordinator,
    )

    with pytest.raises(ValueError, match="incompatible with model dimensions"):
        validate_instance_compute_resources(
            incompatible_instance,
            resources,
            topology=topology,
            node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_network=node_network,
        )


def test_direct_create_validation_handles_malformed_resource_maps() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [dwagon_gpu],
        node_b: [fwuff_gpu],
    }
    placements = place_instance(
        PlaceInstance(
            model_card=_model_card(hidden_size=32, num_key_value_heads=2),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=(
                dwagon_gpu.resource_id,
                fwuff_gpu.resource_id,
            ),
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assignments = instance.shard_assignments
    incomplete_owners = dict(assignments.compute_resource_to_node)
    _ = incomplete_owners.pop(dwagon_gpu.resource_id)
    malformed_assignments = ShardAssignments.model_construct(
        model_id=assignments.model_id,
        runner_to_shard=assignments.runner_to_shard,
        node_to_runner=assignments.node_to_runner,
        compute_resource_to_runner=assignments.compute_resource_to_runner,
        compute_resource_to_node=incomplete_owners,
    )
    malformed_instance = MlxNcclInstance.model_construct(
        instance_id=instance.instance_id,
        shard_assignments=malformed_assignments,
        nccl_coordinator=instance.nccl_coordinator,
    )

    with pytest.raises(ValueError, match="ownership for exactly"):
        validate_instance_compute_resources(malformed_instance, resources)

    unknown_runner_assignments = dict(assignments.compute_resource_to_runner)
    unknown_runner_assignments[dwagon_gpu.resource_id] = RunnerId("unknown-runner")
    malformed_assignments = ShardAssignments.model_construct(
        model_id=assignments.model_id,
        runner_to_shard=assignments.runner_to_shard,
        node_to_runner=assignments.node_to_runner,
        compute_resource_to_runner=unknown_runner_assignments,
        compute_resource_to_node=assignments.compute_resource_to_node,
    )
    malformed_instance = MlxNcclInstance.model_construct(
        instance_id=instance.instance_id,
        shard_assignments=malformed_assignments,
        nccl_coordinator=instance.nccl_coordinator,
    )

    with pytest.raises(ValueError, match="references unknown runner"):
        validate_instance_compute_resources(malformed_instance, resources)


def test_resource_bound_nccl_instance_rejects_partial_resource_map() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-resource-instance"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        use_all_compute_resources=True,
    )
    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    partial_resource_map = dict(
        list(instance.shard_assignments.compute_resource_to_runner.items())[:-1]
    )

    with pytest.raises(ValueError, match="exactly one compute resource per rank"):
        MlxNcclInstance(
            instance_id=instance.instance_id,
            shard_assignments=ShardAssignments(
                model_id=instance.shard_assignments.model_id,
                runner_to_shard=instance.shard_assignments.runner_to_shard,
                node_to_runner=instance.shard_assignments.node_to_runner,
                compute_resource_to_runner=partial_resource_map,
            ),
            nccl_coordinator=instance.nccl_coordinator,
        )


def test_resource_bound_nccl_instance_requires_two_physical_nodes() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    command = PlaceInstance(
        command_id=CommandId("place-resource-instance"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        use_all_compute_resources=True,
    )
    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)

    with pytest.raises(ValueError, match="at least two nodes"):
        MlxNcclInstance(
            instance_id=instance.instance_id,
            shard_assignments=ShardAssignments(
                model_id=instance.shard_assignments.model_id,
                runner_to_shard=instance.shard_assignments.runner_to_shard,
                node_to_runner={
                    node_a: instance.shard_assignments.node_to_runner[node_a]
                },
                compute_resource_to_runner=(
                    instance.shard_assignments.compute_resource_to_runner
                ),
            ),
            nccl_coordinator=instance.nccl_coordinator,
        )


def test_resource_bound_nccl_placement_requires_inventory_for_every_node() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    command = PlaceInstance(
        command_id=CommandId("place-resource-instance"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )

    with pytest.raises(ValueError, match="every node advertises"):
        place_instance(
            command,
            topology,
            {},
            {
                node_a: create_node_memory(1024**3),
                node_b: create_node_memory(1024**3),
            },
            node_network,
            {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
            node_compute_resources={node_a: [_gpu_resource(1)]},
        )


def test_empty_inventory_preserves_legacy_nccl_placement() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    command = PlaceInstance(
        command_id=CommandId("place-legacy-instance"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources={},
    )

    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    assert len(instance.shard_assignments.runner_to_shard) == 2
    assert instance.shard_assignments.compute_resource_to_runner == {}


def test_master_rejects_new_legacy_nccl_instance_with_live_gpu_inventory() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    command = PlaceInstance(
        command_id=CommandId("place-legacy-for-create"),
        model_card=_model_card(),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
    )
    legacy_placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources={},
    )
    legacy_instance = next(iter(legacy_placements.values()))

    with pytest.raises(ValueError, match="explicit compute resource bindings"):
        add_instance_to_placements(
            CreateInstance(instance=legacy_instance),
            topology,
            {},
            {
                node_a: [_gpu_resource(1)],
                node_b: [_gpu_resource(2)],
            },
        )


def test_live_inventory_rejects_contradictory_compute_resource_owner() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology(node_a, node_b)
    resources: dict[NodeId, Sequence[ComputeResource]] = {
        node_a: [_gpu_resource(1), _gpu_resource(2)],
        node_b: [_gpu_resource(3)],
    }
    placements = place_instance(
        PlaceInstance(
            command_id=CommandId("place-owner-validation-instance"),
            model_card=_model_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            use_all_compute_resources=True,
        ),
        topology,
        {},
        {
            node_a: create_node_memory(1024**3),
            node_b: create_node_memory(1024**3),
        },
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    resource_owners = dict(instance.shard_assignments.compute_resource_to_node)
    resource_owners[_gpu_resource(2).resource_id] = node_b
    malformed_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=ShardAssignments(
            model_id=instance.shard_assignments.model_id,
            runner_to_shard=instance.shard_assignments.runner_to_shard,
            node_to_runner=instance.shard_assignments.node_to_runner,
            compute_resource_to_runner=(
                instance.shard_assignments.compute_resource_to_runner
            ),
            compute_resource_to_node=resource_owners,
        ),
        nccl_coordinator=instance.nccl_coordinator,
    )

    with pytest.raises(ValueError, match="but advertised by"):
        validate_instance_compute_resources(malformed_instance, resources)


def test_legacy_tensor_assignments_remain_one_runner_per_node() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, _ = _two_node_topology(node_a, node_b)
    cycle = next(cycle for cycle in topology.get_cycles() if len(cycle) == 2)

    assignments = get_shard_assignments_for_tensor_parallel(_model_card(), cycle)

    assert len(assignments.runner_to_shard) == 2
    assert len(assignments.node_to_runner) == 2
    assert assignments.compute_resource_to_runner == {}
    assert assignments.compute_resource_to_node == {}
