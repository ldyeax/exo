from collections.abc import Sequence

import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import get_shard_assignments_for_tensor_parallel
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.compute_resources import (
    ComputeResource,
    NvidiaGpuComputeResource,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta, MlxNcclInstance
from exo.shared.types.worker.runners import ShardAssignments
from exo.shared.types.worker.shards import Sharding


def _gpu_resource(index: int) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=f"GPU-00000000-0000-0000-0000-{index:012d}",
        pci_bus_id=f"00000000:{index + 32:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
    )


def _model_card(*, hidden_size: int = 30) -> ModelCard:
    return ModelCard(
        model_id=ModelId("resource-placement-model"),
        storage_size=Memory.from_mb(1),
        n_layers=2,
        hidden_size=hidden_size,
        supports_tensor=True,
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


def test_legacy_tensor_assignments_remain_one_runner_per_node() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, _ = _two_node_topology(node_a, node_b)
    cycle = next(cycle for cycle in topology.get_cycles() if len(cycle) == 2)

    assignments = get_shard_assignments_for_tensor_parallel(_model_card(), cycle)

    assert len(assignments.runner_to_shard) == 2
    assert len(assignments.node_to_runner) == 2
    assert assignments.compute_resource_to_runner == {}
