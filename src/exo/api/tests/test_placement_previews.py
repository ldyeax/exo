from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types.api import CreateInstanceParams, PlaceInstanceParams
from exo.master.placement import place_instance
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.compute_resources import (
    ComputeResourceId,
    NvidiaGpuComputeResource,
)
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    MlxNcclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


def _memory() -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=10_000_000,
        ram_available=10_000_000,
        swap_total=0,
        swap_available=0,
    )


def _gpu_resource(index: int) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=f"GPU-00000000-0000-0000-0000-{index:012d}",
        pci_bus_id=f"00000000:{index + 32:02x}:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
    )


def _two_node_topology_and_network(
    node_a: NodeId, node_b: NodeId
) -> tuple[Topology, dict[NodeId, NodeNetworkInfo]]:
    topology = Topology()
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/169.254.0.2/tcp/1234")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=node_b,
            sink=node_a,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/169.254.0.1/tcp/1234")
            ),
        )
    )
    return topology, {
        node_a: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address="169.254.0.1",
                    interface_type="ethernet",
                )
            ]
        ),
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address="169.254.0.2",
                    interface_type="ethernet",
                )
            ]
        ),
    }


async def test_previews_include_nccl_tensor_and_pipeline_error() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1_000),
        n_layers=4,
        hidden_size=32,
        supports_tensor=True,
        num_key_value_heads=2,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    state = State(
        topology=topology,
        node_memory={node_a: _memory(), node_b: _memory()},
        node_network=node_network,
        node_backends={
            node_a: [Backend.MlxCuda],
            node_b: [Backend.MlxCuda],
        },
    )
    api = object.__new__(API)
    api.state = state

    with patch.object(ModelCard, "load", AsyncMock(return_value=model_card)):
        result = await api.get_placement_previews(model_card.model_id)

    nccl_previews = [
        preview
        for preview in result.previews
        if preview.instance_meta == InstanceMeta.MlxNccl
    ]
    assert len(nccl_previews) == 2

    pipeline_preview = next(
        preview for preview in nccl_previews if preview.sharding == Sharding.Pipeline
    )
    assert pipeline_preview.instance is None
    assert pipeline_preview.error is not None
    assert "requires Tensor sharding" in pipeline_preview.error

    tensor_preview = next(
        preview for preview in nccl_previews if preview.sharding == Sharding.Tensor
    )
    assert tensor_preview.error is None
    assert tensor_preview.use_all_compute_resources is False
    assert isinstance(tensor_preview.instance, MlxNcclInstance)
    assert len(tensor_preview.instance.shard_assignments.node_to_runner) == 2


async def test_all_resource_preview_reports_policy_and_rank_weighted_memory() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)
    model_card = ModelCard(
        model_id=ModelId("three-rank-preview-model"),
        storage_size=Memory.from_bytes(9),
        n_layers=2,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    state = State(
        topology=topology,
        node_memory={node_a: _memory(), node_b: _memory()},
        node_network=node_network,
        node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources={
            node_a: [_gpu_resource(1), _gpu_resource(2)],
            node_b: [_gpu_resource(3)],
        },
    )
    api = object.__new__(API)
    api.state = state

    with patch.object(ModelCard, "load", AsyncMock(return_value=model_card)):
        result = await api.get_placement_previews(
            model_card.model_id,
            use_all_compute_resources=True,
        )

    tensor_preview = next(
        preview
        for preview in result.previews
        if preview.instance_meta == InstanceMeta.MlxNccl
        and preview.sharding == Sharding.Tensor
        and preview.error is None
    )
    assert tensor_preview.use_all_compute_resources is True
    assert isinstance(tensor_preview.instance, MlxNcclInstance)
    assert (
        len(tensor_preview.instance.shard_assignments.compute_resource_to_runner) == 3
    )
    assert tensor_preview.memory_delta_by_node == {
        str(node_a): 6,
        str(node_b): 3,
    }


async def test_explicit_resource_policy_propagates_and_preview_is_deduplicated() -> (
    None
):
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)
    first_dwagon_gpu = _gpu_resource(1)
    requested_dwagon_gpu = _gpu_resource(2)
    fwuff_gpu = _gpu_resource(3)
    requested_resource_ids = (
        requested_dwagon_gpu.resource_id,
        fwuff_gpu.resource_id,
    )
    model_card = ModelCard(
        model_id=ModelId("explicit-preview-model"),
        storage_size=Memory.from_bytes(9),
        n_layers=2,
        hidden_size=32,
        supports_tensor=True,
        num_key_value_heads=2,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory={node_a: _memory(), node_b: _memory()},
        node_network=node_network,
        node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources={
            node_a: [first_dwagon_gpu, requested_dwagon_gpu],
            node_b: [fwuff_gpu],
        },
    )
    payload = PlaceInstanceParams(
        model_id=model_card.model_id,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxNccl,
        min_nodes=2,
        requested_compute_resource_ids=requested_resource_ids,
    )
    sent_commands: list[PlaceInstance] = []

    async def capture_command(command: PlaceInstance) -> None:
        sent_commands.append(command)

    with (
        patch.object(ModelCard, "load", AsyncMock(return_value=model_card)),
        patch.object(api, "_send", capture_command),
    ):
        await api.place_instance(payload)
        result = await api.get_placement_previews(
            model_card.model_id,
            requested_compute_resource_ids=list(requested_resource_ids),
        )
        placement = await api.get_placement(
            model_card.model_id,
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
            requested_compute_resource_ids=list(requested_resource_ids),
        )

    assert len(sent_commands) == 1
    sent_command = sent_commands[0]
    assert isinstance(sent_command, PlaceInstance)
    assert sent_command.requested_compute_resource_ids == requested_resource_ids
    successful_previews = [
        preview
        for preview in result.previews
        if preview.error is None and preview.instance is not None
    ]
    assert len(successful_previews) == 1
    preview = successful_previews[0]
    assert preview.requested_compute_resource_ids == requested_resource_ids
    assert isinstance(preview.instance, MlxNcclInstance)
    assignments = preview.instance.shard_assignments
    assert tuple(assignments.compute_resource_to_runner) == requested_resource_ids
    assert first_dwagon_gpu.resource_id not in assignments.compute_resource_to_runner
    assert [
        assignments.runner_to_shard[
            assignments.compute_resource_to_runner[resource_id]
        ].device_rank
        for resource_id in requested_resource_ids
    ] == [0, 1]
    assert all(
        preview.requested_compute_resource_ids == requested_resource_ids
        for preview in result.previews
    )
    assert isinstance(placement, MlxNcclInstance)
    assert tuple(placement.shard_assignments.compute_resource_to_runner) == (
        requested_resource_ids
    )


def test_place_instance_params_rejects_invalid_explicit_resource_policy() -> None:
    resource_id = _gpu_resource(1).resource_id

    with pytest.raises(ValueError, match="must be unique"):
        PlaceInstanceParams(
            model_id=ModelId("duplicate-resource-model"),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            requested_compute_resource_ids=(resource_id, resource_id),
        )
    with pytest.raises(ValueError, match="incompatible"):
        PlaceInstanceParams(
            model_id=ModelId("conflicting-resource-model"),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            use_all_compute_resources=True,
            requested_compute_resource_ids=(resource_id,),
        )
    with pytest.raises(ValueError, match="requires MlxNccl with Tensor"):
        PlaceInstanceParams(
            model_id=ModelId("wrong-engine-resource-model"),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxRing,
            requested_compute_resource_ids=(resource_id,),
        )


async def test_previews_and_direct_create_count_retiring_resources() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)
    model_card = ModelCard(
        model_id=ModelId("retiring-preview-model"),
        storage_size=Memory.from_bytes(9),
        n_layers=2,
        hidden_size=32,
        supports_tensor=True,
        num_key_value_heads=2,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    node_memory = {node_a: _memory(), node_b: _memory()}
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}
    node_compute_resources = {
        node_a: [_gpu_resource(1)],
        node_b: [_gpu_resource(2)],
    }
    ring_placements = place_instance(
        PlaceInstance(
            command_id=CommandId("retiring-preview-ring"),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_compute_resources=node_compute_resources,
    )
    ring_instance = next(iter(ring_placements.values()))
    retiring_compute_resources = {
        resource.resource_id: RunnerId(f"retiring-{node_id}")
        for node_id, resources in node_compute_resources.items()
        for resource in resources
    }
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
        node_compute_resources=node_compute_resources,
        retiring_compute_resources=retiring_compute_resources,
    )

    with patch.object(ModelCard, "load", AsyncMock(return_value=model_card)):
        result = await api.get_placement_previews(model_card.model_id)

    tensor_nccl_preview = next(
        preview
        for preview in result.previews
        if preview.instance_meta == InstanceMeta.MlxNccl
        and preview.sharding == Sharding.Tensor
    )
    assert tensor_nccl_preview.instance is None
    assert tensor_nccl_preview.error is not None
    assert "available NVIDIA GPU" in tensor_nccl_preview.error

    with (
        patch.object(api, "_send", AsyncMock()) as send,
        pytest.raises(HTTPException, match="already occupied"),
    ):
        await api.create_instance(CreateInstanceParams(instance=ring_instance))

    send.assert_not_awaited()


async def test_create_instance_validates_legacy_nccl_with_or_without_inventory() -> (
    None
):
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)
    model_card = ModelCard(
        model_id=ModelId("legacy-create-model"),
        storage_size=Memory.from_bytes(9),
        n_layers=2,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    node_memory = {node_a: _memory(), node_b: _memory()}
    node_backends = {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]}
    legacy_placements = place_instance(
        PlaceInstance(
            command_id=CommandId("legacy-create-placement"),
            model_card=model_card,
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxNccl,
            min_nodes=2,
        ),
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
        node_compute_resources={},
    )
    legacy_instance = next(iter(legacy_placements.values()))
    assert isinstance(legacy_instance, MlxNcclInstance)
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
        node_compute_resources={
            node_a: [_gpu_resource(1)],
            node_b: [_gpu_resource(2)],
        },
    )

    with (
        patch.object(ModelCard, "load", AsyncMock(return_value=model_card)),
        pytest.raises(HTTPException) as raised,
    ):
        await api.create_instance(CreateInstanceParams(instance=legacy_instance))

    assert raised.value.status_code == 400
    assert "explicit compute resource bindings" in str(raised.value.detail)

    api.state = State(
        topology=topology,
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
        node_compute_resources={},
    )
    tampered_instance = MlxNcclInstance(
        instance_id=legacy_instance.instance_id,
        shard_assignments=legacy_instance.shard_assignments,
        nccl_coordinator=legacy_instance.nccl_coordinator.model_copy(
            update={"ip": "203.0.113.99"}
        ),
    )

    with (
        patch.object(api, "_send", AsyncMock()) as send,
        pytest.raises(HTTPException, match="live rank-zero endpoint"),
    ):
        await api.create_instance(CreateInstanceParams(instance=tampered_instance))

    send.assert_not_awaited()


async def test_create_instance_rejects_tampered_resource_or_coordinator() -> None:
    node_a = NodeId("dwagon")
    node_b = NodeId("fwuff")
    topology, node_network = _two_node_topology_and_network(node_a, node_b)
    dwagon_gpu = _gpu_resource(1)
    fwuff_gpu = _gpu_resource(2)
    resources = {node_a: [dwagon_gpu], node_b: [fwuff_gpu]}
    model_card = ModelCard(
        model_id=ModelId("tampered-resource-model"),
        storage_size=Memory.from_bytes(9),
        n_layers=2,
        hidden_size=32,
        supports_tensor=True,
        num_key_value_heads=2,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    placements = place_instance(
        PlaceInstance(
            model_card=model_card,
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
        {node_a: _memory(), node_b: _memory()},
        node_network,
        {node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, MlxNcclInstance)
    unknown_resource_id = ComputeResourceId.from_nvidia_device_uuid("GPU-unknown")
    resource_runners = dict(instance.shard_assignments.compute_resource_to_runner)
    resource_owners = dict(instance.shard_assignments.compute_resource_to_node)
    runner_id = resource_runners.pop(dwagon_gpu.resource_id)
    _ = resource_owners.pop(dwagon_gpu.resource_id)
    resource_runners[unknown_resource_id] = runner_id
    resource_owners[unknown_resource_id] = node_a
    tampered_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=ShardAssignments(
            model_id=instance.shard_assignments.model_id,
            runner_to_shard=instance.shard_assignments.runner_to_shard,
            node_to_runner=instance.shard_assignments.node_to_runner,
            compute_resource_to_runner=resource_runners,
            compute_resource_to_node=resource_owners,
        ),
        nccl_coordinator=instance.nccl_coordinator,
    )
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory={node_a: _memory(), node_b: _memory()},
        node_network=node_network,
        node_backends={node_a: [Backend.MlxCuda], node_b: [Backend.MlxCuda]},
        node_compute_resources=resources,
    )

    with (
        patch.object(api, "_send", AsyncMock()) as send,
        pytest.raises(HTTPException, match="not live"),
    ):
        await api.create_instance(CreateInstanceParams(instance=tampered_instance))

    send.assert_not_awaited()

    tampered_coordinator_instance = MlxNcclInstance(
        instance_id=instance.instance_id,
        shard_assignments=instance.shard_assignments,
        nccl_coordinator=instance.nccl_coordinator.model_copy(
            update={"ip": "203.0.113.99"}
        ),
    )
    with (
        patch.object(api, "_send", AsyncMock()) as send,
        pytest.raises(HTTPException, match="live rank-zero endpoint"),
    ):
        await api.create_instance(
            CreateInstanceParams(instance=tampered_coordinator_instance)
        )

    send.assert_not_awaited()


async def test_create_instance_preserves_embedded_exact_model_revision() -> None:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    model_card = ModelCard(
        model_id=ModelId("pinned-create-model"),
        revision="a" * 40,
        storage_size=Memory.from_bytes(9),
        n_layers=1,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    shard = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("pinned-create-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    api = object.__new__(API)
    api.state = State(node_memory={node_id: _memory()})

    with (
        patch.object(api, "_send", AsyncMock()) as send,
        patch.object(
            ModelCard,
            "load",
            AsyncMock(side_effect=AssertionError("must use embedded model card")),
        ) as load,
    ):
        response = await api.create_instance(CreateInstanceParams(instance=instance))

    load.assert_not_awaited()
    send.assert_awaited_once()
    assert response.model_card == model_card
    assert response.model_card.revision == "a" * 40


async def test_create_instance_rejects_empty_shard_assignments() -> None:
    instance = MlxRingInstance(
        instance_id=InstanceId("empty-create-instance"),
        shard_assignments=ShardAssignments(
            model_id=ModelId("empty-create-model"),
            runner_to_shard={},
            node_to_runner={},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    api = object.__new__(API)
    api.state = State()

    with (
        patch.object(api, "_send", AsyncMock()) as send,
        pytest.raises(HTTPException) as raised,
    ):
        await api.create_instance(CreateInstanceParams(instance=instance))

    send.assert_not_awaited()
    assert raised.value.status_code == 400
    assert "without any shard assignments" in str(raised.value.detail)
