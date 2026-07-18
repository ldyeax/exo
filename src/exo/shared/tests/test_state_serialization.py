import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import InstanceId, MlxNcclInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = Connection(
        source=node_a,
        sink=node_b,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/10001"),
        ),
    )

    state = State()
    state.topology.add_connection(connection)

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert (
        state.topology.to_snapshot().nodes
        == restored_state.topology.to_snapshot().nodes
    )
    assert set(state.topology.to_snapshot().connections) == set(
        restored_state.topology.to_snapshot().connections
    )
    assert restored_state.model_dump_json() == json_repr


def test_nccl_instance_state_serialization_roundtrip() -> None:
    node_ids = (NodeId("node-a"), NodeId("node-b"))
    runner_ids = (RunnerId("runner-a"), RunnerId("runner-b"))
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    shards = {
        runner_id: TensorShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=2,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        for rank, runner_id in enumerate(runner_ids)
    }
    instance_id = InstanceId("nccl-instance")
    instance = MlxNcclInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
        ),
        nccl_coordinator=Host(ip="192.0.2.10", port=5000),
    )
    state = State(instances={instance_id: instance})

    restored_state = State.model_validate_json(state.model_dump_json())

    restored_instance = restored_state.instances[instance_id]
    assert isinstance(restored_instance, MlxNcclInstance)
    assert restored_instance == instance


def test_nccl_instance_rejects_pipeline_shards() -> None:
    node_ids = (NodeId("node-a"), NodeId("node-b"))
    runner_ids = (RunnerId("runner-a"), RunnerId("runner-b"))
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(1),
        n_layers=2,
        hidden_size=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    shards = {
        runner_id: PipelineShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=2,
            start_layer=rank,
            end_layer=rank + 1,
            n_layers=2,
        )
        for rank, runner_id in enumerate(runner_ids)
    }

    with pytest.raises(ValueError, match="requires TensorShardMetadata"):
        MlxNcclInstance(
            instance_id=InstanceId("nccl-instance"),
            shard_assignments=ShardAssignments(
                model_id=model_card.model_id,
                runner_to_shard=shards,
                node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
            ),
            nccl_coordinator=Host(ip="192.0.2.10", port=5000),
        )


@pytest.mark.parametrize(
    "coordinator,error_message",
    [
        (Host(ip="0.0.0.0", port=5000), "concrete IPv4 address"),
        (Host(ip="::1", port=5000), "concrete IPv4 address"),
        (Host(ip="not-an-address", port=5000), "concrete IPv4 address"),
        (Host(ip="192.0.2.10", port=0), "port must be nonzero"),
    ],
)
def test_nccl_instance_rejects_invalid_coordinator(
    coordinator: Host, error_message: str
) -> None:
    node_ids = (NodeId("node-a"), NodeId("node-b"))
    runner_ids = (RunnerId("runner-a"), RunnerId("runner-b"))
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    shards = {
        runner_id: TensorShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=2,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        for rank, runner_id in enumerate(runner_ids)
    }

    with pytest.raises(ValueError, match=error_message):
        MlxNcclInstance(
            instance_id=InstanceId("nccl-instance"),
            shard_assignments=ShardAssignments(
                model_id=model_card.model_id,
                runner_to_shard=shards,
                node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
            ),
            nccl_coordinator=coordinator,
        )
