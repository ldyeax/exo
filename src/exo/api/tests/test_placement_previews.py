from unittest.mock import AsyncMock, patch

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import InstanceMeta, MlxNcclInstance
from exo.shared.types.worker.shards import Sharding


def _memory() -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=10_000_000,
        ram_available=10_000_000,
        swap_total=0,
        swap_available=0,
    )


async def test_previews_include_nccl_tensor_and_pipeline_error() -> None:
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")
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
        node_network={
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
        },
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
    assert isinstance(tensor_preview.instance, MlxNcclInstance)
    assert len(tensor_preview.instance.shard_assignments.node_to_runner) == 2
