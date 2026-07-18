import os
from typing import cast

import mlx.core as mx
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxNcclInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.worker.engines.mlx.utils_mlx import (
    EXO_MLX_DISTRIBUTED_BACKEND,
    _distributed_control_stream,  # pyright: ignore[reportPrivateUsage]
    mlx_distributed_init,
)


def _bound_nccl_instance(rank: int) -> BoundInstance:
    node_ids = [NodeId("node-a"), NodeId("node-b")]
    runner_ids = [RunnerId("runner-a"), RunnerId("runner-b")]
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
        runner_id: TensorShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=2,
            start_layer=0,
            end_layer=2,
            n_layers=2,
        )
        for device_rank, runner_id in enumerate(runner_ids)
    }
    instance = MlxNcclInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
        ),
        nccl_coordinator=Host(ip="192.0.2.10", port=5000),
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[rank],
        bound_node_id=node_ids[rank],
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_mlx_nccl_init_sets_required_environment(
    rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    expected_group = cast(mx.distributed.Group, object())

    def fake_init(*, backend: str, strict: bool) -> mx.distributed.Group:
        calls.append((backend, strict))
        return expected_group

    monkeypatch.setattr(mx.distributed, "init", fake_init)
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx4_0:1,mlx4_0:2")
    monkeypatch.setenv(EXO_MLX_DISTRIBUTED_BACKEND, "unset")
    for variable in ("MLX_RANK", "MLX_WORLD_SIZE", "NCCL_HOST_IP", "NCCL_PORT"):
        monkeypatch.delenv(variable, raising=False)

    group = mlx_distributed_init(_bound_nccl_instance(rank))

    assert group is expected_group
    assert calls == [("nccl", True)]
    assert os.environ["MLX_RANK"] == str(rank)
    assert os.environ["MLX_WORLD_SIZE"] == "2"
    assert os.environ["NCCL_HOST_IP"] == "192.0.2.10"
    assert os.environ["NCCL_PORT"] == "5000"
    assert os.environ["NCCL_IB_HCA"] == "=mlx4_0:1,mlx4_0:2"
    assert os.environ[EXO_MLX_DISTRIBUTED_BACKEND] == "nccl"


def test_nccl_control_collectives_use_default_gpu_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EXO_MLX_DISTRIBUTED_BACKEND, "nccl")

    assert _distributed_control_stream() is None
