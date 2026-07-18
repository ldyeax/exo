import os
from pathlib import Path

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, NodeId
from exo.shared.types.compute_resources import ComputeResourceId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxNcclInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.worker.runner.bootstrap import configure_runner_environment


def _bound_nccl_instance(*, resource_bound: bool = False) -> BoundInstance:
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
    instance = MlxNcclInstance(
        instance_id=InstanceId("instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
            compute_resource_to_runner=(
                {
                    ComputeResourceId.from_nvidia_device_uuid(
                        f"GPU-00000000-0000-0000-0000-00000000000{rank + 1}"
                    ): runner_id
                    for rank, runner_id in enumerate(runner_ids)
                }
                if resource_bound
                else {}
            ),
        ),
        nccl_coordinator=Host(ip="192.0.2.10", port=5000),
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[0],
        bound_node_id=node_ids[0],
    )


def _add_mlx4_infiniband_device(infiniband_devices_path: Path) -> None:
    driver_path = infiniband_devices_path / "mlx4_0" / "device" / "driver"
    driver_path.mkdir(parents=True)
    (driver_path / "module").symlink_to("/sys/module/mlx4_core")


def test_nccl_runner_defaults_to_first_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    configure_runner_environment(_bound_nccl_instance())

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_nccl_runner_preserves_operator_cuda_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)

    configure_runner_environment(_bound_nccl_instance())

    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu_uuid


def test_nccl_resource_bound_runner_overrides_parent_cuda_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-binding")

    configure_runner_environment(_bound_nccl_instance(resource_bound=True))

    assert (
        os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-00000000-0000-0000-0000-000000000001"
    )


def test_nccl_runner_disables_gin_for_mlx4_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_mlx4_infiniband_device(tmp_path)
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert os.environ["NCCL_GIN_ENABLE"] == "0"
    assert os.environ["NCCL_GIN_TYPE"] == "0"


def test_nccl_runner_preserves_operator_gin_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_mlx4_infiniband_device(tmp_path)
    monkeypatch.setenv("NCCL_GIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_GIN_TYPE", "CUDA")

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert os.environ["NCCL_GIN_ENABLE"] == "1"
    assert os.environ["NCCL_GIN_TYPE"] == "CUDA"
