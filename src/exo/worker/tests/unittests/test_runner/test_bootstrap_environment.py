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
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
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


def _bound_ring_instance(*, resource_count: int = 1) -> BoundInstance:
    node_id = NodeId("node-a")
    runner_id = RunnerId("runner-a")
    resource_ids = tuple(
        ComputeResourceId.from_nvidia_device_uuid(
            f"GPU-00000000-0000-0000-0000-{index:012d}"
        )
        for index in range(1, resource_count + 1)
    )
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("ring-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node_id: runner_id},
            compute_resource_to_runner={
                resource_id: runner_id for resource_id in resource_ids
            },
            compute_resource_to_node={
                resource_id: node_id for resource_id in resource_ids
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


def _add_infiniband_device(
    infiniband_devices_path: Path, device_name: str, driver_module: str
) -> None:
    driver_path = infiniband_devices_path / device_name / "device" / "driver"
    driver_path.mkdir(parents=True)
    (driver_path / "module").symlink_to(f"/sys/module/{driver_module}")


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


def test_non_nccl_resource_bound_runner_overrides_parent_cuda_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-binding")

    configure_runner_environment(_bound_ring_instance())

    assert (
        os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-00000000-0000-0000-0000-000000000001"
    )


def test_non_nccl_unbound_runner_preserves_parent_cuda_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-binding")

    configure_runner_environment(_bound_ring_instance(resource_count=0))

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-parent-binding"


def test_non_nccl_runner_rejects_multiple_compute_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-binding")

    with pytest.raises(ValueError, match="exactly one compute resource"):
        configure_runner_environment(_bound_ring_instance(resource_count=2))

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-parent-binding"


def test_nccl_runner_disables_gin_for_mlx4_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_0", "mlx4_core")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert os.environ["NCCL_GIN_ENABLE"] == "0"
    assert os.environ["NCCL_GIN_TYPE"] == "0"


def test_nccl_runner_keeps_gin_available_when_selector_uses_only_mlx5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_0", "mlx4_core")
    _add_infiniband_device(tmp_path, "mlx5_0", "mlx5_core")
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0:1")
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert "NCCL_GIN_ENABLE" not in os.environ
    assert "NCCL_GIN_TYPE" not in os.environ


def test_nccl_runner_disables_gin_when_selector_can_use_mlx4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_0", "mlx4_core")
    _add_infiniband_device(tmp_path, "mlx5_0", "mlx5_core")
    monkeypatch.setenv("NCCL_IB_HCA", "mlx")
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert os.environ["NCCL_GIN_ENABLE"] == "0"
    assert os.environ["NCCL_GIN_TYPE"] == "0"


def test_nccl_runner_honors_mlx4_exclusion_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_0", "mlx4_core")
    _add_infiniband_device(tmp_path, "mlx5_0", "mlx5_core")
    monkeypatch.setenv("NCCL_IB_HCA", "^=mlx4_0")
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert "NCCL_GIN_ENABLE" not in os.environ
    assert "NCCL_GIN_TYPE" not in os.environ


def test_nccl_runner_applies_exact_match_to_entire_hca_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_10", "mlx4_core")
    _add_infiniband_device(tmp_path, "mlx5_0", "mlx5_core")
    monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_0:1,mlx4_1:1")
    monkeypatch.delenv("NCCL_GIN_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_GIN_TYPE", raising=False)

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert "NCCL_GIN_ENABLE" not in os.environ
    assert "NCCL_GIN_TYPE" not in os.environ


def test_nccl_runner_preserves_operator_gin_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _add_infiniband_device(tmp_path, "mlx4_0", "mlx4_core")
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setenv("NCCL_GIN_ENABLE", "1")
    monkeypatch.setenv("NCCL_GIN_TYPE", "CUDA")

    configure_runner_environment(
        _bound_nccl_instance(), infiniband_devices_path=tmp_path
    )

    assert os.environ["NCCL_GIN_ENABLE"] == "1"
    assert os.environ["NCCL_GIN_TYPE"] == "CUDA"
