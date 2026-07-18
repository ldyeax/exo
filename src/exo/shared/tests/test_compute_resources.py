from datetime import datetime, timezone

import pytest

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.compute_resources import (
    ComputeResourceId,
    NvidiaGpuComputeResource,
)
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.state import State
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.utils.info_gatherer.info_gatherer import NodeComputeResources


def _gpu_resource(
    device_uuid: str = "GPU-00000000-0000-0000-0000-000000000001",
) -> NvidiaGpuComputeResource:
    return NvidiaGpuComputeResource.from_device(
        device_uuid=device_uuid,
        pci_bus_id="00000000:27:00.0",
        model_name="NVIDIA GeForce RTX 3090",
        total_memory_bytes=24 * 1024**3,
    )


def test_nvidia_gpu_resource_has_stable_identity_and_roundtrips() -> None:
    resource = _gpu_resource()

    assert resource.resource_id == ComputeResourceId(
        "nvidia-gpu:GPU-00000000-0000-0000-0000-000000000001"
    )
    assert resource.total_memory.in_gb == 24
    assert (
        NvidiaGpuComputeResource.model_validate_json(resource.model_dump_json())
        == resource
    )


def test_nvidia_gpu_resource_rejects_mismatched_identity() -> None:
    with pytest.raises(ValueError, match="derived from the device UUID"):
        NvidiaGpuComputeResource(
            resource_id=ComputeResourceId("nvidia-gpu:another-device"),
            device_uuid="GPU-00000000-0000-0000-0000-000000000001",
            pci_bus_id="00000000:27:00.0",
            model_name="NVIDIA GeForce RTX 3090",
            total_memory=_gpu_resource().total_memory,
        )


def test_compute_resources_are_applied_serialized_and_removed_on_timeout() -> None:
    node_id = NodeId("node-a")
    resource = _gpu_resource()
    state = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=node_id,
            when=datetime.now(timezone.utc).isoformat(),
            info=NodeComputeResources(resources=[resource]),
        ),
        State(),
    )

    assert state.node_compute_resources[node_id] == [resource]
    restored_state = State.model_validate_json(state.model_dump_json())
    assert restored_state.node_compute_resources[node_id] == [resource]

    timed_out_state = apply_node_timed_out(NodeTimedOut(node_id=node_id), state)
    assert node_id not in timed_out_state.node_compute_resources


def test_shard_assignments_accept_optional_compute_resource_bindings() -> None:
    model_id = ModelId("test-model")
    runner_id = RunnerId("runner-a")
    resource_id = _gpu_resource().resource_id
    assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            runner_id: get_pipeline_shard_metadata(model_id, device_rank=0)
        },
        node_to_runner={NodeId("node-a"): runner_id},
        compute_resource_to_runner={resource_id: runner_id},
    )

    assert assignments.compute_resource_to_runner == {resource_id: runner_id}

    legacy_payload = assignments.model_dump()
    del legacy_payload["compute_resource_to_runner"]
    restored_legacy_assignments = ShardAssignments.model_validate(legacy_payload)
    assert restored_legacy_assignments.compute_resource_to_runner == {}


def test_shard_assignments_reject_unknown_resource_runner() -> None:
    model_id = ModelId("test-model")
    runner_id = RunnerId("runner-a")
    with pytest.raises(ValueError, match="does not exist in runner_to_shard"):
        ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                runner_id: get_pipeline_shard_metadata(model_id, device_rank=0)
            },
            node_to_runner={NodeId("node-a"): runner_id},
            compute_resource_to_runner={
                _gpu_resource().resource_id: RunnerId("unknown-runner")
            },
        )
