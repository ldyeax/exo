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
        numa_node=1,
        cpu_affinity=(16, 17, 18, 19),
    )


def test_nvidia_gpu_resource_has_stable_identity_and_roundtrips() -> None:
    resource = _gpu_resource()

    assert resource.resource_id == ComputeResourceId(
        "nvidia-gpu:GPU-00000000-0000-0000-0000-000000000001"
    )
    assert resource.total_memory.in_gb == 24
    assert resource.numa_node == 1
    assert resource.cpu_affinity == (16, 17, 18, 19)
    assert (
        NvidiaGpuComputeResource.model_validate_json(resource.model_dump_json())
        == resource
    )


def test_nvidia_gpu_resource_accepts_legacy_payload_without_locality() -> None:
    resource = NvidiaGpuComputeResource.model_validate_json(
        """{
            "NvidiaGpuComputeResource": {
                "resourceId": "nvidia-gpu:GPU-legacy",
                "deviceUuid": "GPU-legacy",
                "pciBusId": "00000000:27:00.0",
                "modelName": "NVIDIA GeForce RTX 3090",
                "totalMemory": {"inBytes": 25769803776}
            }
        }"""
    )

    assert resource.numa_node is None
    assert resource.cpu_affinity == ()


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
    restored_resource = restored_state.node_compute_resources[node_id][0]
    assert restored_resource.numa_node == 1
    assert restored_resource.cpu_affinity == (16, 17, 18, 19)

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
        compute_resource_to_node={resource_id: NodeId("node-a")},
    )

    assert assignments.compute_resource_to_runner == {resource_id: runner_id}
    assert assignments.compute_resource_to_node == {resource_id: NodeId("node-a")}

    legacy_payload = assignments.model_dump()
    del legacy_payload["compute_resource_to_runner"]
    del legacy_payload["compute_resource_to_node"]
    restored_legacy_assignments = ShardAssignments.model_validate(legacy_payload)
    assert restored_legacy_assignments.compute_resource_to_runner == {}
    assert restored_legacy_assignments.compute_resource_to_node == {}


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


def test_shard_assignments_reject_partial_resource_ownership() -> None:
    model_id = ModelId("test-model")
    runner_id = RunnerId("runner-a")
    with pytest.raises(ValueError, match="cover exactly the bound resources"):
        ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                runner_id: get_pipeline_shard_metadata(model_id, device_rank=0)
            },
            node_to_runner={NodeId("node-a"): runner_id},
            compute_resource_to_runner={_gpu_resource().resource_id: runner_id},
            compute_resource_to_node={
                ComputeResourceId.from_nvidia_device_uuid(
                    "GPU-00000000-0000-0000-0000-000000000002"
                ): NodeId("node-a")
            },
        )


def test_shard_assignments_reject_unknown_resource_owner() -> None:
    model_id = ModelId("test-model")
    runner_id = RunnerId("runner-a")
    resource_id = _gpu_resource().resource_id
    with pytest.raises(ValueError, match="owned by unknown node"):
        ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                runner_id: get_pipeline_shard_metadata(model_id, device_rank=0)
            },
            node_to_runner={NodeId("node-a"): runner_id},
            compute_resource_to_runner={resource_id: runner_id},
            compute_resource_to_node={resource_id: NodeId("node-b")},
        )


def test_shard_assignments_reject_shard_model_id_mismatch() -> None:
    shard = get_pipeline_shard_metadata(ModelId("shard-model"), device_rank=0)

    with pytest.raises(ValueError, match="does not match assignment model"):
        ShardAssignments(
            model_id=ModelId("assignment-model"),
            runner_to_shard={RunnerId("runner-a"): shard},
            node_to_runner={NodeId("node-a"): RunnerId("runner-a")},
        )


@pytest.mark.parametrize(
    "model_card_update",
    [
        {"revision": "b" * 40},
        {"quantization": "different-quantization"},
    ],
)
def test_shard_assignments_require_identical_full_model_cards(
    model_card_update: dict[str, str],
) -> None:
    first_shard = get_pipeline_shard_metadata(
        ModelId("same-model"), device_rank=0, world_size=2
    )
    second_shard = get_pipeline_shard_metadata(
        ModelId("same-model"), device_rank=1, world_size=2
    ).model_copy(
        update={
            "model_card": first_shard.model_card.model_copy(update=model_card_update)
        }
    )

    with pytest.raises(ValueError, match="same complete model card and revision"):
        ShardAssignments(
            model_id=ModelId("same-model"),
            runner_to_shard={
                RunnerId("runner-a"): first_shard,
                RunnerId("runner-b"): second_shard,
            },
            node_to_runner={
                NodeId("node-a"): RunnerId("runner-a"),
                NodeId("node-b"): RunnerId("runner-b"),
            },
        )
