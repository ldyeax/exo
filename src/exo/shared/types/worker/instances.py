from collections.abc import Mapping, Sequence
from enum import Enum
from ipaddress import IPv4Address, ip_address

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.compute_resources import ComputeResource, ComputeResourceId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    MlxNccl = "MlxNccl"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


class MlxNcclInstance(BaseInstance):
    nccl_coordinator: Host

    @model_validator(mode="after")
    def validate_tensor_shards(self) -> "MlxNcclInstance":
        shards = self.shard_assignments.runner_to_shard
        runner_ids = set(shards)
        assigned_runner_ids = set(self.shard_assignments.node_to_runner.values())
        resource_assignments = self.shard_assignments.compute_resource_to_runner
        resource_owners = self.shard_assignments.compute_resource_to_node
        world_size = len(shards)

        try:
            coordinator_ip = ip_address(self.nccl_coordinator.ip)
        except ValueError as error:
            raise ValueError(
                "MlxNcclInstance coordinator must be a concrete IPv4 address"
            ) from error
        if (
            not isinstance(coordinator_ip, IPv4Address)
            or coordinator_ip.is_unspecified
            or coordinator_ip.is_multicast
        ):
            raise ValueError(
                "MlxNcclInstance coordinator must be a concrete IPv4 address"
            )
        if self.nccl_coordinator.port == 0:
            raise ValueError("MlxNcclInstance coordinator port must be nonzero")

        if world_size < 2:
            raise ValueError("MlxNcclInstance requires at least two ranks")
        if resource_assignments:
            resource_runner_ids = list(resource_assignments.values())
            if (
                len(resource_assignments) != world_size
                or len(set(resource_runner_ids)) != world_size
                or set(resource_runner_ids) != runner_ids
            ):
                raise ValueError(
                    "MlxNcclInstance requires exactly one compute resource per rank"
                )
            if len(self.shard_assignments.node_to_runner) < 2:
                raise ValueError("MlxNcclInstance requires at least two nodes")
            if len(assigned_runner_ids) != len(self.shard_assignments.node_to_runner):
                raise ValueError(
                    "MlxNcclInstance node representatives must be unique runners"
                )
            for resource_id in resource_assignments:
                _ = resource_id.nvidia_device_uuid()
            if resource_owners:
                if set(resource_owners.values()) != set(
                    self.shard_assignments.node_to_runner
                ):
                    raise ValueError(
                        "MlxNcclInstance requires resources owned by every node"
                    )
                for (
                    node_id,
                    representative_runner_id,
                ) in self.shard_assignments.node_to_runner.items():
                    representative_resource_ids = [
                        resource_id
                        for resource_id, runner_id in resource_assignments.items()
                        if runner_id == representative_runner_id
                    ]
                    if (
                        len(representative_resource_ids) != 1
                        or resource_owners[representative_resource_ids[0]] != node_id
                    ):
                        raise ValueError(
                            "MlxNcclInstance node representative must use a resource "
                            "owned by that node"
                        )
        elif (
            assigned_runner_ids != runner_ids
            or len(self.shard_assignments.node_to_runner) != world_size
        ):
            raise ValueError("MlxNcclInstance requires exactly one runner per node")
        if not all(isinstance(shard, TensorShardMetadata) for shard in shards.values()):
            raise ValueError("MlxNcclInstance requires TensorShardMetadata")
        if not all(shard.model_card.supports_tensor for shard in shards.values()):
            raise ValueError("MlxNcclInstance requires tensor-capable models")
        if not all(
            shard.start_layer == 0 and shard.end_layer == shard.n_layers
            for shard in shards.values()
        ):
            raise ValueError("MlxNcclInstance tensor shards must contain all layers")
        if {shard.world_size for shard in shards.values()} != {world_size}:
            raise ValueError("MlxNcclInstance shard world sizes are inconsistent")
        if {shard.device_rank for shard in shards.values()} != set(range(world_size)):
            raise ValueError("MlxNcclInstance shard ranks must be contiguous")
        return self


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | MlxNcclInstance


def instance_compute_resource_runners(
    instance: Instance,
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]],
) -> dict[ComputeResourceId, RunnerId]:
    """Return explicit or conservatively inferred GPU bindings for an instance."""
    assignments = instance.shard_assignments
    if assignments.compute_resource_to_runner:
        return dict(assignments.compute_resource_to_runner)
    if not isinstance(instance, MlxRingInstance | MlxNcclInstance):
        return {}
    return {
        resource.resource_id: assignments.node_to_runner[node_id]
        for node_id in assignments.node_to_runner
        for resource in node_compute_resources.get(node_id, ())
    }


class BoundInstance(FrozenModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def bound_compute_resource_ids(self) -> tuple[ComputeResourceId, ...]:
        return tuple(
            resource_id
            for resource_id, runner_id in self.instance.shard_assignments.compute_resource_to_runner.items()
            if runner_id == self.bound_runner_id
        )

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        resource_owners = self.instance.shard_assignments.compute_resource_to_node
        for resource_id in self.bound_compute_resource_ids:
            owner_node_id = resource_owners.get(resource_id)
            if owner_node_id is not None and owner_node_id != self.bound_node_id:
                raise ValueError(
                    f"Compute resource {resource_id} is owned by {owner_node_id}, not "
                    f"bound node {self.bound_node_id}"
                )
        return self
