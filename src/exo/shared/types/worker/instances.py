from enum import Enum
from ipaddress import IPv4Address, ip_address

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
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
        if (
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
        return self
