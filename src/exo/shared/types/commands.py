from typing import cast

from pydantic import Field, field_validator, model_validator

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.compute_resources import ComputeResourceId
from exo.shared.types.instance_link import InstanceLinkId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    use_all_compute_resources: bool = False
    # Nonempty order is the tensor device-rank order (rank zero first).
    requested_compute_resource_ids: tuple[ComputeResourceId, ...] = ()

    @field_validator("requested_compute_resource_ids", mode="before")
    @classmethod
    def deserialize_requested_compute_resource_ids(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_compute_resource_policy(self) -> "PlaceInstance":
        requested_resource_ids = self.requested_compute_resource_ids
        if len(requested_resource_ids) != len(set(requested_resource_ids)):
            raise ValueError("Requested compute resource IDs must be unique")
        if not requested_resource_ids:
            return self
        if self.use_all_compute_resources:
            raise ValueError(
                "Explicit compute resource selection is incompatible with "
                "use_all_compute_resources"
            )
        if (
            self.instance_meta != InstanceMeta.MlxNccl
            or self.sharding != Sharding.Tensor
        ):
            raise ValueError(
                "Explicit compute resource selection currently requires MlxNccl "
                "with Tensor sharding"
            )
        for resource_id in requested_resource_ids:
            _ = resource_id.nvidia_device_uuid()
        return self


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


class SetInstanceLink(BaseCommand):
    link_id: InstanceLinkId
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class DeleteInstanceLink(BaseCommand):
    link_id: InstanceLinkId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
    | SetInstanceLink
    | DeleteInstanceLink
)


class ForwarderCommand(FrozenModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    origin: SystemId
    command: DownloadCommand
