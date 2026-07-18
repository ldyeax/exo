from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Id, NodeId
from exo.shared.types.compute_resources import ComputeResourceId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel
from exo.worker.runner.diagnostics import KnownRunnerDiagnostic


class RunnerId(Id):
    pass


class RunnerError(Exception):
    pass


class BaseRunnerStatus(TaggedModel):
    def is_running(self):
        return isinstance(self, RunnerRunning)


class RunnerIdle(BaseRunnerStatus):
    pass


class RunnerConnecting(BaseRunnerStatus):
    pass


class RunnerConnected(BaseRunnerStatus):
    pass


class RunnerLoading(BaseRunnerStatus):
    layers_loaded: int = 0
    total_layers: int = 0


class RunnerLoaded(BaseRunnerStatus):
    pass


class RunnerWarmingUp(BaseRunnerStatus):
    pass


class RunnerReady(BaseRunnerStatus):
    prefill_server_port: int | None = None


class RunnerRunning(BaseRunnerStatus):
    pass


class RunnerShuttingDown(BaseRunnerStatus):
    pass


class RunnerShutdown(BaseRunnerStatus):
    pass


class RunnerFailed(BaseRunnerStatus):
    error_message: str | None = None
    diagnostics: list[KnownRunnerDiagnostic]


RunnerStatus = (
    RunnerIdle
    | RunnerConnecting
    | RunnerConnected
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerShuttingDown
    | RunnerShutdown
    | RunnerFailed
)


class ShardAssignments(FrozenModel):
    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata]
    node_to_runner: Mapping[NodeId, RunnerId]
    compute_resource_to_runner: Mapping[ComputeResourceId, RunnerId] = {}
    compute_resource_to_node: Mapping[ComputeResourceId, NodeId] = {}

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        for runner_id in self.node_to_runner.values():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                )
        for resource_id, runner_id in self.compute_resource_to_runner.items():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} assigned to compute resource {resource_id} "
                    "does not exist in runner_to_shard"
                )
        if self.compute_resource_to_node and set(self.compute_resource_to_node) != set(
            self.compute_resource_to_runner
        ):
            raise ValueError(
                "Compute resource ownership must cover exactly the bound resources"
            )
        for resource_id, node_id in self.compute_resource_to_node.items():
            if node_id not in self.node_to_runner:
                raise ValueError(
                    f"Compute resource {resource_id} is owned by unknown node {node_id}"
                )

        model_cards = [shard.model_card for shard in self.runner_to_shard.values()]
        for model_card in model_cards:
            if model_card.model_id != self.model_id:
                raise ValueError(
                    f"Shard model card {model_card.model_id} does not match "
                    f"assignment model {self.model_id}"
                )
        if model_cards and any(
            model_card != model_cards[0] for model_card in model_cards[1:]
        ):
            raise ValueError(
                "All runner shards must carry the same complete model card and revision"
            )
        return self
