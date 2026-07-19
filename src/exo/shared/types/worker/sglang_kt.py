from ipaddress import IPv4Address, ip_address
from typing import Annotated, Literal, final

from pydantic import Field, PositiveInt, StringConstraints, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.utils.pydantic_ext import FrozenModel

GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
AbsoluteRuntimePath = Annotated[str, StringConstraints(min_length=2, pattern=r"^/")]
GpuUuid = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        )
    ),
]
ResourceIndex = Annotated[int, Field(ge=0)]
NetworkPort = Annotated[int, Field(ge=1, le=65535)]
StaticMemoryFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
HcaDevice = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9_.-]+:[0-9]+$"),
]

# KTransformers 0.6.3 KT-Kernel methods. New runtime methods require an explicit
# schema update so a persisted launch plan cannot silently change meaning.
KTransformersMethod = Literal[
    "AMXINT4",
    "AMXINT8",
    "BF16",
    "FP8",
    "FP8_PERCHANNEL",
    "LLAMAFILE",
    "MOE_INT4",
    "MOE_INT8",
    "MXFP4",
    "RAWINT4",
]
SglangKtTargetProfile = Literal[
    "glm52_fp8_pp3_sm86_v1",
    "glm47_flash_bf16_sm86_smoke_v1",
    "glm47_flash_bf16_sm86_cpu_routed_experts_control_v1",
    "glm47_flash_bf16_sm86_serving_baseline_v1",
]


def _validate_concrete_ipv4_endpoint(endpoint_name: str, endpoint: Host) -> None:
    try:
        endpoint_ip = ip_address(endpoint.ip)
    except ValueError as error:
        raise ValueError(f"{endpoint_name} must use a concrete IPv4 address") from error
    if (
        not isinstance(endpoint_ip, IPv4Address)
        or endpoint_ip.is_unspecified
        or endpoint_ip.is_multicast
        or endpoint.port == 0
    ):
        raise ValueError(
            f"{endpoint_name} must use a concrete IPv4 address and nonzero port"
        )


@final
class SglangKtStageSpec(FrozenModel):
    """Validated inputs for one logical SGLang pipeline stage.

    This is a launch-plan contract, not an active Exo Instance. Paths are local to
    the assigned node and are intentionally not checked on the planning host.
    """

    pipeline_rank: ResourceIndex
    start_layer: ResourceIndex
    end_layer: PositiveInt
    node_id: NodeId
    gpu_uuid: GpuUuid
    service_endpoint: Host
    model_path: AbsoluteRuntimePath
    ktransformers_weight_path: AbsoluteRuntimePath
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    cpu_infer_threads: PositiveInt
    threadpool_count: PositiveInt
    ktransformers_method: KTransformersMethod
    resident_gpu_experts: ResourceIndex = 0
    max_deferred_experts_per_token: ResourceIndex = 0
    hca_devices: tuple[HcaDevice, ...] = ()

    @model_validator(mode="after")
    def validate_resources(self) -> "SglangKtStageSpec":
        _validate_concrete_ipv4_endpoint(
            "stage service_endpoint", self.service_endpoint
        )
        if self.end_layer <= self.start_layer:
            raise ValueError("stage end_layer must be greater than start_layer")
        if not self.cpu_cores or len(set(self.cpu_cores)) != len(self.cpu_cores):
            raise ValueError("stage cpu_cores must be nonempty and unique")
        if not self.memory_nodes or len(set(self.memory_nodes)) != len(
            self.memory_nodes
        ):
            raise ValueError("stage memory_nodes must be nonempty and unique")
        if self.cpu_infer_threads > len(self.cpu_cores):
            raise ValueError("stage cpu_infer_threads exceeds assigned cpu_cores")
        # SGLang-KT documents --kt-threadpool-count as one-to-one with the
        # selected NUMA nodes.
        if self.threadpool_count != len(self.memory_nodes):
            raise ValueError("stage threadpool_count must equal assigned memory_nodes")
        if len(set(self.hca_devices)) != len(self.hca_devices):
            raise ValueError("stage hca_devices must be unique")
        return self


@final
class SglangKtLaunchPlan(FrozenModel):
    """Pinned, immutable payload for a future ``SglangKtInstance``."""

    model_id: ModelId
    model_revision: GitRevision
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    target_profile: SglangKtTargetProfile
    total_layers: PositiveInt
    context_length: PositiveInt
    max_total_tokens: PositiveInt
    static_memory_fraction: StaticMemoryFraction
    max_concurrent_requests: PositiveInt
    distributed_coordinator: Host
    rank_zero_endpoint: Host
    stages: tuple[SglangKtStageSpec, ...]

    @property
    def pipeline_layer_partition(self) -> tuple[int, ...]:
        return tuple(stage.end_layer - stage.start_layer for stage in self.stages)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "SglangKtLaunchPlan":
        if not self.stages:
            raise ValueError("SGLang/KTransformers launch plan requires stages")

        expected_start_layer = 0
        seen_gpu_uuids: set[str] = set()
        seen_service_endpoints: set[tuple[str, int]] = set()
        cpu_cores_by_node: dict[NodeId, set[int]] = {}

        for expected_rank, stage in enumerate(self.stages):
            if stage.pipeline_rank != expected_rank:
                raise ValueError("pipeline ranks must be ordered and contiguous")
            if stage.start_layer != expected_start_layer:
                raise ValueError("pipeline layer ranges must be ordered and contiguous")
            if stage.gpu_uuid in seen_gpu_uuids:
                raise ValueError("pipeline stages must use distinct GPUs")

            service_endpoint = (
                stage.service_endpoint.ip,
                stage.service_endpoint.port,
            )
            if service_endpoint in seen_service_endpoints:
                raise ValueError("pipeline stages must use distinct service endpoints")

            assigned_cpu_cores = cpu_cores_by_node.setdefault(stage.node_id, set())
            overlapping_cpu_cores = assigned_cpu_cores.intersection(stage.cpu_cores)
            if overlapping_cpu_cores:
                raise ValueError(
                    "pipeline stages on the same node must use disjoint cpu_cores"
                )
            assigned_cpu_cores.update(stage.cpu_cores)

            seen_gpu_uuids.add(stage.gpu_uuid)
            seen_service_endpoints.add(service_endpoint)
            expected_start_layer = stage.end_layer

        if expected_start_layer != self.total_layers:
            raise ValueError("pipeline layer ranges must cover total_layers exactly")
        if self.max_total_tokens > self.context_length:
            raise ValueError("max_total_tokens cannot exceed context_length")

        for endpoint_name, endpoint in (
            ("distributed_coordinator", self.distributed_coordinator),
            ("rank_zero_endpoint", self.rank_zero_endpoint),
        ):
            _validate_concrete_ipv4_endpoint(endpoint_name, endpoint)

        if self.stages[0].service_endpoint != self.rank_zero_endpoint:
            raise ValueError(
                "rank_zero_endpoint must equal the pipeline rank zero service_endpoint"
            )
        if self.distributed_coordinator in (
            stage.service_endpoint for stage in self.stages
        ):
            raise ValueError(
                "distributed_coordinator must differ from every stage service_endpoint"
            )

        return self
