from typing import Final, final

from pydantic import PositiveInt, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    NetworkPort,
    ResourceIndex,
    SglangKtLaunchPlan,
)
from exo.utils.pydantic_ext import FrozenModel

GLM_5_2_FP8_MODEL_ID: Final = ModelId("zai-org/GLM-5.2-FP8")
GLM_5_2_LAYER_COUNT: Final = 78

# KTransformers v0.6.3 is the first release with explicit GLM-5.2 support. Its
# SGLang submodule pins the matching fork revision below.
SUPPORTED_KTRANSFORMERS_REVISION: Final = "ce7c3ddbe93f7ac1f992375eed54058bbc512646"
SUPPORTED_SGLANG_REVISION: Final = "8b636f9008dbad58c0a8e481b03e794739e6c146"
REQUIRED_TRANSFORMERS_VERSION: Final = "5.3.0"

EnvironmentVariable = tuple[str, str]


@final
class SglangKtProcessLaunchSpec(FrozenModel):
    """One inert process-launch description for an external SGLang runtime."""

    pipeline_rank: ResourceIndex
    start_layer: ResourceIndex
    end_layer: PositiveInt
    node_id: NodeId
    gpu_uuid: GpuUuid
    executable: AbsoluteRuntimePath
    arguments: tuple[str, ...]
    environment: tuple[EnvironmentVariable, ...]
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    service_endpoint: Host
    nccl_port: NetworkPort
    model_id: ModelId
    expected_model_revision: GitRevision
    expected_sglang_revision: GitRevision
    expected_ktransformers_revision: GitRevision
    required_transformers_version: str

    @property
    def command(self) -> tuple[str, ...]:
        return (self.executable, *self.arguments)

    @model_validator(mode="after")
    def validate_process_contract(self) -> "SglangKtProcessLaunchSpec":
        if self.end_layer <= self.start_layer:
            raise ValueError("SGLang process layer range must be nonempty")
        if not self.arguments:
            raise ValueError("SGLang process arguments must be nonempty")
        environment_names = tuple(name for name, _value in self.environment)
        if len(set(environment_names)) != len(environment_names):
            raise ValueError("SGLang process environment names must be unique")
        return self


def build_glm_5_2_fp8_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: str,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build pinned PP=stage-count, TP=1 SGLang-KT process descriptions.

    The returned values are inert. A later executor must verify the recorded
    revisions and apply CPU affinity before starting any process.
    """

    _validate_supported_plan(plan)

    pipeline_size = len(plan.stages)
    layer_partition = ",".join(
        str(layer_count) for layer_count in plan.pipeline_layer_partition
    )

    # Each pipeline stage is one logical SGLang node. This permits two logical
    # nodes to share a physical host while satisfying SGLang's world-size rules.
    return tuple(
        SglangKtProcessLaunchSpec(
            pipeline_rank=stage.pipeline_rank,
            start_layer=stage.start_layer,
            end_layer=stage.end_layer,
            node_id=stage.node_id,
            gpu_uuid=stage.gpu_uuid,
            executable=python_executable,
            arguments=(
                "-m",
                "sglang.launch_server",
                "--model-path",
                stage.model_path,
                "--kt-weight-path",
                stage.ktransformers_weight_path,
                "--kt-cpuinfer",
                str(stage.cpu_infer_threads),
                "--kt-threadpool-count",
                str(stage.threadpool_count),
                "--kt-numa-nodes",
                *(str(memory_node) for memory_node in stage.memory_nodes),
                "--kt-num-gpu-experts",
                str(stage.resident_gpu_experts),
                "--kt-method",
                stage.ktransformers_method,
                "--kt-max-deferred-experts-per-token",
                str(stage.max_deferred_experts_per_token),
                "--kt-expert-placement-strategy",
                "uniform",
                "--pp-size",
                str(pipeline_size),
                "--tp-size",
                "1",
                "--nnodes",
                str(pipeline_size),
                "--node-rank",
                str(stage.pipeline_rank),
                "--dist-init-addr",
                str(plan.distributed_coordinator),
                "--host",
                stage.service_endpoint.ip,
                "--port",
                str(stage.service_endpoint.port),
                "--nccl-port",
                str(stage.nccl_port),
                "--context-length",
                str(plan.context_length),
                "--max-running-requests",
                str(plan.max_concurrent_requests),
                "--attention-backend",
                "nsa",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--disable-shared-experts-fusion",
                "--tool-call-parser",
                "glm47",
                "--reasoning-parser",
                "glm45",
                "--served-model-name",
                "GLM5.2",
                "--trust-remote-code",
            ),
            environment=(
                ("CUDA_VISIBLE_DEVICES", stage.gpu_uuid),
                ("NCCL_NET", "IB"),
                ("NCCL_IB_HCA", f"={','.join(stage.hca_devices)}"),
                ("NCCL_GIN_ENABLE", "0"),
                ("NCCL_GIN_TYPE", "0"),
                ("NCCL_NET_GDR_LEVEL", "LOC"),
                ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
                ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
                ("SGLANG_PP_LAYER_PARTITION", layer_partition),
            ),
            cpu_cores=stage.cpu_cores,
            memory_nodes=stage.memory_nodes,
            service_endpoint=stage.service_endpoint,
            nccl_port=stage.nccl_port,
            model_id=plan.model_id,
            expected_model_revision=plan.model_revision,
            expected_sglang_revision=plan.sglang_revision,
            expected_ktransformers_revision=plan.ktransformers_revision,
            required_transformers_version=REQUIRED_TRANSFORMERS_VERSION,
        )
        for stage in plan.stages
    )


def _validate_supported_plan(plan: SglangKtLaunchPlan) -> None:
    if plan.model_id != GLM_5_2_FP8_MODEL_ID:
        raise ValueError(
            "the SGLang-KT launch builder only supports zai-org/GLM-5.2-FP8"
        )
    if plan.total_layers != GLM_5_2_LAYER_COUNT:
        raise ValueError("GLM-5.2 launch plans must contain exactly 78 layers")
    if plan.sglang_revision != SUPPORTED_SGLANG_REVISION:
        raise ValueError(f"SGLang revision must be {SUPPORTED_SGLANG_REVISION}")
    if plan.ktransformers_revision != SUPPORTED_KTRANSFORMERS_REVISION:
        raise ValueError(
            f"KTransformers revision must be {SUPPORTED_KTRANSFORMERS_REVISION}"
        )
    for stage in plan.stages:
        if stage.ktransformers_method != "FP8":
            raise ValueError("GLM-5.2-FP8 stages require KTransformers method FP8")
        if not stage.hca_devices:
            raise ValueError("SGLang-KT InfiniBand stages require hca_devices")
