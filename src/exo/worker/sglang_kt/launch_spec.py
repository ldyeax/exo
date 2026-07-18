from typing import Final, final

from pydantic import PositiveInt, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    HcaDevice,
    KTransformersMethod,
    ResourceIndex,
    SglangKtLaunchPlan,
    SglangKtStageSpec,
)
from exo.utils.pydantic_ext import FrozenModel

GLM_5_2_FP8_MODEL_ID: Final = ModelId("zai-org/GLM-5.2-FP8")
GLM_5_2_LAYER_COUNT: Final = 78
GLM_5_2_FULL_INDEXER_LAYER_STARTS: Final = frozenset(
    (0, 1, 2, *range(6, GLM_5_2_LAYER_COUNT, 4))
)

# KTransformers v0.6.3 is the first release with explicit GLM-5.2 support. Its
# SGLang submodule pins the matching fork revision below.
SUPPORTED_KTRANSFORMERS_REVISION: Final = "ce7c3ddbe93f7ac1f992375eed54058bbc512646"
SUPPORTED_SGLANG_REVISION: Final = "8b636f9008dbad58c0a8e481b03e794739e6c146"
REQUIRED_TRANSFORMERS_DISTRIBUTION: Final = "transformers-kt"
REQUIRED_TRANSFORMERS_VERSION: Final = "5.6.0.post1"
GLM_5_2_KV_CACHE_DTYPE: Final = "fp8_e4m3"

EnvironmentVariable = tuple[str, str]


@final
class SglangKtProcessLaunchSpec(FrozenModel):
    """One inert process-launch description for an external SGLang runtime."""

    plan: SglangKtLaunchPlan
    pipeline_rank: ResourceIndex
    executable: AbsoluteRuntimePath

    @property
    def stage(self) -> SglangKtStageSpec:
        return self.plan.stages[self.pipeline_rank]

    @property
    def start_layer(self) -> ResourceIndex:
        return self.stage.start_layer

    @property
    def end_layer(self) -> PositiveInt:
        return self.stage.end_layer

    @property
    def node_id(self) -> NodeId:
        return self.stage.node_id

    @property
    def gpu_uuid(self) -> GpuUuid:
        return self.stage.gpu_uuid

    @property
    def model_path(self) -> AbsoluteRuntimePath:
        return self.stage.model_path

    @property
    def ktransformers_weight_path(self) -> AbsoluteRuntimePath:
        return self.stage.ktransformers_weight_path

    @property
    def cpu_cores(self) -> tuple[ResourceIndex, ...]:
        return self.stage.cpu_cores

    @property
    def memory_nodes(self) -> tuple[ResourceIndex, ...]:
        return self.stage.memory_nodes

    @property
    def hca_devices(self) -> tuple[HcaDevice, ...]:
        return self.stage.hca_devices

    @property
    def ktransformers_method(self) -> KTransformersMethod:
        return self.stage.ktransformers_method

    @property
    def service_endpoint(self) -> Host:
        return self.stage.service_endpoint

    @property
    def distributed_coordinator(self) -> Host:
        return self.plan.distributed_coordinator

    @property
    def model_id(self) -> ModelId:
        return self.plan.model_id

    @property
    def expected_model_revision(self) -> GitRevision:
        return self.plan.model_revision

    @property
    def expected_sglang_revision(self) -> GitRevision:
        return self.plan.sglang_revision

    @property
    def expected_ktransformers_revision(self) -> GitRevision:
        return self.plan.ktransformers_revision

    @property
    def required_transformers_version(self) -> str:
        return REQUIRED_TRANSFORMERS_VERSION

    @property
    def arguments(self) -> tuple[str, ...]:
        stage = self.stage
        pipeline_size = len(self.plan.stages)
        return (
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
            str(self.plan.distributed_coordinator),
            "--host",
            stage.service_endpoint.ip,
            "--port",
            str(stage.service_endpoint.port),
            "--context-length",
            str(self.plan.context_length),
            "--max-total-tokens",
            str(self.plan.max_total_tokens),
            "--mem-fraction-static",
            str(self.plan.static_memory_fraction),
            "--max-running-requests",
            str(self.plan.max_concurrent_requests),
            "--attention-backend",
            "nsa",
            "--kv-cache-dtype",
            GLM_5_2_KV_CACHE_DTYPE,
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--served-model-name",
            "GLM5.2",
            "--trust-remote-code",
        )

    @property
    def environment(self) -> tuple[EnvironmentVariable, ...]:
        layer_partition = ",".join(
            str(layer_count) for layer_count in self.plan.pipeline_layer_partition
        )
        return (
            ("CUDA_VISIBLE_DEVICES", self.stage.gpu_uuid),
            ("NCCL_NET", "IB"),
            ("NCCL_IB_HCA", f"={','.join(self.stage.hca_devices)}"),
            ("NCCL_GIN_ENABLE", "0"),
            ("NCCL_GIN_TYPE", "0"),
            ("NCCL_NET_GDR_LEVEL", "LOC"),
            ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
            ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
            ("SGLANG_PP_LAYER_PARTITION", layer_partition),
        )

    @property
    def unset_environment_variables(self) -> tuple[str, ...]:
        return ("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE",)

    @property
    def command(self) -> tuple[str, ...]:
        return (self.executable, *self.arguments)

    @model_validator(mode="after")
    def validate_process_contract(self) -> "SglangKtProcessLaunchSpec":
        _validate_supported_plan(self.plan)
        if self.pipeline_rank >= len(self.plan.stages):
            raise ValueError("SGLang process rank is absent from its launch plan")
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

    # Each pipeline stage is one logical SGLang node. This permits two logical
    # nodes to share a physical host while satisfying SGLang's world-size rules.
    return tuple(
        SglangKtProcessLaunchSpec(
            plan=plan,
            pipeline_rank=stage.pipeline_rank,
            executable=python_executable,
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
    invalid_pipeline_starts = tuple(
        stage.start_layer
        for stage in plan.stages[1:]
        if stage.start_layer not in GLM_5_2_FULL_INDEXER_LAYER_STARTS
    )
    if invalid_pipeline_starts:
        raise ValueError(
            "GLM-5.2 pipeline stages must begin on full IndexShare layers; "
            f"invalid starts: {invalid_pipeline_starts}"
        )
    for stage in plan.stages:
        if stage.model_path != stage.ktransformers_weight_path:
            raise ValueError(
                "the audited GLM-5.2 runtime requires model_path and "
                "ktransformers_weight_path to match"
            )
        if stage.ktransformers_method != "FP8":
            raise ValueError("GLM-5.2-FP8 stages require KTransformers method FP8")
        if len(plan.stages) > 1 and stage.max_deferred_experts_per_token != 0:
            raise ValueError(
                "pipeline GLM-5.2 stages require deferred expert work to be disabled"
            )
        if not stage.hca_devices:
            raise ValueError("SGLang-KT InfiniBand stages require hca_devices")
