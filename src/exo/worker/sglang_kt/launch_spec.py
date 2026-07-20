import hashlib
from collections.abc import Mapping
from typing import Annotated, Final, Literal, final

from pydantic import PositiveInt, StringConstraints, model_validator

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
    SglangKtTargetProfile,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.receipt_io import canonical_sglang_kt_json

GLM_5_2_FP8_MODEL_ID: Final = ModelId("zai-org/GLM-5.2-FP8")
GLM_5_2_LAYER_COUNT: Final = 78
GLM_5_2_TARGET_PROFILE: Final[SglangKtTargetProfile] = "glm52_fp8_pp3_sm86_v1"
GLM_5_2_PIPELINE_LAYER_PARTITION: Final = (30, 28, 20)
GLM_5_2_FULL_INDEXER_LAYER_STARTS: Final = frozenset(
    (0, 1, 2, *range(6, GLM_5_2_LAYER_COUNT, 4))
)

GLM_4_7_FLASH_BF16_MODEL_ID: Final = ModelId("zai-org/GLM-4.7-Flash")
GLM_4_7_FLASH_BF16_MODEL_REVISION: Final = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
GLM_4_7_FLASH_BF16_CONFIG_SHA256: Final = (
    "dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb"
)
GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256: Final = (
    "4e7333f341ddc5855aa4253d454e3210d84427fae0159eb956104ff00c437479"
)
GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME: Final = "glm47_flash_bf16_7dd20894.json"
GLM_4_7_FLASH_LAYER_COUNT: Final = 47
GLM_4_7_FLASH_CONTEXT_LENGTH: Final = 202_752
GLM_4_7_FLASH_MAX_TOTAL_TOKENS: Final = 4_096
GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE: Final = 1_024
GLM_4_7_FLASH_ROUTED_EXPERT_COUNT: Final = 64
GLM_4_7_FLASH_TARGET_PROFILE: Final[SglangKtTargetProfile] = (
    "glm47_flash_bf16_sm86_smoke_v1"
)
GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE: Final[SglangKtTargetProfile] = (
    "glm47_flash_bf16_sm86_cpu_routed_experts_control_v1"
)
GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE: Final[SglangKtTargetProfile] = (
    "glm47_flash_bf16_sm86_serving_baseline_v1"
)
GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE: Final[SglangKtTargetProfile] = (
    "glm47_flash_bf16_sm86_pp3_diagnostic_v1"
)
GLM_4_7_FLASH_PP3_PIPELINE_LAYER_PARTITION: Final = (16, 16, 15)
GLM_4_7_FLASH_TARGET_PROFILES: Final = frozenset(
    (
        GLM_4_7_FLASH_TARGET_PROFILE,
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
        GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
        GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
    )
)

# KTransformers v0.6.3 is the first release with explicit GLM-5.2 support. Its
# SGLang submodule pins the matching fork revision below.
SUPPORTED_KTRANSFORMERS_REVISION: Final = "ce7c3ddbe93f7ac1f992375eed54058bbc512646"
SUPPORTED_SGLANG_REVISION: Final = "8b636f9008dbad58c0a8e481b03e794739e6c146"
GLM_4_7_FLASH_KTRANSFORMERS_REVISION: Final = "f9ca69648421f5774215c4da9cf711dccf54f49e"
GLM_4_7_FLASH_SGLANG_REVISION: Final = "3721d710102456b6bf849122e781129dc3f7d9c6"
REQUIRED_TRANSFORMERS_DISTRIBUTION: Final = "transformers-kt"
REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION: Final = "5.6.0.post1"
REQUIRED_TRANSFORMERS_MODULE_VERSION: Final = "5.6.0"
# Backwards-compatible name for callers that only tracked the distribution.
REQUIRED_TRANSFORMERS_VERSION: Final = REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION
GLM_5_2_KV_CACHE_DTYPE: Final = "fp8_e4m3"
GLM_4_7_FLASH_KV_CACHE_DTYPE: Final = "bfloat16"

EnvironmentVariable = tuple[str, str]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SGLANG_KT_PROCESS_LAUNCH_SPEC_CANONICALIZATION: Final = (
    "exo-sglang-kt-process-launch-spec-v1"
)
SglangKtPythonExecutable = str | Mapping[NodeId, str]


@final
class SglangKtProcessLaunchSpec(FrozenModel):
    """One inert process-launch description for an external SGLang runtime."""

    plan: SglangKtLaunchPlan
    pipeline_rank: ResourceIndex
    executable: AbsoluteRuntimePath
    model_contract_sha256: Sha256Digest | None

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
    def target_profile(self) -> SglangKtTargetProfile:
        return self.plan.target_profile

    @property
    def expected_model_revision(self) -> GitRevision:
        return self.plan.model_revision

    @property
    def expected_model_contract_sha256(self) -> str | None:
        return self.model_contract_sha256

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
    def required_transformers_distribution_version(self) -> str:
        return REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION

    @property
    def required_transformers_module_version(self) -> str:
        return REQUIRED_TRANSFORMERS_MODULE_VERSION

    @property
    def attention_backend(self) -> Literal["flashinfer", "nsa"]:
        if self.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
            return "flashinfer"
        return "nsa"

    @property
    def kv_cache_dtype(self) -> Literal["bfloat16", "fp8_e4m3"]:
        if self.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
            return GLM_4_7_FLASH_KV_CACHE_DTYPE
        return GLM_5_2_KV_CACHE_DTYPE

    @property
    def arguments(self) -> tuple[str, ...]:
        stage = self.stage
        pipeline_size = len(self.plan.stages)
        common_arguments = (
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
        )
        if self.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
            glm47_arguments = (
                *common_arguments,
                "--chunked-prefill-size",
                str(GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE),
                "--disable-cuda-graph",
                "--attention-backend",
                self.attention_backend,
                "--kv-cache-dtype",
                self.kv_cache_dtype,
                "--disable-shared-experts-fusion",
                "--tool-call-parser",
                "glm47",
                "--reasoning-parser",
                "glm45",
                "--served-model-name",
                "GLM-4.7-Flash",
                "--trust-remote-code",
            )
            if self.target_profile in (
                GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
                GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
            ):
                return (*glm47_arguments, "--disable-radix-cache")
            return (*glm47_arguments, "--record-kt-gpu-expert-distribution")
        return (
            *common_arguments,
            "--attention-backend",
            self.attention_backend,
            "--kv-cache-dtype",
            self.kv_cache_dtype,
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
        common_environment: tuple[EnvironmentVariable, ...] = (
            ("CUDA_VISIBLE_DEVICES", self.stage.gpu_uuid),
            ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        )
        if self.target_profile == GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE:
            layer_partition = ",".join(
                str(layer_count) for layer_count in self.plan.pipeline_layer_partition
            )
            return (
                common_environment[0],
                ("NCCL_NET", "IB"),
                ("NCCL_IB_HCA", f"={','.join(self.stage.hca_devices)}"),
                ("NCCL_GIN_ENABLE", "0"),
                ("NCCL_GIN_TYPE", "0"),
                ("NCCL_NET_GDR_LEVEL", "LOC"),
                common_environment[1],
                ("SGLANG_PP_LAYER_PARTITION", layer_partition),
            )
        if self.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
            if self.target_profile == GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE:
                return common_environment
            return (
                *common_environment,
                ("SGLANG_KT_HYBRID_TIMING", "1"),
            )

        layer_partition = ",".join(
            str(layer_count) for layer_count in self.plan.pipeline_layer_partition
        )
        return (
            common_environment[0],
            ("NCCL_NET", "IB"),
            ("NCCL_IB_HCA", f"={','.join(self.stage.hca_devices)}"),
            ("NCCL_GIN_ENABLE", "0"),
            ("NCCL_GIN_TYPE", "0"),
            ("NCCL_NET_GDR_LEVEL", "LOC"),
            common_environment[1],
            ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
            ("SGLANG_PP_LAYER_PARTITION", layer_partition),
        )

    @property
    def unset_environment_variables(self) -> tuple[str, ...]:
        return ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF")

    @property
    def unset_environment_variable_prefixes(self) -> tuple[str, ...]:
        # SGLang and NCCL use many behavior-changing environment variables. A
        # launch profile must opt each one back in instead of inheriting shell
        # or service state from a previous, incompatible run.
        return ("NCCL_", "SGLANG_")

    @property
    def command(self) -> tuple[str, ...]:
        return (self.executable, *self.arguments)

    @model_validator(mode="after")
    def validate_process_contract(self) -> "SglangKtProcessLaunchSpec":
        _validate_supported_plan(self.plan)
        if self.pipeline_rank >= len(self.plan.stages):
            raise ValueError("SGLang process rank is absent from its launch plan")
        expected_contract_sha256 = (
            GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
            if self.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
            else None
        )
        if self.model_contract_sha256 != expected_contract_sha256:
            raise ValueError("model contract SHA-256 does not match the target profile")
        return self


def calculate_sglang_kt_process_launch_spec_sha256(
    process_spec: SglangKtProcessLaunchSpec,
) -> str:
    payload = {
        "canonicalization": SGLANG_KT_PROCESS_LAUNCH_SPEC_CANONICALIZATION,
        "process_spec": process_spec.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()


def build_glm_5_2_fp8_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build pinned PP=stage-count, TP=1 SGLang-KT process descriptions.

    The returned values are inert. A later executor must verify the recorded
    revisions and apply CPU affinity before starting any process.
    """

    if plan.target_profile != GLM_5_2_TARGET_PROFILE:
        raise ValueError(
            f"GLM-5.2 builder requires target profile {GLM_5_2_TARGET_PROFILE}"
        )
    return build_sglang_kt_process_launch_specs(plan, python_executable)


def build_glm_4_7_flash_bf16_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build one inert, receipt-gated GLM-4.7-Flash hybrid smoke process."""

    if plan.target_profile != GLM_4_7_FLASH_TARGET_PROFILE:
        raise ValueError(
            "GLM-4.7-Flash builder requires target profile "
            f"{GLM_4_7_FLASH_TARGET_PROFILE}"
        )
    return build_sglang_kt_process_launch_specs(plan, python_executable)


def build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build the receipt-gated zero-resident-GPU-expert control process."""

    if plan.target_profile != GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE:
        raise ValueError(
            "GLM-4.7-Flash CPU-routed-experts builder requires target profile "
            f"{GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE}"
        )
    return build_sglang_kt_process_launch_specs(plan, python_executable)


def build_glm_4_7_flash_bf16_serving_baseline_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build the receipt-gated, instrumentation-free serving baseline."""

    if plan.target_profile != GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE:
        raise ValueError(
            "GLM-4.7-Flash serving builder requires target profile "
            f"{GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE}"
        )
    return build_sglang_kt_process_launch_specs(plan, python_executable)


def build_glm_4_7_flash_bf16_pp3_diagnostic_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build the three-stage GLM-4.7 serving diagnostic process group."""

    if plan.target_profile != GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE:
        raise ValueError(
            "GLM-4.7-Flash PP3 diagnostic builder requires target profile "
            f"{GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE}"
        )
    return build_sglang_kt_process_launch_specs(plan, python_executable)


def build_sglang_kt_process_launch_specs(
    plan: SglangKtLaunchPlan,
    python_executable: SglangKtPythonExecutable,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    """Build inert process descriptions for one statically admitted profile."""

    _validate_supported_plan(plan)
    plan_node_ids = frozenset(stage.node_id for stage in plan.stages)
    if isinstance(python_executable, str):
        executable_by_node: Mapping[NodeId, str] = {
            node_id: python_executable for node_id in plan_node_ids
        }
    else:
        executable_node_ids = frozenset(python_executable)
        if executable_node_ids != plan_node_ids:
            missing_node_ids = (
                ", ".join(sorted(plan_node_ids - executable_node_ids)) or "none"
            )
            unexpected_node_ids = (
                ", ".join(sorted(executable_node_ids - plan_node_ids)) or "none"
            )
            raise ValueError(
                "Python executable mapping keys must exactly match launch plan "
                f"node IDs (missing: {missing_node_ids}; "
                f"unexpected: {unexpected_node_ids})"
            )
        executable_by_node = python_executable

    # Each pipeline stage is one logical SGLang node. This permits two logical
    # nodes to share a physical host while satisfying SGLang's world-size rules.
    return tuple(
        SglangKtProcessLaunchSpec(
            plan=plan,
            pipeline_rank=stage.pipeline_rank,
            executable=executable_by_node[stage.node_id],
            model_contract_sha256=(
                GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
                if plan.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
                else None
            ),
        )
        for stage in plan.stages
    )


def _validate_supported_plan(plan: SglangKtLaunchPlan) -> None:
    if plan.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
        _validate_glm_4_7_flash_bf16_plan(plan)
        return
    _validate_glm_5_2_fp8_plan(plan)


def _validate_glm_5_2_fp8_plan(plan: SglangKtLaunchPlan) -> None:
    if plan.target_profile != GLM_5_2_TARGET_PROFILE:
        raise ValueError(f"unsupported SGLang-KT target profile {plan.target_profile}")
    if plan.model_id != GLM_5_2_FP8_MODEL_ID:
        raise ValueError(
            "the SGLang-KT launch builder only supports zai-org/GLM-5.2-FP8"
        )
    if plan.total_layers != GLM_5_2_LAYER_COUNT:
        raise ValueError("GLM-5.2 launch plans must contain exactly 78 layers")
    if len(plan.stages) != len(GLM_5_2_PIPELINE_LAYER_PARTITION):
        raise ValueError("GLM-5.2 PP=3 target profile requires exactly three stages")
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
    if plan.pipeline_layer_partition != GLM_5_2_PIPELINE_LAYER_PARTITION:
        raise ValueError(
            "GLM-5.2 PP=3 target profile requires the 30,28,20 layer partition"
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


def _validate_glm_4_7_flash_bf16_plan(plan: SglangKtLaunchPlan) -> None:
    if plan.target_profile not in GLM_4_7_FLASH_TARGET_PROFILES:
        raise ValueError(f"unsupported SGLang-KT target profile {plan.target_profile}")
    if plan.model_id != GLM_4_7_FLASH_BF16_MODEL_ID:
        raise ValueError(
            "the GLM-4.7-Flash smoke profile only supports zai-org/GLM-4.7-Flash"
        )
    if plan.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION:
        raise ValueError(
            f"GLM-4.7-Flash model revision must be {GLM_4_7_FLASH_BF16_MODEL_REVISION}"
        )
    if plan.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        raise ValueError(f"SGLang revision must be {GLM_4_7_FLASH_SGLANG_REVISION}")
    if plan.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION:
        raise ValueError(
            f"KTransformers revision must be {GLM_4_7_FLASH_KTRANSFORMERS_REVISION}"
        )
    if plan.total_layers != GLM_4_7_FLASH_LAYER_COUNT:
        raise ValueError("GLM-4.7-Flash launch plans must contain exactly 47 layers")
    if (
        plan.target_profile != GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE
        and len(plan.stages) != 1
    ):
        raise ValueError("GLM-4.7-Flash smoke profile requires PP=1 and TP=1")
    if plan.context_length != GLM_4_7_FLASH_CONTEXT_LENGTH:
        raise ValueError(
            f"GLM-4.7-Flash smoke context length must be {GLM_4_7_FLASH_CONTEXT_LENGTH}"
        )
    if plan.max_total_tokens != GLM_4_7_FLASH_MAX_TOTAL_TOKENS:
        raise ValueError(
            "GLM-4.7-Flash smoke max_total_tokens must be "
            f"{GLM_4_7_FLASH_MAX_TOTAL_TOKENS}"
        )
    if plan.max_concurrent_requests != 1:
        raise ValueError("GLM-4.7-Flash smoke profile requires one running request")
    if plan.static_memory_fraction != 0.8:
        raise ValueError("GLM-4.7-Flash smoke static memory fraction must be 0.8")

    if plan.target_profile == GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE:
        _validate_glm_4_7_flash_bf16_pp3_diagnostic_plan(plan)
        return

    stage = plan.stages[0]
    if stage.model_path != stage.ktransformers_weight_path:
        raise ValueError(
            "the GLM-4.7-Flash BF16 smoke profile requires model_path and "
            "ktransformers_weight_path to match"
        )
    if stage.ktransformers_method != "BF16":
        raise ValueError("GLM-4.7-Flash BF16 stage requires KTransformers method BF16")
    if (
        plan.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        and stage.resident_gpu_experts != 0
    ):
        raise ValueError(
            "GLM-4.7-Flash CPU-routed-experts control requires exactly 0 resident "
            "GPU experts"
        )
    if (
        plan.target_profile
        in (
            GLM_4_7_FLASH_TARGET_PROFILE,
            GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
        )
        and not 1 <= stage.resident_gpu_experts < GLM_4_7_FLASH_ROUTED_EXPERT_COUNT
    ):
        raise ValueError(
            "GLM-4.7-Flash hybrid smoke requires between 1 and 63 resident GPU "
            "experts per MoE layer"
        )
    if stage.max_deferred_experts_per_token != 0:
        raise ValueError("GLM-4.7-Flash smoke requires deferred experts disabled")
    if stage.hca_devices:
        raise ValueError("GLM-4.7-Flash PP=1 smoke profile does not admit HCA devices")


def _validate_glm_4_7_flash_bf16_pp3_diagnostic_plan(
    plan: SglangKtLaunchPlan,
) -> None:
    if len(plan.stages) != len(GLM_4_7_FLASH_PP3_PIPELINE_LAYER_PARTITION):
        raise ValueError("GLM-4.7-Flash PP3 diagnostic profile requires three stages")
    if plan.pipeline_layer_partition != GLM_4_7_FLASH_PP3_PIPELINE_LAYER_PARTITION:
        raise ValueError(
            "GLM-4.7-Flash PP3 diagnostic profile requires the 16,16,15 layer partition"
        )

    for stage in plan.stages:
        if stage.model_path != stage.ktransformers_weight_path:
            raise ValueError(
                "the GLM-4.7-Flash BF16 PP3 diagnostic profile requires model_path "
                "and ktransformers_weight_path to match"
            )
        if stage.ktransformers_method != "BF16":
            raise ValueError(
                "GLM-4.7-Flash BF16 PP3 diagnostic stages require KTransformers "
                "method BF16"
            )
        if not 1 <= stage.resident_gpu_experts < GLM_4_7_FLASH_ROUTED_EXPERT_COUNT:
            raise ValueError(
                "GLM-4.7-Flash PP3 diagnostic stages require between 1 and 63 "
                "resident GPU experts per MoE layer"
            )
        if stage.max_deferred_experts_per_token != 0:
            raise ValueError(
                "GLM-4.7-Flash PP3 diagnostic stages require deferred experts disabled"
            )
        if not stage.hca_devices:
            raise ValueError("GLM-4.7-Flash PP3 diagnostic stages require hca_devices")
