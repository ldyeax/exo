import hashlib
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from exo.shared.types.common import ModelId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    ResourceIndex,
    SglangKtTargetProfile,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_ROUTED_EXPERT_COUNT,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION,
    REQUIRED_TRANSFORMERS_MODULE_VERSION,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)

MODEL_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
MODEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES = 4 * 1024 * 1024
GLM_4_7_FLASH_BF16_INDEX_SHA256 = (
    "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
)

_EXPECTED_TORCH_VERSION = "2.9.1+cu128"
_EXPECTED_CUDA_VERSION = "12.8"
_EXPECTED_GPU_COMPUTE_CAPABILITY = (8, 6)
_EXPECTED_WRAPPED_LAYERS = tuple(range(1, 47))
_EXPECTED_EXPERT_MODULE_TEMPLATE = "model.layers.{layer}.mlp.experts"
_EXPECTED_CPU_BACKEND_WRAPPER = "NativeMoEWrapper"
_EXPECTED_CPU_KERNEL_CLASS = "AMXBF16_MOE"
_EXPECTED_QUANT_METHOD_WRAPPER = "kt_ep"
_EXPECTED_LAYER_ONE_PROBE_SEED = 20_260_719
_EXPECTED_LAYER_ONE_RELATIVE_L1_TOLERANCE = 0.02
_EXPECTED_EXTEND_TOKEN_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
_EXPECTED_EXTEND_POSITIONS = tuple(range(8))
_EXPECTED_LOGITS_SHAPE = (1, 154_880)
_EXPECTED_HIDDEN_SHAPE = (1, 2_048)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

NonemptyText = Annotated[str, StringConstraints(min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ExpertId = Annotated[int, Field(ge=0, lt=GLM_4_7_FLASH_ROUTED_EXPERT_COUNT)]
TokenId = Annotated[int, Field(ge=0, lt=154_880)]
SglangKtModelRuntimeCapability = Literal[
    "glm47_flash_kt_wrapper_active_v1",
    "glm47_flash_kt_wrapper_layers_1_46_v1",
    "glm47_flash_bf16_sm86_short_forward_v1",
    "kt_physical_numa_mapping_v1",
    "kt_process_cpu_affinity_v1",
    "kt_bf16_amx_executed_v1",
    "kt_bf16_cpu_gpu_hybrid_executed_v1",
    "glm47_flash_bf16_cpu_routed_experts_executed_v1",
]

_COMMON_CAPABILITIES: tuple[SglangKtModelRuntimeCapability, ...] = (
    "glm47_flash_kt_wrapper_active_v1",
    "glm47_flash_kt_wrapper_layers_1_46_v1",
    "glm47_flash_bf16_sm86_short_forward_v1",
    "kt_physical_numa_mapping_v1",
    "kt_process_cpu_affinity_v1",
    "kt_bf16_amx_executed_v1",
)


def calculate_glm_4_7_flash_expert_mask_sha256(
    resident_gpu_expert_ids_by_layer: tuple[tuple[int, ...], ...],
) -> str:
    """Hash the exact 47x64 uint8 mask emitted by the pinned SGLang patch."""

    if len(resident_gpu_expert_ids_by_layer) != len(_EXPECTED_WRAPPED_LAYERS):
        raise ValueError("expert mask must describe exactly layers 1 through 46")
    # The pinned KTransformers mask marks the dense layer as fully resident.
    # Routed-expert placement then occupies rows 1 through 46.
    mask = bytearray([1] * GLM_4_7_FLASH_ROUTED_EXPERT_COUNT)
    mask.extend(bytearray(46 * GLM_4_7_FLASH_ROUTED_EXPERT_COUNT))
    for layer_index, resident_expert_ids in zip(
        _EXPECTED_WRAPPED_LAYERS,
        resident_gpu_expert_ids_by_layer,
        strict=True,
    ):
        if resident_expert_ids != tuple(sorted(set(resident_expert_ids))) or any(
            expert_id < 0 or expert_id >= GLM_4_7_FLASH_ROUTED_EXPERT_COUNT
            for expert_id in resident_expert_ids
        ):
            raise ValueError("resident GPU expert IDs must be sorted and unique")
        row_offset = layer_index * GLM_4_7_FLASH_ROUTED_EXPERT_COUNT
        for expert_id in resident_expert_ids:
            mask[row_offset + expert_id] = 1
    return hashlib.sha256(mask).hexdigest()


class SglangKtModelRuntimeValidationReceiptError(ValueError):
    """Raised when a GLM-4.7 model validation receipt is not admissible."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_absolute_normalized_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
        or value != os.path.normpath(value)
    ):
        raise ValueError("receipt binding paths must be normalized absolute paths")
    return value


def _validate_sorted_unique(values: tuple[int, ...], description: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ValueError(f"{description} must be nonempty, sorted, and unique")


@final
class _ModelContractBinding(_StrictModel):
    path: AbsoluteRuntimePath
    receipt_sha256: Sha256Digest
    contract_sha256: Sha256Digest
    config_sha256: Sha256Digest
    index_sha256: Sha256Digest

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)


@final
class _KernelRuntimeValidationBinding(_StrictModel):
    receipt_path: AbsoluteRuntimePath
    receipt_sha256: Sha256Digest

    @field_validator("receipt_path")
    @classmethod
    def validate_receipt_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)


@final
class _ParentBindings(_StrictModel):
    process_spec_sha256: Sha256Digest
    model_contract: _ModelContractBinding
    kernel_runtime_validation: _KernelRuntimeValidationBinding


@final
class _RuntimeLaunchSummary(_StrictModel):
    target_profile: SglangKtTargetProfile
    gpu_uuid: GpuUuid
    gpu_compute_capability: tuple[PositiveInt, ResourceIndex]
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    executed_cpu_backend: Literal["AMX_BF16"]
    model_id: ModelId
    model_revision: GitRevision
    model_config_sha256: Sha256Digest
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    transformers_distribution_version: NonemptyText
    transformers_module_version: NonemptyText
    torch_version: NonemptyText
    cuda_version: NonemptyText
    sgl_kernel_build_id: Sha256Digest
    deep_gemm_build_id: Sha256Digest
    kt_kernel_build_id: Sha256Digest
    ktransformers_method: Literal["BF16"]
    resident_gpu_experts: ResourceIndex
    attention_backend: Literal["flashinfer"]
    kv_cache_dtype: Literal["bfloat16"]
    max_total_tokens: PositiveInt
    static_memory_fraction: float

    @model_validator(mode="after")
    def validate_exact_runtime(self) -> "_RuntimeLaunchSummary":
        if self.target_profile not in (
            GLM_4_7_FLASH_TARGET_PROFILE,
            GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
        ):
            raise ValueError("model receipt target profile is not GLM-4.7 Flash")
        if self.gpu_compute_capability != _EXPECTED_GPU_COMPUTE_CAPABILITY:
            raise ValueError("model receipt requires an exact SM86 GPU")
        _validate_sorted_unique(self.cpu_cores, "runtime CPU cores")
        _validate_sorted_unique(self.memory_nodes, "runtime memory nodes")
        if (
            self.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
            or self.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
            or self.model_config_sha256 != GLM_4_7_FLASH_BF16_CONFIG_SHA256
        ):
            raise ValueError("model receipt checkpoint identity is not pinned")
        if (
            self.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
            or self.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        ):
            raise ValueError("model receipt source revisions are not pinned")
        if (
            self.transformers_distribution_version
            != REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION
            or self.transformers_module_version != REQUIRED_TRANSFORMERS_MODULE_VERSION
            or self.torch_version != _EXPECTED_TORCH_VERSION
            or self.cuda_version != _EXPECTED_CUDA_VERSION
        ):
            raise ValueError("model receipt runtime versions are not pinned")
        if (
            self.max_total_tokens != GLM_4_7_FLASH_MAX_TOTAL_TOKENS
            or self.static_memory_fraction != 0.8
        ):
            raise ValueError("model receipt launch limits are not pinned")
        if self.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE:
            if self.resident_gpu_experts != 0:
                raise ValueError("CPU control receipt cannot have resident GPU experts")
        elif not 1 <= self.resident_gpu_experts < GLM_4_7_FLASH_ROUTED_EXPERT_COUNT:
            raise ValueError("mixed receipt requires between 1 and 63 GPU experts")
        return self


@final
class _WrapperLayerEvidence(_StrictModel):
    layer_index: ResourceIndex
    expert_module_name: NonemptyText
    quant_method_wrapper: Literal["kt_ep"]
    expert_count: PositiveInt
    resident_gpu_expert_ids: tuple[ExpertId, ...]
    cpu_backend_wrapper_class: Literal["NativeMoEWrapper"]
    cpu_kernel_class: Literal["AMXBF16_MOE"]
    global_expert_mask_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_layer(self) -> "_WrapperLayerEvidence":
        if self.expert_module_name != _EXPECTED_EXPERT_MODULE_TEMPLATE.format(
            layer=self.layer_index
        ):
            raise ValueError("wrapper evidence names the wrong expert module")
        if self.expert_count != GLM_4_7_FLASH_ROUTED_EXPERT_COUNT:
            raise ValueError("wrapper evidence requires exactly 64 routed experts")
        if self.resident_gpu_expert_ids != tuple(
            sorted(set(self.resident_gpu_expert_ids))
        ):
            raise ValueError("resident GPU expert IDs must be sorted and unique")
        return self


@final
class _WrapperCoverage(_StrictModel):
    global_expert_mask_sha256: Sha256Digest
    layers: tuple[_WrapperLayerEvidence, ...]


@final
class _NumericalEvidence(_StrictModel):
    shape: tuple[PositiveInt, ...]
    execution_dtype: Literal["torch.bfloat16"]
    reference_dtype: Literal["torch.float32"]
    output_finite: bool
    reference_finite: bool
    mean_absolute_error: float
    maximum_absolute_error: float
    reference_mean_absolute: float
    relative_l1_error: float
    relative_l1_tolerance: float

    @model_validator(mode="after")
    def validate_numerical_evidence(self) -> "_NumericalEvidence":
        metrics = (
            self.mean_absolute_error,
            self.maximum_absolute_error,
            self.reference_mean_absolute,
            self.relative_l1_error,
            self.relative_l1_tolerance,
        )
        if not all(math.isfinite(metric) for metric in metrics):
            raise ValueError("layer-one numerical metrics must be finite")
        if (
            self.shape != _EXPECTED_HIDDEN_SHAPE
            or not self.output_finite
            or not self.reference_finite
            or self.mean_absolute_error < 0
            or self.maximum_absolute_error < 0
            or self.reference_mean_absolute <= 0
            or self.relative_l1_error < 0
            or self.relative_l1_tolerance != _EXPECTED_LAYER_ONE_RELATIVE_L1_TOLERANCE
            or self.relative_l1_error > self.relative_l1_tolerance
        ):
            raise ValueError("layer-one numerical evidence is not admissible")
        return self


@final
class _LayerOneExpertProbe(_StrictModel):
    layer_index: ResourceIndex
    random_seed: int
    probe_invocation_count: PositiveInt
    input_sha256: Sha256Digest
    selected_expert_ids: tuple[ExpertId, ...]
    repeat_selected_expert_ids: tuple[ExpertId, ...]
    cpu_expert_ids: tuple[ExpertId, ...]
    gpu_expert_ids: tuple[ExpertId, ...]
    cpu_backend_wrapper_class: Literal["NativeMoEWrapper"]
    cpu_kernel_class: Literal["AMXBF16_MOE"]
    cpu_submit_count: ResourceIndex
    cpu_sync_count: ResourceIndex
    gpu_forward_count: ResourceIndex
    output_merge_count: ResourceIndex
    output_sha256: Sha256Digest
    repeat_output_sha256: Sha256Digest
    global_expert_mask_sha256: Sha256Digest
    numerical: _NumericalEvidence

    @model_validator(mode="after")
    def validate_deterministic_probe(self) -> "_LayerOneExpertProbe":
        if (
            self.layer_index != 1
            or self.random_seed != _EXPECTED_LAYER_ONE_PROBE_SEED
            or self.probe_invocation_count != 2
            or len(self.selected_expert_ids) != 4
            or len(set(self.selected_expert_ids)) != 4
            or self.repeat_selected_expert_ids != self.selected_expert_ids
            or self.repeat_output_sha256 != self.output_sha256
            or self.cpu_submit_count != 2
            or self.cpu_sync_count != 2
            or self.output_merge_count != 2
        ):
            raise ValueError("layer-one deterministic expert probe is incomplete")
        return self


@final
class _ShortForwardInvocation(_StrictModel):
    forward_mode: Literal["extend", "decode"]
    input_token_ids: tuple[TokenId, ...]
    positions: tuple[ResourceIndex, ...]
    kv_cache_length_before: ResourceIndex
    kv_cache_length_after: PositiveInt
    model_forward_invocation_count: PositiveInt
    logits_shape: tuple[PositiveInt, ...]
    logits_dtype: Literal["torch.float32"]
    logits_finite: bool
    logits_sha256: Sha256Digest
    argmax_token_id: TokenId
    global_expert_mask_sha256_before: Sha256Digest
    global_expert_mask_sha256_after: Sha256Digest


@final
class _WrapperForwardInvocation(_StrictModel):
    layer_index: ResourceIndex
    extend_invocation_count: PositiveInt
    decode_invocation_count: PositiveInt


@final
class _ShortForwardEvidence(_StrictModel):
    random_seed: int
    extend: _ShortForwardInvocation
    decode: _ShortForwardInvocation
    wrapper_invocations: tuple[_WrapperForwardInvocation, ...]

    @model_validator(mode="after")
    def validate_full_short_forward(self) -> "_ShortForwardEvidence":
        extend = self.extend
        decode = self.decode
        if (
            self.random_seed != _EXPECTED_LAYER_ONE_PROBE_SEED
            or extend.forward_mode != "extend"
            or extend.input_token_ids != _EXPECTED_EXTEND_TOKEN_IDS
            or extend.positions != _EXPECTED_EXTEND_POSITIONS
            or extend.kv_cache_length_before != 0
            or extend.kv_cache_length_after != 8
            or extend.model_forward_invocation_count != 1
            or extend.logits_shape != _EXPECTED_LOGITS_SHAPE
            or not extend.logits_finite
            or decode.forward_mode != "decode"
            or decode.input_token_ids != (extend.argmax_token_id,)
            or decode.positions != (8,)
            or decode.kv_cache_length_before != 8
            or decode.kv_cache_length_after != 9
            or decode.model_forward_invocation_count != 1
            or decode.logits_shape != _EXPECTED_LOGITS_SHAPE
            or not decode.logits_finite
        ):
            raise ValueError("full extend/decode short-forward evidence is incomplete")
        invocation_layers = tuple(
            invocation.layer_index for invocation in self.wrapper_invocations
        )
        if invocation_layers != _EXPECTED_WRAPPED_LAYERS or any(
            invocation.extend_invocation_count != 1
            or invocation.decode_invocation_count != 1
            for invocation in self.wrapper_invocations
        ):
            raise ValueError("short forward did not invoke every wrapped layer exactly")
        return self


@final
class _PassedModelRuntimeValidationReceiptV1(_StrictModel):
    schema_version: Literal[1]
    status: Literal["passed"]
    generated_at_utc: NonemptyText
    profiler: Literal["none"]
    validator_sha256: Sha256Digest
    failures: tuple[NonemptyText, ...]
    parents: _ParentBindings
    runtime: _RuntimeLaunchSummary
    wrapper_coverage: _WrapperCoverage
    layer_one_expert_probe: _LayerOneExpertProbe
    short_forward: _ShortForwardEvidence

    @field_validator("generated_at_utc")
    @classmethod
    def validate_generated_at_utc(cls, value: str) -> str:
        try:
            generated_at = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("model receipt generation time is not ISO-8601") from error
        if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
            raise ValueError("model receipt generation time must be UTC")
        return value

    @model_validator(mode="after")
    def validate_model_execution_evidence(
        self,
    ) -> "_PassedModelRuntimeValidationReceiptV1":
        if self.failures:
            raise ValueError("passed model receipt contains failures")
        contract = self.parents.model_contract
        if (
            contract.contract_sha256 != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
            or contract.config_sha256 != GLM_4_7_FLASH_BF16_CONFIG_SHA256
            or contract.index_sha256 != GLM_4_7_FLASH_BF16_INDEX_SHA256
            or contract.config_sha256 != self.runtime.model_config_sha256
        ):
            raise ValueError("model receipt contract parent is not pinned")

        layers = self.wrapper_coverage.layers
        if tuple(layer.layer_index for layer in layers) != _EXPECTED_WRAPPED_LAYERS:
            raise ValueError(
                "wrapper coverage must contain exactly layers 1 through 46"
            )
        global_mask_sha256 = self.wrapper_coverage.global_expert_mask_sha256
        calculated_mask_sha256 = calculate_glm_4_7_flash_expert_mask_sha256(
            tuple(layer.resident_gpu_expert_ids for layer in layers)
        )
        if global_mask_sha256 != calculated_mask_sha256:
            raise ValueError("wrapper expert mask SHA-256 is not canonical")
        resident_count = self.runtime.resident_gpu_experts
        for layer in layers:
            if (
                layer.global_expert_mask_sha256 != global_mask_sha256
                or len(layer.resident_gpu_expert_ids) != resident_count
                or layer.quant_method_wrapper != _EXPECTED_QUANT_METHOD_WRAPPER
                or layer.cpu_backend_wrapper_class != _EXPECTED_CPU_BACKEND_WRAPPER
                or layer.cpu_kernel_class != _EXPECTED_CPU_KERNEL_CLASS
            ):
                raise ValueError("wrapper layer evidence disagrees with the launch")

        layer_one = layers[0]
        probe = self.layer_one_expert_probe
        resident_ids = frozenset(layer_one.resident_gpu_expert_ids)
        expected_gpu_ids = tuple(
            expert_id
            for expert_id in probe.selected_expert_ids
            if expert_id in resident_ids
        )
        expected_cpu_ids = tuple(
            expert_id
            for expert_id in probe.selected_expert_ids
            if expert_id not in resident_ids
        )
        if (
            probe.global_expert_mask_sha256 != global_mask_sha256
            or probe.cpu_expert_ids != expected_cpu_ids
            or probe.gpu_expert_ids != expected_gpu_ids
            or not probe.cpu_expert_ids
        ):
            raise ValueError("layer-one expert route does not match its resident mask")
        is_cpu_control = (
            self.runtime.target_profile
            == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        )
        if is_cpu_control:
            if probe.gpu_expert_ids or probe.gpu_forward_count != 0:
                raise ValueError("CPU control receipt executed a GPU expert route")
        elif not probe.gpu_expert_ids or probe.gpu_forward_count != 2:
            raise ValueError("mixed receipt did not execute both CPU and GPU routes")

        mask_hashes = (
            self.short_forward.extend.global_expert_mask_sha256_before,
            self.short_forward.extend.global_expert_mask_sha256_after,
            self.short_forward.decode.global_expert_mask_sha256_before,
            self.short_forward.decode.global_expert_mask_sha256_after,
        )
        if any(mask_sha256 != global_mask_sha256 for mask_sha256 in mask_hashes):
            raise ValueError("global expert mask changed during the short forward")
        return self


@final
class SglangKtModelRuntimeValidationReceiptObservation(_StrictModel):
    """File-bound, derived facts from one admitted GLM-4.7 model validation."""

    receipt_path: AbsoluteRuntimePath
    receipt_size_bytes: PositiveInt
    receipt_sha256: Sha256Digest
    schema_version: Literal[1]
    generated_at_utc: NonemptyText
    validator_sha256: Sha256Digest
    process_spec_sha256: Sha256Digest
    model_contract_path: AbsoluteRuntimePath
    model_contract_receipt_sha256: Sha256Digest
    model_contract_sha256: Sha256Digest
    model_config_sha256: Sha256Digest
    model_index_sha256: Sha256Digest
    kernel_runtime_validation_receipt_path: AbsoluteRuntimePath
    kernel_runtime_validation_receipt_sha256: Sha256Digest
    target_profile: SglangKtTargetProfile
    gpu_uuid: GpuUuid
    gpu_compute_capability: tuple[PositiveInt, ResourceIndex]
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    executed_cpu_backend: Literal["AMX_BF16"]
    model_id: ModelId
    model_revision: GitRevision
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    transformers_distribution_version: NonemptyText
    transformers_module_version: NonemptyText
    torch_version: NonemptyText
    cuda_version: NonemptyText
    sgl_kernel_build_id: Sha256Digest
    deep_gemm_build_id: Sha256Digest
    kt_kernel_build_id: Sha256Digest
    ktransformers_method: Literal["BF16"]
    resident_gpu_experts: ResourceIndex
    attention_backend: Literal["flashinfer"]
    kv_cache_dtype: Literal["bfloat16"]
    max_total_tokens: PositiveInt
    static_memory_fraction: float
    capabilities: tuple[SglangKtModelRuntimeCapability, ...]
    ktransformers_wrapped_expert_layers: tuple[ResourceIndex, ...]
    global_expert_mask_sha256: Sha256Digest
    layer_one_selected_expert_ids: tuple[ExpertId, ...]
    extend_logits_sha256: Sha256Digest
    decode_logits_sha256: Sha256Digest


def _validate_expected_sha256(value: str, description: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise SglangKtModelRuntimeValidationReceiptError(
            f"expected {description} SHA-256 is invalid"
        )


def _derive_capabilities(
    receipt: _PassedModelRuntimeValidationReceiptV1,
) -> tuple[SglangKtModelRuntimeCapability, ...]:
    route_capability: SglangKtModelRuntimeCapability
    if (
        receipt.runtime.target_profile
        == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    ):
        route_capability = "glm47_flash_bf16_cpu_routed_experts_executed_v1"
    else:
        route_capability = "kt_bf16_cpu_gpu_hybrid_executed_v1"
    return (*_COMMON_CAPABILITIES, route_capability)


def load_sglang_kt_model_runtime_validation_receipt(
    path: Path,
    *,
    expected_validator_sha256: str,
    expected_process_spec_sha256: str,
    expected_model_contract_receipt_sha256: str,
    expected_kernel_receipt_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> SglangKtModelRuntimeValidationReceiptObservation:
    """Load a stable v1 receipt and bind it to independently supplied parents."""

    for value, description in (
        (expected_validator_sha256, "validator"),
        (expected_process_spec_sha256, "process spec"),
        (expected_model_contract_receipt_sha256, "model contract receipt"),
        (expected_kernel_receipt_sha256, "kernel receipt"),
    ):
        _validate_expected_sha256(value, description)
    if expected_receipt_sha256 is not None:
        _validate_expected_sha256(expected_receipt_sha256, "model receipt")

    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=MODEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES,
        )
        if (
            expected_receipt_sha256 is not None
            and bound_file.sha256 != expected_receipt_sha256
        ):
            raise SglangKtModelRuntimeValidationReceiptError(
                "model receipt does not match the expected SHA-256"
            )
        parse_sglang_kt_strict_json(bound_file.contents)
        receipt = _PassedModelRuntimeValidationReceiptV1.model_validate_json(
            bound_file.contents
        )
    except SglangKtModelRuntimeValidationReceiptError:
        raise
    except (RecursionError, SglangKtReceiptFileError, ValidationError) as error:
        raise SglangKtModelRuntimeValidationReceiptError(
            f"invalid SGLang-KTransformers model runtime receipt: {path}"
        ) from error

    parents = receipt.parents
    if receipt.validator_sha256 != expected_validator_sha256:
        raise SglangKtModelRuntimeValidationReceiptError(
            "model receipt validator does not match"
        )
    if parents.process_spec_sha256 != expected_process_spec_sha256:
        raise SglangKtModelRuntimeValidationReceiptError(
            "model receipt process-spec parent does not match"
        )
    if parents.model_contract.receipt_sha256 != expected_model_contract_receipt_sha256:
        raise SglangKtModelRuntimeValidationReceiptError(
            "model receipt model-contract parent does not match"
        )
    if (
        parents.kernel_runtime_validation.receipt_sha256
        != expected_kernel_receipt_sha256
    ):
        raise SglangKtModelRuntimeValidationReceiptError(
            "model receipt kernel parent does not match"
        )

    runtime = receipt.runtime
    contract = parents.model_contract
    kernel = parents.kernel_runtime_validation
    return SglangKtModelRuntimeValidationReceiptObservation(
        receipt_path=str(bound_file.path),
        receipt_size_bytes=len(bound_file.contents),
        receipt_sha256=bound_file.sha256,
        schema_version=MODEL_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        generated_at_utc=receipt.generated_at_utc,
        validator_sha256=receipt.validator_sha256,
        process_spec_sha256=parents.process_spec_sha256,
        model_contract_path=contract.path,
        model_contract_receipt_sha256=contract.receipt_sha256,
        model_contract_sha256=contract.contract_sha256,
        model_config_sha256=runtime.model_config_sha256,
        model_index_sha256=contract.index_sha256,
        kernel_runtime_validation_receipt_path=kernel.receipt_path,
        kernel_runtime_validation_receipt_sha256=kernel.receipt_sha256,
        target_profile=runtime.target_profile,
        gpu_uuid=runtime.gpu_uuid,
        gpu_compute_capability=runtime.gpu_compute_capability,
        cpu_cores=runtime.cpu_cores,
        memory_nodes=runtime.memory_nodes,
        executed_cpu_backend=runtime.executed_cpu_backend,
        model_id=runtime.model_id,
        model_revision=runtime.model_revision,
        sglang_revision=runtime.sglang_revision,
        ktransformers_revision=runtime.ktransformers_revision,
        transformers_distribution_version=(runtime.transformers_distribution_version),
        transformers_module_version=runtime.transformers_module_version,
        torch_version=runtime.torch_version,
        cuda_version=runtime.cuda_version,
        sgl_kernel_build_id=runtime.sgl_kernel_build_id,
        deep_gemm_build_id=runtime.deep_gemm_build_id,
        kt_kernel_build_id=runtime.kt_kernel_build_id,
        ktransformers_method=runtime.ktransformers_method,
        resident_gpu_experts=runtime.resident_gpu_experts,
        attention_backend=runtime.attention_backend,
        kv_cache_dtype=runtime.kv_cache_dtype,
        max_total_tokens=runtime.max_total_tokens,
        static_memory_fraction=runtime.static_memory_fraction,
        capabilities=_derive_capabilities(receipt),
        ktransformers_wrapped_expert_layers=_EXPECTED_WRAPPED_LAYERS,
        global_expert_mask_sha256=(receipt.wrapper_coverage.global_expert_mask_sha256),
        layer_one_selected_expert_ids=(
            receipt.layer_one_expert_probe.selected_expert_ids
        ),
        extend_logits_sha256=receipt.short_forward.extend.logits_sha256,
        decode_logits_sha256=receipt.short_forward.decode.logits_sha256,
    )
