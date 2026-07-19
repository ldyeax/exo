import hashlib
import math
import os
import re
from datetime import datetime
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

from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    HcaDevice,
    NetworkPort,
    ResourceIndex,
    SglangKtTargetProfile,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)

WARM_SERVING_RUN_RECEIPT_SCHEMA_VERSION = 1
WARM_SERVING_RUN_RECEIPT_MAXIMUM_BYTES = 4 * 1024 * 1024
WARM_SERVING_RUN_IDENTITY_CANONICALIZATION = (
    "exo-sglang-kt-warm-serving-run-identity-v1"
)
WARM_SERVING_MINIMUM_WARMUPS = 2
WARM_SERVING_MINIMUM_SAMPLES = 3
GLM_4_7_FLASH_PREFILL_INPUT_TOKENS = 1_024
GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS = 32
GLM_4_7_FLASH_DECODE_INPUT_TOKENS = 128
GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS = 128
GLM_4_7_FLASH_SERVING_SAMPLING_SEED = 20_260_719
GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256 = (
    "ebd3e87b9680a8b3e4bb594b8c2cda63eaead72fd478187bbaa3e2d3f3fadcac"
)
GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256 = (
    "48076354a27d82810c17e0ebb60123c10ee5e8fbde96e8ce53724e1f01fdbc21"
)
GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION = "0.0.0.dev0"
SGLANG_KT_SERVING_SOURCE_BUNDLE_CANONICALIZATION = (
    "exo-sglang-kt-serving-source-bundle-v1"
)
SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH = "scripts/sglang_kt_glm47_serving_client.py"
SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH = (
    "src/exo/worker/sglang_kt/serving_benchmark_receipt.py"
)
SGLANG_KT_SERVING_REQUIRED_SOURCE_PATHS = frozenset(
    (
        SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
        SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH,
    )
)
SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES = 256 * 1024
SGLANG_KT_SERVING_SSE_EVENT_SLACK = 8
SGLANG_KT_SERVING_SSE_LINES_PER_TOKEN_LIMIT = 4

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

NonemptyText = Annotated[str, StringConstraints(min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonnegativeFiniteFloat = Annotated[float, Field(ge=0.0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0.0)]
TokenId = Annotated[int, Field(ge=0, lt=154_880)]
ServingWorkloadKind = Literal["prefill", "decode"]


class SglangKtWarmServingRunReceiptError(ValueError):
    """Raised when warm-serving evidence is not admissible or not bound."""


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
        raise ValueError("serving receipt paths must be normalized absolute paths")
    return value


def _validate_relative_normalized_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or path.is_absolute()
        or not path.parts
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("source paths must be normalized relative POSIX paths")
    return value


def _validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("serving timing values must be finite")
    return value


def _validate_generated_at_utc(value: str) -> str:
    try:
        generated_at = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("generated_at_utc is not an ISO-8601 timestamp") from error
    utc_offset = generated_at.utcoffset()
    if generated_at.tzinfo is None or utc_offset is None:
        raise ValueError("generated_at_utc must be timezone aware")
    if utc_offset.total_seconds() != 0:
        raise ValueError("generated_at_utc must use UTC")
    return value


def calculate_sglang_kt_token_ids_sha256(token_ids: tuple[TokenId, ...]) -> str:
    """Hash token IDs using the same canonical JSON encoding as receipts."""

    if not token_ids or any(
        token_id < 0 or token_id >= 154_880 for token_id in token_ids
    ):
        raise ValueError("token IDs must be nonempty and inside the GLM vocabulary")
    return hashlib.sha256(canonical_sglang_kt_json(list(token_ids))).hexdigest()


def calculate_sglang_kt_length_finish_reason_sha256(completion_tokens: int) -> str:
    """Hash the exact length finish reason required by canonical workloads."""

    if completion_tokens <= 0:
        raise ValueError("completion token count must be positive")
    return hashlib.sha256(
        canonical_sglang_kt_json(
            {"type": "length", "length": completion_tokens},
        )
    ).hexdigest()


@final
class SglangKtServingFileIdentity(_StrictModel):
    path: AbsoluteRuntimePath
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)


@final
class SglangKtServingAdmissionBinding(_StrictModel):
    model_runtime_validation_receipt: SglangKtServingFileIdentity
    kernel_runtime_validation_receipt: SglangKtServingFileIdentity
    model_contract_receipt: SglangKtServingFileIdentity


@final
class SglangKtServingRuntimeIdentity(_StrictModel):
    executable: AbsoluteRuntimePath
    runtime_build_receipt: SglangKtServingFileIdentity
    runtime_build_id: Sha256Digest
    python_version: NonemptyText
    torch_version: NonemptyText
    cuda_version: NonemptyText
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    sgl_kernel_build_id: Sha256Digest
    deep_gemm_build_id: Sha256Digest
    kt_kernel_build_id: Sha256Digest

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)

    @model_validator(mode="after")
    def validate_pinned_sources(self) -> "SglangKtServingRuntimeIdentity":
        if (
            self.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
            or self.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        ):
            raise ValueError("serving runtime source revisions are not pinned")
        return self


@final
class SglangKtServingModelIdentity(_StrictModel):
    model_id: ModelId
    model_revision: GitRevision
    model_path: AbsoluteRuntimePath
    model_config_sha256: Sha256Digest
    model_index_sha256: Sha256Digest
    physical_weight_bytes: PositiveInt

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)

    @model_validator(mode="after")
    def validate_pinned_model(self) -> "SglangKtServingModelIdentity":
        if (
            self.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
            or self.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
            or self.model_config_sha256 != GLM_4_7_FLASH_BF16_CONFIG_SHA256
        ):
            raise ValueError("serving benchmark model identity is not pinned")
        return self


@final
class SglangKtServingProcessSpecIdentity(_StrictModel):
    receipt: SglangKtServingFileIdentity
    process_spec_sha256: Sha256Digest
    launch_argv_sha256: Sha256Digest
    launch_environment_sha256: Sha256Digest
    target_profile: SglangKtTargetProfile
    resident_gpu_experts: ResourceIndex
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]

    @model_validator(mode="after")
    def validate_process_spec(self) -> "SglangKtServingProcessSpecIdentity":
        if self.target_profile != GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE:
            raise ValueError("serving process spec is not the pinned baseline profile")
        if self.resident_gpu_experts < 1:
            raise ValueError("serving baseline requires resident GPU experts")
        if (
            not self.cpu_cores
            or self.cpu_cores != tuple(sorted(set(self.cpu_cores)))
            or not self.memory_nodes
            or self.memory_nodes != tuple(sorted(set(self.memory_nodes)))
        ):
            raise ValueError("serving CPU and memory-node bindings must be exact")
        return self


@final
class SglangKtServingTuningConfigIdentity(_StrictModel):
    relative_path: NonemptyText
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_normalized_path(value)


@final
class SglangKtServingTuningIdentity(_StrictModel):
    mode: Literal["untuned", "tuned"]
    config_directory: AbsoluteRuntimePath | None
    manifest_sha256: Sha256Digest | None
    config_files: tuple[SglangKtServingTuningConfigIdentity, ...]

    @field_validator("config_directory")
    @classmethod
    def validate_config_directory(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_absolute_normalized_path(value)

    @model_validator(mode="after")
    def validate_tuning_bundle(self) -> "SglangKtServingTuningIdentity":
        paths = tuple(item.relative_path for item in self.config_files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("tuning configuration paths must be sorted and unique")
        if self.mode == "untuned":
            if (
                self.config_directory is not None
                or self.manifest_sha256 is not None
                or self.config_files
            ):
                raise ValueError("untuned serving evidence cannot bind tuning files")
        elif (
            self.config_directory is None
            or self.manifest_sha256 is None
            or not self.config_files
        ):
            raise ValueError("tuned serving evidence requires a complete tuning bundle")
        return self


@final
class SglangKtServingClientIdentity(_StrictModel):
    source_file: SglangKtServingFileIdentity
    source_bundle_sha256: Sha256Digest
    protocol_version: Literal[1]
    http_library: NonemptyText
    http_library_version: NonemptyText
    request_timeout_seconds: PositiveFiniteFloat

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        return _validate_finite(value)


@final
class SglangKtServingSourceFileIdentity(_StrictModel):
    relative_path: NonemptyText
    size_bytes: PositiveInt
    sha256: Sha256Digest

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_normalized_path(value)


def calculate_sglang_kt_serving_source_bundle_sha256(
    files: tuple[SglangKtServingSourceFileIdentity, ...],
) -> str:
    """Hash an ordered serving source bundle independently of its checkout path."""

    paths = tuple(file.relative_path for file in files)
    if not files or paths != tuple(sorted(set(paths))):
        raise ValueError("serving source files must be nonempty, sorted, and unique")
    payload = {
        "canonicalization": SGLANG_KT_SERVING_SOURCE_BUNDLE_CANONICALIZATION,
        "files": [file.model_dump(mode="json") for file in files],
    }
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()


@final
class SglangKtServingSourceIdentity(_StrictModel):
    repository_root: AbsoluteRuntimePath
    commit: NonemptyText
    source_bundle_sha256: Sha256Digest
    files: tuple[SglangKtServingSourceFileIdentity, ...]
    dirty_files: tuple[SglangKtServingSourceFileIdentity, ...]

    @field_validator("repository_root")
    @classmethod
    def validate_repository_root(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)

    @field_validator("commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if _GIT_COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("source commit must be a full lowercase Git commit")
        return value

    @model_validator(mode="after")
    def validate_source_files(self) -> "SglangKtServingSourceIdentity":
        file_paths = tuple(item.relative_path for item in self.files)
        if (
            not self.files
            or file_paths != tuple(sorted(set(file_paths)))
            or not SGLANG_KT_SERVING_REQUIRED_SOURCE_PATHS.issubset(file_paths)
        ):
            raise ValueError(
                "source bundle must contain sorted unique serving implementation files"
            )
        if self.source_bundle_sha256 != (
            calculate_sglang_kt_serving_source_bundle_sha256(self.files)
        ):
            raise ValueError("serving source bundle SHA-256 does not match its files")

        dirty_paths = tuple(item.relative_path for item in self.dirty_files)
        if dirty_paths != tuple(sorted(set(dirty_paths))):
            raise ValueError("dirty source paths must be sorted and unique")
        files_by_path = {item.relative_path: item for item in self.files}
        if any(
            files_by_path.get(item.relative_path) != item for item in self.dirty_files
        ):
            raise ValueError("dirty source files must be exact source-bundle members")
        return self


@final
class SglangKtServingTopologyStage(_StrictModel):
    pipeline_rank: ResourceIndex
    node_id: NodeId
    host: NonemptyText
    port: NetworkPort
    gpu_uuid: GpuUuid
    hca_devices: tuple[HcaDevice, ...]

    @model_validator(mode="after")
    def validate_hca_devices(self) -> "SglangKtServingTopologyStage":
        if self.hca_devices != tuple(sorted(set(self.hca_devices))):
            raise ValueError("topology HCA devices must be sorted and unique")
        return self


@final
class SglangKtServingTopologyIdentity(_StrictModel):
    deployment: Literal["local"]
    interconnect: Literal["none"]
    stages: tuple[SglangKtServingTopologyStage, ...]

    @model_validator(mode="after")
    def validate_topology(self) -> "SglangKtServingTopologyIdentity":
        if (
            len(self.stages) != 1
            or self.stages[0].pipeline_rank != 0
            or self.stages[0].hca_devices
        ):
            raise ValueError(
                "schema-v1 serving topology admits one local rank without an HCA"
            )
        return self


@final
class SglangKtServingServerInfoIdentity(_StrictModel):
    node_id: NodeId
    host: NonemptyText
    port: NetworkPort
    canonical_response_sha256: Sha256Digest
    version: Literal["0.0.0.dev0"]
    model_path: AbsoluteRuntimePath
    tp_size: Literal[1]
    pp_size: Literal[1]
    nnodes: Literal[1]
    node_rank: Literal[0]
    disable_radix_cache: Literal[True]

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)


@final
class SglangKtWarmServingRunIdentity(_StrictModel):
    admission: SglangKtServingAdmissionBinding
    runtime: SglangKtServingRuntimeIdentity
    model: SglangKtServingModelIdentity
    process_spec: SglangKtServingProcessSpecIdentity
    tuning: SglangKtServingTuningIdentity
    client: SglangKtServingClientIdentity
    source: SglangKtServingSourceIdentity
    topology: SglangKtServingTopologyIdentity
    server_info: tuple[SglangKtServingServerInfoIdentity, ...]

    @model_validator(mode="after")
    def validate_cross_bindings(self) -> "SglangKtWarmServingRunIdentity":
        if self.tuning.mode != "untuned":
            raise ValueError(
                "schema-v1 serving identity only admits the untuned baseline"
            )
        if len(self.server_info) != 1:
            raise ValueError(
                "schema-v1 serving identity requires rank-zero server_info"
            )
        info = self.server_info[0]
        stage = self.topology.stages[0]
        if (
            (info.host, info.port) != (stage.host, stage.port)
            or info.node_id != stage.node_id
            or info.model_path != self.model.model_path
        ):
            raise ValueError("server_info does not describe the bound local rank zero")

        expected_client_path = PurePosixPath(self.source.repository_root).joinpath(
            SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH
        )
        source_members = {member.relative_path: member for member in self.source.files}
        client_member = source_members.get(SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH)
        if (
            self.client.source_bundle_sha256 != self.source.source_bundle_sha256
            or self.client.source_file.path != expected_client_path.as_posix()
            or client_member is None
            or self.client.source_file.size_bytes != client_member.size_bytes
            or self.client.source_file.sha256 != client_member.sha256
        ):
            raise ValueError("serving client is not an exact source-bundle member")
        return self


def calculate_sglang_kt_warm_serving_run_identity_sha256(
    identity: SglangKtWarmServingRunIdentity,
) -> str:
    payload = {
        "canonicalization": WARM_SERVING_RUN_IDENTITY_CANONICALIZATION,
        "identity": identity.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()


@final
class SglangKtServingSetupEvidence(_StrictModel):
    process_launch_seconds: NonnegativeFiniteFloat
    health_ready_seconds: NonnegativeFiniteFloat
    admission_seconds: NonnegativeFiniteFloat
    server_info_fetch_seconds: NonnegativeFiniteFloat
    health_generate_status_code: Literal[200]
    health_generate_response_sha256: Sha256Digest

    @field_validator(
        "process_launch_seconds",
        "health_ready_seconds",
        "admission_seconds",
        "server_info_fetch_seconds",
    )
    @classmethod
    def validate_timing(cls, value: float) -> float:
        return _validate_finite(value)


@final
class SglangKtServingJitCacheEvidence(_StrictModel):
    cache_directories: tuple[AbsoluteRuntimePath, ...]
    after_penultimate_warmup_manifest_sha256: Sha256Digest
    after_final_warmup_manifest_sha256: Sha256Digest
    after_measurement_manifest_sha256: Sha256Digest

    @field_validator("cache_directories")
    @classmethod
    def validate_cache_directories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("JIT cache directories must be sorted and unique")
        for path in value:
            _validate_absolute_normalized_path(path)
        return value

    @model_validator(mode="after")
    def validate_stability(self) -> "SglangKtServingJitCacheEvidence":
        hashes = (
            self.after_penultimate_warmup_manifest_sha256,
            self.after_final_warmup_manifest_sha256,
            self.after_measurement_manifest_sha256,
        )
        if len(set(hashes)) != 1:
            raise ValueError("JIT cache changed after warmup or during measurement")
        return self


@final
class SglangKtServingWorkloadRequest(_StrictModel):
    kind: ServingWorkloadKind
    input_token_count: PositiveInt
    input_ids_sha256: Sha256Digest
    max_new_tokens: PositiveInt
    sampling_seed: int
    temperature: float
    ignore_eos: Literal[True]
    stream: Literal[True]
    return_logprob: Literal[False]
    log_metrics: Literal[True]

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value != 0.0:
            raise ValueError("warm-serving requests require greedy temperature zero")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> "SglangKtServingWorkloadRequest":
        expected = {
            "prefill": (
                GLM_4_7_FLASH_PREFILL_INPUT_TOKENS,
                GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS,
                GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256,
            ),
            "decode": (
                GLM_4_7_FLASH_DECODE_INPUT_TOKENS,
                GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS,
                GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256,
            ),
        }[self.kind]
        if (
            self.input_token_count,
            self.max_new_tokens,
            self.input_ids_sha256,
        ) != expected or self.sampling_seed != GLM_4_7_FLASH_SERVING_SAMPLING_SEED:
            raise ValueError("workload request is not schema-v1 canonical")
        return self


@final
class SglangKtServingInvocationEvidence(_StrictModel):
    ordinal: PositiveInt
    input_ids_sha256: Sha256Digest
    cache_flush_status_code: Literal[200]
    cache_flush_response_sha256: Sha256Digest
    prompt_tokens: PositiveInt
    completion_tokens: PositiveInt
    cached_tokens: ResourceIndex
    output_ids_sha256: Sha256Digest
    finish_reason_sha256: Sha256Digest
    stream_line_count: PositiveInt
    stream_event_count: PositiveInt
    output_bearing_event_count: PositiveInt
    maximum_stream_line_bytes: PositiveInt
    first_stream_event_output_tokens: PositiveInt
    total_client_seconds: PositiveFiniteFloat
    client_observed_ttft_seconds: PositiveFiniteFloat
    client_observed_generation_window_seconds: PositiveFiniteFloat
    client_observed_decode_tokens_per_second: PositiveFiniteFloat
    ttft_semantics: Literal[
        "client_stream_first_output_event_including_http_and_queue_v1"
    ]

    @field_validator(
        "total_client_seconds",
        "client_observed_ttft_seconds",
        "client_observed_generation_window_seconds",
        "client_observed_decode_tokens_per_second",
    )
    @classmethod
    def validate_timing(cls, value: float) -> float:
        return _validate_finite(value)

    @model_validator(mode="after")
    def validate_client_timings(self) -> "SglangKtServingInvocationEvidence":
        if (
            self.client_observed_ttft_seconds > self.total_client_seconds
            or self.client_observed_generation_window_seconds
            > self.total_client_seconds
            or self.client_observed_ttft_seconds
            + self.client_observed_generation_window_seconds
            > self.total_client_seconds
            or self.first_stream_event_output_tokens > self.completion_tokens
            or self.output_bearing_event_count < 2
            or self.output_bearing_event_count > self.stream_event_count
            or self.stream_event_count
            > self.completion_tokens + SGLANG_KT_SERVING_SSE_EVENT_SLACK
            or self.stream_line_count
            > (self.completion_tokens + 2) * SGLANG_KT_SERVING_SSE_LINES_PER_TOKEN_LIMIT
            or self.stream_line_count < self.stream_event_count + 1
            or self.maximum_stream_line_bytes > SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES
            or self.finish_reason_sha256
            != calculate_sglang_kt_length_finish_reason_sha256(self.completion_tokens)
        ):
            raise ValueError("streaming client timings are internally inconsistent")
        expected_decode_rate = (
            self.completion_tokens - self.first_stream_event_output_tokens
        ) / self.client_observed_generation_window_seconds
        if not math.isclose(
            self.client_observed_decode_tokens_per_second,
            expected_decode_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("decode rate does not match the client event window")
        return self


@final
class SglangKtServingWorkloadEvidence(_StrictModel):
    request: SglangKtServingWorkloadRequest
    warmups: tuple[SglangKtServingInvocationEvidence, ...]
    samples: tuple[SglangKtServingInvocationEvidence, ...]

    @model_validator(mode="after")
    def validate_repetitions(self) -> "SglangKtServingWorkloadEvidence":
        if len(self.warmups) < WARM_SERVING_MINIMUM_WARMUPS:
            raise ValueError("warm-serving evidence requires at least two warmups")
        if len(self.samples) < WARM_SERVING_MINIMUM_SAMPLES:
            raise ValueError("warm-serving evidence requires at least three samples")
        if tuple(item.ordinal for item in self.warmups) != tuple(
            range(1, len(self.warmups) + 1)
        ) or tuple(item.ordinal for item in self.samples) != tuple(
            range(1, len(self.samples) + 1)
        ):
            raise ValueError("warmup and sample ordinals must each be contiguous")

        invocations = (*self.warmups, *self.samples)
        request = self.request
        if any(
            invocation.input_ids_sha256 != request.input_ids_sha256
            or invocation.prompt_tokens != request.input_token_count
            or invocation.completion_tokens != request.max_new_tokens
            or invocation.cached_tokens != 0
            for invocation in invocations
        ):
            raise ValueError("serving invocation does not match its uncached request")
        if len({item.output_ids_sha256 for item in invocations}) != 1:
            raise ValueError("warm-serving output token IDs are not deterministic")
        if len({item.finish_reason_sha256 for item in invocations}) != 1:
            raise ValueError("warm-serving finish reasons are not deterministic")
        return self


@final
class SglangKtServingOwnedServerProcessIdentity(_StrictModel):
    pid: PositiveInt
    proc_start_time_ticks: PositiveInt
    executable: AbsoluteRuntimePath
    argv_sha256: Sha256Digest

    @field_validator("executable")
    @classmethod
    def validate_executable(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)


@final
class SglangKtServingCleanupEvidence(_StrictModel):
    benchmark_completed_normally: Literal[True]
    server_process: SglangKtServingOwnedServerProcessIdentity
    termination_signal: Literal["SIGTERM"]
    server_return_code: Literal[-15, 0]
    forced: Literal[False]
    owned_processes_absent: Literal[True]
    service_host: NonemptyText
    service_port: NetworkPort
    service_port_clear: Literal[True]
    gpu_uuid: GpuUuid
    gpu_process_clear: Literal[True]
    delegated_cgroup_path: AbsoluteRuntimePath
    delegated_cgroup_removed: Literal[True]
    transient_unit_name: NonemptyText
    transient_unit_removed: Literal[True]
    cleanup_completed_at_utc: NonemptyText
    lease_cleanup_scope: Literal["outer_benchmark_wrapper"]

    @field_validator("delegated_cgroup_path")
    @classmethod
    def validate_cgroup_path(cls, value: str) -> str:
        return _validate_absolute_normalized_path(value)

    @field_validator("cleanup_completed_at_utc")
    @classmethod
    def validate_cleanup_timestamp(cls, value: str) -> str:
        return _validate_generated_at_utc(value)


@final
class WarmServingRunReceiptV1(_StrictModel):
    schema_version: Literal[1]
    status: Literal["passed"]
    generated_at_utc: NonemptyText
    evidence_class: Literal["diagnostic", "performance"]
    performance_comparable: bool
    profiler: Literal["none"]
    instrumentation: Literal["none", "debug_timing"]
    radix_cache_disabled: Literal[True]
    max_concurrent_requests: Literal[1]
    identity_sha256: Sha256Digest
    identity: SglangKtWarmServingRunIdentity
    setup: SglangKtServingSetupEvidence
    jit_cache: SglangKtServingJitCacheEvidence
    workloads: tuple[SglangKtServingWorkloadEvidence, ...]
    cleanup: SglangKtServingCleanupEvidence | None

    @field_validator("generated_at_utc")
    @classmethod
    def validate_generated_at_utc(cls, value: str) -> str:
        return _validate_generated_at_utc(value)

    @model_validator(mode="after")
    def validate_complete_run(self) -> "WarmServingRunReceiptV1":
        expected_identity_sha256 = calculate_sglang_kt_warm_serving_run_identity_sha256(
            self.identity
        )
        if self.identity_sha256 != expected_identity_sha256:
            raise ValueError("warm-serving identity SHA-256 does not match")
        if self.performance_comparable != (self.evidence_class == "performance"):
            raise ValueError("performance comparison flag contradicts evidence class")
        if self.evidence_class == "performance" and self.instrumentation != "none":
            raise ValueError("performance evidence cannot enable debug instrumentation")
        if self.performance_comparable and self.cleanup is None:
            raise ValueError("performance evidence requires completed cleanup")
        if tuple(workload.request.kind for workload in self.workloads) != (
            "prefill",
            "decode",
        ):
            raise ValueError("receipt requires ordered prefill and decode workloads")
        if self.cleanup is not None:
            stage = self.identity.topology.stages[0]
            if (
                self.cleanup.server_process.executable
                != self.identity.runtime.executable
                or self.cleanup.server_process.argv_sha256
                != self.identity.process_spec.launch_argv_sha256
                or (self.cleanup.service_host, self.cleanup.service_port)
                != (stage.host, stage.port)
                or self.cleanup.gpu_uuid != stage.gpu_uuid
            ):
                raise ValueError("cleanup does not bind the owned local server")
            if datetime.fromisoformat(self.generated_at_utc) <= datetime.fromisoformat(
                self.cleanup.cleanup_completed_at_utc
            ):
                raise ValueError(
                    "performance receipt must be generated after cleanup completes"
                )
        timeout = self.identity.client.request_timeout_seconds
        if any(
            invocation.total_client_seconds > timeout
            for workload in self.workloads
            for invocation in (*workload.warmups, *workload.samples)
        ):
            raise ValueError("serving invocation exceeded the bound client timeout")
        return self


@final
class SglangKtWarmServingRunReceiptObservation(_StrictModel):
    receipt_path: AbsoluteRuntimePath
    receipt_size_bytes: PositiveInt
    receipt_sha256: Sha256Digest
    receipt: WarmServingRunReceiptV1


def _validate_expected_sha256(value: str, description: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise SglangKtWarmServingRunReceiptError(
            f"expected {description} SHA-256 is invalid"
        )


def _validate_warm_serving_run_receipt_contents(
    contents: bytes,
) -> WarmServingRunReceiptV1:
    parse_sglang_kt_strict_json(contents)
    return WarmServingRunReceiptV1.model_validate_json(contents)


def canonicalize_sglang_kt_warm_serving_run_receipt(payload: object) -> bytes:
    """Validate a warm-serving v1 receipt and return canonical JSON bytes."""

    try:
        contents = canonical_sglang_kt_json(payload)
        if len(contents) > WARM_SERVING_RUN_RECEIPT_MAXIMUM_BYTES:
            raise SglangKtReceiptFileError(
                "warm-serving receipt exceeds the maximum receipt size"
            )
        _validate_warm_serving_run_receipt_contents(contents)
    except (RecursionError, SglangKtReceiptFileError, ValidationError) as error:
        raise SglangKtWarmServingRunReceiptError(
            "invalid SGLang-KTransformers warm-serving receipt payload"
        ) from error
    return contents


def load_sglang_kt_warm_serving_run_receipt(
    path: Path,
    *,
    expected_identity_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> SglangKtWarmServingRunReceiptObservation:
    """Load a stable receipt and bind it to an independently supplied identity."""

    _validate_expected_sha256(expected_identity_sha256, "serving identity")
    if expected_receipt_sha256 is not None:
        _validate_expected_sha256(expected_receipt_sha256, "serving receipt")
    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=WARM_SERVING_RUN_RECEIPT_MAXIMUM_BYTES,
        )
        if (
            expected_receipt_sha256 is not None
            and bound_file.sha256 != expected_receipt_sha256
        ):
            raise SglangKtWarmServingRunReceiptError(
                "warm-serving receipt does not match the expected SHA-256"
            )
        receipt = _validate_warm_serving_run_receipt_contents(bound_file.contents)
    except SglangKtWarmServingRunReceiptError:
        raise
    except (RecursionError, SglangKtReceiptFileError, ValidationError) as error:
        raise SglangKtWarmServingRunReceiptError(
            f"invalid SGLang-KTransformers warm-serving receipt: {path}"
        ) from error
    if receipt.identity_sha256 != expected_identity_sha256:
        raise SglangKtWarmServingRunReceiptError(
            "warm-serving receipt identity does not match"
        )
    return SglangKtWarmServingRunReceiptObservation(
        receipt_path=str(bound_file.path),
        receipt_size_bytes=len(bound_file.contents),
        receipt_sha256=bound_file.sha256,
        receipt=receipt,
    )
