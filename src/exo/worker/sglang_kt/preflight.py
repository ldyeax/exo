from typing import Annotated, Literal, final

from pydantic import PositiveInt, StringConstraints, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    HcaDevice,
    KTransformersMethod,
    ResourceIndex,
    SglangKtTargetProfile,
    StaticMemoryFraction,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KV_CACHE_DTYPE,
    GLM_4_7_FLASH_TARGET_PROFILE,
    GLM_4_7_FLASH_TARGET_PROFILES,
    GLM_5_2_KV_CACHE_DTYPE,
    SglangKtProcessLaunchSpec,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)

ObservedText = Annotated[str, StringConstraints(min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SglangKtRuntimeCapability = Literal[
    "kt_tp_group_local_broadcast_v1",
    "glm52_nsa_sm86_short_forward_v1",
    "kt_physical_numa_mapping_v1",
    "kt_process_cpu_affinity_v1",
    "kt_fp8_amx_executed_v1",
    "glm47_flash_kt_wrapper_active_v1",
    "glm47_flash_kt_wrapper_layers_1_46_v1",
    "glm47_flash_bf16_sm86_short_forward_v1",
    "kt_bf16_amx_executed_v1",
    "kt_bf16_cpu_gpu_hybrid_executed_v1",
    "glm47_flash_bf16_cpu_routed_experts_executed_v1",
]
GLM_5_2_REQUIRED_RUNTIME_CAPABILITIES: frozenset[SglangKtRuntimeCapability] = frozenset(
    (
        "kt_tp_group_local_broadcast_v1",
        "glm52_nsa_sm86_short_forward_v1",
        "kt_physical_numa_mapping_v1",
        "kt_process_cpu_affinity_v1",
        "kt_fp8_amx_executed_v1",
    )
)
GLM_4_7_FLASH_REQUIRED_RUNTIME_CAPABILITIES: frozenset[SglangKtRuntimeCapability] = (
    frozenset(
        (
            "glm47_flash_kt_wrapper_active_v1",
            "glm47_flash_kt_wrapper_layers_1_46_v1",
            "glm47_flash_bf16_sm86_short_forward_v1",
            "kt_physical_numa_mapping_v1",
            "kt_process_cpu_affinity_v1",
            "kt_bf16_amx_executed_v1",
            "kt_bf16_cpu_gpu_hybrid_executed_v1",
        )
    )
)
GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_REQUIRED_RUNTIME_CAPABILITIES: frozenset[
    SglangKtRuntimeCapability
] = frozenset(
    (
        "glm47_flash_kt_wrapper_active_v1",
        "glm47_flash_kt_wrapper_layers_1_46_v1",
        "glm47_flash_bf16_sm86_short_forward_v1",
        "kt_physical_numa_mapping_v1",
        "kt_process_cpu_affinity_v1",
        "kt_bf16_amx_executed_v1",
        "glm47_flash_bf16_cpu_routed_experts_executed_v1",
    )
)
GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS: tuple[ResourceIndex, ...] = tuple(range(1, 47))
# Backwards-compatible name for callers that only know the original GLM-5.2 profile.
REQUIRED_RUNTIME_CAPABILITIES = GLM_5_2_REQUIRED_RUNTIME_CAPABILITIES
PreflightCheck = Literal[
    "host_observation",
    "python_executable",
    "python_implementation",
    "python_version",
    "sglang_revision",
    "ktransformers_revision",
    "transformers_distribution_version",
    "transformers_module_version",
    "runtime_validation_receipt",
    "model_path",
    "ktransformers_weight_path",
    "model_revision_receipt",
    "ktransformers_weight_revision_receipt",
    "gpu_uuid",
    "cpu_cores",
    "memory_nodes",
    "hca_devices",
    "service_endpoint",
    "distributed_coordinator",
]


@final
class SglangKtPythonVersionObservation(FrozenModel):
    major: PositiveInt
    minor: ResourceIndex
    patch: ResourceIndex

    @property
    def release(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        return ".".join(str(component) for component in self.release)


@final
class SglangKtRuntimeObservation(FrozenModel):
    """Facts collected by invoking one planned external Python environment.

    Optional values represent a collector that could not establish that fact.
    The evaluator treats every missing value as a failed check.
    """

    executable: AbsoluteRuntimePath | None = None
    python_implementation: ObservedText | None = None
    python_version: SglangKtPythonVersionObservation | None = None
    sglang_revision: GitRevision | None = None
    ktransformers_revision: GitRevision | None = None
    transformers_distribution_version: ObservedText | None = None
    transformers_module_version: ObservedText | None = None
    torch_version: ObservedText | None = None
    cuda_version: ObservedText | None = None
    sgl_kernel_build_id: ObservedText | None = None
    deep_gemm_build_id: ObservedText | None = None
    kt_kernel_build_id: ObservedText | None = None


@final
class SglangKtRuntimeValidationReceiptObservation(FrozenModel):
    """Evidence from a target-specific runtime smoke test on one GPU.

    The version-only runtime probe never creates this receipt. A future
    validator must execute the named checks on the exact stack and checkpoint
    before the external process group can pass preflight.
    """

    target_profile: SglangKtTargetProfile
    gpu_uuid: GpuUuid
    gpu_compute_capability: tuple[PositiveInt, ResourceIndex]
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    executed_cpu_backend: Literal["AMX", "AMX_BF16"]
    model_id: ModelId
    model_revision: GitRevision
    model_config_sha256: Sha256Digest
    sglang_revision: GitRevision
    ktransformers_revision: GitRevision
    transformers_distribution_version: ObservedText
    transformers_module_version: ObservedText
    torch_version: ObservedText
    cuda_version: ObservedText
    sgl_kernel_build_id: ObservedText
    deep_gemm_build_id: ObservedText
    kt_kernel_build_id: ObservedText
    ktransformers_method: KTransformersMethod
    resident_gpu_experts: ResourceIndex
    attention_backend: Literal["flashinfer", "nsa"]
    kv_cache_dtype: Literal["bfloat16", "fp8_e4m3"]
    max_total_tokens: PositiveInt
    static_memory_fraction: StaticMemoryFraction
    capabilities: tuple[SglangKtRuntimeCapability, ...]
    ktransformers_wrapped_expert_layers: tuple[ResourceIndex, ...] = ()

    @model_validator(mode="after")
    def validate_capabilities(self) -> "SglangKtRuntimeValidationReceiptObservation":
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("runtime validation capabilities must be unique")
        if tuple(sorted(set(self.ktransformers_wrapped_expert_layers))) != (
            self.ktransformers_wrapped_expert_layers
        ):
            raise ValueError(
                "KTransformers wrapped expert layers must be sorted and unique"
            )
        if not self.cpu_cores or len(set(self.cpu_cores)) != len(self.cpu_cores):
            raise ValueError("validated runtime CPU cores must be nonempty and unique")
        if not self.memory_nodes or len(set(self.memory_nodes)) != len(
            self.memory_nodes
        ):
            raise ValueError(
                "validated runtime memory nodes must be nonempty and unique"
            )
        return self


@final
class SglangKtModelSnapshotReceiptObservation(FrozenModel):
    """Observed Exo revision receipt and snapshot completeness for one path."""

    model_path: AbsoluteRuntimePath
    model_id: ModelId
    revision: GitRevision
    weight_format: Literal["safetensors"]
    ktransformers_method: KTransformersMethod
    config_sha256: Sha256Digest
    full_indexer_layer_starts: tuple[ResourceIndex, ...]
    receipt_verified: bool
    snapshot_complete: bool
    contract_path: AbsoluteRuntimePath | None = None
    contract_receipt_sha256: Sha256Digest | None = None
    contract_sha256: Sha256Digest | None = None
    index_sha256: Sha256Digest | None = None
    weight_map_entries: PositiveInt | None = None
    shard_count: PositiveInt | None = None
    physical_weight_bytes: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_indexer_boundaries(self) -> "SglangKtModelSnapshotReceiptObservation":
        starts = self.full_indexer_layer_starts
        if not starts or starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError(
                "full_indexer_layer_starts must be sorted, unique, and begin at zero"
            )
        contract_evidence = (
            self.contract_path,
            self.contract_receipt_sha256,
            self.contract_sha256,
            self.index_sha256,
            self.weight_map_entries,
            self.shard_count,
            self.physical_weight_bytes,
        )
        if any(value is not None for value in contract_evidence) and any(
            value is None for value in contract_evidence
        ):
            raise ValueError("model contract evidence must be complete or absent")
        return self


@final
class SglangKtHostPreflightObservation(FrozenModel):
    """Pre-collected facts for one host; constructing this performs no probes.

    Directory, HCA, and port collections contain only resources the future
    collector established as usable for the planned launch.
    """

    node_id: NodeId
    runtime: SglangKtRuntimeObservation
    runtime_validation_receipts: tuple[
        SglangKtRuntimeValidationReceiptObservation, ...
    ] = ()
    kernel_runtime_validation_receipts: tuple[
        SglangKtKernelRuntimeValidationReceiptObservation, ...
    ] = ()
    readable_directories: tuple[AbsoluteRuntimePath, ...] = ()
    model_snapshot_receipts: tuple[SglangKtModelSnapshotReceiptObservation, ...] = ()
    gpu_uuids: tuple[GpuUuid, ...] = ()
    cpu_cores: tuple[ResourceIndex, ...] = ()
    memory_nodes: tuple[ResourceIndex, ...] = ()
    hca_devices: tuple[HcaDevice, ...] = ()
    available_bind_endpoints: tuple[Host, ...] = ()

    @model_validator(mode="after")
    def validate_unambiguous_facts(self) -> "SglangKtHostPreflightObservation":
        collections: tuple[tuple[str, tuple[object, ...]], ...] = (
            ("readable_directories", self.readable_directories),
            ("gpu_uuids", self.gpu_uuids),
            ("cpu_cores", self.cpu_cores),
            ("memory_nodes", self.memory_nodes),
            ("hca_devices", self.hca_devices),
        )
        for collection_name, values in collections:
            if len(set(values)) != len(values):
                raise ValueError(f"{collection_name} observations must be unique")

        endpoint_keys = tuple(
            (endpoint.ip, endpoint.port) for endpoint in self.available_bind_endpoints
        )
        if len(set(endpoint_keys)) != len(endpoint_keys):
            raise ValueError("available_bind_endpoints observations must be unique")

        receipt_paths = tuple(
            receipt.model_path for receipt in self.model_snapshot_receipts
        )
        if len(set(receipt_paths)) != len(receipt_paths):
            raise ValueError("model snapshot receipt paths must be unique")

        validation_gpu_uuids = tuple(
            receipt.gpu_uuid for receipt in self.runtime_validation_receipts
        )
        if len(set(validation_gpu_uuids)) != len(validation_gpu_uuids):
            raise ValueError("runtime validation receipt GPU UUIDs must be unique")
        kernel_validation_gpu_uuids = tuple(
            receipt.gpu_uuid for receipt in self.kernel_runtime_validation_receipts
        )
        if len(set(kernel_validation_gpu_uuids)) != len(kernel_validation_gpu_uuids):
            raise ValueError(
                "kernel runtime validation receipt GPU UUIDs must be unique"
            )
        return self


@final
class SglangKtRankAdmissionBinding(FrozenModel):
    """Immutable evidence that must still match immediately before launch."""

    pipeline_rank: ResourceIndex
    process_spec_sha256: Sha256Digest
    model_snapshot_receipts: tuple[SglangKtModelSnapshotReceiptObservation, ...] = ()
    model_runtime_validation_receipt: (
        SglangKtRuntimeValidationReceiptObservation | None
    ) = None
    kernel_runtime_validation_receipt: (
        SglangKtKernelRuntimeValidationReceiptObservation | None
    ) = None

    @model_validator(mode="after")
    def validate_unique_snapshot_paths(self) -> "SglangKtRankAdmissionBinding":
        paths = tuple(receipt.model_path for receipt in self.model_snapshot_receipts)
        if len(set(paths)) != len(paths):
            raise ValueError("admission model snapshot paths must be unique")
        return self


def validate_sglang_kt_rank_admission_binding(
    process_spec: SglangKtProcessLaunchSpec,
    binding: SglangKtRankAdmissionBinding,
) -> None:
    if (
        binding.pipeline_rank != process_spec.pipeline_rank
        or binding.process_spec_sha256
        != calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    ):
        raise ValueError("admission binding does not match the process spec digest")

    if process_spec.target_profile not in GLM_4_7_FLASH_TARGET_PROFILES:
        if (
            binding.model_snapshot_receipts
            or binding.model_runtime_validation_receipt is not None
            or binding.kernel_runtime_validation_receipt is not None
        ):
            raise ValueError("target profile does not admit GLM-4.7 evidence")
        return

    expected_snapshot_paths = {
        process_spec.model_path,
        process_spec.ktransformers_weight_path,
    }
    snapshot_receipts = binding.model_snapshot_receipts
    if {receipt.model_path for receipt in snapshot_receipts} != expected_snapshot_paths:
        raise ValueError("admission binding does not cover every model snapshot")
    expected_contract_sha256 = process_spec.expected_model_contract_sha256
    if expected_contract_sha256 is None or any(
        receipt.model_id != process_spec.model_id
        or receipt.revision != process_spec.expected_model_revision
        or receipt.ktransformers_method != process_spec.ktransformers_method
        or receipt.contract_receipt_sha256 != expected_contract_sha256
        or receipt.contract_sha256 != expected_contract_sha256
        or receipt.index_sha256 is None
        or receipt.weight_map_entries is None
        or receipt.shard_count is None
        or receipt.physical_weight_bytes is None
        or not receipt.receipt_verified
        or not receipt.snapshot_complete
        for receipt in snapshot_receipts
    ):
        raise ValueError("admission binding model contract evidence is inconsistent")

    model_runtime_receipt = binding.model_runtime_validation_receipt
    kernel_receipt = binding.kernel_runtime_validation_receipt
    expected_runtime_capabilities = (
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_REQUIRED_RUNTIME_CAPABILITIES
        if process_spec.target_profile
        == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        else GLM_4_7_FLASH_REQUIRED_RUNTIME_CAPABILITIES
    )
    expected_wrapped_layers = GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS
    if not (
        model_runtime_receipt is not None
        and model_runtime_receipt.target_profile == process_spec.target_profile
        and model_runtime_receipt.gpu_uuid == process_spec.gpu_uuid
        and model_runtime_receipt.gpu_compute_capability == (8, 6)
        and model_runtime_receipt.cpu_cores == process_spec.cpu_cores
        and model_runtime_receipt.memory_nodes == process_spec.memory_nodes
        and model_runtime_receipt.executed_cpu_backend == "AMX_BF16"
        and model_runtime_receipt.model_id == process_spec.model_id
        and model_runtime_receipt.model_revision == process_spec.expected_model_revision
        and model_runtime_receipt.model_config_sha256
        == snapshot_receipts[0].config_sha256
        and model_runtime_receipt.sglang_revision
        == process_spec.expected_sglang_revision
        and model_runtime_receipt.ktransformers_revision
        == process_spec.expected_ktransformers_revision
        and model_runtime_receipt.transformers_distribution_version
        == process_spec.required_transformers_distribution_version
        and model_runtime_receipt.transformers_module_version
        == process_spec.required_transformers_module_version
        and kernel_receipt is not None
        and model_runtime_receipt.torch_version == kernel_receipt.torch_version
        and model_runtime_receipt.cuda_version == kernel_receipt.cuda_version
        and model_runtime_receipt.sgl_kernel_build_id
        == kernel_receipt.sgl_kernel_build_id
        and model_runtime_receipt.deep_gemm_build_id
        == kernel_receipt.deep_gemm_build_id
        and model_runtime_receipt.kt_kernel_build_id
        == kernel_receipt.kt_kernel_build_id
        and model_runtime_receipt.ktransformers_method
        == process_spec.ktransformers_method
        and model_runtime_receipt.resident_gpu_experts
        == process_spec.stage.resident_gpu_experts
        and model_runtime_receipt.attention_backend == process_spec.attention_backend
        and model_runtime_receipt.kv_cache_dtype == GLM_4_7_FLASH_KV_CACHE_DTYPE
        and model_runtime_receipt.max_total_tokens == process_spec.plan.max_total_tokens
        and model_runtime_receipt.static_memory_fraction
        == process_spec.plan.static_memory_fraction
        and model_runtime_receipt.ktransformers_wrapped_expert_layers
        == expected_wrapped_layers
        and frozenset(model_runtime_receipt.capabilities)
        == expected_runtime_capabilities
    ):
        raise ValueError("admission binding model runtime evidence is inconsistent")

    if not (
        kernel_receipt.gpu_uuid == process_spec.gpu_uuid
        and kernel_receipt.gpu_compute_capability == (8, 6)
        and kernel_receipt.hostname == str(process_spec.node_id)
        and kernel_receipt.executable == process_spec.executable
        and kernel_receipt.cpu_cores == process_spec.cpu_cores
        and kernel_receipt.memory_nodes == process_spec.memory_nodes
        and len(kernel_receipt.threads_per_subpool)
        == process_spec.stage.threadpool_count
        and sum(kernel_receipt.threads_per_subpool)
        == process_spec.stage.cpu_infer_threads
        and kernel_receipt.sglang_revision == process_spec.expected_sglang_revision
        and kernel_receipt.ktransformers_revision
        == process_spec.expected_ktransformers_revision
        and kernel_receipt.transformers_distribution_version
        == process_spec.required_transformers_distribution_version
        and kernel_receipt.transformers_module_version
        == process_spec.required_transformers_module_version
        and kernel_receipt.capabilities == ("kt_bf16_amx_executed_v1",)
    ):
        raise ValueError("admission binding kernel evidence is inconsistent")


@final
class SglangKtPreflightFailure(FrozenModel):
    pipeline_rank: ResourceIndex
    node_id: NodeId
    check: PreflightCheck
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    message: str


@final
class SglangKtPreflightPassed(FrozenModel):
    process_specs: tuple[SglangKtProcessLaunchSpec, ...]
    admission_bindings: tuple[SglangKtRankAdmissionBinding, ...]

    @model_validator(mode="after")
    def validate_nonempty_group(self) -> "SglangKtPreflightPassed":
        if not self.process_specs:
            raise ValueError("a passed preflight must contain process specs")
        process_ranks = tuple(spec.pipeline_rank for spec in self.process_specs)
        binding_ranks = tuple(
            binding.pipeline_rank for binding in self.admission_bindings
        )
        if len(set(process_ranks)) != len(process_ranks):
            raise ValueError("passed preflight process ranks must be unique")
        if tuple(sorted(binding_ranks)) != tuple(sorted(process_ranks)) or len(
            set(binding_ranks)
        ) != len(binding_ranks):
            raise ValueError(
                "passed preflight requires one admission binding per process rank"
            )
        bindings_by_rank = {
            binding.pipeline_rank: binding for binding in self.admission_bindings
        }
        for process_spec in self.process_specs:
            validate_sglang_kt_rank_admission_binding(
                process_spec, bindings_by_rank[process_spec.pipeline_rank]
            )
        return self


@final
class SglangKtPreflightFailed(FrozenModel):
    failures: tuple[SglangKtPreflightFailure, ...]

    @model_validator(mode="after")
    def validate_failures(self) -> "SglangKtPreflightFailed":
        if not self.failures:
            raise ValueError("a failed preflight must contain failures")
        return self


SglangKtPreflightResult = SglangKtPreflightPassed | SglangKtPreflightFailed


def _build_rank_admission_binding(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
) -> SglangKtRankAdmissionBinding:
    model_snapshot_receipts: tuple[SglangKtModelSnapshotReceiptObservation, ...] = ()
    model_runtime_validation_receipt: (
        SglangKtRuntimeValidationReceiptObservation | None
    ) = None
    kernel_runtime_validation_receipt: (
        SglangKtKernelRuntimeValidationReceiptObservation | None
    ) = None
    if process_spec.target_profile in GLM_4_7_FLASH_TARGET_PROFILES:
        receipts_by_path = {
            receipt.model_path: receipt
            for receipt in observation.model_snapshot_receipts
        }
        planned_paths = tuple(
            dict.fromkeys(
                (
                    process_spec.model_path,
                    process_spec.ktransformers_weight_path,
                )
            )
        )
        try:
            model_snapshot_receipts = tuple(
                receipts_by_path[path] for path in planned_paths
            )
        except KeyError as error:
            raise RuntimeError(
                "successful GLM-4.7 preflight lost model snapshot evidence"
            ) from error
        kernel_runtime_validation_receipt = next(
            (
                receipt
                for receipt in observation.kernel_runtime_validation_receipts
                if receipt.gpu_uuid == process_spec.gpu_uuid
            ),
            None,
        )
        model_runtime_validation_receipt = next(
            (
                receipt
                for receipt in observation.runtime_validation_receipts
                if receipt.gpu_uuid == process_spec.gpu_uuid
            ),
            None,
        )

    binding = SglangKtRankAdmissionBinding(
        pipeline_rank=process_spec.pipeline_rank,
        process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(
            process_spec
        ),
        model_snapshot_receipts=model_snapshot_receipts,
        model_runtime_validation_receipt=model_runtime_validation_receipt,
        kernel_runtime_validation_receipt=kernel_runtime_validation_receipt,
    )
    try:
        validate_sglang_kt_rank_admission_binding(process_spec, binding)
    except ValueError as error:
        raise RuntimeError(
            "successful preflight produced inconsistent admission evidence"
        ) from error
    return binding


def evaluate_sglang_kt_preflight(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
    host_observations: tuple[SglangKtHostPreflightObservation, ...],
) -> SglangKtPreflightResult:
    """Evaluate a complete process group without collecting facts or doing I/O.

    A failed result intentionally contains no process specs, so an executor
    cannot accidentally start ranks that individually passed.
    """

    if not process_specs:
        raise ValueError("SGLang-KT preflight requires process specs")
    launch_plan = process_specs[0].plan
    if any(process_spec.plan != launch_plan for process_spec in process_specs):
        raise ValueError("SGLang-KT preflight specs must share one launch plan")
    pipeline_ranks = tuple(spec.pipeline_rank for spec in process_specs)
    if len(set(pipeline_ranks)) != len(pipeline_ranks):
        raise ValueError("SGLang-KT preflight process ranks must be unique")
    if tuple(sorted(pipeline_ranks)) != tuple(range(len(process_specs))):
        raise ValueError("SGLang-KT preflight process ranks must be contiguous")
    if len(process_specs) != len(launch_plan.stages):
        raise ValueError("SGLang-KT preflight requires every launch-plan rank")

    observations_by_node: dict[NodeId, list[SglangKtHostPreflightObservation]] = {}
    for observation in host_observations:
        observations_by_node.setdefault(observation.node_id, []).append(observation)

    failures: list[SglangKtPreflightFailure] = []
    for process_spec in sorted(process_specs, key=lambda spec: spec.pipeline_rank):
        observations = observations_by_node.get(process_spec.node_id, [])
        if len(observations) != 1:
            _record_failure(
                failures,
                process_spec,
                "host_observation",
                expected=("exactly one host observation",),
                observed=(str(len(observations)),),
                detail=(
                    "no host observation was supplied"
                    if not observations
                    else "multiple host observations were supplied"
                ),
            )
            continue

        observation = observations[0]
        _evaluate_runtime(process_spec, observation.runtime, failures)
        _evaluate_paths_and_receipt(process_spec, observation, failures)
        _evaluate_runtime_validation(process_spec, observation, failures)
        _evaluate_resources(process_spec, observation, failures)
        _evaluate_ports(process_spec, observation, failures)

    if failures:
        return SglangKtPreflightFailed(failures=tuple(failures))
    admission_bindings = tuple(
        _build_rank_admission_binding(
            process_spec,
            observations_by_node[process_spec.node_id][0],
        )
        for process_spec in sorted(
            process_specs, key=lambda candidate: candidate.pipeline_rank
        )
    )
    return SglangKtPreflightPassed(
        process_specs=process_specs,
        admission_bindings=admission_bindings,
    )


def _evaluate_runtime(
    process_spec: SglangKtProcessLaunchSpec,
    runtime: SglangKtRuntimeObservation,
    failures: list[SglangKtPreflightFailure],
) -> None:
    if runtime.executable != process_spec.executable:
        _record_failure(
            failures,
            process_spec,
            "python_executable",
            expected=(process_spec.executable,),
            observed=_optional_observed(runtime.executable),
            detail="the verified Python executable does not match the launch spec",
        )
    if runtime.python_implementation != "CPython":
        _record_failure(
            failures,
            process_spec,
            "python_implementation",
            expected=("CPython",),
            observed=_optional_observed(runtime.python_implementation),
            detail="KTransformers requires a verified CPython runtime",
        )

    python_version = runtime.python_version
    if python_version is None or python_version.release < (3, 11, 0):
        _record_failure(
            failures,
            process_spec,
            "python_version",
            expected=(">=3.11.0",),
            observed=_optional_observed(python_version),
            detail="KTransformers v0.6.3 requires Python 3.11 or newer",
        )
    if runtime.sglang_revision != process_spec.expected_sglang_revision:
        _record_failure(
            failures,
            process_spec,
            "sglang_revision",
            expected=(process_spec.expected_sglang_revision,),
            observed=_optional_observed(runtime.sglang_revision),
            detail="the installed SGLang source revision is not the pinned runtime",
        )
    if runtime.ktransformers_revision != process_spec.expected_ktransformers_revision:
        _record_failure(
            failures,
            process_spec,
            "ktransformers_revision",
            expected=(process_spec.expected_ktransformers_revision,),
            observed=_optional_observed(runtime.ktransformers_revision),
            detail=(
                "the installed KTransformers source revision is not the pinned runtime"
            ),
        )
    if (
        runtime.transformers_distribution_version
        != process_spec.required_transformers_distribution_version
    ):
        _record_failure(
            failures,
            process_spec,
            "transformers_distribution_version",
            expected=(process_spec.required_transformers_distribution_version,),
            observed=_optional_observed(runtime.transformers_distribution_version),
            detail="the installed transformers-kt distribution is not supported",
        )
    if (
        runtime.transformers_module_version
        != process_spec.required_transformers_module_version
    ):
        _record_failure(
            failures,
            process_spec,
            "transformers_module_version",
            expected=(process_spec.required_transformers_module_version,),
            observed=_optional_observed(runtime.transformers_module_version),
            detail="the imported transformers module version is not supported",
        )


def _evaluate_paths_and_receipt(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
    failures: list[SglangKtPreflightFailure],
) -> None:
    readable_directories = set(observation.readable_directories)
    if process_spec.model_path not in readable_directories:
        _record_failure(
            failures,
            process_spec,
            "model_path",
            expected=(process_spec.model_path,),
            observed=observation.readable_directories,
            detail="the model path was not verified as a readable directory",
        )
    if process_spec.ktransformers_weight_path not in readable_directories:
        _record_failure(
            failures,
            process_spec,
            "ktransformers_weight_path",
            expected=(process_spec.ktransformers_weight_path,),
            observed=observation.readable_directories,
            detail="the KTransformers weight path was not verified as readable",
        )

    _evaluate_snapshot_receipt(
        process_spec,
        observation,
        process_spec.model_path,
        "model_revision_receipt",
        "the model snapshot lacks an exact, complete revision receipt",
        failures,
    )
    _evaluate_snapshot_receipt(
        process_spec,
        observation,
        process_spec.ktransformers_weight_path,
        "ktransformers_weight_revision_receipt",
        "the KTransformers weights lack an exact, compatible revision receipt",
        failures,
    )


def _evaluate_snapshot_receipt(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
    snapshot_path: AbsoluteRuntimePath,
    check: Literal["model_revision_receipt", "ktransformers_weight_revision_receipt"],
    detail: str,
    failures: list[SglangKtPreflightFailure],
) -> None:
    expected_config_sha256 = (
        GLM_4_7_FLASH_BF16_CONFIG_SHA256
        if process_spec.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
        else None
    )
    expected_contract_sha256 = process_spec.expected_model_contract_sha256
    receipt = next(
        (
            candidate
            for candidate in observation.model_snapshot_receipts
            if candidate.model_path == snapshot_path
        ),
        None,
    )
    receipt_matches = (
        receipt is not None
        and receipt.model_id == process_spec.model_id
        and receipt.revision == process_spec.expected_model_revision
        and receipt.weight_format == "safetensors"
        and receipt.ktransformers_method == process_spec.ktransformers_method
        and (
            expected_config_sha256 is None
            or receipt.config_sha256 == expected_config_sha256
        )
        and (
            expected_contract_sha256 is None
            or (
                receipt.contract_receipt_sha256 == expected_contract_sha256
                and receipt.contract_sha256 == expected_contract_sha256
                and receipt.index_sha256 is not None
                and receipt.weight_map_entries is not None
                and receipt.shard_count is not None
                and receipt.physical_weight_bytes is not None
            )
        )
        and all(
            stage.start_layer in receipt.full_indexer_layer_starts
            for stage in process_spec.plan.stages
        )
        and receipt.receipt_verified
        and receipt.snapshot_complete
    )
    if not receipt_matches:
        observed = (
            ("<missing>",)
            if receipt is None
            else (
                str(receipt.model_id),
                receipt.revision,
                receipt.weight_format,
                receipt.ktransformers_method,
                receipt.config_sha256,
                *(
                    (
                        f"contract_receipt_sha256={receipt.contract_receipt_sha256}",
                        f"contract_sha256={receipt.contract_sha256}",
                        f"index_sha256={receipt.index_sha256}",
                        f"weight_map_entries={receipt.weight_map_entries}",
                        f"shard_count={receipt.shard_count}",
                        f"physical_weight_bytes={receipt.physical_weight_bytes}",
                    )
                    if expected_contract_sha256 is not None
                    else ()
                ),
                "full_indexer_layer_starts="
                + ",".join(str(start) for start in receipt.full_indexer_layer_starts),
                f"receipt_verified={receipt.receipt_verified}",
                f"snapshot_complete={receipt.snapshot_complete}",
            )
        )
        _record_failure(
            failures,
            process_spec,
            check,
            expected=(
                str(process_spec.model_id),
                process_spec.expected_model_revision,
                "safetensors",
                process_spec.ktransformers_method,
                (
                    "config_sha256=<verified>"
                    if expected_config_sha256 is None
                    else expected_config_sha256
                ),
                *(
                    (
                        f"model_contract_sha256={expected_contract_sha256}",
                        "model_contract_index_and_shards=<verified>",
                    )
                    if expected_contract_sha256 is not None
                    else ()
                ),
                "pipeline starts on verified full indexers",
                "receipt_verified=True",
                "snapshot_complete=True",
            ),
            observed=observed,
            detail=detail,
        )


def _evaluate_runtime_validation(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
    failures: list[SglangKtPreflightFailure],
) -> None:
    expected_cpu_backend: Literal["AMX", "AMX_BF16"]
    expected_kv_cache_dtype: Literal["bfloat16", "fp8_e4m3"]
    required_capabilities: frozenset[SglangKtRuntimeCapability]
    expected_wrapped_expert_layers: tuple[ResourceIndex, ...] | None = None
    if process_spec.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE:
        expected_cpu_backend = "AMX_BF16"
        expected_kv_cache_dtype = GLM_4_7_FLASH_KV_CACHE_DTYPE
        required_capabilities = (
            GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_REQUIRED_RUNTIME_CAPABILITIES
        )
        expected_wrapped_expert_layers = GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS
    elif process_spec.target_profile == GLM_4_7_FLASH_TARGET_PROFILE:
        expected_cpu_backend = "AMX_BF16"
        expected_kv_cache_dtype = GLM_4_7_FLASH_KV_CACHE_DTYPE
        required_capabilities = GLM_4_7_FLASH_REQUIRED_RUNTIME_CAPABILITIES
        expected_wrapped_expert_layers = GLM_4_7_FLASH_WRAPPED_EXPERT_LAYERS
    else:
        expected_cpu_backend = "AMX"
        expected_kv_cache_dtype = GLM_5_2_KV_CACHE_DTYPE
        required_capabilities = GLM_5_2_REQUIRED_RUNTIME_CAPABILITIES

    validation_receipt = next(
        (
            receipt
            for receipt in observation.runtime_validation_receipts
            if receipt.gpu_uuid == process_spec.gpu_uuid
        ),
        None,
    )
    snapshot_receipt = next(
        (
            receipt
            for receipt in observation.model_snapshot_receipts
            if receipt.model_path == process_spec.model_path
        ),
        None,
    )
    kernel_validation_receipt = next(
        (
            receipt
            for receipt in observation.kernel_runtime_validation_receipts
            if receipt.gpu_uuid == process_spec.gpu_uuid
        ),
        None,
    )
    kernel_validation_required = (
        process_spec.target_profile in GLM_4_7_FLASH_TARGET_PROFILES
    )
    kernel_validation_matches = not kernel_validation_required or (
        kernel_validation_receipt is not None
        and kernel_validation_receipt.capabilities == ("kt_bf16_amx_executed_v1",)
        and kernel_validation_receipt.gpu_compute_capability == (8, 6)
        and kernel_validation_receipt.cpu_cores == process_spec.cpu_cores
        and kernel_validation_receipt.memory_nodes == process_spec.memory_nodes
        and kernel_validation_receipt.executable == process_spec.executable
        and len(kernel_validation_receipt.threads_per_subpool)
        == process_spec.stage.threadpool_count
        and sum(kernel_validation_receipt.threads_per_subpool)
        == process_spec.stage.cpu_infer_threads
        and kernel_validation_receipt.hostname == str(process_spec.node_id)
        and kernel_validation_receipt.sglang_revision
        == process_spec.expected_sglang_revision
        and kernel_validation_receipt.ktransformers_revision
        == process_spec.expected_ktransformers_revision
        and kernel_validation_receipt.transformers_distribution_version
        == process_spec.required_transformers_distribution_version
        and kernel_validation_receipt.transformers_module_version
        == process_spec.required_transformers_module_version
        and kernel_validation_receipt.torch_version == observation.runtime.torch_version
        and kernel_validation_receipt.cuda_version == observation.runtime.cuda_version
        and kernel_validation_receipt.sgl_kernel_build_id
        == observation.runtime.sgl_kernel_build_id
        and kernel_validation_receipt.deep_gemm_build_id
        == observation.runtime.deep_gemm_build_id
        and kernel_validation_receipt.kt_kernel_build_id
        == observation.runtime.kt_kernel_build_id
    )
    receipt_matches = (
        validation_receipt is not None
        and snapshot_receipt is not None
        and kernel_validation_matches
        and validation_receipt.target_profile == process_spec.target_profile
        and validation_receipt.gpu_compute_capability == (8, 6)
        and validation_receipt.cpu_cores == process_spec.cpu_cores
        and validation_receipt.memory_nodes == process_spec.memory_nodes
        and validation_receipt.executed_cpu_backend == expected_cpu_backend
        and validation_receipt.model_id == process_spec.model_id
        and validation_receipt.model_revision == process_spec.expected_model_revision
        and validation_receipt.model_config_sha256 == snapshot_receipt.config_sha256
        and validation_receipt.sglang_revision == process_spec.expected_sglang_revision
        and validation_receipt.ktransformers_revision
        == process_spec.expected_ktransformers_revision
        and validation_receipt.transformers_distribution_version
        == process_spec.required_transformers_distribution_version
        and validation_receipt.transformers_module_version
        == process_spec.required_transformers_module_version
        and validation_receipt.torch_version == observation.runtime.torch_version
        and validation_receipt.cuda_version == observation.runtime.cuda_version
        and validation_receipt.sgl_kernel_build_id
        == observation.runtime.sgl_kernel_build_id
        and validation_receipt.deep_gemm_build_id
        == observation.runtime.deep_gemm_build_id
        and validation_receipt.kt_kernel_build_id
        == observation.runtime.kt_kernel_build_id
        and validation_receipt.ktransformers_method == process_spec.ktransformers_method
        and validation_receipt.resident_gpu_experts
        == process_spec.stage.resident_gpu_experts
        and validation_receipt.attention_backend == process_spec.attention_backend
        and validation_receipt.kv_cache_dtype == expected_kv_cache_dtype
        and validation_receipt.max_total_tokens == process_spec.plan.max_total_tokens
        and validation_receipt.static_memory_fraction
        == process_spec.plan.static_memory_fraction
        and (
            expected_wrapped_expert_layers is None
            or validation_receipt.ktransformers_wrapped_expert_layers
            == expected_wrapped_expert_layers
        )
        and frozenset(validation_receipt.capabilities) == required_capabilities
    )
    if receipt_matches:
        return

    observed = (
        ("<missing>",)
        if validation_receipt is None
        else (
            validation_receipt.target_profile,
            validation_receipt.gpu_uuid,
            "compute_capability="
            + ".".join(
                str(component)
                for component in validation_receipt.gpu_compute_capability
            ),
            "cpu_cores=" + ",".join(str(core) for core in validation_receipt.cpu_cores),
            "memory_nodes="
            + ",".join(str(node) for node in validation_receipt.memory_nodes),
            "executed_cpu_backend=" + validation_receipt.executed_cpu_backend,
            str(validation_receipt.model_id),
            validation_receipt.model_revision,
            validation_receipt.model_config_sha256,
            validation_receipt.sglang_revision,
            validation_receipt.ktransformers_revision,
            validation_receipt.transformers_distribution_version,
            validation_receipt.transformers_module_version,
            validation_receipt.torch_version,
            validation_receipt.cuda_version,
            validation_receipt.sgl_kernel_build_id,
            validation_receipt.deep_gemm_build_id,
            validation_receipt.kt_kernel_build_id,
            validation_receipt.ktransformers_method,
            f"resident_gpu_experts={validation_receipt.resident_gpu_experts}",
            validation_receipt.attention_backend,
            validation_receipt.kv_cache_dtype,
            f"max_total_tokens={validation_receipt.max_total_tokens}",
            "static_memory_fraction=" + str(validation_receipt.static_memory_fraction),
            "ktransformers_wrapped_expert_layers="
            + ",".join(
                str(layer)
                for layer in validation_receipt.ktransformers_wrapped_expert_layers
            ),
            *validation_receipt.capabilities,
        )
    )
    _record_failure(
        failures,
        process_spec,
        "runtime_validation_receipt",
        expected=(
            process_spec.target_profile,
            process_spec.gpu_uuid,
            "compute_capability=8.6",
            "cpu_cores=" + ",".join(str(core) for core in process_spec.cpu_cores),
            "memory_nodes=" + ",".join(str(node) for node in process_spec.memory_nodes),
            f"executed_cpu_backend={expected_cpu_backend}",
            str(process_spec.model_id),
            process_spec.expected_model_revision,
            "model_config_sha256=<snapshot receipt>",
            process_spec.expected_sglang_revision,
            process_spec.expected_ktransformers_revision,
            process_spec.required_transformers_distribution_version,
            process_spec.required_transformers_module_version,
            "torch_version=<current runtime>",
            "cuda_version=<current runtime>",
            "sgl_kernel_build_id=<current runtime>",
            "deep_gemm_build_id=<current runtime>",
            "kt_kernel_build_id=<current runtime>",
            process_spec.ktransformers_method,
            f"resident_gpu_experts={process_spec.stage.resident_gpu_experts}",
            process_spec.attention_backend,
            expected_kv_cache_dtype,
            f"max_total_tokens={process_spec.plan.max_total_tokens}",
            "static_memory_fraction=" + str(process_spec.plan.static_memory_fraction),
            *(
                ("kernel_runtime_validation=<exact file-bound receipt>",)
                if kernel_validation_required
                else ()
            ),
            "ktransformers_wrapped_expert_layers="
            + (
                "<not-profile-bound>"
                if expected_wrapped_expert_layers is None
                else ",".join(str(layer) for layer in expected_wrapped_expert_layers)
            ),
            *tuple(sorted(required_capabilities)),
        ),
        observed=observed,
        detail=(
            "the exact target profile lacks bound GPU, model, CPU/NUMA, runtime, "
            "and executed-backend validation evidence"
        ),
    )


def _evaluate_resources(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
    failures: list[SglangKtPreflightFailure],
) -> None:
    if process_spec.gpu_uuid not in observation.gpu_uuids:
        _record_failure(
            failures,
            process_spec,
            "gpu_uuid",
            expected=(process_spec.gpu_uuid,),
            observed=observation.gpu_uuids,
            detail="the assigned GPU UUID is not available on the host",
        )

    missing_cpu_cores = tuple(
        core for core in process_spec.cpu_cores if core not in observation.cpu_cores
    )
    if missing_cpu_cores:
        _record_failure(
            failures,
            process_spec,
            "cpu_cores",
            expected=tuple(str(core) for core in process_spec.cpu_cores),
            observed=tuple(str(core) for core in observation.cpu_cores),
            detail=(
                "assigned CPU cores are unavailable: "
                + ", ".join(str(core) for core in missing_cpu_cores)
            ),
        )

    missing_memory_nodes = tuple(
        node
        for node in process_spec.memory_nodes
        if node not in observation.memory_nodes
    )
    if missing_memory_nodes:
        _record_failure(
            failures,
            process_spec,
            "memory_nodes",
            expected=tuple(str(node) for node in process_spec.memory_nodes),
            observed=tuple(str(node) for node in observation.memory_nodes),
            detail=(
                "assigned NUMA memory nodes are unavailable: "
                + ", ".join(str(node) for node in missing_memory_nodes)
            ),
        )

    missing_hca_devices = tuple(
        device
        for device in process_spec.hca_devices
        if device not in observation.hca_devices
    )
    if missing_hca_devices:
        _record_failure(
            failures,
            process_spec,
            "hca_devices",
            expected=process_spec.hca_devices,
            observed=observation.hca_devices,
            detail=(
                "assigned HCA ports are unavailable: " + ", ".join(missing_hca_devices)
            ),
        )


def _evaluate_ports(
    process_spec: SglangKtProcessLaunchSpec,
    observation: SglangKtHostPreflightObservation,
    failures: list[SglangKtPreflightFailure],
) -> None:
    available_endpoint_keys = {
        (endpoint.ip, endpoint.port)
        for endpoint in observation.available_bind_endpoints
    }
    service_endpoint_key = (
        process_spec.service_endpoint.ip,
        process_spec.service_endpoint.port,
    )
    if service_endpoint_key not in available_endpoint_keys:
        _record_failure(
            failures,
            process_spec,
            "service_endpoint",
            expected=(str(process_spec.service_endpoint),),
            observed=tuple(
                str(endpoint) for endpoint in observation.available_bind_endpoints
            ),
            detail="the stage service endpoint is not available for binding",
        )

    if process_spec.pipeline_rank == 0:
        coordinator_key = (
            process_spec.distributed_coordinator.ip,
            process_spec.distributed_coordinator.port,
        )
        if coordinator_key not in available_endpoint_keys:
            _record_failure(
                failures,
                process_spec,
                "distributed_coordinator",
                expected=(str(process_spec.distributed_coordinator),),
                observed=tuple(
                    str(endpoint) for endpoint in observation.available_bind_endpoints
                ),
                detail="the distributed coordinator endpoint is not available",
            )


def _optional_observed(value: object | None) -> tuple[str, ...]:
    return ("<unobserved>",) if value is None else (str(value),)


def _record_failure(
    failures: list[SglangKtPreflightFailure],
    process_spec: SglangKtProcessLaunchSpec,
    check: PreflightCheck,
    *,
    expected: tuple[str, ...],
    observed: tuple[str, ...],
    detail: str,
) -> None:
    failures.append(
        SglangKtPreflightFailure(
            pipeline_rank=process_spec.pipeline_rank,
            node_id=process_spec.node_id,
            check=check,
            expected=expected,
            observed=observed,
            message=(
                f"pipeline rank {process_spec.pipeline_rank} on "
                f"{process_spec.node_id}: {detail}"
            ),
        )
    )
