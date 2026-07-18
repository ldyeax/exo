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
    StaticMemoryFraction,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import (
    GLM_5_2_KV_CACHE_DTYPE,
    REQUIRED_TRANSFORMERS_VERSION,
    SglangKtProcessLaunchSpec,
)

ObservedText = Annotated[str, StringConstraints(min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SglangKtRuntimeCapability = Literal[
    "kt_tp_group_local_broadcast_v1",
    "glm52_nsa_sm86_short_forward_v1",
    "kt_physical_numa_mapping_v1",
    "kt_process_cpu_affinity_v1",
    "kt_fp8_amx_executed_v1",
]
REQUIRED_RUNTIME_CAPABILITIES: frozenset[SglangKtRuntimeCapability] = frozenset(
    (
        "kt_tp_group_local_broadcast_v1",
        "glm52_nsa_sm86_short_forward_v1",
        "kt_physical_numa_mapping_v1",
        "kt_process_cpu_affinity_v1",
        "kt_fp8_amx_executed_v1",
    )
)
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


@final
class SglangKtRuntimeValidationReceiptObservation(FrozenModel):
    """Evidence from a target-specific runtime smoke test on one GPU.

    The version-only runtime probe never creates this receipt. A future
    validator must execute the named checks on the exact stack and checkpoint
    before the external process group can pass preflight.
    """

    gpu_uuid: GpuUuid
    gpu_compute_capability: tuple[PositiveInt, ResourceIndex]
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    executed_cpu_backend: Literal["AMX"]
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
    kv_cache_dtype: Literal["fp8_e4m3"]
    max_total_tokens: PositiveInt
    static_memory_fraction: StaticMemoryFraction
    capabilities: tuple[SglangKtRuntimeCapability, ...]

    @model_validator(mode="after")
    def validate_capabilities(self) -> "SglangKtRuntimeValidationReceiptObservation":
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("runtime validation capabilities must be unique")
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

    @model_validator(mode="after")
    def validate_indexer_boundaries(self) -> "SglangKtModelSnapshotReceiptObservation":
        starts = self.full_indexer_layer_starts
        if not starts or starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError(
                "full_indexer_layer_starts must be sorted, unique, and begin at zero"
            )
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
        return self


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

    @model_validator(mode="after")
    def validate_nonempty_group(self) -> "SglangKtPreflightPassed":
        if not self.process_specs:
            raise ValueError("a passed preflight must contain process specs")
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
    return SglangKtPreflightPassed(process_specs=process_specs)


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
        != process_spec.required_transformers_version
    ):
        _record_failure(
            failures,
            process_spec,
            "transformers_distribution_version",
            expected=(process_spec.required_transformers_version,),
            observed=_optional_observed(runtime.transformers_distribution_version),
            detail="the installed transformers-kt distribution is not supported",
        )
    if runtime.transformers_module_version != REQUIRED_TRANSFORMERS_VERSION:
        _record_failure(
            failures,
            process_spec,
            "transformers_module_version",
            expected=(REQUIRED_TRANSFORMERS_VERSION,),
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
                "config_sha256=<verified>",
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
    receipt_matches = (
        validation_receipt is not None
        and snapshot_receipt is not None
        and validation_receipt.gpu_compute_capability == (8, 6)
        and validation_receipt.cpu_cores == process_spec.cpu_cores
        and validation_receipt.memory_nodes == process_spec.memory_nodes
        and validation_receipt.executed_cpu_backend == "AMX"
        and validation_receipt.model_id == process_spec.model_id
        and validation_receipt.model_revision == process_spec.expected_model_revision
        and validation_receipt.model_config_sha256 == snapshot_receipt.config_sha256
        and validation_receipt.sglang_revision == process_spec.expected_sglang_revision
        and validation_receipt.ktransformers_revision
        == process_spec.expected_ktransformers_revision
        and validation_receipt.transformers_distribution_version
        == process_spec.required_transformers_version
        and validation_receipt.transformers_module_version
        == process_spec.required_transformers_version
        and validation_receipt.kv_cache_dtype == GLM_5_2_KV_CACHE_DTYPE
        and validation_receipt.max_total_tokens == process_spec.plan.max_total_tokens
        and validation_receipt.static_memory_fraction
        == process_spec.plan.static_memory_fraction
        and REQUIRED_RUNTIME_CAPABILITIES.issubset(validation_receipt.capabilities)
    )
    if receipt_matches:
        return

    observed = (
        ("<missing>",)
        if validation_receipt is None
        else (
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
            validation_receipt.kv_cache_dtype,
            f"max_total_tokens={validation_receipt.max_total_tokens}",
            "static_memory_fraction=" + str(validation_receipt.static_memory_fraction),
            *validation_receipt.capabilities,
        )
    )
    _record_failure(
        failures,
        process_spec,
        "runtime_validation_receipt",
        expected=(
            process_spec.gpu_uuid,
            "compute_capability=8.6",
            "cpu_cores=" + ",".join(str(core) for core in process_spec.cpu_cores),
            "memory_nodes=" + ",".join(str(node) for node in process_spec.memory_nodes),
            "executed_cpu_backend=AMX",
            str(process_spec.model_id),
            process_spec.expected_model_revision,
            "model_config_sha256=<snapshot receipt>",
            process_spec.expected_sglang_revision,
            process_spec.expected_ktransformers_revision,
            process_spec.required_transformers_version,
            GLM_5_2_KV_CACHE_DTYPE,
            f"max_total_tokens={process_spec.plan.max_total_tokens}",
            "static_memory_fraction=" + str(process_spec.plan.static_memory_fraction),
            *tuple(sorted(REQUIRED_RUNTIME_CAPABILITIES)),
        ),
        observed=observed,
        detail=(
            "the exact GPU/runtime stack lacks bound PP=3 broadcast, SM86 NSA, "
            "physical-NUMA, process-affinity, and executed-AMX validation evidence"
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
