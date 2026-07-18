from typing import Annotated, Literal, final

from pydantic import PositiveInt, StringConstraints, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    HcaDevice,
    KTransformersMethod,
    NetworkPort,
    ResourceIndex,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import SglangKtProcessLaunchSpec

ObservedText = Annotated[str, StringConstraints(min_length=1)]
PreflightCheck = Literal[
    "host_observation",
    "python_executable",
    "python_implementation",
    "python_version",
    "sglang_revision",
    "ktransformers_revision",
    "transformers_version",
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
    "nccl_port",
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
    transformers_version: ObservedText | None = None


@final
class SglangKtModelSnapshotReceiptObservation(FrozenModel):
    """Observed Exo revision receipt and snapshot completeness for one path."""

    model_path: AbsoluteRuntimePath
    model_id: ModelId
    revision: GitRevision
    weight_format: Literal["safetensors"]
    ktransformers_method: KTransformersMethod
    receipt_verified: bool
    snapshot_complete: bool


@final
class SglangKtHostPreflightObservation(FrozenModel):
    """Pre-collected facts for one host; constructing this performs no probes.

    Directory, HCA, and port collections contain only resources the future
    collector established as usable for the planned launch.
    """

    node_id: NodeId
    runtime: SglangKtRuntimeObservation
    readable_directories: tuple[AbsoluteRuntimePath, ...] = ()
    model_snapshot_receipts: tuple[SglangKtModelSnapshotReceiptObservation, ...] = ()
    gpu_uuids: tuple[GpuUuid, ...] = ()
    cpu_cores: tuple[ResourceIndex, ...] = ()
    memory_nodes: tuple[ResourceIndex, ...] = ()
    hca_devices: tuple[HcaDevice, ...] = ()
    available_bind_endpoints: tuple[Host, ...] = ()
    available_local_ports: tuple[NetworkPort, ...] = ()

    @model_validator(mode="after")
    def validate_unambiguous_facts(self) -> "SglangKtHostPreflightObservation":
        collections: tuple[tuple[str, tuple[object, ...]], ...] = (
            ("readable_directories", self.readable_directories),
            ("gpu_uuids", self.gpu_uuids),
            ("cpu_cores", self.cpu_cores),
            ("memory_nodes", self.memory_nodes),
            ("hca_devices", self.hca_devices),
            ("available_local_ports", self.available_local_ports),
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
    if runtime.transformers_version != process_spec.required_transformers_version:
        _record_failure(
            failures,
            process_spec,
            "transformers_version",
            expected=(process_spec.required_transformers_version,),
            observed=_optional_observed(runtime.transformers_version),
            detail="the installed Transformers package version is not supported",
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
                "receipt_verified=True",
                "snapshot_complete=True",
            ),
            observed=observed,
            detail=detail,
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

    if process_spec.nccl_port not in observation.available_local_ports:
        _record_failure(
            failures,
            process_spec,
            "nccl_port",
            expected=(str(process_spec.nccl_port),),
            observed=tuple(str(port) for port in observation.available_local_ports),
            detail="the planned NCCL setup port is not available on the host",
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
