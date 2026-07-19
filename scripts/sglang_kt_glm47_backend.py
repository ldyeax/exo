"""Disposable live backend for the pinned GLM-4.7 SGLang-KT probe.

This module is safe to import in the parent process: Torch, SGLang, and
KTransformers are resolved only by :func:`load_glm47_runtime_bindings` inside
the resource-bound disposable child.  It publishes no files and creates no
receipt; callers receive typed, tensor-free evidence only after cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence, Sized
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from scripts.sglang_kt_glm47_reference import (
    GLM47_BF16_DTYPE,
    GLM47_FLOAT32_DTYPE,
    GLM47_LAYER_ONE_ROUTE_WEIGHTS,
    Glm47LayerOneHybridMergeEvidence,
    Glm47LayerOneOutputEvidence,
    Glm47LayerOneReferenceOutputs,
    Glm47TensorDtype,
    Glm47TensorEvidenceBackend,
    Glm47TensorSnapshot,
    build_glm47_layer_one_bf16_tensor_keys,
    build_glm47_layer_one_hybrid_merge_evidence,
    build_glm47_layer_one_output_evidence,
    compute_torch_glm47_layer_one_reference,
    create_torch_glm47_tensor_evidence_backend,
)
from scripts.sglang_kt_glm47_trace import (
    GLM47_ROUTED_LAYER_IDS,
    GPU_METHOD_APPLY,
    LAYER_ONE_KTEP_OUTER_APPLY,
    MODEL_FORWARD,
    NATIVE_WRAPPER_SUBMIT,
    NATIVE_WRAPPER_SYNC,
    QUANT_METHOD_APPLY,
    SGLANG_CPU_SUBMIT,
    SGLANG_CPU_SYNC,
    Glm47TraceSession,
    Glm47TraceTargets,
    TraceProbe,
)

GLM47_LAYER_ONE_RANDOM_SEED: Final = 20_260_719
GLM47_LAYER_ONE_INPUT_SHAPE: Final = (1, 2_048)
GLM47_ROUTED_EXPERT_COUNT: Final = 64
GLM47_EXTEND_TOKEN_IDS: Final = (1, 2, 3, 4, 5, 6, 7, 8)
GLM47_LOGITS_SHAPE: Final = (1, 154_880)

_EXPECTED_COVERAGE_KEYS: Final = frozenset(
    {
        "schema_version",
        "model_architecture",
        "pipeline_parallel_size",
        "layer_count",
        "routed_layer_ids",
        "routed_expert_count",
        "gpu_expert_mask_shape",
        "gpu_expert_mask_sha256",
        "configured_gpu_resident_experts_per_layer",
        "ktransformers_method",
        "wrapper_id",
        "layers",
    }
)
_EXPECTED_COVERAGE_LAYER_KEYS: Final = frozenset(
    {"layer_id", "module_path", "wrapper_id", "gpu_resident_experts"}
)


class Glm47BackendError(RuntimeError):
    """Raised when live execution cannot produce complete exact evidence."""


class Glm47BackendStage(Protocol):
    @property
    def resident_gpu_experts(self) -> int: ...


class Glm47BackendEndpoint(Protocol):
    @property
    def ip(self) -> str: ...

    @property
    def port(self) -> int: ...


class Glm47BackendProcessSpec(Protocol):
    @property
    def arguments(self) -> tuple[str, ...]: ...

    @property
    def model_path(self) -> str: ...

    @property
    def service_endpoint(self) -> Glm47BackendEndpoint: ...

    @property
    def distributed_coordinator(self) -> Glm47BackendEndpoint: ...

    @property
    def stage(self) -> Glm47BackendStage: ...


class Glm47BackendRoute(Protocol):
    @property
    def resident_gpu_expert_ids(self) -> tuple[int, ...]: ...

    @property
    def selected_expert_ids(self) -> tuple[int, ...]: ...

    @property
    def cpu_expert_ids(self) -> tuple[int, ...]: ...

    @property
    def gpu_expert_ids(self) -> tuple[int, ...]: ...


class Glm47ReferenceRunner(Protocol):
    def __call__(
        self,
        *,
        model_path: Path,
        selected_expert_ids: tuple[int, ...],
        cpu_expert_ids: tuple[int, ...],
        gpu_expert_ids: tuple[int, ...],
        hidden_states: object,
    ) -> Glm47LayerOneReferenceOutputs[object]: ...


class Glm47RuntimeLoader(Protocol):
    def __call__(self) -> Glm47RuntimeBindings: ...


@dataclass(frozen=True, slots=True)
class Glm47RuntimeBindings:
    """Lazily imported pinned runtime symbols and tensor evidence backend."""

    torch: object
    inference_mode: Callable[[], AbstractContextManager[None]]
    server_args_type: object
    port_args_type: object
    model_config_type: object
    model_runner_type: object
    sampling_params_type: object
    request_type: object
    schedule_batch_type: object
    forward_batch_type: object
    standard_topk_output_type: object
    standard_dispatch_output_type: object
    speculative_algorithm_none: object
    set_envs_and_config: Callable[[object], object]
    initialize_moe_config: Callable[[object], object]
    initialize_fp8_gemm_config: Callable[[object], object]
    initialize_fp4_gemm_config: Callable[[object], object]
    get_tokenizer: Callable[..., object]
    suppress_other_loggers: Callable[[], object]
    require_mlp_sync: Callable[[object], object]
    get_kt_ep_gpu_experts_masks: Callable[[], object]
    cleanup_dist_env_and_memory: Callable[..., object]
    tensor_evidence_backend: Glm47TensorEvidenceBackend[object]


@dataclass(frozen=True, slots=True)
class Glm47RuntimeEvidence:
    server_arguments: tuple[str, ...]
    service_endpoint: str
    distributed_coordinator: str
    model_runner_nccl_port: int
    server_args_class: str
    model_config_class: str
    model_runner_class: str
    tokenizer_class: str
    tp_size: int
    pp_size: int
    expert_parallel_size: int


@dataclass(frozen=True, slots=True)
class Glm47TensorDigestEvidence:
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    finite: Literal[True]

    def as_json(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class Glm47CapturedTraceOutput:
    kind: Literal["none", "tensor", "hidden_states", "next_token_logits"]
    tensor: Glm47TensorDigestEvidence | None


@dataclass(frozen=True, slots=True)
class Glm47TraceEventEvidence:
    sequence: int
    phase: str | None
    operation: str
    layer_index: int | None
    output: Glm47CapturedTraceOutput


@dataclass(frozen=True, slots=True)
class Glm47TraceCounterEvidence:
    phase: str | None
    operation: str
    layer_index: int | None
    successful_returns: int


@dataclass(frozen=True, slots=True)
class Glm47WrapperLayerEvidence:
    layer_index: int
    expert_module_name: str
    quant_method_wrapper: Literal["kt_ep"]
    expert_count: Literal[64]
    resident_gpu_expert_ids: tuple[int, ...]
    cpu_backend_wrapper_class: Literal["NativeMoEWrapper"]
    cpu_kernel_class: Literal["AMXBF16_MOE"]
    global_expert_mask_sha256: str

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "expert_module_name": self.expert_module_name,
            "quant_method_wrapper": self.quant_method_wrapper,
            "expert_count": self.expert_count,
            "resident_gpu_expert_ids": list(self.resident_gpu_expert_ids),
            "cpu_backend_wrapper_class": self.cpu_backend_wrapper_class,
            "cpu_kernel_class": self.cpu_kernel_class,
            "global_expert_mask_sha256": self.global_expert_mask_sha256,
        }


@dataclass(frozen=True, slots=True)
class Glm47WrapperCoverageEvidence:
    global_expert_mask_sha256: str
    layers: tuple[Glm47WrapperLayerEvidence, ...]

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "global_expert_mask_sha256": self.global_expert_mask_sha256,
            "layers": [layer.as_receipt_json() for layer in self.layers],
        }


@dataclass(frozen=True, slots=True)
class Glm47LayerOneExpertProbeEvidence:
    layer_index: Literal[1]
    random_seed: Literal[20260719]
    probe_invocation_count: Literal[2]
    input_shape: tuple[int, int]
    input_dtype: Literal["torch.bfloat16"]
    input_sha256: str
    selected_expert_ids: tuple[int, int, int, int]
    repeat_selected_expert_ids: tuple[int, int, int, int]
    routing_weights: tuple[float, float, float, float]
    reference_tensor_keys: tuple[str, ...]
    cpu_expert_ids: tuple[int, ...]
    gpu_expert_ids: tuple[int, ...]
    cpu_backend_wrapper_class: Literal["NativeMoEWrapper"]
    cpu_kernel_class: Literal["AMXBF16_MOE"]
    sglang_cpu_submit_count: int
    sglang_cpu_sync_count: int
    native_cpu_submit_count: int
    native_cpu_sync_count: int
    gpu_forward_count: int
    output_merge_count: int
    global_expert_mask_sha256: str
    combined_output: Glm47LayerOneOutputEvidence
    cpu_output: Glm47LayerOneOutputEvidence
    gpu_output: Glm47LayerOneOutputEvidence | None
    hybrid_merge: Glm47LayerOneHybridMergeEvidence | None

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "random_seed": self.random_seed,
            "probe_invocation_count": self.probe_invocation_count,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "input_sha256": self.input_sha256,
            "selected_expert_ids": list(self.selected_expert_ids),
            "repeat_selected_expert_ids": list(self.repeat_selected_expert_ids),
            "routing_weights": list(self.routing_weights),
            "reference_tensor_keys": list(self.reference_tensor_keys),
            "cpu_expert_ids": list(self.cpu_expert_ids),
            "gpu_expert_ids": list(self.gpu_expert_ids),
            "cpu_backend_wrapper_class": self.cpu_backend_wrapper_class,
            "cpu_kernel_class": self.cpu_kernel_class,
            "sglang_cpu_submit_count": self.sglang_cpu_submit_count,
            "sglang_cpu_sync_count": self.sglang_cpu_sync_count,
            "native_cpu_submit_count": self.native_cpu_submit_count,
            "native_cpu_sync_count": self.native_cpu_sync_count,
            "gpu_forward_count": self.gpu_forward_count,
            "output_merge_count": self.output_merge_count,
            "global_expert_mask_sha256": self.global_expert_mask_sha256,
            "combined_output": self.combined_output.as_json(),
            "cpu_output": self.cpu_output.as_json(),
            "gpu_output": (
                self.gpu_output.as_json() if self.gpu_output is not None else None
            ),
            "hybrid_merge": (
                self.hybrid_merge.as_json() if self.hybrid_merge is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class Glm47ShortForwardInvocationEvidence:
    forward_mode: Literal["extend", "decode"]
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    kv_cache_length_before: int
    kv_cache_length_after: int
    model_forward_invocation_count: Literal[1]
    logits_shape: tuple[int, ...]
    logits_dtype: Literal["torch.float32"]
    logits_finite: Literal[True]
    logits_sha256: str
    argmax_token_id: int
    global_expert_mask_sha256_before: str
    global_expert_mask_sha256_after: str

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "forward_mode": self.forward_mode,
            "input_token_ids": list(self.input_token_ids),
            "positions": list(self.positions),
            "kv_cache_length_before": self.kv_cache_length_before,
            "kv_cache_length_after": self.kv_cache_length_after,
            "model_forward_invocation_count": self.model_forward_invocation_count,
            "logits_shape": list(self.logits_shape),
            "logits_dtype": self.logits_dtype,
            "logits_finite": self.logits_finite,
            "logits_sha256": self.logits_sha256,
            "argmax_token_id": self.argmax_token_id,
            "global_expert_mask_sha256_before": (self.global_expert_mask_sha256_before),
            "global_expert_mask_sha256_after": self.global_expert_mask_sha256_after,
        }


@dataclass(frozen=True, slots=True)
class Glm47WrapperForwardInvocationEvidence:
    layer_index: int
    extend_invocation_count: Literal[1]
    decode_invocation_count: Literal[1]

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "extend_invocation_count": self.extend_invocation_count,
            "decode_invocation_count": self.decode_invocation_count,
        }


@dataclass(frozen=True, slots=True)
class Glm47ShortForwardEvidence:
    random_seed: Literal[20260719]
    extend: Glm47ShortForwardInvocationEvidence
    decode: Glm47ShortForwardInvocationEvidence
    wrapper_invocations: tuple[Glm47WrapperForwardInvocationEvidence, ...]

    def as_receipt_json(self) -> dict[str, object]:
        return {
            "random_seed": self.random_seed,
            "extend": self.extend.as_receipt_json(),
            "decode": self.decode.as_receipt_json(),
            "wrapper_invocations": [
                invocation.as_receipt_json() for invocation in self.wrapper_invocations
            ],
        }


@dataclass(frozen=True, slots=True)
class Glm47BackendEvidence:
    runtime: Glm47RuntimeEvidence
    wrapper_coverage: Glm47WrapperCoverageEvidence
    layer_one_expert_probe: Glm47LayerOneExpertProbeEvidence
    short_forward: Glm47ShortForwardEvidence
    trace_events: tuple[Glm47TraceEventEvidence, ...]
    trace_counters: tuple[Glm47TraceCounterEvidence, ...]
    cleanup_completed: Literal[True]


@dataclass(frozen=True, slots=True)
class _Glm47BackendRunResult:
    runtime: Glm47RuntimeEvidence
    wrapper_coverage: Glm47WrapperCoverageEvidence
    layer_one_expert_probe: Glm47LayerOneExpertProbeEvidence
    short_forward: Glm47ShortForwardEvidence
    trace_events: tuple[Glm47TraceEventEvidence, ...]
    trace_counters: tuple[Glm47TraceCounterEvidence, ...]


class _NoPrefixTreeCache:
    def __init__(self, model_runner: object) -> None:
        server_args = _attribute(model_runner, "server_args")
        self.page_size = _attribute(server_args, "page_size")
        self.device = _attribute(model_runner, "device")
        self.token_to_kv_pool_allocator = _attribute(
            model_runner, "token_to_kv_pool_allocator"
        )

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True


def load_glm47_runtime_bindings() -> Glm47RuntimeBindings:
    """Import only the pinned symbols needed by the disposable live child."""

    torch_module = importlib.import_module("torch")
    server_args_module = importlib.import_module("sglang.srt.server_args")
    engine_module = importlib.import_module("sglang.srt.entrypoints.engine")
    moe_module = importlib.import_module("sglang.srt.layers.moe")
    fp8_module = importlib.import_module("sglang.srt.layers.quantization.fp8_utils")
    fp4_module = importlib.import_module("sglang.srt.layers.quantization.fp4_utils")
    model_config_module = importlib.import_module("sglang.srt.configs.model_config")
    model_runner_module = importlib.import_module(
        "sglang.srt.model_executor.model_runner"
    )
    tokenizer_module = importlib.import_module("sglang.srt.utils.hf_transformers_utils")
    common_utils_module = importlib.import_module("sglang.srt.utils.common")
    schedule_batch_module = importlib.import_module(
        "sglang.srt.managers.schedule_batch"
    )
    forward_batch_module = importlib.import_module(
        "sglang.srt.model_executor.forward_batch_info"
    )
    sampling_module = importlib.import_module("sglang.srt.sampling.sampling_params")
    speculative_module = importlib.import_module("sglang.srt.speculative.spec_info")
    topk_module = importlib.import_module("sglang.srt.layers.moe.topk")
    dispatcher_module = importlib.import_module(
        "sglang.srt.layers.moe.token_dispatcher.standard"
    )
    kt_ep_module = importlib.import_module("sglang.srt.layers.moe.kt_ep_wrapper")
    parallel_state_module = importlib.import_module(
        "sglang.srt.distributed.parallel_state"
    )
    speculative_algorithm = _attribute(speculative_module, "SpeculativeAlgorithm")
    return Glm47RuntimeBindings(
        torch=torch_module,
        inference_mode=cast(
            Callable[[], AbstractContextManager[None]],
            _attribute(torch_module, "inference_mode"),
        ),
        server_args_type=_attribute(server_args_module, "ServerArgs"),
        port_args_type=_attribute(server_args_module, "PortArgs"),
        model_config_type=_attribute(model_config_module, "ModelConfig"),
        model_runner_type=_attribute(model_runner_module, "ModelRunner"),
        sampling_params_type=_attribute(sampling_module, "SamplingParams"),
        request_type=_attribute(schedule_batch_module, "Req"),
        schedule_batch_type=_attribute(schedule_batch_module, "ScheduleBatch"),
        forward_batch_type=_attribute(forward_batch_module, "ForwardBatch"),
        standard_topk_output_type=_attribute(topk_module, "StandardTopKOutput"),
        standard_dispatch_output_type=_attribute(
            dispatcher_module, "StandardDispatchOutput"
        ),
        speculative_algorithm_none=_attribute(speculative_algorithm, "NONE"),
        set_envs_and_config=cast(
            Callable[[object], object],
            _attribute(engine_module, "_set_envs_and_config"),
        ),
        initialize_moe_config=cast(
            Callable[[object], object], _attribute(moe_module, "initialize_moe_config")
        ),
        initialize_fp8_gemm_config=cast(
            Callable[[object], object],
            _attribute(fp8_module, "initialize_fp8_gemm_config"),
        ),
        initialize_fp4_gemm_config=cast(
            Callable[[object], object],
            _attribute(fp4_module, "initialize_fp4_gemm_config"),
        ),
        get_tokenizer=cast(
            Callable[..., object], _attribute(tokenizer_module, "get_tokenizer")
        ),
        suppress_other_loggers=cast(
            Callable[[], object],
            _attribute(common_utils_module, "suppress_other_loggers"),
        ),
        require_mlp_sync=cast(
            Callable[[object], object],
            _attribute(common_utils_module, "require_mlp_sync"),
        ),
        get_kt_ep_gpu_experts_masks=cast(
            Callable[[], object],
            _attribute(kt_ep_module, "get_kt_ep_gpu_experts_masks"),
        ),
        cleanup_dist_env_and_memory=cast(
            Callable[..., object],
            _attribute(parallel_state_module, "cleanup_dist_env_and_memory"),
        ),
        tensor_evidence_backend=create_torch_glm47_tensor_evidence_backend(),
    )


def _attribute(owner: object, name: str) -> object:
    try:
        return cast(object, getattr(owner, name))
    except AttributeError as error:
        raise Glm47BackendError(
            f"{type(owner).__module__}.{type(owner).__name__} is missing {name}"
        ) from error


def _set_attribute(owner: object, name: str, value: object) -> None:
    setattr(owner, name, value)


def _call(callable_value: object, /, *args: object, **kwargs: object) -> object:
    if not callable(callable_value):
        raise Glm47BackendError("pinned runtime symbol is not callable")
    return callable_value(*args, **kwargs)


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _require_int_attribute(owner: object, name: str) -> int:
    value = _attribute(owner, name)
    if type(value) is not int:
        raise Glm47BackendError(f"runtime {name} is not an integer")
    return value


def _server_argument_vector(
    process_spec: Glm47BackendProcessSpec,
) -> tuple[str, ...]:
    arguments = process_spec.arguments
    if arguments[:2] != ("-m", "sglang.launch_server") or len(arguments) <= 2:
        raise Glm47BackendError(
            "process spec does not contain exact SGLang launch-server arguments"
        )
    return arguments[2:]


def _parse_server_args(
    process_spec: Glm47BackendProcessSpec,
    runtime: Glm47RuntimeBindings,
) -> tuple[tuple[str, ...], object]:
    server_arguments = _server_argument_vector(process_spec)
    parser = argparse.ArgumentParser()
    _call(_attribute(runtime.server_args_type, "add_cli_args"), parser)
    namespace = parser.parse_args(list(server_arguments))
    server_args = _call(
        _attribute(runtime.server_args_type, "from_cli_args"), namespace
    )
    return server_arguments, server_args


def _require_pp1_tp1_server_args(
    process_spec: Glm47BackendProcessSpec,
    server_args: object,
    route: Glm47BackendRoute,
) -> None:
    exact_values = {
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "nnodes": 1,
        "node_rank": 0,
        "dp_size": 1,
        "kt_method": "BF16",
        "kt_expert_placement_strategy": "uniform",
        "kt_max_deferred_experts_per_token": 0,
        "disable_cuda_graph": True,
        "disable_shared_experts_fusion": True,
        "nccl_port": None,
    }
    mismatches = {
        name: {"expected": expected, "actual": _attribute(server_args, name)}
        for name, expected in exact_values.items()
        if _attribute(server_args, name) != expected
    }
    resident_count = process_spec.stage.resident_gpu_experts
    if _attribute(server_args, "kt_num_gpu_experts") != resident_count:
        mismatches["kt_num_gpu_experts"] = {
            "expected": resident_count,
            "actual": _attribute(server_args, "kt_num_gpu_experts"),
        }
    if _attribute(server_args, "model_path") != process_spec.model_path:
        mismatches["model_path"] = {
            "expected": process_spec.model_path,
            "actual": _attribute(server_args, "model_path"),
        }
    if _attribute(server_args, "kt_weight_path") != process_spec.model_path:
        mismatches["kt_weight_path"] = {
            "expected": process_spec.model_path,
            "actual": _attribute(server_args, "kt_weight_path"),
        }
    if resident_count != len(route.resident_gpu_expert_ids):
        mismatches["route_resident_gpu_experts"] = {
            "expected": resident_count,
            "actual": len(route.resident_gpu_expert_ids),
        }
    if mismatches:
        raise Glm47BackendError(f"parsed ServerArgs are not exact: {mismatches}")


def _bind_model_runner_nccl_port(
    process_spec: Glm47BackendProcessSpec,
    server_args: object,
) -> int:
    """Bind PortArgs' otherwise random auxiliary port to the reserved endpoint."""

    service_endpoint = process_spec.service_endpoint
    coordinator = process_spec.distributed_coordinator
    expected_service_endpoint = f"{service_endpoint.ip}:{service_endpoint.port}"
    expected_coordinator = f"{coordinator.ip}:{coordinator.port}"
    parsed_host = _attribute(server_args, "host")
    parsed_port = _require_int_attribute(server_args, "port")
    parsed_coordinator = _attribute(server_args, "dist_init_addr")
    if (
        type(service_endpoint.port) is not int
        or not 1 <= service_endpoint.port <= 65_535
        or type(coordinator.port) is not int
        or not 1 <= coordinator.port <= 65_535
        or parsed_host != service_endpoint.ip
        or parsed_port != service_endpoint.port
        or parsed_coordinator != expected_coordinator
    ):
        raise Glm47BackendError(
            "parsed SGLang endpoints do not match the reserved process endpoints: "
            f"service={expected_service_endpoint!r}, coordinator={expected_coordinator!r}, "
            f"parsed_host={parsed_host!r}, parsed_port={parsed_port!r}, "
            f"parsed_coordinator={parsed_coordinator!r}"
        )
    if _attribute(server_args, "nccl_port") is not None:
        raise Glm47BackendError("parsed ServerArgs unexpectedly prebind nccl_port")
    _set_attribute(server_args, "nccl_port", service_endpoint.port)
    return service_endpoint.port


def _require_bound_port_args(port_args: object, expected_port: int) -> None:
    actual_port = _require_int_attribute(port_args, "nccl_port")
    if actual_port != expected_port:
        raise Glm47BackendError(
            "PortArgs did not preserve the reserved model-runner NCCL port: "
            f"expected={expected_port}, actual={actual_port}"
        )


def _validate_route(route: Glm47BackendRoute) -> None:
    selected = route.selected_expert_ids
    cpu_ids = route.cpu_expert_ids
    gpu_ids = route.gpu_expert_ids
    resident = route.resident_gpu_expert_ids
    if (
        len(selected) != 4
        or len(set(selected)) != 4
        or any(expert_id < 0 or expert_id >= 64 for expert_id in selected)
        or resident != tuple(sorted(set(resident)))
        or frozenset(cpu_ids) & frozenset(gpu_ids)
        or frozenset(cpu_ids) | frozenset(gpu_ids) != frozenset(selected)
        or cpu_ids
        != tuple(expert_id for expert_id in selected if expert_id not in resident)
        or gpu_ids
        != tuple(expert_id for expert_id in selected if expert_id in resident)
        or not cpu_ids
        or (resident and not gpu_ids)
    ):
        raise Glm47BackendError("deterministic expert route is not exact")


def _load_model_runner(
    runtime: Glm47RuntimeBindings,
    server_args: object,
    port_args: object,
) -> tuple[object, object, object]:
    runtime.suppress_other_loggers()
    model_config = _call(
        _attribute(runtime.model_config_type, "from_server_args"), server_args
    )
    tp_size = _require_int_attribute(server_args, "tp_size")
    expert_parallel_size = _require_int_attribute(server_args, "ep_size")
    if tp_size != 1 or expert_parallel_size != 1:
        raise Glm47BackendError("direct model load supports exact TP1/EP1 only")
    model_runner = _call(
        runtime.model_runner_type,
        model_config=model_config,
        mem_fraction_static=_attribute(server_args, "mem_fraction_static"),
        gpu_id=0,
        tp_rank=0,
        tp_size=tp_size,
        moe_ep_rank=0,
        moe_ep_size=expert_parallel_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=_attribute(port_args, "nccl_port"),
        server_args=server_args,
    )
    tokenizer = runtime.get_tokenizer(
        _attribute(server_args, "tokenizer_path"),
        tokenizer_mode=_attribute(server_args, "tokenizer_mode"),
        trust_remote_code=_attribute(server_args, "trust_remote_code"),
    )
    return model_config, model_runner, tokenizer


def _tensor_values(tensor: object) -> tuple[int, ...]:
    host = _call(_attribute(tensor, "detach"))
    host = _call(_attribute(host, "cpu"))
    host = _call(_attribute(host, "reshape"), -1)
    raw_values = _call(_attribute(host, "tolist"))
    if not isinstance(raw_values, list):
        raise Glm47BackendError("tensor does not contain a flat integer sequence")
    values: list[int] = []
    for value in cast(list[object], raw_values):
        if type(value) is not int:
            raise Glm47BackendError("tensor does not contain a flat integer sequence")
        values.append(value)
    return tuple(values)


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    shape = _attribute(tensor, "shape")
    if not isinstance(shape, Sequence):
        raise Glm47BackendError("tensor shape is not a sequence")
    dimensions: list[int] = []
    for dimension in cast(Sequence[object], shape):
        if type(dimension) is not int or dimension <= 0:
            raise Glm47BackendError("tensor shape is invalid")
        dimensions.append(dimension)
    return tuple(dimensions)


def _snapshot(
    runtime: Glm47RuntimeBindings,
    tensor: object,
    *,
    expected_dtype: Glm47TensorDtype | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> Glm47TensorSnapshot:
    dtype_value = str(_attribute(tensor, "dtype"))
    if dtype_value not in (GLM47_BF16_DTYPE, GLM47_FLOAT32_DTYPE):
        raise Glm47BackendError(f"unsupported evidence tensor dtype: {dtype_value}")
    dtype = dtype_value
    shape = _tensor_shape(tensor)
    if expected_dtype is not None and dtype != expected_dtype:
        raise Glm47BackendError(
            f"tensor dtype {dtype} does not match expected {expected_dtype}"
        )
    if expected_shape is not None and shape != expected_shape:
        raise Glm47BackendError(
            f"tensor shape {shape} does not match expected {expected_shape}"
        )
    snapshot = runtime.tensor_evidence_backend.snapshot(
        tensor,
        expected_dtype=dtype,
        expected_shape=shape,
    )
    snapshot.finite_values()
    return snapshot


def _tensor_digest(snapshot: Glm47TensorSnapshot) -> Glm47TensorDigestEvidence:
    return Glm47TensorDigestEvidence(
        dtype=snapshot.dtype,
        shape=snapshot.shape,
        sha256=snapshot.sha256,
        finite=True,
    )


def _hidden_states(output: object) -> object:
    return _attribute(output, "hidden_states")


def _global_mask_rows(runtime: Glm47RuntimeBindings) -> tuple[tuple[bool, ...], ...]:
    masks = runtime.get_kt_ep_gpu_experts_masks()
    host_masks = _call(_attribute(masks, "cpu"))
    raw_values = _call(_attribute(host_masks, "tolist"))
    if not isinstance(raw_values, list):
        raise Glm47BackendError("global expert mask is not a 47-row list")
    values = cast(list[object], raw_values)
    if len(values) != 47:
        raise Glm47BackendError("global expert mask is not a 47-row list")
    rows: list[tuple[bool, ...]] = []
    for row in values:
        if not isinstance(row, list):
            raise Glm47BackendError("global expert mask row is not length 64")
        row_values = cast(list[object], row)
        if len(row_values) != 64:
            raise Glm47BackendError("global expert mask row is not length 64")
        boolean_row: list[bool] = []
        for value in row_values:
            if type(value) is not bool:
                raise Glm47BackendError("global expert mask is not boolean")
            boolean_row.append(value)
        rows.append(tuple(boolean_row))
    return tuple(rows)


def _global_mask_sha256(runtime: Glm47RuntimeBindings) -> str:
    return hashlib.sha256(
        bytes(int(value) for row in _global_mask_rows(runtime) for value in row)
    ).hexdigest()


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Glm47BackendError(f"{description} is not a mapping")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if type(key) is not str:
            raise Glm47BackendError(f"{description} has a non-string key")
        result[key] = item
    return result


def _require_wrapper_coverage(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    process_spec: Glm47BackendProcessSpec,
    route: Glm47BackendRoute,
) -> Glm47WrapperCoverageEvidence:
    causal_model = _attribute(model_runner, "model")
    decoder = _attribute(causal_model, "model")
    layers = cast(Sequence[object], _attribute(decoder, "layers"))
    if len(layers) != 47:
        raise Glm47BackendError("loaded GLM model does not contain 47 layers")

    receipt = _mapping(
        _attribute(causal_model, "kt_ep_coverage_receipt"), "KT coverage receipt"
    )
    if frozenset(receipt) != _EXPECTED_COVERAGE_KEYS:
        raise Glm47BackendError("KT coverage receipt keys are not exact")
    expected_receipt = {
        "schema_version": 1,
        "model_architecture": "Glm4MoeLiteForCausalLM",
        "pipeline_parallel_size": 1,
        "layer_count": 47,
        "routed_layer_ids": list(GLM47_ROUTED_LAYER_IDS),
        "routed_expert_count": 64,
        "gpu_expert_mask_shape": [47, 64],
        "configured_gpu_resident_experts_per_layer": (
            process_spec.stage.resident_gpu_experts
        ),
        "ktransformers_method": "BF16",
        "wrapper_id": "kt_ep",
    }
    mismatches = {
        key: {"expected": expected, "actual": receipt.get(key)}
        for key, expected in expected_receipt.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise Glm47BackendError(f"KT coverage receipt is not exact: {mismatches}")

    rows = _global_mask_rows(runtime)
    mask_sha256 = _global_mask_sha256(runtime)
    if rows[0] != (True,) * GLM47_ROUTED_EXPERT_COUNT:
        raise Glm47BackendError(
            "KT global expert mask does not mark the dense layer fully resident"
        )
    if receipt.get("gpu_expert_mask_sha256") != mask_sha256:
        raise Glm47BackendError("KT coverage receipt mask SHA-256 is stale")
    expected_resident_ids = route.resident_gpu_expert_ids
    raw_receipt_layers = receipt.get("layers")
    if not isinstance(raw_receipt_layers, list):
        raise Glm47BackendError("KT coverage receipt must contain 46 layers")
    receipt_layers = cast(list[object], raw_receipt_layers)
    if len(receipt_layers) != 46:
        raise Glm47BackendError("KT coverage receipt must contain 46 layers")

    layer_evidence: list[Glm47WrapperLayerEvidence] = []
    for layer_id, receipt_layer_value in zip(
        GLM47_ROUTED_LAYER_IDS, receipt_layers, strict=True
    ):
        module_name = f"model.layers.{layer_id}.mlp.experts"
        receipt_layer = _mapping(receipt_layer_value, "KT coverage layer")
        expected_layer_receipt = {
            "layer_id": layer_id,
            "module_path": module_name,
            "wrapper_id": "kt_ep",
            "gpu_resident_experts": len(expected_resident_ids),
        }
        if (
            frozenset(receipt_layer) != _EXPECTED_COVERAGE_LAYER_KEYS
            or dict(receipt_layer) != expected_layer_receipt
        ):
            raise Glm47BackendError(f"KT coverage layer {layer_id} is not exact")

        experts = _attribute(_attribute(layers[layer_id], "mlp"), "experts")
        quant_method = _attribute(experts, "quant_method")
        wrapper = _attribute(quant_method, "wrapper")
        cpu_kernel = _attribute(wrapper, "moe")
        kt_config = _attribute(quant_method, "kt_config")
        resident_ids = tuple(
            expert_id for expert_id, resident in enumerate(rows[layer_id]) if resident
        )
        linkage = (
            _attribute(experts, "_registry_prefix") == module_name
            and _attribute(quant_method, "_quant_wrapper_id") == "kt_ep"
            and _require_int_attribute(experts, "layer_id") == layer_id
            and _require_int_attribute(
                _attribute(experts, "moe_runner_config"), "layer_id"
            )
            == layer_id
            and _require_int_attribute(kt_config, "layer_idx") == layer_id
            and _type_identity(quant_method)
            == "sglang.srt.layers.moe.kt_ep_wrapper.KTEPWrapperMethod"
            and type(wrapper).__name__ == "NativeMoEWrapper"
            and type(cpu_kernel).__name__ == "AMXBF16_MOE"
            and resident_ids == expected_resident_ids
            and _require_int_attribute(quant_method, "num_gpu_experts")
            == len(expected_resident_ids)
        )
        if not linkage:
            raise Glm47BackendError(
                f"layer {layer_id} does not have exact KTEP/AMX linkage"
            )
        layer_evidence.append(
            Glm47WrapperLayerEvidence(
                layer_index=layer_id,
                expert_module_name=module_name,
                quant_method_wrapper="kt_ep",
                expert_count=64,
                resident_gpu_expert_ids=resident_ids,
                cpu_backend_wrapper_class="NativeMoEWrapper",
                cpu_kernel_class="AMXBF16_MOE",
                global_expert_mask_sha256=mask_sha256,
            )
        )
    return Glm47WrapperCoverageEvidence(
        global_expert_mask_sha256=mask_sha256,
        layers=tuple(layer_evidence),
    )


class _TraceOutputCollector:
    def __init__(self, runtime: Glm47RuntimeBindings) -> None:
        self._runtime = runtime
        self.layer_probe_cpu_outputs: list[object] = []
        self.layer_probe_gpu_outputs: list[object] = []

    def __call__(
        self,
        probe: TraceProbe,
        phase: str | None,
        output: object,
    ) -> Glm47CapturedTraceOutput:
        if output is None:
            return Glm47CapturedTraceOutput(kind="none", tensor=None)

        kind: Literal["tensor", "hidden_states", "next_token_logits"] = "tensor"
        candidate = output
        if hasattr(output, "hidden_states"):
            kind = "hidden_states"
            candidate = _attribute(output, "hidden_states")
        elif hasattr(output, "next_token_logits"):
            kind = "next_token_logits"
            candidate = _attribute(output, "next_token_logits")

        captured_candidate = candidate
        if phase == "layer_probe" and probe.operation in (
            SGLANG_CPU_SYNC,
            GPU_METHOD_APPLY,
        ):
            detached = _call(_attribute(candidate, "detach"))
            captured_candidate = _call(_attribute(detached, "clone"))
        snapshot = _snapshot(self._runtime, captured_candidate)
        if phase == "layer_probe" and probe.operation == SGLANG_CPU_SYNC:
            self.layer_probe_cpu_outputs.append(captured_candidate)
        if phase == "layer_probe" and probe.operation == GPU_METHOD_APPLY:
            self.layer_probe_gpu_outputs.append(captured_candidate)
        return Glm47CapturedTraceOutput(
            kind=kind,
            tensor=_tensor_digest(snapshot),
        )


def _create_layer_probe_inputs(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    route: Glm47BackendRoute,
) -> tuple[object, object, object]:
    generator = _call(_attribute(runtime.torch, "Generator"), device="cpu")
    _call(_attribute(generator, "manual_seed"), GLM47_LAYER_ONE_RANDOM_SEED)
    hidden_float32_cpu = _call(
        _attribute(runtime.torch, "randn"),
        GLM47_LAYER_ONE_INPUT_SHAPE,
        generator=generator,
        dtype=_attribute(runtime.torch, "float32"),
        device="cpu",
    )
    hidden_bfloat16_cpu = _call(
        _attribute(hidden_float32_cpu, "to"),
        dtype=_attribute(runtime.torch, "bfloat16"),
    )
    hidden_bfloat16_device = _call(
        _attribute(hidden_bfloat16_cpu, "to"),
        device=_attribute(model_runner, "device"),
    )
    topk_ids = _call(
        _attribute(runtime.torch, "tensor"),
        (route.selected_expert_ids,),
        dtype=_attribute(runtime.torch, "long"),
        device=_attribute(model_runner, "device"),
    )
    topk_weights = _call(
        _attribute(runtime.torch, "tensor"),
        (GLM47_LAYER_ONE_ROUTE_WEIGHTS,),
        dtype=_attribute(runtime.torch, "float32"),
        device=_attribute(model_runner, "device"),
    )
    router_logits = _call(
        _attribute(runtime.torch, "zeros"),
        (1, GLM47_ROUTED_EXPERT_COUNT),
        dtype=_attribute(runtime.torch, "float32"),
        device=_attribute(model_runner, "device"),
    )
    topk_output = _call(
        runtime.standard_topk_output_type,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
    dispatch_output = _call(
        runtime.standard_dispatch_output_type,
        hidden_states=hidden_bfloat16_device,
        hidden_states_scale=None,
        topk_output=topk_output,
    )
    return hidden_bfloat16_cpu, hidden_bfloat16_device, dispatch_output


def _synchronize(runtime: Glm47RuntimeBindings) -> None:
    cuda = _attribute(runtime.torch, "cuda")
    _call(_attribute(cuda, "synchronize"))


def _require_trace_count(
    trace: Glm47TraceSession,
    operation: str,
    phase: str,
    expected: int,
    *,
    layer_index: int | None = 1,
) -> int:
    observed = trace.successful_return_count(
        TraceProbe(operation, layer_index), phase=phase
    )
    if observed != expected:
        raise Glm47BackendError(
            f"{phase} {operation} successful-return count is {observed}, "
            f"expected {expected}"
        )
    return observed


def _build_layer_probe_evidence(
    runtime: Glm47RuntimeBindings,
    process_spec: Glm47BackendProcessSpec,
    route: Glm47BackendRoute,
    trace: Glm47TraceSession,
    collector: _TraceOutputCollector,
    hidden_bfloat16_cpu: object,
    combined_outputs: tuple[object, object],
    reference_outputs: Glm47LayerOneReferenceOutputs[object],
    mask_sha256: str,
) -> Glm47LayerOneExpertProbeEvidence:
    if reference_outputs.cpu_only is None:
        raise Glm47BackendError("layer-one reference omitted the CPU partition")
    if len(collector.layer_probe_cpu_outputs) != 2:
        raise Glm47BackendError("layer-one probe did not capture two CPU outputs")

    evidence_backend = runtime.tensor_evidence_backend
    combined_evidence = build_glm47_layer_one_output_evidence(
        actual_output=combined_outputs[0],
        repeat_actual_output=combined_outputs[1],
        reference_output=reference_outputs.combined,
        backend=evidence_backend,
    )
    cpu_evidence = build_glm47_layer_one_output_evidence(
        actual_output=collector.layer_probe_cpu_outputs[0],
        repeat_actual_output=collector.layer_probe_cpu_outputs[1],
        reference_output=reference_outputs.cpu_only,
        backend=evidence_backend,
    )
    gpu_evidence: Glm47LayerOneOutputEvidence | None = None
    hybrid_evidence: Glm47LayerOneHybridMergeEvidence | None = None
    expected_gpu_count = 2 if route.gpu_expert_ids else 0
    if route.gpu_expert_ids:
        if (
            reference_outputs.gpu_only is None
            or len(collector.layer_probe_gpu_outputs) != 2
        ):
            raise Glm47BackendError("hybrid layer probe omitted GPU output evidence")
        gpu_evidence = build_glm47_layer_one_output_evidence(
            actual_output=collector.layer_probe_gpu_outputs[0],
            repeat_actual_output=collector.layer_probe_gpu_outputs[1],
            reference_output=reference_outputs.gpu_only,
            backend=evidence_backend,
        )
        hybrid_evidence = build_glm47_layer_one_hybrid_merge_evidence(
            combined_output=combined_outputs[0],
            cpu_output=collector.layer_probe_cpu_outputs[0],
            gpu_output=collector.layer_probe_gpu_outputs[0],
            repeat_cpu_output=collector.layer_probe_cpu_outputs[1],
            repeat_gpu_output=collector.layer_probe_gpu_outputs[1],
            backend=evidence_backend,
        )
    elif collector.layer_probe_gpu_outputs or reference_outputs.gpu_only is not None:
        raise Glm47BackendError("CPU-only layer probe unexpectedly used the GPU")

    sglang_submit_count = _require_trace_count(
        trace, SGLANG_CPU_SUBMIT, "layer_probe", 2
    )
    sglang_sync_count = _require_trace_count(trace, SGLANG_CPU_SYNC, "layer_probe", 2)
    native_submit_count = _require_trace_count(
        trace, NATIVE_WRAPPER_SUBMIT, "layer_probe", 2
    )
    native_sync_count = _require_trace_count(
        trace, NATIVE_WRAPPER_SYNC, "layer_probe", 2
    )
    gpu_forward_count = _require_trace_count(
        trace, GPU_METHOD_APPLY, "layer_probe", expected_gpu_count
    )
    output_merge_count = _require_trace_count(
        trace, LAYER_ONE_KTEP_OUTER_APPLY, "layer_probe", 2
    )
    _require_trace_count(trace, QUANT_METHOD_APPLY, "layer_probe", 2)

    selected = route.selected_expert_ids
    if len(selected) != 4:
        raise Glm47BackendError("layer-one selected expert route is not length four")
    selected_four = selected
    input_snapshot = _snapshot(
        runtime,
        hidden_bfloat16_cpu,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_INPUT_SHAPE,
    )
    return Glm47LayerOneExpertProbeEvidence(
        layer_index=1,
        random_seed=GLM47_LAYER_ONE_RANDOM_SEED,
        probe_invocation_count=2,
        input_shape=GLM47_LAYER_ONE_INPUT_SHAPE,
        input_dtype="torch.bfloat16",
        input_sha256=input_snapshot.sha256,
        selected_expert_ids=selected_four,
        repeat_selected_expert_ids=selected_four,
        routing_weights=GLM47_LAYER_ONE_ROUTE_WEIGHTS,
        reference_tensor_keys=build_glm47_layer_one_bf16_tensor_keys(selected_four),
        cpu_expert_ids=route.cpu_expert_ids,
        gpu_expert_ids=route.gpu_expert_ids,
        cpu_backend_wrapper_class="NativeMoEWrapper",
        cpu_kernel_class="AMXBF16_MOE",
        sglang_cpu_submit_count=sglang_submit_count,
        sglang_cpu_sync_count=sglang_sync_count,
        native_cpu_submit_count=native_submit_count,
        native_cpu_sync_count=native_sync_count,
        gpu_forward_count=gpu_forward_count,
        output_merge_count=output_merge_count,
        global_expert_mask_sha256=mask_sha256,
        combined_output=combined_evidence,
        cpu_output=cpu_evidence,
        gpu_output=gpu_evidence,
        hybrid_merge=hybrid_evidence,
    )


def _new_schedule_batch(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    request: object,
) -> object:
    return _call(
        _attribute(runtime.schedule_batch_type, "init_new"),
        reqs=[request],
        req_to_token_pool=_attribute(model_runner, "req_to_token_pool"),
        token_to_kv_pool_allocator=_attribute(
            model_runner, "token_to_kv_pool_allocator"
        ),
        tree_cache=_NoPrefixTreeCache(model_runner),
        model_config=_attribute(model_runner, "model_config"),
        enable_overlap=False,
        spec_algorithm=runtime.speculative_algorithm_none,
    )


def _forward_batch(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    batch: object,
) -> object:
    if bool(runtime.require_mlp_sync(_attribute(model_runner, "server_args"))):
        raise Glm47BackendError(
            "pinned PP1/TP1 short forward unexpectedly needs MLP sync"
        )
    worker_batch = _call(_attribute(batch, "get_model_worker_batch"))
    return _call(
        _attribute(runtime.forward_batch_type, "init_new"),
        worker_batch,
        model_runner,
    )


def _run_forward_step(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    trace: Glm47TraceSession,
    batch: object,
    request: object,
    *,
    phase: Literal["extend", "decode"],
    mask_sha256_before: str,
) -> tuple[Glm47ShortForwardInvocationEvidence, object]:
    kv_before = _require_int_attribute(request, "kv_committed_len")
    if phase == "extend":
        _call(_attribute(batch, "prepare_for_extend"))
    else:
        _call(_attribute(batch, "prepare_for_decode"))
    forward_batch = _forward_batch(runtime, model_runner, batch)
    input_token_ids = _tensor_values(_attribute(forward_batch, "input_ids"))
    positions = _tensor_values(_attribute(forward_batch, "positions"))
    logits_output: object | None = None
    sampled_token_ids: object | None = None
    with trace.phase(phase):
        runner_output = _call(_attribute(model_runner, "forward"), forward_batch)
        logits_output = _attribute(runner_output, "logits_output")
        sampled_token_ids = _call(
            _attribute(model_runner, "sample"), logits_output, forward_batch
        )
    if logits_output is None or sampled_token_ids is None:
        raise Glm47BackendError(f"{phase} trace phase suppressed model execution")
    _synchronize(runtime)

    logits = _attribute(logits_output, "next_token_logits")
    logits_snapshot = _snapshot(
        runtime,
        logits,
        expected_dtype=GLM47_FLOAT32_DTYPE,
        expected_shape=GLM47_LOGITS_SHAPE,
    )
    argmax = _call(_attribute(logits, "argmax"), dim=-1)
    sampled_values = _tensor_values(sampled_token_ids)
    argmax_values = _tensor_values(argmax)
    if sampled_values != argmax_values or len(argmax_values) != 1:
        raise Glm47BackendError(f"{phase} sampling is not exact greedy argmax")
    kv_after = _require_int_attribute(request, "kv_committed_len")
    mask_sha256_after = _global_mask_sha256(runtime)
    invocation = Glm47ShortForwardInvocationEvidence(
        forward_mode=phase,
        input_token_ids=input_token_ids,
        positions=positions,
        kv_cache_length_before=kv_before,
        kv_cache_length_after=kv_after,
        model_forward_invocation_count=1,
        logits_shape=logits_snapshot.shape,
        logits_dtype="torch.float32",
        logits_finite=True,
        logits_sha256=logits_snapshot.sha256,
        argmax_token_id=argmax_values[0],
        global_expert_mask_sha256_before=mask_sha256_before,
        global_expert_mask_sha256_after=mask_sha256_after,
    )
    return invocation, sampled_token_ids


def _run_short_forward(
    runtime: Glm47RuntimeBindings,
    model_runner: object,
    trace: Glm47TraceSession,
    mask_sha256: str,
) -> Glm47ShortForwardEvidence:
    sampling_params = _call(
        runtime.sampling_params_type,
        temperature=0,
        max_new_tokens=2,
    )
    request = _call(
        runtime.request_type,
        rid="glm47-runtime-validation",
        origin_input_text="",
        origin_input_ids=list(GLM47_EXTEND_TOKEN_IDS),
        sampling_params=sampling_params,
    )
    _set_attribute(request, "fill_ids", list(GLM47_EXTEND_TOKEN_IDS))
    _set_attribute(request, "logprob_start_len", -1)
    prefix_indices = _attribute(request, "prefix_indices")
    if not isinstance(prefix_indices, Sized):
        raise Glm47BackendError("request prefix indices are not sized")
    prefix_length = len(prefix_indices)
    _call(
        _attribute(request, "set_extend_input_len"),
        len(GLM47_EXTEND_TOKEN_IDS) - prefix_length,
    )
    batch = _new_schedule_batch(runtime, model_runner, request)
    extend, first_token = _run_forward_step(
        runtime,
        model_runner,
        trace,
        batch,
        request,
        phase="extend",
        mask_sha256_before=mask_sha256,
    )
    _set_attribute(batch, "output_ids", first_token)
    decode, _second_token = _run_forward_step(
        runtime,
        model_runner,
        trace,
        batch,
        request,
        phase="decode",
        mask_sha256_before=extend.global_expert_mask_sha256_after,
    )
    if (
        extend.input_token_ids != GLM47_EXTEND_TOKEN_IDS
        or extend.positions != tuple(range(8))
        or extend.kv_cache_length_before != 0
        or extend.kv_cache_length_after != 8
        or decode.input_token_ids != (extend.argmax_token_id,)
        or decode.positions != (8,)
        or decode.kv_cache_length_before != 8
        or decode.kv_cache_length_after != 9
        or extend.global_expert_mask_sha256_before != mask_sha256
        or extend.global_expert_mask_sha256_after != mask_sha256
        or decode.global_expert_mask_sha256_before != mask_sha256
        or decode.global_expert_mask_sha256_after != mask_sha256
    ):
        raise Glm47BackendError("extend/decode short-forward invariants are not exact")

    trace.require_exact_layer_apply_coverage(("extend", "decode"))
    _require_trace_count(trace, MODEL_FORWARD, "extend", 1, layer_index=None)
    _require_trace_count(trace, MODEL_FORWARD, "decode", 1, layer_index=None)
    wrapper_invocations = tuple(
        Glm47WrapperForwardInvocationEvidence(
            layer_index=layer_id,
            extend_invocation_count=1,
            decode_invocation_count=1,
        )
        for layer_id in GLM47_ROUTED_LAYER_IDS
    )
    return Glm47ShortForwardEvidence(
        random_seed=GLM47_LAYER_ONE_RANDOM_SEED,
        extend=extend,
        decode=decode,
        wrapper_invocations=wrapper_invocations,
    )


def _trace_evidence(
    trace: Glm47TraceSession,
) -> tuple[
    tuple[Glm47TraceEventEvidence, ...],
    tuple[Glm47TraceCounterEvidence, ...],
]:
    events: list[Glm47TraceEventEvidence] = []
    counts: dict[tuple[str | None, str, int | None], int] = {}
    for event in trace.events:
        output = event.captured_output
        if not event.output_captured or not isinstance(
            output, Glm47CapturedTraceOutput
        ):
            raise Glm47BackendError("trace event omitted copied output evidence")
        events.append(
            Glm47TraceEventEvidence(
                sequence=event.sequence,
                phase=event.phase,
                operation=event.probe.operation,
                layer_index=event.probe.layer_id,
                output=output,
            )
        )
        key = (event.phase, event.probe.operation, event.probe.layer_id)
        counts[key] = counts.get(key, 0) + 1
    counters = tuple(
        Glm47TraceCounterEvidence(
            phase=phase,
            operation=operation,
            layer_index=layer_index,
            successful_returns=count,
        )
        for (phase, operation, layer_index), count in counts.items()
    )
    return tuple(events), counters


def _execute_glm47_backend(
    process_spec: Glm47BackendProcessSpec,
    route: Glm47BackendRoute,
    runtime: Glm47RuntimeBindings,
    reference_runner: Glm47ReferenceRunner,
) -> _Glm47BackendRunResult:
    _validate_route(route)
    server_arguments, server_args = _parse_server_args(process_spec, runtime)
    _require_pp1_tp1_server_args(process_spec, server_args, route)

    runtime.set_envs_and_config(server_args)
    runtime.initialize_moe_config(server_args)
    runtime.initialize_fp8_gemm_config(server_args)
    runtime.initialize_fp4_gemm_config(server_args)
    model_runner_nccl_port = _bind_model_runner_nccl_port(process_spec, server_args)
    port_args = _call(_attribute(runtime.port_args_type, "init_new"), server_args)
    _require_bound_port_args(port_args, model_runner_nccl_port)
    model_config, model_runner, tokenizer = _load_model_runner(
        runtime, server_args, port_args
    )
    runtime_evidence = Glm47RuntimeEvidence(
        server_arguments=server_arguments,
        service_endpoint=(
            f"{process_spec.service_endpoint.ip}:{process_spec.service_endpoint.port}"
        ),
        distributed_coordinator=(
            f"{process_spec.distributed_coordinator.ip}:"
            f"{process_spec.distributed_coordinator.port}"
        ),
        model_runner_nccl_port=model_runner_nccl_port,
        server_args_class=_type_identity(server_args),
        model_config_class=_type_identity(model_config),
        model_runner_class=_type_identity(model_runner),
        tokenizer_class=_type_identity(tokenizer),
        tp_size=_require_int_attribute(server_args, "tp_size"),
        pp_size=_require_int_attribute(server_args, "pp_size"),
        expert_parallel_size=_require_int_attribute(server_args, "ep_size"),
    )
    wrapper_coverage = _require_wrapper_coverage(
        runtime, model_runner, process_spec, route
    )
    initial_mask_sha256 = wrapper_coverage.global_expert_mask_sha256

    trace_targets = Glm47TraceTargets.discover(model_runner)
    collector = _TraceOutputCollector(runtime)
    trace = Glm47TraceSession(trace_targets, output_copier=collector)
    layer_probe: Glm47LayerOneExpertProbeEvidence | None = None
    short_forward: Glm47ShortForwardEvidence | None = None
    trace_events: tuple[Glm47TraceEventEvidence, ...] | None = None
    trace_counters: tuple[Glm47TraceCounterEvidence, ...] | None = None
    with runtime.inference_mode(), trace:
        hidden_cpu, _hidden_device, dispatch_output = _create_layer_probe_inputs(
            runtime, model_runner, route
        )
        decoder = _attribute(_attribute(model_runner, "model"), "model")
        layers = cast(Sequence[object], _attribute(decoder, "layers"))
        layer_one_expert_module = _attribute(_attribute(layers[1], "mlp"), "experts")
        layer_one_quant_method = _attribute(layer_one_expert_module, "quant_method")
        combined_outputs: list[object] = []
        with trace.phase("layer_probe"):
            for _ in range(2):
                result = _call(
                    _attribute(layer_one_quant_method, "apply"),
                    layer_one_expert_module,
                    dispatch_output,
                )
                combined_outputs.append(_hidden_states(result))
        _synchronize(runtime)
        if len(combined_outputs) != 2:
            raise AssertionError("layer-one probe did not return twice")
        if _global_mask_sha256(runtime) != initial_mask_sha256:
            raise Glm47BackendError("layer-one probe changed the global expert mask")

        reference_outputs = reference_runner(
            model_path=Path(process_spec.model_path),
            selected_expert_ids=route.selected_expert_ids,
            cpu_expert_ids=route.cpu_expert_ids,
            gpu_expert_ids=route.gpu_expert_ids,
            hidden_states=hidden_cpu,
        )
        layer_probe = _build_layer_probe_evidence(
            runtime,
            process_spec,
            route,
            trace,
            collector,
            hidden_cpu,
            (combined_outputs[0], combined_outputs[1]),
            reference_outputs,
            initial_mask_sha256,
        )
        short_forward = _run_short_forward(
            runtime,
            model_runner,
            trace,
            initial_mask_sha256,
        )
        expected_gpu_per_forward = 1 if route.resident_gpu_expert_ids else 0
        for phase in ("extend", "decode"):
            for operation in (
                LAYER_ONE_KTEP_OUTER_APPLY,
                SGLANG_CPU_SUBMIT,
                SGLANG_CPU_SYNC,
                NATIVE_WRAPPER_SUBMIT,
                NATIVE_WRAPPER_SYNC,
            ):
                _require_trace_count(trace, operation, phase, 1)
            _require_trace_count(
                trace, GPU_METHOD_APPLY, phase, expected_gpu_per_forward
            )
        trace_events, trace_counters = _trace_evidence(trace)

    if (
        layer_probe is None
        or short_forward is None
        or trace_events is None
        or trace_counters is None
    ):
        raise Glm47BackendError("trace session suppressed backend execution")

    return _Glm47BackendRunResult(
        runtime=runtime_evidence,
        wrapper_coverage=wrapper_coverage,
        layer_one_expert_probe=layer_probe,
        short_forward=short_forward,
        trace_events=trace_events,
        trace_counters=trace_counters,
    )


def run_glm47_backend(
    process_spec: Glm47BackendProcessSpec,
    route: Glm47BackendRoute,
    *,
    runtime_loader: Glm47RuntimeLoader = load_glm47_runtime_bindings,
    reference_runner: Glm47ReferenceRunner = compute_torch_glm47_layer_one_reference,
) -> Glm47BackendEvidence:
    """Run one disposable PP1/TP1 live probe and return only after cleanup."""

    runtime = runtime_loader()
    try:
        result = _execute_glm47_backend(
            process_spec,
            route,
            runtime,
            reference_runner,
        )
    finally:
        runtime.cleanup_dist_env_and_memory(shutdown_ray=False)
    return Glm47BackendEvidence(
        runtime=result.runtime,
        wrapper_coverage=result.wrapper_coverage,
        layer_one_expert_probe=result.layer_one_expert_probe,
        short_forward=result.short_forward,
        trace_events=result.trace_events,
        trace_counters=result.trace_counters,
        cleanup_completed=True,
    )
