"""Pure assembly of the pinned GLM-4.7 model runtime validation receipt."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    canonicalize_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.receipt_io import parse_sglang_kt_strict_json
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)

if TYPE_CHECKING:
    from scripts.sglang_kt_glm47_backend import Glm47BackendEvidence
    from scripts.validate_sglang_kt_glm47_model import Glm47ModelExecutionPreflight

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]


class Glm47ReceiptAssemblyError(ValueError):
    """Raised when independently collected receipt evidence is inconsistent."""


def _json_object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict):
        raise Glm47ReceiptAssemblyError(f"{description} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise Glm47ReceiptAssemblyError(f"{description} must be a JSON object")
    return cast(JsonObject, mapping)


def _json_object_list(value: object, description: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise Glm47ReceiptAssemblyError(f"{description} must be a JSON object list")
    return [_json_object(item, description) for item in cast(list[object], value)]


def _generated_at_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise Glm47ReceiptAssemblyError("receipt generation time must be UTC")
    return value.astimezone(UTC).isoformat()


def _contract_file_sha256(
    preflight: Glm47ModelExecutionPreflight,
    role: Literal["config", "safetensors_index"],
) -> str:
    matches = tuple(
        file.sha256
        for file in preflight.model_contract.contract.files
        if file.role == role
    )
    if len(matches) != 1:
        raise Glm47ReceiptAssemblyError(
            f"model contract must contain exactly one {role} file"
        )
    return matches[0]


def _require_kernel_parent_binding(
    preflight: Glm47ModelExecutionPreflight,
    kernel_runtime: SglangKtKernelRuntimeValidationReceiptObservation,
) -> None:
    process_spec = preflight.process.process_spec
    if (
        kernel_runtime.receipt_path != str(preflight.kernel_runtime_receipt.path)
        or kernel_runtime.receipt_sha256 != preflight.kernel_runtime_receipt.sha256
        or kernel_runtime.gpu_uuid != process_spec.gpu_uuid
        or kernel_runtime.cpu_cores != process_spec.cpu_cores
        or kernel_runtime.memory_nodes != process_spec.memory_nodes
        or kernel_runtime.sglang_revision != process_spec.expected_sglang_revision
        or kernel_runtime.ktransformers_revision
        != process_spec.expected_ktransformers_revision
        or kernel_runtime.transformers_distribution_version
        != process_spec.required_transformers_distribution_version
        or kernel_runtime.transformers_module_version
        != process_spec.required_transformers_module_version
    ):
        raise Glm47ReceiptAssemblyError(
            "kernel runtime evidence does not match the bound preflight"
        )


def _require_backend_runtime_binding(
    preflight: Glm47ModelExecutionPreflight,
    backend: Glm47BackendEvidence,
) -> None:
    process_spec = preflight.process.process_spec
    runtime = backend.runtime
    if (
        runtime.server_arguments != process_spec.arguments[2:]
        or runtime.service_endpoint != str(process_spec.service_endpoint)
        or runtime.distributed_coordinator != str(process_spec.distributed_coordinator)
        or runtime.model_runner_nccl_port != process_spec.service_endpoint.port
        or runtime.tp_size != 1
        or runtime.pp_size != 1
        or runtime.expert_parallel_size != 1
    ):
        raise Glm47ReceiptAssemblyError(
            "backend runtime evidence does not match the bound process spec"
        )


def _runtime_payload(
    preflight: Glm47ModelExecutionPreflight,
    kernel_runtime: SglangKtKernelRuntimeValidationReceiptObservation,
) -> JsonObject:
    process_spec = preflight.process.process_spec
    stage = process_spec.stage
    return {
        "target_profile": process_spec.target_profile,
        "gpu_uuid": process_spec.gpu_uuid,
        "gpu_compute_capability": list(kernel_runtime.gpu_compute_capability),
        "cpu_cores": list(process_spec.cpu_cores),
        "memory_nodes": list(process_spec.memory_nodes),
        "executed_cpu_backend": "AMX_BF16",
        "model_id": str(process_spec.model_id),
        "model_revision": process_spec.expected_model_revision,
        "model_config_sha256": _contract_file_sha256(preflight, "config"),
        "sglang_revision": process_spec.expected_sglang_revision,
        "ktransformers_revision": process_spec.expected_ktransformers_revision,
        "transformers_distribution_version": (
            kernel_runtime.transformers_distribution_version
        ),
        "transformers_module_version": kernel_runtime.transformers_module_version,
        "torch_version": kernel_runtime.torch_version,
        "cuda_version": kernel_runtime.cuda_version,
        "sgl_kernel_build_id": kernel_runtime.sgl_kernel_build_id,
        "deep_gemm_build_id": kernel_runtime.deep_gemm_build_id,
        "kt_kernel_build_id": kernel_runtime.kt_kernel_build_id,
        "ktransformers_method": stage.ktransformers_method,
        "resident_gpu_experts": stage.resident_gpu_experts,
        "attention_backend": process_spec.attention_backend,
        "kv_cache_dtype": process_spec.kv_cache_dtype,
        "max_total_tokens": process_spec.plan.max_total_tokens,
        "static_memory_fraction": process_spec.plan.static_memory_fraction,
    }


def _validator_sources_payload(
    preflight: Glm47ModelExecutionPreflight,
) -> list[JsonValue]:
    return [
        {"path": source.path, "sha256": source.sha256}
        for source in preflight.validator.sources
    ]


def _parents_payload(preflight: Glm47ModelExecutionPreflight) -> JsonObject:
    snapshot = preflight.model_snapshot
    return {
        "process_spec_sha256": preflight.process.process_spec_sha256,
        "model_contract": {
            "path": preflight.model_contract.path,
            "model_path": snapshot.model_path,
            "receipt_sha256": preflight.model_contract.receipt_sha256,
            "contract_sha256": preflight.model_contract.contract_sha256,
            "config_sha256": _contract_file_sha256(preflight, "config"),
            "index_sha256": _contract_file_sha256(preflight, "safetensors_index"),
            "weight_map_entries": snapshot.weight_map_entries,
            "shard_count": snapshot.shard_count,
            "physical_weight_bytes": snapshot.physical_weight_bytes,
        },
        "kernel_runtime_validation": {
            "receipt_path": str(preflight.kernel_runtime_receipt.path),
            "receipt_sha256": preflight.kernel_runtime_receipt.sha256,
        },
    }


def require_glm47_model_runtime_validation_receipt_parent_bindings(
    payload: object,
    *,
    preflight: Glm47ModelExecutionPreflight,
    kernel_runtime: SglangKtKernelRuntimeValidationReceiptObservation,
) -> None:
    """Validate child evidence and bind every independently known parent fact."""

    _require_kernel_parent_binding(preflight, kernel_runtime)
    contents = canonicalize_sglang_kt_model_runtime_validation_receipt(payload)
    root = _json_object(
        parse_sglang_kt_strict_json(contents),
        "model runtime validation receipt",
    )
    if (
        root.get("validator_sha256") != preflight.validator.sha256
        or root.get("validator_sources") != _validator_sources_payload(preflight)
        or root.get("parents") != _parents_payload(preflight)
        or root.get("runtime") != _runtime_payload(preflight, kernel_runtime)
    ):
        raise Glm47ReceiptAssemblyError(
            "model receipt identity does not match its bound parents"
        )

    coverage = _json_object(root.get("wrapper_coverage"), "wrapper coverage")
    layers = _json_object_list(coverage.get("layers"), "wrapper coverage layers")
    expected_resident_ids = list(preflight.route.resident_gpu_expert_ids)
    if any(
        layer.get("resident_gpu_expert_ids") != expected_resident_ids
        for layer in layers
    ):
        raise Glm47ReceiptAssemblyError(
            "model receipt resident experts do not match preflight"
        )

    probe = _json_object(
        root.get("layer_one_expert_probe"),
        "layer-one expert probe",
    )
    expected_route: tuple[tuple[str, object], ...] = (
        ("selected_expert_ids", list(preflight.route.selected_expert_ids)),
        (
            "repeat_selected_expert_ids",
            list(preflight.route.selected_expert_ids),
        ),
        ("cpu_expert_ids", list(preflight.route.cpu_expert_ids)),
        ("gpu_expert_ids", list(preflight.route.gpu_expert_ids)),
    )
    if any(probe.get(name) != expected for name, expected in expected_route):
        raise Glm47ReceiptAssemblyError(
            "model receipt expert route does not match preflight"
        )


def build_glm47_model_runtime_validation_receipt_payload(
    *,
    generated_at_utc: datetime,
    preflight: Glm47ModelExecutionPreflight,
    kernel_runtime: SglangKtKernelRuntimeValidationReceiptObservation,
    backend: Glm47BackendEvidence,
) -> JsonObject:
    """Assemble and validate one deterministic schema-v1 receipt payload.

    The caller supplies time explicitly.  This function performs no filesystem,
    process, clock, Torch, SGLang, or KTransformers operations.
    """

    if not backend.cleanup_completed:
        raise Glm47ReceiptAssemblyError(
            "backend cleanup must complete before receipt assembly"
        )
    _require_backend_runtime_binding(preflight, backend)
    _require_kernel_parent_binding(preflight, kernel_runtime)
    payload: JsonObject = {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": _generated_at_utc(generated_at_utc),
        "profiler": "none",
        "validator_sha256": preflight.validator.sha256,
        "validator_sources": _validator_sources_payload(preflight),
        "failures": [],
        "parents": _parents_payload(preflight),
        "runtime": _runtime_payload(preflight, kernel_runtime),
        "wrapper_coverage": cast(
            JsonObject,
            backend.wrapper_coverage.as_receipt_json(),
        ),
        "layer_one_expert_probe": cast(
            JsonObject,
            backend.layer_one_expert_probe.as_receipt_json(),
        ),
        "short_forward": cast(
            JsonObject,
            backend.short_forward.as_receipt_json(),
        ),
    }
    require_glm47_model_runtime_validation_receipt_parent_bindings(
        payload,
        preflight=preflight,
        kernel_runtime=kernel_runtime,
    )
    return payload
