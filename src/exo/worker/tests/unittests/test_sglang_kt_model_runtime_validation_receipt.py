import hashlib
from pathlib import Path

import pytest

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    GLM_4_7_FLASH_BF16_INDEX_SHA256,
    MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION,
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
    SglangKtModelRuntimeValidationReceiptError,
    SglangKtModelRuntimeValidationReceiptObservation,
    build_glm_4_7_flash_layer_one_reference_tensor_keys,
    calculate_glm_4_7_flash_expert_mask_sha256,
    calculate_sglang_kt_model_runtime_validator_bundle_sha256,
    canonicalize_sglang_kt_model_runtime_validation_receipt,
    load_sglang_kt_model_runtime_validation_receipt,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtRuntimeValidationReceiptObservation,
)
from exo.worker.sglang_kt.receipt_io import canonical_sglang_kt_json

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]

PROCESS_SPEC_SHA256 = "1" * 64
MODEL_CONTRACT_RECEIPT_SHA256 = "2" * 64
KERNEL_RECEIPT_SHA256 = "3" * 64
MODEL_PATH = "/opt/exo/models/glm47"
VALIDATOR_SOURCES = (
    *(
        (
            f"/opt/exo/{relative_path}",
            hashlib.sha256(relative_path.encode()).hexdigest(),
        )
        for relative_path in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    ),
)
VALIDATOR_SHA256 = calculate_sglang_kt_model_runtime_validator_bundle_sha256(
    VALIDATOR_SOURCES
)
GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _object_field(parent: JsonObject, name: str) -> JsonObject:
    value = parent[name]
    if not isinstance(value, dict):
        raise AssertionError(f"{name} is not an object")
    return value


def _array_field(parent: JsonObject, name: str) -> list[JsonValue]:
    value = parent[name]
    if not isinstance(value, list):
        raise AssertionError(f"{name} is not an array")
    return value


def _object_items(parent: JsonObject, name: str) -> list[JsonObject]:
    value = parent[name]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise AssertionError(f"{name} is not an object list")
    return [item for item in value if isinstance(item, dict)]


def _make_runtime(profile: str, resident_gpu_experts: int) -> JsonObject:
    return {
        "target_profile": profile,
        "gpu_uuid": GPU_UUID,
        "gpu_compute_capability": [8, 6],
        "cpu_cores": [0, 1, 2, 3],
        "memory_nodes": [0],
        "executed_cpu_backend": "AMX_BF16",
        "model_id": "zai-org/GLM-4.7-Flash",
        "model_revision": GLM_4_7_FLASH_BF16_MODEL_REVISION,
        "model_config_sha256": GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        "sglang_revision": GLM_4_7_FLASH_SGLANG_REVISION,
        "ktransformers_revision": GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        "transformers_distribution_version": "5.6.0.post1",
        "transformers_module_version": "5.6.0",
        "torch_version": "2.9.1+cu128",
        "cuda_version": "12.8",
        "sgl_kernel_build_id": "5" * 64,
        "deep_gemm_build_id": "6" * 64,
        "kt_kernel_build_id": "7" * 64,
        "ktransformers_method": "BF16",
        "resident_gpu_experts": resident_gpu_experts,
        "attention_backend": "flashinfer",
        "kv_cache_dtype": "bfloat16",
        "max_total_tokens": 4096,
        "static_memory_fraction": 0.8,
    }


def _make_wrapper_layer(
    layer: int,
    resident_ids: list[JsonValue],
    global_mask_sha256: str,
) -> JsonObject:
    return {
        "layer_index": layer,
        "expert_module_name": f"model.layers.{layer}.mlp.experts",
        "quant_method_wrapper": "kt_ep",
        "expert_count": 64,
        "resident_gpu_expert_ids": resident_ids,
        "cpu_backend_wrapper_class": "NativeMoEWrapper",
        "cpu_kernel_class": "AMXBF16_MOE",
        "global_expert_mask_sha256": global_mask_sha256,
    }


def _make_numerical_evidence() -> JsonObject:
    return {
        "shape": [1, 2048],
        "execution_dtype": "torch.bfloat16",
        "reference_dtype": "torch.float32",
        "output_finite": True,
        "reference_finite": True,
        "mean_absolute_error": 0.001,
        "maximum_absolute_error": 0.01,
        "reference_mean_absolute": 0.5,
        "relative_l1_error": 0.002,
        "relative_l1_tolerance": 0.02,
    }


def _make_output_evidence(
    actual_output_sha256: str,
    reference_output_sha256: str,
) -> JsonObject:
    return {
        "actual_output_sha256": actual_output_sha256,
        "repeat_actual_output_sha256": actual_output_sha256,
        "reference_output_sha256": reference_output_sha256,
        "numerical": _make_numerical_evidence(),
    }


def _make_forward_invocation(
    mode: str,
    input_token_ids: list[JsonValue],
    positions: list[JsonValue],
    cache_before: int,
    cache_after: int,
    logits_sha256: str,
    argmax_token_id: int,
    global_mask_sha256: str,
) -> JsonObject:
    return {
        "forward_mode": mode,
        "input_token_ids": input_token_ids,
        "positions": positions,
        "kv_cache_length_before": cache_before,
        "kv_cache_length_after": cache_after,
        "model_forward_invocation_count": 1,
        "logits_shape": [1, 154880],
        "logits_dtype": "torch.float32",
        "logits_finite": True,
        "logits_sha256": logits_sha256,
        "argmax_token_id": argmax_token_id,
        "global_expert_mask_sha256_before": global_mask_sha256,
        "global_expert_mask_sha256_after": global_mask_sha256,
    }


def make_receipt(*, cpu_control: bool) -> JsonObject:
    profile = (
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        if cpu_control
        else GLM_4_7_FLASH_TARGET_PROFILE
    )
    resident_expert_ids = () if cpu_control else (0, 1)
    resident_ids: list[JsonValue] = list(resident_expert_ids)
    global_mask_sha256 = calculate_glm_4_7_flash_expert_mask_sha256(
        tuple(resident_expert_ids for _ in range(46))
    )
    selected_expert_ids = (0, 1, 2, 3) if cpu_control else (0, 1, 4, 5)
    selected_ids: list[JsonValue] = list(selected_expert_ids)
    cpu_ids: list[JsonValue] = list(selected_expert_ids) if cpu_control else [4, 5]
    gpu_ids: list[JsonValue] = [] if cpu_control else [0, 1]
    combined_output = _make_output_evidence("a" * 64, "d" * 64)
    cpu_output = (
        _make_output_evidence("a" * 64, "d" * 64)
        if cpu_control
        else _make_output_evidence("e" * 64, "f" * 64)
    )
    gpu_output = None if cpu_control else _make_output_evidence("7" * 64, "8" * 64)
    return {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": "2026-07-19T12:34:56+00:00",
        "profiler": "none",
        "validator_sha256": VALIDATOR_SHA256,
        "validator_sources": [
            {"path": path, "sha256": sha256} for path, sha256 in VALIDATOR_SOURCES
        ],
        "failures": [],
        "parents": {
            "process_spec_sha256": PROCESS_SPEC_SHA256,
            "model_contract": {
                "path": "/opt/exo/contracts/glm47.json",
                "model_path": MODEL_PATH,
                "receipt_sha256": MODEL_CONTRACT_RECEIPT_SHA256,
                "contract_sha256": GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
                "config_sha256": GLM_4_7_FLASH_BF16_CONFIG_SHA256,
                "index_sha256": GLM_4_7_FLASH_BF16_INDEX_SHA256,
                "weight_map_entries": 9_703,
                "shard_count": 48,
                "physical_weight_bytes": 62_444_175_504,
            },
            "kernel_runtime_validation": {
                "receipt_path": "/var/lib/exo/receipts/kernel.json",
                "receipt_sha256": KERNEL_RECEIPT_SHA256,
            },
        },
        "runtime": _make_runtime(profile, len(resident_ids)),
        "wrapper_coverage": {
            "global_expert_mask_sha256": global_mask_sha256,
            "layers": [
                _make_wrapper_layer(
                    layer,
                    resident_ids.copy(),
                    global_mask_sha256,
                )
                for layer in range(1, 47)
            ],
        },
        "layer_one_expert_probe": {
            "layer_index": 1,
            "random_seed": 20260719,
            "probe_invocation_count": 2,
            "input_shape": [1, 2048],
            "input_dtype": "torch.bfloat16",
            "input_sha256": "9" * 64,
            "selected_expert_ids": selected_ids,
            "repeat_selected_expert_ids": selected_ids.copy(),
            "routing_weights": [0.4, 0.3, 0.2, 0.1],
            "reference_tensor_keys": list(
                build_glm_4_7_flash_layer_one_reference_tensor_keys(selected_expert_ids)
            ),
            "cpu_expert_ids": cpu_ids,
            "gpu_expert_ids": gpu_ids,
            "cpu_backend_wrapper_class": "NativeMoEWrapper",
            "cpu_kernel_class": "AMXBF16_MOE",
            "sglang_cpu_submit_count": 2,
            "sglang_cpu_sync_count": 2,
            "native_cpu_submit_count": 2,
            "native_cpu_sync_count": 2,
            "gpu_forward_count": 0 if cpu_control else 2,
            "output_merge_count": 2,
            "global_expert_mask_sha256": global_mask_sha256,
            "combined_output": combined_output,
            "cpu_output": cpu_output,
            "gpu_output": gpu_output,
            "hybrid_merge": (
                None
                if cpu_control
                else {
                    "combined_output_sha256": "a" * 64,
                    "cpu_output_sha256": "e" * 64,
                    "gpu_output_sha256": "7" * 64,
                    "merged_backend_output_sha256": "a" * 64,
                    "repeat_merged_backend_output_sha256": "a" * 64,
                    "reference_merged_backend_output_sha256": "6" * 64,
                    "numerical": _make_numerical_evidence(),
                }
            ),
        },
        "short_forward": {
            "random_seed": 20260719,
            "extend": _make_forward_invocation(
                "extend",
                list(range(1, 9)),
                list(range(8)),
                0,
                8,
                "b" * 64,
                42,
                global_mask_sha256,
            ),
            "decode": _make_forward_invocation(
                "decode",
                [42],
                [8],
                8,
                9,
                "c" * 64,
                43,
                global_mask_sha256,
            ),
            "wrapper_invocations": [
                {
                    "layer_index": layer,
                    "extend_invocation_count": 1,
                    "decode_invocation_count": 1,
                }
                for layer in range(1, 47)
            ],
        },
    }


def _write_receipt(path: Path, receipt: JsonObject) -> str:
    contents = canonical_sglang_kt_json(receipt)
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def _load(
    path: Path, *, expected_receipt_sha256: str | None = None
) -> SglangKtModelRuntimeValidationReceiptObservation:
    return load_sglang_kt_model_runtime_validation_receipt(
        path,
        expected_validator_sha256=VALIDATOR_SHA256,
        expected_process_spec_sha256=PROCESS_SPEC_SHA256,
        expected_model_contract_receipt_sha256=MODEL_CONTRACT_RECEIPT_SHA256,
        expected_kernel_receipt_sha256=KERNEL_RECEIPT_SHA256,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def test_expert_mask_hash_matches_pinned_runtime_dense_row_layout() -> None:
    resident_experts = tuple((0, 2) for _ in range(46))
    expected_mask = bytearray([1] * 64)
    for _ in range(46):
        expected_mask.extend(
            bytearray((1 if expert_id in {0, 2} else 0) for expert_id in range(64))
        )

    assert calculate_glm_4_7_flash_expert_mask_sha256(resident_experts) == (
        hashlib.sha256(expected_mask).hexdigest()
    )


def test_reference_tensor_keys_match_the_expert_major_oracle_contract() -> None:
    selected_experts = (0, 1, 4, 5)

    assert build_glm_4_7_flash_layer_one_reference_tensor_keys(
        selected_experts
    ) == tuple(
        f"model.layers.1.mlp.experts.{expert_id}.{projection}_proj.weight"
        for expert_id in selected_experts
        for projection in ("gate", "up", "down")
    )


def test_validator_source_bundle_hash_uses_canonical_sorted_evidence() -> None:
    expected_payload: JsonObject = {
        "canonicalization": MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION,
        "sources": [
            {"path": path, "sha256": sha256} for path, sha256 in VALIDATOR_SOURCES
        ],
    }

    assert (
        calculate_sglang_kt_model_runtime_validator_bundle_sha256(VALIDATOR_SOURCES)
        == hashlib.sha256(canonical_sglang_kt_json(expected_payload)).hexdigest()
    )


@pytest.mark.parametrize(
    "sources",
    (
        tuple(reversed(VALIDATOR_SOURCES)),
        (("relative/validator.py", "4" * 64),),
        (VALIDATOR_SOURCES[0], VALIDATOR_SOURCES[0]),
    ),
)
def test_validator_source_bundle_rejects_ambiguous_sources(
    sources: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError):
        calculate_sglang_kt_model_runtime_validator_bundle_sha256(sources)


@pytest.mark.parametrize("cpu_control", (True, False))
def test_canonicalizes_valid_payload_for_loader_round_trip(
    tmp_path: Path,
    cpu_control: bool,
) -> None:
    payload = make_receipt(cpu_control=cpu_control)

    contents = canonicalize_sglang_kt_model_runtime_validation_receipt(payload)

    assert contents == canonical_sglang_kt_json(payload)
    path = tmp_path / "constructed-model-runtime-receipt.json"
    path.write_bytes(contents)
    receipt_sha256 = hashlib.sha256(contents).hexdigest()
    observation = _load(path, expected_receipt_sha256=receipt_sha256)
    assert observation.receipt_sha256 == receipt_sha256
    assert observation.resident_gpu_experts == (0 if cpu_control else 2)


@pytest.mark.parametrize(
    "mutation",
    ("extra_field", "invalid_route", "nonfinite_metric"),
)
def test_canonicalizer_rejects_invalid_payload_mutations(mutation: str) -> None:
    payload = make_receipt(cpu_control=False)
    if mutation == "extra_field":
        payload["claimed_capabilities"] = ["kt_bf16_amx_executed_v1"]
    elif mutation == "invalid_route":
        probe = _object_field(payload, "layer_one_expert_probe")
        probe["gpu_expert_ids"] = [0]
    elif mutation == "nonfinite_metric":
        probe = _object_field(payload, "layer_one_expert_probe")
        combined_output = _object_field(probe, "combined_output")
        numerical = _object_field(combined_output, "numerical")
        numerical["relative_l1_error"] = float("nan")
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(SglangKtModelRuntimeValidationReceiptError):
        canonicalize_sglang_kt_model_runtime_validation_receipt(payload)


@pytest.mark.parametrize(
    ("cpu_control", "route_capability"),
    (
        (True, "glm47_flash_bf16_cpu_routed_experts_executed_v1"),
        (False, "kt_bf16_cpu_gpu_hybrid_executed_v1"),
    ),
)
def test_loads_golden_receipt_and_derives_capabilities(
    tmp_path: Path,
    cpu_control: bool,
    route_capability: str,
) -> None:
    path = tmp_path / "model-runtime-receipt.json"
    receipt_sha256 = _write_receipt(path, make_receipt(cpu_control=cpu_control))

    observation = _load(path, expected_receipt_sha256=receipt_sha256)

    assert observation.receipt_path == str(path)
    assert observation.receipt_sha256 == receipt_sha256
    assert observation.process_spec_sha256 == PROCESS_SPEC_SHA256
    assert observation.model_contract_receipt_sha256 == MODEL_CONTRACT_RECEIPT_SHA256
    assert observation.model_path == MODEL_PATH
    assert observation.model_weight_map_entries == 9_703
    assert observation.model_shard_count == 48
    assert observation.model_physical_weight_bytes == 62_444_175_504
    assert observation.kernel_runtime_validation_receipt_sha256 == KERNEL_RECEIPT_SHA256
    assert observation.model_config_sha256 == GLM_4_7_FLASH_BF16_CONFIG_SHA256
    assert observation.ktransformers_wrapped_expert_layers == tuple(range(1, 47))
    assert observation.capabilities[-1] == route_capability
    assert "kt_bf16_amx_executed_v1" in observation.capabilities
    assert set(SglangKtRuntimeValidationReceiptObservation.model_fields).issubset(
        observation.__class__.model_fields
    )


@pytest.mark.parametrize(
    "parent",
    ("process_spec", "model_contract", "kernel"),
)
def test_rejects_receipt_bound_to_a_different_parent(
    tmp_path: Path,
    parent: str,
) -> None:
    path = tmp_path / "model-runtime-receipt.json"
    _write_receipt(path, make_receipt(cpu_control=False))
    arguments = {
        "expected_validator_sha256": VALIDATOR_SHA256,
        "expected_process_spec_sha256": PROCESS_SPEC_SHA256,
        "expected_model_contract_receipt_sha256": MODEL_CONTRACT_RECEIPT_SHA256,
        "expected_kernel_receipt_sha256": KERNEL_RECEIPT_SHA256,
    }
    arguments[
        f"expected_{parent}_receipt_sha256"
        if parent == "kernel"
        else (
            "expected_model_contract_receipt_sha256"
            if parent == "model_contract"
            else "expected_process_spec_sha256"
        )
    ] = "f" * 64

    with pytest.raises(SglangKtModelRuntimeValidationReceiptError, match="parent"):
        load_sglang_kt_model_runtime_validation_receipt(path, **arguments)


def test_rejects_receipt_from_an_unpinned_validator(tmp_path: Path) -> None:
    path = tmp_path / "model-runtime-receipt.json"
    _write_receipt(path, make_receipt(cpu_control=False))

    with pytest.raises(
        SglangKtModelRuntimeValidationReceiptError,
        match="validator",
    ):
        load_sglang_kt_model_runtime_validation_receipt(
            path,
            expected_validator_sha256="f" * 64,
            expected_process_spec_sha256=PROCESS_SPEC_SHA256,
            expected_model_contract_receipt_sha256=(MODEL_CONTRACT_RECEIPT_SHA256),
            expected_kernel_receipt_sha256=KERNEL_RECEIPT_SHA256,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "status",
        "profiler",
        "failure",
        "claimed_capabilities",
        "validator_source_hash",
        "validator_source_order",
        "contract_hash",
        "config_hash",
        "index_hash",
        "missing_wrapper_layer",
        "expert_module_name",
        "quant_wrapper",
        "expert_count",
        "resident_count",
        "resident_identity_mask",
        "cpu_wrapper",
        "cpu_kernel",
        "layer_mask",
        "input_shape",
        "repeat_route",
        "routing_weights",
        "reference_keys",
        "route_partition",
        "native_cpu_submit_count",
        "repeat_combined_output",
        "missing_gpu_output",
        "missing_hybrid_merge",
        "merge_cpu_output",
        "merged_backend_output",
        "repeat_merge",
        "numerical_tolerance",
        "extend_tokens",
        "nonfinite_logits",
        "decode_shape",
        "missing_forward_layer",
        "forward_count",
        "changed_forward_mask",
    ),
)
def test_rejects_mutated_execution_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt = make_receipt(cpu_control=False)
    validator_sources = _object_items(receipt, "validator_sources")
    parents = _object_field(receipt, "parents")
    contract = _object_field(parents, "model_contract")
    runtime = _object_field(receipt, "runtime")
    coverage = _object_field(receipt, "wrapper_coverage")
    layers = _object_items(coverage, "layers")
    probe = _object_field(receipt, "layer_one_expert_probe")
    combined_output = _object_field(probe, "combined_output")
    numerical = _object_field(combined_output, "numerical")
    hybrid_merge = _object_field(probe, "hybrid_merge")
    short_forward = _object_field(receipt, "short_forward")
    extend = _object_field(short_forward, "extend")
    decode = _object_field(short_forward, "decode")
    invocations = _object_items(short_forward, "wrapper_invocations")

    if mutation == "status":
        receipt["status"] = "failed"
    elif mutation == "profiler":
        receipt["profiler"] = "vtune"
    elif mutation == "failure":
        receipt["failures"] = ["failed check"]
    elif mutation == "claimed_capabilities":
        receipt["capabilities"] = ["kt_bf16_amx_executed_v1"]
    elif mutation == "validator_source_hash":
        validator_sources[0]["sha256"] = "0" * 64
    elif mutation == "validator_source_order":
        _array_field(receipt, "validator_sources").reverse()
    elif mutation == "contract_hash":
        contract["contract_sha256"] = "f" * 64
    elif mutation == "config_hash":
        contract["config_sha256"] = "f" * 64
    elif mutation == "index_hash":
        contract["index_sha256"] = "f" * 64
    elif mutation == "missing_wrapper_layer":
        _array_field(coverage, "layers").pop()
    elif mutation == "expert_module_name":
        layers[0]["expert_module_name"] = "model.layers.2.mlp.experts"
    elif mutation == "quant_wrapper":
        layers[0]["quant_method_wrapper"] = "unwrapped"
    elif mutation == "expert_count":
        layers[0]["expert_count"] = 63
    elif mutation == "resident_count":
        runtime["resident_gpu_experts"] = 3
    elif mutation == "resident_identity_mask":
        layers[1]["resident_gpu_expert_ids"] = [2, 3]
    elif mutation == "cpu_wrapper":
        layers[0]["cpu_backend_wrapper_class"] = "AMXMoEWrapper"
    elif mutation == "cpu_kernel":
        layers[0]["cpu_kernel_class"] = "AVX512_MOE"
    elif mutation == "layer_mask":
        layers[0]["global_expert_mask_sha256"] = "f" * 64
    elif mutation == "input_shape":
        probe["input_shape"] = [2, 2048]
    elif mutation == "repeat_route":
        probe["repeat_selected_expert_ids"] = [1, 0, 4, 5]
    elif mutation == "routing_weights":
        probe["routing_weights"] = [0.3, 0.3, 0.2, 0.2]
    elif mutation == "reference_keys":
        reference_keys = _array_field(probe, "reference_tensor_keys")
        reference_keys[0], reference_keys[1] = reference_keys[1], reference_keys[0]
    elif mutation == "route_partition":
        probe["gpu_expert_ids"] = [0]
    elif mutation == "native_cpu_submit_count":
        probe["native_cpu_submit_count"] = 1
    elif mutation == "repeat_combined_output":
        combined_output["repeat_actual_output_sha256"] = "0" * 64
    elif mutation == "missing_gpu_output":
        probe["gpu_output"] = None
    elif mutation == "missing_hybrid_merge":
        probe["hybrid_merge"] = None
    elif mutation == "merge_cpu_output":
        hybrid_merge["cpu_output_sha256"] = "0" * 64
    elif mutation == "merged_backend_output":
        hybrid_merge["merged_backend_output_sha256"] = "0" * 64
        hybrid_merge["repeat_merged_backend_output_sha256"] = "0" * 64
    elif mutation == "repeat_merge":
        hybrid_merge["repeat_merged_backend_output_sha256"] = "0" * 64
    elif mutation == "numerical_tolerance":
        numerical["relative_l1_error"] = 0.03
    elif mutation == "extend_tokens":
        extend["input_token_ids"] = [1, 2, 3]
    elif mutation == "nonfinite_logits":
        extend["logits_finite"] = False
    elif mutation == "decode_shape":
        decode["logits_shape"] = [1, 154879]
    elif mutation == "missing_forward_layer":
        _array_field(short_forward, "wrapper_invocations").pop()
    elif mutation == "forward_count":
        invocations[0]["decode_invocation_count"] = 2
    elif mutation == "changed_forward_mask":
        decode["global_expert_mask_sha256_after"] = "f" * 64
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    path = tmp_path / "mutated-model-runtime-receipt.json"
    _write_receipt(path, receipt)
    with pytest.raises(SglangKtModelRuntimeValidationReceiptError):
        _load(path)


@pytest.mark.parametrize(
    "mutation",
    ("combined_cpu_mismatch", "gpu_output", "hybrid_merge"),
)
def test_rejects_cpu_control_backend_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt = make_receipt(cpu_control=True)
    probe = _object_field(receipt, "layer_one_expert_probe")
    if mutation == "combined_cpu_mismatch":
        cpu_output = _object_field(probe, "cpu_output")
        cpu_output["actual_output_sha256"] = "0" * 64
        cpu_output["repeat_actual_output_sha256"] = "0" * 64
    elif mutation == "gpu_output":
        probe["gpu_output"] = _make_output_evidence("7" * 64, "8" * 64)
    elif mutation == "hybrid_merge":
        hybrid_receipt = make_receipt(cpu_control=False)
        hybrid_probe = _object_field(hybrid_receipt, "layer_one_expert_probe")
        probe["hybrid_merge"] = hybrid_probe["hybrid_merge"]
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    path = tmp_path / "mutated-cpu-control-model-runtime-receipt.json"
    _write_receipt(path, receipt)
    with pytest.raises(SglangKtModelRuntimeValidationReceiptError):
        _load(path)


def test_rejects_duplicate_keys_before_pydantic_validation(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-key-receipt.json"
    contents = canonical_sglang_kt_json(make_receipt(cpu_control=True)).replace(
        b'"status":"passed"',
        b'"status":"passed","status":"passed"',
        1,
    )
    path.write_bytes(contents)

    with pytest.raises(SglangKtModelRuntimeValidationReceiptError):
        _load(path)


def test_rejects_wrong_raw_receipt_hash(tmp_path: Path) -> None:
    path = tmp_path / "model-runtime-receipt.json"
    _write_receipt(path, make_receipt(cpu_control=True))

    with pytest.raises(SglangKtModelRuntimeValidationReceiptError, match="expected"):
        _load(path, expected_receipt_sha256="f" * 64)
