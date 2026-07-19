import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    KERNEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES,
    SglangKtKernelRuntimeValidationReceiptError,
    load_sglang_kt_kernel_runtime_validation_receipt,
)

_MACHINE_DWAGON_V4_RECEIPT = Path(
    "/var/lib/exo/benchmarks/glm47-kt-kernel-dwagon-20260719-v4/"
    "runtime-validation-receipt.json"
)
_GOLDEN_DWAGON_V4_RECEIPT = (
    Path(__file__).parents[1]
    / "fixtures"
    / "sglang_kt"
    / "glm47_kernel_runtime_v1_dwagon_v4.json"
)
_REAL_DWAGON_V4_RECEIPT_SHA256 = (
    "b4f6fd1718bb3145a17c97cf8113bbfcd186416cfde3cd0fcc9eada301b78eef"
)

type JsonObject = dict[str, object]
type JsonPathPart = str | int


def _golden_receipt_json() -> str:
    return _GOLDEN_DWAGON_V4_RECEIPT.read_text()


def _golden_receipt_document() -> JsonObject:
    return cast(JsonObject, json.loads(_golden_receipt_json()))


def _write_receipt(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "runtime-validation-receipt.json"
    path.write_text(contents)
    return path


def _write_document(tmp_path: Path, document: JsonObject) -> Path:
    return _write_receipt(
        tmp_path,
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError("test fixture value is not a JSON object")
    object_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in object_mapping):
        raise TypeError("test fixture value is not a string-keyed JSON object")
    return cast(JsonObject, object_mapping)


def _json_array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("test fixture value is not a JSON array")
    return cast(list[object], value)


def _replace_json_path(
    document: JsonObject,
    path: tuple[JsonPathPart, ...],
    replacement: object,
) -> None:
    if not path:
        raise ValueError("test mutation path must be nonempty")
    current: object = document
    for part in path[:-1]:
        current = (
            _json_object(current)[part]
            if isinstance(part, str)
            else _json_array(current)[part]
        )
    last_part = path[-1]
    if isinstance(last_part, str):
        _json_object(current)[last_part] = replacement
    else:
        _json_array(current)[last_part] = replacement


def test_loads_real_dwagon_v4_golden_and_derives_only_kernel_capability() -> None:
    observation = load_sglang_kt_kernel_runtime_validation_receipt(
        _GOLDEN_DWAGON_V4_RECEIPT,
        expected_receipt_sha256=_REAL_DWAGON_V4_RECEIPT_SHA256,
    )

    assert observation.receipt_path == str(_GOLDEN_DWAGON_V4_RECEIPT)
    assert observation.receipt_sha256 == _REAL_DWAGON_V4_RECEIPT_SHA256
    assert observation.receipt_size_bytes == _GOLDEN_DWAGON_V4_RECEIPT.stat().st_size
    assert observation.schema_version == 1
    assert observation.capabilities == ("kt_bf16_amx_executed_v1",)
    assert observation.gpu_uuid == "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
    assert observation.gpu_compute_capability == (8, 6)
    assert observation.cpu_cores == tuple(range(16))
    assert observation.allowed_memory_nodes == (0, 1)
    assert observation.memory_nodes == (0,)
    assert observation.threads_per_subpool == (16,)
    assert observation.sglang_revision == GLM_4_7_FLASH_SGLANG_REVISION
    assert observation.ktransformers_revision == GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    assert observation.torch_version == "2.9.1+cu128"
    assert observation.cuda_version == "12.8"


@pytest.mark.skipif(
    not _MACHINE_DWAGON_V4_RECEIPT.is_file(),
    reason="the machine-local dwagon v4 validation receipt is unavailable",
)
def test_checked_in_golden_matches_machine_dwagon_v4_receipt() -> None:
    machine_contents = _MACHINE_DWAGON_V4_RECEIPT.read_bytes()
    golden_contents = _GOLDEN_DWAGON_V4_RECEIPT.read_bytes()

    assert machine_contents == golden_contents
    assert hashlib.sha256(machine_contents).hexdigest() == (
        _REAL_DWAGON_V4_RECEIPT_SHA256
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("schema_version",), 2),
        (("status",), "failed"),
        (("profiler",), "py-spy"),
        (("host", "profiler"), "perf"),
        (("failures",), ["injected failure"]),
        (("capabilities",), []),
        (
            ("capabilities",),
            ["kt_bf16_amx_executed_v1", "kt_bf16_amx_executed_v1"],
        ),
        (
            ("capabilities",),
            ["kt_bf16_amx_executed_v1", "kt_bf16_cpu_gpu_hybrid_executed_v1"],
        ),
        (("generated_at_utc",), "2026-07-19T12:00:00"),
    ),
)
def test_rejects_nonpassing_or_widened_receipt_claims(
    tmp_path: Path,
    path: tuple[JsonPathPart, ...],
    replacement: object,
) -> None:
    document = _golden_receipt_document()
    _replace_json_path(document, path, replacement)
    receipt_path = _write_document(tmp_path, document)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(receipt_path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            ("config", "gpu_uuid"),
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        ),
        (("cuda", "compute_capability"), [9, 0]),
        (("host", "process", "affinity_cpu_ids"), [0]),
        (("config", "worker_pool", "numa_nodes"), [1]),
        (("config", "worker_pool", "threads_per_subpool"), [33]),
        (("provenance", "sglang_revision"), "0" * 40),
        (("provenance", "receipt_path"), "/runtime/other-build-receipt.json"),
        (("runtime_identity", "sgl_kernel_build_id"), "not-a-sha256"),
        (("runtime_identity", "torch_cuda_version"), "13.0"),
        (
            ("provenance", "embedded_provenance", 0, "location"),
            "../unsafe.py",
        ),
    ),
)
def test_rejects_changed_runtime_or_resource_identity(
    tmp_path: Path,
    path: tuple[JsonPathPart, ...],
    replacement: object,
) -> None:
    document = _golden_receipt_document()
    _replace_json_path(document, path, replacement)
    receipt_path = _write_document(tmp_path, document)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(receipt_path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("amx", 1, "qlen"), 1),
        (("amx", 0, "route"), "cuda_stream"),
        (("amx", 0, "extension_sha256"), "e" * 64),
        (("amx", 0, "numerical", "relative_l1_error"), -1.0),
        (("amx", 1, "numerical", "relative_l1_error"), 0.5),
        (("cuda_stream", "default_cuda_stream_id"), 713_304_304),
        (("cuda_stream", "cuda_output_consumer_count"), 0),
        (("cuda_stream", "cpu_input_observed_checksum"), 0.0),
        (("cuda_stream", "cuda_output_consumed_l1"), 20.0),
    ),
)
def test_rejects_incomplete_amx_or_cuda_stream_execution(
    tmp_path: Path,
    path: tuple[JsonPathPart, ...],
    replacement: object,
) -> None:
    document = _golden_receipt_document()
    _replace_json_path(document, path, replacement)
    receipt_path = _write_document(tmp_path, document)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(receipt_path)


def test_rejects_unexpected_keys_and_camel_case_schema(tmp_path: Path) -> None:
    extra_key_document = _golden_receipt_document()
    extra_key_document["unexpected"] = True
    extra_key_path = _write_document(tmp_path, extra_key_document)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(extra_key_path)

    camel_case_document = _golden_receipt_document()
    camel_case_document["schemaVersion"] = camel_case_document.pop("schema_version")
    camel_case_path = _write_document(tmp_path, camel_case_document)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(camel_case_path)


def test_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    receipt_text = _golden_receipt_json().rstrip()
    duplicate_key_path = _write_receipt(
        tmp_path,
        receipt_text[:-1] + ', "status": "failed"}\n',
    )

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(duplicate_key_path)

    document = _golden_receipt_document()
    _replace_json_path(
        document,
        ("cuda", "numerical", "relative_l1_error"),
        1e300,
    )
    overflow_text = json.dumps(document, allow_nan=False).replace("1e+300", "1e400", 1)
    overflow_path = _write_receipt(tmp_path, overflow_text)

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(overflow_path)


def test_rejects_wrong_expected_hash_and_unsafe_file(tmp_path: Path) -> None:
    receipt_path = _write_receipt(tmp_path, _golden_receipt_json())

    with pytest.raises(
        SglangKtKernelRuntimeValidationReceiptError,
        match="expected SHA-256",
    ):
        load_sglang_kt_kernel_runtime_validation_receipt(
            receipt_path,
            expected_receipt_sha256="0" * 64,
        )

    symlink = tmp_path / "receipt-link.json"
    symlink.symlink_to(receipt_path)
    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(symlink)


def test_rejects_oversized_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "oversized.json"
    receipt_path.write_bytes(
        b"{" + b" " * KERNEL_RUNTIME_VALIDATION_RECEIPT_MAXIMUM_BYTES + b"}"
    )

    with pytest.raises(SglangKtKernelRuntimeValidationReceiptError):
        load_sglang_kt_kernel_runtime_validation_receipt(receipt_path)
