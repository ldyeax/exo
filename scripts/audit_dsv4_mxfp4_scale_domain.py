#!/usr/bin/env python3
"""Audit every routed-expert MXFP4 E8M0 scale byte without decoding tensors.

The DeepSeek-V4-Flash checkpoint uses a NumPy/PyTorch dtype that is not
available in every runtime.  This scanner therefore reads the safetensors
headers and byte ranges directly.  It never mutates or materializes the
checkpoint and emits deterministic, hashable evidence for the branch-free
AVX-512 BF16 scale-fold admission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast, final

import numpy as np

RECEIPT_FORMAT: Final = "dsv4-mxfp4-ue8m0-scale-domain"
RECEIPT_VERSION: Final = 1
SCAN_ALGORITHM: Final = "raw-safetensors-f8-e8m0-histogram-sha256-v1"
MODEL_ID: Final = "deepseek-ai/DeepSeek-V4-Flash"
INDEX_FILENAME: Final = "model.safetensors.index.json"
CONFIG_FILENAME: Final = "config.json"
EXPECTED_DTYPE: Final = "F8_E8M0"
DEFAULT_LAYER_COUNT: Final = 43
DEFAULT_EXPERT_COUNT: Final = 256
DEFAULT_SCALE_ELEMENTS: Final = 262_144
FOLD_SAFE_MINIMUM: Final = 2
FOLD_SAFE_MAXIMUM: Final = 252
E8M0_NAN_ENCODING: Final = 255
OCP_MX_SPECIFICATION: Final = "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"
MAXIMUM_JSON_BYTES: Final = 256 * 1024 * 1024
SCALE_KEY_PATTERN: Final = re.compile(
    r"^layers\.(?P<layer>[0-9]+)\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w[123])\.scale$"
)
PROJECTIONS: Final = ("w1", "w2", "w3")

JsonObject = dict[str, Any]


class ScaleDomainAuditError(RuntimeError):
    """Raised when the checkpoint cannot produce unambiguous audit evidence."""


@final
@dataclass(frozen=True)
class ScaleTensor:
    key: str
    shard_name: str
    layer: int
    expert: int
    projection: str


@final
@dataclass(frozen=True)
class AuditExpectations:
    layer_count: int = DEFAULT_LAYER_COUNT
    expert_count: int = DEFAULT_EXPERT_COUNT
    scale_elements: int = DEFAULT_SCALE_ELEMENTS

    def __post_init__(self) -> None:
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.expert_count <= 0:
            raise ValueError("expert_count must be positive")
        if self.scale_elements <= 0:
            raise ValueError("scale_elements must be positive")


@final
class _CliArguments(argparse.Namespace):
    checkpoint: Path
    output: Path
    expected_layers: int
    expected_experts: int
    expected_scale_elements: int
    allow_unsafe: bool


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> JsonObject:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ScaleDomainAuditError(f"cannot stat {label}: {path}") from error
    if size <= 0 or size > MAXIMUM_JSON_BYTES:
        raise ScaleDomainAuditError(f"{label} has invalid size {size}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScaleDomainAuditError(f"cannot parse {label}: {path}") from error
    if not isinstance(value, dict):
        raise ScaleDomainAuditError(f"{label} must be a JSON object: {path}")
    return cast(JsonObject, value)


def _validate_config(
    config: Mapping[str, object], expectations: AuditExpectations
) -> None:
    required: dict[str, object] = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": expectations.layer_count,
        "n_routed_experts": expectations.expert_count,
        "num_experts_per_tok": 6,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ScaleDomainAuditError(
            f"checkpoint is not the expected DeepSeek-V4-Flash geometry: {mismatches}"
        )


def _validate_shard_name(checkpoint: Path, shard_name: object) -> Path:
    if not isinstance(shard_name, str) or not shard_name:
        raise ScaleDomainAuditError("safetensors index has an invalid shard name")
    relative = Path(shard_name)
    if relative.is_absolute() or relative.name != shard_name:
        raise ScaleDomainAuditError(
            f"selected safetensors shard must be a basename: {shard_name!r}"
        )
    path = checkpoint / relative
    if not path.is_file():
        raise ScaleDomainAuditError(f"selected safetensors shard is absent: {path}")
    return path


def _selected_tensors(
    checkpoint: Path,
    index: Mapping[str, object],
    expectations: AuditExpectations,
) -> tuple[list[ScaleTensor], int, int | None]:
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ScaleDomainAuditError("safetensors index has no weight_map object")
    weight_map = cast(dict[object, object], raw_weight_map)
    selected: list[ScaleTensor] = []
    coverage: set[tuple[int, int, str]] = set()
    all_shards: set[str] = set()
    for raw_key, raw_shard in weight_map.items():
        if not isinstance(raw_key, str):
            raise ScaleDomainAuditError("safetensors index has a non-string key")
        if not isinstance(raw_shard, str):
            raise ScaleDomainAuditError(
                f"safetensors index has a non-string shard for {raw_key}"
            )
        all_shards.add(raw_shard)
        match = SCALE_KEY_PATTERN.fullmatch(raw_key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        coordinate = (layer, expert, projection)
        if coordinate in coverage:
            raise ScaleDomainAuditError(
                f"duplicate routed scale coordinate: {coordinate}"
            )
        coverage.add(coordinate)
        _validate_shard_name(checkpoint, raw_shard)
        selected.append(
            ScaleTensor(
                key=raw_key,
                shard_name=raw_shard,
                layer=layer,
                expert=expert,
                projection=projection,
            )
        )

    expected = {
        (layer, expert, projection)
        for layer in range(expectations.layer_count)
        for expert in range(expectations.expert_count)
        for projection in PROJECTIONS
    }
    missing = sorted(expected - coverage)
    extra = sorted(coverage - expected)
    if missing or extra:
        raise ScaleDomainAuditError(
            "routed MXFP4 scale coverage is not the exact expected Cartesian set: "
            f"missing_count={len(missing)} missing_sample={missing[:8]} "
            f"extra_count={len(extra)} extra_sample={extra[:8]}"
        )
    selected.sort(key=lambda tensor: (tensor.shard_name, tensor.key))
    raw_metadata = index.get("metadata")
    indexed_weight_bytes: int | None = None
    if isinstance(raw_metadata, dict):
        total_size = cast(dict[object, object], raw_metadata).get("total_size")
        if isinstance(total_size, int) and not isinstance(total_size, bool):
            indexed_weight_bytes = total_size
    return selected, len(all_shards), indexed_weight_bytes


def _read_exact(source: Any, size: int, *, label: str) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise ScaleDomainAuditError(f"short read while reading {label}")
    return value


def _safetensors_header(path: Path) -> tuple[JsonObject, bytes, int, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            raw_length = _read_exact(source, 8, label=f"{path} header length")
            header_bytes = struct.unpack("<Q", raw_length)[0]
            if (
                header_bytes <= 0
                or header_bytes > MAXIMUM_JSON_BYTES
                or 8 + header_bytes > file_size
            ):
                raise ScaleDomainAuditError(
                    f"invalid safetensors header size {header_bytes}: {path}"
                )
            raw_header = _read_exact(
                source, header_bytes, label=f"{path} safetensors header"
            )
    except OSError as error:
        raise ScaleDomainAuditError(f"cannot read safetensors shard: {path}") from error
    try:
        loaded = json.loads(raw_header)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ScaleDomainAuditError(
            f"cannot parse safetensors header: {path}"
        ) from error
    if not isinstance(loaded, dict):
        raise ScaleDomainAuditError(f"safetensors header is not an object: {path}")
    return cast(JsonObject, loaded), raw_header, 8 + header_bytes, file_size


def _positive_shape(value: object, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ScaleDomainAuditError(f"{key} has an invalid shape")
    shape: list[int] = []
    for dimension in value:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ScaleDomainAuditError(f"{key} has an invalid shape")
        shape.append(dimension)
    return tuple(shape)


def _tensor_range(
    metadata: object,
    *,
    tensor: ScaleTensor,
    data_bytes: int,
    expected_elements: int,
) -> tuple[int, int, tuple[int, ...]]:
    if not isinstance(metadata, dict):
        raise ScaleDomainAuditError(
            f"safetensors header is missing {tensor.key} in {tensor.shard_name}"
        )
    row = cast(dict[object, object], metadata)
    if row.get("dtype") != EXPECTED_DTYPE:
        raise ScaleDomainAuditError(
            f"{tensor.key} must use {EXPECTED_DTYPE}, got {row.get('dtype')!r}"
        )
    shape = _positive_shape(row.get("shape"), key=tensor.key)
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if element_count != expected_elements:
        raise ScaleDomainAuditError(
            f"{tensor.key} has {element_count} scale elements, expected "
            f"{expected_elements}"
        )
    raw_offsets = row.get("data_offsets")
    if (
        not isinstance(raw_offsets, list)
        or len(raw_offsets) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_offsets
        )
    ):
        raise ScaleDomainAuditError(f"{tensor.key} has invalid data_offsets")
    start, end = cast(list[int], raw_offsets)
    if start < 0 or end < start or end > data_bytes:
        raise ScaleDomainAuditError(f"{tensor.key} has out-of-range data_offsets")
    if end - start != element_count:
        raise ScaleDomainAuditError(
            f"{tensor.key} byte range does not match its one-byte E8M0 shape"
        )
    return start, end, shape


def _update_tensor_digest(
    digest: Any, *, key: str, shape: Sequence[int], contents: memoryview
) -> None:
    metadata = json.dumps(
        {"key": key, "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    digest.update(len(contents).to_bytes(8, "little"))
    digest.update(contents)


def audit_checkpoint(
    checkpoint: Path,
    *,
    expectations: AuditExpectations | None = None,
) -> JsonObject:
    if expectations is None:
        expectations = AuditExpectations()
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ScaleDomainAuditError(f"checkpoint is not a directory: {checkpoint}")
    config_path = checkpoint / CONFIG_FILENAME
    index_path = checkpoint / INDEX_FILENAME
    config = _load_json_object(config_path, label="checkpoint config")
    index = _load_json_object(index_path, label="safetensors index")
    _validate_config(config, expectations)
    tensors, index_shard_count, indexed_weight_bytes = _selected_tensors(
        checkpoint, index, expectations
    )

    tensors_by_shard: dict[str, list[ScaleTensor]] = defaultdict(list)
    for tensor in tensors:
        tensors_by_shard[tensor.shard_name].append(tensor)

    histogram = np.zeros(256, dtype=np.int64)
    total_scale_bytes = 0
    global_digest = hashlib.sha256()
    shard_receipts: list[JsonObject] = []
    for shard_name in sorted(tensors_by_shard):
        shard_path = _validate_shard_name(checkpoint, shard_name)
        header, raw_header, data_base, file_size = _safetensors_header(shard_path)
        data_bytes = file_size - data_base
        shard_histogram = np.zeros(256, dtype=np.int64)
        shard_digest = hashlib.sha256()
        shard_scale_bytes = 0
        try:
            with shard_path.open("rb") as source:
                mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    for tensor in tensors_by_shard[shard_name]:
                        start, end, shape = _tensor_range(
                            header.get(tensor.key),
                            tensor=tensor,
                            data_bytes=data_bytes,
                            expected_elements=expectations.scale_elements,
                        )
                        contents = memoryview(mapped)[
                            data_base + start : data_base + end
                        ]
                        values = np.frombuffer(contents, dtype=np.uint8)
                        counts = np.bincount(values, minlength=256)
                        histogram += counts
                        shard_histogram += counts
                        _update_tensor_digest(
                            global_digest,
                            key=tensor.key,
                            shape=shape,
                            contents=contents,
                        )
                        _update_tensor_digest(
                            shard_digest,
                            key=tensor.key,
                            shape=shape,
                            contents=contents,
                        )
                        tensor_bytes = end - start
                        total_scale_bytes += tensor_bytes
                        shard_scale_bytes += tensor_bytes
                        del counts, values, contents
                finally:
                    mapped.close()
        except OSError as error:
            raise ScaleDomainAuditError(
                f"cannot map safetensors shard: {shard_path}"
            ) from error
        shard_receipts.append(
            {
                "file_bytes": file_size,
                "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                "name": shard_name,
                "scale_bytes": shard_scale_bytes,
                "scale_content_sha256": shard_digest.hexdigest(),
                "scale_maximum": int(np.flatnonzero(shard_histogram)[-1]),
                "scale_minimum": int(np.flatnonzero(shard_histogram)[0]),
                "tensor_count": len(tensors_by_shard[shard_name]),
            }
        )

    expected_tensor_count = (
        expectations.layer_count * expectations.expert_count * len(PROJECTIONS)
    )
    expected_scale_bytes = expected_tensor_count * expectations.scale_elements
    if (
        len(tensors) != expected_tensor_count
        or total_scale_bytes != expected_scale_bytes
    ):
        raise ScaleDomainAuditError(
            "audited scale totals do not match the expected geometry: "
            f"tensors={len(tensors)}/{expected_tensor_count} "
            f"bytes={total_scale_bytes}/{expected_scale_bytes}"
        )
    nonzero_encodings = np.flatnonzero(histogram)
    if len(nonzero_encodings) == 0:
        raise ScaleDomainAuditError("routed MXFP4 scale tensors are empty")
    unsafe_count = int(histogram[:FOLD_SAFE_MINIMUM].sum()) + int(
        histogram[FOLD_SAFE_MAXIMUM + 1 :].sum()
    )
    nan_count = int(histogram[E8M0_NAN_ENCODING])
    histogram_receipt = {
        str(index): int(count)
        for index, count in enumerate(histogram.tolist())
        if count
    }
    return {
        "format": RECEIPT_FORMAT,
        "version": RECEIPT_VERSION,
        "scanner": {
            "algorithm": SCAN_ALGORITHM,
            "source_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        },
        "model": {
            "checkpoint_path": str(checkpoint),
            "config_sha256": _sha256_file(config_path),
            "index_sha256": _sha256_file(index_path),
            "indexed_weight_bytes": indexed_weight_bytes,
            "model_id": MODEL_ID,
        },
        "coverage": {
            "expected_experts_per_layer": expectations.expert_count,
            "expected_layer_count": expectations.layer_count,
            "expected_projections": list(PROJECTIONS),
            "expected_scale_elements_per_tensor": expectations.scale_elements,
            "index_shard_count": index_shard_count,
            "selected_shard_count": len(tensors_by_shard),
            "tensor_count": len(tensors),
            "total_scale_bytes": total_scale_bytes,
        },
        "scale_domain": {
            "branchless_fold_admitted": unsafe_count == 0,
            "dtype": EXPECTED_DTYPE,
            "e8m0_nan_count": nan_count,
            "e8m0_nan_encoding": E8M0_NAN_ENCODING,
            "fold_safe_maximum": FOLD_SAFE_MAXIMUM,
            "fold_safe_minimum": FOLD_SAFE_MINIMUM,
            "histogram": histogram_receipt,
            "maximum": int(nonzero_encodings[-1]),
            "minimum": int(nonzero_encodings[0]),
            "ocp_e8m0_has_no_zero": True,
            "ocp_e8m0_minimum_encoding_value": "2^-127",
            "ocp_mx_specification": OCP_MX_SPECIFICATION,
            "safe_scale_bytes": total_scale_bytes - unsafe_count,
            "scale_content_sha256": global_digest.hexdigest(),
            "unsafe_scale_bytes": unsafe_count,
        },
        "selected_shards": shard_receipts,
    }


def validate_fold_safe(receipt: Mapping[str, object]) -> None:
    raw_domain = receipt.get("scale_domain")
    if not isinstance(raw_domain, dict):
        raise ScaleDomainAuditError("receipt has no scale_domain object")
    domain = cast(dict[object, object], raw_domain)
    nan_count = domain.get("e8m0_nan_count")
    if isinstance(nan_count, int) and nan_count > 0:
        raise ScaleDomainAuditError(
            f"checkpoint contains {nan_count} OCP E8M0 NaN scale bytes (255)"
        )
    unsafe_count = domain.get("unsafe_scale_bytes")
    if unsafe_count != 0 or domain.get("branchless_fold_admitted") is not True:
        raise ScaleDomainAuditError(
            "checkpoint contains E8M0 scales outside the branchless BF16 fold "
            f"domain [{FOLD_SAFE_MINIMUM}, {FOLD_SAFE_MAXIMUM}]: "
            f"unsafe_scale_bytes={unsafe_count!r}"
        )


def publish_receipt(path: Path, receipt: Mapping[str, object]) -> str:
    contents = _canonical_json_bytes(receipt)
    digest = hashlib.sha256(contents).hexdigest()
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ScaleDomainAuditError("output path must be absolute and normalized")
    if not path.parent.is_dir():
        raise ScaleDomainAuditError(f"output parent is not a directory: {path.parent}")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ScaleDomainAuditError(
                f"cannot read existing receipt: {path}"
            ) from error
        if existing != contents:
            raise ScaleDomainAuditError(
                f"refusing to overwrite different scale-domain evidence: {path}"
            )
        return digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        temporary_path.unlink()
    except FileExistsError as error:
        raise ScaleDomainAuditError(f"receipt output already exists: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest


def _absolute_normalized_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise argparse.ArgumentTypeError("path must be absolute and normalized")
    return path


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_arguments(arguments: list[str] | None = None) -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=_absolute_normalized_path)
    parser.add_argument("--output", required=True, type=_absolute_normalized_path)
    parser.add_argument(
        "--expected-layers", type=_positive_integer, default=DEFAULT_LAYER_COUNT
    )
    parser.add_argument(
        "--expected-experts", type=_positive_integer, default=DEFAULT_EXPERT_COUNT
    )
    parser.add_argument(
        "--expected-scale-elements",
        type=_positive_integer,
        default=DEFAULT_SCALE_ELEMENTS,
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Publish forensic evidence even when branchless fold admission fails.",
    )
    namespace = _CliArguments()
    parser.parse_args(arguments, namespace=namespace)
    return namespace


def main(arguments: list[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    try:
        receipt = audit_checkpoint(
            parsed.checkpoint,
            expectations=AuditExpectations(
                layer_count=parsed.expected_layers,
                expert_count=parsed.expected_experts,
                scale_elements=parsed.expected_scale_elements,
            ),
        )
        if not parsed.allow_unsafe:
            validate_fold_safe(receipt)
        receipt_sha256 = publish_receipt(parsed.output, receipt)
    except (OSError, ScaleDomainAuditError, ValueError) as error:
        print(f"MXFP4 scale-domain audit failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "branchless_fold_admitted": cast(
                    Mapping[str, object], receipt["scale_domain"]
                )["branchless_fold_admitted"],
                "output": str(parsed.output),
                "receipt_sha256": receipt_sha256,
                "scale_content_sha256": cast(
                    Mapping[str, object], receipt["scale_domain"]
                )["scale_content_sha256"],
                "total_scale_bytes": cast(Mapping[str, object], receipt["coverage"])[
                    "total_scale_bytes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
