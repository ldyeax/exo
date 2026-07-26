#!/usr/bin/env python3
"""Materialize the immutable GLM-5.2 AMXINT4 + Ampere W8 checkpoint.

The source Hugging Face checkpoint contains the complete BF16 model.  This
offline converter writes a much smaller, self-contained GPU-body checkpoint:

* routed and MTP expert tensors are omitted because the separately attested
  KTransformers AMXINT4 checkpoint remains authoritative for them;
* admitted large LinearBase matrices are converted to symmetric,
  per-output-channel W8A16 GPTQ layout (packed INT32 qweight plus BF16 scale);
* each NSA ``kv_b_proj`` is transposed/split offline into compact per-head KC
  and VC matrices in a backend-neutral GPTQ W8 layout used directly by Triton
  or compact-to-compact repacked for grouped Marlin;
* embeddings, norms, routers, sensitive scalars, and the MTP ``eh_proj`` remain
  byte-identical to the BF16 source.

SGLang consumes the compact representation directly.  Ordinary linears and
the Marlin ``kv_b_proj`` backend perform only an INT8-to-INT8 layout repack;
the Triton ``kv_b_proj`` backend bit-extracts the serialized GPTQ words
directly.  Neither backend expands persistent weights back to BF16.

The destination is staged on the destination filesystem, content-addressed,
renamed atomically, and made read-only.  Existing destinations are never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type TensorDisposition = Literal[
    "quantize_int8",
    "quantize_mla_kv_b_w8",
    "preserve_bf16",
    "omit_expert",
]
type OutputTensorKind = Literal[
    "quantized_qweight",
    "quantized_scale",
    "mla_kc_qweight",
    "mla_kc_scale",
    "mla_vc_qweight",
    "mla_vc_scale",
    "preserved",
]

EXPECTED_SOURCE_INDEX_SHA256: Final = (
    "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
)
EXPECTED_SOURCE_CONFIG_SHA256: Final = (
    "817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073"
)
EXPECTED_EXPERT_CONTENT_ID: Final = (
    "3cfb9c32388cd021a725e60022ff312688f90cdfb72ac257f2851cffb5903a07"
)
DEFAULT_SOURCE_PATH: Final = Path("/mnt/sanic/glm52")
DEFAULT_EXPERT_WEIGHT_PATH: Final = Path("/mnt/sanic/glm52-AMXINT4")
DEFAULT_EXPERT_MANIFEST_PATH: Final = Path(
    "/var/lib/exo/shared-host-weights/glm52-amxint4-manifest.json"
)
INDEX_FILENAME: Final = "model.safetensors.index.json"
MANIFEST_FILENAME: Final = "hybrid-checkpoint-manifest.json"
QUANT_CONFIG_FILENAME: Final = "quantize_config.json"
MANIFEST_KIND: Final = "glm52_amxint4_ampere_w8a16_hybrid_checkpoint"
MANIFEST_SCHEMA_VERSION: Final = 1
EXPERT_MANIFEST_KIND: Final = "kt_shared_host_weights_manifest"
EXPERT_CONTENT_KIND: Final = "kt_shared_host_weights_content"
EXPERT_MANIFEST_SCHEMA_VERSION: Final = 1
EXPERT_NUMA_NODES: Final = (0, 1)
DEFAULT_MAXIMUM_SHARD_BYTES: Final = 4 * 1024**3
DEFAULT_QUANTIZATION_CHUNK_BYTES: Final = 128 * 1024**2
MAXIMUM_JSON_BYTES: Final = 64 * 1024**2
COPY_CHUNK_BYTES: Final = 16 * 1024**2
SAFETENSORS_ALIGNMENT: Final = 8
SOURCE_EXPERT_PATTERN: Final = re.compile(r"\.mlp\.experts\.\d+\.")
MLA_KV_B_WEIGHT_PATTERN: Final = re.compile(
    r"^model\.layers\.\d+\.self_attn\.kv_b_proj\.weight$"
)
QUANTIZED_WEIGHT_PATTERN: Final = re.compile(
    r"^(?:"
    r"lm_head|"
    r"model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|o_proj)|"
    r"self_attn\.indexer\.(?:wq_b|wk)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)|"
    r"mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)"
    r")"
    r")\.weight$"
)
ANCILLARY_FILENAMES: Final = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DTYPE_WIDTH_BYTES: Final[Mapping[str, int]] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
QUANTIZATION_DYNAMIC_EXCLUSIONS: Final[tuple[str, ...]] = (
    r"-:.*\.mlp\.experts$",
    r"-:.*\.self_attn\.indexer\.weights_proj$",
)
MLA_NUM_HEADS: Final = 64
MLA_QK_NOPE_HEAD_DIM: Final = 192
MLA_V_HEAD_DIM: Final = 256
MLA_KV_LORA_RANK: Final = 512
MARLIN_W8_PACK_FACTOR: Final = 4
MARLIN_MOE_BLOCK_SIZE_M: Final = 8


class HybridCheckpointError(RuntimeError):
    """Raised when conversion cannot preserve the exact hybrid contract."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    mode: int


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int
    modified_ns: int
    changed_ns: int
    mode: int


@dataclass(frozen=True, slots=True)
class ExpertArtifactIdentity:
    weight_path: Path
    directory_identity: DirectoryIdentity
    manifest_path: Path
    manifest_identity: FileIdentity
    manifest_sha256: str
    content_id: str
    file_identities: Mapping[Path, FileIdentity]


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard_name: str
    absolute_path: Path
    absolute_data_start: int
    absolute_data_end: int

    @property
    def size_bytes(self) -> int:
        return self.absolute_data_end - self.absolute_data_start


@dataclass(frozen=True, slots=True)
class OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    kind: OutputTensorKind
    source_name: str


@dataclass(frozen=True, slots=True)
class OutputUnit:
    source: SourceTensor
    tensors: tuple[OutputTensor, ...]

    @property
    def size_bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors)


@dataclass(frozen=True, slots=True)
class OutputShard:
    filename: str
    units: tuple[OutputUnit, ...]

    @property
    def tensors(self) -> tuple[OutputTensor, ...]:
        return tuple(tensor for unit in self.units for tensor in unit.tensors)

    @property
    def payload_size_bytes(self) -> int:
        return sum(unit.size_bytes for unit in self.units)


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    source_root: Path
    source_index_sha256: str
    source_config_sha256: str
    source_tensors: Mapping[str, SourceTensor]
    source_file_identities: Mapping[Path, FileIdentity]
    dispositions: Mapping[str, TensorDisposition]
    output_shards: tuple[OutputShard, ...]
    omitted_expert_names_sha256: str
    expert_artifact: ExpertArtifactIdentity

    @property
    def output_tensor_count(self) -> int:
        return sum(len(shard.tensors) for shard in self.output_shards)

    @property
    def output_payload_bytes(self) -> int:
        return sum(shard.payload_size_bytes for shard in self.output_shards)


def _canonical_json_bytes(value: JsonValue, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_identity(path: Path) -> FileIdentity:
    try:
        status = path.lstat()
    except OSError as error:
        raise HybridCheckpointError(f"cannot stat required file {path}") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise HybridCheckpointError(
            f"required file must be a regular non-symlink: {path}"
        )
    return FileIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        size_bytes=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
        mode=status.st_mode,
    )


def _directory_identity(path: Path) -> DirectoryIdentity:
    try:
        status = path.lstat()
    except OSError as error:
        raise HybridCheckpointError(f"cannot stat required directory {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(status.st_mode):
        raise HybridCheckpointError(f"required directory must be a non-symlink: {path}")
    return DirectoryIdentity(
        device=status.st_dev,
        inode=status.st_ino,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
        mode=status.st_mode,
    )


def _read_bounded_json(path: Path, description: str) -> JsonObject:
    identity_before = _file_identity(path)
    if not 0 < identity_before.size_bytes <= MAXIMUM_JSON_BYTES:
        raise HybridCheckpointError(
            f"{description} has unsafe size {identity_before.size_bytes}: {path}"
        )
    try:
        contents = path.read_bytes()
        value = cast(object, json.loads(contents))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridCheckpointError(f"cannot read {description}: {path}") from error
    if _file_identity(path) != identity_before:
        raise HybridCheckpointError(f"{description} changed while reading: {path}")
    if not isinstance(value, dict):
        raise HybridCheckpointError(f"{description} must be a JSON object: {path}")
    return cast(JsonObject, value)


def _checked_shape(value: object, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in value
    ):
        raise HybridCheckpointError(f"{tensor_name} has an invalid safetensors shape")
    return tuple(cast(list[int], value))


def _checked_offsets(value: object, tensor_name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(offset, int) for offset in value)
    ):
        raise HybridCheckpointError(
            f"{tensor_name} has invalid safetensors data offsets"
        )
    start, end = cast(list[int], value)
    if start < 0 or end <= start:
        raise HybridCheckpointError(
            f"{tensor_name} has an empty or reversed safetensors extent"
        )
    return start, end


def _read_safetensors_header(
    path: Path,
) -> tuple[JsonObject, int, FileIdentity]:
    identity = _file_identity(path)
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise HybridCheckpointError(
                    f"safetensors shard has a truncated header length: {path}"
                )
            header_length = struct.unpack("<Q", raw_length)[0]
            if (
                header_length <= 0
                or header_length > MAXIMUM_JSON_BYTES
                or 8 + header_length >= identity.size_bytes
            ):
                raise HybridCheckpointError(
                    f"safetensors shard has unsafe header length: {path}"
                )
            raw_header = stream.read(header_length)
    except OSError as error:
        raise HybridCheckpointError(f"cannot read safetensors shard {path}") from error
    if len(raw_header) != header_length:
        raise HybridCheckpointError(f"truncated safetensors header: {path}")
    try:
        value = cast(object, json.loads(raw_header))
    except json.JSONDecodeError as error:
        raise HybridCheckpointError(
            f"invalid safetensors header JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise HybridCheckpointError(f"safetensors header is not an object: {path}")
    return cast(JsonObject, value), 8 + header_length, identity


def _validate_glm52_config(
    config: JsonObject,
    config_path: Path,
    expected_sha256: str | None,
) -> str:
    config_sha256 = _sha256_file(config_path)
    if expected_sha256 is not None and config_sha256 != expected_sha256:
        raise HybridCheckpointError(
            "source config SHA-256 differs from the pinned GLM-5.2 artifact: "
            f"expected={expected_sha256}, actual={config_sha256}"
        )
    expected_values: Mapping[str, JsonValue] = {
        "hidden_size": 6_144,
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2_048,
        "num_attention_heads": MLA_NUM_HEADS,
        "q_lora_rank": 2_048,
        "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
        "v_head_dim": MLA_V_HEAD_DIM,
        "kv_lora_rank": MLA_KV_LORA_RANK,
    }
    failures = [
        f"{key}={config.get(key)!r} (expected {expected!r})"
        for key, expected in expected_values.items()
        if config.get(key) != expected
    ]
    architectures = config.get("architectures")
    if architectures != ["GlmMoeDsaForCausalLM"]:
        failures.append(
            f"architectures={architectures!r} (expected ['GlmMoeDsaForCausalLM'])"
        )
    if failures:
        raise HybridCheckpointError(
            "source config is not the admitted GLM-5.2 topology: " + "; ".join(failures)
        )
    return config_sha256


def _load_source_tensors(
    source_root: Path,
    *,
    expected_index_sha256: str | None,
) -> tuple[
    Mapping[str, SourceTensor],
    Mapping[Path, FileIdentity],
    str,
]:
    index_path = source_root / INDEX_FILENAME
    index = _read_bounded_json(index_path, "source safetensors index")
    index_sha256 = _sha256_file(index_path)
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise HybridCheckpointError(
            "source index SHA-256 differs from the pinned GLM-5.2 artifact: "
            f"expected={expected_index_sha256}, actual={index_sha256}"
        )
    raw_weight_map = index.get("weight_map")
    if (
        not isinstance(raw_weight_map, dict)
        or not raw_weight_map
        or any(
            not isinstance(name, str) or not isinstance(filename, str)
            for name, filename in raw_weight_map.items()
        )
    ):
        raise HybridCheckpointError(
            "source safetensors index weight_map must be nonempty string-to-string"
        )
    weight_map = cast(dict[str, str], raw_weight_map)
    if any(
        Path(filename).name != filename or not filename.endswith(".safetensors")
        for filename in weight_map.values()
    ):
        raise HybridCheckpointError("source index contains unsafe shard paths")

    headers: dict[str, tuple[JsonObject, int, FileIdentity]] = {}
    identities: dict[Path, FileIdentity] = {
        index_path: _file_identity(index_path),
    }
    tensors: dict[str, SourceTensor] = {}
    for tensor_name in sorted(weight_map):
        shard_name = weight_map[tensor_name]
        shard_path = source_root / shard_name
        if shard_name not in headers:
            headers[shard_name] = _read_safetensors_header(shard_path)
            identities[shard_path] = headers[shard_name][2]
        header, data_start, shard_identity = headers[shard_name]
        raw_row = header.get(tensor_name)
        if not isinstance(raw_row, dict):
            raise HybridCheckpointError(
                f"source shard {shard_name} lacks indexed tensor {tensor_name}"
            )
        row = cast(JsonObject, raw_row)
        dtype = row.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPE_WIDTH_BYTES:
            raise HybridCheckpointError(
                f"{tensor_name} has unsupported safetensors dtype {dtype!r}"
            )
        shape = _checked_shape(row.get("shape"), tensor_name)
        offset_start, offset_end = _checked_offsets(
            row.get("data_offsets"),
            tensor_name,
        )
        expected_size = math.prod(shape) * DTYPE_WIDTH_BYTES[dtype]
        if offset_end - offset_start != expected_size:
            raise HybridCheckpointError(
                f"{tensor_name} extent differs from dtype/shape byte size"
            )
        absolute_start = data_start + offset_start
        absolute_end = data_start + offset_end
        if absolute_end > shard_identity.size_bytes:
            raise HybridCheckpointError(
                f"{tensor_name} extent exceeds source shard {shard_name}"
            )
        tensors[tensor_name] = SourceTensor(
            name=tensor_name,
            dtype=dtype,
            shape=shape,
            shard_name=shard_name,
            absolute_path=shard_path,
            absolute_data_start=absolute_start,
            absolute_data_end=absolute_end,
        )
    return tensors, identities, index_sha256


def classify_source_tensor(tensor: SourceTensor) -> TensorDisposition:
    """Return the immutable storage policy for one source tensor."""

    if SOURCE_EXPERT_PATTERN.search(tensor.name):
        return "omit_expert"
    if MLA_KV_B_WEIGHT_PATTERN.fullmatch(tensor.name):
        expected_shape = (
            MLA_NUM_HEADS * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM),
            MLA_KV_LORA_RANK,
        )
        if tensor.dtype != "BF16" or tensor.shape != expected_shape:
            raise HybridCheckpointError(
                "MLA kv_b matrix differs from the admitted GLM-5.2 shape: "
                f"{tensor.name} dtype={tensor.dtype} shape={tensor.shape}, "
                f"expected=BF16{expected_shape}"
            )
        return "quantize_mla_kv_b_w8"
    if QUANTIZED_WEIGHT_PATTERN.fullmatch(tensor.name):
        if tensor.dtype != "BF16" or len(tensor.shape) != 2:
            raise HybridCheckpointError(
                f"quantized matrix must be rank-2 BF16: {tensor.name}"
            )
        output_features, input_features = tensor.shape
        if input_features % 128 != 0 or output_features % 64 != 0:
            raise HybridCheckpointError(
                "quantized matrix is not GPTQ-Marlin aligned "
                f"(N%64, K%128): {tensor.name} shape={tensor.shape}"
            )
        return "quantize_int8"
    return "preserve_bf16"


def _output_unit(
    tensor: SourceTensor,
    disposition: TensorDisposition,
) -> OutputUnit | None:
    if disposition == "omit_expert":
        return None
    if disposition == "preserve_bf16":
        return OutputUnit(
            source=tensor,
            tensors=(
                OutputTensor(
                    name=tensor.name,
                    dtype=tensor.dtype,
                    shape=tensor.shape,
                    size_bytes=tensor.size_bytes,
                    kind="preserved",
                    source_name=tensor.name,
                ),
            ),
        )

    if disposition == "quantize_mla_kv_b_w8":
        stem = tensor.name.removesuffix(".weight")
        kc_qweight = OutputTensor(
            name=f"{stem}.kc_qweight",
            dtype="I32",
            shape=(
                MLA_NUM_HEADS,
                MLA_QK_NOPE_HEAD_DIM // MARLIN_W8_PACK_FACTOR,
                MLA_KV_LORA_RANK,
            ),
            size_bytes=MLA_NUM_HEADS * MLA_QK_NOPE_HEAD_DIM * MLA_KV_LORA_RANK,
            kind="mla_kc_qweight",
            source_name=tensor.name,
        )
        kc_scales = OutputTensor(
            name=f"{stem}.kc_scales",
            dtype="BF16",
            shape=(MLA_NUM_HEADS, 1, MLA_KV_LORA_RANK),
            size_bytes=2 * MLA_NUM_HEADS * MLA_KV_LORA_RANK,
            kind="mla_kc_scale",
            source_name=tensor.name,
        )
        vc_qweight = OutputTensor(
            name=f"{stem}.vc_qweight",
            dtype="I32",
            shape=(
                MLA_NUM_HEADS,
                MLA_KV_LORA_RANK // MARLIN_W8_PACK_FACTOR,
                MLA_V_HEAD_DIM,
            ),
            size_bytes=MLA_NUM_HEADS * MLA_KV_LORA_RANK * MLA_V_HEAD_DIM,
            kind="mla_vc_qweight",
            source_name=tensor.name,
        )
        vc_scales = OutputTensor(
            name=f"{stem}.vc_scales",
            dtype="BF16",
            shape=(MLA_NUM_HEADS, 1, MLA_V_HEAD_DIM),
            size_bytes=2 * MLA_NUM_HEADS * MLA_V_HEAD_DIM,
            kind="mla_vc_scale",
            source_name=tensor.name,
        )
        return OutputUnit(
            source=tensor,
            tensors=(kc_qweight, kc_scales, vc_qweight, vc_scales),
        )

    output_features, input_features = tensor.shape
    stem = tensor.name.removesuffix(".weight")
    qweight = OutputTensor(
        name=f"{stem}.qweight",
        dtype="I32",
        shape=(input_features // 4, output_features),
        size_bytes=input_features * output_features,
        kind="quantized_qweight",
        source_name=tensor.name,
    )
    scales = OutputTensor(
        name=f"{stem}.scales",
        dtype="BF16",
        shape=(1, output_features),
        size_bytes=2 * output_features,
        kind="quantized_scale",
        source_name=tensor.name,
    )
    return OutputUnit(source=tensor, tensors=(qweight, scales))


def _plan_output_shards(
    units: Sequence[OutputUnit],
    maximum_shard_bytes: int,
) -> tuple[OutputShard, ...]:
    if maximum_shard_bytes <= 0:
        raise HybridCheckpointError("maximum shard bytes must be positive")
    shards: list[list[OutputUnit]] = []
    current: list[OutputUnit] = []
    current_bytes = 0
    for unit in units:
        if unit.size_bytes > maximum_shard_bytes:
            raise HybridCheckpointError(
                f"{unit.source.name} output unit is larger than the shard limit"
            )
        if current and current_bytes + unit.size_bytes > maximum_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(unit)
        current_bytes += unit.size_bytes
    if current:
        shards.append(current)
    width = max(5, len(str(len(shards))))
    return tuple(
        OutputShard(
            filename=(
                f"model-{index:0{width}d}-of-{len(shards):0{width}d}.safetensors"
            ),
            units=tuple(shard_units),
        )
        for index, shard_units in enumerate(shards, start=1)
    )


def _expert_artifact_identity(
    expert_weight_path: Path,
    expert_manifest_path: Path,
    expected_content_id: str,
) -> ExpertArtifactIdentity:
    if not expert_weight_path.is_absolute() or not expert_weight_path.is_dir():
        raise HybridCheckpointError(
            f"expert weight path must be an existing absolute directory: "
            f"{expert_weight_path}"
        )
    directory_identity = _directory_identity(expert_weight_path)
    if directory_identity.mode & 0o222:
        raise HybridCheckpointError(
            f"AMXINT4 expert directory must be immutable: {expert_weight_path}"
        )
    manifest_identity = _file_identity(expert_manifest_path)
    if manifest_identity.mode & 0o222:
        raise HybridCheckpointError(
            f"AMXINT4 expert manifest must be immutable: {expert_manifest_path}"
        )
    manifest = _read_bounded_json(expert_manifest_path, "AMXINT4 expert manifest")
    if set(manifest) != {
        "content_id",
        "files",
        "kind",
        "numa_nodes",
        "schema_version",
    }:
        raise HybridCheckpointError(
            "AMXINT4 expert manifest has unexpected or missing fields"
        )
    if (
        manifest.get("kind") != EXPERT_MANIFEST_KIND
        or manifest.get("schema_version") != EXPERT_MANIFEST_SCHEMA_VERSION
        or manifest.get("numa_nodes") != list(EXPERT_NUMA_NODES)
    ):
        raise HybridCheckpointError(
            "AMXINT4 expert manifest has an unsupported schema or NUMA order"
        )
    content_id = manifest.get("content_id")
    if content_id != expected_content_id:
        raise HybridCheckpointError(
            "AMXINT4 expert content ID differs from the pinned artifact: "
            f"expected={expected_content_id}, actual={content_id!r}"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise HybridCheckpointError("AMXINT4 expert manifest has no files")

    content_rows: list[JsonValue] = []
    relative_paths: list[str] = []
    expected_hashes: dict[str, str] = {}
    expected_sizes: dict[str, int] = {}
    for raw_file in cast(list[object], raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise HybridCheckpointError(
                "AMXINT4 expert manifest file row has an invalid schema"
            )
        file_row = cast(dict[str, object], raw_file)
        relative_path = file_row.get("path")
        sha256 = file_row.get("sha256")
        size_bytes = file_row.get("size_bytes")
        if not isinstance(relative_path, str) or not relative_path:
            raise HybridCheckpointError(
                "AMXINT4 expert manifest contains an invalid file path"
            )
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or len(pure_path.parts) != 1
            or pure_path.name != relative_path
            or relative_path in {".", ".."}
            or "\x00" in relative_path
        ):
            raise HybridCheckpointError(
                f"AMXINT4 expert manifest contains an unsafe path: {relative_path!r}"
            )
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise HybridCheckpointError(
                f"AMXINT4 expert manifest metadata is invalid: {relative_path}"
            )
        relative_paths.append(relative_path)
        expected_hashes[relative_path] = sha256
        expected_sizes[relative_path] = size_bytes
        content_rows.append(
            {
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise HybridCheckpointError(
            "AMXINT4 expert manifest paths must be unique and sorted"
        )
    calculated_content_id = _sha256_bytes(
        _canonical_json_bytes(
            {
                "files": content_rows,
                "kind": EXPERT_CONTENT_KIND,
                "schema_version": EXPERT_MANIFEST_SCHEMA_VERSION,
            }
        )
    )
    if calculated_content_id != expected_content_id:
        raise HybridCheckpointError(
            "AMXINT4 expert content ID does not authenticate its file table"
        )

    actual_entries = tuple(sorted(expert_weight_path.iterdir()))
    actual_names = tuple(entry.name for entry in actual_entries)
    if actual_names != tuple(relative_paths):
        raise HybridCheckpointError(
            "AMXINT4 expert directory contents differ from its manifest"
        )
    file_identities: dict[Path, FileIdentity] = {}
    for entry in actual_entries:
        identity_before = _file_identity(entry)
        if identity_before.mode & 0o222:
            raise HybridCheckpointError(
                f"AMXINT4 expert file must be immutable: {entry}"
            )
        if identity_before.size_bytes != expected_sizes[entry.name]:
            raise HybridCheckpointError(
                f"AMXINT4 expert file size differs from its manifest: {entry}"
            )
        observed_sha256 = _sha256_file(entry)
        identity_after = _file_identity(entry)
        if identity_after != identity_before:
            raise HybridCheckpointError(
                f"AMXINT4 expert file changed while hashing: {entry}"
            )
        if observed_sha256 != expected_hashes[entry.name]:
            raise HybridCheckpointError(
                f"AMXINT4 expert file hash differs from its manifest: {entry}"
            )
        file_identities[entry] = identity_before

    manifest_sha256 = _sha256_file(expert_manifest_path)
    if _file_identity(expert_manifest_path) != manifest_identity:
        raise HybridCheckpointError(
            "AMXINT4 expert manifest changed while being authenticated"
        )
    if _directory_identity(expert_weight_path) != directory_identity:
        raise HybridCheckpointError(
            "AMXINT4 expert directory changed while being authenticated"
        )
    return ExpertArtifactIdentity(
        weight_path=expert_weight_path,
        directory_identity=directory_identity,
        manifest_path=expert_manifest_path,
        manifest_identity=manifest_identity,
        manifest_sha256=manifest_sha256,
        content_id=expected_content_id,
        file_identities=file_identities,
    )


def build_conversion_plan(
    source_root: Path,
    *,
    expert_weight_path: Path,
    expert_manifest_path: Path,
    expected_source_index_sha256: str | None,
    expected_source_config_sha256: str | None,
    expected_expert_content_id: str,
    maximum_shard_bytes: int,
) -> ConversionPlan:
    """Build and validate the exact conversion without writing output."""

    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise HybridCheckpointError(f"source root is not a directory: {source_root}")
    config_path = source_root / "config.json"
    config = _read_bounded_json(config_path, "source model config")
    config_sha256 = _validate_glm52_config(
        config,
        config_path,
        expected_source_config_sha256,
    )
    source_tensors, identities, index_sha256 = _load_source_tensors(
        source_root,
        expected_index_sha256=expected_source_index_sha256,
    )
    identities = dict(identities)
    identities[config_path] = _file_identity(config_path)

    dispositions: dict[str, TensorDisposition] = {
        name: classify_source_tensor(tensor) for name, tensor in source_tensors.items()
    }
    omitted_names = sorted(
        name
        for name, disposition in dispositions.items()
        if disposition == "omit_expert"
    )
    if not omitted_names or not any(
        name.startswith("model.layers.78.mlp.experts.") for name in omitted_names
    ):
        raise HybridCheckpointError(
            "source does not contain both routed and MTP expert tensors"
        )
    quantized_names = [
        name
        for name, disposition in dispositions.items()
        if disposition == "quantize_int8"
    ]
    required_quantized_names = {
        "lm_head.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.78.self_attn.q_a_proj.weight",
        "model.layers.78.mlp.shared_experts.gate_proj.weight",
    }
    missing_quantized = sorted(required_quantized_names.difference(quantized_names))
    if missing_quantized:
        raise HybridCheckpointError(
            f"source lacks required GPU-body matrices: {missing_quantized}"
        )
    mla_kv_b_names = [
        name
        for name, disposition in dispositions.items()
        if disposition == "quantize_mla_kv_b_w8"
    ]
    if len(mla_kv_b_names) != 79:
        raise HybridCheckpointError(
            "expected one compact MLA kv_b matrix for each target and MTP "
            f"layer, observed {len(mla_kv_b_names)}"
        )
    required_mla_names = {
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.78.self_attn.kv_b_proj.weight",
    }
    if not required_mla_names.issubset(mla_kv_b_names):
        raise HybridCheckpointError(
            "source lacks required target/MTP MLA kv_b matrices"
        )

    units = tuple(
        unit
        for name in sorted(source_tensors)
        if (unit := _output_unit(source_tensors[name], dispositions[name])) is not None
    )
    output_names = [tensor.name for unit in units for tensor in unit.tensors]
    if len(output_names) != len(set(output_names)):
        raise HybridCheckpointError("conversion produced duplicate output tensor names")
    output_shards = _plan_output_shards(units, maximum_shard_bytes)
    expert_artifact = _expert_artifact_identity(
        expert_weight_path.resolve(strict=True),
        expert_manifest_path.resolve(strict=True),
        expected_expert_content_id,
    )
    return ConversionPlan(
        source_root=source_root,
        source_index_sha256=index_sha256,
        source_config_sha256=config_sha256,
        source_tensors=source_tensors,
        source_file_identities=identities,
        dispositions=dispositions,
        output_shards=output_shards,
        omitted_expert_names_sha256=_sha256_bytes(
            ("\n".join(omitted_names) + "\n").encode()
        ),
        expert_artifact=expert_artifact,
    )


def _safetensors_header(
    tensors: Sequence[OutputTensor],
) -> tuple[bytes, Mapping[str, tuple[int, int]]]:
    offsets: dict[str, tuple[int, int]] = {}
    rows: JsonObject = {
        "__metadata__": {"format": "pt"},
    }
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: item.name):
        start = cursor
        cursor += tensor.size_bytes
        offsets[tensor.name] = (start, cursor)
        rows[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, cursor],
        }
    raw_header = _canonical_json_bytes(rows)
    padding = (-len(raw_header)) % SAFETENSORS_ALIGNMENT
    padded_header = raw_header + b" " * padding
    return struct.pack("<Q", len(padded_header)) + padded_header, offsets


def _copy_extent(
    source: SourceTensor,
    destination_descriptor: int,
    destination_offset: int,
) -> None:
    source_descriptor = os.open(source.absolute_path, os.O_RDONLY)
    try:
        remaining = source.size_bytes
        source_offset = source.absolute_data_start
        output_offset = destination_offset
        while remaining:
            chunk_size = min(remaining, COPY_CHUNK_BYTES)
            chunk = os.pread(source_descriptor, chunk_size, source_offset)
            if len(chunk) != chunk_size:
                raise HybridCheckpointError(
                    f"source tensor became truncated while copying {source.name}"
                )
            written = os.pwrite(destination_descriptor, chunk, output_offset)
            if written != chunk_size:
                raise HybridCheckpointError(
                    f"short output write while copying {source.name}"
                )
            remaining -= chunk_size
            source_offset += chunk_size
            output_offset += chunk_size
    finally:
        os.close(source_descriptor)


def _bf16_bits_to_float32(
    values: NDArray[np.uint16],
) -> NDArray[np.float32]:
    widened = values.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


def _float32_to_bf16_bits(
    values: NDArray[np.float32],
) -> NDArray[np.uint16]:
    """Round finite float32 values to BF16 with round-to-nearest-even."""

    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _quantize_matrix(
    source: SourceTensor,
    destination_path: Path,
    data_start: int,
    output_offsets: Mapping[str, tuple[int, int]],
    *,
    chunk_bytes: int,
) -> None:
    if source.dtype != "BF16" or len(source.shape) != 2:
        raise AssertionError("quantization source contract was not validated")
    output_features, input_features = source.shape
    stem = source.name.removesuffix(".weight")
    qweight_name = f"{stem}.qweight"
    scales_name = f"{stem}.scales"
    qweight_offset = data_start + output_offsets[qweight_name][0]
    scales_offset = data_start + output_offsets[scales_name][0]

    source_view = np.memmap(
        source.absolute_path,
        mode="r",
        dtype="<u2",
        offset=source.absolute_data_start,
        shape=(output_features, input_features),
        order="C",
    )
    qweight_view = np.memmap(
        destination_path,
        mode="r+",
        dtype="<i4",
        offset=qweight_offset,
        shape=(input_features // 4, output_features),
        order="C",
    )
    scales_view = np.memmap(
        destination_path,
        mode="r+",
        dtype="<u2",
        offset=scales_offset,
        shape=(1, output_features),
        order="C",
    )
    bytes_per_output_row = input_features * (
        np.dtype(np.uint16).itemsize
        + np.dtype(np.float32).itemsize
        + np.dtype(np.int16).itemsize
        + np.dtype(np.uint8).itemsize
    )
    rows_per_chunk = max(1, chunk_bytes // bytes_per_output_row)
    try:
        for row_start in range(0, output_features, rows_per_chunk):
            row_end = min(output_features, row_start + rows_per_chunk)
            source_bits = np.asarray(
                source_view[row_start:row_end],
                dtype=np.uint16,
            )
            weights = _bf16_bits_to_float32(source_bits)
            maximum = np.max(weights, axis=1)
            minimum = np.min(weights, axis=1)
            scales = np.maximum(
                np.abs(maximum / np.float32(127.0)),
                np.abs(minimum / np.float32(-128.0)),
            ).astype(np.float32)
            scales = np.where(
                np.isfinite(scales) & (scales > 0),
                scales,
                np.float32(1.0),
            ).astype(np.float32)
            scale_bits = _float32_to_bf16_bits(scales)
            stored_scales = _bf16_bits_to_float32(scale_bits)
            signed = np.rint(weights / stored_scales[:, np.newaxis])
            signed = np.clip(signed, -128, 127).astype(np.int16)
            unsigned = (signed + np.int16(128)).astype(np.uint8)
            packed = (
                unsigned[:, 0::4].astype(np.uint32)
                | (unsigned[:, 1::4].astype(np.uint32) << np.uint32(8))
                | (unsigned[:, 2::4].astype(np.uint32) << np.uint32(16))
                | (unsigned[:, 3::4].astype(np.uint32) << np.uint32(24))
            )
            qweight_view[:, row_start:row_end] = packed.T.view(np.int32)
            scales_view[0, row_start:row_end] = scale_bits
        qweight_view.flush()
        scales_view.flush()
    finally:
        del source_view
        del qweight_view
        del scales_view


def _quantize_packed_w8_matrix(
    weights: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.uint16]]:
    """Quantize logical ``[K, N]`` to canonical per-head GPTQ W8.

    Four K-adjacent signed INT8 lanes are biased by 128 and serialized in
    little-endian order into each INT32 word.  Scales remain in logical
    per-output-channel order. Marlin performs only a compact layout repack
    after loading.
    """

    if weights.ndim != 2:
        raise HybridCheckpointError(
            f"packed W8 input must be rank two, got {weights.shape}"
        )
    input_features, output_features = weights.shape
    if input_features % MARLIN_W8_PACK_FACTOR != 0 or output_features <= 0:
        raise HybridCheckpointError(
            f"packed W8 matrix cannot pack four K lanes: shape={weights.shape}"
        )
    if not np.isfinite(weights).all():
        raise HybridCheckpointError("packed W8 matrix contains NaN or infinity")

    maximum = np.max(weights, axis=0)
    minimum = np.min(weights, axis=0)
    scales = np.maximum(
        np.abs(maximum / np.float32(127.0)),
        np.abs(minimum / np.float32(-128.0)),
    ).astype(np.float32)
    scales = np.where(
        np.isfinite(scales) & (scales > 0),
        scales,
        np.float32(1.0),
    ).astype(np.float32)
    scale_bits = _float32_to_bf16_bits(scales)
    stored_scales = _bf16_bits_to_float32(scale_bits)
    signed = np.rint(weights / stored_scales[np.newaxis, :])
    signed = np.clip(signed, -128, 127).astype(np.int16)
    unsigned = (signed + np.int16(128)).astype(np.uint32)
    packed = (
        unsigned[0::4]
        | (unsigned[1::4] << np.uint32(8))
        | (unsigned[2::4] << np.uint32(16))
        | (unsigned[3::4] << np.uint32(24))
    )
    return packed.view(np.int32), scale_bits.reshape((1, output_features))


def _quantize_mla_kv_b_matrix(
    source: SourceTensor,
    destination_path: Path,
    data_start: int,
    output_offsets: Mapping[str, tuple[int, int]],
) -> None:
    if source.dtype != "BF16" or source.shape != (
        MLA_NUM_HEADS * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM),
        MLA_KV_LORA_RANK,
    ):
        raise AssertionError("MLA kv_b source contract was not validated")
    stem = source.name.removesuffix(".weight")
    output_specs = {
        "kc_qweight": (
            "<i4",
            (
                MLA_NUM_HEADS,
                MLA_QK_NOPE_HEAD_DIM // MARLIN_W8_PACK_FACTOR,
                MLA_KV_LORA_RANK,
            ),
        ),
        "kc_scales": (
            "<u2",
            (MLA_NUM_HEADS, 1, MLA_KV_LORA_RANK),
        ),
        "vc_qweight": (
            "<i4",
            (
                MLA_NUM_HEADS,
                MLA_KV_LORA_RANK // MARLIN_W8_PACK_FACTOR,
                MLA_V_HEAD_DIM,
            ),
        ),
        "vc_scales": (
            "<u2",
            (MLA_NUM_HEADS, 1, MLA_V_HEAD_DIM),
        ),
    }
    output_views: dict[str, np.memmap] = {}
    for suffix, (dtype, shape) in output_specs.items():
        name = f"{stem}.{suffix}"
        output_views[suffix] = np.memmap(
            destination_path,
            mode="r+",
            dtype=dtype,
            offset=data_start + output_offsets[name][0],
            shape=shape,
            order="C",
        )

    source_view = np.memmap(
        source.absolute_path,
        mode="r",
        dtype="<u2",
        offset=source.absolute_data_start,
        shape=(
            MLA_NUM_HEADS,
            MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM,
            MLA_KV_LORA_RANK,
        ),
        order="C",
    )
    try:
        for head in range(MLA_NUM_HEADS):
            head_weights = _bf16_bits_to_float32(
                np.asarray(source_view[head], dtype=np.uint16)
            )
            kc_weight = np.asarray(
                head_weights[:MLA_QK_NOPE_HEAD_DIM, :],
                dtype=np.float32,
            )
            vc_weight = np.asarray(
                head_weights[MLA_QK_NOPE_HEAD_DIM:, :].T,
                dtype=np.float32,
                order="C",
            )
            kc_qweight, kc_scales = _quantize_packed_w8_matrix(kc_weight)
            vc_qweight, vc_scales = _quantize_packed_w8_matrix(vc_weight)
            output_views["kc_qweight"][head] = kc_qweight
            output_views["kc_scales"][head] = kc_scales
            output_views["vc_qweight"][head] = vc_qweight
            output_views["vc_scales"][head] = vc_scales
        for view in output_views.values():
            view.flush()
    finally:
        del source_view
        for name in tuple(output_views):
            del output_views[name]


def _write_output_shard(
    shard: OutputShard,
    destination_path: Path,
    *,
    quantization_chunk_bytes: int,
) -> None:
    header, offsets = _safetensors_header(shard.tensors)
    total_size = len(header) + shard.payload_size_bytes
    descriptor = os.open(
        destination_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.ftruncate(descriptor, total_size)
        if os.pwrite(descriptor, header, 0) != len(header):
            raise HybridCheckpointError(
                f"short safetensors header write: {destination_path}"
            )
        for unit in shard.units:
            if unit.tensors[0].kind == "preserved":
                output_offset = len(header) + offsets[unit.tensors[0].name][0]
                _copy_extent(unit.source, descriptor, output_offset)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    for unit in shard.units:
        if unit.tensors[0].kind == "quantized_qweight":
            _quantize_matrix(
                unit.source,
                destination_path,
                len(header),
                offsets,
                chunk_bytes=quantization_chunk_bytes,
            )
        elif unit.tensors[0].kind == "mla_kc_qweight":
            _quantize_mla_kv_b_matrix(
                unit.source,
                destination_path,
                len(header),
                offsets,
            )
    descriptor = os.open(destination_path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quantization_config() -> JsonObject:
    return {
        "bits": 8,
        "checkpoint_format": "gptq",
        "desc_act": False,
        "dynamic": {pattern: {} for pattern in QUANTIZATION_DYNAMIC_EXCLUSIONS},
        "group_size": -1,
        "lm_head": True,
        "quant_method": "gptq",
        "sym": True,
        "exo_mla_kv_b_w8": {
            "bits": 8,
            "block_size_m": MARLIN_MOE_BLOCK_SIZE_M,
            "format": "gptq_packed_rows_per_head_v1",
            "implicit_bias": 128,
            "kv_lora_rank": MLA_KV_LORA_RANK,
            "num_attention_heads": MLA_NUM_HEADS,
            "pack_axis": "K",
            "pack_order": "little_endian_k_lanes_0_1_2_3",
            "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
            "scale_compute_dtype": "float32",
            "scale_dtype": "bfloat16",
            "v_head_dim": MLA_V_HEAD_DIM,
        },
    }


def _write_exclusive(path: Path, contents: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(contents)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise HybridCheckpointError(f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_ancillary_files(plan: ConversionPlan, stage_root: Path) -> list[str]:
    copied: list[str] = []
    for filename in ANCILLARY_FILENAMES:
        source = plan.source_root / filename
        if not source.exists():
            continue
        _file_identity(source)
        destination = stage_root / filename
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        source_tensor = SourceTensor(
            name=filename,
            dtype="U8",
            shape=(_file_identity(source).size_bytes,),
            shard_name=filename,
            absolute_path=source,
            absolute_data_start=0,
            absolute_data_end=_file_identity(source).size_bytes,
        )
        output_descriptor = os.open(destination, os.O_WRONLY)
        try:
            _copy_extent(source_tensor, output_descriptor, 0)
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
        copied.append(filename)
    return copied


def _write_model_metadata(
    plan: ConversionPlan,
    stage_root: Path,
) -> list[str]:
    copied = _copy_ancillary_files(plan, stage_root)
    source_config = _read_bounded_json(
        plan.source_root / "config.json",
        "source model config",
    )
    quantization_config = _quantization_config()
    source_config["quantization_config"] = quantization_config
    _write_exclusive(
        stage_root / "config.json",
        _canonical_json_bytes(source_config, pretty=True),
    )
    _write_exclusive(
        stage_root / QUANT_CONFIG_FILENAME,
        _canonical_json_bytes(quantization_config, pretty=True),
    )

    weight_map: JsonObject = {}
    total_size = 0
    for shard in plan.output_shards:
        for tensor in shard.tensors:
            weight_map[tensor.name] = shard.filename
            total_size += tensor.size_bytes
    index: JsonObject = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    _write_exclusive(
        stage_root / INDEX_FILENAME,
        _canonical_json_bytes(index, pretty=True),
    )
    return sorted(
        {
            *copied,
            "config.json",
            QUANT_CONFIG_FILENAME,
            INDEX_FILENAME,
        }
    )


def _disposition_summary(
    plan: ConversionPlan,
    disposition: TensorDisposition,
) -> JsonObject:
    names = [
        name for name, actual in plan.dispositions.items() if actual == disposition
    ]
    source_bytes = sum(plan.source_tensors[name].size_bytes for name in names)
    return {
        "source_tensor_count": len(names),
        "source_bytes": source_bytes,
    }


def _tensor_manifest(plan: ConversionPlan) -> list[JsonValue]:
    rows: list[JsonValue] = []
    for shard in plan.output_shards:
        for tensor in sorted(shard.tensors, key=lambda item: item.name):
            source = plan.source_tensors[tensor.source_name]
            rows.append(
                {
                    "dtype": tensor.dtype,
                    "kind": tensor.kind,
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "size_bytes": tensor.size_bytes,
                    "source_dtype": source.dtype,
                    "source_name": tensor.source_name,
                    "source_shape": list(source.shape),
                    "output_shard": shard.filename,
                }
            )
    return rows


def _validate_expert_artifact_identity(
    artifact: ExpertArtifactIdentity,
) -> None:
    if _directory_identity(artifact.weight_path) != artifact.directory_identity:
        raise HybridCheckpointError(
            "AMXINT4 expert directory changed during conversion"
        )
    if _file_identity(artifact.manifest_path) != artifact.manifest_identity:
        raise HybridCheckpointError("AMXINT4 expert manifest changed during conversion")
    if _sha256_file(artifact.manifest_path) != artifact.manifest_sha256:
        raise HybridCheckpointError(
            "AMXINT4 expert manifest content changed during conversion"
        )
    changed_files = [
        str(path)
        for path, identity in artifact.file_identities.items()
        if _file_identity(path) != identity
    ]
    if changed_files:
        raise HybridCheckpointError(
            f"AMXINT4 expert files changed during conversion: {changed_files[:3]}"
        )
    actual_names = tuple(sorted(entry.name for entry in artifact.weight_path.iterdir()))
    expected_names = tuple(sorted(path.name for path in artifact.file_identities))
    if actual_names != expected_names:
        raise HybridCheckpointError(
            "AMXINT4 expert directory contents changed during conversion"
        )


def _validate_input_identities(plan: ConversionPlan) -> None:
    changed = [
        str(path)
        for path, identity in plan.source_file_identities.items()
        if _file_identity(path) != identity
    ]
    if changed:
        raise HybridCheckpointError(
            f"source files changed during conversion: {changed[:3]}"
        )
    if _sha256_file(plan.source_root / INDEX_FILENAME) != plan.source_index_sha256:
        raise HybridCheckpointError("source index content changed during conversion")
    if _sha256_file(plan.source_root / "config.json") != plan.source_config_sha256:
        raise HybridCheckpointError("source config content changed during conversion")
    _validate_expert_artifact_identity(plan.expert_artifact)


def _manifest_quantization_contract() -> JsonObject:
    return {
        "activation_dtype": "BF16",
        "algorithm": "symmetric_per_output_channel_absmax_rne",
        "checkpoint_layout": {
            "ordinary_linear": "gptq_packed_rows",
            "mla_kv_b": "gptq_packed_rows_per_head_v1",
        },
        "group_size": -1,
        "kernel": {
            "ordinary_linear": "gptq_marlin_w8a16",
            "mla_kv_b": "grouped_marlin_w8a16",
        },
        "mla_kv_b": {
            "block_size_m": MARLIN_MOE_BLOCK_SIZE_M,
            "force_absorbed_mla": True,
            "implicit_bias": 128,
            "kv_lora_rank": MLA_KV_LORA_RANK,
            "num_attention_heads": MLA_NUM_HEADS,
            "pack_axis": "K",
            "pack_order": "little_endian_k_lanes_0_1_2_3",
            "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
            "scale_compute_dtype": "float32",
            "scale_dtype": "bfloat16",
            "v_head_dim": MLA_V_HEAD_DIM,
        },
        "serialized_scale_dtype": "BF16",
        "serialized_weight_dtype": "INT8_biased_by_128_packed_in_INT32",
        "temporary_bf16_expansion_at_load": False,
        "ordinary_linear_compact_to_compact_marlin_repack_at_load": True,
        "mla_kv_b_compact_to_compact_marlin_repack_at_load": True,
        "dynamic_exclusions": list(QUANTIZATION_DYNAMIC_EXCLUSIONS),
    }


def _plan_receipt(plan: ConversionPlan) -> JsonObject:
    return {
        "expert_checkpoint": {
            "content_id": plan.expert_artifact.content_id,
            "manifest_path": str(plan.expert_artifact.manifest_path),
            "manifest_sha256": plan.expert_artifact.manifest_sha256,
            "method": "AMXINT4",
            "weight_path": str(plan.expert_artifact.weight_path),
        },
        "kind": MANIFEST_KIND,
        "output": {
            "payload_bytes": plan.output_payload_bytes,
            "shard_count": len(plan.output_shards),
            "tensor_count": plan.output_tensor_count,
        },
        "policy": {
            "omit_expert": _disposition_summary(plan, "omit_expert"),
            "preserve_bf16": _disposition_summary(plan, "preserve_bf16"),
            "quantize_int8": _disposition_summary(plan, "quantize_int8"),
            "quantize_mla_kv_b_w8": _disposition_summary(plan, "quantize_mla_kv_b_w8"),
            "omitted_expert_names_sha256": plan.omitted_expert_names_sha256,
        },
        "quantization": _manifest_quantization_contract(),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_checkpoint": {
            "config_sha256": plan.source_config_sha256,
            "index_sha256": plan.source_index_sha256,
            "path": str(plan.source_root),
            "tensor_count": len(plan.source_tensors),
        },
    }


def _make_read_only(root: Path) -> None:
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise HybridCheckpointError(
                f"staged checkpoint contains an unexpected entry: {entry}"
            )
        entry.chmod(0o444)
    root.chmod(0o555)


def materialize_checkpoint(
    plan: ConversionPlan,
    destination: Path,
    *,
    quantization_chunk_bytes: int,
) -> Path:
    """Atomically materialize a validated plan and return its manifest."""

    if quantization_chunk_bytes <= 0:
        raise HybridCheckpointError("quantization chunk bytes must be positive")
    if not destination.is_absolute():
        raise HybridCheckpointError("destination must be absolute")
    if destination.exists() or destination.is_symlink():
        raise HybridCheckpointError(f"destination already exists: {destination}")
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    stage_root = destination_parent / (
        f".{destination.name}.staging-{uuid.uuid4().hex}"
    )
    stage_root.mkdir(mode=0o700)
    try:
        metadata_files = _write_model_metadata(plan, stage_root)
        for index, shard in enumerate(plan.output_shards, start=1):
            print(
                f"[{index}/{len(plan.output_shards)}] writing "
                f"{shard.filename} ({shard.payload_size_bytes / 1024**3:.3f} GiB)",
                flush=True,
            )
            _write_output_shard(
                shard,
                stage_root / shard.filename,
                quantization_chunk_bytes=quantization_chunk_bytes,
            )

        _validate_input_identities(plan)
        content_files = sorted(
            [
                *metadata_files,
                *(shard.filename for shard in plan.output_shards),
            ]
        )
        file_rows: list[JsonValue] = []
        for filename in content_files:
            path = stage_root / filename
            file_rows.append(
                {
                    "path": filename,
                    "sha256": _sha256_file(path),
                    "size_bytes": _file_identity(path).size_bytes,
                }
            )
        # Keep the content ID stable over the executable conversion contract and
        # hashed checkpoint files. The verbose tensor/source provenance table is
        # derived metadata; runtime admission separately binds the SHA-256 of the
        # complete manifest, including that table and the creation timestamp.
        content_contract = _plan_receipt(plan)
        content_contract["files"] = file_rows
        content_id = _sha256_bytes(_canonical_json_bytes(content_contract))
        manifest = dict(content_contract)
        manifest["content_id"] = content_id
        manifest["created_at_utc"] = _utc_now()
        manifest["immutability"] = {
            "directory_mode": "0555",
            "file_mode": "0444",
            "materialized_offline": True,
        }
        manifest["tensors"] = _tensor_manifest(plan)
        _write_exclusive(
            stage_root / MANIFEST_FILENAME,
            _canonical_json_bytes(manifest, pretty=True),
        )

        directory_descriptor = os.open(stage_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _make_read_only(stage_root)
        os.replace(stage_root, destination)
        parent_descriptor = os.open(destination_parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return destination / MANIFEST_FILENAME
    except BaseException:
        if stage_root.exists():
            stage_root.chmod(0o700)
            for entry in stage_root.iterdir():
                if entry.is_file() and not entry.is_symlink():
                    entry.chmod(0o600)
            shutil.rmtree(stage_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--expert-weight-path",
        type=Path,
        default=DEFAULT_EXPERT_WEIGHT_PATH,
    )
    parser.add_argument(
        "--expert-manifest",
        type=Path,
        default=DEFAULT_EXPERT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--expected-source-index-sha256",
        default=EXPECTED_SOURCE_INDEX_SHA256,
    )
    parser.add_argument(
        "--expected-source-config-sha256",
        default=EXPECTED_SOURCE_CONFIG_SHA256,
    )
    parser.add_argument(
        "--expected-expert-content-id",
        default=EXPECTED_EXPERT_CONTENT_ID,
    )
    parser.add_argument(
        "--maximum-shard-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_SHARD_BYTES,
    )
    parser.add_argument(
        "--quantization-chunk-bytes",
        type=int,
        default=DEFAULT_QUANTIZATION_CHUNK_BYTES,
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the conversion receipt without writing output",
    )
    return parser


def _optional_sha256(value: str) -> str | None:
    if value == "none":
        return None
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HybridCheckpointError(
            "expected SHA-256 must be lowercase hexadecimal or 'none'"
        )
    return value


def main() -> int:
    arguments = _parser().parse_args()
    try:
        expected_expert_content_id = cast(
            str,
            arguments.expected_expert_content_id,
        )
        if re.fullmatch(r"[0-9a-f]{64}", expected_expert_content_id) is None:
            raise HybridCheckpointError(
                "expected expert content ID must be lowercase SHA-256"
            )
        plan = build_conversion_plan(
            cast(Path, arguments.source),
            expert_weight_path=cast(Path, arguments.expert_weight_path),
            expert_manifest_path=cast(Path, arguments.expert_manifest),
            expected_source_index_sha256=_optional_sha256(
                cast(str, arguments.expected_source_index_sha256)
            ),
            expected_source_config_sha256=_optional_sha256(
                cast(str, arguments.expected_source_config_sha256)
            ),
            expected_expert_content_id=expected_expert_content_id,
            maximum_shard_bytes=cast(int, arguments.maximum_shard_bytes),
        )
        if cast(bool, arguments.plan_only):
            print(
                _canonical_json_bytes(_plan_receipt(plan), pretty=True).decode(),
                end="",
            )
            return 0
        manifest_path = materialize_checkpoint(
            plan,
            cast(Path, arguments.destination),
            quantization_chunk_bytes=cast(
                int,
                arguments.quantization_chunk_bytes,
            ),
        )
        manifest = _read_bounded_json(manifest_path, "hybrid checkpoint manifest")
        print(
            f"{manifest_path} content_id={manifest['content_id']} "
            f"payload_bytes={plan.output_payload_bytes}"
        )
    except (HybridCheckpointError, OSError, ValueError) as error:
        print(
            f"GLM-5.2 hybrid checkpoint materialization failed: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
