#!/usr/bin/env python3
"""Build and validate provenance-bound OSCAR INT2 calibration for DSV4 Flash.

This is deliberately an offline pipeline.  A capture hook writes the tensors
described by ``dsv4_oscar_int2_capture_v1.schema.json``; this program verifies
and reduces those tensors to covariance statistics, fits the OSCAR rotations
and clipping policy, and emits one self-contained artifact for target and
DSpark draft execution.

DeepSeek-V4's cache row is a *shared* K=V representation.  Consequently this
pipeline never emits separate K and V rotations.  For every layer it derives
one 448-dimensional orthogonal map from OSCAR's attention-weighted value SST
covariance, then admits it only when the held-out joint query/value metric also
improves.  The runtime contract is exactly::

    cache = latent @ R
    query = q_nope @ R
    output_nope = output_rotated @ R.T

The 64-dimensional RoPE tail is excluded from the rotation and remains exact
BF16.  C4 layers additionally receive a separate 128-dimensional scorer-only
rotation.  There is no identity, generic INT2, or Hadamard-only fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast, final

import torch

ARTIFACT_FORMAT: Final = "dsv4-oscar-int2-calibration"
ARTIFACT_VERSION: Final = 2
STATISTICS_FORMAT: Final = "dsv4-oscar-int2-statistics"
STATISTICS_VERSION: Final = 2
CAPTURE_FORMAT: Final = "dsv4-oscar-int2-capture"
CAPTURE_VERSION: Final = 1
CHECKPOINT_FORMAT: Final = "dsv4-checkpoint-fingerprint"
CHECKPOINT_VERSION: Final = 1
ADMISSION_FORMAT: Final = "dsv4-oscar-int2-admission"
ADMISSION_VERSION: Final = 1
PROMPT_FORMAT: Final = "dsv4-oscar-calibration-prompts"
PROMPT_VERSION: Final = 1

ALGORITHM: Final = (
    "oscar-dsv4-compressed-history-shared-v-sst-u-pbr-h64x7-c4-k-qqt-u-h128-pbr"
)
STORAGE_LAYOUT: Final = "oscar-int2-asym-g64-v1"
CLIP_MODE: Final = "per_row_quantile"
CLIP_SEMANTICS: Final = "per_row_per_group_abs_order_stat_then_affine_u2_v1"
OSCAR_SOURCE_COMMIT: Final = "797e39c7ccf442c5c6789f87c5a692ee6e98b263"
OSCAR_ROTATION_SOURCE_SHA256: Final = (
    "5f89868fe3fd80cecb7202b802744a83eb0e2d273c6a9960df458b8436b10b2f"
)
OSCAR_CLIP_SOURCE_SHA256: Final = (
    "c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157"
)

NUM_LAYERS: Final = 43
NUM_ATTENTION_HEADS: Final = 64
NUM_KV_HEADS: Final = 1
HEAD_DIM: Final = 512
LATENT_DIM: Final = 448
ROPE_DIM: Final = 64
INDEX_HEAD_DIM: Final = 128
INDEX_HEADS: Final = 64
GROUP_SIZE: Final = 64
NUM_GROUPS: Final = LATENT_DIM // GROUP_SIZE
INDEX_GROUP_SIZE: Final = 128
INDEX_NUM_GROUPS: Final = INDEX_HEAD_DIM // INDEX_GROUP_SIZE
EXPECTED_COMPRESSION_RATIOS: Final = (
    0,
    0,
    *(value for _ in range(20) for value in (4, 128)),
    4,
)
EXPECTED_COMPRESSED_LAYER_IDS: Final = frozenset(
    layer_id for layer_id, ratio in enumerate(EXPECTED_COMPRESSION_RATIOS) if ratio != 0
)
EXPECTED_C4_LAYER_IDS: Final = frozenset(
    layer_id for layer_id, ratio in enumerate(EXPECTED_COMPRESSION_RATIOS) if ratio == 4
)
SHARED_ROTATION_COMPOSITION: Final = "u-pbr-h64x7"
C4_ROTATION_COMPOSITION: Final = "u-h128-pbr"
SHARED_ROTATION_OBJECTIVE: Final = "attention_weighted_value_sst"
C4_ROTATION_OBJECTIVE: Final = "query_qqt"
SHARED_LATENT_SOURCE: Final = "compressed_history_only"
CONSUMER_SCOPE: Final = "target_compressed_history_layers"
DEFAULT_CLIP_GRID: Final = (0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0)


JsonObject = dict[str, Any]
Split = Literal["train", "heldout"]


@final
@dataclass(frozen=True)
class ModelGeometry:
    model_id: str
    model_type: str
    config_sha256: str
    checkpoint_sha256: str
    compression_ratios: tuple[int, ...]
    num_hidden_layers: int = NUM_LAYERS
    num_attention_heads: int = NUM_ATTENTION_HEADS
    num_key_value_heads: int = NUM_KV_HEADS
    head_dim: int = HEAD_DIM
    latent_dim: int = LATENT_DIM
    rope_dim: int = ROPE_DIM
    index_head_dim: int = INDEX_HEAD_DIM
    index_n_heads: int = INDEX_HEADS


@final
@dataclass(frozen=True)
class CalibrationGates:
    maximum_orthogonality_error: float = 2.0e-5
    maximum_heldout_relative_error: float = 0.55
    minimum_improvement_vs_unrotated: float = 0.0
    minimum_train_rows: int = 32
    minimum_heldout_rows: int = 16


@final
@dataclass(frozen=True)
class ArtifactExpectations:
    checkpoint_sha256: str
    config_sha256: str
    prompt_manifest_sha256: str
    capture_manifest_sha256: str | None = None
    statistics_sha256: str | None = None


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _load_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(JsonObject, loaded)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value.lower()


def _resolved_path(base: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(raw)
    return (path if path.is_absolute() else base / path).resolve()


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(_canonical_json_bytes(list(cpu.shape)))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _update_tree_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        digest.update(_tensor_content_digest(value).encode())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(_canonical_json_bytes(str(key)))
            _update_tree_digest(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_tree_digest(digest, item)
        digest.update(b"]")
        return
    digest.update(b"scalar:")
    digest.update(_canonical_json_bytes(value))


def tree_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_tree_digest(digest, value)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _checkpoint_files(checkpoint_dir: Path) -> tuple[Path, ...]:
    config_path = checkpoint_dir / "config.json"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "checkpoint must contain config.json and model.safetensors.index.json"
        )
    index = _load_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint weight index must contain a non-empty weight_map")
    raw_shard_names = tuple(weight_map.values())
    if not all(isinstance(name, str) and name for name in raw_shard_names):
        raise ValueError("checkpoint weight_map contains an invalid shard name")
    shard_names = sorted(set(cast(tuple[str, ...], raw_shard_names)))
    paths = (config_path, index_path, *(checkpoint_dir / name for name in shard_names))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing {missing[0]}")
    return tuple(paths)


def build_checkpoint_fingerprint(checkpoint_dir: Path) -> JsonObject:
    checkpoint_dir = checkpoint_dir.resolve()
    files: list[JsonObject] = []
    for path in _checkpoint_files(checkpoint_dir):
        files.append(
            {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    aggregate = hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
    config = _load_json_object(checkpoint_dir / "config.json")
    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "model_type": config.get("model_type"),
        "files": files,
        "checkpoint_sha256": aggregate,
    }


def verify_checkpoint_fingerprint(
    fingerprint_path: Path, checkpoint_dir: Path
) -> JsonObject:
    expected = _load_json_object(fingerprint_path)
    if (
        expected.get("format") != CHECKPOINT_FORMAT
        or expected.get("format_version") != CHECKPOINT_VERSION
    ):
        raise ValueError("incompatible checkpoint fingerprint format")
    actual = build_checkpoint_fingerprint(checkpoint_dir)
    if actual != expected:
        raise ValueError("checkpoint fingerprint does not match checkpoint contents")
    return actual


def validate_model_config(
    config_path: Path,
    *,
    model_id: str,
    checkpoint_sha256: str,
) -> ModelGeometry:
    config = _load_json_object(config_path)
    required: dict[str, object] = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "qk_rope_head_dim": ROPE_DIM,
        "index_head_dim": INDEX_HEAD_DIM,
        "index_n_heads": INDEX_HEADS,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"DeepSeek-V4 Flash config {key} must be {expected!r}, "
                f"got {config.get(key)!r}"
            )
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < NUM_LAYERS:
        raise ValueError("config compress_ratios must cover all 43 target layers")
    target_ratios = tuple(int(value) for value in ratios[:NUM_LAYERS])
    if target_ratios != EXPECTED_COMPRESSION_RATIOS:
        raise ValueError("config compression topology is not DeepSeek-V4 Flash")
    config_sha256 = sha256_file(config_path)
    return ModelGeometry(
        model_id=model_id,
        model_type="deepseek_v4",
        config_sha256=config_sha256,
        checkpoint_sha256=_require_sha256(checkpoint_sha256, label="checkpoint_sha256"),
        compression_ratios=target_ratios,
    )


def _geometry_dict(geometry: ModelGeometry) -> JsonObject:
    return {
        "model_id": geometry.model_id,
        "model_type": geometry.model_type,
        "checkpoint_sha256": geometry.checkpoint_sha256,
        "config_sha256": geometry.config_sha256,
        "num_hidden_layers": geometry.num_hidden_layers,
        "num_attention_heads": geometry.num_attention_heads,
        "num_key_value_heads": geometry.num_key_value_heads,
        "head_dim": geometry.head_dim,
        "latent_dim": geometry.latent_dim,
        "rope_dim": geometry.rope_dim,
        "index_head_dim": geometry.index_head_dim,
        "index_n_heads": geometry.index_n_heads,
        "compression_ratios": torch.tensor(
            geometry.compression_ratios, dtype=torch.int16
        ),
    }


def _geometry_from_dict(value: Mapping[str, object]) -> ModelGeometry:
    expected_keys = {
        "model_id",
        "model_type",
        "checkpoint_sha256",
        "config_sha256",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "latent_dim",
        "rope_dim",
        "index_head_dim",
        "index_n_heads",
        "compression_ratios",
    }
    if set(value) != expected_keys:
        raise ValueError("model geometry keys do not match the frozen artifact ABI")
    ratios = _require_tensor(
        value.get("compression_ratios"),
        shape=(NUM_LAYERS,),
        dtype=torch.int16,
        label="model.compression_ratios",
    )
    ratio_tuple = tuple(int(item) for item in ratios.to(torch.int64).tolist())
    model_id = value.get("model_id")
    model_type = value.get("model_type")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model.model_id must be a non-empty string")
    if not isinstance(model_type, str):
        raise TypeError("model.model_type must be a string")
    geometry = ModelGeometry(
        model_id=model_id,
        model_type=model_type,
        checkpoint_sha256=_require_sha256(
            value.get("checkpoint_sha256"), label="model.checkpoint_sha256"
        ),
        config_sha256=_require_sha256(
            value.get("config_sha256"), label="model.config_sha256"
        ),
        compression_ratios=ratio_tuple,
        num_hidden_layers=int(value.get("num_hidden_layers", -1)),
        num_attention_heads=int(value.get("num_attention_heads", -1)),
        num_key_value_heads=int(value.get("num_key_value_heads", -1)),
        head_dim=int(value.get("head_dim", -1)),
        latent_dim=int(value.get("latent_dim", -1)),
        rope_dim=int(value.get("rope_dim", -1)),
        index_head_dim=int(value.get("index_head_dim", -1)),
        index_n_heads=int(value.get("index_n_heads", -1)),
    )
    if geometry != ModelGeometry(
        model_id=geometry.model_id,
        model_type="deepseek_v4",
        config_sha256=geometry.config_sha256,
        checkpoint_sha256=geometry.checkpoint_sha256,
        compression_ratios=EXPECTED_COMPRESSION_RATIOS,
    ):
        raise ValueError("artifact model geometry is incompatible with DSV4 Flash")
    return geometry


def build_hadamard(size: int) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    matrix = torch.ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        ) / math.sqrt(2.0)
    return matrix


def build_block_hadamard(size: int, block_size: int = GROUP_SIZE) -> torch.Tensor:
    if size % block_size != 0:
        raise ValueError(f"{size=} must be divisible by {block_size=}")
    block = build_hadamard(block_size)
    return torch.block_diag(*(block for _ in range(size // block_size)))


def _bit_reverse(value: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def balanced_bit_reversal_order(
    size: int, block_size: int = GROUP_SIZE
) -> torch.Tensor:
    """Return a mixed-radix bit-reversal order for 448 = 7 x 64.

    Reversing the six-bit intra-block digit and moving it ahead of the radix-7
    digit spreads consecutive high-energy eigen-directions across all seven
    quantization groups.  It is a true permutation, so it preserves exact
    orthogonality unlike cropping a padded 512-dimensional transform.
    """

    if size % block_size != 0 or block_size & (block_size - 1):
        raise ValueError("balanced bit reversal needs power-of-two uniform blocks")
    outer = size // block_size
    bits = int(math.log2(block_size))
    order = [
        _bit_reverse(index % block_size, bits) * outer + index // block_size
        for index in range(size)
    ]
    if sorted(order) != list(range(size)):
        raise AssertionError("mixed-radix bit reversal is not bijective")
    return torch.tensor(order, dtype=torch.int64)


def make_balancing_permutation(
    eigenvalues: torch.Tensor, block_size: int = GROUP_SIZE
) -> torch.Tensor:
    size = eigenvalues.numel()
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    destinations = balanced_bit_reversal_order(size, block_size)
    permutation = torch.empty(size, dtype=torch.int64)
    permutation[destinations] = sorted_indices
    return torch.eye(size, dtype=torch.float64)[:, permutation]


def compose_oscar_rotation(
    hessian: torch.Tensor, *, hadamard_block_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the dimension-specific OSCAR rotation.

    Official OSCAR uses ``U @ H @ Pbr`` for one complete power-of-two
    Hadamard.  The DSV4 shared latent is 448 = 7 x 64, so its transform is
    seven independent H64 blocks.  In that case the spectrum-balancing
    permutation must run *before* the block transform; placing it after the
    blocks merely permutes already-isolated mixtures and concentrates related
    eigendirections inside individual quantization groups.  The C4 scorer is
    a complete H128 and therefore retains the official composition.
    """

    hessian64 = hessian.to(dtype=torch.float64, device="cpu")
    hessian64 = (hessian64 + hessian64.T) * 0.5
    if hessian64.ndim != 2 or hessian64.shape[0] != hessian64.shape[1]:
        raise ValueError("OSCAR Hessian must be square")
    if hessian64.shape[0] % hadamard_block_size != 0:
        raise ValueError("OSCAR Hessian dimension must use complete Hadamard blocks")
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian64)
    hadamard = build_block_hadamard(hessian64.shape[0], hadamard_block_size)
    permutation = make_balancing_permutation(eigenvalues, hadamard_block_size)
    if hessian64.shape[0] == hadamard_block_size:
        rotation = eigenvectors @ hadamard @ permutation
    elif hessian64.shape[0] == LATENT_DIM and hadamard_block_size == GROUP_SIZE:
        rotation = eigenvectors @ permutation @ hadamard
    else:
        raise ValueError(
            "unsupported OSCAR rotation geometry; expected shared 448/G64 "
            "or C4 128/G128"
        )
    return rotation.to(torch.float32).contiguous(), eigenvalues.to(
        torch.float32
    ).contiguous()


def orthogonality_max_abs(rotation: torch.Tensor) -> float:
    identity = torch.eye(rotation.shape[0], dtype=torch.float64)
    gram = rotation.to(torch.float64).T @ rotation.to(torch.float64)
    return float((gram - identity).abs().max())


def _normalise_covariance(covariance: torch.Tensor) -> torch.Tensor:
    covariance = (covariance.to(torch.float64) + covariance.to(torch.float64).T) * 0.5
    trace = float(torch.trace(covariance))
    if not math.isfinite(trace) or trace <= 0.0:
        raise ValueError("calibration covariance must have positive finite trace")
    return covariance / trace


def shared_value_sst_hessian(value_covariance: torch.Tensor) -> torch.Tensor:
    """Return OSCAR's normalized attention-weighted V/SST objective.

    DSV4 uses one latent row for both K and V, so runtime cannot carry the
    separate K/QQT and V/SST rotations used by ordinary GQA models.  The
    shared map is therefore fit from the official V-side SST covariance while
    admission still measures the reconstructed row against both held-out QQT
    and SST covariances.  This avoids contaminating the spectral fit with a
    non-OSCAR equal-covariance interpolation.
    """

    return _normalise_covariance(value_covariance)


def _clip_indices(
    clip_ratios: torch.Tensor, group_size: int = GROUP_SIZE
) -> torch.Tensor:
    indices = torch.floor(clip_ratios.to(torch.float64) * group_size).to(torch.int16)
    return indices.clamp_(min=0, max=group_size - 1)


def fake_quantize_oscar_int2(
    rows: torch.Tensor,
    clip_ratios: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for OSCAR per-row quantile clip + asymmetric uint2."""

    rows32 = rows.to(dtype=torch.float32, device="cpu")
    if rows32.ndim != 2 or rows32.shape[1] % group_size != 0:
        raise ValueError("INT2 input must be [rows, complete G64 groups]")
    groups = rows32.view(rows32.shape[0], -1, group_size)
    num_groups = groups.shape[1]
    if clip_ratios.shape != (num_groups,):
        raise ValueError(f"clip_ratios must have shape [{num_groups}]")
    indices = _clip_indices(clip_ratios, group_size).to(torch.int64)
    sorted_absolute = groups.abs().sort(dim=-1).values
    gather_index = indices.view(1, num_groups, 1).expand(groups.shape[0], -1, -1)
    thresholds = sorted_absolute.gather(-1, gather_index).squeeze(-1)
    clipped = torch.minimum(
        torch.maximum(groups, -thresholds.unsqueeze(-1)), thresholds.unsqueeze(-1)
    )
    minimum = clipped.amin(dim=-1, keepdim=True)
    maximum = clipped.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum).clamp_min(1.0e-8) / 3.0
    zero = -minimum / scale
    quantized = (clipped / scale + zero + 0.5).floor().clamp_(0.0, 3.0)
    dequantized = (quantized - zero) * scale
    return dequantized.reshape_as(rows32), thresholds


def _quadratic_relative_error(
    original: torch.Tensor, reconstructed: torch.Tensor, hessian: torch.Tensor
) -> float:
    original64 = original.to(torch.float64)
    error64 = reconstructed.to(torch.float64) - original64
    hessian64 = hessian.to(torch.float64)
    numerator = torch.einsum("bi,ij,bj->", error64, hessian64, error64)
    denominator = torch.einsum("bi,ij,bj->", original64, hessian64, original64)
    denominator = denominator.clamp_min(torch.finfo(torch.float64).eps)
    return float(numerator / denominator)


def select_group_clip_ratios(
    rows: torch.Tensor,
    rotation: torch.Tensor,
    hessian: torch.Tensor,
    *,
    candidates: Sequence[float] = DEFAULT_CLIP_GRID,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, JsonObject]:
    if not candidates or any(not 0.0 < item <= 1.0 for item in candidates):
        raise ValueError("clip candidates must be in (0, 1]")
    rotated = rows.to(torch.float32) @ rotation.to(torch.float32)
    rotated_hessian = (
        rotation.T.to(torch.float64)
        @ hessian.to(torch.float64)
        @ rotation.to(torch.float64)
    )
    if rows.shape[1] % group_size != 0:
        raise ValueError("clip calibration rows must use complete quant groups")
    num_groups = rows.shape[1] // group_size
    selected: list[float] = []
    sweep: list[list[float]] = []
    for group in range(num_groups):
        start = group * group_size
        stop = start + group_size
        group_rows = rotated[:, start:stop]
        group_hessian = rotated_hessian[start:stop, start:stop]
        errors: list[float] = []
        for ratio in candidates:
            reconstructed, _ = fake_quantize_oscar_int2(
                group_rows,
                torch.tensor([ratio], dtype=torch.float32),
                group_size=group_size,
            )
            errors.append(
                _quadratic_relative_error(group_rows, reconstructed, group_hessian)
            )
        best = min(range(len(errors)), key=errors.__getitem__)
        selected.append(float(candidates[best]))
        sweep.append(errors)
    ratios = torch.tensor(selected, dtype=torch.float32)
    _, per_row_thresholds = fake_quantize_oscar_int2(
        rotated, ratios, group_size=group_size
    )
    diagnostic_thresholds = per_row_thresholds.median(dim=0).values.to(torch.float32)
    return (
        ratios,
        diagnostic_thresholds,
        {
            "candidate_ratios": [float(item) for item in candidates],
            "train_group_relative_error": sweep,
        },
    )


def _covariance(rows: torch.Tensor) -> torch.Tensor:
    rows64 = rows.reshape(-1, rows.shape[-1]).to(torch.float64)
    if rows64.shape[0] == 0 or not bool(torch.isfinite(rows64).all()):
        raise ValueError("covariance input must contain finite rows")
    return (rows64.T @ rows64 / rows64.shape[0]).contiguous()


def _attention_weighted_value_covariance(
    rows: torch.Tensor, query_covariance: torch.Tensor
) -> torch.Tensor:
    rows64 = rows.to(torch.float64)
    query64 = query_covariance.to(torch.float64)
    weights = (rows64 @ query64 * rows64).sum(dim=1).clamp_min(0.0)
    total = weights.sum()
    if not math.isfinite(float(total)) or float(total) <= 0.0:
        raise ValueError("attention importance weights are degenerate")
    weights = weights * (rows64.shape[0] / total)
    weighted = rows64 * weights.sqrt().unsqueeze(1)
    return (weighted.T @ weighted / rows64.shape[0]).contiguous()


def _load_tensor_ref(
    base: Path,
    reference: object,
    *,
    expected_tail: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    if not isinstance(reference, dict):
        raise TypeError(f"{label} must be a tensor file reference")
    path = _resolved_path(base, reference.get("path"), label=f"{label}.path")
    expected_sha = _require_sha256(reference.get("sha256"), label=f"{label}.sha256")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} file is absent or has the wrong hash")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"{label} file must contain one tensor")
    tensor = loaded.detach().to(device="cpu")
    if not tensor.is_floating_point() or tensor.ndim != len(expected_tail) + 1:
        raise ValueError(f"{label} has an incompatible dtype or rank")
    if tuple(tensor.shape[1:]) != expected_tail or tensor.shape[0] == 0:
        raise ValueError(
            f"{label} must have shape [rows,{','.join(map(str, expected_tail))}]"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains non-finite values")
    declared_shape = reference.get("shape")
    if declared_shape != list(tensor.shape):
        raise ValueError(f"{label} declared shape does not match tensor")
    return tensor.to(torch.float32).contiguous()


def _prompt_provenance(
    prompt_path: Path, expected_sha256: str
) -> tuple[dict[str, Split], int]:
    if sha256_file(prompt_path) != expected_sha256:
        raise ValueError("calibration prompt file hash mismatch")
    prompt_document = _load_json_object(prompt_path)
    if (
        prompt_document.get("format") != PROMPT_FORMAT
        or prompt_document.get("format_version") != PROMPT_VERSION
    ):
        raise ValueError("incompatible calibration prompt format")
    if set(prompt_document) != {"format", "format_version", "prompts"}:
        raise ValueError("calibration prompt document has unsupported fields")
    prompts = prompt_document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("calibration prompt file contains no prompts")
    prompt_splits: dict[str, Split] = {}
    total_tokens = 0
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise TypeError("each prompt entry must be an object")
        if set(prompt) != {"id", "split", "text", "token_count"}:
            raise ValueError("calibration prompt entry has unsupported fields")
        prompt_id = prompt.get("id")
        split = prompt.get("split")
        text = prompt.get("text")
        token_count = prompt.get("token_count")
        if (
            not isinstance(prompt_id, str)
            or prompt_id in prompt_splits
            or split not in ("train", "heldout")
            or not isinstance(text, str)
            or not text
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise ValueError("invalid calibration prompt entry")
        prompt_splits[prompt_id] = cast(Split, split)
        total_tokens += token_count
    if set(prompt_splits.values()) != {"train", "heldout"}:
        raise ValueError("prompt set must contain train and heldout prompts")
    return prompt_splits, total_tokens


def _sample_rows(rows: torch.Tensor, maximum_rows: int, seed: str) -> torch.Tensor:
    if rows.shape[0] <= maximum_rows:
        return rows.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed[:16], 16))
    indices = torch.randperm(rows.shape[0], generator=generator)[:maximum_rows]
    return rows[indices].contiguous()


def _layer_chunks(
    chunks: Sequence[object], layer_id: int, split: Split
) -> list[JsonObject]:
    selected = [
        cast(JsonObject, chunk)
        for chunk in chunks
        if isinstance(chunk, dict)
        and chunk.get("layer_id") == layer_id
        and chunk.get("split") == split
    ]
    if not selected:
        raise ValueError(f"capture has no {split} chunk for layer {layer_id}")
    return selected


def collect_statistics(
    capture_manifest_path: Path,
    output_path: Path,
    *,
    maximum_sample_rows: int = 2048,
) -> JsonObject:
    if maximum_sample_rows <= 0:
        raise ValueError("maximum_sample_rows must be positive")
    manifest_path = capture_manifest_path.resolve()
    manifest = _load_json_object(manifest_path)
    if (
        manifest.get("format") != CAPTURE_FORMAT
        or manifest.get("format_version") != CAPTURE_VERSION
    ):
        raise ValueError("incompatible OSCAR capture manifest")
    if set(manifest) != {"format", "format_version", "model", "prompts", "chunks"}:
        raise ValueError("capture manifest has unsupported fields")
    base = manifest_path.parent
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise TypeError("capture model metadata must be an object")
    if set(model) != {
        "model_id",
        "checkpoint_path",
        "checkpoint_fingerprint_path",
        "checkpoint_fingerprint_sha256",
    }:
        raise ValueError("capture model metadata has unsupported fields")
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("capture model_id is missing")
    checkpoint_dir = _resolved_path(
        base, model.get("checkpoint_path"), label="model.checkpoint_path"
    )
    fingerprint_path = _resolved_path(
        base,
        model.get("checkpoint_fingerprint_path"),
        label="model.checkpoint_fingerprint_path",
    )
    fingerprint_sha = _require_sha256(
        model.get("checkpoint_fingerprint_sha256"),
        label="model.checkpoint_fingerprint_sha256",
    )
    if sha256_file(fingerprint_path) != fingerprint_sha:
        raise ValueError("checkpoint fingerprint document hash mismatch")
    fingerprint = verify_checkpoint_fingerprint(fingerprint_path, checkpoint_dir)
    checkpoint_sha = _require_sha256(
        fingerprint.get("checkpoint_sha256"), label="checkpoint checkpoint_sha256"
    )
    geometry = validate_model_config(
        checkpoint_dir / "config.json",
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha,
    )
    prompts = manifest.get("prompts")
    if not isinstance(prompts, dict):
        raise TypeError("capture prompts metadata must be an object")
    if set(prompts) != {"path", "sha256"}:
        raise ValueError("capture prompts metadata has unsupported fields")
    prompt_path = _resolved_path(base, prompts.get("path"), label="prompts.path")
    prompt_sha = _require_sha256(prompts.get("sha256"), label="prompts.sha256")
    prompt_splits, calibration_prompt_tokens = _prompt_provenance(
        prompt_path, prompt_sha
    )
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("capture chunks must be a list")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("capture chunk must be an object")
        layer_id = chunk.get("layer_id")
        split = chunk.get("split")
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or not 0 <= layer_id < NUM_LAYERS
            or split not in ("train", "heldout")
        ):
            raise ValueError("capture chunk layer/split is invalid")
        ratio = geometry.compression_ratios[layer_id]
        expected_chunk_keys = {
            "layer_id",
            "split",
            "prompt_ids",
            "attention_query_nope",
            "swa_latent",
        }
        if ratio != 0:
            expected_chunk_keys.add("compressed_latent")
        if ratio == 4:
            expected_chunk_keys.update({"c4_scorer_query", "c4_scorer_key"})
        if set(chunk) != expected_chunk_keys:
            raise ValueError(
                f"capture chunk layer {layer_id} does not match its exact topology"
            )

    referenced_prompt_ids: set[str] = set()
    layers: dict[int, JsonObject] = {}
    total_rows: dict[Split, int] = {"train": 0, "heldout": 0}
    capture_input_hashes: list[str] = []
    for layer_id, compress_ratio in enumerate(geometry.compression_ratios):
        # Ratio-0 layers live entirely in the protected BF16 SWA pool.  They
        # never call an OSCAR writer or attention consumer and therefore must
        # not receive a synthetic calibration entry.
        if compress_ratio == 0:
            continue
        layer_statistics: JsonObject = {
            "layer_id": layer_id,
            "compress_ratio": compress_ratio,
        }
        for split in cast(tuple[Split, Split], ("train", "heldout")):
            query_parts: list[torch.Tensor] = []
            latent_parts: list[torch.Tensor] = []
            scorer_query_parts: list[torch.Tensor] = []
            scorer_key_parts: list[torch.Tensor] = []
            for chunk_index, chunk in enumerate(_layer_chunks(chunks, layer_id, split)):
                prompt_ids = chunk.get("prompt_ids")
                if not isinstance(prompt_ids, list) or not prompt_ids:
                    raise ValueError("capture chunk must bind at least one prompt id")
                for prompt_id in prompt_ids:
                    if (
                        not isinstance(prompt_id, str)
                        or prompt_splits.get(prompt_id) != split
                    ):
                        raise ValueError(
                            "capture chunk prompt split/provenance mismatch"
                        )
                    referenced_prompt_ids.add(prompt_id)
                label = f"layer{layer_id}.{split}.chunk{chunk_index}"
                query = _load_tensor_ref(
                    base,
                    chunk.get("attention_query_nope"),
                    expected_tail=(NUM_ATTENTION_HEADS, LATENT_DIM),
                    label=f"{label}.attention_query_nope",
                )
                query_parts.append(query.reshape(-1, LATENT_DIM))
                for name in (
                    "attention_query_nope",
                    "compressed_latent",
                    "c4_scorer_query",
                    "c4_scorer_key",
                ):
                    reference = chunk.get(name)
                    if isinstance(reference, dict):
                        capture_input_hashes.append(
                            _require_sha256(
                                reference.get("sha256"), label=f"{label}.{name}.sha256"
                            )
                        )
                # Only compressor outputs enter the INT2 history pages.  SWA
                # rows remain BF16 and mixing them into this distribution can
                # make a bad history rotation appear to pass heldout gates.
                compressed = _load_tensor_ref(
                    base,
                    chunk.get("compressed_latent"),
                    expected_tail=(LATENT_DIM,),
                    label=f"{label}.compressed_latent",
                )
                latent_parts.append(compressed)
                if compress_ratio == 4:
                    scorer_query_parts.append(
                        _load_tensor_ref(
                            base,
                            chunk.get("c4_scorer_query"),
                            expected_tail=(INDEX_HEADS, INDEX_HEAD_DIM),
                            label=f"{label}.c4_scorer_query",
                        ).reshape(-1, INDEX_HEAD_DIM)
                    )
                    scorer_key_parts.append(
                        _load_tensor_ref(
                            base,
                            chunk.get("c4_scorer_key"),
                            expected_tail=(INDEX_HEAD_DIM,),
                            label=f"{label}.c4_scorer_key",
                        )
                    )
                elif (
                    chunk.get("c4_scorer_query") is not None
                    or chunk.get("c4_scorer_key") is not None
                ):
                    raise ValueError("only C4 layers may provide scorer tensors")
            query_rows = torch.cat(query_parts, dim=0)
            latent_rows = torch.cat(latent_parts, dim=0)
            query_covariance = _covariance(query_rows)
            value_covariance = _attention_weighted_value_covariance(
                latent_rows, query_covariance
            )
            seed = hashlib.sha256(
                f"{sha256_file(manifest_path)}:{layer_id}:{split}:attention".encode()
            ).hexdigest()
            split_statistics: JsonObject = {
                "query_covariance": query_covariance,
                "value_covariance": value_covariance,
                "latent_samples": _sample_rows(latent_rows, maximum_sample_rows, seed),
                "query_row_count": query_rows.shape[0],
                "latent_row_count": latent_rows.shape[0],
            }
            total_rows[split] += latent_rows.shape[0]
            if compress_ratio == 4:
                scorer_query_rows = torch.cat(scorer_query_parts, dim=0)
                scorer_key_rows = torch.cat(scorer_key_parts, dim=0)
                scorer_seed = hashlib.sha256(f"{seed}:c4".encode()).hexdigest()
                split_statistics["c4_scorer"] = {
                    "query_covariance": _covariance(scorer_query_rows),
                    "key_samples": _sample_rows(
                        scorer_key_rows, maximum_sample_rows, scorer_seed
                    ),
                    "query_row_count": scorer_query_rows.shape[0],
                    "key_row_count": scorer_key_rows.shape[0],
                }
            layer_statistics[split] = split_statistics
        layers[layer_id] = layer_statistics
    if referenced_prompt_ids != set(prompt_splits):
        raise ValueError("capture does not account for every hashed calibration prompt")
    capture_sha = sha256_file(manifest_path)
    statistics: JsonObject = {
        "format": STATISTICS_FORMAT,
        "format_version": STATISTICS_VERSION,
        "algorithm": ALGORITHM,
        "model": _geometry_dict(geometry),
        "provenance": {
            "prompt_manifest_sha256": prompt_sha,
            "capture_manifest_sha256": capture_sha,
            "checkpoint_fingerprint_sha256": fingerprint_sha,
            "capture_input_set_sha256": hashlib.sha256(
                _canonical_json_bytes(sorted(capture_input_hashes))
            ).hexdigest(),
            "calibration_prompt_tokens": calibration_prompt_tokens,
            "train_latent_rows": total_rows["train"],
            "heldout_latent_rows": total_rows["heldout"],
            "shared_latent_source": SHARED_LATENT_SOURCE,
            "shared_rotation_objective": SHARED_ROTATION_OBJECTIVE,
            "c4_rotation_objective": C4_ROTATION_OBJECTIVE,
            "oscar_source_commit": OSCAR_SOURCE_COMMIT,
            "oscar_rotation_source_sha256": OSCAR_ROTATION_SOURCE_SHA256,
            "oscar_clip_source_sha256": OSCAR_CLIP_SOURCE_SHA256,
        },
        "layers": layers,
    }
    statistics["statistics_sha256"] = tree_sha256(statistics)
    _atomic_torch_save(statistics, output_path.resolve())
    return statistics


def _require_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"{label} must be {dtype} with shape {shape}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")
    return value.to(device="cpu").contiguous()


def _require_positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_sample_matrix(value: object, *, width: int, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] != width
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite FP32 with shape [rows,{width}]")
    return value.to(device="cpu").contiguous()


def _heldout_metrics(
    rows: torch.Tensor,
    rotation: torch.Tensor,
    clip_ratios: torch.Tensor,
    unrotated_clip_ratios: torch.Tensor,
    query_hessian: torch.Tensor,
    value_hessian: torch.Tensor | None,
    *,
    group_size: int,
) -> JsonObject:
    rotated = rows.to(torch.float32) @ rotation.to(torch.float32)
    quantized_rotated, _ = fake_quantize_oscar_int2(
        rotated, clip_ratios, group_size=group_size
    )
    reconstructed = quantized_rotated @ rotation.T.to(torch.float32)
    unrotated_quantized, _ = fake_quantize_oscar_int2(
        rows, unrotated_clip_ratios, group_size=group_size
    )
    query_error = _quadratic_relative_error(rows, reconstructed, query_hessian)
    unrotated_query_error = _quadratic_relative_error(
        rows, unrotated_quantized, query_hessian
    )
    if value_hessian is None:
        value_error = query_error
        unrotated_value_error = unrotated_query_error
    else:
        value_error = _quadratic_relative_error(rows, reconstructed, value_hessian)
        unrotated_value_error = _quadratic_relative_error(
            rows, unrotated_quantized, value_hessian
        )
    joint_error = 0.5 * (query_error + value_error)
    unrotated_joint_error = 0.5 * (unrotated_query_error + unrotated_value_error)
    return {
        "query_relative_error": query_error,
        "value_relative_error": value_error,
        "joint_relative_error": joint_error,
        "unrotated_query_relative_error": unrotated_query_error,
        "unrotated_value_relative_error": unrotated_value_error,
        "unrotated_joint_relative_error": unrotated_joint_error,
        "improvement_vs_unrotated": (
            (unrotated_joint_error - joint_error)
            / max(unrotated_joint_error, torch.finfo(torch.float64).eps)
        ),
        "row_count": rows.shape[0],
    }


def _fit_domain(
    *,
    domain: str,
    train_rows: torch.Tensor,
    heldout_rows: torch.Tensor,
    train_query_covariance: torch.Tensor,
    heldout_query_covariance: torch.Tensor,
    train_value_covariance: torch.Tensor | None,
    heldout_value_covariance: torch.Tensor | None,
    statistics_sha256: str,
    layer_id: int,
    gates: CalibrationGates,
    clip_candidates: Sequence[float],
    group_size: int,
) -> JsonObject:
    if domain == "attention_shared_latent":
        if train_value_covariance is None or heldout_value_covariance is None:
            raise ValueError("shared-latent OSCAR requires V/SST covariances")
    elif domain == "c4_scorer":
        if train_value_covariance is not None or heldout_value_covariance is not None:
            raise ValueError("C4 OSCAR must use the K/QQT objective")
    else:
        raise ValueError(f"unsupported OSCAR calibration domain: {domain}")
    if train_rows.shape[0] < gates.minimum_train_rows:
        raise ValueError(f"{domain} layer {layer_id} has too few train rows")
    if heldout_rows.shape[0] < gates.minimum_heldout_rows:
        raise ValueError(f"{domain} layer {layer_id} has too few heldout rows")
    if train_value_covariance is None:
        train_hessian = _normalise_covariance(train_query_covariance)
        rotation_objective = C4_ROTATION_OBJECTIVE
    else:
        train_hessian = shared_value_sst_hessian(train_value_covariance)
        rotation_objective = SHARED_ROTATION_OBJECTIVE
    rotation, eigenvalues = compose_oscar_rotation(
        train_hessian, hadamard_block_size=group_size
    )
    rotation_composition = (
        C4_ROTATION_COMPOSITION
        if train_rows.shape[1] == group_size
        else SHARED_ROTATION_COMPOSITION
    )
    orthogonality = orthogonality_max_abs(rotation)
    if orthogonality > gates.maximum_orthogonality_error:
        raise ValueError(
            f"{domain} layer {layer_id} rotation is not sufficiently orthogonal"
        )
    clip_ratios, diagnostic_thresholds, clip_sweep = select_group_clip_ratios(
        train_rows,
        rotation,
        train_hessian,
        candidates=clip_candidates,
        group_size=group_size,
    )
    identity = torch.eye(train_rows.shape[1], dtype=torch.float32)
    unrotated_clip_ratios, _, unrotated_clip_sweep = select_group_clip_ratios(
        train_rows,
        identity,
        train_hessian,
        candidates=clip_candidates,
        group_size=group_size,
    )
    metrics = _heldout_metrics(
        heldout_rows,
        rotation,
        clip_ratios,
        unrotated_clip_ratios,
        heldout_query_covariance,
        heldout_value_covariance,
        group_size=group_size,
    )
    if metrics["joint_relative_error"] > gates.maximum_heldout_relative_error:
        raise ValueError(f"{domain} layer {layer_id} fails heldout error gate")
    if metrics["improvement_vs_unrotated"] < gates.minimum_improvement_vs_unrotated:
        raise ValueError(f"{domain} layer {layer_id} fails heldout improvement gate")
    payload: JsonObject = {
        "domain": domain,
        "rotation_objective": rotation_objective,
        "rotation_composition": rotation_composition,
        "rotation": rotation,
        "eigenvalues": eigenvalues,
        "clip_mode": CLIP_MODE,
        "clip_ratios": clip_ratios,
        "clip_indices": _clip_indices(clip_ratios, group_size),
        "clip_thresholds": diagnostic_thresholds,
        "clip_semantics": CLIP_SEMANTICS,
        "clip_calibration": {
            **clip_sweep,
            "unrotated_baseline": {
                **unrotated_clip_sweep,
                "selected_ratios": unrotated_clip_ratios,
            },
        },
        "orthogonality_max_abs": orthogonality,
        "heldout_metrics": metrics,
        "provenance_sha256": "",
    }
    payload["provenance_sha256"] = tree_sha256(
        {
            "statistics_sha256": statistics_sha256,
            "layer_id": layer_id,
            **{
                key: value
                for key, value in payload.items()
                if key != "provenance_sha256"
            },
        }
    )
    return payload


def _validate_statistics(statistics: Mapping[str, object]) -> tuple[ModelGeometry, str]:
    if (
        statistics.get("format") != STATISTICS_FORMAT
        or statistics.get("format_version") != STATISTICS_VERSION
    ):
        raise ValueError("incompatible OSCAR statistics format")
    if set(statistics) != {
        "format",
        "format_version",
        "algorithm",
        "model",
        "provenance",
        "layers",
        "statistics_sha256",
    }:
        raise ValueError("statistics keys do not match the frozen calibration ABI")
    if statistics.get("algorithm") != ALGORITHM:
        raise ValueError("statistics were not collected for shared-latent OSCAR")
    recorded_sha = _require_sha256(
        statistics.get("statistics_sha256"), label="statistics_sha256"
    )
    calculated_sha = tree_sha256(
        {key: value for key, value in statistics.items() if key != "statistics_sha256"}
    )
    if recorded_sha != calculated_sha:
        raise ValueError("OSCAR statistics content digest mismatch")
    model = statistics.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("statistics model must be a mapping")
    geometry = _geometry_from_dict(model)
    provenance = statistics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("statistics provenance must be a mapping")
    if set(provenance) != {
        "prompt_manifest_sha256",
        "capture_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "capture_input_set_sha256",
        "calibration_prompt_tokens",
        "train_latent_rows",
        "heldout_latent_rows",
        "shared_latent_source",
        "shared_rotation_objective",
        "c4_rotation_objective",
        "oscar_source_commit",
        "oscar_rotation_source_sha256",
        "oscar_clip_source_sha256",
    }:
        raise ValueError("statistics provenance keys do not match the frozen ABI")
    for key in (
        "prompt_manifest_sha256",
        "capture_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "capture_input_set_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"statistics.provenance.{key}")
    for key in (
        "calibration_prompt_tokens",
        "train_latent_rows",
        "heldout_latent_rows",
    ):
        _require_positive_int(provenance.get(key), label=f"statistics.provenance.{key}")
    if provenance.get("oscar_source_commit") != OSCAR_SOURCE_COMMIT:
        raise ValueError("statistics OSCAR source commit mismatch")
    if provenance.get("shared_latent_source") != SHARED_LATENT_SOURCE:
        raise ValueError("statistics shared latent source is not compressed history")
    if provenance.get("shared_rotation_objective") != SHARED_ROTATION_OBJECTIVE:
        raise ValueError("statistics shared rotation objective is not V/SST")
    if provenance.get("c4_rotation_objective") != C4_ROTATION_OBJECTIVE:
        raise ValueError("statistics C4 rotation objective is not K/QQT")
    if provenance.get("oscar_rotation_source_sha256") != OSCAR_ROTATION_SOURCE_SHA256:
        raise ValueError("statistics OSCAR rotation source hash mismatch")
    if provenance.get("oscar_clip_source_sha256") != OSCAR_CLIP_SOURCE_SHA256:
        raise ValueError("statistics OSCAR clip source hash mismatch")
    layers = statistics.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != EXPECTED_COMPRESSED_LAYER_IDS:
        raise ValueError("statistics must contain exactly compressed layers 2..42")
    return geometry, recorded_sha


def calibrate_artifact(
    statistics_path: Path,
    output_path: Path,
    *,
    gates: CalibrationGates | None = None,
    clip_candidates: Sequence[float] = DEFAULT_CLIP_GRID,
) -> JsonObject:
    gates = gates or CalibrationGates()
    loaded = torch.load(statistics_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("statistics artifact must contain a dictionary")
    statistics = cast(JsonObject, loaded)
    geometry, statistics_sha = _validate_statistics(statistics)
    raw_layers = cast(Mapping[int, object], statistics["layers"])
    layers: dict[int, JsonObject] = {}
    train_rows = 0
    heldout_rows = 0
    train_source_rows = 0
    heldout_source_rows = 0
    for layer_id, compress_ratio in enumerate(geometry.compression_ratios):
        if compress_ratio == 0:
            continue
        raw_layer = raw_layers[layer_id]
        if not isinstance(raw_layer, Mapping):
            raise TypeError(f"statistics layer {layer_id} must be a mapping")
        if set(raw_layer) != {
            "layer_id",
            "compress_ratio",
            "train",
            "heldout",
        }:
            raise ValueError(f"statistics layer {layer_id} has unsupported fields")
        if (
            raw_layer.get("layer_id") != layer_id
            or raw_layer.get("compress_ratio") != compress_ratio
        ):
            raise ValueError(f"statistics layer {layer_id} metadata mismatch")
        train = raw_layer.get("train")
        heldout = raw_layer.get("heldout")
        if not isinstance(train, Mapping) or not isinstance(heldout, Mapping):
            raise TypeError(f"statistics layer {layer_id} split missing")
        expected_split_keys = {
            "query_covariance",
            "value_covariance",
            "latent_samples",
            "query_row_count",
            "latent_row_count",
        }
        if compress_ratio == 4:
            expected_split_keys.add("c4_scorer")
        if set(train) != expected_split_keys or set(heldout) != expected_split_keys:
            raise ValueError(f"statistics layer {layer_id} split fields mismatch")
        train_samples = _require_sample_matrix(
            train.get("latent_samples"),
            width=LATENT_DIM,
            label=f"layer{layer_id}.train.latent_samples",
        )
        heldout_samples = _require_sample_matrix(
            heldout.get("latent_samples"),
            width=LATENT_DIM,
            label=f"layer{layer_id}.heldout.latent_samples",
        )
        train_query_row_count = _require_positive_int(
            train.get("query_row_count"),
            label=f"layer{layer_id}.train.query_row_count",
        )
        heldout_query_row_count = _require_positive_int(
            heldout.get("query_row_count"),
            label=f"layer{layer_id}.heldout.query_row_count",
        )
        train_latent_row_count = _require_positive_int(
            train.get("latent_row_count"),
            label=f"layer{layer_id}.train.latent_row_count",
        )
        heldout_latent_row_count = _require_positive_int(
            heldout.get("latent_row_count"),
            label=f"layer{layer_id}.heldout.latent_row_count",
        )
        if (
            train_query_row_count < train_samples.shape[0]
            or heldout_query_row_count < heldout_samples.shape[0]
            or train_latent_row_count < train_samples.shape[0]
            or heldout_latent_row_count < heldout_samples.shape[0]
        ):
            raise ValueError(f"statistics layer {layer_id} row counts are inconsistent")
        train_source_rows += train_latent_row_count
        heldout_source_rows += heldout_latent_row_count
        train_query = _require_tensor(
            train.get("query_covariance"),
            shape=(LATENT_DIM, LATENT_DIM),
            dtype=torch.float64,
            label=f"layer{layer_id}.train.query_covariance",
        )
        heldout_query = _require_tensor(
            heldout.get("query_covariance"),
            shape=(LATENT_DIM, LATENT_DIM),
            dtype=torch.float64,
            label=f"layer{layer_id}.heldout.query_covariance",
        )
        train_value = _require_tensor(
            train.get("value_covariance"),
            shape=(LATENT_DIM, LATENT_DIM),
            dtype=torch.float64,
            label=f"layer{layer_id}.train.value_covariance",
        )
        heldout_value = _require_tensor(
            heldout.get("value_covariance"),
            shape=(LATENT_DIM, LATENT_DIM),
            dtype=torch.float64,
            label=f"layer{layer_id}.heldout.value_covariance",
        )
        attention_payload = _fit_domain(
            domain="attention_shared_latent",
            train_rows=train_samples,
            heldout_rows=heldout_samples,
            train_query_covariance=train_query,
            heldout_query_covariance=heldout_query,
            train_value_covariance=train_value,
            heldout_value_covariance=heldout_value,
            statistics_sha256=statistics_sha,
            layer_id=layer_id,
            gates=gates,
            clip_candidates=clip_candidates,
            group_size=GROUP_SIZE,
        )
        layer_payload: JsonObject = {
            "layer_id": layer_id,
            "compress_ratio": compress_ratio,
            "attention_shared_latent": attention_payload,
        }
        train_rows += train_samples.shape[0]
        heldout_rows += heldout_samples.shape[0]
        if compress_ratio == 4:
            train_scorer = train.get("c4_scorer")
            heldout_scorer = heldout.get("c4_scorer")
            if not isinstance(train_scorer, Mapping) or not isinstance(
                heldout_scorer, Mapping
            ):
                raise ValueError(f"C4 layer {layer_id} scorer statistics missing")
            expected_scorer_keys = {
                "query_covariance",
                "key_samples",
                "query_row_count",
                "key_row_count",
            }
            if (
                set(train_scorer) != expected_scorer_keys
                or set(heldout_scorer) != expected_scorer_keys
            ):
                raise ValueError(f"C4 layer {layer_id} scorer fields mismatch")
            train_keys = _require_sample_matrix(
                train_scorer.get("key_samples"),
                width=INDEX_HEAD_DIM,
                label=f"layer{layer_id}.train.c4.key_samples",
            )
            heldout_keys = _require_sample_matrix(
                heldout_scorer.get("key_samples"),
                width=INDEX_HEAD_DIM,
                label=f"layer{layer_id}.heldout.c4.key_samples",
            )
            train_c4_query_rows = _require_positive_int(
                train_scorer.get("query_row_count"),
                label=f"layer{layer_id}.train.c4.query_row_count",
            )
            heldout_c4_query_rows = _require_positive_int(
                heldout_scorer.get("query_row_count"),
                label=f"layer{layer_id}.heldout.c4.query_row_count",
            )
            train_c4_key_rows = _require_positive_int(
                train_scorer.get("key_row_count"),
                label=f"layer{layer_id}.train.c4.key_row_count",
            )
            heldout_c4_key_rows = _require_positive_int(
                heldout_scorer.get("key_row_count"),
                label=f"layer{layer_id}.heldout.c4.key_row_count",
            )
            if (
                train_c4_query_rows < train_keys.shape[0]
                or heldout_c4_query_rows < heldout_keys.shape[0]
                or train_c4_key_rows < train_keys.shape[0]
                or heldout_c4_key_rows < heldout_keys.shape[0]
            ):
                raise ValueError(f"C4 layer {layer_id} scorer row counts inconsistent")
            train_c4_query = _require_tensor(
                train_scorer.get("query_covariance"),
                shape=(INDEX_HEAD_DIM, INDEX_HEAD_DIM),
                dtype=torch.float64,
                label=f"layer{layer_id}.train.c4.query_covariance",
            )
            heldout_c4_query = _require_tensor(
                heldout_scorer.get("query_covariance"),
                shape=(INDEX_HEAD_DIM, INDEX_HEAD_DIM),
                dtype=torch.float64,
                label=f"layer{layer_id}.heldout.c4.query_covariance",
            )
            layer_payload["c4_scorer"] = _fit_domain(
                domain="c4_scorer",
                train_rows=train_keys,
                heldout_rows=heldout_keys,
                train_query_covariance=train_c4_query,
                heldout_query_covariance=heldout_c4_query,
                train_value_covariance=None,
                heldout_value_covariance=None,
                statistics_sha256=statistics_sha,
                layer_id=layer_id,
                gates=gates,
                clip_candidates=clip_candidates,
                group_size=INDEX_GROUP_SIZE,
            )
        layers[layer_id] = layer_payload
    stats_provenance = statistics.get("provenance")
    if not isinstance(stats_provenance, Mapping):
        raise TypeError("statistics provenance must be a mapping")
    if (
        stats_provenance.get("train_latent_rows") != train_source_rows
        or stats_provenance.get("heldout_latent_rows") != heldout_source_rows
    ):
        raise ValueError("statistics aggregate source row counts mismatch")
    artifact: JsonObject = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "algorithm": ALGORITHM,
        "model": _geometry_dict(geometry),
        "quantization": {
            "bits": 2,
            "group_size": GROUP_SIZE,
            "num_groups": NUM_GROUPS,
            "storage_layout": STORAGE_LAYOUT,
            "codes_bytes_per_token": 112,
            "scale_zero_bytes_per_token": 28,
            "rope_bytes_per_token": 128,
            "logical_bytes_per_token": 268,
            "padded_bytes_per_token": 272,
        },
        "provenance": {
            **dict(stats_provenance),
            "statistics_sha256": statistics_sha,
            "train_sample_rows": train_rows,
            "heldout_sample_rows": heldout_rows,
            "consumer_scope": CONSUMER_SCOPE,
            "shared_rotation_composition": SHARED_ROTATION_COMPOSITION,
            "c4_rotation_composition": C4_ROTATION_COMPOSITION,
            "calibrator_source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "layers": layers,
    }
    artifact["artifact_provenance_sha256"] = tree_sha256(artifact)
    validate_artifact(artifact)
    _atomic_torch_save(artifact, output_path.resolve())
    return artifact


def _validate_domain_payload(
    payload: object,
    *,
    domain: str,
    dimension: int,
    group_size: int,
    num_groups: int,
    layer_id: int,
    statistics_sha256: str,
) -> None:
    if not isinstance(payload, Mapping) or payload.get("domain") != domain:
        raise ValueError(f"layer {layer_id} missing exact {domain} domain")
    expected_keys = {
        "domain",
        "rotation_objective",
        "rotation_composition",
        "rotation",
        "eigenvalues",
        "clip_mode",
        "clip_ratios",
        "clip_indices",
        "clip_thresholds",
        "clip_semantics",
        "clip_calibration",
        "orthogonality_max_abs",
        "heldout_metrics",
        "provenance_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(f"layer {layer_id} {domain} fields do not match the ABI")
    expected_composition = (
        SHARED_ROTATION_COMPOSITION
        if domain == "attention_shared_latent"
        else C4_ROTATION_COMPOSITION
    )
    expected_objective = (
        SHARED_ROTATION_OBJECTIVE
        if domain == "attention_shared_latent"
        else C4_ROTATION_OBJECTIVE
    )
    if payload.get("rotation_objective") != expected_objective:
        raise ValueError(f"layer {layer_id} {domain} rotation objective mismatch")
    if payload.get("rotation_composition") != expected_composition:
        raise ValueError(f"layer {layer_id} {domain} rotation composition mismatch")
    rotation = _require_tensor(
        payload.get("rotation"),
        shape=(dimension, dimension),
        dtype=torch.float32,
        label=f"layer{layer_id}.{domain}.rotation",
    )
    eigenvalues = _require_tensor(
        payload.get("eigenvalues"),
        shape=(dimension,),
        dtype=torch.float32,
        label=f"layer{layer_id}.{domain}.eigenvalues",
    )
    if not bool(torch.all(eigenvalues[1:] >= eigenvalues[:-1])):
        raise ValueError(f"layer {layer_id} {domain} eigenvalues are not ordered")
    ratios = _require_tensor(
        payload.get("clip_ratios"),
        shape=(num_groups,),
        dtype=torch.float32,
        label=f"layer{layer_id}.{domain}.clip_ratios",
    )
    indices = _require_tensor(
        payload.get("clip_indices"),
        shape=(num_groups,),
        dtype=torch.int16,
        label=f"layer{layer_id}.{domain}.clip_indices",
    )
    thresholds = _require_tensor(
        payload.get("clip_thresholds"),
        shape=(num_groups,),
        dtype=torch.float32,
        label=f"layer{layer_id}.{domain}.clip_thresholds",
    )
    if not bool(torch.all(torch.isfinite(thresholds) & (thresholds > 0.0))):
        raise ValueError(
            f"layer {layer_id} {domain} clip thresholds must be finite and positive"
        )
    if (
        payload.get("clip_mode") != CLIP_MODE
        or payload.get("clip_semantics") != CLIP_SEMANTICS
    ):
        raise ValueError(f"layer {layer_id} {domain} clip semantics mismatch")
    if not bool(torch.all((ratios > 0.0) & (ratios <= 1.0))):
        raise ValueError(f"layer {layer_id} {domain} clip ratios invalid")
    if not torch.equal(indices, _clip_indices(ratios, group_size)):
        raise ValueError(f"layer {layer_id} {domain} clip indices invalid")
    clip_calibration = payload.get("clip_calibration")
    if not isinstance(clip_calibration, Mapping) or set(clip_calibration) != {
        "candidate_ratios",
        "train_group_relative_error",
        "unrotated_baseline",
    }:
        raise ValueError(f"layer {layer_id} {domain} clip proof is malformed")
    candidates = clip_calibration.get("candidate_ratios")
    group_errors = clip_calibration.get("train_group_relative_error")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(candidate, float)
            or not math.isfinite(candidate)
            or not 0.0 < candidate <= 1.0
            for candidate in candidates
        )
        or len(set(candidates)) != len(candidates)
        or not isinstance(group_errors, list)
        or len(group_errors) != num_groups
    ):
        raise ValueError(f"layer {layer_id} {domain} clip sweep is invalid")
    for group, errors in enumerate(group_errors):
        if (
            not isinstance(errors, list)
            or len(errors) != len(candidates)
            or any(
                not isinstance(error, float) or not math.isfinite(error) or error < 0.0
                for error in errors
            )
        ):
            raise ValueError(f"layer {layer_id} {domain} clip errors are invalid")
        selected = min(range(len(errors)), key=errors.__getitem__)
        if abs(float(ratios[group]) - candidates[selected]) > 1.0e-6:
            raise ValueError(f"layer {layer_id} {domain} clip optimum mismatch")
    unrotated = clip_calibration.get("unrotated_baseline")
    if not isinstance(unrotated, Mapping) or set(unrotated) != {
        "candidate_ratios",
        "train_group_relative_error",
        "selected_ratios",
    }:
        raise ValueError(f"layer {layer_id} {domain} baseline clip proof malformed")
    if unrotated.get("candidate_ratios") != candidates:
        raise ValueError(f"layer {layer_id} {domain} baseline candidate mismatch")
    baseline_errors = unrotated.get("train_group_relative_error")
    baseline_ratios = _require_tensor(
        unrotated.get("selected_ratios"),
        shape=(num_groups,),
        dtype=torch.float32,
        label=f"layer{layer_id}.{domain}.baseline_clip_ratios",
    )
    if not isinstance(baseline_errors, list) or len(baseline_errors) != num_groups:
        raise ValueError(f"layer {layer_id} {domain} baseline clip errors malformed")
    for group, errors in enumerate(baseline_errors):
        if (
            not isinstance(errors, list)
            or len(errors) != len(candidates)
            or any(
                not isinstance(error, float) or not math.isfinite(error) or error < 0.0
                for error in errors
            )
        ):
            raise ValueError(f"layer {layer_id} {domain} baseline errors invalid")
        selected = min(range(len(errors)), key=errors.__getitem__)
        if abs(float(baseline_ratios[group]) - candidates[selected]) > 1.0e-6:
            raise ValueError(f"layer {layer_id} {domain} baseline optimum mismatch")
    recorded_orthogonality = payload.get("orthogonality_max_abs")
    if not isinstance(recorded_orthogonality, float):
        raise TypeError("orthogonality proof must be a float")
    calculated_orthogonality = orthogonality_max_abs(rotation)
    if (
        calculated_orthogonality > CalibrationGates().maximum_orthogonality_error
        or abs(recorded_orthogonality - calculated_orthogonality) > 1.0e-7
    ):
        raise ValueError(f"layer {layer_id} {domain} orthogonality proof invalid")
    identity_deviation = float(
        (rotation - torch.eye(dimension, dtype=torch.float32)).abs().max()
    )
    if identity_deviation <= 1.0e-3:
        raise ValueError(
            f"layer {layer_id} {domain} uses an identity placeholder rotation"
        )
    metrics = payload.get("heldout_metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError(f"layer {layer_id} {domain} heldout metrics missing")
    required_metrics = {
        "query_relative_error",
        "value_relative_error",
        "joint_relative_error",
        "unrotated_query_relative_error",
        "unrotated_value_relative_error",
        "unrotated_joint_relative_error",
        "improvement_vs_unrotated",
        "row_count",
    }
    if set(metrics) != required_metrics:
        raise ValueError(f"layer {layer_id} {domain} heldout metric fields mismatch")
    for key in required_metrics:
        value = metrics.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"layer {layer_id} {domain} heldout metric {key} invalid")
    query_error = float(metrics["query_relative_error"])
    value_error = float(metrics["value_relative_error"])
    joint_error = float(metrics["joint_relative_error"])
    unrotated_query_error = float(metrics["unrotated_query_relative_error"])
    unrotated_value_error = float(metrics["unrotated_value_relative_error"])
    unrotated_joint_error = float(metrics["unrotated_joint_relative_error"])
    improvement = float(metrics["improvement_vs_unrotated"])
    row_count = metrics["row_count"]
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        raise TypeError(f"layer {layer_id} {domain} row_count must be an integer")
    if (
        min(
            query_error,
            value_error,
            joint_error,
            unrotated_query_error,
            unrotated_value_error,
            unrotated_joint_error,
        )
        < 0.0
    ):
        raise ValueError(f"layer {layer_id} {domain} heldout error is negative")
    epsilon = torch.finfo(torch.float64).eps
    expected_improvement = (unrotated_joint_error - joint_error) / max(
        unrotated_joint_error, epsilon
    )
    if (
        abs(joint_error - 0.5 * (query_error + value_error)) > 1.0e-7
        or abs(
            unrotated_joint_error
            - 0.5 * (unrotated_query_error + unrotated_value_error)
        )
        > 1.0e-7
        or abs(improvement - expected_improvement) > 1.0e-7
    ):
        raise ValueError(f"layer {layer_id} {domain} heldout metric proof mismatch")
    if (
        joint_error > CalibrationGates().maximum_heldout_relative_error
        or improvement < CalibrationGates().minimum_improvement_vs_unrotated
        or row_count < CalibrationGates().minimum_heldout_rows
    ):
        raise ValueError(f"layer {layer_id} {domain} heldout admission gate failed")
    recorded_provenance = _require_sha256(
        payload.get("provenance_sha256"),
        label=f"layer{layer_id}.{domain}.provenance_sha256",
    )
    calculated_provenance = tree_sha256(
        {
            "statistics_sha256": statistics_sha256,
            "layer_id": layer_id,
            **{
                key: value
                for key, value in payload.items()
                if key != "provenance_sha256"
            },
        }
    )
    if recorded_provenance != calculated_provenance:
        raise ValueError(f"layer {layer_id} {domain} provenance proof mismatch")


def validate_artifact(
    artifact: Mapping[str, object],
    expectations: ArtifactExpectations | None = None,
) -> None:
    if (
        artifact.get("format") != ARTIFACT_FORMAT
        or artifact.get("format_version") != ARTIFACT_VERSION
    ):
        raise ValueError("incompatible DSV4 OSCAR INT2 artifact format")
    if set(artifact) != {
        "format",
        "format_version",
        "algorithm",
        "model",
        "quantization",
        "provenance",
        "layers",
        "artifact_provenance_sha256",
    }:
        raise ValueError("artifact keys do not match the frozen runtime ABI")
    if artifact.get("algorithm") != ALGORITHM:
        raise ValueError("artifact is not the calibrated shared-latent OSCAR algorithm")
    recorded_digest = _require_sha256(
        artifact.get("artifact_provenance_sha256"),
        label="artifact_provenance_sha256",
    )
    calculated_digest = tree_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "artifact_provenance_sha256"
        }
    )
    if recorded_digest != calculated_digest:
        raise ValueError("artifact content/provenance digest mismatch")
    model = artifact.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("artifact model must be a mapping")
    geometry = _geometry_from_dict(model)
    quantization = artifact.get("quantization")
    expected_quantization = {
        "bits": 2,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "storage_layout": STORAGE_LAYOUT,
        "codes_bytes_per_token": 112,
        "scale_zero_bytes_per_token": 28,
        "rope_bytes_per_token": 128,
        "logical_bytes_per_token": 268,
        "padded_bytes_per_token": 272,
    }
    if quantization != expected_quantization:
        raise ValueError("artifact quantization/storage contract mismatch")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("artifact provenance must be a mapping")
    if set(provenance) != {
        "prompt_manifest_sha256",
        "capture_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "capture_input_set_sha256",
        "calibration_prompt_tokens",
        "train_latent_rows",
        "heldout_latent_rows",
        "shared_latent_source",
        "shared_rotation_objective",
        "c4_rotation_objective",
        "oscar_source_commit",
        "oscar_rotation_source_sha256",
        "oscar_clip_source_sha256",
        "statistics_sha256",
        "train_sample_rows",
        "heldout_sample_rows",
        "consumer_scope",
        "shared_rotation_composition",
        "c4_rotation_composition",
        "calibrator_source_sha256",
    }:
        raise ValueError("artifact provenance keys do not match the frozen ABI")
    for key in (
        "prompt_manifest_sha256",
        "capture_manifest_sha256",
        "statistics_sha256",
        "checkpoint_fingerprint_sha256",
        "capture_input_set_sha256",
        "calibrator_source_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"provenance.{key}")
    if provenance.get("oscar_source_commit") != OSCAR_SOURCE_COMMIT:
        raise ValueError("artifact OSCAR source commit mismatch")
    if provenance.get("oscar_rotation_source_sha256") != OSCAR_ROTATION_SOURCE_SHA256:
        raise ValueError("artifact OSCAR rotation source hash mismatch")
    if provenance.get("oscar_clip_source_sha256") != OSCAR_CLIP_SOURCE_SHA256:
        raise ValueError("artifact OSCAR clip source hash mismatch")
    if provenance.get("shared_latent_source") != SHARED_LATENT_SOURCE:
        raise ValueError("artifact shared latent source is not compressed history")
    if provenance.get("shared_rotation_objective") != SHARED_ROTATION_OBJECTIVE:
        raise ValueError("artifact shared rotation objective is not V/SST")
    if provenance.get("c4_rotation_objective") != C4_ROTATION_OBJECTIVE:
        raise ValueError("artifact C4 rotation objective is not K/QQT")
    for key in (
        "calibration_prompt_tokens",
        "train_latent_rows",
        "heldout_latent_rows",
        "train_sample_rows",
        "heldout_sample_rows",
    ):
        _require_positive_int(provenance.get(key), label=f"provenance.{key}")
    if provenance.get("consumer_scope") != CONSUMER_SCOPE:
        raise ValueError("artifact consumer scope mismatch")
    if provenance.get("shared_rotation_composition") != SHARED_ROTATION_COMPOSITION:
        raise ValueError("artifact shared rotation composition mismatch")
    if provenance.get("c4_rotation_composition") != C4_ROTATION_COMPOSITION:
        raise ValueError("artifact C4 rotation composition mismatch")
    if provenance.get("calibrator_source_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("artifact calibrator source hash mismatch")
    statistics_sha256 = cast(str, provenance["statistics_sha256"])
    layers = artifact.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != EXPECTED_COMPRESSED_LAYER_IDS:
        raise ValueError("artifact must contain exactly compressed layers 2..42")
    for layer_id, compress_ratio in enumerate(geometry.compression_ratios):
        if compress_ratio == 0:
            continue
        layer = layers[layer_id]
        if not isinstance(layer, Mapping):
            raise TypeError(f"artifact layer {layer_id} must be a mapping")
        expected_keys = {"layer_id", "compress_ratio", "attention_shared_latent"}
        if compress_ratio == 4:
            expected_keys.add("c4_scorer")
        if set(layer) != expected_keys:
            raise ValueError(f"artifact layer {layer_id} domain coverage mismatch")
        if (
            layer.get("layer_id") != layer_id
            or layer.get("compress_ratio") != compress_ratio
        ):
            raise ValueError(f"artifact layer {layer_id} metadata mismatch")
        _validate_domain_payload(
            layer.get("attention_shared_latent"),
            domain="attention_shared_latent",
            dimension=LATENT_DIM,
            group_size=GROUP_SIZE,
            num_groups=NUM_GROUPS,
            layer_id=layer_id,
            statistics_sha256=statistics_sha256,
        )
        if compress_ratio == 4:
            _validate_domain_payload(
                layer.get("c4_scorer"),
                domain="c4_scorer",
                dimension=INDEX_HEAD_DIM,
                group_size=INDEX_GROUP_SIZE,
                num_groups=INDEX_NUM_GROUPS,
                layer_id=layer_id,
                statistics_sha256=statistics_sha256,
            )
    if expectations is not None:
        comparisons = {
            "checkpoint_sha256": (
                geometry.checkpoint_sha256,
                expectations.checkpoint_sha256,
            ),
            "config_sha256": (geometry.config_sha256, expectations.config_sha256),
            "prompt_manifest_sha256": (
                provenance.get("prompt_manifest_sha256"),
                expectations.prompt_manifest_sha256,
            ),
            "capture_manifest_sha256": (
                provenance.get("capture_manifest_sha256"),
                expectations.capture_manifest_sha256,
            ),
            "statistics_sha256": (
                provenance.get("statistics_sha256"),
                expectations.statistics_sha256,
            ),
        }
        for label, (actual, expected) in comparisons.items():
            if expected is not None and actual != expected:
                raise ValueError(f"artifact expected {label} mismatch")


def load_validated_artifact(
    artifact_path: Path,
    expectations: ArtifactExpectations | None = None,
) -> JsonObject:
    path = artifact_path.resolve()
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError("OSCAR calibration artifact is absent")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("OSCAR calibration artifact must contain a dictionary")
    artifact = cast(JsonObject, loaded)
    validate_artifact(artifact, expectations)
    return artifact


def admit_artifact_for_checkpoint(
    *,
    artifact_path: Path,
    checkpoint_path: Path,
    checkpoint_fingerprint_path: Path,
    model_id: str,
    receipt_path: Path,
) -> JsonObject:
    """Rehash and bind an OSCAR artifact to the exact on-disk model.

    This intentionally hashes every shard referenced by the checkpoint index on
    every admission.  It does not trust path, size, mtime, or a stale receipt.
    Launchers run it once before SGLang forks workers and can then distribute the
    resulting receipt and already-validated artifact file hash.
    """

    artifact_file = artifact_path.resolve()
    checkpoint_dir = checkpoint_path.resolve()
    fingerprint_file = checkpoint_fingerprint_path.resolve()
    if not artifact_file.is_file():
        raise FileNotFoundError("OSCAR calibration artifact is absent")
    if not fingerprint_file.is_file():
        raise FileNotFoundError("checkpoint fingerprint receipt is absent")
    artifact = load_validated_artifact(artifact_file)
    artifact_model = artifact.get("model")
    artifact_provenance = artifact.get("provenance")
    if not isinstance(artifact_model, Mapping) or not isinstance(
        artifact_provenance, Mapping
    ):
        raise TypeError("validated OSCAR artifact metadata is unavailable")
    expected_fingerprint_sha = cast(
        str, artifact_provenance["checkpoint_fingerprint_sha256"]
    )
    actual_fingerprint_sha = sha256_file(fingerprint_file)
    if actual_fingerprint_sha != expected_fingerprint_sha:
        raise ValueError("artifact checkpoint fingerprint receipt hash mismatch")
    fingerprint = verify_checkpoint_fingerprint(fingerprint_file, checkpoint_dir)
    checkpoint_sha = _require_sha256(
        fingerprint.get("checkpoint_sha256"), label="admission checkpoint_sha256"
    )
    actual_geometry = validate_model_config(
        checkpoint_dir / "config.json",
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha,
    )
    artifact_geometry = _geometry_from_dict(artifact_model)
    if actual_geometry != artifact_geometry:
        raise ValueError("OSCAR artifact is not bound to this exact checkpoint/model")
    receipt: JsonObject = {
        "format": ADMISSION_FORMAT,
        "format_version": ADMISSION_VERSION,
        "admitted": True,
        "model_id": model_id,
        "artifact_path": str(artifact_file),
        "artifact_file_sha256": sha256_file(artifact_file),
        "artifact_provenance_sha256": cast(str, artifact["artifact_provenance_sha256"]),
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": actual_geometry.config_sha256,
        "checkpoint_fingerprint_path": str(fingerprint_file),
        "checkpoint_fingerprint_sha256": actual_fingerprint_sha,
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
    }
    receipt["admission_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _atomic_json_save(receipt, receipt_path.resolve())
    return receipt


def _parse_expectations(path: Path) -> ArtifactExpectations:
    value = _load_json_object(path)
    return ArtifactExpectations(
        checkpoint_sha256=_require_sha256(
            value.get("checkpoint_sha256"), label="expected checkpoint_sha256"
        ),
        config_sha256=_require_sha256(
            value.get("config_sha256"), label="expected config_sha256"
        ),
        prompt_manifest_sha256=_require_sha256(
            value.get("prompt_manifest_sha256"),
            label="expected prompt_manifest_sha256",
        ),
        capture_manifest_sha256=(
            _require_sha256(
                value.get("capture_manifest_sha256"),
                label="expected capture_manifest_sha256",
            )
            if value.get("capture_manifest_sha256") is not None
            else None
        ),
        statistics_sha256=(
            _require_sha256(
                value.get("statistics_sha256"), label="expected statistics_sha256"
            )
            if value.get("statistics_sha256") is not None
            else None
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser("fingerprint-checkpoint")
    fingerprint.add_argument("--checkpoint", type=Path, required=True)
    fingerprint.add_argument("--output", type=Path, required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--capture-manifest", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--maximum-sample-rows", type=int, default=2048)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--statistics", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--minimum-train-rows", type=int, default=32)
    calibrate.add_argument("--minimum-heldout-rows", type=int, default=16)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--expected-metadata", type=Path)
    admit = subparsers.add_parser("admit", aliases=["validate-model"])
    admit.add_argument("--artifact", type=Path, required=True)
    admit.add_argument("--checkpoint", type=Path, required=True)
    admit.add_argument("--checkpoint-fingerprint", type=Path, required=True)
    admit.add_argument("--model-id", required=True)
    admit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fingerprint-checkpoint":
        _atomic_json_save(build_checkpoint_fingerprint(args.checkpoint), args.output)
    elif args.command == "collect":
        if args.maximum_sample_rows <= 0:
            raise ValueError("--maximum-sample-rows must be positive")
        collect_statistics(
            args.capture_manifest,
            args.output,
            maximum_sample_rows=args.maximum_sample_rows,
        )
    elif args.command == "calibrate":
        if (
            args.minimum_train_rows < CalibrationGates().minimum_train_rows
            or args.minimum_heldout_rows < CalibrationGates().minimum_heldout_rows
        ):
            raise ValueError(
                "calibration row gates may be tightened but never weakened below "
                "32 train / 16 heldout"
            )
        calibrate_artifact(
            args.statistics,
            args.output,
            gates=CalibrationGates(
                minimum_train_rows=args.minimum_train_rows,
                minimum_heldout_rows=args.minimum_heldout_rows,
            ),
        )
    elif args.command == "validate":
        expectations = (
            _parse_expectations(args.expected_metadata)
            if args.expected_metadata is not None
            else None
        )
        load_validated_artifact(args.artifact, expectations)
    elif args.command in {"admit", "validate-model"}:
        admit_artifact_for_checkpoint(
            artifact_path=args.artifact,
            checkpoint_path=args.checkpoint,
            checkpoint_fingerprint_path=args.checkpoint_fingerprint,
            model_id=args.model_id,
            receipt_path=args.output,
        )
    else:
        raise AssertionError(f"unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
