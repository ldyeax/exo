#!/usr/bin/env python3
"""Validate the DSV4 Flash 0731 inputs and build a logical GPU-expert plan.

The default action is read-only. Passing ``--write-plan`` materializes the
small torch mask consumed by the accumulated KTransformers/SGLang fork; it
never starts a server or loads model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

MODEL_REVISION: Final = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_WEIGHT_BYTES: Final = 166_878_536_440
ROUTED_LAYER_COUNT: Final = 43
EXPERT_COUNT: Final = 256
SHARD_COUNT: Final = 48
DEFAULT_GPU_EXPERTS_PER_LAYER: Final = 48


class PreparationError(RuntimeError):
    """Raised when a launch input does not match the pinned contract."""


@dataclass(frozen=True)
class PreparationReceipt:
    model_path: str
    model_revision: str
    model_weight_bytes: int
    shard_count: int
    ordering_path: str
    ordering_sha256: str
    gpu_experts_per_layer: int
    gpu_expert_slots: int
    plan_path: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"expected a JSON object: {path}")
    return value


def validate_ordering(path: Path) -> tuple[tuple[int, ...], ...]:
    payload = load_json_object(path)
    raw_rows = payload.get("physical_to_logical_map")
    if not isinstance(raw_rows, list) or len(raw_rows) != ROUTED_LAYER_COUNT:
        raise PreparationError(
            f"expert ordering must contain {ROUTED_LAYER_COUNT} layers"
        )

    expected = set(range(EXPERT_COUNT))
    rows: list[tuple[int, ...]] = []
    for layer_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, list) or len(raw_row) != EXPERT_COUNT:
            raise PreparationError(
                f"expert ordering layer {layer_index} must contain {EXPERT_COUNT} IDs"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_row):
            raise PreparationError(
                f"expert ordering layer {layer_index} contains a non-integer ID"
            )
        row = tuple(raw_row)
        if set(row) != expected:
            raise PreparationError(
                f"expert ordering layer {layer_index} is not a complete permutation"
            )
        rows.append(row)
    return tuple(rows)


def _metadata_revision(model_path: Path) -> str:
    metadata_path = (
        model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    )
    try:
        first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as error:
        raise PreparationError(
            f"cannot read model revision metadata: {metadata_path}"
        ) from error
    return first_line.strip()


def validate_model(model_path: Path) -> tuple[int, int]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config = load_json_object(config_path)
    index = load_json_object(index_path)

    expected_config: dict[str, object] = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": ROUTED_LAYER_COUNT,
        "n_routed_experts": EXPERT_COUNT,
        "num_experts_per_tok": 6,
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "max_position_embeddings": 1_048_576,
        "torch_dtype": "bfloat16",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise PreparationError(f"model config does not match DSV4 0731: {mismatches}")

    revision = _metadata_revision(model_path)
    if revision != MODEL_REVISION:
        raise PreparationError(
            f"model revision mismatch: actual={revision} expected={MODEL_REVISION}"
        )

    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise PreparationError("invalid safetensors index structure")
    total_size = metadata.get("total_size")
    if total_size != MODEL_WEIGHT_BYTES:
        raise PreparationError(
            f"model weight size mismatch: actual={total_size} expected={MODEL_WEIGHT_BYTES}"
        )

    raw_shard_names = tuple(weight_map.values())
    if any(not isinstance(name, str) for name in raw_shard_names):
        raise PreparationError("safetensors index contains a non-string shard name")
    shard_names = sorted(
        {name for name in raw_shard_names if isinstance(name, str)}
    )
    if len(shard_names) != SHARD_COUNT:
        raise PreparationError(
            f"expected {SHARD_COUNT} unique safetensors shards, got {len(shard_names)}"
        )
    missing = [name for name in shard_names if not (model_path / name).is_file()]
    empty = [name for name in shard_names if (model_path / name).stat().st_size == 0]
    incomplete = list(model_path.glob("*.incomplete"))
    if missing or empty or incomplete:
        raise PreparationError(
            f"incomplete model: missing={missing} empty={empty} partial={incomplete}"
        )
    return int(total_size), len(shard_names)


def write_mask_plan(
    output_path: Path,
    ordering: tuple[tuple[int, ...], ...],
    *,
    ordering_sha256: str,
    gpu_experts_per_layer: int,
) -> None:
    if not 0 <= gpu_experts_per_layer <= EXPERT_COUNT:
        raise PreparationError(
            f"GPU experts per layer must be in [0, {EXPERT_COUNT}]"
        )
    try:
        import torch
    except ImportError as error:
        raise PreparationError("torch is required only when writing the mask plan") from error

    mask = torch.zeros(
        (ROUTED_LAYER_COUNT, EXPERT_COUNT), dtype=torch.bool, device="cpu"
    )
    for layer_index, row in enumerate(ordering):
        selected = torch.tensor(
            row[:gpu_experts_per_layer], dtype=torch.int64, device="cpu"
        )
        mask[layer_index, selected] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(
        {
            "gpu_experts_mask": mask,
            "model_revision": MODEL_REVISION,
            "ordering_sha256": ordering_sha256,
            "gpu_experts_per_layer": gpu_experts_per_layer,
        },
        temporary_path,
    )
    os.replace(temporary_path, output_path)


def prepare(
    *,
    model_path: Path,
    ordering_path: Path,
    gpu_experts_per_layer: int,
    plan_path: Path | None,
) -> PreparationReceipt:
    model_weight_bytes, shard_count = validate_model(model_path)
    ordering = validate_ordering(ordering_path)
    ordering_sha256 = sha256_file(ordering_path)
    if plan_path is not None:
        write_mask_plan(
            plan_path,
            ordering,
            ordering_sha256=ordering_sha256,
            gpu_experts_per_layer=gpu_experts_per_layer,
        )
    return PreparationReceipt(
        model_path=str(model_path.resolve()),
        model_revision=MODEL_REVISION,
        model_weight_bytes=model_weight_bytes,
        shard_count=shard_count,
        ordering_path=str(ordering_path.resolve()),
        ordering_sha256=ordering_sha256,
        gpu_experts_per_layer=gpu_experts_per_layer,
        gpu_expert_slots=ROUTED_LAYER_COUNT * gpu_experts_per_layer,
        plan_path=str(plan_path.resolve()) if plan_path is not None else None,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ordering", required=True, type=Path)
    parser.add_argument(
        "--gpu-experts-per-layer",
        type=int,
        default=DEFAULT_GPU_EXPERTS_PER_LAYER,
    )
    parser.add_argument(
        "--write-plan",
        type=Path,
        help="Write the validated torch mask plan; omitted for read-only preflight.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        receipt = prepare(
            model_path=arguments.model,
            ordering_path=arguments.ordering,
            gpu_experts_per_layer=arguments.gpu_experts_per_layer,
            plan_path=arguments.write_plan,
        )
    except PreparationError as error:
        raise SystemExit(f"DSV4 preparation failed: {error}") from error
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
