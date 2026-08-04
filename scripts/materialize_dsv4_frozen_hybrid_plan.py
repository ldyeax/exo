#!/usr/bin/env python3
"""Materialize a reviewable frozen DSV4 hybrid placement as a runtime plan."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, final

import torch

MANIFEST_FORMAT: Final = "dsv4_frozen_hybrid_expert_placement_v1"
PLAN_FORMAT: Final = "sglang_kt_hybrid_expert_shard_v1"
MaterializationAction = Literal["created", "replaced", "reused"]


@final
@dataclass(frozen=True)
class FrozenPlacement:
    """Validated placement tensors and their immutable identity."""

    name: str
    gpu_masks_by_rank: torch.Tensor
    cpu_expert_ids_by_rank: tuple[torch.Tensor, torch.Tensor]
    gpu_experts_per_rank: int
    cpu_experts_per_rank: int
    hot_prefix_count: int
    placement_semantics_sha256: str
    provenance: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def placement_semantics_sha256(
    gpu_masks_by_rank: torch.Tensor,
    cpu_expert_ids_by_rank: tuple[torch.Tensor, torch.Tensor],
) -> str:
    serialized = json.dumps(
        {
            "gpu_experts_mask_by_rank": gpu_masks_by_rank.to(torch.uint8).tolist(),
            "cpu_expert_ids_by_rank": [
                expert_ids.to(torch.int64).tolist()
                for expert_ids in cpu_expert_ids_by_rank
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _required_integer(manifest: dict[str, object], key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"manifest field {key!r} must be an integer")
    return value


def _required_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def _parse_gpu_expert_ids(
    raw_gpu_expert_ids: object,
    *,
    rank_count: int,
    num_layers: int,
    num_experts: int,
    gpu_experts_per_rank: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if (
        not isinstance(raw_gpu_expert_ids, list)
        or len(raw_gpu_expert_ids) != rank_count
    ):
        raise ValueError(
            "gpu_expert_ids_by_rank must contain exactly one entry per rank"
        )
    parsed_ranks: list[tuple[tuple[int, ...], ...]] = []
    for rank, raw_layers in enumerate(raw_gpu_expert_ids):
        if not isinstance(raw_layers, list) or len(raw_layers) != num_layers:
            raise ValueError(
                f"gpu_expert_ids_by_rank[{rank}] must contain {num_layers} layers"
            )
        parsed_layers: list[tuple[int, ...]] = []
        for layer, raw_expert_ids in enumerate(raw_layers):
            if not isinstance(raw_expert_ids, list):
                raise TypeError(
                    f"GPU expert IDs for rank {rank} layer {layer} must be a list"
                )
            if any(
                not isinstance(expert_id, int) or isinstance(expert_id, bool)
                for expert_id in raw_expert_ids
            ):
                raise ValueError(
                    f"GPU expert IDs for rank {rank} layer {layer} must be integers"
                )
            expert_ids = tuple(cast(list[int], raw_expert_ids))
            if len(expert_ids) != gpu_experts_per_rank:
                raise ValueError(
                    f"rank {rank} layer {layer} must contain "
                    f"{gpu_experts_per_rank} GPU experts"
                )
            if expert_ids != tuple(sorted(set(expert_ids))):
                raise ValueError(
                    f"GPU expert IDs for rank {rank} layer {layer} must be "
                    "unique and sorted"
                )
            if expert_ids and (expert_ids[0] < 0 or expert_ids[-1] >= num_experts):
                raise ValueError(
                    f"GPU expert ID for rank {rank} layer {layer} is out of range"
                )
            parsed_layers.append(expert_ids)
        parsed_ranks.append(tuple(parsed_layers))
    return tuple(parsed_ranks)


def load_frozen_placement(manifest_path: Path) -> FrozenPlacement:
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("frozen placement manifest must contain a JSON object")
    manifest = cast(dict[str, object], loaded)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"frozen placement format must be {MANIFEST_FORMAT!r}, "
            f"got {manifest.get('format')!r}"
        )

    name = _required_string(manifest, "name")
    num_layers = _required_integer(manifest, "num_layers")
    num_experts = _required_integer(manifest, "num_experts")
    rank_count = _required_integer(manifest, "rank_count")
    gpu_experts_per_rank = _required_integer(manifest, "gpu_experts_per_rank")
    cpu_experts_per_rank = _required_integer(manifest, "cpu_experts_per_rank")
    hot_prefix_count = _required_integer(manifest, "hot_prefix_count")
    expected_semantics_sha256 = _required_string(manifest, "placement_semantics_sha256")
    if num_layers <= 0 or num_experts <= 0:
        raise ValueError("num_layers and num_experts must be positive")
    if rank_count != 2:
        raise ValueError("frozen DSV4 placement currently requires exactly two ranks")
    if gpu_experts_per_rank < 0 or cpu_experts_per_rank < 0:
        raise ValueError("GPU and CPU expert counts must be non-negative")
    if rank_count * (gpu_experts_per_rank + cpu_experts_per_rank) != num_experts:
        raise ValueError("rank-local GPU and CPU counts must exactly cover all experts")
    if not 0 <= hot_prefix_count <= rank_count * gpu_experts_per_rank:
        raise ValueError("hot_prefix_count must fit inside the GPU expert union")

    raw_ownership = manifest.get("rank0_ownership_hex")
    if not isinstance(raw_ownership, list) or len(raw_ownership) != num_layers:
        raise ValueError(
            f"rank0_ownership_hex must contain exactly {num_layers} layers"
        )
    hexadecimal_width = (num_experts + 3) // 4
    rank0_ownership: list[int] = []
    for layer, raw_bitset in enumerate(raw_ownership):
        if (
            not isinstance(raw_bitset, str)
            or len(raw_bitset) != hexadecimal_width
            or any(character not in "0123456789abcdef" for character in raw_bitset)
        ):
            raise ValueError(
                f"rank0 ownership for layer {layer} must be a lowercase "
                f"{hexadecimal_width}-digit hexadecimal bitset"
            )
        bitset = int(raw_bitset, 16)
        if bitset >> num_experts:
            raise ValueError(f"rank0 ownership for layer {layer} has excess bits")
        expected_rank_owned_count = gpu_experts_per_rank + cpu_experts_per_rank
        if bitset.bit_count() != expected_rank_owned_count:
            raise ValueError(
                f"rank0 ownership for layer {layer} must contain "
                f"{expected_rank_owned_count} experts"
            )
        rank0_ownership.append(bitset)

    gpu_expert_ids_by_rank = _parse_gpu_expert_ids(
        manifest.get("gpu_expert_ids_by_rank"),
        rank_count=rank_count,
        num_layers=num_layers,
        num_experts=num_experts,
        gpu_experts_per_rank=gpu_experts_per_rank,
    )
    gpu_masks_by_rank = torch.zeros(
        (rank_count, num_layers, num_experts), dtype=torch.bool
    )
    cpu_expert_ids_by_rank: list[list[list[int]]] = [[], []]
    for layer, rank0_bitset in enumerate(rank0_ownership):
        owner_by_expert = [
            0 if (rank0_bitset >> expert_id) & 1 else 1
            for expert_id in range(num_experts)
        ]
        for rank in range(rank_count):
            gpu_expert_ids = gpu_expert_ids_by_rank[rank][layer]
            if any(owner_by_expert[expert_id] != rank for expert_id in gpu_expert_ids):
                raise ValueError(
                    f"GPU placement escapes rank ownership at rank {rank} layer {layer}"
                )
            gpu_masks_by_rank[rank, layer, list(gpu_expert_ids)] = True
            gpu_expert_id_set = set(gpu_expert_ids)
            cpu_expert_ids = [
                expert_id
                for expert_id, owner in enumerate(owner_by_expert)
                if owner == rank and expert_id not in gpu_expert_id_set
            ]
            if len(cpu_expert_ids) != cpu_experts_per_rank:
                raise ValueError(
                    f"rank {rank} layer {layer} does not leave "
                    f"{cpu_experts_per_rank} CPU experts"
                )
            cpu_expert_ids_by_rank[rank].append(cpu_expert_ids)

    cpu_tensors = cast(
        tuple[torch.Tensor, torch.Tensor],
        tuple(
            torch.tensor(expert_ids, dtype=torch.int64)
            for expert_ids in cpu_expert_ids_by_rank
        ),
    )
    actual_semantics_sha256 = placement_semantics_sha256(gpu_masks_by_rank, cpu_tensors)
    if actual_semantics_sha256 != expected_semantics_sha256:
        raise ValueError(
            "frozen placement semantic SHA-256 mismatch: "
            f"expected {expected_semantics_sha256}, got {actual_semantics_sha256}"
        )
    raw_provenance = manifest.get("provenance", {})
    if not isinstance(raw_provenance, dict):
        raise TypeError("provenance must be a JSON object")
    return FrozenPlacement(
        name=name,
        gpu_masks_by_rank=gpu_masks_by_rank,
        cpu_expert_ids_by_rank=cpu_tensors,
        gpu_experts_per_rank=gpu_experts_per_rank,
        cpu_experts_per_rank=cpu_experts_per_rank,
        hot_prefix_count=hot_prefix_count,
        placement_semantics_sha256=actual_semantics_sha256,
        provenance=cast(dict[str, object], raw_provenance),
    )


def build_runtime_plan(
    placement: FrozenPlacement,
    *,
    manifest_path: Path,
    source_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "format": PLAN_FORMAT,
        "gpu_experts_mask_by_rank": placement.gpu_masks_by_rank,
        "cpu_expert_ids_by_rank": list(placement.cpu_expert_ids_by_rank),
        "gpu_rank_counts": torch.tensor(
            [placement.gpu_experts_per_rank] * 2, dtype=torch.int64
        ),
        "cpu_rank_counts": torch.tensor(
            [placement.cpu_experts_per_rank] * 2, dtype=torch.int64
        ),
        "global_num_experts": torch.tensor(
            placement.gpu_masks_by_rank.shape[-1], dtype=torch.int64
        ),
        "gpu_union_expert_count": torch.tensor(
            placement.gpu_experts_per_rank * 2, dtype=torch.int64
        ),
        "gpu_selection_strategy": "frozen-multi-prompt-critical-tail",
        "gpu_profile_hot_prefix_experts_per_layer": torch.tensor(
            placement.hot_prefix_count, dtype=torch.int64
        ),
        "rank_ownership_constraint": "frozen-baseline-plan",
        "placement_semantics_sha256": placement.placement_semantics_sha256,
        "frozen_placement_name": placement.name,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "frozen_placement_provenance": placement.provenance,
    }


def _runtime_plan_values_equal(expected: object, actual: object) -> bool:
    """Compare a loaded weights-only plan without trusting its claimed hashes."""

    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and expected.dtype == actual.dtype
            and tuple(expected.shape) == tuple(actual.shape)
            and torch.equal(expected.cpu(), actual.cpu())
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        actual_dict = cast(dict[object, object], actual)
        return expected.keys() == actual_dict.keys() and all(
            _runtime_plan_values_equal(value, actual_dict[key])
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        actual_sequence = cast(list[object] | tuple[object, ...], actual)
        return len(expected) == len(actual_sequence) and all(
            _runtime_plan_values_equal(left, right)
            for left, right in zip(expected, actual_sequence, strict=True)
        )
    return expected == actual


def _is_reusable_runtime_plan(plan: dict[str, object], output_path: Path) -> bool:
    if output_path.is_symlink():
        raise ValueError(f"refusing to reuse or replace symlink plan: {output_path}")
    if not output_path.exists():
        return False
    if not output_path.is_file():
        raise ValueError(f"runtime plan output is not a regular file: {output_path}")
    try:
        loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError):
        return False
    return isinstance(loaded, dict) and _runtime_plan_values_equal(plan, loaded)


def write_runtime_plan(
    plan: dict[str, object], output_path: Path
) -> MaterializationAction:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_existed = output_path.exists()
    if _is_reusable_runtime_plan(plan, output_path):
        return "reused"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        # Passing a path makes torch.save derive the ZIP archive root from that
        # path.  The randomized atomic-temporary filename therefore changed the
        # byte-level SHA-256 on every launcher invocation.  A BytesIO object uses
        # PyTorch's stable ``archive/`` root; the resulting deterministic bytes
        # can still be installed through an atomic temporary-file replacement.
        serialized_plan = io.BytesIO()
        torch.save(plan, serialized_plan)
        with temporary_path.open("wb") as temporary_file:
            temporary_file.write(serialized_plan.getbuffer())
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        return "replaced" if output_existed else "created"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source_manifest_sha256 = sha256_file(arguments.manifest)
    placement = load_frozen_placement(arguments.manifest)
    if sha256_file(arguments.manifest) != source_manifest_sha256:
        raise ValueError("frozen placement manifest changed while loading")
    plan = build_runtime_plan(
        placement,
        manifest_path=arguments.manifest,
        source_manifest_sha256=source_manifest_sha256,
    )
    materialization = write_runtime_plan(plan, arguments.output)
    output_sha256 = sha256_file(arguments.output)
    print(
        json.dumps(
            {
                "materialization": materialization,
                "output": str(arguments.output.resolve()),
                "output_sha256": output_sha256,
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "placement_semantics_sha256": placement.placement_semantics_sha256,
                "gpu_rank_counts": [placement.gpu_experts_per_rank] * 2,
                "cpu_rank_counts": [placement.cpu_experts_per_rank] * 2,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
