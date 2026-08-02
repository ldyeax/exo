#!/usr/bin/env python3
"""Build an exact-cover, rank-local GPU/CPU expert placement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, final

import torch

type GpuSelectionStrategy = Literal[
    "profile-hot", "ordering", "profile-hot-prefix-profile-fill"
]


@final
@dataclass(frozen=True)
class LayerGpuSelection:
    hottest_first: tuple[int, ...]
    used_primary_ordering_fallback: bool
    used_fill_ordering_fallback: bool
    primary_profile_expert_count: int
    fill_profile_expert_count: int
    source_ordering_expert_count: int


def parse_rank_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(","))
    if not counts or any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError("rank counts must be non-negative integers")
    return counts


def parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_layer_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ordering layer indices must be comma-separated integers"
        ) from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "ordering layer indices must be non-negative integers"
        )
    return indices


def reduce_profile(raw_counts: torch.Tensor) -> torch.Tensor:
    if (
        raw_counts.dtype == torch.bool
        or raw_counts.is_floating_point()
        or raw_counts.is_complex()
    ):
        raise TypeError(
            f"logical_count must use an integer dtype, got {raw_counts.dtype}"
        )
    if raw_counts.ndim == 2:
        frequency = raw_counts.to(dtype=torch.int64, device="cpu")
    elif raw_counts.ndim == 3:
        frequency = raw_counts.sum(dim=0, dtype=torch.int64).cpu()
    else:
        raise ValueError(
            "logical_count must have shape [layers, experts] or "
            f"[samples, layers, experts], got {tuple(raw_counts.shape)}"
        )
    if frequency.numel() and int(frequency.min()) < 0:
        raise ValueError("logical_count must be non-negative")
    return frequency


def load_profile_frequency(profile_path: Path, *, label: str) -> torch.Tensor:
    loaded_profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded_profile, dict) or not isinstance(
        loaded_profile.get("logical_count"), torch.Tensor
    ):
        raise ValueError(f"{label} must contain a logical_count tensor")
    return reduce_profile(loaded_profile["logical_count"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_by_load(
    expert_ids: list[int],
    frequency: torch.Tensor,
    rank_counts: tuple[int, ...],
) -> list[list[int]]:
    if sum(rank_counts) != len(expert_ids):
        raise ValueError(
            f"rank counts sum to {sum(rank_counts)}, expected {len(expert_ids)}"
        )
    assignments: list[list[int]] = [[] for _ in rank_counts]
    assigned_load = [0 for _ in rank_counts]
    for expert_id in sorted(
        expert_ids, key=lambda current: (-int(frequency[current]), current)
    ):
        eligible_ranks = [
            rank
            for rank, capacity in enumerate(rank_counts)
            if len(assignments[rank]) < capacity
        ]
        if not eligible_ranks:
            raise RuntimeError("no rank has capacity for the remaining expert")
        selected_rank = min(
            eligible_ranks,
            key=lambda rank: (assigned_load[rank], len(assignments[rank]), rank),
        )
        assignments[selected_rank].append(expert_id)
        assigned_load[selected_rank] += int(frequency[expert_id])
    for assignment in assignments:
        assignment.sort()
    return assignments


def assign_hybrid_layer(
    frequency: torch.Tensor,
    hottest_first: Sequence[int],
    gpu_rank_counts: tuple[int, ...],
    cpu_rank_counts: tuple[int, ...],
) -> tuple[list[list[int]], list[list[int]]]:
    """Assign every expert exactly once across rank-local GPU and CPU tiers."""
    num_experts = frequency.numel()
    if len(gpu_rank_counts) != len(cpu_rank_counts):
        raise ValueError("GPU and CPU rank counts must have the same rank count")
    if sum(gpu_rank_counts) + sum(cpu_rank_counts) != num_experts:
        raise ValueError("GPU and CPU rank counts must sum to the global expert count")
    if sorted(hottest_first) != list(range(num_experts)):
        raise ValueError("hottest-first ordering must be an expert permutation")

    gpu_expert_count = sum(gpu_rank_counts)
    gpu_expert_ids = list(hottest_first[:gpu_expert_count])
    gpu_expert_set = set(gpu_expert_ids)
    cpu_expert_ids = [
        expert_id for expert_id in range(num_experts) if expert_id not in gpu_expert_set
    ]
    gpu_assignments = assign_by_load(gpu_expert_ids, frequency, gpu_rank_counts)
    cpu_assignments = assign_by_load(cpu_expert_ids, frequency, cpu_rank_counts)

    assigned = [
        expert_id
        for rank_assignment in (*gpu_assignments, *cpu_assignments)
        for expert_id in rank_assignment
    ]
    if sorted(assigned) != list(range(num_experts)):
        raise RuntimeError("hybrid assignment is not an exact disjoint cover")
    return gpu_assignments, cpu_assignments


def load_hottest_first_ordering(
    ordering_path: Path,
    *,
    num_layers: int,
    num_experts: int,
    layer_indices: tuple[int, ...] | None = None,
) -> list[list[int]]:
    with ordering_path.open("r", encoding="utf-8") as ordering_file:
        loaded = json.load(ordering_file)
    raw_ordering = (
        loaded.get("physical_to_logical_map") if isinstance(loaded, dict) else None
    )
    if not isinstance(raw_ordering, list):
        raise ValueError("ordering must contain physical_to_logical_map rows")
    if layer_indices is None:
        if len(raw_ordering) != num_layers:
            raise ValueError(
                f"ordering must contain {num_layers} physical_to_logical_map rows"
            )
        selected_rows = raw_ordering
    else:
        if len(layer_indices) != num_layers:
            raise ValueError(
                "ordering layer index count must match profile layer count: "
                f"indices={len(layer_indices)} profile={num_layers}"
            )
        if any(index >= len(raw_ordering) for index in layer_indices):
            raise ValueError(
                "ordering layer index is outside physical_to_logical_map: "
                f"rows={len(raw_ordering)} indices={layer_indices}"
            )
        selected_rows = [raw_ordering[index] for index in layer_indices]
    ordering: list[list[int]] = []
    for layer_idx, raw_layer in enumerate(selected_rows):
        if not isinstance(raw_layer, list) or any(
            not isinstance(expert_id, int) for expert_id in raw_layer
        ):
            raise TypeError(f"ordering layer {layer_idx} must contain integer IDs")
        if sorted(raw_layer) != list(range(num_experts)):
            raise ValueError(
                f"ordering layer {layer_idx} is not a {num_experts}-expert permutation"
            )
        ordering.append(raw_layer)
    return ordering


def validate_gpu_selection_configuration(
    *,
    strategy: GpuSelectionStrategy,
    gpu_expert_count: int,
    num_experts: int,
    profile_hot_prefix_experts_per_layer: int | None,
    fill_profile_provided: bool,
) -> None:
    if gpu_expert_count < 0 or gpu_expert_count > num_experts:
        raise ValueError(
            f"GPU expert count must be between 0 and {num_experts}, "
            f"got {gpu_expert_count}"
        )
    if strategy == "profile-hot-prefix-profile-fill":
        if profile_hot_prefix_experts_per_layer is None:
            raise ValueError(
                "profile-hot-prefix-profile-fill requires "
                "--profile-hot-prefix-experts-per-layer"
            )
        if not fill_profile_provided:
            raise ValueError(
                "profile-hot-prefix-profile-fill requires --gpu-fill-profile"
            )
        if not 0 <= profile_hot_prefix_experts_per_layer <= gpu_expert_count:
            raise ValueError(
                "profile-hot prefix expert count must be between 0 and the total "
                f"GPU expert count ({gpu_expert_count}), got "
                f"{profile_hot_prefix_experts_per_layer}"
            )
        return
    if profile_hot_prefix_experts_per_layer is not None:
        raise ValueError(
            "--profile-hot-prefix-experts-per-layer is only valid with "
            "--gpu-selection profile-hot-prefix-profile-fill"
        )
    if fill_profile_provided:
        raise ValueError(
            "--gpu-fill-profile is only valid with "
            "--gpu-selection profile-hot-prefix-profile-fill"
        )


def rank_profile_hottest_first(
    frequency: torch.Tensor, fallback_ordering: list[int]
) -> tuple[list[int], bool]:
    if frequency.ndim != 1:
        raise ValueError(
            f"frequency must be one-dimensional, got {tuple(frequency.shape)}"
        )
    num_experts = frequency.numel()
    if sorted(fallback_ordering) != list(range(num_experts)):
        raise ValueError("fallback ordering must be an expert permutation")
    if frequency.numel() and int(frequency.min()) < 0:
        raise ValueError("expert frequencies must be non-negative")
    if int(frequency.sum()) == 0:
        # Preserve the validated fwuff ordering for an unobserved layer rather
        # than inventing an expert-ID placement.
        return list(fallback_ordering), True
    return (
        sorted(
            range(num_experts),
            key=lambda expert_id: (-int(frequency[expert_id]), expert_id),
        ),
        False,
    )


def select_hottest_first(
    frequency: torch.Tensor,
    fallback_ordering: list[int],
    *,
    strategy: GpuSelectionStrategy,
    gpu_expert_count: int,
    profile_hot_prefix_experts_per_layer: int | None = None,
    fill_frequency: torch.Tensor | None = None,
) -> LayerGpuSelection:
    """Choose one layer's GPU union while retaining real-profile load weights."""
    if frequency.ndim != 1:
        raise ValueError(
            f"frequency must be one-dimensional, got {tuple(frequency.shape)}"
        )
    num_experts = frequency.numel()
    validate_gpu_selection_configuration(
        strategy=strategy,
        gpu_expert_count=gpu_expert_count,
        num_experts=num_experts,
        profile_hot_prefix_experts_per_layer=(profile_hot_prefix_experts_per_layer),
        fill_profile_provided=fill_frequency is not None,
    )
    if sorted(fallback_ordering) != list(range(num_experts)):
        raise ValueError("fallback ordering must be an expert permutation")
    if frequency.numel() and int(frequency.min()) < 0:
        raise ValueError("expert frequencies must be non-negative")
    if strategy == "ordering":
        return LayerGpuSelection(
            hottest_first=tuple(fallback_ordering),
            used_primary_ordering_fallback=False,
            used_fill_ordering_fallback=False,
            primary_profile_expert_count=0,
            fill_profile_expert_count=0,
            source_ordering_expert_count=gpu_expert_count,
        )

    profile_ordering, used_primary_fallback = rank_profile_hottest_first(
        frequency, fallback_ordering
    )
    if strategy == "profile-hot":
        return LayerGpuSelection(
            hottest_first=tuple(profile_ordering),
            used_primary_ordering_fallback=used_primary_fallback,
            used_fill_ordering_fallback=False,
            primary_profile_expert_count=(
                0 if used_primary_fallback else gpu_expert_count
            ),
            fill_profile_expert_count=0,
            source_ordering_expert_count=(
                gpu_expert_count if used_primary_fallback else 0
            ),
        )
    if strategy != "profile-hot-prefix-profile-fill":
        raise ValueError(f"unknown GPU selection strategy: {strategy}")
    if profile_hot_prefix_experts_per_layer is None or fill_frequency is None:
        raise RuntimeError("validated profile-fill inputs are unexpectedly absent")
    if tuple(fill_frequency.shape) != tuple(frequency.shape):
        raise ValueError(
            "fill frequency must have the same shape as primary frequency, got "
            f"{tuple(fill_frequency.shape)} and {tuple(frequency.shape)}"
        )
    fill_ordering, used_fill_fallback = rank_profile_hottest_first(
        fill_frequency, fallback_ordering
    )

    selected_gpu_experts = profile_ordering[:profile_hot_prefix_experts_per_layer]
    selected_gpu_expert_set = set(selected_gpu_experts)
    for expert_id in fill_ordering:
        if len(selected_gpu_experts) == gpu_expert_count:
            break
        if expert_id not in selected_gpu_expert_set:
            selected_gpu_experts.append(expert_id)
            selected_gpu_expert_set.add(expert_id)
    remaining_experts = [
        expert_id
        for expert_id in fallback_ordering
        if expert_id not in selected_gpu_expert_set
    ]
    fill_expert_count = gpu_expert_count - profile_hot_prefix_experts_per_layer
    return LayerGpuSelection(
        hottest_first=tuple((*selected_gpu_experts, *remaining_experts)),
        used_primary_ordering_fallback=used_primary_fallback,
        used_fill_ordering_fallback=used_fill_fallback,
        primary_profile_expert_count=(
            0 if used_primary_fallback else profile_hot_prefix_experts_per_layer
        ),
        fill_profile_expert_count=(0 if used_fill_fallback else fill_expert_count),
        source_ordering_expert_count=(
            (profile_hot_prefix_experts_per_layer if used_primary_fallback else 0)
            + (fill_expert_count if used_fill_fallback else 0)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ordering", type=Path, required=True)
    parser.add_argument(
        "--ordering-layer-indices",
        type=parse_layer_indices,
        help=(
            "Optional comma-separated source-ordering rows for a shorter "
            "profile, such as a three-stage DSpark draft profile"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-rank-counts", type=parse_rank_counts, required=True)
    parser.add_argument("--cpu-rank-counts", type=parse_rank_counts, required=True)
    parser.add_argument(
        "--gpu-selection",
        choices=(
            "profile-hot",
            "ordering",
            "profile-hot-prefix-profile-fill",
        ),
        default="profile-hot",
        help=(
            "Select the GPU union from per-layer profile frequency (default), "
            "reproduce the supplied static ordering, or select a profile-hot "
            "prefix and fill the union from a secondary profile"
        ),
    )
    parser.add_argument(
        "--gpu-fill-profile",
        type=Path,
        help=(
            "Secondary logical-count profile used only to fill the GPU union; "
            "required only for profile-hot-prefix-profile-fill"
        ),
    )
    parser.add_argument(
        "--profile-hot-prefix-experts-per-layer",
        type=parse_non_negative_int,
        help=(
            "Number of experts selected from each non-empty profile layer before "
            "filling the GPU union from the secondary profile; required only for "
            "profile-hot-prefix-profile-fill"
        ),
    )
    arguments = parser.parse_args()

    frequency = load_profile_frequency(arguments.profile, label="profile")
    num_layers, num_experts = frequency.shape
    if num_layers == 0 or num_experts == 0:
        raise ValueError("profile must contain at least one layer and one expert")
    ordering = load_hottest_first_ordering(
        arguments.ordering,
        num_layers=num_layers,
        num_experts=num_experts,
        layer_indices=arguments.ordering_layer_indices,
    )

    if len(arguments.gpu_rank_counts) != len(arguments.cpu_rank_counts):
        raise ValueError("GPU and CPU rank counts must have the same rank count")
    rank_count = len(arguments.gpu_rank_counts)
    gpu_expert_count = sum(arguments.gpu_rank_counts)
    try:
        validate_gpu_selection_configuration(
            strategy=arguments.gpu_selection,
            gpu_expert_count=gpu_expert_count,
            num_experts=num_experts,
            profile_hot_prefix_experts_per_layer=(
                arguments.profile_hot_prefix_experts_per_layer
            ),
            fill_profile_provided=arguments.gpu_fill_profile is not None,
        )
    except ValueError as error:
        parser.error(str(error))
    if sum(arguments.gpu_rank_counts) + sum(arguments.cpu_rank_counts) != num_experts:
        parser.error(
            "GPU and CPU rank counts must sum to the global expert count "
            f"({num_experts})"
        )

    fill_frequency: torch.Tensor | None = None
    if arguments.gpu_fill_profile is not None:
        fill_frequency = load_profile_frequency(
            arguments.gpu_fill_profile, label="GPU fill profile"
        )
        if tuple(fill_frequency.shape) != tuple(frequency.shape):
            raise ValueError(
                "GPU fill profile logical_count must match profile shape "
                f"{tuple(frequency.shape)}, got {tuple(fill_frequency.shape)}"
            )
    gpu_masks_by_rank = torch.zeros(
        (rank_count, num_layers, num_experts), dtype=torch.bool
    )
    cpu_expert_ids_by_rank: list[list[list[int]]] = [[] for _ in range(rank_count)]
    primary_zero_count_fallback_layers: list[int] = []
    fill_zero_count_fallback_layers: list[int] = []
    primary_profile_selected_counts_by_layer: list[int] = []
    fill_profile_selected_counts_by_layer: list[int] = []
    source_ordering_selected_counts_by_layer: list[int] = []
    for layer_idx in range(num_layers):
        selection = select_hottest_first(
            frequency[layer_idx],
            ordering[layer_idx],
            strategy=arguments.gpu_selection,
            gpu_expert_count=gpu_expert_count,
            profile_hot_prefix_experts_per_layer=(
                arguments.profile_hot_prefix_experts_per_layer
            ),
            fill_frequency=(
                fill_frequency[layer_idx] if fill_frequency is not None else None
            ),
        )
        if selection.used_primary_ordering_fallback:
            primary_zero_count_fallback_layers.append(layer_idx)
        if selection.used_fill_ordering_fallback:
            fill_zero_count_fallback_layers.append(layer_idx)
        primary_profile_selected_counts_by_layer.append(
            selection.primary_profile_expert_count
        )
        fill_profile_selected_counts_by_layer.append(
            selection.fill_profile_expert_count
        )
        source_ordering_selected_counts_by_layer.append(
            selection.source_ordering_expert_count
        )
        gpu_assignments, cpu_assignments = assign_hybrid_layer(
            frequency[layer_idx],
            selection.hottest_first,
            arguments.gpu_rank_counts,
            arguments.cpu_rank_counts,
        )
        for rank in range(rank_count):
            gpu_masks_by_rank[rank, layer_idx, gpu_assignments[rank]] = True
            cpu_expert_ids_by_rank[rank].append(cpu_assignments[rank])

    cpu_tensors_by_rank = [
        torch.tensor(rank_assignments, dtype=torch.int64)
        for rank_assignments in cpu_expert_ids_by_rank
    ]
    total_frequency = frequency.sum().clamp_min(1).to(torch.float64)
    gpu_activation_fractions = torch.tensor(
        [
            frequency[gpu_masks_by_rank[rank]].sum() / total_frequency
            for rank in range(rank_count)
        ],
        dtype=torch.float64,
    )
    cpu_activation_fractions = torch.stack(
        [
            frequency.gather(1, cpu_tensors_by_rank[rank]).sum().to(torch.float64)
            / total_frequency
            for rank in range(rank_count)
        ]
    )
    output = {
        "format": "sglang_kt_hybrid_expert_shard_v1",
        "gpu_experts_mask_by_rank": gpu_masks_by_rank,
        "cpu_expert_ids_by_rank": cpu_tensors_by_rank,
        "gpu_rank_counts": torch.tensor(arguments.gpu_rank_counts, dtype=torch.int64),
        "cpu_rank_counts": torch.tensor(arguments.cpu_rank_counts, dtype=torch.int64),
        "global_num_experts": torch.tensor(num_experts, dtype=torch.int64),
        "gpu_union_expert_count": torch.tensor(gpu_expert_count, dtype=torch.int64),
        "source_profile": str(arguments.profile.resolve()),
        "source_profile_sha256": sha256_file(arguments.profile),
        "source_ordering": str(arguments.ordering.resolve()),
        "source_ordering_sha256": sha256_file(arguments.ordering),
        "gpu_selection_strategy": arguments.gpu_selection,
        "gpu_primary_profile_selected_counts_by_layer": torch.tensor(
            primary_profile_selected_counts_by_layer, dtype=torch.int64
        ),
        "gpu_fill_profile_selected_counts_by_layer": torch.tensor(
            fill_profile_selected_counts_by_layer, dtype=torch.int64
        ),
        "gpu_source_ordering_selected_counts_by_layer": torch.tensor(
            source_ordering_selected_counts_by_layer, dtype=torch.int64
        ),
        "zero_count_layer_fallback": "source_ordering",
        "zero_count_fallback_layers": torch.tensor(
            primary_zero_count_fallback_layers, dtype=torch.int64
        ),
        "primary_zero_count_fallback_layers": torch.tensor(
            primary_zero_count_fallback_layers, dtype=torch.int64
        ),
        "fill_zero_count_fallback_layers": torch.tensor(
            fill_zero_count_fallback_layers, dtype=torch.int64
        ),
        "gpu_activation_fractions": gpu_activation_fractions,
        "cpu_activation_fractions": cpu_activation_fractions,
    }
    if arguments.profile_hot_prefix_experts_per_layer is not None:
        output["gpu_profile_hot_prefix_experts_per_layer"] = torch.tensor(
            arguments.profile_hot_prefix_experts_per_layer, dtype=torch.int64
        )
    if arguments.gpu_fill_profile is not None:
        output["source_fill_profile"] = str(arguments.gpu_fill_profile.resolve())
        output["source_fill_profile_sha256"] = sha256_file(arguments.gpu_fill_profile)
    if arguments.ordering_layer_indices is not None:
        output["source_ordering_layer_indices"] = torch.tensor(
            arguments.ordering_layer_indices, dtype=torch.int64
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, arguments.output)
    print(
        f"wrote {arguments.output}: layers={num_layers} experts={num_experts} "
        f"gpu_rank_counts={arguments.gpu_rank_counts} "
        f"cpu_rank_counts={arguments.cpu_rank_counts} "
        f"gpu_selection={arguments.gpu_selection} "
        "profile_hot_prefix_experts_per_layer="
        f"{arguments.profile_hot_prefix_experts_per_layer} "
        f"primary_zero_count_fallback_layers={primary_zero_count_fallback_layers} "
        f"fill_zero_count_fallback_layers={fill_zero_count_fallback_layers} "
        "primary_profile_selected_count_range="
        f"{min(primary_profile_selected_counts_by_layer)}-"
        f"{max(primary_profile_selected_counts_by_layer)} "
        "fill_profile_selected_count_range="
        f"{min(fill_profile_selected_counts_by_layer)}-"
        f"{max(fill_profile_selected_counts_by_layer)} "
        "source_ordering_selected_count_range="
        f"{min(source_ordering_selected_counts_by_layer)}-"
        f"{max(source_ordering_selected_counts_by_layer)} "
        "gpu_activation_fractions="
        f"{','.join(f'{float(value):.6f}' for value in gpu_activation_fractions)} "
        "cpu_activation_fractions="
        f"{','.join(f'{float(value):.6f}' for value in cpu_activation_fractions)}"
    )


if __name__ == "__main__":
    main()
