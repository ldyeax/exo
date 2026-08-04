#!/usr/bin/env python3
"""Build and cross-validate target-MoE critical-tail placement variants.

The runtime waits for both rank-local CPU shards at every MoE layer. Therefore
the placement objective here is the sum of the slower rank's per-layer cost,
not global expert frequency. Each named input profile is treated as one prompt
fold so a narrow route trace cannot win without a held-out receipt exposing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, final

import torch

try:
    from scripts.build_dsv4_kt_hybrid_shard_plan import (
        load_hottest_first_ordering,
        load_profile_frequency,
        sha256_file,
    )
except ModuleNotFoundError:
    from build_dsv4_kt_hybrid_shard_plan import (
        load_hottest_first_ordering,
        load_profile_frequency,
        sha256_file,
    )


PLAN_FORMAT = "sglang_kt_hybrid_expert_shard_v1"
SWEEP_RECEIPT_FORMAT = "dsv4_critical_tail_placement_sweep_v1"
COMPACT_PROFILE_FORMAT = "dsv4_compact_logical_count_profile_v1"


@final
@dataclass(frozen=True)
class PromptCostProfile:
    label: str
    path: Path
    normalized_cost: torch.Tensor
    raw_cost: torch.Tensor
    sample_count: int
    source_kind: str


@final
@dataclass(frozen=True)
class PlacementVariant:
    gpu_masks_by_rank: torch.Tensor
    cpu_expert_ids_by_rank: tuple[torch.Tensor, ...]
    gpu_expert_ids_by_rank: tuple[torch.Tensor, ...]
    metrics: dict[str, object]


def parse_named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("profile must be LABEL=/path/to/profile")
    if any(character.isspace() for character in label):
        raise argparse.ArgumentTypeError("profile label must not contain whitespace")
    return label, Path(raw_path)


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must contain comma-separated positive integers"
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parsed


def parse_prefix_list(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for part in value.split(","):
        if part == "full":
            parsed.append(None)
            continue
        try:
            count = int(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "hot-prefix counts must be non-negative integers or 'full'"
            ) from error
        if count < 0:
            raise argparse.ArgumentTypeError("hot-prefix counts must be non-negative")
        parsed.append(count)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one hot-prefix count is required")
    return tuple(parsed)


def _validate_count_tensor(counts: torch.Tensor, *, label: str) -> torch.Tensor:
    if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
        raise TypeError(f"{label} must use an integer dtype, got {counts.dtype}")
    counts = counts.to(dtype=torch.int64, device="cpu")
    if counts.numel() and int(counts.min()) < 0:
        raise ValueError(f"{label} must be non-negative")
    return counts


def _physical_to_logical_counts(
    physical_counts: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    physical_counts = _validate_count_tensor(
        physical_counts, label="global_physical_count"
    )
    physical_to_logical_map = _validate_count_tensor(
        physical_to_logical_map, label="physical_to_logical_map"
    )
    if physical_counts.ndim != 3:
        raise ValueError(
            "global_physical_count samples must have shape [samples,layers,experts]"
        )
    if tuple(physical_counts.shape[1:]) != tuple(physical_to_logical_map.shape):
        raise ValueError(
            "physical count/map shape mismatch: "
            f"{tuple(physical_counts.shape[1:])} vs "
            f"{tuple(physical_to_logical_map.shape)}"
        )
    num_experts = physical_counts.shape[-1]
    expected_ids = torch.arange(num_experts, dtype=torch.int64)
    if any(
        not torch.equal(torch.sort(layer_map).values, expected_ids)
        for layer_map in physical_to_logical_map
    ):
        raise ValueError("each physical-to-logical map row must be a permutation")
    logical_counts = torch.zeros_like(physical_counts)
    logical_counts.scatter_add_(
        2,
        physical_to_logical_map.unsqueeze(0).expand_as(physical_counts),
        physical_counts,
    )
    return logical_counts


def _cost_from_samples(
    samples: torch.Tensor,
    *,
    active_expert_cost: float,
    additional_row_cost: float,
) -> torch.Tensor:
    active = samples > 0
    additional_rows = torch.clamp(samples - 1, min=0)
    return (
        active.sum(dim=0).to(torch.float64) * active_expert_cost
        + additional_rows.sum(dim=0).to(torch.float64) * additional_row_cost
    ) / samples.shape[0]


def _normalize_layer_cost(raw_cost: torch.Tensor) -> torch.Tensor:
    layer_totals = raw_cost.sum(dim=1, keepdim=True)
    return torch.where(
        layer_totals > 0,
        raw_cost / layer_totals.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(raw_cost),
    )


def _filter_sampled_logical_counts(
    logical_count: torch.Tensor,
    *,
    label: str,
    routes_per_layer: int | None,
) -> torch.Tensor:
    """Retain samples whose every layer has the requested route count.

    Recorder ``logical_count`` dumps interleave target decode samples with
    all-zero draft samples and much larger prefill samples.  Apply the same
    all-layer route-shape predicate used by the raw-record loader so those
    phases cannot silently contaminate a decode placement sweep.
    """
    if routes_per_layer is None:
        return logical_count
    sample_matches = torch.all(
        logical_count.sum(dim=2) == routes_per_layer,
        dim=1,
    )
    selected = logical_count[sample_matches]
    if selected.shape[0] == 0:
        raise ValueError(
            f"profile {label!r} has no logical_count samples matching "
            f"routes_per_layer={routes_per_layer}"
        )
    return selected


def load_prompt_cost_profile(
    label: str,
    path: Path,
    *,
    routes_per_layer: int | None,
    active_expert_cost: float,
    additional_row_cost: float,
) -> PromptCostProfile:
    if path.suffix.lower() == ".json":
        frequency = load_profile_frequency(path, label=f"profile {label!r}")
        raw_cost = frequency.to(torch.float64) * active_expert_cost
        return PromptCostProfile(
            label=label,
            path=path.resolve(),
            normalized_cost=_normalize_layer_cost(raw_cost),
            raw_cost=raw_cost,
            sample_count=1,
            source_kind="aggregate_sparse_logical_count",
        )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"profile {label!r} must contain a dictionary")

    if loaded.get("format") == COMPACT_PROFILE_FORMAT:
        summed_count = loaded.get("logical_count")
        active_count = loaded.get("active_expert_count")
        additional_count = loaded.get("additional_row_count")
        raw_sample_count = loaded.get("sample_count")
        raw_routes_per_layer = loaded.get("routes_per_layer")
        if not all(
            isinstance(item, torch.Tensor)
            for item in (
                summed_count,
                active_count,
                additional_count,
                raw_sample_count,
                raw_routes_per_layer,
            )
        ):
            raise TypeError(f"compact profile {label!r} has malformed tensor fields")
        summed_count = _validate_count_tensor(
            summed_count, label=f"compact profile {label!r} logical_count"
        )
        active_count = _validate_count_tensor(
            active_count, label=f"compact profile {label!r} active_expert_count"
        )
        additional_count = _validate_count_tensor(
            additional_count,
            label=f"compact profile {label!r} additional_row_count",
        )
        if (
            summed_count.ndim != 2
            or active_count.shape != summed_count.shape
            or additional_count.shape != summed_count.shape
        ):
            raise ValueError(
                f"compact profile {label!r} count fields must share [layers,experts]"
            )
        if not torch.equal(summed_count, active_count + additional_count):
            raise ValueError(
                f"compact profile {label!r} logical-count accounting is inconsistent"
            )
        sample_count = int(raw_sample_count)
        compact_routes_per_layer = int(raw_routes_per_layer)
        if sample_count <= 0 or compact_routes_per_layer <= 0:
            raise ValueError(f"compact profile {label!r} metadata must be positive")
        if (
            routes_per_layer is not None
            and routes_per_layer != compact_routes_per_layer
        ):
            raise ValueError(
                f"compact profile {label!r} was filtered for "
                f"routes_per_layer={compact_routes_per_layer}, requested "
                f"{routes_per_layer}"
            )
        raw_cost = (
            active_count.to(torch.float64) * active_expert_cost
            + additional_count.to(torch.float64) * additional_row_cost
        ) / sample_count
        return PromptCostProfile(
            label=label,
            path=path.resolve(),
            normalized_cost=_normalize_layer_cost(raw_cost),
            raw_cost=raw_cost,
            sample_count=sample_count,
            source_kind="compact_sampled_logical_count",
        )

    records = loaded.get("records")
    physical_to_logical_map = loaded.get("last_physical_to_logical_map")
    if isinstance(records, list) and isinstance(physical_to_logical_map, torch.Tensor):
        selected_counts: list[torch.Tensor] = []
        for record in records:
            if not isinstance(record, dict) or record.get("gatherer_key") != "primary":
                continue
            counts = record.get("global_physical_count")
            if not isinstance(counts, torch.Tensor):
                continue
            if routes_per_layer is not None and not bool(
                torch.all(counts.sum(dim=1) == routes_per_layer)
            ):
                continue
            selected_counts.append(counts)
        if not selected_counts:
            raise ValueError(
                f"profile {label!r} has no primary records matching "
                f"routes_per_layer={routes_per_layer}"
            )
        samples = _physical_to_logical_counts(
            torch.stack(selected_counts), physical_to_logical_map
        )
        raw_cost = _cost_from_samples(
            samples,
            active_expert_cost=active_expert_cost,
            additional_row_cost=additional_row_cost,
        )
        source_kind = "expert_distribution_records"
        sample_count = samples.shape[0]
    else:
        logical_count = loaded.get("logical_count")
        if not isinstance(logical_count, torch.Tensor):
            raise ValueError(f"profile {label!r} must contain records or logical_count")
        logical_count = _validate_count_tensor(
            logical_count, label=f"profile {label!r} logical_count"
        )
        if logical_count.ndim == 3:
            logical_count = _filter_sampled_logical_counts(
                logical_count,
                label=label,
                routes_per_layer=routes_per_layer,
            )
            raw_cost = _cost_from_samples(
                logical_count,
                active_expert_cost=active_expert_cost,
                additional_row_cost=additional_row_cost,
            )
            source_kind = "sampled_logical_count"
            sample_count = logical_count.shape[0]
        elif logical_count.ndim == 2:
            raw_cost = logical_count.to(torch.float64) * active_expert_cost
            source_kind = "aggregate_logical_count"
            sample_count = 1
        else:
            raise ValueError(
                f"profile {label!r} logical_count must have two or three dimensions"
            )
    return PromptCostProfile(
        label=label,
        path=path.resolve(),
        normalized_cost=_normalize_layer_cost(raw_cost),
        raw_cost=raw_cost,
        sample_count=sample_count,
        source_kind=source_kind,
    )


def robust_expert_order(
    prompt_cost: torch.Tensor,
    fallback_order: Sequence[int],
    *,
    tail_weight: float,
) -> list[int]:
    if prompt_cost.ndim != 2:
        raise ValueError("prompt cost must have shape [prompts,experts]")
    num_experts = prompt_cost.shape[1]
    if sorted(fallback_order) != list(range(num_experts)):
        raise ValueError("fallback order must be an expert permutation")
    if prompt_cost.numel() and float(prompt_cost.min()) < 0:
        raise ValueError("prompt cost must be non-negative")
    if float(prompt_cost.sum()) == 0:
        return list(fallback_order)
    score = prompt_cost.mean(dim=0) + tail_weight * prompt_cost.max(dim=0).values
    fallback_position = {
        expert_id: position for position, expert_id in enumerate(fallback_order)
    }
    return sorted(
        range(num_experts),
        key=lambda expert_id: (
            -float(score[expert_id]),
            fallback_position[expert_id],
            expert_id,
        ),
    )


def select_gpu_union(
    prompt_cost: torch.Tensor,
    fill_frequency: torch.Tensor,
    fallback_order: Sequence[int],
    *,
    gpu_expert_count: int,
    hot_prefix_count: int,
    tail_weight: float,
) -> list[int]:
    if not 0 <= hot_prefix_count <= gpu_expert_count:
        raise ValueError("hot-prefix count must fit inside the GPU expert union")
    primary_order = robust_expert_order(
        prompt_cost, fallback_order, tail_weight=tail_weight
    )
    fill_order = sorted(
        range(fill_frequency.numel()),
        key=lambda expert_id: (
            -int(fill_frequency[expert_id]),
            fallback_order.index(expert_id),
            expert_id,
        ),
    )
    selected = list(primary_order[:hot_prefix_count])
    selected_set = set(selected)
    for expert_id in (*fill_order, *fallback_order):
        if len(selected) == gpu_expert_count:
            break
        if expert_id not in selected_set:
            selected.append(expert_id)
            selected_set.add(expert_id)
    if len(selected) != gpu_expert_count:
        raise RuntimeError("failed to fill GPU expert union")
    return selected


def load_baseline_ownership(
    baseline_plan_path: Path,
    *,
    num_layers: int,
    num_experts: int,
) -> torch.Tensor:
    loaded = torch.load(baseline_plan_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("baseline plan must contain a dictionary")
    gpu_masks = loaded.get("gpu_experts_mask_by_rank")
    cpu_expert_ids_by_rank = loaded.get("cpu_expert_ids_by_rank")
    if not isinstance(gpu_masks, torch.Tensor) or not isinstance(
        cpu_expert_ids_by_rank, (list, tuple)
    ):
        raise ValueError(
            "baseline plan must contain GPU masks and CPU expert IDs by rank"
        )
    if tuple(gpu_masks.shape) != (2, num_layers, num_experts):
        raise ValueError(
            "baseline GPU mask shape mismatch: "
            f"expected {(2, num_layers, num_experts)}, got {tuple(gpu_masks.shape)}"
        )
    if len(cpu_expert_ids_by_rank) != 2:
        raise ValueError("baseline plan must contain exactly two rank shards")
    ownership = gpu_masks.to(dtype=torch.bool, device="cpu").clone()
    for rank, expert_ids_by_layer in enumerate(cpu_expert_ids_by_rank):
        if not isinstance(expert_ids_by_layer, torch.Tensor) or tuple(
            expert_ids_by_layer.shape[:1]
        ) != (num_layers,):
            raise ValueError("baseline CPU expert IDs must have one row per layer")
        ownership[rank].scatter_(
            1,
            expert_ids_by_layer.to(dtype=torch.int64, device="cpu"),
            True,
        )
    if not bool(torch.all(ownership.sum(dim=0) == 1)):
        raise ValueError(
            "baseline GPU+CPU ownership must be a disjoint exact cover per layer"
        )
    return ownership


def select_gpu_by_baseline_ownership(
    prompt_cost: torch.Tensor,
    fill_frequency: torch.Tensor,
    fallback_order: Sequence[int],
    ownership_by_rank: torch.Tensor,
    *,
    gpu_experts_per_rank: int,
    hot_prefix_count: int,
    tail_weight: float,
) -> list[list[int]]:
    gpu_expert_count = gpu_experts_per_rank * 2
    if not 0 <= hot_prefix_count <= gpu_expert_count:
        raise ValueError("hot-prefix count must fit inside the GPU expert union")
    if tuple(ownership_by_rank.shape) != (2, prompt_cost.shape[1]):
        raise ValueError("baseline ownership must have shape [2,experts]")
    if not bool(torch.all(ownership_by_rank.sum(dim=0) == 1)):
        raise ValueError("baseline ownership must assign every expert to one rank")
    primary_order = robust_expert_order(
        prompt_cost, fallback_order, tail_weight=tail_weight
    )
    fallback_position = {
        expert_id: position for position, expert_id in enumerate(fallback_order)
    }
    fill_order = sorted(
        range(fill_frequency.numel()),
        key=lambda expert_id: (
            -int(fill_frequency[expert_id]),
            fallback_position[expert_id],
            expert_id,
        ),
    )
    owner_by_expert = ownership_by_rank.to(torch.int64).argmax(dim=0).tolist()
    selected: list[list[int]] = [[], []]
    selected_set: set[int] = set()
    for expert_id in primary_order:
        if len(selected_set) == hot_prefix_count:
            break
        owner = owner_by_expert[expert_id]
        if len(selected[owner]) < gpu_experts_per_rank:
            selected[owner].append(expert_id)
            selected_set.add(expert_id)
    for rank in range(2):
        for expert_id in (*fill_order, *fallback_order):
            if len(selected[rank]) == gpu_experts_per_rank:
                break
            if owner_by_expert[expert_id] == rank and expert_id not in selected_set:
                selected[rank].append(expert_id)
                selected_set.add(expert_id)
        if len(selected[rank]) != gpu_experts_per_rank:
            raise RuntimeError(f"failed to fill baseline-owned GPU rank {rank}")
        selected[rank].sort()
    return selected


def assign_for_critical_tail(
    expert_ids: Sequence[int],
    prompt_cost: torch.Tensor,
    rank_counts: tuple[int, ...],
    *,
    tail_weight: float,
) -> list[list[int]]:
    if sum(rank_counts) != len(expert_ids):
        raise ValueError("rank capacities must cover the supplied experts")
    rank_loads = torch.zeros(
        (prompt_cost.shape[0], len(rank_counts)), dtype=torch.float64
    )
    assignments: list[list[int]] = [[] for _ in rank_counts]
    ordered_experts = sorted(
        expert_ids,
        key=lambda expert_id: (
            -float(prompt_cost[:, expert_id].max()),
            -float(prompt_cost[:, expert_id].mean()),
            expert_id,
        ),
    )
    for expert_id in ordered_experts:
        eligible_ranks = [
            rank
            for rank, capacity in enumerate(rank_counts)
            if len(assignments[rank]) < capacity
        ]
        selected_rank = min(
            eligible_ranks,
            key=lambda rank: _assignment_objective(
                rank_loads,
                prompt_cost[:, expert_id],
                rank,
                len(assignments[rank]),
                tail_weight=tail_weight,
            ),
        )
        assignments[selected_rank].append(expert_id)
        rank_loads[:, selected_rank] += prompt_cost[:, expert_id]
    for assignment in assignments:
        assignment.sort()
    return assignments


def _assignment_objective(
    rank_loads: torch.Tensor,
    expert_cost: torch.Tensor,
    rank: int,
    assigned_count: int,
    *,
    tail_weight: float,
) -> tuple[float, float, float, int, int]:
    candidate = rank_loads.clone()
    candidate[:, rank] += expert_cost
    prompt_critical = candidate.max(dim=1).values
    return (
        float(prompt_critical.mean() + tail_weight * prompt_critical.max()),
        float(candidate[:, rank].max()),
        float(candidate[:, rank].mean()),
        assigned_count,
        rank,
    )


def score_cpu_placement(
    prompt_cost: torch.Tensor,
    cpu_expert_ids_by_rank: Sequence[torch.Tensor],
) -> dict[str, object]:
    prompt_count, num_layers, _ = prompt_cost.shape
    rank_count = len(cpu_expert_ids_by_rank)
    layer_rank_cost = torch.zeros(
        (prompt_count, num_layers, rank_count), dtype=torch.float64
    )
    for rank, expert_ids_by_layer in enumerate(cpu_expert_ids_by_rank):
        for layer in range(num_layers):
            layer_rank_cost[:, layer, rank] = prompt_cost[
                :, layer, expert_ids_by_layer[layer]
            ].sum(dim=1)
    prompt_critical_tail = layer_rank_cost.max(dim=2).values.sum(dim=1)
    prompt_total_cpu = layer_rank_cost.sum(dim=(1, 2))
    layer_rank_skew = (
        layer_rank_cost.max(dim=2).values - layer_rank_cost.min(dim=2).values
    )
    return {
        "prompt_critical_tail": prompt_critical_tail.tolist(),
        "mean_critical_tail": float(prompt_critical_tail.mean()),
        "worst_critical_tail": float(prompt_critical_tail.max()),
        "prompt_total_cpu_cost": prompt_total_cpu.tolist(),
        "mean_layer_rank_skew": float(layer_rank_skew.mean()),
        "worst_layer_rank_skew": float(layer_rank_skew.max()),
    }


def build_placement_variant(
    prompt_cost: torch.Tensor,
    fill_frequency: torch.Tensor,
    fallback_ordering: Sequence[Sequence[int]],
    *,
    gpu_experts_per_rank: int,
    hot_prefix_count: int,
    tail_weight: float,
    ownership_by_rank: torch.Tensor | None = None,
) -> PlacementVariant:
    if prompt_cost.ndim != 3:
        raise ValueError("prompt cost must have shape [prompts,layers,experts]")
    _, num_layers, num_experts = prompt_cost.shape
    rank_count = 2
    gpu_expert_count = gpu_experts_per_rank * rank_count
    remaining_experts = num_experts - gpu_expert_count
    if remaining_experts < 0 or remaining_experts % rank_count:
        raise ValueError("GPU union must leave an equal integer CPU shard per rank")
    cpu_experts_per_rank = remaining_experts // rank_count
    gpu_masks = torch.zeros((rank_count, num_layers, num_experts), dtype=torch.bool)
    cpu_assignments_by_rank: list[list[list[int]]] = [[], []]
    gpu_assignments_by_rank: list[list[list[int]]] = [[], []]
    for layer in range(num_layers):
        if ownership_by_rank is None:
            gpu_union = select_gpu_union(
                prompt_cost[:, layer],
                fill_frequency[layer],
                fallback_ordering[layer],
                gpu_expert_count=gpu_expert_count,
                hot_prefix_count=hot_prefix_count,
                tail_weight=tail_weight,
            )
            gpu_set = set(gpu_union)
            cpu_union = [
                expert_id
                for expert_id in range(num_experts)
                if expert_id not in gpu_set
            ]
            gpu_assignments = assign_for_critical_tail(
                gpu_union,
                prompt_cost[:, layer],
                (gpu_experts_per_rank, gpu_experts_per_rank),
                tail_weight=tail_weight,
            )
            cpu_assignments = assign_for_critical_tail(
                cpu_union,
                prompt_cost[:, layer],
                (cpu_experts_per_rank, cpu_experts_per_rank),
                tail_weight=tail_weight,
            )
        else:
            gpu_assignments = select_gpu_by_baseline_ownership(
                prompt_cost[:, layer],
                fill_frequency[layer],
                fallback_ordering[layer],
                ownership_by_rank[:, layer],
                gpu_experts_per_rank=gpu_experts_per_rank,
                hot_prefix_count=hot_prefix_count,
                tail_weight=tail_weight,
            )
            cpu_assignments = []
            for rank in range(rank_count):
                gpu_set = set(gpu_assignments[rank])
                owned_experts = torch.where(ownership_by_rank[rank, layer])[0].tolist()
                cpu_assignment = [
                    expert_id for expert_id in owned_experts if expert_id not in gpu_set
                ]
                if len(cpu_assignment) != cpu_experts_per_rank:
                    raise ValueError(
                        "baseline ownership does not leave the requested equal CPU shard"
                    )
                cpu_assignments.append(cpu_assignment)
        for rank in range(rank_count):
            gpu_masks[rank, layer, gpu_assignments[rank]] = True
            gpu_assignments_by_rank[rank].append(gpu_assignments[rank])
            cpu_assignments_by_rank[rank].append(cpu_assignments[rank])
    cpu_tensors = tuple(
        torch.tensor(assignments, dtype=torch.int64)
        for assignments in cpu_assignments_by_rank
    )
    gpu_tensors = tuple(
        torch.tensor(assignments, dtype=torch.int64)
        for assignments in gpu_assignments_by_rank
    )
    return PlacementVariant(
        gpu_masks_by_rank=gpu_masks,
        cpu_expert_ids_by_rank=cpu_tensors,
        gpu_expert_ids_by_rank=gpu_tensors,
        metrics=score_cpu_placement(prompt_cost, cpu_tensors),
    )


def cross_validate_variant(
    prompt_cost: torch.Tensor,
    fill_frequency: torch.Tensor,
    fallback_ordering: Sequence[Sequence[int]],
    *,
    gpu_experts_per_rank: int,
    hot_prefix_count: int,
    tail_weight: float,
    ownership_by_rank: torch.Tensor | None = None,
) -> list[dict[str, object]]:
    if prompt_cost.shape[0] < 2:
        return []
    receipts: list[dict[str, object]] = []
    for held_out in range(prompt_cost.shape[0]):
        training_indices = [
            index for index in range(prompt_cost.shape[0]) if index != held_out
        ]
        trained = build_placement_variant(
            prompt_cost[training_indices],
            fill_frequency,
            fallback_ordering,
            gpu_experts_per_rank=gpu_experts_per_rank,
            hot_prefix_count=hot_prefix_count,
            tail_weight=tail_weight,
            ownership_by_rank=ownership_by_rank,
        )
        held_out_cost = prompt_cost[held_out : held_out + 1]
        held_out_metrics = score_cpu_placement(
            held_out_cost, trained.cpu_expert_ids_by_rank
        )
        oracle = build_placement_variant(
            held_out_cost,
            fill_frequency,
            fallback_ordering,
            gpu_experts_per_rank=gpu_experts_per_rank,
            hot_prefix_count=hot_prefix_count,
            tail_weight=tail_weight,
            ownership_by_rank=ownership_by_rank,
        )
        held_out_tail = float(held_out_metrics["mean_critical_tail"])
        oracle_tail = float(oracle.metrics["mean_critical_tail"])
        receipts.append(
            {
                "held_out_index": held_out,
                "critical_tail": held_out_tail,
                "oracle_critical_tail": oracle_tail,
                "oracle_regret_ratio": (
                    held_out_tail / oracle_tail if oracle_tail > 0 else 1.0
                ),
                "mean_layer_rank_skew": held_out_metrics["mean_layer_rank_skew"],
                "worst_layer_rank_skew": held_out_metrics["worst_layer_rank_skew"],
            }
        )
    return receipts


def _plan_semantics_sha256(
    gpu_masks: torch.Tensor, cpu_expert_ids_by_rank: Sequence[torch.Tensor]
) -> str:
    serialized = json.dumps(
        {
            "gpu_experts_mask_by_rank": gpu_masks.to(torch.uint8).tolist(),
            "cpu_expert_ids_by_rank": [
                expert_ids.tolist() for expert_ids in cpu_expert_ids_by_rank
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_variant_plan(
    path: Path,
    variant: PlacementVariant,
    profiles: Sequence[PromptCostProfile],
    *,
    ordering_path: Path,
    fill_profile_path: Path,
    gpu_experts_per_rank: int,
    hot_prefix_count: int,
    tail_weight: float,
    active_expert_cost: float,
    additional_row_cost: float,
    baseline_plan_path: Path | None,
) -> None:
    num_experts = variant.gpu_masks_by_rank.shape[-1]
    gpu_union_expert_count = gpu_experts_per_rank * 2
    cpu_experts_per_rank = (num_experts - gpu_union_expert_count) // 2
    aggregate_cost = torch.stack(
        [profile.normalized_cost for profile in profiles]
    ).mean(dim=0)
    total_cost = aggregate_cost.sum().clamp_min(torch.finfo(torch.float64).tiny)
    gpu_fractions = torch.tensor(
        [
            aggregate_cost[variant.gpu_masks_by_rank[rank]].sum() / total_cost
            for rank in range(2)
        ],
        dtype=torch.float64,
    )
    cpu_fractions = torch.tensor(
        [
            aggregate_cost.gather(1, variant.cpu_expert_ids_by_rank[rank]).sum()
            / total_cost
            for rank in range(2)
        ],
        dtype=torch.float64,
    )
    plan: dict[str, object] = {
        "format": PLAN_FORMAT,
        "gpu_experts_mask_by_rank": variant.gpu_masks_by_rank,
        "cpu_expert_ids_by_rank": list(variant.cpu_expert_ids_by_rank),
        "gpu_rank_counts": torch.tensor(
            [gpu_experts_per_rank, gpu_experts_per_rank], dtype=torch.int64
        ),
        "cpu_rank_counts": torch.tensor(
            [cpu_experts_per_rank, cpu_experts_per_rank], dtype=torch.int64
        ),
        "global_num_experts": torch.tensor(num_experts, dtype=torch.int64),
        "gpu_union_expert_count": torch.tensor(
            gpu_union_expert_count, dtype=torch.int64
        ),
        "gpu_activation_fractions": gpu_fractions,
        "cpu_activation_fractions": cpu_fractions,
        "gpu_selection_strategy": "multi-prompt-critical-tail-prefix-fill",
        "gpu_profile_hot_prefix_experts_per_layer": torch.tensor(
            hot_prefix_count, dtype=torch.int64
        ),
        "source_profiles": [str(profile.path) for profile in profiles],
        "source_profile_sha256s": [sha256_file(profile.path) for profile in profiles],
        "source_ordering": str(ordering_path.resolve()),
        "source_ordering_sha256": sha256_file(ordering_path),
        "source_fill_profile": str(fill_profile_path.resolve()),
        "source_fill_profile_sha256": sha256_file(fill_profile_path),
        "critical_tail_weight": float(tail_weight),
        "active_expert_cost": float(active_expert_cost),
        "additional_row_cost": float(additional_row_cost),
        "rank_ownership_constraint": (
            "baseline-plan" if baseline_plan_path is not None else "rebalanced"
        ),
        "placement_semantics_sha256": _plan_semantics_sha256(
            variant.gpu_masks_by_rank, variant.cpu_expert_ids_by_rank
        ),
    }
    if baseline_plan_path is not None:
        plan["source_baseline_plan"] = str(baseline_plan_path.resolve())
        plan["source_baseline_plan_sha256"] = sha256_file(baseline_plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(plan, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="append",
        type=parse_named_path,
        required=True,
        help="Named prompt fold as LABEL=/path/to/profile; repeat for CV",
    )
    parser.add_argument("--fill-profile", type=Path, required=True)
    parser.add_argument("--ordering", type=Path, required=True)
    parser.add_argument(
        "--baseline-plan",
        type=Path,
        help=(
            "Keep each expert inside this plan's immutable rank-local GPU+CPU "
            "ownership union; required for runtime hotspot promotion candidates"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-experts-per-rank",
        type=parse_positive_int_list,
        default=(13, 14),
    )
    parser.add_argument(
        "--hot-prefix-counts",
        type=parse_prefix_list,
        default=(0, 13, 26, None),
        help="Union counts from multi-prompt cost before fill; 'full' is per-g full",
    )
    parser.add_argument(
        "--routes-per-layer",
        type=int,
        default=36,
        help=(
            "Keep raw records and sampled logical-count rows whose every layer "
            "has this route total; use -1 to retain every sample"
        ),
    )
    parser.add_argument("--active-expert-cost", type=float, default=1.0)
    parser.add_argument("--additional-row-cost", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.active_expert_cost <= 0:
        parser.error("--active-expert-cost must be positive")
    if arguments.additional_row_cost < 0:
        parser.error("--additional-row-cost must be non-negative")
    if arguments.tail_weight < 0:
        parser.error("--tail-weight must be non-negative")

    profile_specs: list[tuple[str, Path]] = arguments.profile
    labels = [label for label, _ in profile_specs]
    if len(set(labels)) != len(labels):
        parser.error("profile labels must be unique")
    routes_per_layer = (
        None if arguments.routes_per_layer < 0 else arguments.routes_per_layer
    )
    profiles = [
        load_prompt_cost_profile(
            label,
            path,
            routes_per_layer=routes_per_layer,
            active_expert_cost=arguments.active_expert_cost,
            additional_row_cost=arguments.additional_row_cost,
        )
        for label, path in profile_specs
    ]
    profile_shapes = {tuple(profile.normalized_cost.shape) for profile in profiles}
    if len(profile_shapes) != 1:
        raise ValueError(f"prompt profile shapes must match, got {profile_shapes}")
    num_layers, num_experts = profiles[0].normalized_cost.shape
    prompt_cost = torch.stack([profile.normalized_cost for profile in profiles])
    fill_frequency = load_profile_frequency(
        arguments.fill_profile, label="fill profile"
    )
    if tuple(fill_frequency.shape) != (num_layers, num_experts):
        raise ValueError(
            "fill profile shape mismatch: "
            f"expected {(num_layers, num_experts)}, got {tuple(fill_frequency.shape)}"
        )
    fallback_ordering = load_hottest_first_ordering(
        arguments.ordering,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    ownership_by_rank = (
        load_baseline_ownership(
            arguments.baseline_plan,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        if arguments.baseline_plan is not None
        else None
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_receipt: dict[str, object] = {
        "format": SWEEP_RECEIPT_FORMAT,
        "profiles": [
            {
                "label": profile.label,
                "path": str(profile.path),
                "sha256": sha256_file(profile.path),
                "sample_count": profile.sample_count,
                "source_kind": profile.source_kind,
            }
            for profile in profiles
        ],
        "routes_per_layer": routes_per_layer,
        "active_expert_cost": arguments.active_expert_cost,
        "additional_row_cost": arguments.additional_row_cost,
        "tail_weight": arguments.tail_weight,
        "rank_ownership_constraint": (
            "baseline-plan" if arguments.baseline_plan is not None else "rebalanced"
        ),
        "baseline_plan": (
            None
            if arguments.baseline_plan is None
            else {
                "path": str(arguments.baseline_plan.resolve()),
                "sha256": sha256_file(arguments.baseline_plan),
            }
        ),
        "variants": [],
    }
    variants_receipt = sweep_receipt["variants"]
    assert isinstance(variants_receipt, list)
    seen_variants: set[tuple[int, int]] = set()
    for gpu_experts_per_rank in arguments.gpu_experts_per_rank:
        gpu_union_expert_count = gpu_experts_per_rank * 2
        for requested_prefix in arguments.hot_prefix_counts:
            hot_prefix_count = (
                gpu_union_expert_count if requested_prefix is None else requested_prefix
            )
            if hot_prefix_count > gpu_union_expert_count:
                continue
            variant_key = (gpu_experts_per_rank, hot_prefix_count)
            if variant_key in seen_variants:
                continue
            seen_variants.add(variant_key)
            variant = build_placement_variant(
                prompt_cost,
                fill_frequency,
                fallback_ordering,
                gpu_experts_per_rank=gpu_experts_per_rank,
                hot_prefix_count=hot_prefix_count,
                tail_weight=arguments.tail_weight,
                ownership_by_rank=ownership_by_rank,
            )
            variant_name = f"g{gpu_experts_per_rank}-p{hot_prefix_count}"
            plan_path = arguments.output_dir / f"{variant_name}.pt"
            _write_variant_plan(
                plan_path,
                variant,
                profiles,
                ordering_path=arguments.ordering,
                fill_profile_path=arguments.fill_profile,
                gpu_experts_per_rank=gpu_experts_per_rank,
                hot_prefix_count=hot_prefix_count,
                tail_weight=arguments.tail_weight,
                active_expert_cost=arguments.active_expert_cost,
                additional_row_cost=arguments.additional_row_cost,
                baseline_plan_path=arguments.baseline_plan,
            )
            cross_validation = cross_validate_variant(
                prompt_cost,
                fill_frequency,
                fallback_ordering,
                gpu_experts_per_rank=gpu_experts_per_rank,
                hot_prefix_count=hot_prefix_count,
                tail_weight=arguments.tail_weight,
                ownership_by_rank=ownership_by_rank,
            )
            cross_validation_summary = None
            if cross_validation:
                held_out_tails = [
                    float(fold["critical_tail"]) for fold in cross_validation
                ]
                regret_ratios = [
                    float(fold["oracle_regret_ratio"]) for fold in cross_validation
                ]
                cross_validation_summary = {
                    "mean_critical_tail": sum(held_out_tails) / len(held_out_tails),
                    "worst_critical_tail": max(held_out_tails),
                    "mean_oracle_regret_ratio": sum(regret_ratios) / len(regret_ratios),
                    "worst_oracle_regret_ratio": max(regret_ratios),
                }
            variants_receipt.append(
                {
                    "name": variant_name,
                    "plan": str(plan_path.resolve()),
                    "plan_sha256": sha256_file(plan_path),
                    "gpu_experts_per_rank": gpu_experts_per_rank,
                    "hot_prefix_count": hot_prefix_count,
                    "training_metrics": variant.metrics,
                    "cross_validation_summary": cross_validation_summary,
                    "cross_validation": [
                        {
                            **fold,
                            "held_out_label": profiles[
                                int(fold["held_out_index"])
                            ].label,
                        }
                        for fold in cross_validation
                    ],
                }
            )
    sweep_receipt["cross_validated_ranking"] = [
        variant["name"]
        for variant in sorted(
            variants_receipt,
            key=lambda variant: (
                (
                    variant["cross_validation_summary"]["worst_critical_tail"]
                    if isinstance(variant["cross_validation_summary"], dict)
                    else variant["training_metrics"]["worst_critical_tail"]
                ),
                (
                    variant["cross_validation_summary"]["mean_critical_tail"]
                    if isinstance(variant["cross_validation_summary"], dict)
                    else variant["training_metrics"]["mean_critical_tail"]
                ),
                variant["name"],
            ),
        )
    ]
    receipt_path = arguments.output_dir / "sweep-receipt.json"
    receipt_path.write_text(
        json.dumps(sweep_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
