#!/usr/bin/env python3
"""Build a byte-budgeted, variable-width DSV4 hybrid expert plan.

Uniformly adding one GPU expert to every target and draft layer spends roughly
586 MiB per rank on the current checkpoint.  This planner instead promotes
rank-owned CPU experts only in layers where the multi-prompt slow-rank tail is
predicted to fall.  The default permits different per-layer widths on the two
EP ranks while enforcing the same total byte budget per rank.  A conservative
rank-symmetric mode is available for the first runtime qualification.

The v2 plan stores CPU IDs in a padded tensor plus explicit per-layer counts.
That representation is ``torch.load(weights_only=True)`` compatible and avoids
using object/ragged tensors in a launch-time artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

import torch

try:
    from scripts.build_dsv4_critical_tail_plan_sweep import (
        PromptCostProfile,
        load_prompt_cost_profile,
        parse_named_path,
    )
    from scripts.build_dsv4_kt_hybrid_shard_plan import sha256_file
except ModuleNotFoundError:
    from build_dsv4_critical_tail_plan_sweep import (
        PromptCostProfile,
        load_prompt_cost_profile,
        parse_named_path,
    )
    from build_dsv4_kt_hybrid_shard_plan import sha256_file


VARIABLE_PLAN_FORMAT = "sglang_kt_hybrid_expert_shard_v2_variable"
VARIABLE_PLAN_RECEIPT_FORMAT = "dsv4_variable_residency_plan_v1"
DEFAULT_EXPERT_WEIGHT_BYTES = 13_369_344


@final
@dataclass(frozen=True)
class BaselinePlacement:
    gpu_masks_by_rank: torch.Tensor
    cpu_expert_ids_by_rank: tuple[torch.Tensor, torch.Tensor]


@final
@dataclass(frozen=True)
class Promotion:
    step: int
    rank: int
    layer: int
    expert_id: int
    objective_before: float
    objective_after: float
    critical_tail_reduction_by_profile: tuple[float, ...]


@final
@dataclass(frozen=True)
class VariablePlacement:
    gpu_masks_by_rank: torch.Tensor
    cpu_expert_ids_padded_by_rank: torch.Tensor
    cpu_rank_counts_by_layer: torch.Tensor
    gpu_rank_counts_by_layer: torch.Tensor
    promotions: tuple[Promotion, ...]
    critical_tail_before: torch.Tensor
    critical_tail_after: torch.Tensor


def _integer_tensor(value: object, *, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device="cpu")
    else:
        tensor = torch.as_tensor(value, device="cpu")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{label} must use an integer dtype")
    return tensor.to(dtype=torch.int64).contiguous()


def load_baseline_placement(path: Path) -> BaselinePlacement:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("baseline plan must contain a dictionary")
    raw_gpu_masks = loaded.get("gpu_experts_mask_by_rank")
    raw_cpu_shards = loaded.get("cpu_expert_ids_by_rank")
    if not isinstance(raw_gpu_masks, torch.Tensor) or not isinstance(
        raw_cpu_shards, (list, tuple)
    ):
        raise TypeError(
            "baseline plan must contain gpu_experts_mask_by_rank and "
            "cpu_expert_ids_by_rank"
        )
    gpu_masks = raw_gpu_masks.to(device="cpu", dtype=torch.bool).contiguous()
    if gpu_masks.ndim != 3 or gpu_masks.shape[0] != 2:
        raise ValueError("baseline GPU masks must have shape [2,layers,experts]")
    if len(raw_cpu_shards) != 2:
        raise ValueError("baseline plan must contain exactly two CPU rank shards")
    cpu_shards = tuple(
        _integer_tensor(shard, label=f"baseline CPU rank {rank}")
        for rank, shard in enumerate(raw_cpu_shards)
    )
    if any(shard.ndim != 2 for shard in cpu_shards):
        raise ValueError("baseline CPU rank shards must be rectangular tensors")
    num_layers = gpu_masks.shape[1]
    num_experts = gpu_masks.shape[2]
    if any(shard.shape[0] != num_layers for shard in cpu_shards):
        raise ValueError("baseline CPU shards must have one row per layer")
    expected = torch.arange(num_experts, dtype=torch.int64)
    gpu_counts = gpu_masks.sum(dim=2)
    if not bool(torch.all(gpu_counts[0] == gpu_counts[1])):
        raise ValueError(
            "variable-width planning requires equal GPU widths across EP ranks"
        )
    for layer in range(num_layers):
        assigned = torch.cat(
            [
                cpu_shards[0][layer],
                cpu_shards[1][layer],
                torch.where(gpu_masks[0, layer])[0],
                torch.where(gpu_masks[1, layer])[0],
            ]
        )
        if assigned.numel() != num_experts or not torch.equal(
            torch.sort(assigned).values, expected
        ):
            raise ValueError(
                "baseline GPU and CPU shards must form an exact disjoint cover "
                f"at layer {layer}"
            )
    return BaselinePlacement(
        gpu_masks_by_rank=gpu_masks,
        cpu_expert_ids_by_rank=cast(tuple[torch.Tensor, torch.Tensor], cpu_shards),
    )


def _objective(critical_tail: torch.Tensor, *, tail_weight: float) -> float:
    return float(critical_tail.mean() + tail_weight * critical_tail.max())


def _initial_rank_costs(
    prompt_cost: torch.Tensor,
    cpu_expert_ids_by_rank: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    profile_count, num_layers, _ = prompt_cost.shape
    rank_costs = torch.zeros((profile_count, num_layers, 2), dtype=torch.float64)
    for rank, cpu_ids_by_layer in enumerate(cpu_expert_ids_by_rank):
        for layer in range(num_layers):
            rank_costs[:, layer, rank] = prompt_cost[
                :, layer, cpu_ids_by_layer[layer]
            ].sum(dim=1)
    return rank_costs


def _best_layer_pair(
    *,
    prompt_cost: torch.Tensor,
    layer: int,
    cpu_ids_by_rank: tuple[list[list[int]], list[list[int]]],
    rank_costs: torch.Tensor,
    current_critical_tail: torch.Tensor,
    tail_weight: float,
) -> tuple[float, int, int, torch.Tensor] | None:
    rank0_ids = cpu_ids_by_rank[0][layer]
    rank1_ids = cpu_ids_by_rank[1][layer]
    if not rank0_ids or not rank1_ids:
        return None
    rank0_expert_cost = prompt_cost[:, layer, rank0_ids]
    rank1_expert_cost = prompt_cost[:, layer, rank1_ids]
    current_layer_tail = rank_costs[:, layer].max(dim=1).values
    # [profiles, rank0 candidates, rank1 candidates]
    candidate_layer_tail = torch.maximum(
        rank_costs[:, layer, 0, None, None] - rank0_expert_cost[:, :, None],
        rank_costs[:, layer, 1, None, None] - rank1_expert_cost[:, None, :],
    )
    candidate_critical_tail = (
        current_critical_tail[:, None, None]
        - current_layer_tail[:, None, None]
        + candidate_layer_tail
    )
    candidate_objective = candidate_critical_tail.mean(dim=0) + tail_weight * (
        candidate_critical_tail.max(dim=0).values
    )
    flat_index = int(torch.argmin(candidate_objective).item())
    rank1_count = len(rank1_ids)
    rank0_index, rank1_index = divmod(flat_index, rank1_count)
    return (
        float(candidate_objective[rank0_index, rank1_index]),
        rank0_ids[rank0_index],
        rank1_ids[rank1_index],
        candidate_critical_tail[:, rank0_index, rank1_index].clone(),
    )


def _best_rank_layer_expert(
    *,
    prompt_cost: torch.Tensor,
    rank: int,
    layer: int,
    cpu_ids_by_rank: tuple[list[list[int]], list[list[int]]],
    rank_costs: torch.Tensor,
    current_critical_tail: torch.Tensor,
    tail_weight: float,
) -> tuple[float, float, int, torch.Tensor] | None:
    expert_ids = cpu_ids_by_rank[rank][layer]
    if not expert_ids:
        return None
    other_rank = 1 - rank
    current_layer_tail = rank_costs[:, layer].max(dim=1).values
    candidate_layer_tail = torch.maximum(
        rank_costs[:, layer, rank, None] - prompt_cost[:, layer, expert_ids],
        rank_costs[:, layer, other_rank, None],
    )
    candidate_critical_tail = (
        current_critical_tail[:, None]
        - current_layer_tail[:, None]
        + candidate_layer_tail
    )
    candidate_objective = candidate_critical_tail.mean(dim=0) + tail_weight * (
        candidate_critical_tail.max(dim=0).values
    )
    minimum_objective = candidate_objective.min()
    objective_ties = torch.isclose(
        candidate_objective,
        minimum_objective,
        rtol=0.0,
        atol=torch.finfo(candidate_objective.dtype).eps * 8,
    )
    local_reduction = prompt_cost[:, layer, expert_ids].mean(dim=0)
    tie_score = torch.where(
        objective_ties,
        local_reduction,
        torch.full_like(local_reduction, -1.0),
    )
    candidate_index = int(torch.argmax(tie_score).item())
    return (
        float(candidate_objective[candidate_index]),
        float(local_reduction[candidate_index]),
        expert_ids[candidate_index],
        candidate_critical_tail[:, candidate_index].clone(),
    )


def _pad_cpu_ids(
    cpu_ids_by_rank: tuple[list[list[int]], list[list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_count = len(cpu_ids_by_rank)
    num_layers = len(cpu_ids_by_rank[0])
    counts = torch.tensor(
        [
            [len(cpu_ids_by_rank[rank][layer]) for layer in range(num_layers)]
            for rank in range(rank_count)
        ],
        dtype=torch.int64,
    )
    max_count = int(counts.max().item()) if counts.numel() else 0
    padded = torch.full((rank_count, num_layers, max_count), -1, dtype=torch.int64)
    for rank in range(rank_count):
        for layer in range(num_layers):
            layer_ids = cpu_ids_by_rank[rank][layer]
            if layer_ids:
                padded[rank, layer, : len(layer_ids)] = torch.tensor(
                    layer_ids, dtype=torch.int64
                )
    return padded, counts


def build_variable_placement(
    prompt_cost: torch.Tensor,
    baseline: BaselinePlacement,
    *,
    extra_bytes_per_rank: int,
    expert_weight_bytes: int = DEFAULT_EXPERT_WEIGHT_BYTES,
    max_extra_experts_per_layer: int = 8,
    tail_weight: float = 1.0,
    minimum_objective_reduction: float = 0.0,
    rank_symmetric_widths: bool = False,
) -> VariablePlacement:
    """Promote rank-owned experts under an exact per-rank byte budget."""
    if prompt_cost.ndim != 3:
        raise ValueError("prompt cost must have shape [profiles,layers,experts]")
    if prompt_cost.dtype != torch.float64:
        prompt_cost = prompt_cost.to(dtype=torch.float64, device="cpu")
    if prompt_cost.numel() and float(prompt_cost.min()) < 0:
        raise ValueError("prompt cost must be non-negative")
    expected_shape = tuple(baseline.gpu_masks_by_rank.shape[1:])
    if tuple(prompt_cost.shape[1:]) != expected_shape:
        raise ValueError(
            f"prompt cost shape mismatch: expected {expected_shape}, "
            f"got {tuple(prompt_cost.shape[1:])}"
        )
    if extra_bytes_per_rank < 0:
        raise ValueError("extra byte budget must be non-negative")
    if expert_weight_bytes <= 0:
        raise ValueError("expert weight bytes must be positive")
    if max_extra_experts_per_layer < 0:
        raise ValueError("max extra experts per layer must be non-negative")
    if tail_weight < 0 or minimum_objective_reduction < 0:
        raise ValueError("objective weights and thresholds must be non-negative")

    promotion_budget_per_rank = extra_bytes_per_rank // expert_weight_bytes
    gpu_masks = baseline.gpu_masks_by_rank.clone()
    cpu_ids_by_rank: tuple[list[list[int]], list[list[int]]] = (
        [sorted(row.tolist()) for row in baseline.cpu_expert_ids_by_rank[0]],
        [sorted(row.tolist()) for row in baseline.cpu_expert_ids_by_rank[1]],
    )
    rank_costs = _initial_rank_costs(prompt_cost, baseline.cpu_expert_ids_by_rank)
    critical_tail_before = rank_costs.max(dim=2).values.sum(dim=1)
    current_critical_tail = critical_tail_before.clone()
    initial_gpu_widths = gpu_masks.sum(dim=2)
    extra_by_rank_layer = torch.zeros((2, gpu_masks.shape[1]), dtype=torch.int64)
    promotions_by_rank = [0, 0]
    promotions: list[Promotion] = []

    maximum_promotion_events = (
        promotion_budget_per_rank
        if rank_symmetric_widths
        else promotion_budget_per_rank * 2
    )
    for step in range(maximum_promotion_events):
        objective_before = _objective(current_critical_tail, tail_weight=tail_weight)
        if rank_symmetric_widths:
            best_pair: tuple[float, int, int, int, torch.Tensor] | None = None
            for layer in range(gpu_masks.shape[1]):
                if int(extra_by_rank_layer[0, layer]) >= max_extra_experts_per_layer:
                    continue
                layer_result = _best_layer_pair(
                    prompt_cost=prompt_cost,
                    layer=layer,
                    cpu_ids_by_rank=cpu_ids_by_rank,
                    rank_costs=rank_costs,
                    current_critical_tail=current_critical_tail,
                    tail_weight=tail_weight,
                )
                if layer_result is None:
                    continue
                candidate_objective, expert0, expert1, candidate_tail = layer_result
                candidate = (
                    candidate_objective,
                    layer,
                    expert0,
                    expert1,
                    candidate_tail,
                )
                if best_pair is None or candidate[:4] < best_pair[:4]:
                    best_pair = candidate
            if best_pair is None:
                break
            objective_after, layer, expert0, expert1, candidate_tail = best_pair
            if objective_before - objective_after <= minimum_objective_reduction:
                break
            reduction = current_critical_tail - candidate_tail
            for rank, expert_id in enumerate((expert0, expert1)):
                cpu_ids_by_rank[rank][layer].remove(expert_id)
                gpu_masks[rank, layer, expert_id] = True
                rank_costs[:, layer, rank] -= prompt_cost[:, layer, expert_id]
                promotions_by_rank[rank] += 1
                extra_by_rank_layer[rank, layer] += 1
                promotions.append(
                    Promotion(
                        step=step,
                        rank=rank,
                        layer=layer,
                        expert_id=expert_id,
                        objective_before=objective_before,
                        objective_after=objective_after,
                        critical_tail_reduction_by_profile=tuple(reduction.tolist()),
                    )
                )
            current_critical_tail = candidate_tail
            continue

        best_single: tuple[float, float, int, int, int, torch.Tensor] | None = None
        if promotions_by_rank[0] == promotions_by_rank[1]:
            allowed_ranks = range(2)
        else:
            allowed_ranks = (int(promotions_by_rank[0] > promotions_by_rank[1]),)
        for rank in allowed_ranks:
            if promotions_by_rank[rank] >= promotion_budget_per_rank:
                continue
            for layer in range(gpu_masks.shape[1]):
                if int(extra_by_rank_layer[rank, layer]) >= max_extra_experts_per_layer:
                    continue
                layer_result = _best_rank_layer_expert(
                    prompt_cost=prompt_cost,
                    rank=rank,
                    layer=layer,
                    cpu_ids_by_rank=cpu_ids_by_rank,
                    rank_costs=rank_costs,
                    current_critical_tail=current_critical_tail,
                    tail_weight=tail_weight,
                )
                if layer_result is None:
                    continue
                (
                    candidate_objective,
                    local_reduction,
                    expert_id,
                    candidate_tail,
                ) = layer_result
                candidate = (
                    candidate_objective,
                    -local_reduction,
                    rank,
                    layer,
                    expert_id,
                    candidate_tail,
                )
                if best_single is None or candidate[:5] < best_single[:5]:
                    best_single = candidate
        if best_single is None:
            break
        (
            objective_after,
            _negative_local_reduction,
            rank,
            layer,
            expert_id,
            candidate_tail,
        ) = best_single
        # The rank budgets advance in lockstep. A zero-gain first half can
        # expose a gain when the other rank is promoted, so the independent
        # mode spends the admitted budget in full. The threshold is reserved
        # for the paired/symmetric mode where one event covers both ranks.
        reduction = current_critical_tail - candidate_tail
        cpu_ids_by_rank[rank][layer].remove(expert_id)
        gpu_masks[rank, layer, expert_id] = True
        rank_costs[:, layer, rank] -= prompt_cost[:, layer, expert_id]
        promotions_by_rank[rank] += 1
        extra_by_rank_layer[rank, layer] += 1
        promotions.append(
            Promotion(
                step=step,
                rank=rank,
                layer=layer,
                expert_id=expert_id,
                objective_before=objective_before,
                objective_after=objective_after,
                critical_tail_reduction_by_profile=tuple(reduction.tolist()),
            )
        )
        current_critical_tail = candidate_tail

    padded_cpu_ids, cpu_counts = _pad_cpu_ids(cpu_ids_by_rank)
    gpu_counts = gpu_masks.sum(dim=2).to(dtype=torch.int64)
    if promotions_by_rank[0] != promotions_by_rank[1]:
        raise RuntimeError(
            "planner could not spend an equal expert-weight budget on both ranks"
        )
    if rank_symmetric_widths and not bool(torch.all(gpu_counts[0] == gpu_counts[1])):
        raise AssertionError("symmetric planner produced rank-asymmetric GPU widths")
    if not bool(torch.all(gpu_counts - initial_gpu_widths == extra_by_rank_layer)):
        raise AssertionError("planner promotion accounting drifted")
    return VariablePlacement(
        gpu_masks_by_rank=gpu_masks,
        cpu_expert_ids_padded_by_rank=padded_cpu_ids,
        cpu_rank_counts_by_layer=cpu_counts,
        gpu_rank_counts_by_layer=gpu_counts,
        promotions=tuple(promotions),
        critical_tail_before=critical_tail_before,
        critical_tail_after=current_critical_tail,
    )


def _semantics_sha256(placement: VariablePlacement) -> str:
    serialized = json.dumps(
        {
            "gpu_experts_mask_by_rank": placement.gpu_masks_by_rank.to(
                torch.uint8
            ).tolist(),
            "cpu_expert_ids_padded_by_rank": (
                placement.cpu_expert_ids_padded_by_rank.tolist()
            ),
            "cpu_rank_counts_by_layer": (placement.cpu_rank_counts_by_layer.tolist()),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_plan_payload(
    placement: VariablePlacement,
    profiles: Sequence[PromptCostProfile],
    *,
    baseline_path: Path,
    extra_bytes_per_rank: int,
    expert_weight_bytes: int,
    max_extra_experts_per_layer: int,
    tail_weight: float,
    rank_symmetric_widths: bool,
) -> dict[str, object]:
    gpu_counts = placement.gpu_rank_counts_by_layer
    return {
        "format": VARIABLE_PLAN_FORMAT,
        "gpu_experts_mask_by_rank": placement.gpu_masks_by_rank,
        "cpu_expert_ids_padded_by_rank": (placement.cpu_expert_ids_padded_by_rank),
        "cpu_rank_counts_by_layer": placement.cpu_rank_counts_by_layer,
        "gpu_rank_counts_by_layer": placement.gpu_rank_counts_by_layer,
        "global_num_experts": torch.tensor(
            placement.gpu_masks_by_rank.shape[2], dtype=torch.int64
        ),
        "max_gpu_experts_per_rank_per_layer": torch.tensor(
            int(gpu_counts.max().item()), dtype=torch.int64
        ),
        "min_gpu_experts_per_rank_per_layer": torch.tensor(
            int(gpu_counts.min().item()), dtype=torch.int64
        ),
        "expert_weight_bytes": torch.tensor(expert_weight_bytes, dtype=torch.int64),
        "extra_bytes_per_rank": torch.tensor(extra_bytes_per_rank, dtype=torch.int64),
        "selected_extra_bytes_per_rank": torch.tensor(
            (len(placement.promotions) // 2) * expert_weight_bytes,
            dtype=torch.int64,
        ),
        "max_extra_experts_per_layer": torch.tensor(
            max_extra_experts_per_layer, dtype=torch.int64
        ),
        "critical_tail_weight": float(tail_weight),
        "rank_symmetric_widths": rank_symmetric_widths,
        "source_baseline_plan": str(baseline_path.resolve()),
        "source_baseline_plan_sha256": sha256_file(baseline_path),
        "source_profiles": [str(profile.path) for profile in profiles],
        "source_profile_sha256s": [sha256_file(profile.path) for profile in profiles],
        "placement_semantics_sha256": _semantics_sha256(placement),
    }


def build_receipt(
    placement: VariablePlacement,
    profiles: Sequence[PromptCostProfile],
    *,
    output_path: Path,
    baseline_path: Path,
    extra_bytes_per_rank: int,
    expert_weight_bytes: int,
    rank_symmetric_widths: bool,
) -> dict[str, object]:
    before = placement.critical_tail_before
    after = placement.critical_tail_after
    reductions = before - after
    selected_slots_per_rank = len(placement.promotions) // 2
    baseline_masks = placement.gpu_masks_by_rank.clone()
    for promotion in placement.promotions:
        baseline_masks[promotion.rank, promotion.layer, promotion.expert_id] = False
    baseline_gpu_union = baseline_masks.any(dim=0)
    selected_gpu_union = placement.gpu_masks_by_rank.any(dim=0)
    route_coverage: list[tuple[float, float]] = []
    for profile in profiles:
        total_cost = float(profile.normalized_cost.sum())
        before_coverage = float(profile.normalized_cost[baseline_gpu_union].sum())
        after_coverage = float(profile.normalized_cost[selected_gpu_union].sum())
        route_coverage.append(
            (
                before_coverage / total_cost if total_cost > 0 else 0.0,
                after_coverage / total_cost if total_cost > 0 else 0.0,
            )
        )
    return {
        "format": VARIABLE_PLAN_RECEIPT_FORMAT,
        "plan": str(output_path.resolve()),
        "plan_sha256": sha256_file(output_path),
        "placement_semantics_sha256": _semantics_sha256(placement),
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        },
        "profiles": [
            {
                "label": profile.label,
                "path": str(profile.path),
                "sha256": sha256_file(profile.path),
                "sample_count": profile.sample_count,
                "source_kind": profile.source_kind,
                "critical_tail_before": float(before[index]),
                "critical_tail_after": float(after[index]),
                "predicted_reduction": float(reductions[index]),
                "predicted_reduction_ratio": (
                    float(reductions[index] / before[index])
                    if float(before[index]) > 0
                    else 0.0
                ),
                "gpu_route_cost_fraction_before": route_coverage[index][0],
                "gpu_route_cost_fraction_after": route_coverage[index][1],
            }
            for index, profile in enumerate(profiles)
        ],
        "budget": {
            "extra_bytes_per_rank": extra_bytes_per_rank,
            "expert_weight_bytes": expert_weight_bytes,
            "slot_budget_per_rank": extra_bytes_per_rank // expert_weight_bytes,
            "selected_slots_per_rank": selected_slots_per_rank,
            "selected_bytes_per_rank": selected_slots_per_rank * expert_weight_bytes,
            "unused_bytes_per_rank": extra_bytes_per_rank
            - selected_slots_per_rank * expert_weight_bytes,
        },
        "critical_tail_summary": {
            "mean_before": float(before.mean()),
            "mean_after": float(after.mean()),
            "mean_reduction_ratio": (
                float((before.mean() - after.mean()) / before.mean())
                if float(before.mean()) > 0
                else 0.0
            ),
            "worst_before": float(before.max()),
            "worst_after": float(after.max()),
            "worst_reduction_ratio": (
                float((before.max() - after.max()) / before.max())
                if float(before.max()) > 0
                else 0.0
            ),
            "mean_gpu_route_cost_fraction_before": sum(
                item[0] for item in route_coverage
            )
            / len(route_coverage),
            "mean_gpu_route_cost_fraction_after": sum(
                item[1] for item in route_coverage
            )
            / len(route_coverage),
        },
        "gpu_width": {
            "minimum": int(placement.gpu_rank_counts_by_layer.min().item()),
            "maximum": int(placement.gpu_rank_counts_by_layer.max().item()),
            "rank_symmetric": rank_symmetric_widths,
            "by_rank_and_layer": placement.gpu_rank_counts_by_layer.tolist(),
        },
        "promotions": [
            {
                "step": promotion.step,
                "rank": promotion.rank,
                "layer": promotion.layer,
                "expert_id": promotion.expert_id,
                "objective_before": promotion.objective_before,
                "objective_after": promotion.objective_after,
                "critical_tail_reduction_by_profile": list(
                    promotion.critical_tail_reduction_by_profile
                ),
            }
            for promotion in placement.promotions
        ],
        "runtime_requirements": [
            "hybrid v2 padded CPU shard loading with per-layer counts",
            "--kt-num-gpu-experts interpreted as a per-layer admission ceiling",
            (
                "rank-symmetric per-layer GPU widths"
                if rank_symmetric_widths
                else "rank-specific per-layer GPU widths with equal total rank bytes"
            ),
            "target plan SHA-256 bound by launcher and loader",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="append",
        type=parse_named_path,
        required=True,
        help="Named prompt fold as LABEL=/path/to/profile; repeat for robustness",
    )
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-mib-per-rank",
        type=float,
        required=True,
        help="Hard additional GPU-weight budget for each rank",
    )
    parser.add_argument(
        "--expert-weight-bytes", type=int, default=DEFAULT_EXPERT_WEIGHT_BYTES
    )
    parser.add_argument("--max-extra-experts-per-layer", type=int, default=8)
    parser.add_argument("--routes-per-layer", type=int, default=72)
    parser.add_argument("--active-expert-cost", type=float, default=1.0)
    parser.add_argument("--additional-row-cost", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--minimum-objective-reduction", type=float, default=0.0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for small planner tensors; one avoids threadpool overhead",
    )
    parser.add_argument(
        "--rank-symmetric-widths",
        action="store_true",
        help="promote one expert on both ranks in the same layer per slot step",
    )
    arguments = parser.parse_args()
    if (
        not math.isfinite(arguments.extra_mib_per_rank)
        or arguments.extra_mib_per_rank < 0
    ):
        parser.error("--extra-mib-per-rank must be finite and non-negative")
    if arguments.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(arguments.torch_threads)
    labels = [label for label, _path in arguments.profile]
    if len(labels) != len(set(labels)):
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
        for label, path in arguments.profile
    ]
    shapes = {tuple(profile.normalized_cost.shape) for profile in profiles}
    if len(shapes) != 1:
        raise ValueError(f"prompt profile shapes must match, got {shapes}")
    baseline = load_baseline_placement(arguments.baseline_plan)
    prompt_cost = torch.stack([profile.normalized_cost for profile in profiles])
    extra_bytes_per_rank = int(arguments.extra_mib_per_rank * 1024 * 1024)
    placement = build_variable_placement(
        prompt_cost,
        baseline,
        extra_bytes_per_rank=extra_bytes_per_rank,
        expert_weight_bytes=arguments.expert_weight_bytes,
        max_extra_experts_per_layer=arguments.max_extra_experts_per_layer,
        tail_weight=arguments.tail_weight,
        minimum_objective_reduction=arguments.minimum_objective_reduction,
        rank_symmetric_widths=arguments.rank_symmetric_widths,
    )
    payload = build_plan_payload(
        placement,
        profiles,
        baseline_path=arguments.baseline_plan,
        extra_bytes_per_rank=extra_bytes_per_rank,
        expert_weight_bytes=arguments.expert_weight_bytes,
        max_extra_experts_per_layer=arguments.max_extra_experts_per_layer,
        tail_weight=arguments.tail_weight,
        rank_symmetric_widths=arguments.rank_symmetric_widths,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    receipt = build_receipt(
        placement,
        profiles,
        output_path=arguments.output,
        baseline_path=arguments.baseline_plan,
        extra_bytes_per_rank=extra_bytes_per_rank,
        expert_weight_bytes=arguments.expert_weight_bytes,
        rank_symmetric_widths=arguments.rank_symmetric_widths,
    )
    receipt_path = arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
