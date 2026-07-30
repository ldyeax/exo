#!/usr/bin/env python3
"""Build disjoint GPU, opposite-NUMA, fwuff, and local MXFP4 tiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_activation_counts(profile_path: Path) -> torch.Tensor:
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    if not isinstance(profile, dict):
        raise ValueError("expert profile must be a dictionary")
    logical_count = profile.get("logical_count")
    if not isinstance(logical_count, torch.Tensor):
        raise ValueError("expert profile must contain logical_count")
    if logical_count.ndim == 3:
        logical_count = logical_count.sum(dim=0, dtype=torch.int64)
    elif logical_count.ndim == 2:
        logical_count = logical_count.to(dtype=torch.int64)
    else:
        raise ValueError(
            "logical_count must have shape [layers, experts] or "
            f"[samples, layers, experts], got {tuple(logical_count.shape)}"
        )
    return logical_count.cpu().contiguous()


def load_gpu_mask(mask_path: Path, expected_shape: torch.Size) -> torch.Tensor:
    plan = torch.load(mask_path, map_location="cpu", weights_only=True)
    if not isinstance(plan, dict):
        raise ValueError("GPU mask plan must be a dictionary")
    mask = plan.get("gpu_experts_mask")
    if not isinstance(mask, torch.Tensor):
        raise ValueError("GPU mask plan must contain gpu_experts_mask")
    if mask.shape != expected_shape:
        raise ValueError(
            "GPU mask shape does not match the expert profile: "
            f"expected={tuple(expected_shape)}, actual={tuple(mask.shape)}"
        )
    return mask.to(device="cpu", dtype=torch.bool).contiguous()


def make_gpu_mask(
    activation_counts: torch.Tensor,
    gpu_experts_per_layer: int,
) -> torch.Tensor:
    num_layers, num_experts = activation_counts.shape
    total_gpu_experts = gpu_experts_per_layer * num_layers
    if not 0 <= total_gpu_experts <= activation_counts.numel():
        raise ValueError("GPU expert count is outside the profile")
    mask = torch.zeros(activation_counts.numel(), dtype=torch.bool, device="cpu")
    if total_gpu_experts:
        # Match kt_kernel.generate_gpu_experts_masks exactly.
        top_indices = torch.topk(
            activation_counts.reshape(-1),
            k=total_gpu_experts,
            largest=True,
            sorted=False,
        ).indices
        mask[top_indices] = True
    return mask.reshape(num_layers, num_experts)


def make_pipeline_gpu_mask(
    activation_counts: torch.Tensor,
    *,
    gpu_experts_per_layer_by_stage: tuple[int, ...],
    stage_layer_partition: tuple[int, ...],
) -> torch.Tensor:
    if len(gpu_experts_per_layer_by_stage) != len(stage_layer_partition):
        raise ValueError(
            "GPU budgets and pipeline layer partition must have the same length"
        )
    if sum(stage_layer_partition) != activation_counts.shape[0]:
        raise ValueError("pipeline layer partition must cover every profile layer")
    if any(layer_count <= 0 for layer_count in stage_layer_partition):
        raise ValueError("pipeline layer counts must be positive")

    staged_mask = torch.zeros_like(activation_counts, dtype=torch.bool)
    layer_start = 0
    masks_by_budget = {}
    for gpu_budget, layer_count in zip(
        gpu_experts_per_layer_by_stage,
        stage_layer_partition,
        strict=True,
    ):
        if gpu_budget not in masks_by_budget:
            masks_by_budget[gpu_budget] = make_gpu_mask(
                activation_counts,
                gpu_budget,
            )
        layer_end = layer_start + layer_count
        staged_mask[layer_start:layer_end] = masks_by_budget[gpu_budget][
            layer_start:layer_end
        ]
        layer_start = layer_end
    return staged_mask


def make_pipeline_gpu_mask_by_slots(
    activation_counts: torch.Tensor,
    *,
    gpu_expert_slots_by_stage: tuple[int, ...],
    stage_layer_partition: tuple[int, ...],
) -> torch.Tensor:
    """Select an exact number of hottest slots inside each pipeline stage."""
    if len(gpu_expert_slots_by_stage) != len(stage_layer_partition):
        raise ValueError(
            "GPU slot counts and pipeline layer partition must have the same length"
        )
    if sum(stage_layer_partition) != activation_counts.shape[0]:
        raise ValueError("pipeline layer partition must cover every profile layer")
    mask = torch.zeros_like(activation_counts, dtype=torch.bool)
    layer_start = 0
    for slot_count, layer_count in zip(
        gpu_expert_slots_by_stage,
        stage_layer_partition,
        strict=True,
    ):
        layer_end = layer_start + layer_count
        stage_counts = activation_counts[layer_start:layer_end]
        if not 0 <= slot_count <= stage_counts.numel():
            raise ValueError(
                f"GPU slot count {slot_count} is outside stage capacity "
                f"{stage_counts.numel()}"
            )
        if slot_count:
            stage_mask = mask[layer_start:layer_end].reshape(-1)
            stage_mask[
                torch.topk(
                    stage_counts.reshape(-1),
                    k=slot_count,
                    largest=True,
                    sorted=False,
                ).indices
            ] = True
        layer_start = layer_end
    return mask


def choose_balanced_sidecar_ids(
    frequency: torch.Tensor,
    available_ids: torch.Tensor,
    sidecar_count: int,
) -> torch.Tensor:
    if sidecar_count > available_ids.numel():
        raise ValueError("opposite-NUMA sidecar count exceeds available experts")
    ordered = sorted(
        (int(expert_id) for expert_id in available_ids),
        key=lambda expert_id: (-int(frequency[expert_id]), expert_id),
    )
    target_load = sum(int(frequency[expert_id]) for expert_id in ordered) / 2
    selected = []
    selected_set = set()
    selected_load = 0
    for expert_id in ordered:
        if len(selected) == sidecar_count:
            break
        candidate_load = selected_load + int(frequency[expert_id])
        if candidate_load <= target_load or not selected:
            selected.append(expert_id)
            selected_set.add(expert_id)
            selected_load = candidate_load
    if len(selected) < sidecar_count:
        # Fill fixed storage slots with the coldest remaining experts. This
        # minimally perturbs the frequency-balanced split.
        for expert_id in reversed(ordered):
            if expert_id not in selected_set:
                selected.append(expert_id)
                selected_set.add(expert_id)
                if len(selected) == sidecar_count:
                    break
    return torch.tensor(sorted(selected), dtype=torch.int64)


def choose_ids_toward_load(
    frequency: torch.Tensor,
    available_ids: torch.Tensor,
    sidecar_count: int,
    target_load: float,
) -> torch.Tensor:
    """Choose a fixed slot count while approaching a requested route load."""
    if sidecar_count > available_ids.numel():
        raise ValueError("sidecar count exceeds available experts")
    if sidecar_count == 0:
        return torch.empty(0, dtype=torch.int64)
    ordered = sorted(
        (int(expert_id) for expert_id in available_ids),
        key=lambda expert_id: (-int(frequency[expert_id]), expert_id),
    )
    selected = []
    selected_set = set()
    selected_load = 0
    for expert_id in ordered:
        if len(selected) == sidecar_count:
            break
        candidate_load = selected_load + int(frequency[expert_id])
        if candidate_load <= target_load or not selected:
            selected.append(expert_id)
            selected_set.add(expert_id)
            selected_load = candidate_load
    if len(selected) < sidecar_count:
        # Fill storage slots with cold experts after reaching the load target.
        for expert_id in reversed(ordered):
            if expert_id not in selected_set:
                selected.append(expert_id)
                selected_set.add(expert_id)
                if len(selected) == sidecar_count:
                    break
    return torch.tensor(sorted(selected), dtype=torch.int64)


def make_plan(
    *,
    remote_expert_ids: torch.Tensor,
    activation_counts: torch.Tensor,
    profile_path: Path,
    tier: str,
) -> dict:
    selected_counts = activation_counts[: remote_expert_ids.shape[0]].gather(
        1, remote_expert_ids
    )
    return {
        "remote_expert_ids": remote_expert_ids,
        "activation_counts": selected_counts,
        "profile_path": str(profile_path.resolve()),
        "tier": tier,
        "hidden_size": 7168,
        "intermediate_size": 3072,
        "topk": 6,
        "swiglu_limit": 10.0,
        "num_experts": activation_counts.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--opposite-numa-output", required=True, type=Path)
    parser.add_argument("--fwuff-output", required=True, type=Path)
    parser.add_argument(
        "--gpu-mask-output",
        type=Path,
        help="Optional explicit logical GPU expert mask plan output.",
    )
    parser.add_argument(
        "--gpu-mask-input",
        type=Path,
        help="Reuse an exact logical GPU expert mask instead of regenerating it.",
    )
    parser.add_argument("--serve-layers", type=int, default=48)
    parser.add_argument("--gpu-experts-per-layer", type=int, default=5)
    parser.add_argument(
        "--gpu-experts-per-layer-by-stage",
        help="Comma-separated rank-local GPU budgets, e.g. 7,4,12.",
    )
    parser.add_argument(
        "--gpu-expert-slots-by-stage",
        help="Exact comma-separated GPU slot counts per pipeline stage.",
    )
    parser.add_argument(
        "--stage-layer-partition",
        default="23,25,13",
        help="Comma-separated pipeline layer counts.",
    )
    parser.add_argument("--opposite-numa-experts-per-layer", type=int, default=32)
    parser.add_argument("--fwuff-experts-per-layer", type=int, default=8)
    parser.add_argument(
        "--fwuff-cold-experts-per-layer",
        type=int,
        help=(
            "Cold experts retained on fwuff per layer. Defaults to "
            "--fwuff-experts-per-layer."
        ),
    )
    parser.add_argument(
        "--fwuff-balanced-experts-per-layer",
        type=int,
        default=0,
        help="Additional profile-balanced fwuff experts per layer.",
    )
    parser.add_argument(
        "--fwuff-target-share",
        type=float,
        default=1 / 3,
        help="Target share of non-GPU route load for the complete fwuff tier.",
    )
    args = parser.parse_args()

    counts = load_activation_counts(args.profile)
    num_layers, num_experts = counts.shape
    stage_partition = tuple(
        int(value) for value in args.stage_layer_partition.split(",")
    )
    if not 0 < args.serve_layers <= num_layers:
        raise ValueError("--serve-layers is outside the profile")
    if args.gpu_experts_per_layer_by_stage and args.gpu_expert_slots_by_stage:
        raise ValueError("Use either GPU budgets or exact GPU slot counts, not both")
    if args.gpu_mask_input and (
        args.gpu_experts_per_layer_by_stage or args.gpu_expert_slots_by_stage
    ):
        raise ValueError("An input GPU mask cannot be combined with GPU stage budgets")
    gpu_slots: tuple[int, ...] = ()
    if args.gpu_mask_input:
        gpu_mask = load_gpu_mask(args.gpu_mask_input, counts.shape)
        gpu_budgets: tuple[int, ...] = ()
    elif args.gpu_expert_slots_by_stage:
        gpu_slots = tuple(
            int(value) for value in args.gpu_expert_slots_by_stage.split(",")
        )
        gpu_mask = make_pipeline_gpu_mask_by_slots(
            counts,
            gpu_expert_slots_by_stage=gpu_slots,
            stage_layer_partition=stage_partition,
        )
        gpu_budgets: tuple[int, ...] = ()
    elif args.gpu_experts_per_layer_by_stage:
        gpu_budgets = tuple(
            int(value) for value in args.gpu_experts_per_layer_by_stage.split(",")
        )
        gpu_mask = make_pipeline_gpu_mask(
            counts,
            gpu_experts_per_layer_by_stage=gpu_budgets,
            stage_layer_partition=stage_partition,
        )
    else:
        gpu_budgets = (args.gpu_experts_per_layer,)
        gpu_mask = make_gpu_mask(counts, args.gpu_experts_per_layer)
    fwuff_cold_count = (
        args.fwuff_cold_experts_per_layer
        if args.fwuff_cold_experts_per_layer is not None
        else args.fwuff_experts_per_layer
    )
    fwuff_balanced_count = args.fwuff_balanced_experts_per_layer
    fwuff_total_count = fwuff_cold_count + fwuff_balanced_count
    if fwuff_cold_count < 0 or fwuff_balanced_count < 0:
        raise ValueError("fwuff expert counts must be nonnegative")
    if not 0.0 <= args.fwuff_target_share <= 1.0:
        raise ValueError("--fwuff-target-share must be between zero and one")
    if (
        gpu_mask[: args.serve_layers].sum(dim=1)
        + args.opposite_numa_experts_per_layer
        + fwuff_total_count
        >= num_experts
    ).any():
        raise ValueError("configured tiers leave no native local experts")
    fwuff_ids_by_layer = []
    opposite_ids_by_layer = []
    all_ids = torch.arange(num_experts, dtype=torch.int64)
    for layer_idx in range(args.serve_layers):
        frequency = counts[layer_idx]
        non_gpu_ids = all_ids[~gpu_mask[layer_idx]]
        fwuff_order = torch.argsort(frequency[non_gpu_ids], stable=True)
        fwuff_cold_ids = non_gpu_ids[fwuff_order[:fwuff_cold_count]]
        if fwuff_balanced_count:
            fwuff_cold_mask = torch.zeros(num_experts, dtype=torch.bool)
            fwuff_cold_mask[fwuff_cold_ids] = True
            balanced_available_ids = all_ids[~(gpu_mask[layer_idx] | fwuff_cold_mask)]
            non_gpu_load = int(frequency[non_gpu_ids].sum())
            remaining_target_load = max(
                non_gpu_load * args.fwuff_target_share
                - int(frequency[fwuff_cold_ids].sum()),
                0.0,
            )
            fwuff_balanced_ids = choose_ids_toward_load(
                frequency,
                balanced_available_ids,
                fwuff_balanced_count,
                remaining_target_load,
            )
            fwuff_ids = torch.cat((fwuff_cold_ids, fwuff_balanced_ids)).sort().values
        else:
            # Preserve the historical cold-only storage order exactly.
            fwuff_ids = fwuff_cold_ids
        fwuff_mask = torch.zeros(num_experts, dtype=torch.bool)
        fwuff_mask[fwuff_ids] = True
        available_mask = ~(gpu_mask[layer_idx] | fwuff_mask)
        available_ids = all_ids[available_mask]
        opposite_ids = choose_balanced_sidecar_ids(
            frequency,
            available_ids,
            args.opposite_numa_experts_per_layer,
        )
        if torch.isin(opposite_ids, fwuff_ids).any():
            raise RuntimeError("opposite-NUMA and fwuff tiers overlap")
        if gpu_mask[layer_idx, opposite_ids].any():
            raise RuntimeError("opposite-NUMA and GPU tiers overlap")
        fwuff_ids_by_layer.append(fwuff_ids)
        opposite_ids_by_layer.append(opposite_ids)

    fwuff_ids = torch.stack(fwuff_ids_by_layer)
    opposite_ids = torch.stack(opposite_ids_by_layer)
    args.fwuff_output.parent.mkdir(parents=True, exist_ok=True)
    args.opposite_numa_output.parent.mkdir(parents=True, exist_ok=True)
    if args.gpu_mask_output is not None:
        args.gpu_mask_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gpu_experts_mask": gpu_mask,
                "activation_counts": counts,
                "profile_path": str(args.profile.resolve()),
                "stage_layer_partition": stage_partition
                if (
                    args.gpu_expert_slots_by_stage
                    or args.gpu_experts_per_layer_by_stage
                )
                else None,
                "gpu_expert_slots_by_stage": gpu_slots
                if args.gpu_expert_slots_by_stage
                else None,
                "gpu_mask_input": str(args.gpu_mask_input.resolve())
                if args.gpu_mask_input
                else None,
            },
            args.gpu_mask_output,
        )
    torch.save(
        make_plan(
            remote_expert_ids=fwuff_ids,
            activation_counts=counts,
            profile_path=args.profile,
            tier="fwuff-cold"
            if not fwuff_balanced_count
            else "fwuff-cold-plus-balanced",
        ),
        args.fwuff_output,
    )
    torch.save(
        make_plan(
            remote_expert_ids=opposite_ids,
            activation_counts=counts,
            profile_path=args.profile,
            tier="opposite-numa-balanced",
        ),
        args.opposite_numa_output,
    )

    served_counts = counts[: args.serve_layers]
    served_routes = int(served_counts.sum().item())
    gpu_routes = int(
        served_counts.masked_select(gpu_mask[: args.serve_layers]).sum().item()
    )
    fwuff_routes = int(served_counts.gather(1, fwuff_ids).sum().item())
    opposite_routes = int(served_counts.gather(1, opposite_ids).sum().item())
    native_local_routes = served_routes - gpu_routes - fwuff_routes - opposite_routes
    print(
        f"wrote tiers for {args.serve_layers}/{num_layers} layers: "
        f"GPU={gpu_routes / max(served_routes, 1):.4%}, "
        f"opposite_NUMA={opposite_routes / max(served_routes, 1):.4%}, "
        f"fwuff={fwuff_routes / max(served_routes, 1):.4%}, "
        f"native_local={native_local_routes / max(served_routes, 1):.4%}, "
        f"fwuff_cold_slots={fwuff_cold_count}, "
        f"fwuff_balanced_slots={fwuff_balanced_count}, "
        f"GPU_budgets={gpu_budgets}, "
        f"GPU_slots_by_stage="
        f"{gpu_slots if args.gpu_expert_slots_by_stage else None}"
    )
    layer_start = 0
    for stage_index, stage_layer_count in enumerate(stage_partition):
        layer_end = min(
            layer_start + stage_layer_count,
            args.serve_layers,
        )
        if layer_start >= args.serve_layers:
            break
        stage_counts = counts[layer_start:layer_end]
        stage_routes = int(stage_counts.sum())
        stage_gpu_routes = int(
            stage_counts.masked_select(gpu_mask[layer_start:layer_end]).sum()
        )
        stage_fwuff_routes = int(
            stage_counts.gather(
                1,
                fwuff_ids[layer_start:layer_end],
            ).sum()
        )
        stage_opposite_routes = int(
            stage_counts.gather(
                1,
                opposite_ids[layer_start:layer_end],
            ).sum()
        )
        stage_local_routes = (
            stage_routes - stage_gpu_routes - stage_fwuff_routes - stage_opposite_routes
        )
        print(
            f"stage{stage_index}[{layer_start}:{layer_end}]: "
            f"GPU={stage_gpu_routes / max(stage_routes, 1):.4%}, "
            f"opposite_NUMA="
            f"{stage_opposite_routes / max(stage_routes, 1):.4%}, "
            f"fwuff={stage_fwuff_routes / max(stage_routes, 1):.4%}, "
            f"native_local={stage_local_routes / max(stage_routes, 1):.4%}"
        )
        layer_start = layer_end


if __name__ == "__main__":
    main()
