#!/usr/bin/env python3
"""Build a memory-neutral two-host, three-pool DSV4 decode layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_tensor(path: Path, key: str) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not isinstance(data.get(key), torch.Tensor):
        raise ValueError(f"{path} must contain tensor {key!r}")
    return data[key].to(device="cpu").contiguous()


def choose_tier(
    frequency: torch.Tensor,
    available_ids: torch.Tensor,
    *,
    expert_count: int,
    target_load: float,
) -> torch.Tensor:
    """Choose a fixed-width tier close to a target call load."""
    if not 0 <= expert_count <= available_ids.numel():
        raise ValueError("tier expert count exceeds available experts")
    ordered = sorted(
        (int(expert_id) for expert_id in available_ids),
        key=lambda expert_id: (-int(frequency[expert_id]), expert_id),
    )
    selected = []
    selected_set = set()
    selected_load = 0
    for expert_id in ordered:
        if len(selected) == expert_count:
            break
        candidate_load = selected_load + int(frequency[expert_id])
        if candidate_load <= target_load or not selected:
            selected.append(expert_id)
            selected_set.add(expert_id)
            selected_load = candidate_load
    if len(selected) < expert_count:
        for expert_id in reversed(ordered):
            if expert_id not in selected_set:
                selected.append(expert_id)
                selected_set.add(expert_id)
                if len(selected) == expert_count:
                    break
    return torch.tensor(sorted(selected), dtype=torch.int64)


def make_plan(
    expert_ids: torch.Tensor,
    calls: torch.Tensor,
    *,
    tier: str,
    active_layer_ranges: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "remote_expert_ids": expert_ids,
        "activation_counts": calls.gather(1, expert_ids),
        "tier": tier,
        "active_layer_ranges": active_layer_ranges,
        "hidden_size": 7168,
        "intermediate_size": 3072,
        "topk": 6,
        "swiglu_limit": 10.0,
        "num_experts": calls.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-call-profile", type=Path, required=True)
    parser.add_argument("--gpu-mask-plan", type=Path, required=True)
    parser.add_argument("--dwagon-numa0-output", type=Path, required=True)
    parser.add_argument("--dwagon-numa1-output", type=Path, required=True)
    parser.add_argument("--fwuff-output", type=Path, required=True)
    parser.add_argument("--stage-layer-partition", default="23,25,13")
    parser.add_argument("--dwagon-experts-per-layer", type=int, default=64)
    parser.add_argument("--fwuff-experts-per-layer", type=int, default=36)
    parser.add_argument(
        "--pp2-dwagon-tiers",
        choices=(1, 2),
        type=int,
        default=2,
        help=(
            "Use one or both dwagon NUMA sidecars for PP2. One tier reduces "
            "per-layer RPC joins and leaves the other socket plan inactive."
        ),
    )
    parser.add_argument(
        "--fwuff-placement",
        choices=("cold", "balanced"),
        default="cold",
        help=(
            "Choose least-called fwuff experts for capacity offload, or choose "
            "a fixed-width tier targeting one third of non-GPU decode calls."
        ),
    )
    args = parser.parse_args()

    calls = load_tensor(args.decode_call_profile, "logical_count").to(dtype=torch.int64)
    gpu_mask = load_tensor(args.gpu_mask_plan, "gpu_experts_mask").to(dtype=torch.bool)
    if tuple(calls.shape) != tuple(gpu_mask.shape):
        raise ValueError("decode profile and GPU mask shapes differ")
    stage_partition = tuple(
        int(value) for value in args.stage_layer_partition.split(",")
    )
    if len(stage_partition) != 3 or sum(stage_partition) != calls.shape[0]:
        raise ValueError("bidirectional layout requires three complete stages")
    stage0_end = stage_partition[0]
    stage1_end = stage0_end + stage_partition[1]
    num_layers, num_experts = calls.shape
    all_ids = torch.arange(num_experts, dtype=torch.int64)

    numa0_rows = []
    numa1_rows = []
    fwuff_rows = []
    tier_calls = {
        "gpu": [0, 0, 0],
        "dwagon_numa0": [0, 0, 0],
        "dwagon_numa1": [0, 0, 0],
        "fwuff": [0, 0, 0],
        "local": [0, 0, 0],
    }
    for layer_idx in range(num_layers):
        frequency = calls[layer_idx]
        available = all_ids[~gpu_mask[layer_idx]]
        stage_idx = 0 if layer_idx < stage0_end else 1 if layer_idx < stage1_end else 2
        tier_calls["gpu"][stage_idx] += int(frequency[gpu_mask[layer_idx]].sum())

        if layer_idx < stage1_end:
            if args.fwuff_placement == "cold":
                cold_order = torch.argsort(frequency[available], stable=True)
                fwuff_ids = available[cold_order[: args.fwuff_experts_per_layer]]
            else:
                fwuff_ids = choose_tier(
                    frequency,
                    available,
                    expert_count=args.fwuff_experts_per_layer,
                    target_load=int(frequency[available].sum()) / 3,
                )
            after_fwuff = available[~torch.isin(available, fwuff_ids)]
            target = int(frequency[after_fwuff].sum()) / 2
            opposite_ids = choose_tier(
                frequency,
                after_fwuff,
                expert_count=args.dwagon_experts_per_layer,
                target_load=target,
            )
            placeholder_ids = opposite_ids.clone()
            if stage_idx == 0:
                numa0_ids = opposite_ids
                numa1_ids = placeholder_ids
                tier_calls["dwagon_numa0"][0] += int(frequency[numa0_ids].sum())
            else:
                numa0_ids = placeholder_ids
                numa1_ids = opposite_ids
                tier_calls["dwagon_numa1"][1] += int(frequency[numa1_ids].sum())
            tier_calls["fwuff"][stage_idx] += int(frequency[fwuff_ids].sum())
            owned = gpu_mask[layer_idx].clone()
            owned[fwuff_ids] = True
            owned[opposite_ids] = True
            tier_calls["local"][stage_idx] += int(frequency[~owned].sum())
            fwuff_rows.append(fwuff_ids)
        else:
            total_cpu_load = int(frequency[available].sum())
            numa0_ids = choose_tier(
                frequency,
                available,
                expert_count=args.dwagon_experts_per_layer,
                target_load=total_cpu_load / (args.pp2_dwagon_tiers + 1),
            )
            owned = gpu_mask[layer_idx].clone()
            owned[numa0_ids] = True
            tier_calls["dwagon_numa0"][2] += int(frequency[numa0_ids].sum())
            if args.pp2_dwagon_tiers == 2:
                after_numa0 = available[~torch.isin(available, numa0_ids)]
                numa1_ids = choose_tier(
                    frequency,
                    after_numa0,
                    expert_count=args.dwagon_experts_per_layer,
                    target_load=total_cpu_load / 3,
                )
                owned[numa1_ids] = True
                tier_calls["dwagon_numa1"][2] += int(frequency[numa1_ids].sum())
            else:
                # Keep the serialized plan rectangular. These stage-2 rows
                # are inactive for NUMA 1 and therefore need no ownership.
                numa1_ids = numa0_ids.clone()
            tier_calls["local"][2] += int(frequency[~owned].sum())

        numa0_rows.append(numa0_ids)
        numa1_rows.append(numa1_ids)

    numa0 = torch.stack(numa0_rows)
    numa1 = torch.stack(numa1_rows)
    fwuff = torch.stack(fwuff_rows)
    for layer_idx in range(num_layers):
        active_ids = [all_ids[gpu_mask[layer_idx]]]
        if layer_idx < stage0_end:
            active_ids.extend((numa0[layer_idx], fwuff[layer_idx]))
        elif layer_idx < stage1_end:
            active_ids.extend((numa1[layer_idx], fwuff[layer_idx]))
        else:
            active_ids.append(numa0[layer_idx])
            if args.pp2_dwagon_tiers == 2:
                active_ids.append(numa1[layer_idx])
        assigned = torch.cat(active_ids)
        if torch.unique(assigned).numel() != assigned.numel():
            raise RuntimeError(f"tier overlap at layer {layer_idx}")

    for output in (
        args.dwagon_numa0_output,
        args.dwagon_numa1_output,
        args.fwuff_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        make_plan(
            numa0,
            calls,
            tier="dwagon-numa0-bidirectional",
            active_layer_ranges=((0, stage0_end), (stage1_end, num_layers)),
        ),
        args.dwagon_numa0_output,
    )
    torch.save(
        make_plan(
            numa1,
            calls,
            tier="dwagon-numa1-bidirectional",
            active_layer_ranges=((stage0_end, stage1_end),)
            if args.pp2_dwagon_tiers == 1
            else (
                (stage0_end, stage1_end),
                (stage1_end, num_layers),
            ),
        ),
        args.dwagon_numa1_output,
    )
    torch.save(
        make_plan(
            fwuff,
            calls[:stage1_end],
            tier=f"fwuff-{args.fwuff_placement}-bidirectional",
            active_layer_ranges=((0, stage1_end),),
        ),
        args.fwuff_output,
    )

    print(
        f"expert slots: numa0={(stage_partition[0] + stage_partition[2]) * args.dwagon_experts_per_layer}, "
        f"numa1={(stage_partition[1] + (stage_partition[2] if args.pp2_dwagon_tiers == 2 else 0)) * args.dwagon_experts_per_layer}, "
        f"fwuff={stage1_end * args.fwuff_experts_per_layer}"
    )
    for stage_idx in range(3):
        stage_total = sum(values[stage_idx] for values in tier_calls.values())
        print(
            f"stage{stage_idx}: "
            + ", ".join(
                f"{name}={values[stage_idx]} "
                f"({values[stage_idx] / max(stage_total, 1):.2%})"
                for name, values in tier_calls.items()
            )
        )


if __name__ == "__main__":
    main()
