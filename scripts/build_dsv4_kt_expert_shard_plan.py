#!/usr/bin/env python3
"""Build a lossless, profile-guided native-MXFP4 expert shard plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_rank_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(","))
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("rank counts must be positive integers")
    return counts


def reduce_profile(raw_counts: torch.Tensor) -> torch.Tensor:
    if raw_counts.ndim == 2:
        return raw_counts.to(dtype=torch.int64, device="cpu")
    if raw_counts.ndim == 3:
        return raw_counts.sum(dim=0, dtype=torch.int64).cpu()
    raise ValueError(
        "logical_count must have shape [layers, experts] or "
        f"[samples, layers, experts], got {tuple(raw_counts.shape)}"
    )


def assign_layer(
    frequency: torch.Tensor, rank_counts: tuple[int, ...], remote_rank: int
) -> list[list[int]]:
    num_experts = frequency.numel()
    if sum(rank_counts) != num_experts:
        raise ValueError(
            f"rank counts sum to {sum(rank_counts)}, expected {num_experts}"
        )
    remote_count = rank_counts[remote_rank]
    ordered_cold = sorted(
        range(num_experts), key=lambda expert_id: (int(frequency[expert_id]), expert_id)
    )
    remote_ids = ordered_cold[:remote_count]
    remote_set = set(remote_ids)

    assignments: list[list[int]] = [[] for _ in rank_counts]
    assignments[remote_rank] = remote_ids
    assigned_load = [0 for _ in rank_counts]
    local_ranks = [rank for rank in range(len(rank_counts)) if rank != remote_rank]
    ordered_hot = sorted(
        (expert_id for expert_id in range(num_experts) if expert_id not in remote_set),
        key=lambda expert_id: (-int(frequency[expert_id]), expert_id),
    )
    for expert_id in ordered_hot:
        eligible = [
            rank for rank in local_ranks if len(assignments[rank]) < rank_counts[rank]
        ]
        if not eligible:
            raise RuntimeError("no rank has capacity for the remaining expert")
        selected_rank = min(
            eligible,
            key=lambda rank: (assigned_load[rank], len(assignments[rank]), rank),
        )
        assignments[selected_rank].append(expert_id)
        assigned_load[selected_rank] += int(frequency[expert_id])

    for rank, expected_count in enumerate(rank_counts):
        if len(assignments[rank]) != expected_count:
            raise RuntimeError(
                f"rank {rank} received {len(assignments[rank])}, expected {expected_count}"
            )
        assignments[rank].sort()
    flattened = sorted(expert for shard in assignments for expert in shard)
    if flattened != list(range(num_experts)):
        raise RuntimeError("expert assignment is not an exact disjoint partition")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-counts", type=parse_rank_counts, required=True)
    parser.add_argument("--remote-rank", type=int, required=True)
    args = parser.parse_args()

    loaded = torch.load(args.profile, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not isinstance(
        loaded.get("logical_count"), torch.Tensor
    ):
        raise ValueError("profile must contain a logical_count tensor")
    frequency = reduce_profile(loaded["logical_count"])
    num_layers, num_experts = frequency.shape
    if not 0 <= args.remote_rank < len(args.rank_counts):
        raise ValueError("remote rank is outside rank-counts")

    assignments_by_layer = [
        assign_layer(frequency[layer_idx], args.rank_counts, args.remote_rank)
        for layer_idx in range(num_layers)
    ]
    expert_ids_by_rank = [
        torch.tensor(
            [layer_assignment[rank] for layer_assignment in assignments_by_layer],
            dtype=torch.int64,
        )
        for rank in range(len(args.rank_counts))
    ]
    remote_frequency = frequency.gather(1, expert_ids_by_rank[args.remote_rank]).sum()
    total_frequency = frequency.sum()
    output = {
        "format": "sglang_kt_cpu_expert_shard_v1",
        "expert_ids_by_rank": expert_ids_by_rank,
        "rank_counts": torch.tensor(args.rank_counts, dtype=torch.int64),
        "remote_rank": torch.tensor(args.remote_rank, dtype=torch.int64),
        "global_num_experts": torch.tensor(num_experts, dtype=torch.int64),
        "source_profile": str(args.profile.resolve()),
        "remote_activation_fraction": (
            remote_frequency.to(torch.float64)
            / total_frequency.clamp_min(1).to(torch.float64)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"wrote {args.output}: layers={num_layers} experts={num_experts} "
        f"rank_counts={args.rank_counts} remote_activation_fraction="
        f"{float(output['remote_activation_fraction']):.6f}"
    )


if __name__ == "__main__":
    main()
