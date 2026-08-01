#!/usr/bin/env python3
"""Convert DSV4 route-volume traces into decode expert-call frequencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-profiles",
        type=Path,
        nargs="+",
        required=True,
        help="Per-pipeline-rank recorder profiles in rank order.",
    )
    parser.add_argument(
        "--stage-layer-partition",
        default="23,25,13",
        help="Comma-separated pipeline layer counts.",
    )
    parser.add_argument(
        "--decode-routes-per-layer",
        type=int,
        default=36,
        help="Expected tokens times top-k for one target verification.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage_partition = tuple(
        int(value) for value in args.stage_layer_partition.split(",")
    )
    if len(stage_partition) != len(args.rank_profiles):
        raise ValueError("rank profile count must match the pipeline layer partition")
    loaded_profiles = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.rank_profiles
    ]
    raw_counts = []
    for path, profile in zip(args.rank_profiles, loaded_profiles, strict=True):
        if not isinstance(profile, dict) or not isinstance(
            profile.get("logical_count"), torch.Tensor
        ):
            raise ValueError(f"profile lacks logical_count tensor: {path}")
        tensor = profile["logical_count"]
        if tensor.ndim != 3:
            raise ValueError(f"profile logical_count must be three-dimensional: {path}")
        raw_counts.append(tensor.to(device="cpu"))

    shape = raw_counts[0].shape[1:]
    if any(tuple(tensor.shape[1:]) != tuple(shape) for tensor in raw_counts):
        raise ValueError("rank profiles have different layer/expert shapes")
    if sum(stage_partition) != shape[0]:
        raise ValueError("pipeline layer partition does not cover the profile layers")

    call_counts = torch.zeros(shape, dtype=torch.int64)
    route_counts = torch.zeros(shape, dtype=torch.int64)
    decode_records_by_rank = []
    layer_start = 0
    for tensor, layer_count in zip(raw_counts, stage_partition, strict=True):
        layer_end = layer_start + layer_count
        stage = tensor[:, layer_start:layer_end]
        decode_rows = stage[:, 0].sum(dim=-1) == args.decode_routes_per_layer
        if not bool(decode_rows.any()):
            raise ValueError(
                f"rank stage {layer_start}:{layer_end} has no decode records "
                f"with {args.decode_routes_per_layer} routes"
            )
        selected = stage[decode_rows]
        per_layer_totals = selected.sum(dim=-1)
        if not bool(torch.all(per_layer_totals == args.decode_routes_per_layer)):
            raise ValueError(
                f"decode record route total changes inside stage "
                f"{layer_start}:{layer_end}"
            )
        call_counts[layer_start:layer_end] = (selected > 0).sum(
            dim=0, dtype=torch.int64
        )
        route_counts[layer_start:layer_end] = selected.sum(dim=0, dtype=torch.int64)
        decode_records_by_rank.append(int(decode_rows.sum().item()))
        layer_start = layer_end

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Existing placement loaders consume logical_count. A count here
            # represents one native expert weight-streaming call, irrespective
            # of how many correlated verification tokens share that call.
            "logical_count": call_counts,
            "decode_route_count": route_counts,
            "decode_records_by_rank": decode_records_by_rank,
            "decode_routes_per_layer": args.decode_routes_per_layer,
            "stage_layer_partition": stage_partition,
            "source_profiles": [str(path.resolve()) for path in args.rank_profiles],
            "metric": "distinct_decode_expert_calls",
        },
        args.output,
    )
    print(
        f"wrote {args.output}: shape={tuple(call_counts.shape)} "
        f"decode_records={decode_records_by_rank} "
        f"expert_calls={int(call_counts.sum().item())}"
    )


if __name__ == "__main__":
    main()
