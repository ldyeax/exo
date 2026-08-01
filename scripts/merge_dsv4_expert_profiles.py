#!/usr/bin/env python3
"""Merge target-only SGLang expert profiles from pipeline ranks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def reduce_logical_count(profile_path: Path) -> torch.Tensor:
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    if not isinstance(profile, dict):
        raise ValueError(f"{profile_path} is not a profile dictionary")
    logical_count = profile.get("logical_count")
    if not isinstance(logical_count, torch.Tensor):
        raise ValueError(f"{profile_path} has no logical_count tensor")
    if logical_count.ndim == 3:
        logical_count = logical_count.sum(dim=0, dtype=torch.int64)
    elif logical_count.ndim == 2:
        logical_count = logical_count.to(dtype=torch.int64)
    else:
        raise ValueError(
            f"{profile_path} logical_count must be 2-D or 3-D, "
            f"got {tuple(logical_count.shape)}"
        )
    return logical_count.cpu().contiguous()


def merge_profiles(profile_paths: list[Path]) -> torch.Tensor:
    if not profile_paths:
        raise ValueError("at least one profile is required")
    merged = reduce_logical_count(profile_paths[0])
    for profile_path in profile_paths[1:]:
        rank_count = reduce_logical_count(profile_path)
        if rank_count.shape != merged.shape:
            raise ValueError(
                "pipeline-rank profiles have different shapes: "
                f"{tuple(merged.shape)} versus {tuple(rank_count.shape)} "
                f"from {profile_path}"
            )
        merged.add_(rank_count)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("profiles", nargs="+", type=Path)
    args = parser.parse_args()

    logical_count = merge_profiles(args.profiles)
    populated_layers = int((logical_count.sum(dim=1) > 0).sum().item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logical_count": logical_count,
            "source_profiles": [str(profile.resolve()) for profile in args.profiles],
        },
        args.output,
    )
    print(
        f"wrote {args.output}: shape={tuple(logical_count.shape)} "
        f"routes={int(logical_count.sum().item())} "
        f"populated_layers={populated_layers}"
    )


if __name__ == "__main__":
    main()
