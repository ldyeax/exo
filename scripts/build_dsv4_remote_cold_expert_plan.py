#!/usr/bin/env python3
"""Build a profile-guided native-MXFP4 cold-expert sidecar plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--serve-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--remote-experts-per-layer", type=int, default=8)
    parser.add_argument(
        "--selection",
        choices=("cold", "hot"),
        default="cold",
        help="Use hot only for a route-forcing integration oracle.",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    return parser.parse_args()


def load_activation_counts(
    profile_path: Path,
    *,
    serve_layers: int,
    num_experts: int,
) -> torch.Tensor:
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    if not isinstance(profile, dict) or "logical_count" not in profile:
        raise ValueError(
            f"{profile_path} must contain a logical_count tensor"
        )
    logical_count = profile["logical_count"]
    if not isinstance(logical_count, torch.Tensor):
        logical_count = torch.as_tensor(logical_count)
    if logical_count.ndim == 3:
        logical_count = logical_count.sum(dim=0)
    if logical_count.ndim != 2:
        raise ValueError(
            "logical_count must have shape [samples, layers, experts] or "
            f"[layers, experts], got {tuple(logical_count.shape)}"
        )
    if logical_count.shape[0] < serve_layers:
        raise ValueError(
            f"profile has {logical_count.shape[0]} layers, need {serve_layers}"
        )
    if logical_count.shape[1] != num_experts:
        raise ValueError(
            f"profile has {logical_count.shape[1]} experts, need {num_experts}"
        )
    return logical_count[:serve_layers].to(torch.int64).contiguous()


def main() -> None:
    args = parse_args()
    if not 0 < args.remote_experts_per_layer < args.num_experts:
        raise ValueError(
            "--remote-experts-per-layer must be between 1 and num_experts-1"
        )
    activation_counts = load_activation_counts(
        args.profile,
        serve_layers=args.serve_layers,
        num_experts=args.num_experts,
    )
    # Stable sorting makes equal-count ties deterministic by global expert ID.
    remote_expert_ids = torch.argsort(
        activation_counts,
        dim=1,
        descending=args.selection == "hot",
        stable=True,
    )[:, : args.remote_experts_per_layer].contiguous()
    selected_counts = torch.gather(
        activation_counts, 1, remote_expert_ids
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "remote_expert_ids": remote_expert_ids,
            "activation_counts": selected_counts,
            "profile_path": str(args.profile.resolve()),
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "topk": args.topk,
            "swiglu_limit": args.swiglu_limit,
            "num_experts": args.num_experts,
        },
        args.output,
    )
    print(
        "wrote",
        args.output,
        f"layers={args.serve_layers}",
        f"remote/layer={args.remote_experts_per_layer}",
        f"selection={args.selection}",
        f"selected_routes={int(selected_counts.sum().item())}",
        f"profile_routes={int(activation_counts.sum().item())}",
    )


if __name__ == "__main__":
    main()
