#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed-width subset of an existing DSV4 remote-expert "
            "plan without changing the other ownership tiers."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts-per-layer", type=int, required=True)
    parser.add_argument(
        "--selection",
        choices=("hottest", "coldest"),
        default="hottest",
        help="Rank the input tier's experts by its recorded activation counts.",
    )
    parser.add_argument("--tier", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.experts_per_layer <= 0:
        raise ValueError("--experts-per-layer must be positive")

    plan: dict[str, Any] = torch.load(
        arguments.input, map_location="cpu", weights_only=False
    )
    remote_expert_ids = plan["remote_expert_ids"]
    activation_counts = plan["activation_counts"]
    if not isinstance(remote_expert_ids, torch.Tensor) or not isinstance(
        activation_counts, torch.Tensor
    ):
        raise TypeError("plan expert IDs and activation counts must be tensors")
    if remote_expert_ids.shape != activation_counts.shape:
        raise ValueError("plan expert IDs and activation counts have different shapes")
    if remote_expert_ids.ndim != 2:
        raise ValueError("plan tensors must have shape [layers, experts]")
    if arguments.experts_per_layer > remote_expert_ids.shape[1]:
        raise ValueError("requested subset is wider than the input plan")

    subset_ids = torch.full(
        (remote_expert_ids.shape[0], arguments.experts_per_layer),
        -1,
        dtype=remote_expert_ids.dtype,
    )
    subset_counts = torch.zeros(
        (activation_counts.shape[0], arguments.experts_per_layer),
        dtype=activation_counts.dtype,
    )
    descending = arguments.selection == "hottest"

    for layer_index in range(remote_expert_ids.shape[0]):
        valid = remote_expert_ids[layer_index] >= 0
        layer_ids = remote_expert_ids[layer_index][valid]
        layer_counts = activation_counts[layer_index][valid]
        if layer_ids.numel() == 0:
            continue
        if layer_ids.numel() < arguments.experts_per_layer:
            raise ValueError(
                f"layer {layer_index} has only {layer_ids.numel()} input experts"
            )
        order = torch.argsort(layer_counts, descending=descending, stable=True)
        selected = order[: arguments.experts_per_layer]
        subset_ids[layer_index] = layer_ids[selected]
        subset_counts[layer_index] = layer_counts[selected]

    output_plan = dict(plan)
    output_plan["remote_expert_ids"] = subset_ids
    output_plan["activation_counts"] = subset_counts
    output_plan["tier"] = arguments.tier
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_plan, arguments.output)

    active_rows = (subset_ids >= 0).any(dim=1)
    print(
        f"wrote {arguments.output}: shape={tuple(subset_ids.shape)} "
        f"active_layers={int(active_rows.sum())} "
        f"expert_slots={int((subset_ids >= 0).sum())}"
    )


if __name__ == "__main__":
    main()
