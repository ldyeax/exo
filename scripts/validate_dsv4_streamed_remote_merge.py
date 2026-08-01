#!/usr/bin/env python3
"""Validate pinned receive and chunked GPU merge for a live expert sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
from sglang.srt.layers.moe.kt_remote_sidecar import KTExpertSidecarClient


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:29562")
    parser.add_argument(
        "--plan",
        default=(
            "/var/lib/exo/plans/dsv4-pro-profile7/balanced-fwuff12/opposite-numa64.pt"
        ),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--merge-chunk-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = torch.load(args.plan, map_location="cpu", weights_only=True)
    expert_ids = plan["remote_expert_ids"][args.layer]
    expert_ids = expert_ids[expert_ids >= 0][:6].to(torch.int64)
    if expert_ids.numel() != 6:
        raise RuntimeError("validation plan does not expose six remote experts")

    torch.manual_seed(42)
    device = torch.device(args.device)
    hidden_states = torch.randn(
        args.tokens,
        int(plan["hidden_size"]),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = expert_ids.to(device).repeat(args.tokens, 1)
    topk_weights = torch.rand(
        args.tokens,
        6,
        dtype=torch.float32,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    client = KTExpertSidecarClient.get(args.endpoint)
    reference = client.forward(
        layer_idx=args.layer,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    ).cpu()
    pinned_output = client.forward(
        layer_idx=args.layer,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        return_cpu=True,
    )

    selected_indices = torch.arange(args.tokens, device=device)
    staging = torch.empty(
        args.merge_chunk_tokens,
        hidden_states.shape[-1],
        dtype=hidden_states.dtype,
        device=device,
    )
    streamed = torch.zeros_like(hidden_states)
    for token_start in range(0, args.tokens, args.merge_chunk_tokens):
        token_end = min(
            token_start + args.merge_chunk_tokens,
            args.tokens,
        )
        row_count = token_end - token_start
        transfer = staging[:row_count]
        transfer.copy_(
            pinned_output[token_start:token_end],
            non_blocking=True,
        )
        streamed.index_add_(
            0,
            selected_indices[token_start:token_end],
            transfer,
        )
    torch.cuda.synchronize(device)
    streamed_cpu = streamed.cpu()

    result = {
        "endpoint": args.endpoint,
        "layer": args.layer,
        "tokens": args.tokens,
        "hidden_size": hidden_states.shape[-1],
        "merge_chunk_tokens": args.merge_chunk_tokens,
        "pinned_output": pinned_output.is_pinned(),
        "cpu_receive_byte_exact": torch.equal(reference, pinned_output),
        "streamed_merge_byte_exact": torch.equal(reference, streamed_cpu),
        "reference_sha256": tensor_sha256(reference),
        "pinned_sha256": tensor_sha256(pinned_output),
        "streamed_sha256": tensor_sha256(streamed_cpu),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["cpu_receive_byte_exact"]:
        raise SystemExit(1)
    if not result["streamed_merge_byte_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
