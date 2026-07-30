#!/usr/bin/env python3
"""Validate the low-memory DSV4 output projection at checkpoint geometry."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(tensor_bytes.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.tokens <= args.chunk_size:
        raise ValueError("--tokens must exceed --chunk-size to exercise the tail")

    device = torch.device(args.device)
    dtype = torch.bfloat16
    groups = 16
    group_width = 4096
    low_rank = 1024
    hidden_size = 7168

    torch.manual_seed(42)
    torch.cuda.set_device(device)
    attention_output = torch.randn(
        args.tokens,
        groups,
        group_width,
        device=device,
        dtype=dtype,
    )
    wo_a = torch.randn(
        groups,
        low_rank,
        group_width,
        device=device,
        dtype=dtype,
    )
    wo_b = torch.randn(
        hidden_size,
        groups * low_rank,
        device=device,
        dtype=dtype,
    )

    torch.cuda.reset_peak_memory_stats(device)
    reference_base_bytes = torch.cuda.memory_allocated(device)
    reference_low_rank = torch.einsum("tgd,grd->tgr", attention_output, wo_a)
    reference = torch.nn.functional.linear(reference_low_rank.flatten(1), wo_b)
    torch.cuda.synchronize(device)
    reference_peak_bytes = (
        torch.cuda.max_memory_allocated(device) - reference_base_bytes
    )

    output_buffer = torch.full(
        (args.tokens, hidden_size),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    output_pointer = output_buffer.data_ptr()
    del reference_low_rank
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    streamed_base_bytes = torch.cuda.memory_allocated(device)
    for token_start in range(0, args.tokens, args.chunk_size):
        token_end = min(token_start + args.chunk_size, args.tokens)
        low_rank_chunk = torch.einsum(
            "tgd,grd->tgr",
            attention_output[token_start:token_end],
            wo_a,
        )
        output_buffer[token_start:token_end].copy_(
            torch.nn.functional.linear(low_rank_chunk.flatten(1), wo_b)
        )
    torch.cuda.synchronize(device)
    streamed_peak_bytes = (
        torch.cuda.max_memory_allocated(device) - streamed_base_bytes
    )

    reference_float = reference.float()
    streamed_float = output_buffer.float()
    reference_absolute_mean = reference_float.abs().mean()
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.flatten(),
        streamed_float.flatten(),
        dim=0,
    ).item()
    absolute_difference = (reference_float - streamed_float).abs()
    result = {
        "geometry": {
            "tokens": args.tokens,
            "groups": groups,
            "group_width": group_width,
            "low_rank": low_rank,
            "hidden_size": hidden_size,
            "chunk_size": args.chunk_size,
        },
        "device": str(device),
        "dtype": str(dtype),
        "output_pointer_preserved": output_buffer.data_ptr() == output_pointer,
        "cosine": cosine,
        "reference_absolute_mean": reference_absolute_mean.item(),
        "mean_absolute_difference": absolute_difference.mean().item(),
        "relative_mean_absolute_difference": (
            absolute_difference.mean() / reference_absolute_mean
        ).item(),
        "max_absolute_difference": absolute_difference.max().item(),
        "reference_sha256": tensor_sha256(reference),
        "streamed_sha256": tensor_sha256(output_buffer),
        "reference_peak_allocated_bytes": reference_peak_bytes,
        "streamed_peak_allocated_bytes": streamed_peak_bytes,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
