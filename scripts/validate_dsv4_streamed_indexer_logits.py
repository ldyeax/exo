#!/usr/bin/env python3
"""Validate caller-owned, row-streamed SM86 C4 indexer logits and top-k."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
from sglang.jit_kernel.dsv4 import topk_transform_512
from sglang.srt.layers.attention.dsv4.tilelang_kernel import (
    tilelang_fp8_bf16_paged_mqa_logits,
)


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--rows-per-tile", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_seq_len % 64 != 0:
        raise ValueError("max sequence length must be divisible by 64")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    num_heads = 64
    head_dim = 128
    page_size = 64
    num_blocks = args.max_seq_len // page_size

    q = torch.randn(
        (args.rows, 1, num_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (args.rows, num_heads),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    cache = torch.empty(
        (num_blocks, page_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    cache_pages = cache.view(num_blocks, page_size * (head_dim + 4))
    cache_pages[:, : page_size * head_dim].view(torch.float8_e4m3fn).copy_(
        torch.randn(
            (num_blocks, page_size * head_dim),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).to(torch.float8_e4m3fn)
    )
    cache_pages[:, page_size * head_dim :].view(torch.float32).fill_(0.015625)
    seq_lens = torch.full(
        (args.rows,),
        args.max_seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    page_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device="cuda",
    ).repeat(args.rows, 1)

    logits_reference = tilelang_fp8_bf16_paged_mqa_logits(
        q,
        cache,
        weights,
        seq_lens,
        page_table,
        None,
        args.max_seq_len,
        False,
    )
    topk_reference = torch.empty(
        (args.rows, 512),
        dtype=torch.int32,
        device="cuda",
    )
    topk_transform_512(
        logits_reference,
        seq_lens,
        page_table,
        topk_reference,
        page_size,
    )

    projection_bytes = args.rows * num_heads * head_dim * 2
    quant_bytes = args.rows * num_heads * head_dim
    output_offset = (projection_bytes + quant_bytes + 3) & ~3
    output_bytes = args.rows_per_tile * args.max_seq_len * 4
    workspace = torch.empty(
        output_offset + output_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    logits_storage = workspace[output_offset : output_offset + output_bytes].view(
        torch.float32
    )
    topk_streamed = torch.empty_like(topk_reference)
    caller_pointers_preserved = True

    for row_start in range(0, args.rows, args.rows_per_tile):
        row_end = min(row_start + args.rows_per_tile, args.rows)
        tile_rows = row_end - row_start
        logits_output = logits_storage[: tile_rows * args.max_seq_len].view(
            tile_rows, args.max_seq_len
        )
        logits_tile = tilelang_fp8_bf16_paged_mqa_logits(
            q[row_start:row_end],
            cache,
            weights[row_start:row_end],
            seq_lens[row_start:row_end],
            page_table[row_start:row_end],
            None,
            args.max_seq_len,
            False,
            logits_output=logits_output,
        )
        caller_pointers_preserved &= logits_tile.data_ptr() == logits_output.data_ptr()
        if not torch.equal(
            logits_reference[row_start:row_end],
            logits_tile,
        ):
            raise SystemExit(f"logits mismatch in rows [{row_start}, {row_end})")
        topk_transform_512(
            logits_tile,
            seq_lens[row_start:row_end],
            page_table[row_start:row_end],
            topk_streamed[row_start:row_end],
            page_size,
        )

    torch.cuda.synchronize()
    topk_equal = torch.equal(topk_reference, topk_streamed)
    sorted_topk_equal = torch.equal(
        torch.sort(topk_reference, dim=1).values,
        torch.sort(topk_streamed, dim=1).values,
    )
    mismatched_elements = int((topk_reference != topk_streamed).sum().item())
    mismatched_rows = int((topk_reference != topk_streamed).any(dim=1).sum().item())
    report = {
        "rows": args.rows,
        "max_seq_len": args.max_seq_len,
        "rows_per_tile": args.rows_per_tile,
        "caller_pointers_preserved": caller_pointers_preserved,
        "logits_bitwise_equal": True,
        "topk_bitwise_equal": topk_equal,
        "topk_sets_equal": sorted_topk_equal,
        "topk_mismatched_elements": mismatched_elements,
        "topk_mismatched_rows": mismatched_rows,
        "logits_sha256": _sha256(logits_reference),
        "topk_sha256": _sha256(topk_reference),
        "workspace_bytes": workspace.numel(),
        "reused_logits_bytes": output_bytes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    # The upstream top-k kernel does not define tie ordering: two calls over
    # identical logits can permute equal-score page IDs. Selection-set equality
    # is therefore the invariant; logits themselves remain byte-exact.
    if not caller_pointers_preserved or not sorted_topk_equal:
        raise SystemExit("streamed indexer coherence gate failed")


if __name__ == "__main__":
    main()
