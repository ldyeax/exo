#!/usr/bin/env python3
"""Prove DSV4 TileLang MHC post can overwrite its dead residual input."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
from sglang.srt.layers.mhc import mhc_post_tilelang


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device)
    residual = torch.randn(
        args.tokens,
        args.hc,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    residual_inplace = residual.clone()
    hidden_states = torch.randn(
        args.tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    comb = torch.randn(
        args.tokens,
        args.hc,
        args.hc,
        dtype=torch.float32,
        device=device,
    )
    post = torch.randn(
        args.tokens,
        args.hc,
        dtype=torch.float32,
        device=device,
    )

    reference = torch.empty_like(residual)
    mhc_post_tilelang(
        comb,
        residual,
        post,
        hidden_states,
        reference,
        args.hc,
        args.hidden_size,
    )
    input_pointer = residual_inplace.data_ptr()
    mhc_post_tilelang(
        comb,
        residual_inplace,
        post,
        hidden_states,
        residual_inplace,
        args.hc,
        args.hidden_size,
    )
    torch.cuda.synchronize(device)

    byte_exact = torch.equal(reference, residual_inplace)
    result = {
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "hc": args.hc,
        "dtype": str(residual.dtype),
        "bytes_per_residual": residual.numel() * residual.element_size(),
        "output_pointer_preserved": residual_inplace.data_ptr() == input_pointer,
        "byte_exact": byte_exact,
        "reference_sha256": tensor_sha256(reference),
        "inplace_sha256": tensor_sha256(residual_inplace),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not byte_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
