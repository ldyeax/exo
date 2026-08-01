#!/usr/bin/env python3
"""Validate caller-owned DSV4 embedding output on the actual checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark"),
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.model / "model.safetensors.index.json"
    index = __import__("json").loads(index_path.read_text())
    shard_path = args.model / index["weight_map"]["embed.weight"]
    with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor("embed.weight").cuda()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        weight.shape[0],
        (args.tokens,),
        device="cuda",
        generator=generator,
    )
    reference = torch_functional.embedding(input_ids, weight)
    output = torch.empty_like(reference)
    output_pointer = output.data_ptr()
    layer = SimpleNamespace(weight=weight)
    method = UnquantizedEmbeddingMethod()

    actual = method.embedding_into(layer, input_ids, output)
    torch.cuda.synchronize()
    if actual.data_ptr() != output_pointer:
        raise AssertionError("embedding_into rebound caller output")
    if not torch.equal(reference.view(torch.uint16), actual.view(torch.uint16)):
        difference = (reference.float() - actual.float()).abs()
        raise AssertionError(
            "caller-owned embedding differs from F.embedding: "
            f"max_abs={difference.max().item()}"
        )

    del reference
    torch.cuda.empty_cache()
    for _ in range(3):
        method.embedding_into(layer, input_ids, output)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    method.embedding_into(layer, input_ids, output)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    if peak_delta != 0:
        raise AssertionError(f"steady embedding_into allocated {peak_delta} bytes")

    print(f"weight_shape={tuple(weight.shape)} dtype={weight.dtype}")
    print(f"output_shape={tuple(output.shape)} pointer_preserved=True")
    print(f"byte_exact=True sha256={tensor_sha256(output)}")
    print(f"steady_alloc_delta_bytes={peak_delta}")


if __name__ == "__main__":
    main()
