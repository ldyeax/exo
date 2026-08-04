#!/usr/bin/env python3
"""Benchmark monolithic versus split-history DSV4 Oscar INT2 attention.

The benchmark uses the production H64 decoder and captures both providers in
CUDA graphs.  It measures the C4-512 shape, the short C128 shape seen by the
2,694-token OpenCode workload, and a full C128-4096 shape separately.  This
prevents a C4 optimization from being promoted by silently regressing the 20
C128 layers in the target model.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from sglang.kernels.ops.attention.dsv4.oscar_int2_decode import (
    BF16_BYTES_PER_TOKEN,
    SPLIT_HISTORY_SPLIT_MAP,
    SPLIT_HISTORY_WORKSPACE_BYTES,
    SPLIT_HISTORY_WORKSPACE_FLOATS,
    OscarInt2SplitHistoryWorkspace,
    decode_sparse_attention_oscar_int2,
)
from sglang.kernels.ops.attention.dsv4.oscar_int2_storage import (
    HEAD_DIM,
    ROPE_OFFSET_BYTES,
    SCALE_ZERO_OFFSET_BYTES,
    STORAGE_BYTES_PER_TOKEN,
)

FORMAT: Final = "dsv4_oscar_int2_split_decode_benchmark_v1"
NUM_HEADS: Final = 64
SWA_LENGTH: Final = 128
SWA_PAGE_SIZE: Final = 256
C4_LAYER_COUNT: Final = 21
C128_LAYER_COUNT: Final = 20


@dataclass(frozen=True)
class Shape:
    name: str
    extra_length: int
    extra_page_size: int


SHAPES: Final = (
    Shape("c4_512", 512, 64),
    Shape("c128_short_21", 21, 2),
    Shape("c128_full_4096", 4096, 2),
)


def parse_positive_integer_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("token shapes must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("token shapes must not contain duplicates")
    unsupported = sorted(set(values) - set(SPLIT_HISTORY_SPLIT_MAP))
    if unsupported:
        raise ValueError(f"split-history has no static map for T={unsupported}")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,5")
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _capture(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def _measure_alternating(
    monolithic: torch.cuda.CUDAGraph,
    split: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, object]:
    for _ in range(warmup):
        monolithic.replay()
        split.replay()
    torch.cuda.synchronize()

    provider_samples: dict[str, list[float]] = {"monolithic": [], "split": []}
    providers = {"monolithic": monolithic, "split": split}
    for sample_index in range(samples):
        order = (
            ("monolithic", "split")
            if sample_index % 2 == 0
            else ("split", "monolithic")
        )
        for name in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(iterations):
                providers[name].replay()
            end.record()
            end.synchronize()
            provider_samples[name].append(begin.elapsed_time(end) / iterations)

    monolithic_median = float(statistics.median(provider_samples["monolithic"]))
    split_median = float(statistics.median(provider_samples["split"]))
    return {
        "monolithic_graph_ms": monolithic_median,
        "split_graph_ms": split_median,
        "split_speedup": monolithic_median / split_median,
        "split_savings_ms": monolithic_median - split_median,
        "samples_ms": provider_samples,
    }


def _make_extra_storage(
    shape: Shape,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    page_count = (shape.extra_length + shape.extra_page_size - 1) // (
        shape.extra_page_size
    )
    storage = torch.zeros(
        (page_count, shape.extra_page_size, STORAGE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    storage[..., :ROPE_OFFSET_BYTES].random_(0, 256, generator=generator)
    as_bfloat16 = storage.view(torch.bfloat16)
    rope_start = ROPE_OFFSET_BYTES // 2
    metadata_start = SCALE_ZERO_OFFSET_BYTES // 2
    rope = torch.randn(
        (*storage.shape[:2], (SCALE_ZERO_OFFSET_BYTES - ROPE_OFFSET_BYTES) // 2),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    as_bfloat16[..., rope_start:metadata_start].copy_(rope)
    metadata = as_bfloat16[..., metadata_start : metadata_start + 14]
    metadata[..., 0::2].fill_(0.125)
    metadata[..., 1::2].fill_(1.5)
    return storage.view(page_count, -1)


def _benchmark_shape(
    *,
    shape: Shape,
    num_tokens: int,
    generator: torch.Generator,
    device: torch.device,
    workspace: OscarInt2SplitHistoryWorkspace,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, object]:
    query = torch.randn(
        (num_tokens, NUM_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    sink = torch.linspace(-1.0, 0.0, NUM_HEADS, dtype=torch.float32, device=device)
    extra_storage = _make_extra_storage(
        shape,
        generator=generator,
        device=device,
    )
    extra_width = ((shape.extra_length + 63) // 64) * 64
    extra_indices = torch.full(
        (num_tokens, extra_width), -1, dtype=torch.int32, device=device
    )
    extra_indices[:, : shape.extra_length] = torch.arange(
        shape.extra_length, dtype=torch.int32, device=device
    )
    extra_lengths = torch.full(
        (num_tokens,), shape.extra_length, dtype=torch.int32, device=device
    )

    swa_storage = torch.zeros(
        (1, SWA_PAGE_SIZE * BF16_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    swa_values = swa_storage.view(torch.bfloat16).view(
        1, SWA_PAGE_SIZE, HEAD_DIM
    )
    swa_values[:, :SWA_LENGTH].copy_(
        torch.randn(
            (1, SWA_LENGTH, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
    )
    swa_indices = torch.arange(
        SWA_LENGTH, dtype=torch.int32, device=device
    ).expand(num_tokens, -1)
    swa_lengths = torch.full(
        (num_tokens,), SWA_LENGTH, dtype=torch.int32, device=device
    )
    monolithic_output = torch.empty_like(query)
    split_output = torch.empty_like(query)

    def run(out: torch.Tensor, *, split: bool) -> None:
        decode_sparse_attention_oscar_int2(
            q=query,
            swa_cache=swa_storage,
            swa_indices=swa_indices,
            swa_lens=swa_lengths,
            scale=float(HEAD_DIM**-0.5),
            attn_sink=sink,
            out=out,
            swa_block_size=SWA_PAGE_SIZE,
            extra_cache=extra_storage,
            extra_indices=extra_indices,
            extra_lens=extra_lengths,
            extra_block_size=shape.extra_page_size,
            split_workspace=workspace if split else None,
        )

    monolithic_graph = _capture(lambda: run(monolithic_output, split=False))
    split_graph = _capture(lambda: run(split_output, split=True))
    monolithic_graph.replay()
    split_graph.replay()
    torch.cuda.synchronize()
    difference = (monolithic_output.float() - split_output.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        monolithic_output.float().flatten(),
        split_output.float().flatten(),
        dim=0,
    ).item()
    torch.testing.assert_close(
        split_output,
        monolithic_output,
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    timing = _measure_alternating(
        monolithic_graph,
        split_graph,
        warmup=warmup,
        iterations=iterations,
        samples=samples,
    )
    return {
        "shape": shape.name,
        "num_tokens": num_tokens,
        "num_splits": SPLIT_HISTORY_SPLIT_MAP[num_tokens],
        "extra_length": shape.extra_length,
        "swa_length": SWA_LENGTH,
        "parity": {
            "mean_absolute_difference": difference.mean().item(),
            "maximum_absolute_difference": difference.max().item(),
            "cosine_similarity": cosine,
            "passed": True,
        },
        "timing": timing,
    }


def main() -> int:
    arguments = parse_arguments()
    token_shapes = parse_positive_integer_list(arguments.tokens)
    if arguments.warmup < 0 or arguments.iterations <= 0 or arguments.samples <= 0:
        raise ValueError("invalid timing counts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(arguments.device)
    device = torch.device("cuda", arguments.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 6):
        raise RuntimeError(f"Oscar split-history benchmark requires SM86, got {capability}")

    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    storage = torch.empty(
        SPLIT_HISTORY_WORKSPACE_FLOATS,
        dtype=torch.float32,
        device=device,
    )
    workspace = OscarInt2SplitHistoryWorkspace.from_backend_storage(storage)
    results = [
        _benchmark_shape(
            shape=shape,
            num_tokens=num_tokens,
            generator=generator,
            device=device,
            workspace=workspace,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            samples=arguments.samples,
        )
        for num_tokens in token_shapes
        for shape in SHAPES
    ]

    weighted_current_prompt: dict[str, object] = {}
    for num_tokens in token_shapes:
        by_shape = {
            result["shape"]: result
            for result in results
            if result["num_tokens"] == num_tokens
        }
        c4 = by_shape["c4_512"]["timing"]
        c128 = by_shape["c128_short_21"]["timing"]
        assert isinstance(c4, dict) and isinstance(c128, dict)
        monolithic_ms = (
            C4_LAYER_COUNT * float(c4["monolithic_graph_ms"])
            + C128_LAYER_COUNT * float(c128["monolithic_graph_ms"])
        )
        split_ms = (
            C4_LAYER_COUNT * float(c4["split_graph_ms"])
            + C128_LAYER_COUNT * float(c128["split_graph_ms"])
        )
        weighted_current_prompt[str(num_tokens)] = {
            "monolithic_41_layer_ms": monolithic_ms,
            "split_41_layer_ms": split_ms,
            "split_speedup": monolithic_ms / split_ms,
            "split_savings_ms": monolithic_ms - split_ms,
        }

    receipt = {
        "format": FORMAT,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "seed": arguments.seed,
        "workspace_bytes": SPLIT_HISTORY_WORKSPACE_BYTES,
        "workspace_fixed_address": workspace.fixed_data_ptr,
        "split_map": {str(key): value for key, value in SPLIT_HISTORY_SPLIT_MAP.items()},
        "warmup": arguments.warmup,
        "iterations_per_sample": arguments.iterations,
        "samples": arguments.samples,
        "results": results,
        "weighted_current_prompt": weighted_current_prompt,
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
