#!/usr/bin/env python3
"""Benchmark SM86 DeepSeek-V4 FP8-SWA/BF16-C128 sparse decode.

The benchmark mirrors TP2 production geometry: 64 local heads, a 128-token
SWA tail, two-token C128 pages, and a statically padded 4096-entry C128 index
row.  It times both ordinary launches and captured graph replay for B1 decode
and B5 target verification at short (21) and full-window (4096) C128 lengths.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import triton
from sglang.kernels.ops.attention.dsv4 import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsv4.fp8_storage import (
    prime_e4m3fn_decode_lut,
)
from sglang.srt.layers.attention.nsa.v4_mixed_c128_bf16_kernel import (
    decode_sparse_attention_fp8_swa_bf16_extra,
)
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    decode_sparse_attention_triton,
)

HEAD_DIM = 512
NUM_HEADS = 64
SWA_PAGE_SIZE = 128
C128_PAGE_SIZE = 2
MAX_C128_TOKENS = 4096
# `/tmp/dsv4-local-checkpoint-0731/config.json` has exactly twenty entries
# whose compression ratio is 128.  Each maps to one physical C128 pool.
NUM_C128_LAYERS = 20
CONTEXT_TOKENS = 524_288


def _fp8_page_bytes(page_size: int) -> int:
    return math.ceil(page_size * 584 / 576) * 576


def _timings(fn: Callable[[], None], warmup_ms: int, rep_ms: int) -> dict[str, float]:
    quantiles = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
        return_mode="median",
    )
    if isinstance(quantiles, float):
        low = median = high = quantiles
    else:
        low, median, high = (float(value) for value in quantiles)
    return {"p20_ms": low, "median_ms": median, "p80_ms": high}


def _capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _build_case(batch_size: int, extra_length: int, seed: int) -> dict[str, object]:
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(seed)

    swa_values = torch.randn(
        (SWA_PAGE_SIZE, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    swa_cache = torch.empty(
        (1, _fp8_page_bytes(SWA_PAGE_SIZE)), device=device, dtype=torch.uint8
    )
    swa_locations = torch.arange(SWA_PAGE_SIZE, device=device, dtype=torch.int32)
    fused_store_cache(
        swa_values,
        swa_cache,
        swa_locations,
        page_size=SWA_PAGE_SIZE,
        type="flashmla",
    )
    swa_decoded = dequantize_k_cache_paged(
        swa_cache, swa_locations, SWA_PAGE_SIZE
    )[:, 0]

    extra_values = torch.randn(
        (MAX_C128_TOKENS, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    num_extra_pages = MAX_C128_TOKENS // C128_PAGE_SIZE
    extra_fp8 = torch.empty(
        (num_extra_pages, _fp8_page_bytes(C128_PAGE_SIZE)),
        device=device,
        dtype=torch.uint8,
    )
    extra_bf16 = torch.empty(
        (num_extra_pages, C128_PAGE_SIZE * HEAD_DIM * 2),
        device=device,
        dtype=torch.uint8,
    )
    extra_locations = torch.arange(
        MAX_C128_TOKENS, device=device, dtype=torch.int32
    )
    fused_store_cache(
        extra_values,
        extra_fp8,
        extra_locations,
        page_size=C128_PAGE_SIZE,
        type="flashmla",
    )
    extra_bf16.view(torch.bfloat16).reshape(
        num_extra_pages, C128_PAGE_SIZE, HEAD_DIM
    ).copy_(extra_values.view(num_extra_pages, C128_PAGE_SIZE, HEAD_DIM))

    q = torch.randn(
        (batch_size, NUM_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    swa_indices = torch.arange(
        SWA_PAGE_SIZE, device=device, dtype=torch.int32
    ).expand(batch_size, -1)
    swa_lens = torch.full(
        (batch_size,), SWA_PAGE_SIZE, device=device, dtype=torch.int32
    )
    extra_indices = torch.full(
        (batch_size, MAX_C128_TOKENS), -1, device=device, dtype=torch.int32
    )
    extra_indices[:, :extra_length] = torch.arange(
        extra_length, device=device, dtype=torch.int32
    )
    extra_lens = torch.full(
        (batch_size,), extra_length, device=device, dtype=torch.int32
    )
    sink = torch.linspace(-0.5, 0.5, NUM_HEADS, device=device, dtype=torch.float32)
    out_fp8 = torch.empty_like(q)
    out_mixed = torch.empty_like(q)

    return {
        "q": q,
        "swa_cache": swa_cache,
        "swa_decoded": swa_decoded,
        "swa_indices": swa_indices,
        "swa_lens": swa_lens,
        "extra_fp8": extra_fp8,
        "extra_bf16": extra_bf16,
        "extra_values": extra_values,
        "extra_indices": extra_indices,
        "extra_lens": extra_lens,
        "sink": sink,
        "out_fp8": out_fp8,
        "out_mixed": out_mixed,
    }


def _run_case(
    batch_size: int,
    extra_length: int,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    case = _build_case(batch_size, extra_length, seed=301 + batch_size + extra_length)
    scale = HEAD_DIM**-0.5

    def fp8() -> None:
        decode_sparse_attention_triton(
            q=case["q"],
            swa_cache=case["swa_cache"],
            swa_indices=case["swa_indices"],
            swa_lens=case["swa_lens"],
            scale=scale,
            attn_sink=case["sink"],
            out=case["out_fp8"],
            extra_cache=case["extra_fp8"],
            extra_indices=case["extra_indices"],
            extra_lens=case["extra_lens"],
            swa_block_size=SWA_PAGE_SIZE,
            extra_block_size=C128_PAGE_SIZE,
        )

    def mixed() -> None:
        decode_sparse_attention_fp8_swa_bf16_extra(
            q=case["q"],
            swa_cache=case["swa_cache"],
            swa_indices=case["swa_indices"],
            swa_lens=case["swa_lens"],
            scale=scale,
            attn_sink=case["sink"],
            out=case["out_mixed"],
            extra_cache=case["extra_bf16"],
            extra_indices=case["extra_indices"],
            extra_lens=case["extra_lens"],
            swa_block_size=SWA_PAGE_SIZE,
            extra_block_size=C128_PAGE_SIZE,
        )

    fp8()
    mixed()
    torch.cuda.synchronize()
    eager_fp8 = _timings(fp8, warmup_ms, rep_ms)
    eager_mixed = _timings(mixed, warmup_ms, rep_ms)
    fp8_graph = _capture(fp8)
    mixed_graph = _capture(mixed)
    graph_fp8 = _timings(fp8_graph.replay, warmup_ms, rep_ms)
    graph_mixed = _timings(mixed_graph.replay, warmup_ms, rep_ms)

    fp8()
    mixed()
    torch.cuda.synchronize()
    difference = (
        case["out_mixed"].float() - case["out_fp8"].float()
    ).abs()
    selected = torch.cat(
        (case["extra_values"][:extra_length], case["swa_decoded"]), dim=0
    ).float()
    logits = torch.einsum("bhd,nd->bhn", case["q"].float(), selected) * scale
    logits = torch.cat(
        (
            case["sink"].float().view(1, NUM_HEADS, 1).expand(batch_size, -1, -1),
            logits,
        ),
        dim=-1,
    )
    probabilities = logits.softmax(dim=-1)[..., 1:]
    reference = torch.einsum("bhn,nd->bhd", probabilities, selected)
    fp8_error = (case["out_fp8"].float() - reference).abs()
    mixed_error = (case["out_mixed"].float() - reference).abs()
    result = {
        "batch_size": batch_size,
        "swa_length": SWA_PAGE_SIZE,
        "extra_length": extra_length,
        "extra_index_width": MAX_C128_TOKENS,
        "eager": {"fp8": eager_fp8, "mixed": eager_mixed},
        "cuda_graph": {"fp8": graph_fp8, "mixed": graph_mixed},
        "eager_speedup": eager_fp8["median_ms"] / eager_mixed["median_ms"],
        "cuda_graph_speedup": graph_fp8["median_ms"] / graph_mixed["median_ms"],
        "output_delta_vs_fp8": {
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        },
        "error_vs_bf16_extra_reference": {
            "fp8_max_abs": float(fp8_error.max().item()),
            "fp8_mean_abs": float(fp8_error.mean().item()),
            "mixed_max_abs": float(mixed_error.max().item()),
            "mixed_mean_abs": float(mixed_error.mean().item()),
        },
    }
    del fp8_graph, mixed_graph
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--rep-ms", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("NVIDIA CUDA is required")
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 6):
        raise SystemExit(f"exact SM86 is required, got SM{capability[0]}{capability[1]}")
    prime_e4m3fn_decode_lut(torch.device("cuda", 0))

    started = time.time()
    cases = [
        _run_case(batch, extra, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
        for batch in (1, 5)
        for extra in (21, 4096)
    ]
    c128_tokens = CONTEXT_TOKENS // 128
    num_pages = (c128_tokens + C128_PAGE_SIZE + 1) // C128_PAGE_SIZE
    fp8_page_bytes = _fp8_page_bytes(C128_PAGE_SIZE)
    bf16_page_bytes = C128_PAGE_SIZE * HEAD_DIM * 2
    receipt = {
        "schema": "dsv4_selective_c128_bf16_sm86_benchmark_v1",
        "timestamp_unix": started,
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(capability),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "geometry": {
            "heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "swa_page_size": SWA_PAGE_SIZE,
            "c128_page_size": C128_PAGE_SIZE,
            "c128_layers_per_tp_rank": NUM_C128_LAYERS,
            "context_tokens": CONTEXT_TOKENS,
        },
        "storage": {
            "fp8_c128_page_bytes": fp8_page_bytes,
            "bf16_c128_page_bytes": bf16_page_bytes,
            "c128_pages_including_allocator_padding": num_pages,
            "per_gpu_delta_bytes": num_pages
            * (bf16_page_bytes - fp8_page_bytes)
            * NUM_C128_LAYERS,
        },
        "cases": cases,
        "summary": {
            "median_cuda_graph_speedup": statistics.median(
                float(case["cuda_graph_speedup"]) for case in cases
            ),
            "minimum_cuda_graph_speedup": min(
                float(case["cuda_graph_speedup"]) for case in cases
            ),
        },
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
