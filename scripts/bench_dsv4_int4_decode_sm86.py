"""Microbenchmark the production SM86 DeepSeek-V4 sparse decode paths.

The benchmark intentionally uses fixed, zero-filled cache payloads: it measures
the same memory transactions and decode instructions without conflating kernel
timing with cache construction.  Run once per requested batch/compression shape.
"""

import argparse
import math

import torch
import triton
from sglang.kernels.ops.attention.dsv4.bf16_decode import (
    decode_sparse_attention_bf16,
)
from sglang.kernels.ops.attention.dsv4.fp8_storage import prime_e4m3fn_decode_lut
from sglang.kernels.ops.attention.dsv4.int4_decode import (
    decode_sparse_attention_int4,
)
from sglang.kernels.ops.attention.dsv4.int4_storage import int4_main_page_bytes
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    decode_sparse_attention_triton,
)


def fp8_page_bytes(page_size: int) -> int:
    return math.ceil(page_size * 584 / 576) * 576


def bench_case(batch: int, extra_topk: int, extra_page_size: int) -> None:
    device = torch.device("cuda")
    heads = 16
    dim = 512
    swa_topk = 128
    swa_page_size = 128
    swa_pages = math.ceil(swa_topk / swa_page_size)
    extra_pages = math.ceil(extra_topk / extra_page_size)
    q = torch.randn((batch, heads, dim), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q)
    sink = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32, device=device)
    swa_indices = torch.arange(swa_topk, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    extra_indices = torch.arange(extra_topk, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    swa_lens = torch.full((batch,), swa_topk, dtype=torch.int32, device=device)
    extra_lens = torch.full((batch,), extra_topk, dtype=torch.int32, device=device)
    scale = dim**-0.5

    int4_swa = torch.zeros(
        (swa_pages, int4_main_page_bytes(swa_page_size)),
        dtype=torch.uint8,
        device=device,
    )
    int4_extra = torch.zeros(
        (extra_pages, int4_main_page_bytes(extra_page_size)),
        dtype=torch.uint8,
        device=device,
    )
    fp8_swa = torch.zeros(
        (swa_pages, fp8_page_bytes(swa_page_size)),
        dtype=torch.uint8,
        device=device,
    )
    fp8_extra = torch.zeros(
        (extra_pages, fp8_page_bytes(extra_page_size)),
        dtype=torch.uint8,
        device=device,
    )
    bf16_swa = torch.zeros(
        (swa_pages, swa_page_size * dim), dtype=torch.bfloat16, device=device
    )
    bf16_extra = torch.zeros(
        (extra_pages, extra_page_size * dim), dtype=torch.bfloat16, device=device
    )
    prime_e4m3fn_decode_lut(device)

    def run_int4() -> None:
        decode_sparse_attention_int4(
            q,
            int4_swa,
            swa_indices,
            swa_lens,
            scale,
            sink,
            out,
            swa_page_size,
            int4_extra,
            extra_indices,
            extra_lens,
            extra_page_size,
        )

    def run_fp8() -> None:
        decode_sparse_attention_triton(
            q,
            fp8_swa,
            swa_indices,
            swa_lens,
            scale,
            sink,
            out,
            fp8_extra,
            extra_indices,
            extra_lens,
            swa_block_size=swa_page_size,
            extra_block_size=extra_page_size,
        )

    def run_bf16() -> None:
        decode_sparse_attention_bf16(
            q,
            bf16_swa,
            swa_indices,
            swa_lens,
            scale,
            sink,
            out,
            swa_page_size,
            bf16_extra,
            extra_indices,
            extra_lens,
            extra_page_size,
        )

    timings = {}
    for name, fn in (("int4", run_int4), ("fp8", run_fp8), ("bf16", run_bf16)):
        fn()
        torch.cuda.synchronize()
        timings[name] = triton.testing.do_bench(fn, warmup=500, rep=500)
    ratio = timings["int4"] / timings["fp8"]
    print(
        f"B{batch} extra={extra_topk} page={extra_page_size}: "
        f"INT4={timings['int4']:.6f} ms FP8={timings['fp8']:.6f} ms "
        f"BF16={timings['bf16']:.6f} ms INT4/FP8={ratio:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--extra-topk", type=int, required=True)
    parser.add_argument("--extra-page-size", type=int, required=True)
    args = parser.parse_args()
    bench_case(args.batch, args.extra_topk, args.extra_page_size)
