#!/usr/bin/env python3
"""Benchmark the graph-replayed DeepSeek-V4 Oscar C4 selection pipeline.

The baseline graph runs RoPE, Oscar query rotation / INT2 C4 scoring, and the
exact top-512 transform as separate stages.  The candidate folds RoPE into the
Oscar rotation and statically elides scoring for graph tiers no larger than
top-k.  Exact global top-k remains a synchronization boundary for longer
histories, so those shapes intentionally retain one caller-owned FP32 score
workspace before the existing top-k kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Final

import torch
from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.oscar_int2_c4_indexer import (
    CLIP_MODE,
    HEAD_DIM,
    NUM_HEADS,
    PAGE_SIZE,
    oscar_int2_c4_paged_mqa_logits_triton,
    pack_oscar_int2_c4_pages_reference,
    validate_oscar_int2_c4_calibration,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512

TOP_K: Final = 512


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,5")
    parser.add_argument("--sequence-lengths", default="511,512,513,2694")
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--candidate-mode",
        choices=("hybrid", "fused-rope"),
        default="hybrid",
        help=(
            "hybrid elides static <=top-k work but retains the faster narrow "
            "RoPE kernel for long tiers; fused-rope measures the rejected "
            "combined RoPE + Oscar transform."
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_positive_integers(text: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _make_calibration(device: torch.device):
    generator = torch.Generator().manual_seed(20260804)
    permutation = torch.randperm(HEAD_DIM, generator=generator)
    signs = torch.where(
        torch.arange(HEAD_DIM) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(torch.bfloat16)
    rotation = torch.zeros((HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16)
    rotation[torch.arange(HEAD_DIM), permutation] = signs
    return validate_oscar_int2_c4_calibration(
        rotation.to(device),
        torch.tensor([2.75], dtype=torch.float32, device=device),
        torch.tensor([0.95], dtype=torch.float32, device=device),
        torch.tensor([121], dtype=torch.int16, device=device),
        layer_id=2,
        clip_mode=CLIP_MODE,
        clip_provenance="sm86-fused-c4-pipeline-benchmark",
    )


def _capture(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    function()
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure_alternating(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, object]:
    for _ in range(warmup):
        baseline.replay()
        candidate.replay()
    torch.cuda.synchronize()

    values: dict[str, list[float]] = {"baseline": [], "fused": []}
    providers = {"baseline": baseline, "fused": candidate}
    for sample_index in range(samples):
        order = (
            ("baseline", "fused")
            if sample_index % 2 == 0
            else (
                "fused",
                "baseline",
            )
        )
        for provider in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(iterations):
                providers[provider].replay()
            end.record()
            end.synchronize()
            values[provider].append(begin.elapsed_time(end) / iterations)

    baseline_ms = float(statistics.median(values["baseline"]))
    fused_ms = float(statistics.median(values["fused"]))
    return {
        "baseline_graph_ms": baseline_ms,
        "fused_graph_ms": fused_ms,
        "speedup": baseline_ms / fused_ms,
        "savings_ms": baseline_ms - fused_ms,
        "samples_ms": values,
    }


def _benchmark_shape(
    *,
    rows: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
    candidate_mode: str,
) -> dict[str, object]:
    # The production allocator exposes page-rounded score workspaces.  A tier
    # up through 512 is intentionally fixed at exactly top-k so the candidate
    # can remove both scorer launches at graph construction time.
    max_sequence_length = (
        TOP_K
        if sequence_length <= TOP_K
        else ((sequence_length + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    )
    page_count = max_sequence_length // PAGE_SIZE
    calibration = _make_calibration(device)
    raw_pages = torch.randn(
        (page_count, PAGE_SIZE, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    storage = pack_oscar_int2_c4_pages_reference(raw_pages, calibration)
    raw_query = torch.randn(
        (rows, 1, NUM_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    baseline_query = torch.empty_like(raw_query)
    fused_query = torch.empty_like(raw_query)
    frequencies = torch.polar(
        torch.ones((64, 32), dtype=torch.float32, device=device),
        torch.randn((64, 32), generator=generator, device=device),
    )
    frequencies_real = torch.view_as_real(frequencies).flatten(-2)
    positions = torch.arange(rows, dtype=torch.int64, device=device).mul_(7).add_(3)
    weights = torch.randn(
        (rows, NUM_HEADS), generator=generator, dtype=torch.float32, device=device
    )
    sequence_lengths = torch.full(
        (rows,), sequence_length, dtype=torch.int32, device=device
    )
    page_table = (
        torch.arange(page_count, dtype=torch.int32, device=device)[None, :]
        .expand(rows, -1)
        .contiguous()
    )
    baseline_scores = torch.full(
        (rows, max_sequence_length), -12345.0, dtype=torch.float32, device=device
    )
    fused_scores = torch.full_like(baseline_scores, -12345.0)
    baseline_rotated = torch.empty(
        (rows, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    fused_rotated = torch.full_like(baseline_rotated, -321.0)
    baseline_indices = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
    fused_indices = torch.empty_like(baseline_indices)

    def run_baseline() -> None:
        baseline_query.copy_(raw_query)
        fused_rope_inplace(baseline_query[..., -64:], None, frequencies, positions)
        oscar_int2_c4_paged_mqa_logits_triton(
            baseline_query,
            storage,
            weights,
            sequence_lengths,
            page_table,
            None,
            max_sequence_length,
            False,
            calibration=calibration,
            out=baseline_scores,
            rotated_query_out=baseline_rotated,
        )
        topk_transform_512(
            baseline_scores,
            sequence_lengths,
            page_table,
            baseline_indices,
            PAGE_SIZE,
        )

    def run_fused() -> None:
        fused_query.copy_(raw_query)
        fuse_rope = candidate_mode == "fused-rope"
        if candidate_mode == "hybrid" and max_sequence_length > TOP_K:
            fused_rope_inplace(fused_query[..., -64:], None, frequencies, positions)
        oscar_int2_c4_paged_mqa_logits_triton(
            fused_query,
            storage,
            weights,
            sequence_lengths,
            page_table,
            None,
            max_sequence_length,
            False,
            calibration=calibration,
            out=fused_scores,
            rotated_query_out=fused_rotated,
            freqs_cis_real=frequencies_real if fuse_rope else None,
            positions=positions if fuse_rope else None,
            selection_topk=TOP_K,
        )
        topk_transform_512(
            fused_scores,
            sequence_lengths,
            page_table,
            fused_indices,
            PAGE_SIZE,
        )

    baseline_graph = _capture(run_baseline)
    candidate_graph = _capture(run_fused)
    baseline_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()

    exact_index_order = torch.equal(fused_indices, baseline_indices)
    # TopKKernel does not promise a stable output ordering, including when it
    # sees identical scores in two separately captured graphs.  Sparse
    # attention consumes the selected set, so compare that exact contract.
    torch.testing.assert_close(
        fused_indices.sort(dim=1).values,
        baseline_indices.sort(dim=1).values,
        rtol=0,
        atol=0,
    )
    if sequence_length > TOP_K:
        torch.testing.assert_close(fused_rotated, baseline_rotated, rtol=0, atol=0)
        torch.testing.assert_close(
            fused_scores[:, :sequence_length],
            baseline_scores[:, :sequence_length],
            rtol=0,
            atol=0,
        )
    else:
        if not torch.all(fused_rotated == -321.0).item():
            raise AssertionError("static <=top-k graph unexpectedly rotated its query")
        if not torch.all(fused_scores == -12345.0).item():
            raise AssertionError("static <=top-k graph unexpectedly wrote scores")

    timing = _measure_alternating(
        baseline_graph,
        candidate_graph,
        warmup=warmup,
        iterations=iterations,
        samples=samples,
    )
    return {
        "rows": rows,
        "sequence_length": sequence_length,
        "max_sequence_length": max_sequence_length,
        "static_scorer_elision": max_sequence_length <= TOP_K,
        "candidate_mode": candidate_mode,
        "topk_selection_set_parity": True,
        "exact_topk_order": exact_index_order,
        "exact_long_score_parity": True if sequence_length > TOP_K else None,
        "timing": timing,
    }


def main() -> int:
    args = parse_arguments()
    rows_values = parse_positive_integers(args.rows, name="rows")
    sequence_lengths = parse_positive_integers(
        args.sequence_lengths, name="sequence-lengths"
    )
    if any(rows > 8 for rows in rows_values):
        raise ValueError("rows must be in [1, 8]")
    if args.warmup < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("timing counts are invalid")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) != (8, 6):
        raise RuntimeError("this benchmark is qualified only on exact SM86")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = [
        _benchmark_shape(
            rows=rows,
            sequence_length=sequence_length,
            generator=generator,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            candidate_mode=args.candidate_mode,
        )
        for rows in rows_values
        for sequence_length in sequence_lengths
    ]
    receipt = {
        "schema_version": 1,
        "benchmark": "dsv4_oscar_c4_fused_pipeline",
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "top_k": TOP_K,
        "candidate_mode": args.candidate_mode,
        "rows": list(rows_values),
        "sequence_lengths": list(sequence_lengths),
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "results": results,
    }
    serialized = json.dumps(receipt, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
