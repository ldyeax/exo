#!/usr/bin/env python3
"""Benchmark the graph-safe DeepSeek-V4 tiny-row routing specialization."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from sglang.srt.layers.quantization import v4_triton_kernels_moe as v4_moe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-counts", default="14,22")
    parser.add_argument("--rows", default="1,2,3,4,5,6")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def make_routes(rows: int, expert_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    route_ids = torch.full((rows, 6), -1, dtype=torch.int32, device="cuda")
    route_weights = torch.empty((rows, 6), dtype=torch.float32, device="cuda")
    weights = torch.arange(1, 7, dtype=torch.float32, device="cuda")
    weights /= weights.sum()
    for row in range(rows):
        valid = min(6, expert_count)
        route_ids[row, :valid] = (
            torch.arange(valid, dtype=torch.int32, device="cuda") + row * 3
        ) % expert_count
        route_weights[row] = weights.roll(row)
    return route_ids, route_weights


def capture(function: Callable[[], Any]) -> tuple[torch.cuda.CUDAGraph, Any]:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_replay(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    gpu_microseconds: list[float] = []
    wall_microseconds: list[float] = []
    for _ in range(samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start_event.record()
        for _ in range(iterations):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_microseconds.append(start_event.elapsed_time(end_event) * 1000 / iterations)
        wall_microseconds.append((wall_end - wall_start) / 1000 / iterations)
    return {
        "gpu_median_us": statistics.median(gpu_microseconds),
        "gpu_min_us": min(gpu_microseconds),
        "wall_median_us": statistics.median(wall_microseconds),
        "wall_min_us": min(wall_microseconds),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("this benchmark is qualified only on SM86")
    expert_counts = [int(value) for value in args.expert_counts.split(",")]
    row_counts = [int(value) for value in args.rows.split(",")]
    os.environ.pop(v4_moe._SMALL_ROW_ROUTING_ENV, None)

    results: list[dict[str, Any]] = []
    for expert_count in expert_counts:
        if not 1 <= expert_count <= v4_moe._SMALL_ROW_ROUTING_MAX_EXPERTS:
            raise ValueError(f"expert count outside router admission: {expert_count}")
        for rows in row_counts:
            route_ids, route_weights = make_routes(rows, expert_count)
            baseline_graph, baseline_output = capture(
                lambda route_ids=route_ids,
                route_weights=route_weights,
                expert_count=expert_count: (
                    v4_moe._make_routing_data_v4(route_ids, route_weights, expert_count)
                )
            )
            baseline = measure_replay(
                baseline_graph,
                warmup=args.warmup,
                iterations=args.iterations,
                samples=args.samples,
            )
            del baseline_output, baseline_graph
            torch.cuda.empty_cache()

            candidate_graph, candidate_output = capture(
                lambda route_ids=route_ids,
                route_weights=route_weights,
                expert_count=expert_count: (
                    v4_moe._make_small_row_routing_data_v4(
                        route_ids, route_weights, expert_count
                    )
                )
            )
            candidate = measure_replay(
                candidate_graph,
                warmup=args.warmup,
                iterations=args.iterations,
                samples=args.samples,
            )
            del candidate_output, candidate_graph
            torch.cuda.empty_cache()

            results.append(
                {
                    "expert_count": expert_count,
                    "rows": rows,
                    "baseline": baseline,
                    "candidate": candidate,
                    "gpu_speedup": (
                        baseline["gpu_median_us"] / candidate["gpu_median_us"]
                    ),
                    "wall_speedup": (
                        baseline["wall_median_us"] / candidate["wall_median_us"]
                    ),
                }
            )

    receipt = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "results": results,
    }
    serialized = json.dumps(receipt, indent=2)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
