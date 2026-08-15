#!/usr/bin/env python3
"""Tune the real DeepSeek-V4 MXFP4 resident-GPU MoE path on SM86.

This benchmark deliberately exercises the complete production operation:
top-k routing, both simulated-MXFP4 ``matmul_ogs`` calls, the 2604B clamp,
SwiGLU, weighted scatter/reduction, and caller-owned output.  Each candidate is
captured in a CUDA graph, replayed with one through three live local routes per
row, and compared with the package-default output for the same checkpoint
weights and inputs.

The package does not expose block-N as a public constraint.  The benchmark
therefore scopes a monkey-patch of its block-N heuristic to each capture.  No
installed package or model file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from sglang.srt.layers.quantization import v4_triton_kernels_moe as v4_moe


@dataclass(frozen=True)
class KernelConfig:
    block_n: int | None
    split_k: int | None
    num_stages: int | None

    @property
    def name(self) -> str:
        def show(value: int | None) -> str:
            return "default" if value is None else str(value)

        return (
            f"bn{show(self.block_n)}-sk{show(self.split_k)}-st{show(self.num_stages)}"
        )


@dataclass(frozen=True)
class ExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    gate_scale: torch.Tensor
    up_scale: torch.Tensor
    down_scale: torch.Tensor


def parse_integer_or_default_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        value = None if item == "default" else int(item)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("configuration list may not be empty")
    return values


def parse_integer_list(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list may not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"integer list contains duplicates: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/tmp/dsv4-local-checkpoint-0731"),
    )
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--expert-counts", default="14,22")
    parser.add_argument("--rows", default="1,2,3,4,5,6")
    parser.add_argument("--live-routes", default="1,2,3")
    parser.add_argument("--block-n", default="64,128,256")
    parser.add_argument("--split-k", default="1,2,default")
    parser.add_argument("--num-stages", default="2,3,4")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument(
        "--fused-t5-moe",
        action="store_true",
        help=(
            "Benchmark the interleaved W13 fused-activation path. The flag "
            "controls both weight conversion and execution so a mixed layout "
            "cannot be measured accidentally."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the full receipt only to --output.",
    )
    return parser.parse_args()


def load_experts(
    model_path: Path,
    *,
    layer: int,
    expert_count: int,
) -> list[ExpertWeights]:
    with (model_path / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]

    names_by_file: dict[str, list[tuple[int, str, str]]] = {}
    for expert in range(expert_count):
        prefix = f"layers.{layer}.ffn.experts.{expert}"
        names = {
            "gate": f"{prefix}.w1.weight",
            "up": f"{prefix}.w3.weight",
            "down": f"{prefix}.w2.weight",
            "gate_scale": f"{prefix}.w1.scale",
            "up_scale": f"{prefix}.w3.scale",
            "down_scale": f"{prefix}.w2.scale",
        }
        for field, name in names.items():
            names_by_file.setdefault(weight_map[name], []).append((expert, field, name))

    tensors: list[dict[str, torch.Tensor]] = [{} for _ in range(expert_count)]
    for filename, entries in names_by_file.items():
        with safe_open(model_path / filename, framework="pt", device="cpu") as handle:
            for expert, field, name in entries:
                tensors[expert][field] = handle.get_tensor(name).contiguous()
    return [ExpertWeights(**expert) for expert in tensors]


def convert_experts(experts: list[ExpertWeights]) -> tuple[Any, Any, Any, Any]:
    w13 = torch.stack(
        [torch.cat((expert.gate, expert.up), dim=0) for expert in experts]
    ).cuda()
    w2 = torch.stack([expert.down for expert in experts]).cuda()
    w13_scale = torch.stack(
        [torch.cat((expert.gate_scale, expert.up_scale), dim=0) for expert in experts]
    ).cuda()
    w2_scale = torch.stack([expert.down_scale for expert in experts]).cuda()
    return v4_moe.convert_v4_weights_to_triton_kernels(
        w13,
        w13_scale,
        w2,
        w2_scale,
    )


def fill_routes(
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    expert_count: int,
    live_routes: int,
) -> None:
    rows, top_k = route_ids.shape
    ids = torch.full(
        (rows, top_k),
        -1,
        dtype=torch.int32,
        device=route_ids.device,
    )
    # Adjacent verification rows usually overlap on popular experts.  This
    # pattern retains one shared expert and adds deterministic row-specific
    # experts, rather than unrealistically routing every row to disjoint sets.
    for row in range(rows):
        ids[row, 0] = 0
        for column in range(1, live_routes):
            ids[row, column] = (row * 3 + column * 5) % expert_count
    base_weights = torch.tensor(
        [0.31, 0.23, 0.17, 0.13, 0.09, 0.07],
        dtype=torch.float32,
        device=route_ids.device,
    )
    weights = torch.stack([base_weights.roll(row) for row in range(rows)])
    route_ids.copy_(ids)
    route_weights.copy_(weights)


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_outputs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    candidate_float = candidate.float()
    reference_float = reference.float()
    difference = (candidate_float - reference_float).abs()
    reference_abs_mean = reference_float.abs().mean().clamp_min(1e-12)
    relative_mean = difference.mean() / reference_abs_mean
    cosine = torch.nn.functional.cosine_similarity(
        candidate_float.flatten(),
        reference_float.flatten(),
        dim=0,
    )
    return {
        "mean_abs_difference": difference.mean().item(),
        "max_abs_difference": difference.max().item(),
        "relative_mean_difference": relative_mean.item(),
        "cosine_similarity": cosine.item(),
        "passed": bool(cosine.item() >= 0.999 and relative_mean.item() <= 0.01),
    }


def measure_graph(
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
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_begin = time.perf_counter_ns()
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_microseconds.append(begin.elapsed_time(end) * 1000 / iterations)
        wall_microseconds.append((wall_end - wall_begin) / 1000 / iterations)
    return {
        "gpu_median_us": statistics.median(gpu_microseconds),
        "gpu_min_us": min(gpu_microseconds),
        "gpu_max_us": max(gpu_microseconds),
        "wall_median_us": statistics.median(wall_microseconds),
        "wall_min_us": min(wall_microseconds),
    }


@contextmanager
def package_kernel_config(config: KernelConfig) -> Iterator[None]:
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags

    original_compute_block_n = opt_flags.opt_flags_nvidia.compute_block_n
    original_constraints = dict(opt_flags._opt_flags_constraints)
    try:
        opt_flags._opt_flags_constraints.clear()
        opt_flags._opt_flags_constraints["is_persistent"] = False
        if config.split_k is not None:
            opt_flags._opt_flags_constraints["split_k"] = config.split_k
        if config.num_stages is not None:
            opt_flags._opt_flags_constraints["num_stages"] = config.num_stages
        if config.block_n is not None:
            block_n = config.block_n

            def fixed_block_n(_n: int, _arch: Any, _precision: Any) -> int:
                return block_n

            opt_flags.opt_flags_nvidia.compute_block_n = fixed_block_n
        yield
    finally:
        opt_flags.opt_flags_nvidia.compute_block_n = original_compute_block_n
        opt_flags._opt_flags_constraints.clear()
        opt_flags._opt_flags_constraints.update(original_constraints)


@contextmanager
def record_opt_flags(records: list[dict[str, Any]]) -> Iterator[None]:
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags

    original = opt_flags.make_default_opt_flags_nvidia

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        entry = {
            "m": args[4],
            "n": args[5],
            "k": args[6],
            **asdict(result),
        }
        if entry not in records:
            records.append(entry)
        return result

    opt_flags.make_default_opt_flags_nvidia = wrapped
    try:
        yield
    finally:
        opt_flags.make_default_opt_flags_nvidia = original


def capture_candidate(
    function: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    # Warm eager calls compile both Triton GEMMs before graph capture.
    output = function()
    output = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def make_configs(args: argparse.Namespace) -> list[KernelConfig]:
    configs = [KernelConfig(None, None, None)]
    for block_n in parse_integer_or_default_list(args.block_n):
        for split_k in parse_integer_or_default_list(args.split_k):
            for num_stages in parse_integer_or_default_list(args.num_stages):
                config = KernelConfig(block_n, split_k, num_stages)
                if config not in configs:
                    configs.append(config)
    return configs


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("this benchmark is qualified only on exact SM86")
    if args.warmup < 0 or args.iterations < 1 or args.samples < 1:
        raise ValueError("invalid timing counts")

    expert_counts = parse_integer_list(args.expert_counts)
    rows_values = parse_integer_list(args.rows)
    live_route_values = parse_integer_list(args.live_routes)
    if any(count < 6 or count > 22 for count in expert_counts):
        raise ValueError("expert counts must be in [6, 22]")
    if any(rows < 1 or rows > 6 for rows in rows_values):
        raise ValueError("row counts must be in [1, 6]")
    if any(routes < 1 or routes > 3 for routes in live_route_values):
        raise ValueError("live route counts must be in [1, 3]")

    os.environ[v4_moe._SMALL_ROW_ROUTING_ENV] = "1"
    os.environ[v4_moe._SM86_FUSED_T5_MOE_ENV] = "1" if args.fused_t5_moe else "0"
    # Apply the package's architecture patch before wrapping its flag chooser.
    v4_moe._patch_strided_mxfp()
    torch.manual_seed(args.seed)
    configs = make_configs(args)
    results: list[dict[str, Any]] = []

    for expert_count in expert_counts:
        experts = load_experts(
            args.model,
            layer=args.layer,
            expert_count=expert_count,
        )
        intermediate_size = experts[0].gate.shape[0]
        hidden_size = experts[0].gate.shape[1] * 2
        converted = convert_experts(experts)
        del experts

        for rows in rows_values:
            hidden_states = (
                torch.randn(
                    rows,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                * 0.25
            ).contiguous()
            caller_output = torch.empty_like(hidden_states)
            route_ids = torch.empty(
                (rows, 6),
                dtype=torch.int32,
                device="cuda",
            )
            route_weights = torch.empty(
                (rows, 6),
                dtype=torch.float32,
                device="cuda",
            )

            def apply(
                hidden_states: torch.Tensor = hidden_states,
                converted_weights: tuple[Any, Any, Any, Any] = converted,
                route_weights: torch.Tensor = route_weights,
                route_ids: torch.Tensor = route_ids,
                intermediate_size: int = intermediate_size,
                expert_count: int = expert_count,
                caller_output: torch.Tensor = caller_output,
            ) -> torch.Tensor:
                return v4_moe.apply_v4_triton_kernels_moe(
                    hidden_states=hidden_states,
                    w13_swiz=converted_weights[0],
                    w13_pcg=converted_weights[1],
                    w2_swiz=converted_weights[2],
                    w2_pcg=converted_weights[3],
                    topk_weights=route_weights,
                    topk_ids=route_ids,
                    intermediate_size=intermediate_size,
                    num_experts=expert_count,
                    swiglu_limit=args.swiglu_limit,
                    caller_output=caller_output,
                    fused_t5_moe=args.fused_t5_moe,
                )

            references: dict[int, torch.Tensor] = {}
            with package_kernel_config(KernelConfig(None, None, None)):
                for live_routes in live_route_values:
                    fill_routes(
                        route_ids,
                        route_weights,
                        expert_count=expert_count,
                        live_routes=live_routes,
                    )
                    references[live_routes] = apply().detach().cpu().clone()
                torch.cuda.synchronize()

            for config in configs:
                result: dict[str, Any] = {
                    "expert_count": expert_count,
                    "rows": rows,
                    "config": asdict(config),
                    "config_name": config.name,
                }
                observed_flags: list[dict[str, Any]] = []
                try:
                    fill_routes(
                        route_ids,
                        route_weights,
                        expert_count=expert_count,
                        live_routes=live_route_values[0],
                    )
                    with (
                        package_kernel_config(config),
                        record_opt_flags(observed_flags),
                    ):
                        graph, output = capture_candidate(apply)
                    result["observed_opt_flags"] = observed_flags
                    route_results: list[dict[str, Any]] = []
                    for live_routes in live_route_values:
                        fill_routes(
                            route_ids,
                            route_weights,
                            expert_count=expert_count,
                            live_routes=live_routes,
                        )
                        graph.replay()
                        torch.cuda.synchronize()
                        candidate = output.detach().cpu().clone()
                        first_hash = tensor_sha256(candidate)
                        graph.replay()
                        torch.cuda.synchronize()
                        second_hash = tensor_sha256(output)
                        comparison = compare_outputs(
                            candidate,
                            references[live_routes],
                        )
                        timing = measure_graph(
                            graph,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            samples=args.samples,
                        )
                        route_results.append(
                            {
                                "live_routes_per_row": live_routes,
                                "comparison": comparison,
                                "graph_replay_exact": first_hash == second_hash,
                                "output_sha256": first_hash,
                                "timing": timing,
                            }
                        )
                    result["routes"] = route_results
                    result["passed"] = all(
                        route["comparison"]["passed"] and route["graph_replay_exact"]
                        for route in route_results
                    )
                    del graph, output
                except Exception as error:
                    torch.cuda.synchronize()
                    result.update(
                        {
                            "passed": False,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "observed_opt_flags": observed_flags,
                        }
                    )
                results.append(result)
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    receipt = {
        "schema_version": 1,
        "benchmark": "dsv4_mxfp4_small_batch_gemm",
        "host": platform.node(),
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "triton_version": __import__("triton").__version__,
        "seed": args.seed,
        "swiglu_limit": args.swiglu_limit,
        "fused_t5_moe": args.fused_t5_moe,
        "expert_counts": expert_counts,
        "rows": rows_values,
        "live_routes_per_row": live_route_values,
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "configs": [asdict(config) for config in configs],
        "results": results,
    }
    serialized = json.dumps(receipt, indent=2)
    if not args.quiet:
        print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
