#!/usr/bin/env python3
"""Compare a native MXFP4 GPU expert against the KTransformers CPU kernel.

This deliberately bypasses the language-model scheduler.  Both backends
receive the same packed E2M1 weights, UE8M0 scales, BF16 activations, compact
expert IDs, and routing weights, making it a focused numerical oracle for the
portable SM86 GPU MoE path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import socket
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors import safe_open

_REQUEST_HEADER = struct.Struct("!4sIIII")
_RESPONSE_HEADER = struct.Struct("!4sII")


@dataclass(frozen=True)
class ExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    gate_scale: torch.Tensor
    up_scale: torch.Tensor
    down_scale: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument(
        "--experts",
        default="351,31,327,148,300,267,268",
        help="Comma-separated global expert IDs, in compact GPU storage order.",
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        help="Use global experts [0, count) instead of --experts.",
    )
    parser.add_argument(
        "--fixed-routes",
        action="store_true",
        help="Route every token to compact experts [0, top-k).",
    )
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=52)
    parser.add_argument("--cpu-forward-repeats", type=int, default=1)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip the independent SM86 comparison for CPU dispatch A/B tests.",
    )
    parser.add_argument(
        "--native-extension",
        type=Path,
        help="Load this native kt_kernel_ext binary for an isolated A/B oracle.",
    )
    parser.add_argument(
        "--sidecar-endpoint",
        help="Also compare against a running native-MXFP4 sidecar at HOST:PORT.",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path(
            "/root/exo/vendor/ktransformers/third_party/sglang/python/sglang/"
            "srt/layers/quantization/v4_triton_kernels_moe.py"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON result to this artifact path.",
    )
    return parser.parse_args()


def load_kernel(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_dsv4_mxfp4_oracle_kernel", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import portable MXFP4 kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_native_extension(path: Path | None) -> Any:
    if path is None:
        from kt_kernel import kt_kernel_ext

        return kt_kernel_ext
    module_name = "kt_kernel_ext"
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(path.resolve()))
    spec = importlib.util.spec_from_file_location(
        module_name, path.resolve(), loader=loader
    )
    if spec is None:
        raise ImportError(f"Cannot load native extension from {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_expert(
    model_path: Path,
    weight_map: dict[str, str],
    *,
    layer: int,
    expert: int,
) -> ExpertWeights:
    prefix = f"layers.{layer}.ffn.experts.{expert}"
    names = {
        "gate": f"{prefix}.w1.weight",
        "up": f"{prefix}.w3.weight",
        "down": f"{prefix}.w2.weight",
        "gate_scale": f"{prefix}.w1.scale",
        "up_scale": f"{prefix}.w3.scale",
        "down_scale": f"{prefix}.w2.scale",
    }
    by_file: dict[str, list[tuple[str, str]]] = {}
    for field, name in names.items():
        by_file.setdefault(weight_map[name], []).append((field, name))

    tensors: dict[str, torch.Tensor] = {}
    for filename, fields in by_file.items():
        with safe_open(model_path / filename, framework="pt", device="cpu") as handle:
            for field, name in fields:
                tensors[field] = handle.get_tensor(name).contiguous()
    return ExpertWeights(**tensors)


def make_cpu_output(
    native_extension: Any,
    experts: list[ExpertWeights],
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    layer: int,
    cpu_threads: int,
    forward_repeats: int,
    swiglu_limit: float,
) -> tuple[torch.Tensor, list[float], list[str]]:
    hidden_size = experts[0].gate.shape[1] * 2
    intermediate_size = experts[0].gate.shape[0]
    worker_config = native_extension.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [0]
    worker_config.subpool_thread_count = [cpu_threads]
    cpu_infer = native_extension.CPUInfer(worker_config)
    config = native_extension.moe.MOEConfig(
        len(experts),
        topk_ids.shape[1],
        hidden_size,
        intermediate_size,
        0,
    )
    config.layer_idx = layer
    config.max_len = hidden_states.shape[0]
    config.pool = cpu_infer.backend_
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.swiglu_limit = swiglu_limit
    config.gate_projs = [[expert.gate.data_ptr() for expert in experts]]
    config.up_projs = [[expert.up.data_ptr() for expert in experts]]
    config.down_projs = [[expert.down.data_ptr() for expert in experts]]
    config.gate_scales = [[expert.gate_scale.data_ptr() for expert in experts]]
    config.up_scales = [[expert.up_scale.data_ptr() for expert in experts]]
    config.down_scales = [[expert.down_scale.data_ptr() for expert in experts]]

    moe = native_extension.moe.AMXFP4_KGroup_MOE(config)
    physical_to_logical = torch.arange(len(experts), dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()

    batch_sizes = torch.tensor([hidden_states.shape[0]], dtype=torch.int32)
    # The pybind task accepts only a pointer; the native MoE reads int64_t
    # expert IDs, matching KTEPWrapper's pinned torch.long staging buffer.
    topk_ids_int64 = topk_ids.to(torch.int64).contiguous()
    topk_weights_float32 = topk_weights.to(torch.float32).contiguous()
    output = torch.empty_like(hidden_states)
    elapsed_milliseconds = []
    output_sha256s = []
    for _ in range(forward_repeats):
        start_time = time.perf_counter()
        cpu_infer.submit(
            moe.forward_task(
                batch_sizes.data_ptr(),
                topk_ids.shape[1],
                topk_ids_int64.data_ptr(),
                topk_weights_float32.data_ptr(),
                hidden_states.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpu_infer.sync()
        elapsed_milliseconds.append((time.perf_counter() - start_time) * 1000)
        output_sha256s.append(
            hashlib.sha256(output.view(torch.uint8).numpy().tobytes()).hexdigest()
        )
    return output, elapsed_milliseconds, output_sha256s


def make_gpu_output(
    kernel: Any,
    experts: list[ExpertWeights],
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    swiglu_limit: float,
) -> torch.Tensor:
    w13 = torch.stack(
        [torch.cat((expert.gate, expert.up), dim=0) for expert in experts]
    ).cuda()
    w2 = torch.stack([expert.down for expert in experts]).cuda()
    w13_scale = torch.stack(
        [torch.cat((expert.gate_scale, expert.up_scale), dim=0) for expert in experts]
    ).cuda()
    w2_scale = torch.stack([expert.down_scale for expert in experts]).cuda()
    converted = kernel.convert_v4_weights_to_triton_kernels(
        w13,
        w13_scale,
        w2,
        w2_scale,
    )
    return kernel.apply_v4_triton_kernels_moe(
        hidden_states=hidden_states.cuda(),
        w13_swiz=converted[0],
        w13_pcg=converted[1],
        w2_swiz=converted[2],
        w2_pcg=converted[3],
        topk_weights=topk_weights.cuda(),
        topk_ids=topk_ids.cuda(),
        intermediate_size=experts[0].gate.shape[0],
        num_experts=len(experts),
        swiglu_limit=swiglu_limit,
    ).cpu()


def _recv_exact(connection: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("native-MXFP4 sidecar closed the connection")
        offset += received
    return data


def make_sidecar_output(
    endpoint: str,
    hidden_states: torch.Tensor,
    global_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    layer: int,
) -> tuple[torch.Tensor, float]:
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError(
            f"--sidecar-endpoint must have the form HOST:PORT, got {endpoint!r}"
        )
    hidden_states = hidden_states.to(torch.bfloat16).contiguous()
    global_topk_ids = global_topk_ids.to(torch.int64).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    batch_size, hidden_size = hidden_states.shape
    top_k = global_topk_ids.shape[1]
    expected_bytes = batch_size * hidden_size * 2
    start_time = time.perf_counter()
    with socket.create_connection((host, int(port_text)), timeout=120.0) as connection:
        connection.settimeout(1200.0)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.sendall(
            _REQUEST_HEADER.pack(b"KTR1", layer, batch_size, hidden_size, top_k)
        )
        connection.sendall(memoryview(hidden_states.view(torch.uint8).numpy()))
        connection.sendall(memoryview(global_topk_ids.view(torch.uint8).numpy()))
        connection.sendall(memoryview(topk_weights.view(torch.uint8).numpy()))
        magic, status, payload_size = _RESPONSE_HEADER.unpack(
            _recv_exact(connection, _RESPONSE_HEADER.size)
        )
        payload = _recv_exact(connection, payload_size)
    elapsed_milliseconds = (time.perf_counter() - start_time) * 1000
    if magic != b"KTO1":
        raise RuntimeError("native-MXFP4 sidecar returned an invalid response magic")
    if status != 0:
        raise RuntimeError(
            "native-MXFP4 sidecar failed: " + payload.decode("utf-8", errors="replace")
        )
    if payload_size != expected_bytes:
        raise RuntimeError(
            "native-MXFP4 sidecar returned an invalid payload size: "
            f"expected={expected_bytes}, actual={payload_size}"
        )
    return (
        torch.frombuffer(payload, dtype=torch.bfloat16)
        .reshape(batch_size, hidden_size)
        .clone(),
        elapsed_milliseconds,
    )


def main() -> int:
    args = parse_args()
    expert_ids = (
        list(range(args.expert_count))
        if args.expert_count is not None
        else [int(value) for value in args.experts.split(",") if value]
    )
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("Expert IDs must be unique")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not args.fixed_routes and args.top_k > len(expert_ids):
        raise ValueError(
            "--top-k may exceed --experts only with --fixed-routes, which "
            "fills the remaining routes with the -1 sentinel"
        )
    if args.cpu_forward_repeats < 1:
        raise ValueError("--cpu-forward-repeats must be positive")

    with (args.model / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    experts = [
        load_expert(args.model, weight_map, layer=args.layer, expert=expert)
        for expert in expert_ids
    ]
    hidden_size = experts[0].gate.shape[1] * 2
    torch.manual_seed(7)
    hidden_states = (
        torch.randn(args.tokens, hidden_size, dtype=torch.bfloat16) * args.input_scale
    ).contiguous()
    if args.fixed_routes:
        route_ids = torch.arange(args.top_k, dtype=torch.int32)
        route_ids[route_ids >= len(experts)] = -1
        topk_ids = route_ids.repeat(args.tokens, 1)
    else:
        topk_ids = torch.stack(
            [torch.randperm(len(experts))[: args.top_k] for _ in range(args.tokens)]
        ).to(torch.int32)
    topk_weights = torch.rand(args.tokens, args.top_k, dtype=torch.float32).contiguous()
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    cpu_output, cpu_elapsed_milliseconds, cpu_output_sha256s = make_cpu_output(
        load_native_extension(args.native_extension),
        experts,
        hidden_states,
        topk_ids,
        topk_weights,
        layer=args.layer,
        cpu_threads=args.cpu_threads,
        forward_repeats=args.cpu_forward_repeats,
        swiglu_limit=args.swiglu_limit,
    )
    gpu_output = None
    if not args.skip_gpu:
        gpu_output = make_gpu_output(
            load_kernel(args.kernel),
            experts,
            hidden_states,
            topk_ids,
            topk_weights,
            swiglu_limit=args.swiglu_limit,
        )
    sidecar_output = None
    sidecar_elapsed_milliseconds = None
    if args.sidecar_endpoint:
        expert_id_lookup = torch.tensor(expert_ids, dtype=torch.int64)
        valid_topk_ids = topk_ids >= 0
        global_topk_ids = torch.where(
            valid_topk_ids,
            expert_id_lookup[topk_ids.to(torch.int64).clamp_min(0)],
            -1,
        )
        sidecar_output, sidecar_elapsed_milliseconds = make_sidecar_output(
            args.sidecar_endpoint,
            hidden_states,
            global_topk_ids,
            topk_weights,
            layer=args.layer,
        )
    cpu_float = cpu_output.float()
    result = {
        "layer": args.layer,
        "global_expert_ids": expert_ids,
        "tokens": args.tokens,
        "top_k": args.top_k,
        "hidden_size": hidden_size,
        "intermediate_size": experts[0].gate.shape[0],
        "input_scale": args.input_scale,
        "swiglu_limit": args.swiglu_limit,
        "cpu_abs_mean": cpu_float.abs().mean().item(),
        "cpu_forward_repeats": args.cpu_forward_repeats,
        "cpu_forward_min_ms": min(cpu_elapsed_milliseconds),
        "cpu_forward_median_ms": statistics.median(cpu_elapsed_milliseconds),
        "cpu_output_sha256": cpu_output_sha256s[-1],
        "cpu_output_repeat_sha256s": cpu_output_sha256s,
        "cpu_output_repeat_exact": len(set(cpu_output_sha256s)) == 1,
        "cpu_head": cpu_float.flatten()[:8].tolist(),
    }
    gpu_passed = True
    if gpu_output is not None:
        gpu_float = gpu_output.float()
        difference = (gpu_float - cpu_float).abs()
        relative_mean = difference.mean() / cpu_float.abs().mean().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            gpu_float.flatten(), cpu_float.flatten(), dim=0
        )
        row_cosine = torch.nn.functional.cosine_similarity(gpu_float, cpu_float, dim=-1)
        row_relative_mean = difference.mean(dim=-1) / cpu_float.abs().mean(
            dim=-1
        ).clamp_min(1e-12)
        result.update(
            {
                "gpu_abs_mean": gpu_float.abs().mean().item(),
                "gpu_output_sha256": hashlib.sha256(
                    gpu_output.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "mean_abs_difference": difference.mean().item(),
                "max_abs_difference": difference.max().item(),
                "relative_mean_difference": relative_mean.item(),
                "cosine_similarity": cosine.item(),
                "row_relative_mean_difference": row_relative_mean.tolist(),
                "row_cosine_similarity": row_cosine.tolist(),
                "gpu_head": gpu_float.flatten()[:8].tolist(),
            }
        )
        gpu_passed = cosine.item() >= 0.99 and relative_mean.item() <= 0.10
    sidecar_passed = True
    if sidecar_output is not None:
        sidecar_float = sidecar_output.float()
        sidecar_difference = (sidecar_float - cpu_float).abs()
        sidecar_relative_mean = (
            sidecar_difference.mean() / cpu_float.abs().mean().clamp_min(1e-12)
        )
        sidecar_cosine = torch.nn.functional.cosine_similarity(
            sidecar_float.flatten(), cpu_float.flatten(), dim=0
        )
        result.update(
            {
                "sidecar_endpoint": args.sidecar_endpoint,
                "sidecar_elapsed_ms": sidecar_elapsed_milliseconds,
                "sidecar_output_sha256": hashlib.sha256(
                    sidecar_output.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "sidecar_mean_abs_difference": sidecar_difference.mean().item(),
                "sidecar_max_abs_difference": sidecar_difference.max().item(),
                "sidecar_relative_mean_difference": sidecar_relative_mean.item(),
                "sidecar_cosine_similarity": sidecar_cosine.item(),
                "sidecar_head": sidecar_float.flatten()[:8].tolist(),
            }
        )
        sidecar_passed = (
            sidecar_cosine.item() >= 0.99 and sidecar_relative_mean.item() <= 0.10
        )
    serialized_result = json.dumps(result, indent=2)
    print(serialized_result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized_result + "\n")
    return 0 if gpu_passed and sidecar_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
