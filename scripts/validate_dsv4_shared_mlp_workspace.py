#!/usr/bin/env python3
"""Validate the production DSV4 shared-MLP alias schedule on SM86."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sglang.jit_kernel.dsv4 import silu_and_mul_clamp
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    apply_fp8_marlin_linear_into,
    prepare_fp8_layer_for_marlin,
)


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _prepare_marlin_layer(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_size: int,
    output_size: int,
) -> SimpleNamespace:
    layer = SimpleNamespace(
        input_size_per_partition=input_size,
        output_size_per_partition=output_size,
        orig_dtype=torch.bfloat16,
        weight_block_size=[128, 128],
        weight=torch.nn.Parameter(weight.cuda(), requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(scale.cuda(), requires_grad=False),
        input_scale=None,
        bias=None,
    )
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


def _run_linear(
    layer: SimpleNamespace,
    input_tensor: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    arguments = {
        "input": input_tensor,
        "weight": layer.weight,
        "weight_scale": layer.weight_scale,
        "workspace": layer.workspace,
        "size_n": layer.output_size_per_partition,
        "size_k": layer.input_size_per_partition,
        "bias": None,
    }
    if output is None:
        return apply_fp8_marlin_linear(**arguments)
    return apply_fp8_marlin_linear_into(**arguments, output=output)


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-shard",
        type=Path,
        default=Path(
            "/mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark/"
            "model-00002-of-00066.safetensors"
        ),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_size = 7168
    intermediate_size = 3072
    gate_up_size = 2 * intermediate_size
    prefix = f"layers.{args.layer}.ffn.shared_experts"

    gate_weight = torch.cat(
        [
            _load_tensor(args.checkpoint_shard, f"{prefix}.w1.weight"),
            _load_tensor(args.checkpoint_shard, f"{prefix}.w3.weight"),
        ],
        dim=0,
    )
    gate_scale = torch.cat(
        [
            _load_tensor(args.checkpoint_shard, f"{prefix}.w1.scale"),
            _load_tensor(args.checkpoint_shard, f"{prefix}.w3.scale"),
        ],
        dim=0,
    )
    down_weight = _load_tensor(args.checkpoint_shard, f"{prefix}.w2.weight")
    down_scale = _load_tensor(args.checkpoint_shard, f"{prefix}.w2.scale")

    gate_layer = _prepare_marlin_layer(
        gate_weight,
        gate_scale,
        input_size=hidden_size,
        output_size=gate_up_size,
    )
    down_layer = _prepare_marlin_layer(
        down_weight,
        down_scale,
        input_size=intermediate_size,
        output_size=hidden_size,
    )
    del gate_weight, gate_scale, down_weight, down_scale

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    input_tensor = torch.randn(
        (args.rows, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    gate_reference = _run_linear(gate_layer, input_tensor)
    activation_reference = torch.empty(
        (args.rows, intermediate_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    silu_and_mul_clamp(gate_reference, activation_reference, 10.0)
    final_reference = _run_linear(down_layer, activation_reference)

    workspace_bytes = 512 * 1024 * 1024
    workspace = torch.empty(
        workspace_bytes // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate_elements = args.rows * gate_up_size
    down_elements = args.rows * hidden_size
    activation_elements = args.rows * intermediate_size
    activation_offset = max(gate_elements, down_elements)
    live_elements = activation_offset + activation_elements
    if live_elements > workspace.numel():
        raise ValueError("shared MLP alias schedule exceeds the Q workspace")

    gate_output = workspace[:gate_elements].view(args.rows, gate_up_size)
    activation_output = workspace[
        activation_offset : activation_offset + activation_elements
    ].view(args.rows, intermediate_size)
    down_output = workspace[:down_elements].view(args.rows, hidden_size)

    gate_result = _run_linear(gate_layer, input_tensor, gate_output)
    torch.cuda.synchronize()
    gate_equal = torch.equal(gate_reference, gate_result)
    gate_pointer_preserved = gate_result.data_ptr() == gate_output.data_ptr()

    silu_and_mul_clamp(gate_result, activation_output, 10.0)
    final_result = _run_linear(down_layer, activation_output, down_output)
    torch.cuda.synchronize()
    final_equal = torch.equal(final_reference, final_result)
    final_pointer_preserved = final_result.data_ptr() == down_output.data_ptr()

    # Repeat after all kernels have compiled so the allocation delta measures
    # only the steady caller-owned dataflow.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocation_before = torch.cuda.memory_allocated()
    gate_result = _run_linear(gate_layer, input_tensor, gate_output)
    silu_and_mul_clamp(gate_result, activation_output, 10.0)
    final_result = _run_linear(down_layer, activation_output, down_output)
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated()
    peak_allocation = torch.cuda.max_memory_allocated()

    report = {
        "rows": args.rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "workspace_bytes": workspace_bytes,
        "gate_output_bytes": gate_elements * torch.bfloat16.itemsize,
        "activation_offset_bytes": activation_offset * torch.bfloat16.itemsize,
        "activation_output_bytes": activation_elements * torch.bfloat16.itemsize,
        "down_output_bytes": down_elements * torch.bfloat16.itemsize,
        "live_workspace_bytes": live_elements * torch.bfloat16.itemsize,
        "gate_bitwise_equal": gate_equal,
        "final_bitwise_equal": final_equal,
        "gate_pointer_preserved": gate_pointer_preserved,
        "final_pointer_preserved": final_pointer_preserved,
        "steady_allocated_delta_bytes": allocation_after - allocation_before,
        "steady_peak_delta_bytes": peak_allocation - allocation_before,
        "gate_sha256": _sha256(gate_reference),
        "final_sha256": _sha256(final_reference),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not all(
        (
            gate_equal,
            final_equal,
            gate_pointer_preserved,
            final_pointer_preserved,
            allocation_after == allocation_before,
        )
    ):
        raise SystemExit("shared MLP workspace coherence gate failed")


if __name__ == "__main__":
    main()
