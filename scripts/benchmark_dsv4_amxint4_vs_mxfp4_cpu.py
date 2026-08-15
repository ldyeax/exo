#!/usr/bin/env python3
"""Compare DSV4-shaped KTransformers AMXINT4 and native MXFP4 CPU MoE.

The benchmark intentionally starts from one synthesized *native MXFP4* weight
set.  Those packed weights are decoded exactly to BF16 and the same BF16
tensors are passed to the official AMXINT4 online converter.  Consequently the
two backends see equivalent source weights rather than unrelated random byte
buffers.  A route bank spanning more packed weight bytes than socket LLC keeps
the small-batch result representative of sequential DSV4 layers.

Run each extension in a fresh process.  A production-shape invocation is::

  numactl --physcpubind=0-55 --membind=0 \
    python scripts/benchmark_dsv4_amxint4_vs_mxfp4_cpu.py

The final stdout line begins with ``DSV4_AMXINT4_BENCH_JSON=`` so callers can
persist a receipt without mixing native-kernel diagnostics into the JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import kt_kernel_ext
import torch
from safetensors import safe_open

HIDDEN_SIZE: Final = 4096
INTERMEDIATE_SIZE: Final = 2048
TOP_K: Final = 6
GROUP_SIZE: Final = 32
M_VALUES: Final = (1, 2, 3, 4, 5, 6)
E2M1_VALUES: Final = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)
E2M1_THRESHOLDS: Final = torch.tensor(
    (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32
)
BACKEND_CLASSES: Final = {
    "mxfp4": "AMXFP4_KGroup_MOE",
    "amxint4": "AMXInt4_MOE",
    "rawint4": "AMXInt4_KGroup_MOE",
}


@dataclass(frozen=True)
class ProjectionWeights:
    packed: torch.Tensor
    scales: torch.Tensor
    bf16: torch.Tensor


@dataclass(frozen=True)
class SourceWeights:
    gate: ProjectionWeights
    up: ProjectionWeights
    down: ProjectionWeights
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RawInt4Projection:
    packed: torch.Tensor
    scales: torch.Tensor


@dataclass(frozen=True)
class RawInt4Weights:
    gate: RawInt4Projection
    up: RawInt4Projection
    down: RawInt4Projection


@dataclass(frozen=True)
class InputCase:
    batch_size: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    hidden_states: torch.Tensor
    output: torch.Tensor


def _parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[1]) if len(bounds) == 2 else start
        cpus.update(range(start, end + 1))
    return cpus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _validate_host(*, numa_node: int, threads: int) -> dict[str, Any]:
    node_cpus = _parse_cpu_list(
        Path(f"/sys/devices/system/node/node{numa_node}/cpulist").read_text(
            encoding="ascii"
        )
    )
    affinity = set(os.sched_getaffinity(0))
    if not affinity or not affinity.issubset(node_cpus):
        raise RuntimeError(
            f"process affinity {sorted(affinity)} is not confined to NUMA node "
            f"{numa_node} ({sorted(node_cpus)})"
        )
    if len(affinity) != threads:
        raise RuntimeError(
            f"expected exactly {threads} affinity CPUs, observed {len(affinity)}: "
            f"{sorted(affinity)}"
        )
    cpu_flags = set()
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("flags"):
            cpu_flags = set(line.split(":", 1)[1].split())
            break
    required_flags = {"amx_tile", "amx_int8", "amx_bf16", "avx512_bf16"}
    missing = required_flags - cpu_flags
    if missing:
        raise RuntimeError(f"host is missing required CPU flags: {sorted(missing)}")
    return {
        "cpu_model": _cpu_model(),
        "hostname": platform.node(),
        "numa_node": numa_node,
        "worker_threads": threads,
        "process_affinity": sorted(affinity),
        "node_cpu_count_including_smt": len(node_cpus),
        "required_cpu_flags": sorted(required_flags),
    }


def _synthesize_projection(
    *,
    generator: torch.Generator,
    expert_count: int,
    output_features: int,
    input_features: int,
    base_scale_exponent: int,
    row_chunk: int,
) -> ProjectionWeights:
    if input_features % GROUP_SIZE != 0:
        raise ValueError("input features must be divisible by MXFP4 group size")
    packed = torch.empty(
        (expert_count, output_features, input_features // 2), dtype=torch.uint8
    )
    scales = torch.empty(
        (expert_count, output_features, input_features // GROUP_SIZE),
        dtype=torch.uint8,
    )
    bf16 = torch.empty(
        (expert_count, output_features, input_features), dtype=torch.bfloat16
    )
    for expert_id in range(expert_count):
        # Neighboring power-of-two scales make the generated experts distinct
        # while retaining an exactly representable native UE8M0 source.
        scale_exponent = base_scale_exponent + (expert_id % 3) - 1
        scale_byte = scale_exponent + 127
        scale = math.ldexp(1.0, scale_exponent)
        scales[expert_id].fill_(scale_byte)
        for row_start in range(0, output_features, row_chunk):
            row_end = min(row_start + row_chunk, output_features)
            row_count = row_end - row_start
            normalized = torch.randn(
                (row_count, input_features), generator=generator, dtype=torch.float32
            ).mul_(1.7)
            magnitude_code = torch.bucketize(
                normalized.abs(), E2M1_THRESHOLDS, right=False
            ).to(torch.uint8)
            code = magnitude_code + torch.signbit(normalized).to(torch.uint8) * 8
            values = E2M1_VALUES[code.to(torch.int64)].mul_(scale)
            bf16[expert_id, row_start:row_end].copy_(values.to(torch.bfloat16))
            code_pairs = code.reshape(row_count, input_features // 2, 2)
            packed_chunk = code_pairs[..., 0] | (code_pairs[..., 1] << 4)
            packed[expert_id, row_start:row_end].copy_(packed_chunk)
    return ProjectionWeights(
        packed=packed.contiguous(),
        scales=scales.contiguous(),
        bf16=bf16.contiguous(),
    )


def _synthesize_weights(
    *, expert_count: int, seed: int, row_chunk: int
) -> SourceWeights:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gate = _synthesize_projection(
        generator=generator,
        expert_count=expert_count,
        output_features=INTERMEDIATE_SIZE,
        input_features=HIDDEN_SIZE,
        base_scale_exponent=-8,
        row_chunk=row_chunk,
    )
    up = _synthesize_projection(
        generator=generator,
        expert_count=expert_count,
        output_features=INTERMEDIATE_SIZE,
        input_features=HIDDEN_SIZE,
        base_scale_exponent=-8,
        row_chunk=row_chunk,
    )
    down = _synthesize_projection(
        generator=generator,
        expert_count=expert_count,
        output_features=HIDDEN_SIZE,
        input_features=INTERMEDIATE_SIZE,
        base_scale_exponent=-8,
        row_chunk=row_chunk,
    )
    return SourceWeights(
        gate=gate,
        up=up,
        down=down,
        metadata={
            "kind": "synthesized_native_mxfp4",
            "seed": seed,
            "description": "native MXFP4 decoded exactly to shared BF16",
        },
    )


def _dequantize_projection(
    *, packed: torch.Tensor, scales: torch.Tensor, row_chunk: int
) -> torch.Tensor:
    expert_count, output_features, packed_features = packed.shape
    input_features = packed_features * 2
    if scales.shape != (
        expert_count,
        output_features,
        input_features // GROUP_SIZE,
    ):
        raise ValueError(
            f"invalid MXFP4 scale shape {tuple(scales.shape)} for packed shape "
            f"{tuple(packed.shape)}"
        )
    bf16 = torch.empty(
        (expert_count, output_features, input_features), dtype=torch.bfloat16
    )
    for expert_id in range(expert_count):
        for row_start in range(0, output_features, row_chunk):
            row_end = min(row_start + row_chunk, output_features)
            packed_chunk = packed[expert_id, row_start:row_end]
            low = packed_chunk & 0x0F
            high = (packed_chunk >> 4) & 0x0F
            codes = torch.stack((low, high), dim=-1).reshape(
                row_end - row_start, input_features
            )
            decoded = E2M1_VALUES[codes.to(torch.int64)]
            scale_bits = scales[expert_id, row_start:row_end].to(torch.int32) << 23
            scale_values = scale_bits.view(torch.float32).repeat_interleave(
                GROUP_SIZE, dim=-1
            )
            bf16[expert_id, row_start:row_end].copy_(
                decoded.mul_(scale_values).to(torch.bfloat16)
            )
    return bf16.contiguous()


def _tensor_content_sha256(tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        # NumPy does not expose torch.bfloat16. Hash the physical payload so
        # independently built native extensions can be checked for bitwise
        # output parity without a dtype-dependent serialization step.
        if contiguous.dtype == torch.bfloat16:
            contiguous = contiguous.view(torch.uint8)
        array = contiguous.numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_checkpoint_weights(
    *,
    checkpoint: Path,
    layer: int,
    expert_ids: tuple[int, ...],
    row_chunk: int,
) -> SourceWeights:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError(f"checkpoint index has no weight_map: {index_path}")
    projection_names = {"gate": "w1", "up": "w3", "down": "w2"}
    requested: dict[str, tuple[str, str, int]] = {}
    for projection, checkpoint_name in projection_names.items():
        for local_expert_id, checkpoint_expert_id in enumerate(expert_ids):
            for suffix in ("weight", "scale"):
                key = (
                    f"layers.{layer}.ffn.experts.{checkpoint_expert_id}."
                    f"{checkpoint_name}.{suffix}"
                )
                filename = weight_map.get(key)
                if not isinstance(filename, str):
                    raise KeyError(f"checkpoint index is missing {key}")
                requested[key] = (projection, suffix, local_expert_id)

    loaded: dict[tuple[str, str, int], torch.Tensor] = {}
    keys_by_file: dict[str, list[str]] = {}
    for key in requested:
        filename = weight_map[key]
        keys_by_file.setdefault(filename, []).append(key)
    for filename, keys in keys_by_file.items():
        shard_path = checkpoint / filename
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_tensor(key)
                if tensor.dtype != torch.uint8:
                    tensor = tensor.view(torch.uint8)
                loaded[requested[key]] = tensor.contiguous()

    projections: dict[str, ProjectionWeights] = {}
    for projection in projection_names:
        packed = torch.stack(
            [loaded[(projection, "weight", index)] for index in range(len(expert_ids))]
        ).contiguous()
        scales = torch.stack(
            [loaded[(projection, "scale", index)] for index in range(len(expert_ids))]
        ).contiguous()
        projections[projection] = ProjectionWeights(
            packed=packed,
            scales=scales,
            bf16=_dequantize_projection(
                packed=packed, scales=scales, row_chunk=row_chunk
            ),
        )
    source_hash = _tensor_content_sha256(
        (
            projections["gate"].packed,
            projections["gate"].scales,
            projections["up"].packed,
            projections["up"].scales,
            projections["down"].packed,
            projections["down"].scales,
        )
    )
    return SourceWeights(
        gate=projections["gate"],
        up=projections["up"],
        down=projections["down"],
        metadata={
            "kind": "deepseek_v4_flash_checkpoint_mxfp4",
            "checkpoint": str(checkpoint.resolve(strict=True)),
            "index_sha256": _sha256(index_path),
            "layer": layer,
            "expert_ids": list(expert_ids),
            "selected_native_tensor_sha256": source_hash,
            "selected_shards": sorted(keys_by_file),
            "description": "checkpoint MXFP4 decoded exactly to shared BF16",
        },
    )


def _make_worker_pool(*, numa_node: int, threads: int) -> Any:
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [numa_node]
    worker_config.subpool_thread_count = [threads]
    return kt_kernel_ext.CPUInfer(worker_config)


def _configure_common_moe(
    *, cpu_infer: Any, expert_count: int, clamp_limit: float
) -> Any:
    config = kt_kernel_ext.moe.MOEConfig(
        expert_count, TOP_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0
    )
    # The generic AMXINT4 activation buffer tiles M by 32 even though decode
    # supplies only one to six live rows.  Production uses a much larger
    # chunked-prefill allocation; 32 is the smallest equivalent legal arena.
    config.max_len = 32
    config.pool = cpu_infer.backend_
    config.swiglu_limit = clamp_limit
    config.swiglu_alpha = 0.0
    return config


def _build_mxfp4(
    *, cpu_infer: Any, weights: SourceWeights, expert_count: int, clamp_limit: float
) -> Any:
    config = _configure_common_moe(
        cpu_infer=cpu_infer,
        expert_count=expert_count,
        clamp_limit=clamp_limit,
    )
    config.quant_config.bits = 4
    config.quant_config.group_size = GROUP_SIZE
    config.quant_config.zero_point = False
    config.gate_proj = weights.gate.packed.data_ptr()
    config.up_proj = weights.up.packed.data_ptr()
    config.down_proj = weights.down.packed.data_ptr()
    config.gate_scale = weights.gate.scales.data_ptr()
    config.up_scale = weights.up.scales.data_ptr()
    config.down_scale = weights.down.scales.data_ptr()
    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(config)
    physical_to_logical = torch.arange(expert_count, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe


def _build_amxint4(
    *, cpu_infer: Any, weights: SourceWeights, expert_count: int, clamp_limit: float
) -> Any:
    config = _configure_common_moe(
        cpu_infer=cpu_infer,
        expert_count=expert_count,
        clamp_limit=clamp_limit,
    )
    config.gate_proj = weights.gate.bf16.data_ptr()
    config.up_proj = weights.up.bf16.data_ptr()
    config.down_proj = weights.down.bf16.data_ptr()
    moe = kt_kernel_ext.moe.AMXInt4_MOE(config)
    physical_to_logical = torch.arange(expert_count, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe


def _quantize_rawint4_projection(
    weight: torch.Tensor, *, row_chunk: int
) -> RawInt4Projection:
    expert_count, output_features, input_features = weight.shape
    if input_features % GROUP_SIZE != 0:
        raise ValueError("RAWINT4 input features must be divisible by group size")
    packed = torch.empty(
        (expert_count, output_features, input_features // 2), dtype=torch.uint8
    )
    scales = torch.empty(
        (expert_count, output_features, input_features // GROUP_SIZE),
        dtype=torch.bfloat16,
    )
    for expert_id in range(expert_count):
        for row_start in range(0, output_features, row_chunk):
            row_end = min(row_start + row_chunk, output_features)
            blocks = (
                weight[expert_id, row_start:row_end]
                .to(torch.float32)
                .reshape(row_end - row_start, input_features // GROUP_SIZE, GROUP_SIZE)
            )
            block_scales = blocks.abs().amax(dim=-1).div_(7.0)
            block_scales.masked_fill_(block_scales == 0, 1.0)
            scales[expert_id, row_start:row_end].copy_(block_scales.to(torch.bfloat16))
            encoded = (
                torch.round(blocks / block_scales.unsqueeze(-1))
                .clamp_(-8, 7)
                .to(torch.int16)
                .add_(8)
                .to(torch.uint8)
                .reshape(row_end - row_start, input_features)
            )
            pairs = encoded.reshape(row_end - row_start, input_features // 2, 2)
            packed[expert_id, row_start:row_end].copy_(
                pairs[..., 0] | (pairs[..., 1] << 4)
            )
    return RawInt4Projection(packed=packed.contiguous(), scales=scales.contiguous())


def _quantize_rawint4(weights: SourceWeights, *, row_chunk: int) -> RawInt4Weights:
    return RawInt4Weights(
        gate=_quantize_rawint4_projection(weights.gate.bf16, row_chunk=row_chunk),
        up=_quantize_rawint4_projection(weights.up.bf16, row_chunk=row_chunk),
        down=_quantize_rawint4_projection(weights.down.bf16, row_chunk=row_chunk),
    )


def _build_rawint4(
    *,
    cpu_infer: Any,
    weights: RawInt4Weights,
    expert_count: int,
    clamp_limit: float,
) -> Any:
    config = _configure_common_moe(
        cpu_infer=cpu_infer,
        expert_count=expert_count,
        clamp_limit=clamp_limit,
    )
    config.quant_config.bits = 4
    config.quant_config.group_size = GROUP_SIZE
    config.quant_config.zero_point = False
    config.gate_proj = weights.gate.packed.data_ptr()
    config.up_proj = weights.up.packed.data_ptr()
    config.down_proj = weights.down.packed.data_ptr()
    config.gate_scale = weights.gate.scales.data_ptr()
    config.up_scale = weights.up.scales.data_ptr()
    config.down_scale = weights.down.scales.data_ptr()
    moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(config)
    physical_to_logical = torch.arange(expert_count, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe


def _make_cases(
    *, expert_count: int, route_case_count: int, seed: int
) -> dict[int, list[InputCase]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cases: dict[int, list[InputCase]] = {}
    for batch_size in M_VALUES:
        batch_cases: list[InputCase] = []
        for _ in range(route_case_count):
            expert_ids = torch.stack(
                [
                    torch.randperm(expert_count, generator=generator)[:TOP_K]
                    for _ in range(batch_size)
                ]
            ).to(torch.int64)
            routing_logits = torch.randn(
                (batch_size, TOP_K), generator=generator, dtype=torch.float32
            )
            routing_weights = torch.softmax(routing_logits, dim=-1).contiguous()
            hidden_states = (
                torch.randn(
                    (batch_size, HIDDEN_SIZE),
                    generator=generator,
                    dtype=torch.float32,
                )
                .mul_(0.25)
                .to(torch.bfloat16)
                .contiguous()
            )
            batch_cases.append(
                InputCase(
                    batch_size=torch.tensor([batch_size], dtype=torch.int32),
                    expert_ids=expert_ids.contiguous(),
                    routing_weights=routing_weights,
                    hidden_states=hidden_states,
                    output=torch.empty_like(hidden_states),
                )
            )
        cases[batch_size] = batch_cases
    return cases


def _run_forward(moe: Any, cpu_infer: Any, case: InputCase) -> None:
    cpu_infer.submit(
        moe.forward_task(
            case.batch_size.data_ptr(),
            TOP_K,
            case.expert_ids.data_ptr(),
            case.routing_weights.data_ptr(),
            case.hidden_states.data_ptr(),
            case.output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()


def _reference_output(
    *, case: InputCase, weights: SourceWeights, clamp_limit: float
) -> torch.Tensor:
    result = torch.zeros(
        (case.hidden_states.shape[0], HIDDEN_SIZE), dtype=torch.float32
    )
    expert_ids = case.expert_ids
    for expert_id_tensor in torch.unique(expert_ids):
        expert_id = int(expert_id_tensor.item())
        occurrences = torch.nonzero(expert_ids == expert_id, as_tuple=False)
        token_ids = occurrences[:, 0]
        route_slots = occurrences[:, 1]
        hidden = case.hidden_states[token_ids].to(torch.float32)
        gate = hidden @ weights.gate.bf16[expert_id].to(torch.float32).T
        up = hidden @ weights.up.bf16[expert_id].to(torch.float32).T
        if clamp_limit > 0.0:
            gate.clamp_(max=clamp_limit)
            up.clamp_(min=-clamp_limit, max=clamp_limit)
        activated = torch.nn.functional.silu(gate).mul_(up)
        expert_output = activated @ weights.down.bf16[expert_id].to(torch.float32).T
        route_scale = case.routing_weights[token_ids, route_slots].unsqueeze(1)
        result.index_add_(0, token_ids, expert_output.mul_(route_scale))
    return result


def _quality(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = output.to(torch.float32)
    difference = actual - reference
    reference_l1 = reference.abs().mean().item()
    dot = torch.dot(actual.flatten(), reference.flatten()).item()
    actual_norm = torch.linalg.vector_norm(actual).item()
    reference_norm = torch.linalg.vector_norm(reference).item()
    return {
        "relative_l1": difference.abs().mean().item() / max(reference_l1, 1e-30),
        "cosine_similarity": dot / max(actual_norm * reference_norm, 1e-30),
        "maximum_absolute_error": difference.abs().max().item(),
        "reference_mean_absolute": reference_l1,
    }


def _measure_backend(
    *,
    moe: Any,
    cpu_infer: Any,
    cases: list[InputCase],
    warmup_rounds: int,
    timed_rounds: int,
    iterations_per_round: int,
) -> dict[str, Any]:
    for index in range(warmup_rounds * len(cases)):
        _run_forward(moe, cpu_infer, cases[index % len(cases)])
    round_medians_us: list[float] = []
    all_samples_us: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_index in range(timed_rounds):
            samples_us: list[float] = []
            for iteration in range(iterations_per_round):
                case_index = (round_index * iterations_per_round + iteration) % len(
                    cases
                )
                start_ns = time.perf_counter_ns()
                _run_forward(moe, cpu_infer, cases[case_index])
                elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
                samples_us.append(elapsed_us)
            round_medians_us.append(statistics.median(samples_us))
            all_samples_us.extend(samples_us)
    finally:
        if gc_was_enabled:
            gc.enable()
    sorted_samples = sorted(all_samples_us)
    return {
        "sample_count": len(all_samples_us),
        "median_us": statistics.median(all_samples_us),
        "minimum_us": sorted_samples[0],
        "p10_us": sorted_samples[int(0.10 * (len(sorted_samples) - 1))],
        "p90_us": sorted_samples[int(0.90 * (len(sorted_samples) - 1))],
        "round_medians_us": round_medians_us,
    }


def _parse_backends(value: str) -> tuple[str, ...]:
    backends = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not backends or len(set(backends)) != len(backends):
        raise argparse.ArgumentTypeError("backends must be a non-empty unique list")
    unknown = set(backends) - BACKEND_CLASSES.keys()
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown backends: {sorted(unknown)}")
    return backends


def _parse_expert_ids(value: str) -> tuple[int, ...]:
    try:
        expert_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expert IDs must be decimal integers"
        ) from error
    if not expert_ids or len(set(expert_ids)) != len(expert_ids):
        raise argparse.ArgumentTypeError("expert IDs must be a non-empty unique list")
    if any(expert_id < 0 or expert_id >= 256 for expert_id in expert_ids):
        raise argparse.ArgumentTypeError("expert IDs must be in [0, 256)")
    return expert_ids


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", type=_parse_backends, default=("mxfp4", "amxint4")
    )
    parser.add_argument("--expert-count", type=int, default=12)
    parser.add_argument("--route-cases", type=int, default=12)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--timed-rounds", type=int, default=5)
    parser.add_argument("--iterations-per-round", type=int, default=24)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="optionally persist the canonical JSON receipt to this path",
    )
    parser.add_argument(
        "--expert-ids",
        type=_parse_expert_ids,
        default=(0, 7, 23, 41, 64, 89, 113, 137, 166, 193, 224, 255),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if (arguments.checkpoint is None) != (arguments.layer is None):
        raise ValueError("--checkpoint and --layer must be supplied together")
    expert_count = (
        len(arguments.expert_ids)
        if arguments.checkpoint is not None
        else arguments.expert_count
    )
    if expert_count < TOP_K:
        raise ValueError(f"expert count must be at least top-k ({TOP_K})")
    for name in (
        "route_cases",
        "warmup_rounds",
        "timed_rounds",
        "iterations_per_round",
        "threads",
        "row_chunk",
    ):
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name} must be positive")

    os.environ["KT_AMX_FUSED_ACTIVATION"] = "0"
    os.environ.setdefault("KT_AMX_FINE_GRAINED_DECODE", "1")
    os.environ.setdefault("KT_WORKER_SPIN_US", "1000")
    os.environ.setdefault("KT_MXFP4_AMX_MIN_EXPERT_TOKENS", "5")
    os.environ.setdefault("KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS", "2")
    host = _validate_host(numa_node=arguments.numa_node, threads=arguments.threads)
    extension_path = Path(kt_kernel_ext.__file__).resolve(strict=True)
    missing_bindings = [
        BACKEND_CLASSES[name]
        for name in arguments.backends
        if not hasattr(kt_kernel_ext.moe, BACKEND_CLASSES[name])
    ]
    if missing_bindings:
        raise RuntimeError(f"extension is missing backend bindings: {missing_bindings}")

    # The reference is not timed. Keep its OpenMP use bounded so it cannot
    # perturb the dedicated native worker pool during measurement.
    torch.set_num_threads(min(8, arguments.threads))
    torch.set_num_interop_threads(1)
    if arguments.checkpoint is None:
        weights = _synthesize_weights(
            expert_count=expert_count,
            seed=arguments.seed,
            row_chunk=arguments.row_chunk,
        )
    else:
        weights = _load_checkpoint_weights(
            checkpoint=arguments.checkpoint,
            layer=arguments.layer,
            expert_ids=arguments.expert_ids,
            row_chunk=arguments.row_chunk,
        )
    cases = _make_cases(
        expert_count=expert_count,
        route_case_count=arguments.route_cases,
        seed=arguments.seed + 1,
    )
    cpu_infer = _make_worker_pool(
        numa_node=arguments.numa_node, threads=arguments.threads
    )
    moes: dict[str, Any] = {}
    rawint4_weights = (
        _quantize_rawint4(weights, row_chunk=arguments.row_chunk)
        if "rawint4" in arguments.backends
        else None
    )
    for backend in arguments.backends:
        if backend == "mxfp4":
            moes[backend] = _build_mxfp4(
                cpu_infer=cpu_infer,
                weights=weights,
                expert_count=expert_count,
                clamp_limit=arguments.swiglu_limit,
            )
        elif backend == "amxint4":
            moes[backend] = _build_amxint4(
                cpu_infer=cpu_infer,
                weights=weights,
                expert_count=expert_count,
                clamp_limit=arguments.swiglu_limit,
            )
        elif backend == "rawint4":
            assert rawint4_weights is not None
            moes[backend] = _build_rawint4(
                cpu_infer=cpu_infer,
                weights=rawint4_weights,
                expert_count=expert_count,
                clamp_limit=arguments.swiglu_limit,
            )
        else:  # pragma: no cover - guarded by argparse
            raise AssertionError(backend)

    results: dict[str, Any] = {}
    quality_outputs: dict[int, dict[str, torch.Tensor]] = {
        batch_size: {} for batch_size in M_VALUES
    }
    for backend, moe in moes.items():
        backend_rows: dict[str, Any] = {}
        for batch_size in M_VALUES:
            quality_case = cases[batch_size][0]
            _run_forward(moe, cpu_infer, quality_case)
            output = quality_case.output.clone()
            quality_outputs[batch_size][backend] = output
            reference = _reference_output(
                case=quality_case,
                weights=weights,
                clamp_limit=arguments.swiglu_limit,
            )
            timing = _measure_backend(
                moe=moe,
                cpu_infer=cpu_infer,
                cases=cases[batch_size],
                warmup_rounds=arguments.warmup_rounds,
                timed_rounds=arguments.timed_rounds,
                iterations_per_round=arguments.iterations_per_round,
            )
            backend_rows[str(batch_size)] = {
                "output_sha256": _tensor_content_sha256((output,)),
                "quality_vs_bf16_source": _quality(output, reference),
                "timing": timing,
            }
        results[backend] = backend_rows

    candidate_speedups: dict[str, float] = {}
    candidate_quality: dict[str, dict[str, Any]] = {}
    if "mxfp4" in arguments.backends:
        for candidate in arguments.backends:
            if candidate == "mxfp4":
                continue
            speedups = []
            quality_rows: dict[str, Any] = {}
            for batch_size in M_VALUES:
                baseline_us = results["mxfp4"][str(batch_size)]["timing"]["median_us"]
                candidate_us = results[candidate][str(batch_size)]["timing"][
                    "median_us"
                ]
                speedup = baseline_us / candidate_us
                results[candidate][str(batch_size)]["speedup_over_mxfp4"] = speedup
                speedups.append(speedup)
                quality_rows[str(batch_size)] = _quality(
                    quality_outputs[batch_size][candidate],
                    quality_outputs[batch_size]["mxfp4"].to(torch.float32),
                )
            candidate_speedups[candidate] = math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            )
            candidate_quality[candidate] = quality_rows

    packed_bytes_per_expert = HIDDEN_SIZE * INTERMEDIATE_SIZE * 3 // 2 + (
        INTERMEDIATE_SIZE * (HIDDEN_SIZE // GROUP_SIZE) * 2
        + HIDDEN_SIZE * (INTERMEDIATE_SIZE // GROUP_SIZE)
    )
    scale_fold_telemetry = (
        kt_kernel_ext.mxfp4_avx_scale_fold_telemetry()
        if hasattr(kt_kernel_ext, "mxfp4_avx_scale_fold_telemetry")
        else None
    )
    cpu_dispatch_telemetry = {
        name: getattr(cpu_infer, name)()
        for name in (
            "task_queue_affinity",
            "single_numa_inline_dispatch",
            "worker_pool_affinity",
        )
        if hasattr(cpu_infer, name)
    }
    receipt = {
        "schema_version": 1,
        "benchmark": "dsv4_amxint4_vs_native_mxfp4_cpu",
        "timestamp_unix": time.time(),
        "extension": {
            "path": str(extension_path),
            "sha256": _sha256(extension_path),
        },
        "host": host,
        "shape": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "top_k": TOP_K,
            "batch_sizes": list(M_VALUES),
            "expert_count": expert_count,
            "mxfp4_group_size": GROUP_SIZE,
            "packed_bytes_per_expert_including_scales": packed_bytes_per_expert,
            "route_bank_packed_bytes": packed_bytes_per_expert * expert_count,
        },
        "methodology": {
            "source": weights.metadata,
            "seed": arguments.seed,
            "route_case_count": arguments.route_cases,
            "warmup_rounds": arguments.warmup_rounds,
            "timed_rounds": arguments.timed_rounds,
            "iterations_per_round": arguments.iterations_per_round,
            "timing_clock": "time.perf_counter_ns around CPUInfer submit+sync",
            "swiglu_limit": arguments.swiglu_limit,
            "fused_activation": False,
            "fine_grained_decode": os.environ["KT_AMX_FINE_GRAINED_DECODE"] == "1",
            "worker_spin_us": int(os.environ["KT_WORKER_SPIN_US"]),
            "mxfp4_amx_min_expert_tokens": int(
                os.environ["KT_MXFP4_AMX_MIN_EXPERT_TOKENS"]
            ),
            "mxfp4_avx_tiled_min_expert_tokens": int(
                os.environ["KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS"]
            ),
            "mxfp4_avx_scale_fold_mode": os.environ.get(
                "KT_MXFP4_AVX_SCALE_FOLD_MODE", "off"
            ),
        },
        "mxfp4_avx_scale_fold_telemetry": scale_fold_telemetry,
        "cpu_dispatch_telemetry": cpu_dispatch_telemetry,
        "backends": list(arguments.backends),
        "results": results,
        "candidate_geometric_speedup_over_mxfp4_m1_to_m6": candidate_speedups,
        "candidate_quality_vs_mxfp4": candidate_quality,
    }
    canonical_receipt = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(canonical_receipt, encoding="utf-8")
    print("DSV4_AMXINT4_BENCH_JSON=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
