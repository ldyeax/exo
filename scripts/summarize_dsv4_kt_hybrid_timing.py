#!/usr/bin/env python3
"""Summarize graph-safe KT hybrid JSONL receipts across local TP/EP/PP ranks."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

TIMING_FORMAT = "sglang_kt_hybrid_timing_v1"
SUMMARY_FORMAT = "dsv4_kt_hybrid_timing_summary_v1"


class _ParsedArguments(Protocol):
    receipts: list[Path]
    output: Path | None


def load_receipts(paths: Iterable[Path]) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as receipt_file:
            for line_number, line in enumerate(receipt_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                loaded = cast(object, json.loads(stripped))
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"{path}:{line_number} is not a {TIMING_FORMAT} receipt"
                    )
                loaded_receipt = cast(dict[str, object], loaded)
                if loaded_receipt.get("format") != TIMING_FORMAT:
                    raise ValueError(
                        f"{path}:{line_number} is not a {TIMING_FORMAT} receipt"
                    )
                receipts.append(loaded_receipt)
    if not receipts:
        raise ValueError("no timing receipts were loaded")
    return receipts


def _number(receipt: dict[str, object], field: str) -> float:
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"receipt field {field!r} must be numeric")
    return float(value)


def _integer(receipt: dict[str, object], field: str) -> int:
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"receipt field {field!r} must be an integer")
    return value


def _pipeline_rank(receipt: dict[str, object]) -> int:
    """Return PP rank while retaining compatibility with pre-PP receipts."""
    value = receipt.get("pp_rank", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("receipt field 'pp_rank' must be an integer")
    return value


def summarize(receipts: list[dict[str, object]]) -> dict[str, object]:
    by_layer_rank: dict[tuple[int, int, int, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    by_step_layer: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(
        list
    )
    for receipt in receipts:
        layer = _integer(receipt, "layer")
        pp_rank = _pipeline_rank(receipt)
        tp_rank = _integer(receipt, "tp_rank")
        ep_rank = _integer(receipt, "ep_rank")
        step = _integer(receipt, "step")
        num_tokens = _integer(receipt, "num_tokens")
        by_layer_rank[(layer, pp_rank, tp_rank, ep_rank)].append(receipt)
        by_step_layer[(step, layer, num_tokens)].append(receipt)

    layer_rank_summary: list[dict[str, object]] = []
    for (layer, pp_rank, tp_rank, ep_rank), rank_receipts in sorted(
        by_layer_rank.items()
    ):
        cpu_wait = [_number(receipt, "cpu_wait_ms") for receipt in rank_receipts]
        total = [_number(receipt, "total_ms") for receipt in rank_receipts]
        active_experts = [
            _number(receipt, "cpu_active_experts")
            for receipt in rank_receipts
            if receipt.get("cpu_active_experts") is not None
        ]
        estimated_bytes = [
            _number(receipt, "estimated_cpu_weight_stream_bytes")
            for receipt in rank_receipts
            if receipt.get("estimated_cpu_weight_stream_bytes") is not None
        ]
        layer_rank_summary.append(
            {
                "layer": layer,
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "ep_rank": ep_rank,
                "samples": len(rank_receipts),
                "mean_cpu_wait_ms": statistics.fmean(cpu_wait),
                "median_cpu_wait_ms": statistics.median(cpu_wait),
                "mean_total_ms": statistics.fmean(total),
                "mean_cpu_active_experts": (
                    statistics.fmean(active_experts) if active_experts else None
                ),
                "mean_estimated_cpu_weight_stream_bytes": (
                    statistics.fmean(estimated_bytes) if estimated_bytes else None
                ),
            }
        )

    paired_layer_skews: list[float] = []
    paired_expert_skews: list[float] = []
    paired_weight_byte_skews: list[float] = []
    paired_step_layers: dict[int, dict[int, list[dict[str, object]]]] = defaultdict(
        dict
    )
    expected_layers = {_integer(receipt, "layer") for receipt in receipts}
    for (step, layer, _num_tokens), layer_receipts in by_step_layer.items():
        unique_ranks = {
            (
                _pipeline_rank(receipt),
                _integer(receipt, "tp_rank"),
                _integer(receipt, "ep_rank"),
            )
            for receipt in layer_receipts
        }
        if len(unique_ranks) < 2:
            continue
        cpu_wait = [_number(receipt, "cpu_wait_ms") for receipt in layer_receipts]
        paired_layer_skews.append(max(cpu_wait) - min(cpu_wait))
        active_experts = [
            _number(receipt, "cpu_active_experts")
            for receipt in layer_receipts
            if receipt.get("cpu_active_experts") is not None
        ]
        if len(active_experts) >= 2:
            paired_expert_skews.append(max(active_experts) - min(active_experts))
        estimated_bytes = [
            _number(receipt, "estimated_cpu_weight_stream_bytes")
            for receipt in layer_receipts
            if receipt.get("estimated_cpu_weight_stream_bytes") is not None
        ]
        if len(estimated_bytes) >= 2:
            paired_weight_byte_skews.append(max(estimated_bytes) - min(estimated_bytes))
        paired_step_layers[step][layer] = layer_receipts

    cycle_critical_cpu_wait_ms = [
        sum(
            max(_number(receipt, "cpu_wait_ms") for receipt in layer_receipts)
            for layer_receipts in layers.values()
        )
        for _, layers in sorted(paired_step_layers.items())
        if set(layers) == expected_layers
    ]

    layers_by_pipeline_rank: dict[int, set[int]] = defaultdict(set)
    stage_step_layers: dict[tuple[int, int], dict[int, list[dict[str, object]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for receipt in receipts:
        pp_rank = _pipeline_rank(receipt)
        layer = _integer(receipt, "layer")
        step = _integer(receipt, "step")
        layers_by_pipeline_rank[pp_rank].add(layer)
        stage_step_layers[(pp_rank, step)][layer].append(receipt)
    stage_cycles: dict[int, list[float]] = defaultdict(list)
    for (pp_rank, _step), layers in stage_step_layers.items():
        if set(layers) != layers_by_pipeline_rank[pp_rank]:
            continue
        stage_cycles[pp_rank].append(
            sum(
                max(_number(receipt, "cpu_wait_ms") for receipt in layer_receipts)
                for layer_receipts in layers.values()
            )
        )
    pipeline_stage_summary: list[dict[str, object]] = []
    for pp_rank, layers in sorted(layers_by_pipeline_rank.items()):
        cycles = stage_cycles.get(pp_rank, [])
        pipeline_stage_summary.append(
            {
                "pp_rank": pp_rank,
                "layer_count": len(layers),
                "layers": sorted(layers),
                "complete_cycle_count": len(cycles),
                "mean_cycle_critical_cpu_wait_ms": (
                    statistics.fmean(cycles) if cycles else None
                ),
                "median_cycle_critical_cpu_wait_ms": (
                    statistics.median(cycles) if cycles else None
                ),
                "max_cycle_critical_cpu_wait_ms": max(cycles) if cycles else None,
            }
        )
    latest_numa_by_rank: dict[str, object] = {}
    for receipt in sorted(
        receipts, key=lambda item: _integer(item, "end_monotonic_ns")
    ):
        numa_pages = receipt.get("numa_pages_by_node")
        if isinstance(numa_pages, dict):
            rank_key = (
                f"pp{_pipeline_rank(receipt)}-"
                f"tp{_integer(receipt, 'tp_rank')}-"
                f"ep{_integer(receipt, 'ep_rank')}"
            )
            latest_numa_by_rank[rank_key] = numa_pages

    return {
        "format": SUMMARY_FORMAT,
        "receipt_count": len(receipts),
        "rank_count": len(
            {
                (
                    _pipeline_rank(receipt),
                    _integer(receipt, "tp_rank"),
                    _integer(receipt, "ep_rank"),
                )
                for receipt in receipts
            }
        ),
        "layer_rank": layer_rank_summary,
        "pipeline_stage": pipeline_stage_summary,
        "paired_layer_count": len(paired_layer_skews),
        "mean_paired_cpu_wait_skew_ms": (
            statistics.fmean(paired_layer_skews) if paired_layer_skews else None
        ),
        "worst_paired_cpu_wait_skew_ms": (
            max(paired_layer_skews) if paired_layer_skews else None
        ),
        "mean_paired_active_expert_skew": (
            statistics.fmean(paired_expert_skews) if paired_expert_skews else None
        ),
        "mean_paired_estimated_weight_byte_skew": (
            statistics.fmean(paired_weight_byte_skews)
            if paired_weight_byte_skews
            else None
        ),
        "complete_cycle_count": len(cycle_critical_cpu_wait_ms),
        "mean_cycle_critical_cpu_wait_ms": (
            statistics.fmean(cycle_critical_cpu_wait_ms)
            if cycle_critical_cpu_wait_ms
            else None
        ),
        "latest_numa_pages_by_rank": latest_numa_by_rank,
        "external_counter_join": {
            "timestamp_fields": ["start_monotonic_ns", "end_monotonic_ns"],
            "note": "Join Intel PCM/perf samples by the monotonic interval; estimated weight bytes are not measured DRAM traffic.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = cast(_ParsedArguments, cast(object, parser.parse_args()))
    summary = summarize(load_receipts(arguments.receipts))
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
