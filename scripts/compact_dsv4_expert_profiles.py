#!/usr/bin/env python3
"""Compact sampled DSV4 expert recorders without changing planner costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

try:
    from scripts.build_dsv4_critical_tail_plan_sweep import (
        COMPACT_PROFILE_FORMAT,
        parse_named_path,
    )
    from scripts.build_dsv4_kt_hybrid_shard_plan import sha256_file
except ModuleNotFoundError:
    from build_dsv4_critical_tail_plan_sweep import (
        COMPACT_PROFILE_FORMAT,
        parse_named_path,
    )
    from build_dsv4_kt_hybrid_shard_plan import sha256_file


RECEIPT_FORMAT = "dsv4_compact_logical_count_profile_receipt_v1"


def build_compact_payload(
    source_path: Path,
    *,
    routes_per_layer: int,
) -> dict[str, object]:
    loaded = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("expert recorder must contain a dictionary")
    raw_count = loaded.get("logical_count")
    if not isinstance(raw_count, torch.Tensor):
        raise TypeError("expert recorder must contain logical_count")
    if (
        raw_count.ndim != 3
        or raw_count.dtype == torch.bool
        or raw_count.is_floating_point()
        or raw_count.is_complex()
    ):
        raise TypeError("logical_count must be integer [samples,layers,experts]")
    logical_count = raw_count.to(device="cpu", dtype=torch.int64)
    if logical_count.numel() and int(logical_count.min()) < 0:
        raise ValueError("logical_count must be non-negative")
    selected = logical_count[
        torch.all(logical_count.sum(dim=2) == routes_per_layer, dim=1)
    ]
    if selected.shape[0] == 0:
        raise ValueError("no samples match the requested route count")
    summed_count = selected.sum(dim=0)
    active_count = (selected > 0).sum(dim=0)
    additional_count = torch.clamp(selected - 1, min=0).sum(dim=0)
    if not torch.equal(summed_count, active_count + additional_count):
        raise AssertionError("compacted logical-count accounting drifted")
    maximum = max(
        int(summed_count.max()),
        int(active_count.max()),
        int(additional_count.max()),
    )
    output_dtype = (
        torch.int32 if maximum <= torch.iinfo(torch.int32).max else torch.int64
    )
    tensors = {
        "logical_count": summed_count.to(output_dtype).contiguous(),
        "active_expert_count": active_count.to(output_dtype).contiguous(),
        "additional_row_count": additional_count.to(output_dtype).contiguous(),
    }
    semantic_digest = hashlib.sha256()
    semantic_digest.update(routes_per_layer.to_bytes(8, "big"))
    semantic_digest.update(selected.shape[0].to_bytes(8, "big"))
    for name in sorted(tensors):
        semantic_digest.update(name.encode("utf-8"))
        semantic_digest.update(tensors[name].numpy().tobytes(order="C"))
    return {
        "format": COMPACT_PROFILE_FORMAT,
        **tensors,
        "sample_count": torch.tensor(selected.shape[0], dtype=torch.int64),
        "routes_per_layer": torch.tensor(routes_per_layer, dtype=torch.int64),
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "semantic_sha256": semantic_digest.hexdigest(),
    }


def compact_profile(
    source_path: Path,
    output_path: Path,
    *,
    routes_per_layer: int,
) -> dict[str, object]:
    payload = build_compact_payload(
        source_path,
        routes_per_layer=routes_per_layer,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    receipt = {
        "format": RECEIPT_FORMAT,
        "profile": str(output_path.resolve()),
        "profile_sha256": sha256_file(output_path),
        "semantic_sha256": payload["semantic_sha256"],
        "source": str(source_path.resolve()),
        "source_sha256": payload["source_sha256"],
        "sample_count": int(cast(torch.Tensor, payload["sample_count"])),
        "routes_per_layer": routes_per_layer,
        "shape": list(cast(torch.Tensor, payload["logical_count"]).shape),
    }
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="append", type=parse_named_path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routes-per-layer", type=int, default=72)
    arguments = parser.parse_args()
    if arguments.routes_per_layer <= 0:
        parser.error("--routes-per-layer must be positive")
    labels = [label for label, _path in arguments.profile]
    if len(labels) != len(set(labels)):
        parser.error("profile labels must be unique")
    for label, source_path in arguments.profile:
        output_path = (
            arguments.output_dir
            / f"dsv4_flash_hotspot_{label}_routes{arguments.routes_per_layer}.pt"
        )
        compact_profile(
            source_path,
            output_path,
            routes_per_layer=arguments.routes_per_layer,
        )
        print(output_path)


if __name__ == "__main__":
    main()
