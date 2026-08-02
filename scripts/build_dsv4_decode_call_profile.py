#!/usr/bin/env python3
"""Convert DSV4 route-volume traces into decode expert-call frequencies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

Profile = dict[str, object]
PerPassRecord = tuple[int, int, str, torch.Tensor]


def _load_profile(path: Path) -> Profile:
    profile = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(profile, dict):
        raise ValueError(f"profile is not a dictionary: {path}")
    return cast(Profile, profile)


def _validate_count_tensor(tensor: torch.Tensor, path: Path) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"profile logical_count must be three-dimensional: {path}")
    if tensor.dtype == torch.bool or tensor.is_floating_point():
        raise ValueError(f"profile logical_count must use an integer dtype: {path}")
    tensor = tensor.to(device="cpu", dtype=torch.int64).contiguous()
    if bool((tensor < 0).any()):
        raise ValueError(f"profile logical_count contains negative values: {path}")
    return tensor


def _summarize_decode_records(
    counts: torch.Tensor,
    decode_routes_per_layer: int,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    per_layer_totals = counts.sum(dim=-1)
    fully_populated_rows = torch.all(per_layer_totals > 0, dim=-1)
    inconsistent_decode_rows = (
        fully_populated_rows
        & (per_layer_totals[:, 0] == decode_routes_per_layer)
        & ~torch.all(per_layer_totals == decode_routes_per_layer, dim=-1)
    )
    if bool(inconsistent_decode_rows.any()):
        raise ValueError(f"decode record route total changes inside {label}")
    decode_rows = fully_populated_rows & torch.all(
        per_layer_totals == decode_routes_per_layer, dim=-1
    )
    if not bool(decode_rows.any()):
        raise ValueError(
            f"{label} has no decode records with {decode_routes_per_layer} routes"
        )
    selected = counts[decode_rows]
    return (
        (selected > 0).sum(dim=0, dtype=torch.int64),
        selected.sum(dim=0, dtype=torch.int64),
        int(decode_rows.sum().item()),
    )


def build_pipeline_profile(
    profile_paths: list[Path],
    stage_partition: tuple[int, ...],
    decode_routes_per_layer: int,
) -> Profile:
    """Build a profile from legacy stat dumps with disjoint PP layer ranges."""
    if not profile_paths:
        raise ValueError("at least one rank profile is required")
    if len(stage_partition) != len(profile_paths):
        raise ValueError("rank profile count must match the pipeline layer partition")
    if any(layer_count <= 0 for layer_count in stage_partition):
        raise ValueError("pipeline layer counts must be positive")

    raw_counts: list[torch.Tensor] = []
    for path in profile_paths:
        profile = _load_profile(path)
        logical_count = profile.get("logical_count")
        if not isinstance(logical_count, torch.Tensor):
            raise ValueError(f"profile lacks logical_count tensor: {path}")
        raw_counts.append(_validate_count_tensor(logical_count, path))

    shape = raw_counts[0].shape[1:]
    if any(tuple(tensor.shape[1:]) != tuple(shape) for tensor in raw_counts):
        raise ValueError("rank profiles have different layer/expert shapes")
    if sum(stage_partition) != shape[0]:
        raise ValueError("pipeline layer partition does not cover the profile layers")

    call_counts = torch.zeros(shape, dtype=torch.int64)
    route_counts = torch.zeros(shape, dtype=torch.int64)
    decode_records_by_rank: list[int] = []
    layer_start = 0
    for tensor, layer_count in zip(raw_counts, stage_partition, strict=True):
        layer_end = layer_start + layer_count
        stage = tensor[:, layer_start:layer_end]
        stage_calls, stage_routes, decode_record_count = _summarize_decode_records(
            stage,
            decode_routes_per_layer,
            label=f"rank stage {layer_start}:{layer_end}",
        )
        call_counts[layer_start:layer_end] = stage_calls
        route_counts[layer_start:layer_end] = stage_routes
        decode_records_by_rank.append(decode_record_count)
        layer_start = layer_end

    return {
        "logical_count": call_counts,
        "decode_route_count": route_counts,
        "decode_records_by_rank": decode_records_by_rank,
        "decode_routes_per_layer": decode_routes_per_layer,
        "stage_layer_partition": stage_partition,
        "source_profiles": [str(path.resolve()) for path in profile_paths],
        "profile_topology": "pipeline",
        "metric": "distinct_decode_expert_calls",
    }


def _logical_counts_from_per_pass_profile(
    path: Path,
) -> tuple[int, torch.Tensor, list[PerPassRecord]]:
    profile = _load_profile(path)
    records = profile.get("records")
    physical_to_logical_map = profile.get("last_physical_to_logical_map")
    if not isinstance(records, list):
        raise ValueError(f"per-pass profile lacks records list: {path}")
    if not records:
        raise ValueError(f"per-pass profile has no records: {path}")
    if not isinstance(physical_to_logical_map, torch.Tensor):
        raise ValueError(f"per-pass profile lacks physical-to-logical map: {path}")
    if physical_to_logical_map.ndim != 2:
        raise ValueError(f"physical-to-logical map must be two-dimensional: {path}")
    if physical_to_logical_map.numel() == 0:
        raise ValueError(f"physical-to-logical map must not be empty: {path}")
    if (
        physical_to_logical_map.dtype == torch.bool
        or physical_to_logical_map.is_floating_point()
    ):
        raise ValueError(f"physical-to-logical map must use an integer dtype: {path}")
    physical_to_logical_map = physical_to_logical_map.to(
        device="cpu", dtype=torch.int64
    ).contiguous()
    if bool((physical_to_logical_map < 0).any()):
        raise ValueError(f"physical-to-logical map contains negative values: {path}")
    logical_expert_count = int(physical_to_logical_map.max().item()) + 1
    observed_logical_experts = torch.unique(physical_to_logical_map)
    if observed_logical_experts.tolist() != list(range(logical_expert_count)):
        raise ValueError(f"physical-to-logical map has missing logical experts: {path}")

    profile_rank: int | None = None
    logical_records: list[PerPassRecord] = []
    occurrences_by_forward_pass: dict[int, int] = {}
    expected_physical_shape = tuple(physical_to_logical_map.shape)
    for record_index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(
                f"per-pass record {record_index} is not a dictionary: {path}"
            )
        rank = raw_record.get("rank")
        forward_pass_id = raw_record.get("forward_pass_id")
        gatherer_key = raw_record.get("gatherer_key")
        physical_count = raw_record.get("global_physical_count")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"per-pass record {record_index} has invalid rank: {path}")
        if (
            isinstance(forward_pass_id, bool)
            or not isinstance(forward_pass_id, int)
            or forward_pass_id < 0
        ):
            raise ValueError(
                f"per-pass record {record_index} has invalid forward_pass_id: {path}"
            )
        if profile_rank is None:
            profile_rank = rank
        elif rank != profile_rank:
            raise ValueError(f"per-pass profile mixes recorder ranks: {path}")
        if not isinstance(gatherer_key, str) or not gatherer_key:
            raise ValueError(
                f"per-pass record {record_index} has invalid gatherer_key: {path}"
            )
        if not isinstance(physical_count, torch.Tensor):
            raise ValueError(
                f"per-pass record {record_index} lacks global_physical_count: {path}"
            )
        if tuple(physical_count.shape) != expected_physical_shape:
            raise ValueError(
                f"per-pass record {record_index} physical count shape "
                f"{tuple(physical_count.shape)} does not match map "
                f"{expected_physical_shape}: {path}"
            )
        if physical_count.dtype == torch.bool or physical_count.is_floating_point():
            raise ValueError(
                f"per-pass record {record_index} counts must use an integer dtype: {path}"
            )
        physical_count = physical_count.to(device="cpu", dtype=torch.int64).contiguous()
        if bool((physical_count < 0).any()):
            raise ValueError(
                f"per-pass record {record_index} contains negative counts: {path}"
            )
        logical_count = torch.zeros(
            (expected_physical_shape[0], logical_expert_count), dtype=torch.int64
        )
        logical_count.scatter_add_(
            dim=1,
            index=physical_to_logical_map,
            src=physical_count,
        )
        occurrence = occurrences_by_forward_pass.get(forward_pass_id, 0)
        occurrences_by_forward_pass[forward_pass_id] = occurrence + 1
        logical_records.append(
            (forward_pass_id, occurrence, gatherer_key, logical_count)
        )

    assert profile_rank is not None
    return profile_rank, physical_to_logical_map, logical_records


def build_replicated_per_pass_profile(
    profile_paths: list[Path],
    decode_routes_per_layer: int,
    active_prefix_layer_count: int | None = None,
) -> Profile:
    """Build one call profile from identical per-pass TP/EP route replicas."""
    if not profile_paths:
        raise ValueError("at least one rank profile is required")

    profiles_by_rank: dict[int, tuple[Path, torch.Tensor, list[PerPassRecord]]] = {}
    for path in profile_paths:
        rank, physical_to_logical_map, records = _logical_counts_from_per_pass_profile(
            path
        )
        if rank in profiles_by_rank:
            raise ValueError(f"multiple per-pass profiles report recorder rank {rank}")
        profiles_by_rank[rank] = (path, physical_to_logical_map, records)

    reference_rank = min(profiles_by_rank)
    _, reference_map, reference_records = profiles_by_rank[reference_rank]
    for rank, (path, physical_to_logical_map, records) in profiles_by_rank.items():
        if not torch.equal(physical_to_logical_map, reference_map):
            raise ValueError(
                f"replica rank {rank} physical-to-logical map differs from rank "
                f"{reference_rank}: profile={path}"
            )
        if len(records) != len(reference_records):
            raise ValueError(
                f"replica rank {rank} forward-pass IDs differ from rank "
                f"{reference_rank} in ordered record sequence: "
                f"record_count={len(records)}, "
                f"reference_record_count={len(reference_records)}, profile={path}"
            )
        for record_index, (record, reference_record) in enumerate(
            zip(records, reference_records, strict=True)
        ):
            record_metadata = record[:3]
            reference_metadata = reference_record[:3]
            if record_metadata != reference_metadata:
                raise ValueError(
                    f"replica rank {rank} forward-pass IDs differ from rank "
                    f"{reference_rank} at ordered record {record_index}: "
                    f"record={record_metadata}, reference={reference_metadata}, "
                    f"profile={path}"
                )
            if not torch.equal(record[3], reference_record[3]):
                forward_pass_id, occurrence, gatherer_key = record_metadata
                raise ValueError(
                    f"replica rank {rank} routes differ from rank {reference_rank} "
                    f"at ordered record {record_index}, forward_pass_id "
                    f"{forward_pass_id}, occurrence {occurrence}, "
                    f"gatherer_key {gatherer_key}"
                )

    profile_layer_count = int(reference_map.shape[0])
    if active_prefix_layer_count is not None:
        if (
            isinstance(active_prefix_layer_count, bool)
            or active_prefix_layer_count <= 0
        ):
            raise ValueError("active prefix layer count must be positive")
        if active_prefix_layer_count > profile_layer_count:
            raise ValueError(
                "active prefix layer count exceeds per-pass profile layer count: "
                f"active={active_prefix_layer_count}, profile={profile_layer_count}"
            )

    replicated_counts = torch.stack([record[3] for record in reference_records])
    if active_prefix_layer_count is not None:
        replicated_counts = replicated_counts[:, :active_prefix_layer_count]
    call_counts, route_counts, decode_record_count = _summarize_decode_records(
        replicated_counts,
        decode_routes_per_layer,
        label="replicated per-pass profile",
    )
    recorder_ranks = sorted(profiles_by_rank)
    output: Profile = {
        "logical_count": call_counts,
        "decode_route_count": route_counts,
        "decode_records_by_rank": [decode_record_count for _ in recorder_ranks],
        "decode_routes_per_layer": decode_routes_per_layer,
        "stage_layer_partition": (int(call_counts.shape[0]),),
        "source_profiles": [str(path.resolve()) for path in profile_paths],
        "profile_topology": "replicated_per_pass",
        "replica_ranks": recorder_ranks,
        "replicated_forward_pass_count": len(
            {forward_pass_id for forward_pass_id, _, _, _ in reference_records}
        ),
        "replicated_record_count": len(reference_records),
        "metric": "distinct_decode_expert_calls",
    }
    if active_prefix_layer_count is not None:
        output["active_prefix_layer_count"] = active_prefix_layer_count
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank-profiles",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Recorder profiles in pipeline-rank order, or one per replicated "
            "TP/EP rank with --profile-topology replicated-per-pass."
        ),
    )
    parser.add_argument(
        "--profile-topology",
        "--topology",
        choices=("pipeline", "replicated", "replicated-per-pass"),
        default="pipeline",
        help="Recorder topology and on-disk schema.",
    )
    parser.add_argument(
        "--stage-layer-partition",
        default="23,25,13",
        help="Comma-separated pipeline layer counts (pipeline topology only).",
    )
    parser.add_argument(
        "--decode-routes-per-layer",
        type=int,
        default=36,
        help="Expected tokens times top-k for one target verification.",
    )
    parser.add_argument(
        "--active-prefix-layer-count",
        type=int,
        help=(
            "Only classify and summarize records across this many leading layers "
            "of a replicated per-pass profile (for example, a 3-layer DSpark "
            "draft)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_routes_per_layer <= 0:
        raise ValueError("decode routes per layer must be positive")
    if (
        args.active_prefix_layer_count is not None
        and args.active_prefix_layer_count <= 0
    ):
        raise ValueError("active prefix layer count must be positive")
    if (
        args.profile_topology == "pipeline"
        and args.active_prefix_layer_count is not None
    ):
        raise ValueError(
            "active prefix layer count is only valid for replicated per-pass profiles"
        )

    if args.profile_topology == "pipeline":
        stage_partition = tuple(
            int(value) for value in args.stage_layer_partition.split(",")
        )
        output = build_pipeline_profile(
            args.rank_profiles,
            stage_partition,
            args.decode_routes_per_layer,
        )
    else:
        output = build_replicated_per_pass_profile(
            args.rank_profiles,
            args.decode_routes_per_layer,
            active_prefix_layer_count=args.active_prefix_layer_count,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    logical_count = output["logical_count"]
    assert isinstance(logical_count, torch.Tensor)
    print(
        f"wrote {args.output}: shape={tuple(logical_count.shape)} "
        f"decode_records={output['decode_records_by_rank']} "
        f"expert_calls={int(logical_count.sum().item())}"
    )


if __name__ == "__main__":
    main()
