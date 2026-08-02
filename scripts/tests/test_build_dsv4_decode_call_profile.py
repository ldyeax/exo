from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from scripts.build_dsv4_decode_call_profile import (
    build_pipeline_profile,
    build_replicated_per_pass_profile,
    main,
)


def _save_per_pass_profile(
    path: Path,
    *,
    rank: int,
    counts_by_pass: dict[int, torch.Tensor],
    physical_to_logical_map: torch.Tensor | None = None,
) -> None:
    first_count = next(iter(counts_by_pass.values()))
    if physical_to_logical_map is None:
        physical_to_logical_map = torch.arange(first_count.shape[1]).repeat(
            first_count.shape[0], 1
        )
    torch.save(
        {
            "records": [
                {
                    "forward_pass_id": forward_pass_id,
                    "rank": rank,
                    "gatherer_key": "primary",
                    "global_physical_count": counts,
                }
                for forward_pass_id, counts in counts_by_pass.items()
            ],
            "last_physical_to_logical_map": physical_to_logical_map,
        },
        path,
    )


def _save_ordered_per_pass_profile(
    path: Path,
    *,
    rank: int,
    records: list[tuple[int, torch.Tensor]],
) -> None:
    first_count = records[0][1]
    physical_to_logical_map = torch.arange(first_count.shape[1]).repeat(
        first_count.shape[0], 1
    )
    torch.save(
        {
            "records": [
                {
                    "forward_pass_id": forward_pass_id,
                    "rank": rank,
                    "gatherer_key": "primary",
                    "global_physical_count": counts,
                }
                for forward_pass_id, counts in records
            ],
            "last_physical_to_logical_map": physical_to_logical_map,
        },
        path,
    )


def test_pipeline_profile_retains_legacy_disjoint_stage_behavior(
    tmp_path: Path,
) -> None:
    rank_zero = torch.zeros((3, 2, 4), dtype=torch.int32)
    rank_one = torch.zeros((3, 2, 4), dtype=torch.int32)
    rank_zero[0, 0] = torch.tensor([2, 2, 0, 0])
    rank_zero[0, 1] = torch.tensor([1, 3, 0, 0])
    rank_one[1, 0] = torch.tensor([0, 0, 3, 1])
    rank_one[1, 1] = torch.tensor([0, 0, 2, 2])
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    torch.save({"logical_count": rank_zero}, rank_zero_path)
    torch.save({"logical_count": rank_one}, rank_one_path)

    output = build_pipeline_profile(
        [rank_zero_path, rank_one_path],
        stage_partition=(1, 1),
        decode_routes_per_layer=4,
    )

    assert torch.equal(
        output["logical_count"],
        torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.int64),
    )
    assert output["decode_records_by_rank"] == [1, 1]
    assert output["profile_topology"] == "pipeline"


def test_command_line_defaults_to_legacy_pipeline_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "rank0.pt"
    output_path = tmp_path / "calls.pt"
    torch.save(
        {"logical_count": torch.tensor([[[2, 2]]], dtype=torch.int32)},
        profile_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_decode_call_profile.py",
            "--rank-profiles",
            str(profile_path),
            "--stage-layer-partition",
            "1",
            "--decode-routes-per-layer",
            "4",
            "--output",
            str(output_path),
        ],
    )

    main()

    output = torch.load(output_path, map_location="cpu", weights_only=True)
    assert output["profile_topology"] == "pipeline"
    assert output["stage_layer_partition"] == (1,)


def test_replicated_per_pass_profiles_are_verified_and_counted_once(
    tmp_path: Path,
) -> None:
    prefill = torch.tensor([[10, 10, 0, 0], [10, 10, 0, 0]], dtype=torch.int32)
    decode_one = torch.tensor([[2, 2, 0, 0], [1, 3, 0, 0]], dtype=torch.int32)
    decode_two = torch.tensor([[0, 1, 1, 2], [0, 2, 1, 1]], dtype=torch.int32)
    records = {9: prefill, 11: decode_one, 12: decode_two}
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    _save_per_pass_profile(rank_zero_path, rank=0, counts_by_pass=records)
    _save_per_pass_profile(
        rank_one_path,
        rank=1,
        counts_by_pass=records,
    )

    output = build_replicated_per_pass_profile(
        [rank_one_path, rank_zero_path], decode_routes_per_layer=4
    )

    assert torch.equal(
        output["logical_count"],
        torch.tensor([[1, 2, 1, 1], [1, 2, 1, 1]], dtype=torch.int64),
    )
    assert torch.equal(
        output["decode_route_count"],
        torch.tensor([[2, 3, 1, 2], [1, 5, 1, 1]], dtype=torch.int64),
    )
    assert output["decode_records_by_rank"] == [2, 2]
    assert output["replica_ranks"] == [0, 1]
    assert output["replicated_forward_pass_count"] == 3
    assert output["profile_topology"] == "replicated_per_pass"


def test_replicated_per_pass_profile_allows_target_and_draft_to_share_ids(
    tmp_path: Path,
) -> None:
    prefill = torch.tensor([[10, 10, 0], [10, 10, 0]], dtype=torch.int32)
    draft = torch.tensor([[2, 2, 0], [0, 0, 0]], dtype=torch.int32)
    target = torch.tensor([[2, 2, 0], [1, 0, 3]], dtype=torch.int32)
    records = [(1, prefill), (1, draft), (7, target)]
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    _save_ordered_per_pass_profile(rank_zero_path, rank=0, records=records)
    _save_ordered_per_pass_profile(rank_one_path, rank=1, records=records)

    output = build_replicated_per_pass_profile(
        [rank_zero_path, rank_one_path], decode_routes_per_layer=4
    )

    assert torch.equal(
        output["logical_count"],
        torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.int64),
    )
    assert output["decode_records_by_rank"] == [1, 1]
    assert output["replicated_forward_pass_count"] == 2
    assert output["replicated_record_count"] == 3


def test_replicated_per_pass_profile_extracts_active_draft_layer_prefix(
    tmp_path: Path,
) -> None:
    target = torch.tensor(
        [[9, 9, 9, 9]] * 5,
        dtype=torch.int32,
    )
    draft_one = torch.tensor(
        [
            [20, 10, 0, 0],
            [0, 15, 15, 0],
            [0, 0, 10, 20],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    draft_two = torch.tensor(
        [
            [0, 30, 0, 0],
            [10, 10, 10, 0],
            [5, 5, 5, 15],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    records = [(4, target), (4, draft_one), (8, draft_two)]
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    _save_ordered_per_pass_profile(rank_zero_path, rank=0, records=records)
    _save_ordered_per_pass_profile(rank_one_path, rank=1, records=records)

    output = build_replicated_per_pass_profile(
        [rank_zero_path, rank_one_path],
        decode_routes_per_layer=30,
        active_prefix_layer_count=3,
    )

    assert torch.equal(
        output["logical_count"],
        torch.tensor(
            [[1, 2, 0, 0], [1, 2, 2, 0], [1, 1, 2, 2]],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        output["decode_route_count"],
        torch.tensor(
            [[20, 40, 0, 0], [10, 25, 25, 0], [5, 5, 15, 35]],
            dtype=torch.int64,
        ),
    )
    assert output["decode_records_by_rank"] == [2, 2]
    assert output["stage_layer_partition"] == (3,)
    assert output["active_prefix_layer_count"] == 3


def test_command_line_extracts_active_draft_layer_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "rank0.pt"
    output_path = tmp_path / "draft-calls.pt"
    target = torch.tensor([[20, 16], [20, 16], [20, 16], [20, 16]])
    draft = torch.tensor([[20, 10], [15, 15], [10, 20], [0, 0]])
    _save_ordered_per_pass_profile(
        profile_path,
        rank=0,
        records=[(1, target), (1, draft)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_decode_call_profile.py",
            "--rank-profiles",
            str(profile_path),
            "--profile-topology",
            "replicated-per-pass",
            "--decode-routes-per-layer",
            "30",
            "--active-prefix-layer-count",
            "3",
            "--output",
            str(output_path),
        ],
    )

    main()

    output = torch.load(output_path, map_location="cpu", weights_only=True)
    assert torch.equal(output["logical_count"], torch.ones((3, 2), dtype=torch.int64))
    assert torch.equal(
        output["decode_route_count"],
        torch.tensor([[20, 10], [15, 15], [10, 20]], dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("topology", "active_prefix_layer_count", "message"),
    [
        ("replicated-per-pass", "0", "active prefix layer count must be positive"),
        ("pipeline", "3", "only valid for replicated per-pass profiles"),
    ],
)
def test_command_line_validates_active_prefix_layer_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    active_prefix_layer_count: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_decode_call_profile.py",
            "--rank-profiles",
            str(tmp_path / "unused.pt"),
            "--profile-topology",
            topology,
            "--active-prefix-layer-count",
            active_prefix_layer_count,
            "--output",
            str(tmp_path / "unused-output.pt"),
        ],
    )

    with pytest.raises(ValueError, match=message):
        main()


def test_replicated_per_pass_profile_converts_physical_to_logical_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rank0.pt"
    _save_per_pass_profile(
        path,
        rank=0,
        counts_by_pass={7: torch.tensor([[1, 2, 3, 0]], dtype=torch.int32)},
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]], dtype=torch.int64),
    )

    output = build_replicated_per_pass_profile([path], decode_routes_per_layer=6)

    assert torch.equal(
        output["logical_count"], torch.tensor([[1, 1]], dtype=torch.int64)
    )
    assert torch.equal(
        output["decode_route_count"], torch.tensor([[4, 2]], dtype=torch.int64)
    )


def test_replicated_per_pass_profile_rejects_divergent_replica(
    tmp_path: Path,
) -> None:
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    _save_per_pass_profile(
        rank_zero_path,
        rank=0,
        counts_by_pass={3: torch.tensor([[2, 2]], dtype=torch.int32)},
    )
    _save_per_pass_profile(
        rank_one_path,
        rank=1,
        counts_by_pass={3: torch.tensor([[1, 3]], dtype=torch.int32)},
    )

    with pytest.raises(ValueError, match="routes differ"):
        build_replicated_per_pass_profile(
            [rank_zero_path, rank_one_path], decode_routes_per_layer=4
        )


def test_replicated_per_pass_profile_rejects_incomplete_replica(
    tmp_path: Path,
) -> None:
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    count = torch.tensor([[2, 2]], dtype=torch.int32)
    _save_per_pass_profile(rank_zero_path, rank=0, counts_by_pass={3: count, 4: count})
    _save_per_pass_profile(rank_one_path, rank=1, counts_by_pass={3: count})

    with pytest.raises(ValueError, match="forward-pass IDs differ"):
        build_replicated_per_pass_profile(
            [rank_zero_path, rank_one_path], decode_routes_per_layer=4
        )


def test_replicated_per_pass_profile_rejects_reordered_replica(
    tmp_path: Path,
) -> None:
    rank_zero_path = tmp_path / "rank0.pt"
    rank_one_path = tmp_path / "rank1.pt"
    first = torch.tensor([[2, 2]], dtype=torch.int32)
    second = torch.tensor([[1, 3]], dtype=torch.int32)
    _save_ordered_per_pass_profile(
        rank_zero_path, rank=0, records=[(3, first), (4, second)]
    )
    _save_ordered_per_pass_profile(
        rank_one_path, rank=1, records=[(4, second), (3, first)]
    )

    with pytest.raises(ValueError, match="ordered record"):
        build_replicated_per_pass_profile(
            [rank_zero_path, rank_one_path], decode_routes_per_layer=4
        )
