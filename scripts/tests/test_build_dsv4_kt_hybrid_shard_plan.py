from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.build_dsv4_kt_hybrid_shard_plan import (
    GpuSelectionStrategy,
    assign_hybrid_layer,
    load_hottest_first_ordering,
    main,
    reduce_profile,
    select_hottest_first,
    sha256_file,
)


def test_ordering_can_select_rows_for_a_short_draft_profile(tmp_path: Path) -> None:
    ordering_path = tmp_path / "ordering.json"
    ordering_path.write_text(
        json.dumps(
            {
                "physical_to_logical_map": [
                    [0, 1, 2, 3],
                    [1, 0, 3, 2],
                    [2, 3, 0, 1],
                    [3, 2, 1, 0],
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = load_hottest_first_ordering(
        ordering_path,
        num_layers=2,
        num_experts=4,
        layer_indices=(3, 1),
    )

    assert selected == [[3, 2, 1, 0], [1, 0, 3, 2]]


def test_hybrid_assignment_is_exact_cover_and_tier_balanced() -> None:
    frequency = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3], dtype=torch.int64)

    gpu_assignments, cpu_assignments = assign_hybrid_layer(
        frequency,
        hottest_first=list(range(8)),
        gpu_rank_counts=(1, 1),
        cpu_rank_counts=(3, 3),
    )

    assert sorted(
        expert_id
        for assignment in (*gpu_assignments, *cpu_assignments)
        for expert_id in assignment
    ) == list(range(8))
    assert [len(assignment) for assignment in gpu_assignments] == [1, 1]
    assert [len(assignment) for assignment in cpu_assignments] == [3, 3]
    assert [
        sum(int(frequency[expert_id]) for expert_id in assignment)
        for assignment in gpu_assignments
    ] == [10, 9]
    assert [
        sum(int(frequency[expert_id]) for expert_id in assignment)
        for assignment in cpu_assignments
    ] == [17, 16]


def test_hybrid_assignment_rejects_incomplete_capacity() -> None:
    with pytest.raises(ValueError, match="global expert count"):
        assign_hybrid_layer(
            torch.ones(8, dtype=torch.int64),
            hottest_first=list(range(8)),
            gpu_rank_counts=(1, 1),
            cpu_rank_counts=(2, 3),
        )


def test_hybrid_assignment_rejects_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="permutation"):
        assign_hybrid_layer(
            torch.ones(4, dtype=torch.int64),
            hottest_first=[0, 1, 1, 3],
            gpu_rank_counts=(1, 1),
            cpu_rank_counts=(1, 1),
        )


def test_profile_hot_selection_uses_frequency_and_deterministic_id_ties() -> None:
    selection = select_hottest_first(
        torch.tensor([2, 9, 9, 0], dtype=torch.int64),
        [3, 2, 0, 1],
        strategy="profile-hot",
        gpu_expert_count=2,
    )

    assert selection.hottest_first == (1, 2, 0, 3)
    assert not selection.used_primary_ordering_fallback
    assert not selection.used_fill_ordering_fallback
    assert selection.primary_profile_expert_count == 2
    assert selection.fill_profile_expert_count == 0
    assert selection.source_ordering_expert_count == 0


def test_profile_hot_selection_uses_ordering_for_zero_count_layer() -> None:
    selection = select_hottest_first(
        torch.zeros(4, dtype=torch.int64),
        [3, 1, 0, 2],
        strategy="profile-hot",
        gpu_expert_count=2,
    )

    assert selection.hottest_first == (3, 1, 0, 2)
    assert selection.used_primary_ordering_fallback
    assert selection.primary_profile_expert_count == 0
    assert selection.source_ordering_expert_count == 2


def test_profile_hot_prefix_profile_fill_builds_deterministic_union() -> None:
    selection = select_hottest_first(
        torch.tensor([100, 90, 80, 70, 60, 50, 40, 30], dtype=torch.int64),
        [1, 7, 0, 6, 5, 4, 3, 2],
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=2,
        fill_frequency=torch.tensor([1, 2, 3, 4, 5, 6, 90, 100], dtype=torch.int64),
    )

    # Experts 0 and 1 come from the live profile. The secondary-profile fill
    # contributes experts 7 and 6; source ordering only orders the unused tail.
    assert selection.hottest_first == (0, 1, 7, 6, 5, 4, 3, 2)
    assert not selection.used_primary_ordering_fallback
    assert not selection.used_fill_ordering_fallback
    assert selection.primary_profile_expert_count == 2
    assert selection.fill_profile_expert_count == 2
    assert selection.source_ordering_expert_count == 0


def test_profile_fill_preserves_stored_expert_id_tie_breaking() -> None:
    selection = select_hottest_first(
        torch.tensor([100, 90, 80, 70, 60, 50], dtype=torch.int64),
        [5, 4, 3, 2, 1, 0],
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=2,
        fill_frequency=torch.tensor([0, 0, 7, 7, 0, 0], dtype=torch.int64),
    )

    # The old profile-hot plan breaks nonzero frequency ties by expert ID,
    # independent of source ordering.
    assert selection.hottest_first[:4] == (0, 1, 2, 3)


def test_profile_fill_tracks_primary_and_fill_zero_fallbacks_independently() -> None:
    primary_zero = select_hottest_first(
        torch.zeros(6, dtype=torch.int64),
        [5, 3, 1, 4, 2, 0],
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=2,
        fill_frequency=torch.tensor([60, 50, 40, 30, 20, 10], dtype=torch.int64),
    )
    assert primary_zero.hottest_first[:4] == (5, 3, 0, 1)
    assert primary_zero.used_primary_ordering_fallback
    assert not primary_zero.used_fill_ordering_fallback
    assert primary_zero.primary_profile_expert_count == 0
    assert primary_zero.fill_profile_expert_count == 2
    assert primary_zero.source_ordering_expert_count == 2

    fill_zero = select_hottest_first(
        torch.tensor([60, 50, 40, 30, 20, 10], dtype=torch.int64),
        [5, 3, 1, 4, 2, 0],
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=2,
        fill_frequency=torch.zeros(6, dtype=torch.int64),
    )
    assert fill_zero.hottest_first[:4] == (0, 1, 5, 3)
    assert not fill_zero.used_primary_ordering_fallback
    assert fill_zero.used_fill_ordering_fallback
    assert fill_zero.primary_profile_expert_count == 2
    assert fill_zero.fill_profile_expert_count == 0
    assert fill_zero.source_ordering_expert_count == 2


def test_profile_fill_supports_zero_and_full_primary_prefix_boundaries() -> None:
    primary = torch.tensor([60, 50, 40, 30, 20, 10], dtype=torch.int64)
    fill = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
    ordering = [2, 1, 0, 5, 4, 3]

    fill_only = select_hottest_first(
        primary,
        ordering,
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=0,
        fill_frequency=fill,
    )
    primary_only = select_hottest_first(
        primary,
        ordering,
        strategy="profile-hot-prefix-profile-fill",
        gpu_expert_count=4,
        profile_hot_prefix_experts_per_layer=4,
        fill_frequency=fill,
    )

    assert fill_only.hottest_first[:4] == (5, 4, 3, 2)
    assert primary_only.hottest_first[:4] == (0, 1, 2, 3)


def test_profile_fill_rejects_frequency_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        select_hottest_first(
            torch.ones(6, dtype=torch.int64),
            list(range(6)),
            strategy="profile-hot-prefix-profile-fill",
            gpu_expert_count=4,
            profile_hot_prefix_experts_per_layer=2,
            fill_frequency=torch.ones(5, dtype=torch.int64),
        )


@pytest.mark.parametrize(
    ("strategy", "prefix_count", "provide_fill", "message"),
    [
        (
            "profile-hot-prefix-profile-fill",
            None,
            True,
            "requires --profile-hot-prefix-experts-per-layer",
        ),
        (
            "profile-hot-prefix-profile-fill",
            5,
            True,
            "between 0 and the total GPU expert count",
        ),
        (
            "profile-hot-prefix-profile-fill",
            2,
            False,
            "requires --gpu-fill-profile",
        ),
        (
            "profile-hot",
            2,
            False,
            "only valid with --gpu-selection profile-hot-prefix-profile-fill",
        ),
        (
            "profile-hot",
            None,
            True,
            "--gpu-fill-profile is only valid",
        ),
    ],
)
def test_profile_hot_prefix_configuration_is_strictly_validated(
    strategy: GpuSelectionStrategy,
    prefix_count: int | None,
    provide_fill: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_hottest_first(
            torch.ones(8, dtype=torch.int64),
            list(range(8)),
            strategy=strategy,
            gpu_expert_count=4,
            profile_hot_prefix_experts_per_layer=prefix_count,
            fill_frequency=(torch.ones(8, dtype=torch.int64) if provide_fill else None),
        )


@pytest.mark.parametrize(
    "invalid_counts",
    [
        torch.ones((1, 4), dtype=torch.float32),
        torch.ones((1, 4), dtype=torch.bool),
    ],
)
def test_reduce_profile_rejects_non_integer_counts(
    invalid_counts: torch.Tensor,
) -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        reduce_profile(invalid_counts)


def test_reduce_profile_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reduce_profile(torch.tensor([[1, 0, -1]], dtype=torch.int64))


def test_main_serializes_profile_hot_masks_and_fallback_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile_path = tmp_path / "profile.pt"
    ordering_path = tmp_path / "ordering.json"
    output_path = tmp_path / "hybrid.pt"
    frequency = torch.tensor(
        [
            [1, 100, 50, 80, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    torch.save({"logical_count": frequency}, profile_path)
    ordering_path.write_text(
        json.dumps(
            {
                "physical_to_logical_map": [
                    [7, 6, 5, 4, 3, 2, 1, 0],
                    [6, 4, 2, 0, 7, 5, 3, 1],
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_kt_hybrid_shard_plan.py",
            "--profile",
            str(profile_path),
            "--ordering",
            str(ordering_path),
            "--output",
            str(output_path),
            "--gpu-rank-counts",
            "1,1",
            "--cpu-rank-counts",
            "3,3",
        ],
    )

    main()

    plan = torch.load(output_path, map_location="cpu", weights_only=True)
    gpu_masks = plan["gpu_experts_mask_by_rank"]
    assert tuple(gpu_masks.shape) == (2, 2, 8)
    assert set(torch.where(gpu_masks[:, 0].any(dim=0))[0].tolist()) == {1, 3}
    assert set(torch.where(gpu_masks[:, 1].any(dim=0))[0].tolist()) == {4, 6}
    assert plan["gpu_selection_strategy"] == "profile-hot"
    torch.testing.assert_close(
        plan["gpu_primary_profile_selected_counts_by_layer"],
        torch.tensor([2, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_fill_profile_selected_counts_by_layer"],
        torch.tensor([0, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_source_ordering_selected_counts_by_layer"],
        torch.tensor([0, 2], dtype=torch.int64),
    )
    assert plan["zero_count_layer_fallback"] == "source_ordering"
    torch.testing.assert_close(
        plan["zero_count_fallback_layers"], torch.tensor([1], dtype=torch.int64)
    )


def test_main_serializes_profile_hot_prefix_receipt_and_real_profile_balance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile_path = tmp_path / "profile.pt"
    fill_profile_path = tmp_path / "fill-profile.pt"
    ordering_path = tmp_path / "ordering.json"
    output_path = tmp_path / "hybrid.pt"
    frequency = torch.tensor([[100, 90, 80, 70, 60, 50, 40, 30]], dtype=torch.int64)
    torch.save({"logical_count": frequency}, profile_path)
    torch.save(
        {
            "logical_count": torch.tensor(
                [[1, 2, 3, 4, 5, 6, 90, 100]], dtype=torch.int64
            )
        },
        fill_profile_path,
    )
    ordering_path.write_text(
        json.dumps({"physical_to_logical_map": [[1, 7, 0, 6, 5, 4, 3, 2]]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_kt_hybrid_shard_plan.py",
            "--profile",
            str(profile_path),
            "--ordering",
            str(ordering_path),
            "--output",
            str(output_path),
            "--gpu-rank-counts",
            "2,2",
            "--cpu-rank-counts",
            "2,2",
            "--gpu-selection",
            "profile-hot-prefix-profile-fill",
            "--profile-hot-prefix-experts-per-layer",
            "2",
            "--gpu-fill-profile",
            str(fill_profile_path),
        ],
    )

    main()

    plan = torch.load(output_path, map_location="cpu", weights_only=True)
    gpu_masks = plan["gpu_experts_mask_by_rank"]
    assert set(torch.where(gpu_masks[:, 0].any(dim=0))[0].tolist()) == {0, 1, 6, 7}
    # Real counts, rather than source positions or fabricated blend scores,
    # balance both GPU ranks to 130 calls: {100, 30} and {90, 40}.
    assert [int(frequency[0][gpu_masks[rank, 0]].sum()) for rank in range(2)] == [
        130,
        130,
    ]
    assert plan["gpu_selection_strategy"] == "profile-hot-prefix-profile-fill"
    torch.testing.assert_close(
        plan["gpu_profile_hot_prefix_experts_per_layer"],
        torch.tensor(2, dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_primary_profile_selected_counts_by_layer"],
        torch.tensor([2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_fill_profile_selected_counts_by_layer"],
        torch.tensor([2], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_source_ordering_selected_counts_by_layer"],
        torch.tensor([0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan["gpu_union_expert_count"], torch.tensor(4, dtype=torch.int64)
    )
    assert plan["source_fill_profile"] == str(fill_profile_path.resolve())
    assert plan["source_profile_sha256"] == sha256_file(profile_path)
    assert plan["source_fill_profile_sha256"] == sha256_file(fill_profile_path)
    assert plan["source_ordering_sha256"] == sha256_file(ordering_path)


def test_main_rejects_prefix_count_for_an_unrelated_selection_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pt"
    ordering_path = tmp_path / "ordering.json"
    output_path = tmp_path / "hybrid.pt"
    torch.save({"logical_count": torch.ones((1, 4), dtype=torch.int64)}, profile_path)
    ordering_path.write_text(
        json.dumps({"physical_to_logical_map": [[0, 1, 2, 3]]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dsv4_kt_hybrid_shard_plan.py",
            "--profile",
            str(profile_path),
            "--ordering",
            str(ordering_path),
            "--output",
            str(output_path),
            "--gpu-rank-counts",
            "1,1",
            "--cpu-rank-counts",
            "1,1",
            "--gpu-selection",
            "profile-hot",
            "--profile-hot-prefix-experts-per-layer",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert "only valid with --gpu-selection" in capsys.readouterr().err
    assert not output_path.exists()
