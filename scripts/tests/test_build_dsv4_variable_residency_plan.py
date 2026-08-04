from pathlib import Path

import pytest
import torch

from scripts.build_dsv4_critical_tail_plan_sweep import load_prompt_cost_profile
from scripts.build_dsv4_variable_residency_plan import (
    BaselinePlacement,
    VariablePlacement,
    build_variable_placement,
    load_baseline_placement,
)
from scripts.compact_dsv4_expert_profiles import compact_profile


def _baseline() -> BaselinePlacement:
    gpu_masks = torch.zeros((2, 3, 6), dtype=torch.bool)
    gpu_masks[0, :, 0] = True
    gpu_masks[1, :, 3] = True
    return BaselinePlacement(
        gpu_masks_by_rank=gpu_masks,
        cpu_expert_ids_by_rank=(
            torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.int64),
            torch.tensor([[4, 5], [4, 5], [4, 5]], dtype=torch.int64),
        ),
    )


def _skewed_prompt_cost() -> torch.Tensor:
    cost = torch.zeros((2, 3, 6), dtype=torch.float64)
    # Rank 0 owns the critical layer-0 expert, while rank 1 owns the critical
    # layer-1 expert. Independent widths should spend one slot on each without
    # wasting the paired slot in the other rank/layer.
    cost[:, 0, 1] = torch.tensor([10.0, 8.0])
    cost[:, 0, 2] = 1.0
    cost[:, 0, 4:6] = 0.5
    cost[:, 1, 4] = torch.tensor([9.0, 11.0])
    cost[:, 1, 5] = 1.0
    cost[:, 1, 1:3] = 0.5
    cost[:, 2, 1:3] = 1.0
    cost[:, 2, 4:6] = 1.0
    return cost


def _assert_exact_cover(placement: VariablePlacement) -> None:
    gpu_masks = placement.gpu_masks_by_rank
    padded_cpu = placement.cpu_expert_ids_padded_by_rank
    cpu_counts = placement.cpu_rank_counts_by_layer
    expected = list(range(gpu_masks.shape[2]))
    for layer in range(gpu_masks.shape[1]):
        assigned: list[int] = []
        for rank in range(2):
            assigned.extend(torch.where(gpu_masks[rank, layer])[0].tolist())
            count = int(cpu_counts[rank, layer])
            assigned.extend(padded_cpu[rank, layer, :count].tolist())
            assert torch.all(padded_cpu[rank, layer, count:] == -1)
        assert sorted(assigned) == expected


def test_independent_widths_concentrate_each_rank_on_its_critical_layer() -> None:
    placement = build_variable_placement(
        _skewed_prompt_cost(),
        _baseline(),
        extra_bytes_per_rank=10,
        expert_weight_bytes=10,
        max_extra_experts_per_layer=8,
        tail_weight=1.0,
    )

    assert len(placement.promotions) == 2
    assert {
        (item.rank, item.layer, item.expert_id) for item in placement.promotions
    } == {
        (0, 0, 1),
        (1, 1, 4),
    }
    assert placement.gpu_rank_counts_by_layer.tolist() == [
        [2, 1, 1],
        [1, 2, 1],
    ]
    assert torch.all(placement.critical_tail_after < placement.critical_tail_before)
    _assert_exact_cover(placement)


def test_symmetric_mode_keeps_rank_widths_equal() -> None:
    placement = build_variable_placement(
        _skewed_prompt_cost(),
        _baseline(),
        extra_bytes_per_rank=10,
        expert_weight_bytes=10,
        max_extra_experts_per_layer=8,
        tail_weight=1.0,
        rank_symmetric_widths=True,
    )

    assert len(placement.promotions) == 2
    torch.testing.assert_close(
        placement.gpu_rank_counts_by_layer[0],
        placement.gpu_rank_counts_by_layer[1],
    )
    _assert_exact_cover(placement)


def test_byte_budget_and_per_rank_layer_ceiling_are_hard_bounds() -> None:
    placement = build_variable_placement(
        _skewed_prompt_cost(),
        _baseline(),
        extra_bytes_per_rank=29,
        expert_weight_bytes=10,
        max_extra_experts_per_layer=1,
        tail_weight=0.0,
    )

    promotions_by_rank = [
        sum(item.rank == rank for item in placement.promotions) for rank in range(2)
    ]
    assert promotions_by_rank == [2, 2]
    extra_width = placement.gpu_rank_counts_by_layer - 1
    assert int(extra_width.max()) <= 1
    _assert_exact_cover(placement)


def test_baseline_loader_rejects_rank_asymmetric_widths(tmp_path: Path) -> None:
    baseline = _baseline()
    baseline.gpu_masks_by_rank[1, 0, 4] = True
    path = tmp_path / "asymmetric.pt"
    torch.save(
        {
            "gpu_experts_mask_by_rank": baseline.gpu_masks_by_rank,
            "cpu_expert_ids_by_rank": list(baseline.cpu_expert_ids_by_rank),
        },
        path,
    )

    with pytest.raises(ValueError, match="equal GPU widths"):
        load_baseline_placement(path)


def test_zero_budget_preserves_baseline() -> None:
    baseline = _baseline()
    placement = build_variable_placement(
        _skewed_prompt_cost(),
        baseline,
        extra_bytes_per_rank=9,
        expert_weight_bytes=10,
    )

    assert not placement.promotions
    torch.testing.assert_close(placement.gpu_masks_by_rank, baseline.gpu_masks_by_rank)
    torch.testing.assert_close(
        placement.critical_tail_after, placement.critical_tail_before
    )
    _assert_exact_cover(placement)


def test_compact_profile_preserves_sampled_route_costs(tmp_path: Path) -> None:
    logical_count = torch.tensor(
        [
            [[2, 1, 0, 0], [0, 1, 2, 0]],
            [[1, 0, 2, 0], [1, 1, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.int32,
    )
    source = tmp_path / "raw.pt"
    compact = tmp_path / "compact.pt"
    torch.save({"logical_count": logical_count}, source)
    receipt = compact_profile(source, compact, routes_per_layer=3)

    raw_profile = load_prompt_cost_profile(
        "raw",
        source,
        routes_per_layer=3,
        active_expert_cost=1.0,
        additional_row_cost=0.25,
    )
    compact_profile_cost = load_prompt_cost_profile(
        "compact",
        compact,
        routes_per_layer=3,
        active_expert_cost=1.0,
        additional_row_cost=0.25,
    )

    assert receipt["sample_count"] == 2
    assert compact_profile_cost.source_kind == "compact_sampled_logical_count"
    torch.testing.assert_close(compact_profile_cost.raw_cost, raw_profile.raw_cost)
    torch.testing.assert_close(
        compact_profile_cost.normalized_cost, raw_profile.normalized_cost
    )
