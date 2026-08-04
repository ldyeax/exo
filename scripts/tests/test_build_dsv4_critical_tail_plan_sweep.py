from pathlib import Path

import pytest
import torch

from scripts.build_dsv4_critical_tail_plan_sweep import (
    assign_for_critical_tail,
    build_placement_variant,
    load_prompt_cost_profile,
)


def test_recorder_profile_uses_distinct_expert_and_additional_row_cost(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "routes.pt"
    torch.save(
        {
            "records": [
                {
                    "forward_pass_id": 1,
                    "rank": 0,
                    "gatherer_key": "primary",
                    "global_physical_count": torch.tensor(
                        [[1, 0, 2, 1]], dtype=torch.int32
                    ),
                },
                {
                    "forward_pass_id": 2,
                    "rank": 0,
                    "gatherer_key": "primary",
                    "global_physical_count": torch.tensor(
                        [[20, 0, 0, 0]], dtype=torch.int32
                    ),
                },
            ],
            "last_physical_to_logical_map": torch.tensor(
                [[2, 0, 3, 1]], dtype=torch.int64
            ),
        },
        profile_path,
    )

    profile = load_prompt_cost_profile(
        "coding",
        profile_path,
        routes_per_layer=4,
        active_expert_cost=1.0,
        additional_row_cost=0.25,
    )

    assert profile.sample_count == 1
    assert profile.source_kind == "expert_distribution_records"
    torch.testing.assert_close(
        profile.raw_cost,
        torch.tensor([[0.0, 1.0, 1.0, 1.25]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        profile.normalized_cost.sum(dim=1), torch.ones(1, dtype=torch.float64)
    )


def test_sampled_logical_profile_filters_zero_draft_and_prefill_samples(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "sampled-routes.pt"
    decode_first = torch.tensor([[1, 0, 2, 1], [0, 4, 0, 0]], dtype=torch.int64)
    decode_second = torch.tensor([[0, 2, 0, 2], [1, 1, 1, 1]], dtype=torch.int64)
    torch.save(
        {
            "logical_count": torch.stack(
                [
                    torch.zeros_like(decode_first),
                    decode_first,
                    torch.full_like(decode_first, 20),
                    torch.stack([decode_first[0], torch.full((4,), 20)]),
                    decode_second,
                ]
            )
        },
        profile_path,
    )

    profile = load_prompt_cost_profile(
        "mixed-phases",
        profile_path,
        routes_per_layer=4,
        active_expert_cost=1.0,
        additional_row_cost=0.25,
    )

    assert profile.sample_count == 2
    assert profile.source_kind == "sampled_logical_count"
    torch.testing.assert_close(
        profile.raw_cost,
        torch.tensor(
            [[0.5, 0.625, 0.625, 1.125], [0.5, 1.375, 0.5, 0.5]],
            dtype=torch.float64,
        ),
    )


def test_sampled_logical_profile_rejects_an_empty_route_shape(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "prefill-only.pt"
    torch.save(
        {"logical_count": torch.full((2, 3, 4), 8, dtype=torch.int64)},
        profile_path,
    )

    with pytest.raises(
        ValueError,
        match=r"no logical_count samples matching routes_per_layer=4",
    ):
        load_prompt_cost_profile(
            "prefill-only",
            profile_path,
            routes_per_layer=4,
            active_expert_cost=1.0,
            additional_row_cost=0.25,
        )


def test_critical_tail_assignment_balances_each_prompt() -> None:
    prompt_cost = torch.tensor(
        [[10.0, 9.0, 0.0, 0.0], [0.0, 0.0, 10.0, 9.0]],
        dtype=torch.float64,
    )

    assignments = assign_for_critical_tail(
        [0, 1, 2, 3], prompt_cost, (2, 2), tail_weight=1.0
    )

    assert sorted((*assignments[0], *assignments[1])) == [0, 1, 2, 3]
    rank_loads = torch.tensor(
        [
            [prompt_cost[prompt, assignment].sum() for assignment in assignments]
            for prompt in range(2)
        ]
    )
    assert rank_loads.max(dim=1).values.tolist() == [10.0, 10.0]


def test_placement_variant_is_an_exact_rank_local_cover() -> None:
    prompt_cost = torch.tensor(
        [
            [
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 3, 5, 7, 8, 6, 4, 2],
                [2, 4, 6, 8, 7, 5, 3, 1],
            ],
        ],
        dtype=torch.float64,
    )
    prompt_cost /= prompt_cost.sum(dim=2, keepdim=True)
    fill_frequency = torch.ones((2, 8), dtype=torch.int64)
    ordering = [list(range(8)), list(reversed(range(8)))]

    variant = build_placement_variant(
        prompt_cost,
        fill_frequency,
        ordering,
        gpu_experts_per_rank=1,
        hot_prefix_count=2,
        tail_weight=1.0,
    )

    assert tuple(variant.gpu_masks_by_rank.shape) == (2, 2, 8)
    for layer in range(2):
        assigned = []
        for rank in range(2):
            assigned.extend(variant.gpu_expert_ids_by_rank[rank][layer].tolist())
            assigned.extend(variant.cpu_expert_ids_by_rank[rank][layer].tolist())
        assert sorted(assigned) == list(range(8))
    assert variant.metrics["worst_layer_rank_skew"] >= 0


def test_baseline_constraint_preserves_immutable_rank_ownership() -> None:
    prompt_cost = torch.tensor(
        [
            [
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ]
        ],
        dtype=torch.float64,
    )
    prompt_cost /= prompt_cost.sum(dim=2, keepdim=True)
    ownership = torch.zeros((2, 2, 8), dtype=torch.bool)
    ownership[0, :, :4] = True
    ownership[1, :, 4:] = True

    variant = build_placement_variant(
        prompt_cost,
        torch.ones((2, 8), dtype=torch.int64),
        [list(range(8)), list(reversed(range(8)))],
        gpu_experts_per_rank=1,
        hot_prefix_count=2,
        tail_weight=1.0,
        ownership_by_rank=ownership,
    )

    for layer in range(2):
        for rank in range(2):
            candidate_ownership = torch.zeros(8, dtype=torch.bool)
            candidate_ownership[variant.gpu_expert_ids_by_rank[rank][layer]] = True
            candidate_ownership[variant.cpu_expert_ids_by_rank[rank][layer]] = True
            torch.testing.assert_close(candidate_ownership, ownership[rank, layer])
