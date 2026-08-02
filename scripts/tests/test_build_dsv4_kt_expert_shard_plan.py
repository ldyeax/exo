from __future__ import annotations

import pytest
import torch

from scripts.build_dsv4_kt_expert_shard_plan import assign_layer_balanced


def test_balanced_assignment_is_lossless_capacity_bounded_and_load_balanced() -> None:
    frequency = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3], dtype=torch.int64)

    assignments = assign_layer_balanced(frequency, (4, 4))

    assert sorted(expert for rank in assignments for expert in rank) == list(range(8))
    assert [len(rank) for rank in assignments] == [4, 4]
    loads = [sum(int(frequency[expert]) for expert in rank) for rank in assignments]
    assert loads == [26, 26]


def test_balanced_assignment_rejects_incomplete_capacity() -> None:
    with pytest.raises(ValueError, match="rank counts sum"):
        assign_layer_balanced(torch.ones(4, dtype=torch.int64), (1, 2))
