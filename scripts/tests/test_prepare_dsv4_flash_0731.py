from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prepare_dsv4_flash_0731 as prepare


def _ordering_payload() -> dict[str, object]:
    return {
        "physical_to_logical_map": [
            list(range(prepare.EXPERT_COUNT))
            for _ in range(prepare.ROUTED_LAYER_COUNT)
        ]
    }


def test_validate_ordering_accepts_complete_permutations(tmp_path: Path) -> None:
    path = tmp_path / "ordering.json"
    path.write_text(json.dumps(_ordering_payload()), encoding="utf-8")

    ordering = prepare.validate_ordering(path)

    assert len(ordering) == prepare.ROUTED_LAYER_COUNT
    assert ordering[0][:4] == (0, 1, 2, 3)


def test_validate_ordering_rejects_duplicates(tmp_path: Path) -> None:
    payload = _ordering_payload()
    rows = payload["physical_to_logical_map"]
    assert isinstance(rows, list)
    first_row = rows[0]
    assert isinstance(first_row, list)
    first_row[-1] = first_row[0]
    path = tmp_path / "ordering.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(prepare.PreparationError, match="complete permutation"):
        prepare.validate_ordering(path)


def test_write_mask_plan_selects_prefix_per_layer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    ordering = tuple(
        tuple(reversed(range(prepare.EXPERT_COUNT)))
        for _ in range(prepare.ROUTED_LAYER_COUNT)
    )
    output = tmp_path / "mask.pt"

    prepare.write_mask_plan(
        output,
        ordering,
        ordering_sha256="0" * 64,
        gpu_experts_per_layer=3,
    )

    payload = torch.load(output, map_location="cpu", weights_only=True)
    mask = payload["gpu_experts_mask"]
    assert tuple(mask.shape) == (prepare.ROUTED_LAYER_COUNT, prepare.EXPERT_COUNT)
    assert int(mask.sum().item()) == prepare.ROUTED_LAYER_COUNT * 3
    assert mask[0, -3:].tolist() == [True, True, True]


def test_validate_model_rejects_wrong_revision(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": prepare.ROUTED_LAYER_COUNT,
                "n_routed_experts": prepare.EXPERT_COUNT,
                "num_experts_per_tok": 6,
                "num_nextn_predict_layers": 1,
                "dspark_block_size": 5,
                "max_position_embeddings": 1_048_576,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": prepare.MODEL_WEIGHT_BYTES}, "weight_map": {}}),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "config.json.metadata").write_text(
        "wrong-revision\n", encoding="utf-8"
    )

    with pytest.raises(prepare.PreparationError, match="revision mismatch"):
        prepare.validate_model(tmp_path)
