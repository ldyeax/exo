import json
from pathlib import Path

import pytest

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.eos_token_ids import (
    get_configured_eos_token_ids,
    get_eos_token_ids_for_model,
)


@pytest.mark.parametrize(
    ("model_id", "expected_eos_token_ids"),
    [
        (
            ModelId("mlx-community/GLM-4.7-Flash-4bit"),
            [154820, 154827, 154829],
        ),
        (
            ModelId("mlx-community/GLM-4.7-4bit"),
            [151336, 151329, 151338],
        ),
        (
            ModelId("mlx-community/GLM-5-MXFP4-Q8"),
            [154820, 154827, 154829],
        ),
        (
            ModelId("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
            [151645, 151643],
        ),
        (
            ModelId("mlx-community/Qwen3-Coder-Next-4bit"),
            [151645, 151643],
        ),
        (
            ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
            [151645, 151643],
        ),
    ],
)
def test_eos_token_ids_match_model_generation_config(
    model_id: ModelId, expected_eos_token_ids: list[int]
) -> None:
    assert get_eos_token_ids_for_model(model_id) == expected_eos_token_ids


def test_generation_config_eos_ids_take_precedence(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [128001, 128008, 128009]})
    )
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 128009}))

    assert get_configured_eos_token_ids(tmp_path) == [128001, 128008, 128009]


def test_model_config_supports_nested_and_scalar_eos_ids(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 248044}})
    )

    assert get_configured_eos_token_ids(tmp_path) == [248044]


def test_invalid_or_missing_configured_eos_ids_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text("not json")
    (tmp_path / "config.json").write_text(
        json.dumps({"eos_token_id": [151645, "151643"]})
    )

    assert get_configured_eos_token_ids(tmp_path) is None
