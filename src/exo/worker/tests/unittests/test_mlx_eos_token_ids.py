import pytest

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.eos_token_ids import get_eos_token_ids_for_model


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
    ],
)
def test_glm_eos_token_ids_match_tokenizer_family(
    model_id: ModelId, expected_eos_token_ids: list[int]
) -> None:
    assert get_eos_token_ids_for_model(model_id) == expected_eos_token_ids
