import pytest

from exo.worker.engines.mlx.vision_policy import (
    EXO_MLX_VISION_LOADING,
    get_mlx_vision_loading_mode,
)


@pytest.mark.parametrize("mode", ["eager", "lazy", "disabled"])
def test_mlx_vision_loading_modes_are_accepted(mode: str) -> None:
    assert get_mlx_vision_loading_mode({EXO_MLX_VISION_LOADING: mode}) == mode


def test_mlx_vision_loading_defaults_to_eager() -> None:
    assert get_mlx_vision_loading_mode({}) == "eager"


def test_mlx_vision_loading_is_case_insensitive() -> None:
    assert get_mlx_vision_loading_mode({EXO_MLX_VISION_LOADING: " Lazy "}) == "lazy"


def test_invalid_mlx_vision_loading_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match=EXO_MLX_VISION_LOADING):
        get_mlx_vision_loading_mode({EXO_MLX_VISION_LOADING: "sometimes"})
