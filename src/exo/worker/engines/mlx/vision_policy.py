import os
from collections.abc import Mapping
from typing import Literal

EXO_MLX_VISION_LOADING = "EXO_MLX_VISION_LOADING"
MlxVisionLoadingMode = Literal["eager", "lazy", "disabled"]
_VALID_VISION_LOADING_MODES = frozenset(("eager", "lazy", "disabled"))


def get_mlx_vision_loading_mode(
    environment: Mapping[str, str] | None = None,
) -> MlxVisionLoadingMode:
    values = os.environ if environment is None else environment
    mode = values.get(EXO_MLX_VISION_LOADING, "eager").strip().lower()
    if mode not in _VALID_VISION_LOADING_MODES:
        valid_modes = ", ".join(sorted(_VALID_VISION_LOADING_MODES))
        raise ValueError(
            f"{EXO_MLX_VISION_LOADING} must be one of: {valid_modes}; got {mode!r}"
        )
    return mode
