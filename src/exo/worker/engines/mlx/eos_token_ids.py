import json
from pathlib import Path
from typing import cast

from exo.shared.types.common import ModelId


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """Return explicit EOS token IDs required by known model families."""
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    if "glm-5" in model_id_lower or "glm-4.7-flash" in model_id_lower:
        # <|endoftext|>, <|user|>, and <|observation|>.
        return [154820, 154827, 154829]
    if "glm" in model_id_lower:
        # The original GLM-4.7 and older tokenizer vocabulary.
        return [151336, 151329, 151338]
    if "gpt-oss" in model_id_lower:
        return [200002, 200012]
    if any(
        family in model_id_lower
        for family in ("qwen3.5", "qwen-3.5", "qwen3.6", "qwen-3.6")
    ):
        # <|im_end|> and <|endoftext|>.
        return [248046, 248044]
    if "qwen3-coder" in model_id_lower or "qwen3-next" in model_id_lower:
        # Qwen3 Coder/Next generation configs declare both tokens as EOS.
        return [151645, 151643]
    if "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def get_configured_eos_token_ids(model_path: Path) -> list[int] | None:
    """Read EOS IDs declared by a local checkpoint, preferring generation config."""
    for config_name in ("generation_config.json", "config.json"):
        try:
            config_value = cast(
                object, json.loads((model_path / config_name).read_text())
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(config_value, dict):
            continue

        config = cast(dict[str, object], config_value)
        eos_value = config.get("eos_token_id")
        if eos_value is None and isinstance(config.get("text_config"), dict):
            text_config = cast(dict[str, object], config["text_config"])
            eos_value = text_config.get("eos_token_id")

        if isinstance(eos_value, int) and not isinstance(eos_value, bool):
            return [eos_value]
        if isinstance(eos_value, list):
            raw_eos_token_ids = cast(list[object], eos_value)
            eos_token_ids = [
                token_id
                for token_id in raw_eos_token_ids
                if isinstance(token_id, int) and not isinstance(token_id, bool)
            ]
            if eos_token_ids and len(eos_token_ids) == len(raw_eos_token_ids):
                return list(dict.fromkeys(eos_token_ids))
    return None
