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
    if "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None
