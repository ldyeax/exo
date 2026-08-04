from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.dsv4_oscar_int2_calibration import _prompt_provenance, sha256_file
from scripts.dsv4_oscar_int2_runtime_capture import (
    _default_row_limits,
    _prompt_document,
    projected_raw_tensor_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_MANIFEST = REPO_ROOT / "scripts" / "data" / "dsv4_oscar_int2_prompts.json"
EXPECTED_MANIFEST_SHA256 = (
    "66a3b469f2dc1a2da230d413b521e3c760d1086b94f41ef69bbe82bd2971e1f3"
)
# Raw-text counts from the checked-in checkpoint's tokenizer.json, without
# chat-template special tokens.  The separate whitespace assertion remains a
# tokenizer-independent lower bound for every C128 capture observation.
EXPECTED_CHECKPOINT_TOKEN_COUNTS = {
    "train-50f64a9ece076520": 229,
    "train-d032c0ec1bdaedb6": 245,
    "train-1a2fc72aba993a74": 235,
    "train-d7fb4f9a884632f6": 239,
    "train-4ebc6bae97b4bbfc": 235,
    "train-73a24565c4befdec": 243,
    "train-58e22032ccef74ba": 236,
    "train-655d77ec324409f3": 246,
    "heldout-eff2369de950f15e": 232,
    "heldout-66941f1a1a5bd499": 223,
    "heldout-57184f566606016b": 226,
    "heldout-fcb6c4c661cec783": 230,
}


def test_checked_in_oscar_prompts_have_stable_real_capture_provenance() -> None:
    assert sha256_file(PROMPT_MANIFEST) == EXPECTED_MANIFEST_SHA256
    document = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    assert set(document) == {"format", "format_version", "prompts"}
    assert document["format"] == "dsv4-oscar-calibration-prompts"
    assert document["format_version"] == 1

    prompts = document["prompts"]
    assert len(prompts) == 12
    assert sum(prompt["split"] == "train" for prompt in prompts) == 8
    assert sum(prompt["split"] == "heldout" for prompt in prompts) == 4
    assert len({prompt["id"] for prompt in prompts}) == len(prompts)
    assert sum(prompt["token_count"] for prompt in prompts) == 2819

    for prompt in prompts:
        assert set(prompt) == {"id", "split", "token_count", "text"}
        whitespace_tokens = prompt["text"].split()
        assert len(whitespace_tokens) >= 128
        assert prompt["token_count"] == EXPECTED_CHECKPOINT_TOKEN_COUNTS[prompt["id"]]
        text_digest = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        assert prompt["id"] == f"{prompt['split']}-{text_digest[:16]}"
        assert "http://" not in prompt["text"]
        assert "https://" not in prompt["text"]

    prompt_splits, total_tokens = _prompt_provenance(
        PROMPT_MANIFEST, EXPECTED_MANIFEST_SHA256
    )
    assert len(prompt_splits) == 12
    assert total_tokens == 2819


def test_checked_in_oscar_prompts_fit_bounded_initial_tp2_capture() -> None:
    _, prompt_splits = _prompt_document(PROMPT_MANIFEST)
    limits = _default_row_limits(prompt_splits)
    assert limits == {
        "attention_query_nope": {"train": 4, "heldout": 4},
        "swa_latent": {"train": 4, "heldout": 4},
        "compressed_latent": {"train": 128, "heldout": 128},
        "c4_scorer_query": {"train": 4, "heldout": 4},
        "c4_scorer_key": {"train": 128, "heldout": 128},
    }
    assert projected_raw_tensor_bytes(prompt_splits, limits) < 256 * 1024 * 1024
