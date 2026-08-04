from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_dsv4_oscar_int2_long_prompts import build_long_prompt_document
from scripts.dsv4_oscar_int2_calibration import _prompt_provenance, sha256_file
from scripts.dsv4_oscar_int2_runtime_capture import (
    _default_row_limits,
    _prompt_document,
    projected_raw_tensor_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LONG_PROMPT_MANIFEST = (
    REPO_ROOT / "scripts" / "data" / "dsv4_oscar_int2_prompts_long.json"
)
EXPECTED_LONG_MANIFEST_SHA256 = (
    "faada3ce6083b110c513f2370586503140c89d3741da328bfafffc1a6e501434"
)


def _source_document() -> dict[str, object]:
    prompts = []
    for split, count in (("train", 8), ("heldout", 4)):
        for index in range(count):
            text = f"{split} source case {index} " + "evidence " * 32
            prompts.append(
                {
                    "id": f"{split}-{index}",
                    "split": split,
                    "token_count": len(text.split()),
                    "text": text,
                }
            )
    return {
        "format": "dsv4-oscar-calibration-prompts",
        "format_version": 1,
        "prompts": prompts,
    }


def test_long_prompt_builder_keeps_train_and_heldout_text_disjoint() -> None:
    document = build_long_prompt_document(
        _source_document(), count_tokens=lambda text: len(text.split()) * 8
    )
    prompts = document["prompts"]
    assert isinstance(prompts, list)
    assert len(prompts) == 12
    train_text = "\n".join(
        prompt["text"] for prompt in prompts if prompt["split"] == "train"
    )
    heldout_text = "\n".join(
        prompt["text"] for prompt in prompts if prompt["split"] == "heldout"
    )
    assert "train source case" in train_text
    assert "heldout source case" not in train_text
    assert "heldout source case" in heldout_text
    assert "train source case" not in heldout_text
    for prompt in prompts:
        digest = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        assert prompt["id"] == f"{prompt['split']}-{digest[:16]}"
        assert prompt["token_count"] >= 768


def test_long_prompt_builder_rejects_wrong_split_geometry() -> None:
    document = _source_document()
    prompts = document["prompts"]
    assert isinstance(prompts, list)
    prompts.pop()
    with pytest.raises(ValueError, match="eight train and four heldout"):
        build_long_prompt_document(document, count_tokens=lambda text: 1024)


def test_checked_in_long_prompt_corpus_has_stable_capture_provenance() -> None:
    assert sha256_file(LONG_PROMPT_MANIFEST) == EXPECTED_LONG_MANIFEST_SHA256
    document = json.loads(LONG_PROMPT_MANIFEST.read_text(encoding="utf-8"))
    prompts = document["prompts"]
    counts = [prompt["token_count"] for prompt in prompts]
    assert len(prompts) == 12
    assert sum(prompt["split"] == "train" for prompt in prompts) == 8
    assert sum(prompt["split"] == "heldout" for prompt in prompts) == 4
    assert sum(counts) == 13_172
    assert min(counts) == 1_069
    assert max(counts) == 1_127
    for prompt in prompts:
        digest = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        assert prompt["id"] == f"{prompt['split']}-{digest[:16]}"
    prompt_splits, total_tokens = _prompt_provenance(
        LONG_PROMPT_MANIFEST, EXPECTED_LONG_MANIFEST_SHA256
    )
    assert total_tokens == 13_172
    limits = _default_row_limits(prompt_splits)
    assert projected_raw_tensor_bytes(prompt_splits, limits) < 256 * 1024 * 1024
    _, runtime_splits = _prompt_document(LONG_PROMPT_MANIFEST)
    assert runtime_splits == prompt_splits
