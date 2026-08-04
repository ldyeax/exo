#!/usr/bin/env python3
"""Build the deterministic long-form OSCAR calibration prompt corpus.

The short source corpus is useful for capture-pipeline smoke tests, but a
roughly 256-token request produces too few C128 history boundaries to calibrate
the cache honestly.  This builder combines only prompts from the same split so
the long held-out requests never contain training text.  It records tokenizer
counts and content-derived identifiers in the generated version-1 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, cast

from tokenizers import Tokenizer

PROMPT_FORMAT: Final = "dsv4-oscar-calibration-prompts"
PROMPT_FORMAT_VERSION: Final = 1
SPLITS: Final = ("train", "heldout")
TRAIN_OFFSETS: Final = (0, 1, 3, 5)
HELDOUT_OFFSETS: Final = (0, 1, 2, 3)


def _load_document(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("source prompt manifest must contain one JSON object")
    return cast(dict[str, object], loaded)


def _validate_source_prompts(document: dict[str, object]) -> list[dict[str, object]]:
    if (
        document.get("format") != PROMPT_FORMAT
        or document.get("format_version") != PROMPT_FORMAT_VERSION
        or set(document) != {"format", "format_version", "prompts"}
    ):
        raise ValueError("source prompt manifest does not use the version-1 ABI")
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("source prompt manifest contains no prompts")
    prompts: list[dict[str, object]] = []
    for raw_prompt in raw_prompts:
        if not isinstance(raw_prompt, dict):
            raise TypeError("source prompt entries must be JSON objects")
        if set(raw_prompt) != {"id", "split", "token_count", "text"}:
            raise ValueError("source prompt entry fields do not match version 1")
        split = raw_prompt.get("split")
        text = raw_prompt.get("text")
        if split not in SPLITS or not isinstance(text, str) or not text:
            raise ValueError("source prompt entry has an invalid split or text")
        prompts.append(cast(dict[str, object], raw_prompt))
    return prompts


def build_long_prompt_document(
    source_document: dict[str, object],
    *,
    count_tokens: Callable[[str], int],
) -> dict[str, object]:
    """Combine four same-split tasks into each deterministic long request."""

    source_prompts = _validate_source_prompts(source_document)
    prompts_by_split = {
        split: [prompt for prompt in source_prompts if prompt["split"] == split]
        for split in SPLITS
    }
    if len(prompts_by_split["train"]) != 8 or len(prompts_by_split["heldout"]) != 4:
        raise ValueError("OSCAR long corpus requires exactly eight train and four heldout prompts")

    generated_prompts: list[dict[str, object]] = []
    for split in SPLITS:
        split_prompts = prompts_by_split[split]
        offsets = TRAIN_OFFSETS if split == "train" else HELDOUT_OFFSETS
        for prompt_index in range(len(split_prompts)):
            selected = [
                split_prompts[(prompt_index + offset) % len(split_prompts)]
                for offset in offsets
            ]
            sections = [
                (
                    f"Calibration case {section_index + 1} of {len(selected)}. "
                    "Treat this as an independent repository task; preserve its "
                    "constraints and reason through concrete evidence before the "
                    "next case.\n\n"
                    f"{selected_prompt['text']}"
                )
                for section_index, selected_prompt in enumerate(selected)
            ]
            text = (
                "Work through the following four independent engineering cases in "
                "order. Keep their assumptions separate, identify cross-case lessons "
                "only after each analysis, and finish with a short synthesis that does "
                "not invent tool results.\n\n"
                + "\n\n".join(sections)
            )
            token_count = count_tokens(text)
            if token_count < 768:
                raise ValueError(
                    f"generated {split} prompt {prompt_index} has only {token_count} tokens"
                )
            text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            generated_prompts.append(
                {
                    "id": f"{split}-{text_digest[:16]}",
                    "split": split,
                    "token_count": token_count,
                    "text": text,
                }
            )

    return {
        "format": PROMPT_FORMAT,
        "format_version": PROMPT_FORMAT_VERSION,
        "prompts": generated_prompts,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    source = arguments.source.resolve()
    tokenizer_path = arguments.tokenizer_json.resolve()
    output = arguments.output.resolve()
    if not source.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("source manifest and tokenizer JSON must be regular files")
    if output == source:
        raise ValueError("long prompt output must not overwrite its short source corpus")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    document = build_long_prompt_document(
        _load_document(source),
        count_tokens=lambda text: len(tokenizer.encode(text).ids),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
