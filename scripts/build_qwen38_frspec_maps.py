#!/usr/bin/env python3
"""Build deterministic Qwen3.8 FR-Spec maps for English/code agent workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

DEFAULT_SIZES = (32 * 1024, 64 * 1024, 96 * 1024)
TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".rs",
        ".sh",
        ".svelte",
        ".toml",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
    }
)
SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
    }
)

# Tokenize domain syntax even when a caller supplies a very small corpus. The
# merge-rank half of each map supplies general English; these seeds ensure the
# domain half sees Python, async/concurrency, patches, shell, and Qwen/OpenCode
# tool-call delimiters.
DOMAIN_SEED = r"""
Implement a production-quality asynchronous bounded worker pool in Python.
Use asyncio.Queue, TaskGroup, dataclasses, type hints, graceful cancellation,
backpressure, structured error collection, docstrings, and an executable test.

async def worker(queue: asyncio.Queue[WorkItem | None]) -> None:
    while item := await queue.get():
        try:
            result = await process(item)
        except Exception as error:
            errors.append(WorkerError(item=item, cause=error))
        finally:
            queue.task_done()

<|im_start|>assistant
<think>
Inspect the repository, run focused tests, and make the smallest correct patch.
</think>
I will update the implementation and verify formatting, lint, and tests.
<tool_call>
<function=exec_command>
<parameter=cmd>
rg -n "TODO|FIXME" src tests && pytest -q
</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
3 passed in 0.42s
</tool_response>
Please explain the failure, preserve public APIs, and return a concise report.
<|im_end|>

*** Begin Patch
*** Update File: src/example.py
@@
-raise NotImplementedError
+return await implementation(request)
*** End Patch

from typing import Final, Generic, TypeVar
import contextlib
import logging
import pathlib
import subprocess
import unittest
"""


@dataclass(frozen=True)
class MapReceipt:
    path: str
    sha256: str
    size_bytes: int
    token_count: int
    corpus_token_coverage: float
    domain_seed_token_coverage: float
    forced_added_token_count: int
    forced_domain_seed_token_count: int
    merge_rank_core_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Qwen model snapshot containing tokenizer.json and config.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--size",
        dest="sizes",
        type=int,
        action="append",
        help="Map size; repeat for a sweep (default: 32768, 65536, 98304).",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively scan a source/documentation tree; repeatable.",
    )
    parser.add_argument(
        "--corpus-file",
        type=Path,
        action="append",
        default=[],
        help="Add an explicit transcript or benchmark output; repeatable.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=128 * 1024,
        help="Deterministic prefix read from each corpus file.",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Maximum aggregate corpus bytes, in sorted path order.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Require every emitted map size to divide this TP degree.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_corpus_files(
    roots: Sequence[Path],
    explicit_files: Sequence[Path],
    *,
    excluded_paths: Sequence[Path] = (),
    excluded_roots: Sequence[Path] = (),
) -> list[Path]:
    resolved_excluded_paths = {path.resolve() for path in excluded_paths}
    resolved_excluded_roots = tuple(path.resolve() for path in excluded_roots)

    def is_excluded(path: Path) -> bool:
        resolved_path = path.resolve()
        return resolved_path in resolved_excluded_paths or any(
            resolved_path.is_relative_to(root) for root in resolved_excluded_roots
        )

    paths = {
        path.resolve()
        for path in explicit_files
        if path.is_file() and not is_excluded(path)
    }
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if is_excluded(path):
                continue
            relative_parts = path.relative_to(root).parts[:-1]
            if any(part in SKIPPED_DIRECTORY_NAMES for part in relative_parts):
                continue
            paths.add(path.resolve())
    return sorted(paths, key=lambda path: str(path))


def read_corpus(
    paths: Iterable[Path], *, max_file_bytes: int, max_total_bytes: int
) -> tuple[list[tuple[str, str]], int]:
    documents: list[tuple[str, str]] = []
    total_bytes = 0
    for path in paths:
        remaining = max_total_bytes - total_bytes
        if remaining <= 0:
            break
        read_size = min(max_file_bytes, remaining)
        raw = path.read_bytes()[:read_size]
        if b"\x00" in raw:
            continue
        documents.append((str(path), raw.decode("utf-8", errors="ignore")))
        total_bytes += len(raw)
    return documents, total_bytes


def token_counts(tokenizer: Tokenizer, documents: Iterable[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for document in documents:
        counts.update(tokenizer.encode(document, add_special_tokens=False).ids)
    return counts


def select_token_ids(
    *,
    size: int,
    base_vocab_size: int,
    tokenizer_vocab_size: int,
    forced_token_ids: Sequence[int],
    counts: Counter[int],
) -> tuple[list[int], int]:
    if size > tokenizer_vocab_size:
        raise ValueError(
            f"Map size {size} exceeds tokenizer vocabulary {tokenizer_vocab_size}."
        )

    # Keep half the budget as a contiguous BPE merge-rank core. Qwen's BPE ids
    # encode merge order, making this the robust general-English prior. Fill the
    # other half with observed English/Python/OpenCode tokens by descending
    # frequency, then return to merge rank for any remaining slots.
    merge_rank_core_count = min(size // 2, base_vocab_size)
    selected = set(range(merge_rank_core_count))
    selected.update(forced_token_ids)
    if len(selected) > size:
        raise ValueError(
            f"Map size {size} cannot hold {len(selected)} forced/core token ids."
        )

    ranked_observed_ids = sorted(
        (
            token_id
            for token_id in counts
            if 0 <= token_id < tokenizer_vocab_size and token_id not in selected
        ),
        key=lambda token_id: (-counts[token_id], token_id),
    )
    for token_id in ranked_observed_ids:
        if len(selected) == size:
            break
        selected.add(token_id)

    for token_id in range(tokenizer_vocab_size):
        if len(selected) == size:
            break
        selected.add(token_id)

    if len(selected) != size:
        raise AssertionError(f"Selected {len(selected)} ids for requested size {size}.")
    return sorted(selected), merge_rank_core_count


def coverage(counts: Counter[int], selected_token_ids: set[int]) -> float:
    total = counts.total()
    if total == 0:
        return 1.0
    covered = sum(
        count for token_id, count in counts.items() if token_id in selected_token_ids
    )
    return covered / total


def main() -> None:
    args = parse_args()
    sizes = tuple(args.sizes or DEFAULT_SIZES)
    if args.tp_size < 1:
        raise ValueError("--tp-size must be positive.")
    for size in sizes:
        if size <= 0 or size % args.tp_size != 0:
            raise ValueError(
                f"Map size {size} must be positive and divisible by TP={args.tp_size}."
            )

    tokenizer_path = args.model_path / "tokenizer.json"
    config_path = args.model_path / "config.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    base_vocab_size = tokenizer.get_vocab_size(with_added_tokens=False)
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    added_token_ids = sorted(tokenizer.get_added_tokens_decoder())

    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    model_vocab_size = int(text_config["vocab_size"])
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab {tokenizer_vocab_size} exceeds model vocab {model_vocab_size}."
        )

    # The generator and generated receipts must not train their own successor.
    # Without these exclusions, formatting this file or regenerating into a
    # scanned tree can perturb a boundary token in the next map.
    corpus_paths = discover_corpus_files(
        args.corpus_root,
        args.corpus_file,
        excluded_paths=[Path(__file__)],
        excluded_roots=[args.output_dir],
    )
    documents, corpus_bytes = read_corpus(
        corpus_paths,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    corpus_counts = token_counts(tokenizer, (text for _, text in documents))
    domain_seed_counts = token_counts(tokenizer, [DOMAIN_SEED])
    ranking_counts = corpus_counts + domain_seed_counts
    # Domain syntax is a correctness-of-acceleration requirement: missing one
    # of these tokens cannot change verified output, but it deterministically
    # terminates the draft chain at that position. Force the small seed set in
    # addition to every tokenizer-added control token.
    forced_token_ids = sorted(set(added_token_ids) | set(domain_seed_counts))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[MapReceipt] = []
    for size in sizes:
        token_ids, merge_rank_core_count = select_token_ids(
            size=size,
            base_vocab_size=base_vocab_size,
            tokenizer_vocab_size=tokenizer_vocab_size,
            forced_token_ids=forced_token_ids,
            counts=ranking_counts,
        )
        token_id_tensor = torch.tensor(token_ids, dtype=torch.int64)
        output_path = (
            args.output_dir / f"qwen38-frspec-english-python-opencode-{size}.pt"
        )
        torch.save(token_id_tensor, output_path)
        selected_token_ids = set(token_ids)
        receipts.append(
            MapReceipt(
                path=str(output_path),
                sha256=sha256_file(output_path),
                size_bytes=output_path.stat().st_size,
                token_count=size,
                corpus_token_coverage=coverage(corpus_counts, selected_token_ids),
                domain_seed_token_coverage=coverage(
                    domain_seed_counts, selected_token_ids
                ),
                forced_added_token_count=len(added_token_ids),
                forced_domain_seed_token_count=len(domain_seed_counts),
                merge_rank_core_count=merge_rank_core_count,
            )
        )

    receipt = {
        "algorithm": (
            "sorted unique union of the lowest size/2 base-BPE merge ranks, all "
            "tokenizer-added control tokens, all curated Python/OpenCode seed "
            "tokens, corpus tokens by (-frequency,id), then remaining ids by "
            "merge rank"
        ),
        "base_vocab_size": base_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": model_vocab_size,
        "tp_size": args.tp_size,
        "corpus_file_count": len(documents),
        "corpus_bytes": corpus_bytes,
        "corpus_paths": [path for path, _ in documents],
        "maps": [asdict(map_receipt) for map_receipt in receipts],
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(receipt_path)
    for map_receipt in receipts:
        print(
            f"{map_receipt.token_count}: {map_receipt.path} "
            f"sha256={map_receipt.sha256} "
            f"corpus_coverage={map_receipt.corpus_token_coverage:.6%} "
            f"seed_coverage={map_receipt.domain_seed_token_coverage:.6%}"
        )


if __name__ == "__main__":
    main()
