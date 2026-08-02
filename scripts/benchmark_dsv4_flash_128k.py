#!/usr/bin/env python3
"""Benchmark DSV4 at an exact token shape without printing model output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmark_dsv4_flash_fwuff_baseline")
    parser.add_argument("--url", default="http://127.0.0.1:30010/generate")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731"),
    )
    # Defaults reproduce the actual fwuff performance record.  Larger context
    # capacity is a separate launch choice, not part of this benchmark shape.
    parser.add_argument("--input-tokens", type=int, default=2_694)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="report every N streamed completion tokens to stderr (0 disables)",
    )
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


def load_message_encoder(model_path: Path) -> Callable[..., str]:
    encoder_path = model_path / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("dsv4_release_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DSV4 encoder from {encoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = getattr(module, "encode_messages", None)
    if not callable(encoder):
        raise TypeError("DSV4 release encoder has no callable encode_messages")
    return encoder


def build_exact_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    encode_messages: Callable[..., str],
    target_tokens: int,
) -> list[int]:
    instruction = (
        "\nWrite at least 250 words explaining how careful measurement improves "
        "computer-system performance."
    )

    def encode_with_fill(fill_tokens: int) -> list[int]:
        prompt = encode_messages(
            [
                {
                    "role": "user",
                    "content": (" a" * fill_tokens) + instruction,
                }
            ],
            thinking_mode="chat",
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    empty_ids = encode_with_fill(0)
    if len(empty_ids) >= target_tokens:
        raise ValueError(
            f"target input length {target_tokens} is not larger than framing length"
        )

    lower = 0
    upper = target_tokens
    while lower <= upper:
        middle = (lower + upper) // 2
        prompt_ids = encode_with_fill(middle)
        prompt_length = len(prompt_ids)
        if prompt_length == target_tokens:
            return prompt_ids
        if prompt_length < target_tokens:
            lower = middle + 1
        else:
            upper = middle - 1

    for fill_tokens in range(max(0, upper - 8), lower + 9):
        prompt_ids = encode_with_fill(fill_tokens)
        if len(prompt_ids) == target_tokens:
            return prompt_ids
    raise ValueError(f"could not construct an exact {target_tokens}-token prompt")


def run_benchmark(
    url: str,
    input_ids: list[int],
    output_tokens: int,
    timeout: float,
    progress_every: int = 0,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    arrivals: list[tuple[int, float]] = []
    final_text = ""
    final_metadata: dict[str, Any] = {}
    next_progress_token = progress_every
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            event_payload = line.removeprefix(b"data: ")
            if event_payload == b"[DONE]":
                continue
            event = json.loads(event_payload)
            metadata = event.get("meta_info") or {}
            completion_tokens = metadata.get("completion_tokens")
            if isinstance(completion_tokens, int):
                elapsed = time.perf_counter() - started
                if not arrivals or completion_tokens > arrivals[-1][0]:
                    arrivals.append((completion_tokens, elapsed))
                    if progress_every > 0 and completion_tokens >= next_progress_token:
                        print(
                            f"completion_tokens={completion_tokens} elapsed={elapsed:.3f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_progress_token = (
                            completion_tokens // progress_every + 1
                        ) * progress_every
            generated_text = event.get("text")
            if isinstance(generated_text, str):
                final_text = generated_text
            if isinstance(metadata, dict):
                final_metadata = metadata
    elapsed_seconds = time.perf_counter() - started

    if status != 200 or not arrivals:
        raise RuntimeError("DSV4 benchmark request produced no token-bearing events")
    completion_tokens = arrivals[-1][0]
    if completion_tokens != output_tokens:
        raise RuntimeError(
            f"expected {output_tokens} completion tokens, received {completion_tokens}"
        )

    time_to_first_token = arrivals[0][1]
    decode_seconds = elapsed_seconds - time_to_first_token
    decode_token_count = max(0, completion_tokens - 1)
    return {
        "http_status": status,
        "input_tokens": len(input_ids),
        "completion_tokens": completion_tokens,
        "server_prompt_tokens": final_metadata.get("prompt_tokens"),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "time_to_first_token_seconds": round(time_to_first_token, 6),
        "prefill_tokens_per_second": round(len(input_ids) / time_to_first_token, 6),
        "decode_seconds": round(decode_seconds, 6),
        "decode_tokens_per_second": round(
            decode_token_count / decode_seconds if decode_seconds > 0 else 0.0,
            6,
        ),
        "total_tokens_per_second": round(
            (len(input_ids) + completion_tokens) / elapsed_seconds, 6
        ),
        "stream_events": len(arrivals),
        "output_bytes": len(final_text.encode("utf-8")),
        "output_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    encode_messages = load_message_encoder(args.model_path)
    input_ids = build_exact_prompt_ids(
        tokenizer,
        encode_messages,
        args.input_tokens,
    )
    result = run_benchmark(
        args.url,
        input_ids,
        args.output_tokens,
        args.timeout,
        args.progress_every,
    )
    result_json = json.dumps(result, sort_keys=True)
    if args.output_file is not None:
        args.output_file.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
