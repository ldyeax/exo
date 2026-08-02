#!/usr/bin/env python3
"""Run a deterministic DSV4 coherency check without printing generated text."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30010/v1/chat/completions")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731"),
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def generate_chat_completion(
    url: str,
    model_path: Path,
    expected_marker: str,
    timeout: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": str(model_path),
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Reply with exactly the word {expected_marker} and nothing else."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.load(response)
        status = response.status
    elapsed_seconds = time.perf_counter() - started

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("DSV4 response did not contain a non-empty choices list")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TypeError("DSV4 response choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("DSV4 response choice did not contain a message object")
    generated_text = message.get("content")
    if not isinstance(generated_text, str):
        raise TypeError("DSV4 response message did not contain string content")
    usage = response_payload.get("usage", {})
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    return {
        "http_status": status,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "completion_tokens": completion_tokens,
        "finish_reason": choice.get("finish_reason"),
        "output_bytes": len(generated_text.encode("utf-8")),
        "output_sha256": hashlib.sha256(generated_text.encode("utf-8")).hexdigest(),
        "expected_marker_count": generated_text.count(expected_marker),
    }


def main() -> int:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    expected_marker = "PINEAPPLE"
    try:
        results = [
            generate_chat_completion(
                args.url,
                args.model_path,
                expected_marker,
                args.timeout,
                args.max_new_tokens,
            )
            for _ in range(args.repetitions)
        ]
    except (
        ImportError,
        OSError,
        TypeError,
        ValueError,
        urllib.error.HTTPError,
    ) as error:
        print(json.dumps({"coherent": False, "error_type": type(error).__name__}))
        return 1

    output_hashes = {result["output_sha256"] for result in results}
    deterministic = len(output_hashes) == 1
    coherent = all(
        result["http_status"] == 200 and result["expected_marker_count"] >= 1
        for result in results
    )
    print(
        json.dumps(
            {"coherent": coherent, "deterministic": deterministic, "runs": results},
            sort_keys=True,
        )
    )
    return 0 if coherent else 1


if __name__ == "__main__":
    raise SystemExit(main())
