#!/usr/bin/env python3
"""Benchmark DSV4 at an exact input shape with a semantic acceptance gate.

The script deliberately keeps model text out of its receipt.  A throughput
receipt is emitted only after the generated text passes semantic, degeneracy,
stream-integrity, and termination checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast


class PromptTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class ServingTokenizer(PromptTokenizer, Protocol):
    @property
    def eos_token_id(self) -> int | list[int] | None: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class TokenizerFactory(Protocol):
    def from_pretrained(self, model_path: str) -> ServingTokenizer: ...


class MessageEncoder(Protocol):
    def __call__(
        self, messages: list[dict[str, str]], *, thinking_mode: str
    ) -> str: ...


class HttpResponse(Protocol):
    status: int

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def __iter__(self) -> Iterator[bytes]: ...


RECEIPT_VERSION = 2
MINIMUM_OUTPUT_WORDS = 120
TAIL_WORD_COUNT = 48
REQUIRED_ENDING = "Measurement before claims."
REQUIRED_SEMANTIC_MARKERS = {
    "profile": "aurora-73",
    "baseline": "46",
    "improvement": "27",
    "reproducibility": "reproducibility",
    "thermal_headroom": "thermal headroom",
    "semantic_validation": "semantic validation",
}
WORD_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*")
SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\s\"')\]]|$)")
REPEATED_NGRAM_SIZE = 4
COPY_NGRAM_SIZE = 8


class BenchmarkValidationError(RuntimeError):
    """A safe benchmark receipt cannot be produced."""

    def __init__(self, issue_codes: Iterable[str]) -> None:
        self.issue_codes = tuple(sorted(set(issue_codes)))
        super().__init__("benchmark output rejected: " + ", ".join(self.issue_codes))


@dataclass(frozen=True)
class SemanticAssessment:
    issue_codes: tuple[str, ...]
    word_count: int
    unique_word_ratio: float
    dominant_word_ratio: float
    repeated_four_gram_ratio: float
    tail_unique_word_ratio: float
    printable_character_ratio: float
    alphabetic_character_ratio: float
    sentence_count: int
    prompt_copy_eight_gram_ratio: float | None
    semantic_markers_present: int

    @property
    def passed(self) -> bool:
        return not self.issue_codes

    def safe_receipt(self) -> dict[str, Any]:
        """Return metrics only; generated model text is intentionally omitted."""
        return {
            "passed": self.passed,
            "issue_codes": list(self.issue_codes),
            "word_count": self.word_count,
            "unique_word_ratio": round(self.unique_word_ratio, 6),
            "dominant_word_ratio": round(self.dominant_word_ratio, 6),
            "repeated_four_gram_ratio": round(self.repeated_four_gram_ratio, 6),
            "tail_unique_word_ratio": round(self.tail_unique_word_ratio, 6),
            "printable_character_ratio": round(self.printable_character_ratio, 6),
            "alphabetic_character_ratio": round(self.alphabetic_character_ratio, 6),
            "sentence_count": self.sentence_count,
            "prompt_copy_eight_gram_ratio": (
                round(self.prompt_copy_eight_gram_ratio, 6)
                if self.prompt_copy_eight_gram_ratio is not None
                else None
            ),
            "semantic_markers_present": self.semantic_markers_present,
            "semantic_markers_required": len(REQUIRED_SEMANTIC_MARKERS),
            "required_ending_present": "missing_required_ending"
            not in self.issue_codes,
        }


@dataclass(frozen=True)
class BenchmarkArguments:
    url: str
    model_path: Path
    input_tokens: int
    output_tokens: int
    timeout: float
    progress_every: int
    ignore_eos: bool
    output_file: Path | None


def parse_args() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(prog="benchmark_dsv4_flash_fwuff_baseline")
    parser.add_argument("--url", default="http://127.0.0.1:30010/generate")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731"),
    )
    # Defaults reproduce the actual fwuff performance-record token shape.
    # Context capacity is a launch choice, not part of this benchmark shape.
    parser.add_argument("--input-tokens", type=int, default=2_694)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="report every N streamed completion tokens to stderr (0 disables)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "force the exact requested output-token count; this is opt-in and "
            "is accepted only when output IDs prove no early EOS occurred"
        ),
    )
    parser.add_argument("--output-file", type=Path)
    arguments = parser.parse_args()
    return BenchmarkArguments(
        url=cast(str, arguments.url),
        model_path=cast(Path, arguments.model_path),
        input_tokens=cast(int, arguments.input_tokens),
        output_tokens=cast(int, arguments.output_tokens),
        timeout=cast(float, arguments.timeout),
        progress_every=cast(int, arguments.progress_every),
        ignore_eos=cast(bool, arguments.ignore_eos),
        output_file=cast(Path | None, arguments.output_file),
    )


def load_message_encoder(model_path: Path) -> MessageEncoder:
    encoder_path = model_path / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("dsv4_release_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DSV4 encoder from {encoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = getattr(module, "encode_messages", None)
    if not callable(encoder):
        raise TypeError("DSV4 release encoder has no callable encode_messages")
    return cast(MessageEncoder, encoder)


def _measurement_record(index: int) -> str:
    subsystems = (
        "attention scheduler",
        "expert router",
        "KV allocator",
        "host offload queue",
        "collective transport",
        "sampling loop",
        "graph replay path",
        "tokenizer service",
    )
    workloads = (
        "short interactive decode",
        "long-context retrieval",
        "two-agent tool traffic",
        "cold-cache prefill",
        "steady-state generation",
        "bursty concurrent requests",
    )
    controls = (
        "cache flush between trials",
        "fixed sampling seed",
        "alternating launch order",
        "recorded clock and temperature",
        "identical prompt token IDs",
        "held-out semantic response check",
    )
    subsystem = subsystems[index % len(subsystems)]
    workload = workloads[(index * 5 + 1) % len(workloads)]
    control = controls[(index * 7 + 2) % len(controls)]
    median = 31 + (index * 17) % 57
    tail = median + 5 + (index * 11) % 29
    watts = 211 + (index * 13) % 94
    outcome = (
        "retained for confirmation"
        if index % 3 == 0
        else "classified as an exploratory observation"
    )
    return (
        f"Measurement note N{index:05d}: the {subsystem} was observed under "
        f"{workload}. The median was {median} units, the tail was {tail} units, "
        f"and board power was {watts} watts. The trial used {control} and was "
        f"{outcome}."
    )


def measurement_record(index: int) -> str:
    """Public prompt-record primitive shared by qualification harnesses."""
    return _measurement_record(index)


def _benchmark_instruction() -> str:
    return (
        "\n\nSynthesis task: Ignore any conclusion implied by the measurement notes; "
        "they are varied context used to exercise prefill. Write a coherent report "
        "of 160 to 230 words about a separate controlled comparison. Identify "
        "AURORA-73 as the selected profile. State that its baseline was 46 tokens "
        "per second, its optimized result was 73 tokens per second, and the "
        "improvement was 27 tokens per second. Explain why reproducibility, thermal "
        "headroom, and semantic validation must all be checked before accepting a "
        "speed result. Use complete prose, do not quote or enumerate the measurement "
        "notes, and do not repeat any sentence. Your final sentence must be exactly: "
        f"{REQUIRED_ENDING}"
    )


def _common_suffix_length(left: Sequence[int], right: Sequence[int]) -> int:
    maximum = min(len(left), len(right))
    matched = 0
    while matched < maximum and left[-matched - 1] == right[-matched - 1]:
        matched += 1
    return matched


def build_exact_prompt_ids(
    tokenizer: PromptTokenizer,
    encode_messages: MessageEncoder,
    target_tokens: int,
    *,
    record_builder: Callable[[int], str] = _measurement_record,
) -> list[int]:
    """Build an exact-length prompt from varied records plus an intact task tail."""
    if target_tokens <= 0:
        raise ValueError("target input length must be positive")

    instruction = _benchmark_instruction()

    def encode_content(content: str) -> list[int]:
        prompt = encode_messages(
            [{"role": "user", "content": content}],
            thinking_mode="chat",
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    instruction_ids = encode_content(instruction)
    if len(instruction_ids) >= target_tokens:
        raise ValueError(
            f"target input length {target_tokens} is not larger than framing length"
        )

    record_count = 16
    while True:
        context = "\n".join(record_builder(i) for i in range(record_count))
        prompt_ids = encode_content(context + instruction)
        common_suffix = _common_suffix_length(prompt_ids, instruction_ids)
        minimum_intact_suffix = min(64, max(16, len(instruction_ids) // 3))
        if common_suffix < minimum_intact_suffix:
            raise ValueError(
                "tokenizer did not preserve the semantic instruction suffix"
            )
        if len(prompt_ids) > target_tokens:
            prefix_length = target_tokens - common_suffix
            if prefix_length <= 0:
                raise ValueError(
                    "target input length cannot retain semantic prompt framing"
                )
            exact_ids = prompt_ids[:prefix_length] + prompt_ids[-common_suffix:]
            if len(exact_ids) != target_tokens:
                raise AssertionError("exact prompt construction changed token shape")
            return exact_ids
        record_count *= 2
        if record_count > 1_048_576:
            raise ValueError(
                f"could not construct an exact {target_tokens}-token prompt"
            )


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(text)]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _repeated_ngram_ratio(words: Sequence[str], size: int) -> float:
    if len(words) < size:
        return 0.0
    ngrams = [
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    ]
    counts = Counter(ngrams)
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return _ratio(repeated_occurrences, len(ngrams))


def _prompt_copy_ratio(
    prompt_ids: Sequence[int], output_ids: Sequence[int], size: int
) -> float | None:
    if len(prompt_ids) < size or len(output_ids) < size:
        return None
    output_ngrams = [
        tuple(output_ids[index : index + size])
        for index in range(len(output_ids) - size + 1)
    ]
    output_ngram_set = set(output_ngrams)
    present_in_prompt: set[tuple[int, ...]] = set()
    for index in range(len(prompt_ids) - size + 1):
        ngram = tuple(prompt_ids[index : index + size])
        if ngram in output_ngram_set:
            present_in_prompt.add(ngram)
    copied = sum(ngram in present_in_prompt for ngram in output_ngrams)
    return _ratio(copied, len(output_ngrams))


def assess_semantic_output(
    output_text: str,
    *,
    prompt_ids: Sequence[int] = (),
    output_ids: Sequence[int] = (),
) -> SemanticAssessment:
    """Conservatively reject incoherent, copied, or degenerately repeated output."""
    words = _words(output_text)
    word_counts = Counter(words)
    word_count = len(words)
    unique_word_ratio = _ratio(len(word_counts), word_count)
    dominant_word_ratio = _ratio(max(word_counts.values(), default=0), word_count)
    repeated_ratio = _repeated_ngram_ratio(words, REPEATED_NGRAM_SIZE)
    tail_words = words[-TAIL_WORD_COUNT:]
    tail_unique_ratio = _ratio(len(set(tail_words)), len(tail_words))
    character_count = len(output_text)
    printable_count = sum(
        character.isprintable() or character in "\n\r\t" for character in output_text
    )
    visible_count = sum(not character.isspace() for character in output_text)
    alphabetic_count = sum(character.isalpha() for character in output_text)
    printable_ratio = _ratio(printable_count, character_count)
    alphabetic_ratio = _ratio(alphabetic_count, visible_count)
    sentence_count = len(SENTENCE_END_PATTERN.findall(output_text))
    folded_output = output_text.casefold()
    markers_present = sum(
        marker in folded_output for marker in REQUIRED_SEMANTIC_MARKERS.values()
    )
    copy_ratio = (
        _prompt_copy_ratio(prompt_ids, output_ids, COPY_NGRAM_SIZE)
        if prompt_ids and output_ids
        else None
    )

    issue_codes: set[str] = set()
    if word_count < MINIMUM_OUTPUT_WORDS:
        issue_codes.add("output_too_short")
    for name, marker in REQUIRED_SEMANTIC_MARKERS.items():
        if marker not in folded_output:
            issue_codes.add(f"missing_marker_{name}")
    if not output_text.rstrip().casefold().endswith(REQUIRED_ENDING.casefold()):
        issue_codes.add("missing_required_ending")
    if unique_word_ratio < 0.35:
        issue_codes.add("low_lexical_diversity")
    if dominant_word_ratio > 0.10:
        issue_codes.add("dominant_repeated_word")
    if repeated_ratio > 0.08:
        issue_codes.add("repeated_four_gram")
    if len(tail_words) < TAIL_WORD_COUNT or tail_unique_ratio < 0.35:
        issue_codes.add("degenerate_output_tail")
    if printable_ratio < 0.995 or any(
        unicodedata.category(character) in {"Cc", "Cs"} and character not in "\n\r\t"
        for character in output_text
    ):
        issue_codes.add("invalid_character_mix")
    if alphabetic_ratio < 0.55:
        issue_codes.add("low_alphabetic_content")
    if sentence_count < 5:
        issue_codes.add("too_few_sentences")
    if copy_ratio is not None and copy_ratio > 0.35:
        issue_codes.add("prompt_copying")

    return SemanticAssessment(
        issue_codes=tuple(sorted(issue_codes)),
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        dominant_word_ratio=dominant_word_ratio,
        repeated_four_gram_ratio=repeated_ratio,
        tail_unique_word_ratio=tail_unique_ratio,
        printable_character_ratio=printable_ratio,
        alphabetic_character_ratio=alphabetic_ratio,
        sentence_count=sentence_count,
        prompt_copy_eight_gram_ratio=copy_ratio,
        semantic_markers_present=markers_present,
    )


def _update_output_ids(
    accumulated: list[int], event_output_ids: object, completion_tokens: int
) -> None:
    if not isinstance(event_output_ids, list):
        raise BenchmarkValidationError(("malformed_output_ids",))
    raw_ids = cast(list[object], event_output_ids)
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in raw_ids
    ):
        raise BenchmarkValidationError(("malformed_output_ids",))
    typed_ids = cast(list[int], raw_ids)
    if len(typed_ids) == completion_tokens:
        if (
            len(typed_ids) < len(accumulated)
            or typed_ids[: len(accumulated)] != accumulated
        ):
            raise BenchmarkValidationError(("output_id_prefix_changed",))
        accumulated[:] = typed_ids
        return
    if len(accumulated) + len(typed_ids) == completion_tokens:
        accumulated.extend(typed_ids)
        return
    raise BenchmarkValidationError(("inconsistent_output_ids",))


def _merge_stream_text(accumulated: str, event_text: str) -> str:
    if not accumulated or event_text.startswith(accumulated):
        return event_text
    if accumulated.endswith(event_text):
        return accumulated
    return accumulated + event_text


def _finish_reason_type(raw_finish_reason: object) -> str | None:
    if isinstance(raw_finish_reason, str):
        return raw_finish_reason
    if isinstance(raw_finish_reason, dict):
        typed_finish_reason = cast(dict[object, object], raw_finish_reason)
        raw_type = typed_finish_reason.get("type")
        if isinstance(raw_type, str):
            return raw_type
    return None


def _hash_token_ids(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError("token IDs must be non-negative")
        digest.update(token_id.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Public stable token-ID digest used in redacted benchmark receipts."""
    return _hash_token_ids(token_ids)


def run_benchmark(
    url: str,
    input_ids: list[int],
    output_tokens: int,
    timeout: float,
    progress_every: int = 0,
    *,
    ignore_eos: bool = False,
    decode_output_ids: Callable[[list[int]], str] | None = None,
    terminal_token_ids: Sequence[int] = (),
    request_id: str | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Run one request and return throughput only after strict validation."""
    if not input_ids or output_tokens <= 0 or timeout <= 0:
        raise ValueError("input IDs, output token count, and timeout must be positive")
    if diagnostic_only and not ignore_eos:
        raise ValueError("diagnostic-only requests must use a fixed forced length")
    payload: dict[str, object] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
            "sampling_seed": 0,
        },
        "stream": True,
        "return_logprob": False,
        "log_metrics": True,
    }
    if request_id is not None:
        if not request_id:
            raise ValueError("request_id must be non-empty when provided")
        payload["rid"] = request_id
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    arrivals: list[tuple[int, float]] = []
    output_text = ""
    output_ids: list[int] = []
    saw_output_ids = False
    server_prompt_tokens: int | None = None
    server_cached_tokens: int | None = None
    finish_reason: str | None = None
    event_count = 0
    saw_done = False
    next_progress_token = progress_every
    status = 0
    with cast(
        HttpResponse, urllib.request.urlopen(request, timeout=timeout)
    ) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            event_payload = line.removeprefix(b"data:").lstrip()
            if event_payload == b"[DONE]":
                saw_done = True
                continue
            try:
                decoded_event = cast(object, json.loads(event_payload))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BenchmarkValidationError(("invalid_stream_json",)) from error
            if not isinstance(decoded_event, dict):
                raise BenchmarkValidationError(("stream_event_not_object",))
            event = cast(dict[str, object], decoded_event)
            event_count += 1
            raw_metadata = event.get("meta_info")
            if not isinstance(raw_metadata, dict):
                continue
            metadata = cast(dict[str, object], raw_metadata)
            completion_tokens = metadata.get("completion_tokens")
            if not isinstance(completion_tokens, int) or isinstance(
                completion_tokens, bool
            ):
                continue
            if arrivals and completion_tokens < arrivals[-1][0]:
                raise BenchmarkValidationError(("completion_count_moved_backward",))

            prompt_tokens = metadata.get("prompt_tokens")
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
                if server_prompt_tokens is None:
                    server_prompt_tokens = prompt_tokens
                elif server_prompt_tokens != prompt_tokens:
                    raise BenchmarkValidationError(("prompt_count_changed",))

            cached_tokens = metadata.get("cached_tokens")
            if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
                if cached_tokens < 0:
                    raise BenchmarkValidationError(("negative_cached_tokens",))
                if server_cached_tokens is None:
                    server_cached_tokens = cached_tokens
                elif server_cached_tokens != cached_tokens:
                    raise BenchmarkValidationError(("cached_token_count_changed",))

            event_output_ids = event.get("output_ids")
            if event_output_ids is not None:
                _update_output_ids(output_ids, event_output_ids, completion_tokens)
                saw_output_ids = True

            event_text = event.get("text")
            if isinstance(event_text, str):
                output_text = _merge_stream_text(output_text, event_text)

            raw_finish_reason = metadata.get("finish_reason")
            parsed_finish_reason = _finish_reason_type(raw_finish_reason)
            if parsed_finish_reason is not None:
                finish_reason = parsed_finish_reason

            if not arrivals or completion_tokens > arrivals[-1][0]:
                elapsed = time.perf_counter() - started
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
    elapsed_seconds = time.perf_counter() - started

    protocol_issues: set[str] = set()
    if status != 200:
        protocol_issues.add("non_200_response")
    if not arrivals:
        protocol_issues.add("no_token_bearing_events")
        raise BenchmarkValidationError(protocol_issues)
    completion_tokens = arrivals[-1][0]
    if completion_tokens <= 0 or completion_tokens > output_tokens:
        protocol_issues.add("invalid_completion_count")
    if ignore_eos and completion_tokens != output_tokens:
        protocol_issues.add("forced_generation_not_exact_length")
    if server_prompt_tokens is not None and server_prompt_tokens != len(input_ids):
        protocol_issues.add("server_prompt_count_mismatch")
    if not saw_done:
        protocol_issues.add("missing_stream_done")
    if finish_reason is None:
        protocol_issues.add("missing_finish_reason")
    elif ignore_eos and finish_reason != "length":
        protocol_issues.add("unexpected_forced_finish_reason")
    elif not ignore_eos and finish_reason != "stop":
        protocol_issues.add("non_natural_finish_reason")

    if saw_output_ids and len(output_ids) != completion_tokens:
        protocol_issues.add("incomplete_output_id_stream")
    if decode_output_ids is not None and len(output_ids) == completion_tokens:
        output_text = decode_output_ids(output_ids)

    terminal_ids = frozenset(terminal_token_ids)
    if (
        ignore_eos
        and not diagnostic_only
        and (not terminal_ids or len(output_ids) != completion_tokens)
    ):
        protocol_issues.add("forced_generation_eos_scan_unavailable")
    terminal_positions: list[int] = []
    if terminal_ids and output_ids:
        terminal_positions = [
            index
            for index, token_id in enumerate(output_ids)
            if token_id in terminal_ids
        ]
        if not diagnostic_only and any(
            position < len(output_ids) - 1 for position in terminal_positions
        ):
            protocol_issues.add("tokens_after_terminal_token")

    semantic = assess_semantic_output(
        output_text,
        prompt_ids=input_ids,
        output_ids=output_ids if len(output_ids) == completion_tokens else (),
    )
    all_issues = (
        protocol_issues
        if diagnostic_only
        else set(semantic.issue_codes) | protocol_issues
    )
    if all_issues:
        raise BenchmarkValidationError(all_issues)

    first_completion_count, time_to_first_token = arrivals[0]
    last_token_at = arrivals[-1][1]
    decode_seconds = last_token_at - time_to_first_token
    decode_token_count = completion_tokens - first_completion_count
    if decode_token_count <= 0 or decode_seconds <= 0:
        raise BenchmarkValidationError(("insufficient_decode_timing_events",))

    semantic_receipt = semantic.safe_receipt()
    semantic_receipt["enforced"] = not diagnostic_only
    output_token_ids_available = len(output_ids) == completion_tokens
    return {
        "receipt_version": RECEIPT_VERSION,
        "accepted": True,
        "performance_claim_eligible": not diagnostic_only,
        "diagnostic_only": diagnostic_only,
        "receipt_safety": (
            "diagnostic_forced_length"
            if diagnostic_only
            else ("validated_forced_length" if ignore_eos else "validated_natural_stop")
        ),
        "http_status": status,
        "input_tokens": len(input_ids),
        "requested_max_completion_tokens": output_tokens,
        "completion_tokens": completion_tokens,
        "exact_requested_token_shape": completion_tokens == output_tokens,
        "server_prompt_tokens": server_prompt_tokens,
        "server_cached_tokens": server_cached_tokens,
        "request_id": request_id,
        "ignore_eos": ignore_eos,
        "finish_reason": finish_reason,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "time_to_first_token_seconds": round(time_to_first_token, 6),
        "prefill_tokens_per_second": round(len(input_ids) / time_to_first_token, 6),
        "decode_seconds": round(decode_seconds, 6),
        "decode_tokens_per_second": round(decode_token_count / decode_seconds, 6),
        "total_tokens_per_second": round(
            (len(input_ids) + completion_tokens) / elapsed_seconds, 6
        ),
        "first_stream_completion_tokens": first_completion_count,
        "stream_events": len(arrivals),
        "event_count": event_count,
        "saw_done": saw_done,
        "input_sha256": _hash_token_ids(input_ids),
        "output_bytes": len(output_text.encode("utf-8")),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "output_token_ids_available": output_token_ids_available,
        "output_token_ids_sha256": (
            _hash_token_ids(output_ids) if output_token_ids_available else None
        ),
        "terminal_token_scan": (
            "observed_not_gated"
            if diagnostic_only and terminal_ids and output_ids
            else (
                "unavailable_diagnostic_only"
                if diagnostic_only
                else ("verified" if terminal_ids and output_ids else "unused")
            )
        ),
        "terminal_token_count": len(terminal_positions),
        "semantic_validation": semantic_receipt,
    }


def _terminal_token_ids(tokenizer: ServingTokenizer) -> tuple[int, ...]:
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool):
        return (eos_token_id,)
    if isinstance(eos_token_id, list) and not any(
        isinstance(token_id, bool) for token_id in eos_token_id
    ):
        return tuple(eos_token_id)
    return ()


def terminal_token_ids(tokenizer: ServingTokenizer) -> tuple[int, ...]:
    """Public terminal-token projection for sibling benchmark harnesses."""
    return _terminal_token_ids(tokenizer)


def load_tokenizer(model_path: Path) -> ServingTokenizer:
    transformers_module = importlib.import_module("transformers")
    raw_factory = cast(object, transformers_module.PreTrainedTokenizerFast)
    factory = cast(TokenizerFactory, raw_factory)
    return factory.from_pretrained(str(model_path))


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_path)
    encode_messages = load_message_encoder(args.model_path)
    input_ids = build_exact_prompt_ids(
        tokenizer,
        encode_messages,
        args.input_tokens,
    )

    def decode_output_ids(token_ids: list[int]) -> str:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    try:
        result = run_benchmark(
            args.url,
            input_ids,
            args.output_tokens,
            args.timeout,
            args.progress_every,
            ignore_eos=args.ignore_eos,
            decode_output_ids=decode_output_ids,
            terminal_token_ids=_terminal_token_ids(tokenizer),
        )
    except BenchmarkValidationError as error:
        rejection = {
            "receipt_version": RECEIPT_VERSION,
            "accepted": False,
            "performance_claim_eligible": False,
            "validation_issue_codes": list(error.issue_codes),
        }
        print(json.dumps(rejection, sort_keys=True), file=sys.stderr)
        return 2

    result_json = json.dumps(result, sort_keys=True)
    if args.output_file is not None:
        args.output_file.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
