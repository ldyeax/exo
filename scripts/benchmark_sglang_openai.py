#!/usr/bin/env python3
"""Benchmark an SGLang OpenAI chat endpoint with deterministic requests.

The client uses only the Python standard library.  It records token usage,
end-to-end request time, completion-token throughput, and exact output hashes.
Streaming mode additionally records client TTFT and post-first-token rate;
nonstream mode records SGLang server timing metadata.  Generated text is omitted
from JSONL receipts; use ``--show-content`` when a human needs to inspect the
responses for semantic coherence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self, TextIO, cast

SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 600.0
MAXIMUM_ERROR_DETAIL_CHARACTERS = 4096

type JsonObject = dict[str, object]


class BenchmarkError(RuntimeError):
    """The benchmark could not produce a trustworthy result."""


class ReadableResponse(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def read(self) -> bytes: ...


class LineReadableResponse(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def readline(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    endpoint: str
    model: str
    prompt: str
    max_tokens: int
    warmups: int
    samples: int
    timeout_seconds: float
    output_jsonl: Path | None
    show_content: bool
    enable_thinking: bool | None
    stream: bool
    api_key: str | None
    api_key_environment_variable: str

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)

    def public_json(self) -> JsonObject:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "prompt_utf8_bytes": len(self.prompt.encode("utf-8")),
            "max_tokens": self.max_tokens,
            "warmups": self.warmups,
            "samples": self.samples,
            "timeout_seconds": self.timeout_seconds,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": self.stream,
            },
            "chat_template_kwargs": (
                None
                if self.enable_thinking is None
                else {"enable_thinking": self.enable_thinking}
            ),
            "api_key_configured": self.api_key is not None,
            "api_key_environment_variable": self.api_key_environment_variable,
        }


@dataclass(frozen=True, slots=True)
class ServerMetadata:
    raw: JsonObject
    e2e_latency_seconds: float | None
    reported_decode_throughput: float | None
    request_received_timestamp: float | None
    api_server_dispatch_finish_timestamp: float | None
    request_finished_timestamp: float | None
    response_sent_to_client_timestamp: float | None
    forward_entry_timestamp: float | None
    prefill_finished_timestamp: float | None
    speculative_accept_rate: float | None
    speculative_accept_length: float | None
    speculative_cap_length: float | None
    speculative_verify_count: int | None

    @property
    def scheduler_prefill_seconds(self) -> float | None:
        if (
            self.forward_entry_timestamp is None
            or self.prefill_finished_timestamp is None
        ):
            return None
        return self.prefill_finished_timestamp - self.forward_entry_timestamp

    @property
    def scheduler_decode_seconds(self) -> float | None:
        if (
            self.prefill_finished_timestamp is None
            or self.request_finished_timestamp is None
        ):
            return None
        return self.request_finished_timestamp - self.prefill_finished_timestamp

    def prefill_tokens_per_second(self, prompt_tokens: int) -> float | None:
        duration = self.scheduler_prefill_seconds
        if duration is None or duration <= 0.0:
            return None
        return prompt_tokens / duration

    def decode_tokens_per_second(self, completion_tokens: int) -> float | None:
        duration = self.scheduler_decode_seconds
        if duration is None or duration <= 0.0 or completion_tokens <= 1:
            return None
        return (completion_tokens - 1) / duration


@dataclass(frozen=True, slots=True)
class CompletionResult:
    phase: str
    index: int
    started_at_utc: str
    wall_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None
    content: str
    reasoning_content: str | None
    finish_reason: str | None
    response_id: str | None
    client_ttft_seconds: float | None
    sse_json_events: int | None
    sse_nonempty_text_events: int | None
    server_metadata: ServerMetadata | None

    @property
    def completion_tokens_per_second(self) -> float:
        return self.completion_tokens / self.wall_seconds

    @property
    def content_sha256(self) -> str:
        return sha256_text(self.content)

    @property
    def post_first_token_seconds(self) -> float | None:
        if self.client_ttft_seconds is None:
            return None
        duration = self.wall_seconds - self.client_ttft_seconds
        return duration if duration >= 0.0 else None

    @property
    def post_first_token_completion_tokens_per_second(self) -> float | None:
        duration = self.post_first_token_seconds
        if self.completion_tokens <= 1 or duration is None or duration <= 0.0:
            return None
        return (self.completion_tokens - 1) / duration

    @property
    def server_prefill_tokens_per_second(self) -> float | None:
        if self.server_metadata is None:
            return None
        return self.server_metadata.prefill_tokens_per_second(self.prompt_tokens)

    @property
    def server_decode_tokens_per_second(self) -> float | None:
        if self.server_metadata is None:
            return None
        return self.server_metadata.decode_tokens_per_second(self.completion_tokens)

    def receipt(
        self, *, campaign_id: str, configuration: BenchmarkConfiguration
    ) -> JsonObject:
        reasoning_hash = (
            sha256_text(self.reasoning_content)
            if self.reasoning_content is not None
            else None
        )
        receipt: JsonObject = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "completion",
            "campaign_id": campaign_id,
            "recorded_at_utc": utc_now(),
            "phase": self.phase,
            "index": self.index,
            "started_at_utc": self.started_at_utc,
            "endpoint": configuration.endpoint,
            "model": configuration.model,
            "prompt_sha256": configuration.prompt_sha256,
            "max_tokens": configuration.max_tokens,
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": configuration.stream,
            },
            "wall_seconds": self.wall_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "completion_tokens_per_second": self.completion_tokens_per_second,
            "content_sha256": self.content_sha256,
            "content_utf8_bytes": len(self.content.encode("utf-8")),
            "reasoning_content_sha256": reasoning_hash,
            "reasoning_content_utf8_bytes": (
                len(self.reasoning_content.encode("utf-8"))
                if self.reasoning_content is not None
                else None
            ),
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
        }
        if configuration.stream:
            receipt.update(
                {
                    "client_ttft_seconds": self.client_ttft_seconds,
                    "post_first_token_seconds": self.post_first_token_seconds,
                    "post_first_token_completion_tokens_per_second": (
                        self.post_first_token_completion_tokens_per_second
                    ),
                    "full_wall_completion_tokens_per_second": (
                        self.completion_tokens_per_second
                    ),
                    "sse_json_events": self.sse_json_events,
                    "sse_nonempty_text_events": self.sse_nonempty_text_events,
                }
            )
        elif self.server_metadata is not None:
            receipt.update(
                {
                    "server_meta_info": self.server_metadata.raw,
                    "server_e2e_latency_seconds": (
                        self.server_metadata.e2e_latency_seconds
                    ),
                    # SGLang's reported decode throughput starts at its first
                    # IPC flush, so it is stream-interval-biased. Preserve it
                    # as raw evidence, but do not present it as decode speed.
                    "server_reported_decode_throughput": (
                        self.server_metadata.reported_decode_throughput
                    ),
                    "server_request_received_timestamp": (
                        self.server_metadata.request_received_timestamp
                    ),
                    "server_api_dispatch_finish_timestamp": (
                        self.server_metadata.api_server_dispatch_finish_timestamp
                    ),
                    "server_request_finished_timestamp": (
                        self.server_metadata.request_finished_timestamp
                    ),
                    "server_response_sent_to_client_timestamp": (
                        self.server_metadata.response_sent_to_client_timestamp
                    ),
                    "server_forward_entry_timestamp": (
                        self.server_metadata.forward_entry_timestamp
                    ),
                    "server_prefill_finished_timestamp": (
                        self.server_metadata.prefill_finished_timestamp
                    ),
                    "server_scheduler_prefill_seconds": (
                        self.server_metadata.scheduler_prefill_seconds
                    ),
                    "server_prefill_tokens_per_second": (
                        self.server_prefill_tokens_per_second
                    ),
                    "server_scheduler_decode_seconds": (
                        self.server_metadata.scheduler_decode_seconds
                    ),
                    "server_decode_tokens_per_second": (
                        self.server_decode_tokens_per_second
                    ),
                    "server_speculative_accept_rate": (
                        self.server_metadata.speculative_accept_rate
                    ),
                    "server_speculative_accept_length": (
                        self.server_metadata.speculative_accept_length
                    ),
                    "server_speculative_cap_length": (
                        self.server_metadata.speculative_cap_length
                    ),
                    "server_speculative_verify_count": (
                        self.server_metadata.speculative_verify_count
                    ),
                }
            )
        return receipt


@dataclass(frozen=True, slots=True)
class StreamingResponsePayload:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None
    content: str
    reasoning_content: str | None
    finish_reason: str | None
    response_id: str | None
    client_ttft_seconds: float | None
    sse_json_events: int
    sse_nonempty_text_events: int


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with millisecond resolution."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of the exact UTF-8 representation of text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="full OpenAI-compatible /v1/chat/completions endpoint URL",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model name or served-model alias sent in each request",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="literal user prompt")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file whose complete contents become the user prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_integer,
        default=DEFAULT_MAX_TOKENS,
        help="maximum completion tokens per request",
    )
    parser.add_argument(
        "--warmups",
        type=nonnegative_integer,
        default=1,
        help="warmup requests excluded from aggregate statistics",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=3,
        help="measured requests included in aggregate statistics",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP connect/read timeout for each request",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="append per-request records and one campaign summary to this JSONL file",
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="print response content for human semantic-coherence inspection",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "use OpenAI streaming SSE and record client-observed TTFT plus "
            "post-first-token throughput"
        ),
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        action="store_true",
        default=None,
        help="explicitly enable the model chat template's thinking mode",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="explicitly disable the model chat template's thinking mode",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable containing an optional bearer API key",
    )
    return parser


def _validated_endpoint(value: str) -> str:
    endpoint = value.strip()
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BenchmarkError("--endpoint must be an absolute HTTP(S) URL")
    if parsed.fragment:
        raise BenchmarkError("--endpoint must not contain a URL fragment")
    return endpoint


def configuration_from_arguments(
    arguments: argparse.Namespace,
) -> BenchmarkConfiguration:
    model = cast(str, arguments.model).strip()
    if not model:
        raise BenchmarkError("--model must not be empty")

    prompt_argument = cast(str | None, arguments.prompt)
    prompt_path = cast(Path | None, arguments.prompt_file)
    if prompt_argument is not None:
        prompt = prompt_argument
    elif prompt_path is not None:
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkError(
                f"cannot read UTF-8 prompt file {prompt_path}: {error}"
            ) from error
    else:  # argparse enforces this; retain a defensive check for direct callers.
        raise BenchmarkError("exactly one of --prompt or --prompt-file is required")
    if not prompt:
        raise BenchmarkError("the prompt must not be empty")

    api_key_environment_variable = cast(str, arguments.api_key_env).strip()
    if not api_key_environment_variable:
        raise BenchmarkError("--api-key-env must not be empty")
    api_key = os.environ.get(api_key_environment_variable)
    if api_key == "":
        api_key = None

    return BenchmarkConfiguration(
        endpoint=_validated_endpoint(cast(str, arguments.endpoint)),
        model=model,
        prompt=prompt,
        max_tokens=cast(int, arguments.max_tokens),
        warmups=cast(int, arguments.warmups),
        samples=cast(int, arguments.samples),
        timeout_seconds=cast(float, arguments.timeout_seconds),
        output_jsonl=cast(Path | None, arguments.output_jsonl),
        show_content=cast(bool, arguments.show_content),
        enable_thinking=cast(bool | None, arguments.enable_thinking),
        stream=cast(bool, arguments.stream),
        api_key=api_key,
        api_key_environment_variable=api_key_environment_variable,
    )


def _json_object(value: object, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{context} must be a JSON object")
    unknown_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in unknown_mapping):
        raise BenchmarkError(f"{context} contains a non-string key")
    return cast(JsonObject, unknown_mapping)


def _nonnegative_usage_integer(usage: JsonObject, key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"response usage.{key} must be a nonnegative integer")
    return value


def _optional_string(mapping: JsonObject, key: str, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkError(f"{context}.{key} must be a string or null")
    return value


def _message_text(value: object, context: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise BenchmarkError(f"{context} must be a string or a list of text parts")

    text_parts: list[str] = []
    for part_index, raw_part in enumerate(cast(list[object], value)):
        part = _json_object(raw_part, f"{context}[{part_index}]")
        text = part.get("text")
        if not isinstance(text, str):
            raise BenchmarkError(f"{context}[{part_index}].text must be a string")
        text_parts.append(text)
    return "".join(text_parts)


def _response_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8", errors="replace")
    except OSError:
        detail = ""
    detail = detail.strip()
    if len(detail) > MAXIMUM_ERROR_DETAIL_CHARACTERS:
        detail = detail[:MAXIMUM_ERROR_DETAIL_CHARACTERS] + "..."
    return f": {detail}" if detail else ""


def _raise_for_response_error(document: JsonObject) -> None:
    if document.get("error") is None:
        return
    error_value = json.dumps(document["error"], ensure_ascii=False)
    if len(error_value) > MAXIMUM_ERROR_DETAIL_CHARACTERS:
        error_value = error_value[:MAXIMUM_ERROR_DETAIL_CHARACTERS] + "..."
    raise BenchmarkError(f"server returned an error object: {error_value}")


def _optional_finite_number(
    mapping: JsonObject, key: str, context: str
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{context}.{key} must be a finite number or null")
    parsed = float(value)
    if not float("-inf") < parsed < float("inf"):
        raise BenchmarkError(f"{context}.{key} must be a finite number or null")
    return parsed


def _optional_nonnegative_integer(
    mapping: JsonObject, key: str, context: str
) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{context}.{key} must be a nonnegative integer or null")
    return value


def _server_metadata(choice: JsonObject) -> ServerMetadata | None:
    value = choice.get("meta_info")
    if value is None:
        return None
    meta_info = _json_object(value, "response.choices[0].meta_info")
    return ServerMetadata(
        raw=meta_info,
        e2e_latency_seconds=_optional_finite_number(
            meta_info, "e2e_latency", "response.choices[0].meta_info"
        ),
        reported_decode_throughput=_optional_finite_number(
            meta_info, "decode_throughput", "response.choices[0].meta_info"
        ),
        request_received_timestamp=_optional_finite_number(
            meta_info, "request_received_ts", "response.choices[0].meta_info"
        ),
        api_server_dispatch_finish_timestamp=_optional_finite_number(
            meta_info,
            "api_server_dispatch_finish_ts",
            "response.choices[0].meta_info",
        ),
        request_finished_timestamp=_optional_finite_number(
            meta_info, "request_finished_ts", "response.choices[0].meta_info"
        ),
        response_sent_to_client_timestamp=_optional_finite_number(
            meta_info,
            "response_sent_to_client_ts",
            "response.choices[0].meta_info",
        ),
        forward_entry_timestamp=_optional_finite_number(
            meta_info, "forward_entry_time", "response.choices[0].meta_info"
        ),
        prefill_finished_timestamp=_optional_finite_number(
            meta_info, "prefill_finished_time", "response.choices[0].meta_info"
        ),
        speculative_accept_rate=_optional_finite_number(
            meta_info, "spec_accept_rate", "response.choices[0].meta_info"
        ),
        speculative_accept_length=_optional_finite_number(
            meta_info, "spec_accept_length", "response.choices[0].meta_info"
        ),
        speculative_cap_length=_optional_finite_number(
            meta_info, "spec_cap_length", "response.choices[0].meta_info"
        ),
        speculative_verify_count=_optional_nonnegative_integer(
            meta_info, "spec_verify_ct", "response.choices[0].meta_info"
        ),
    )


def request_document(configuration: BenchmarkConfiguration) -> JsonObject:
    request_document: JsonObject = {
        "model": configuration.model,
        "messages": [{"role": "user", "content": configuration.prompt}],
        "max_tokens": configuration.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": configuration.stream,
    }
    if configuration.stream:
        request_document["stream_options"] = {"include_usage": True}
    else:
        request_document["return_meta_info"] = True
    if configuration.enable_thinking is not None:
        request_document["chat_template_kwargs"] = {
            "enable_thinking": configuration.enable_thinking
        }
    return request_document


def iter_sse_data(response: LineReadableResponse) -> Iterator[bytes]:
    """Yield complete SSE ``data`` payloads from a byte-oriented response."""

    data_lines: list[bytes] = []
    while True:
        raw_line = response.readline()
        if raw_line == b"":
            if data_lines:
                yield b"\n".join(data_lines)
            return

        line = raw_line.rstrip(b"\r\n")
        if line == b"":
            if data_lines:
                yield b"\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(b":"):
            continue

        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            data_lines.append(value)


def _stream_delta_text(delta: JsonObject, key: str, context: str) -> str:
    value = delta.get(key)
    if value is None:
        return ""
    return _message_text(value, f"{context}.{key}")


def read_streaming_response(
    response: LineReadableResponse,
    *,
    started_at: float,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamingResponsePayload:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_seen = False
    finish_reason: str | None = None
    response_id: str | None = None
    usage: JsonObject | None = None
    client_ttft_seconds: float | None = None
    sse_json_events = 0
    sse_nonempty_text_events = 0
    saw_done = False

    for raw_data in iter_sse_data(response):
        if raw_data.strip() == b"[DONE]":
            saw_done = True
            break
        try:
            decoded = cast(object, json.loads(raw_data.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BenchmarkError("stream contained invalid UTF-8 JSON data") from error

        document = _json_object(decoded, "stream event")
        _raise_for_response_error(document)
        sse_json_events += 1

        event_response_id = _optional_string(document, "id", "stream event")
        if event_response_id is not None:
            response_id = event_response_id

        usage_value = document.get("usage")
        if usage_value is not None:
            usage = _json_object(usage_value, "stream event.usage")

        choices_value = document.get("choices")
        if choices_value is None:
            continue
        if not isinstance(choices_value, list):
            raise BenchmarkError("stream event.choices must be a list")
        if not choices_value:
            continue

        choice = _json_object(
            cast(list[object], choices_value)[0], "stream event.choices[0]"
        )
        event_finish_reason = _optional_string(
            choice, "finish_reason", "stream event.choices[0]"
        )
        if event_finish_reason is not None:
            finish_reason = event_finish_reason

        delta_value = choice.get("delta")
        if delta_value is None:
            continue
        delta = _json_object(delta_value, "stream event.choices[0].delta")
        content = _stream_delta_text(delta, "content", "stream event.choices[0].delta")
        reasoning = _stream_delta_text(
            delta, "reasoning_content", "stream event.choices[0].delta"
        )
        if delta.get("reasoning_content") is not None:
            reasoning_seen = True
        content_parts.append(content)
        reasoning_parts.append(reasoning)

        if content != "" or reasoning != "":
            sse_nonempty_text_events += 1
            if client_ttft_seconds is None:
                client_ttft_seconds = max(clock() - started_at, 0.0)

    if not saw_done:
        raise BenchmarkError("stream ended before the [DONE] sentinel")
    if usage is None:
        raise BenchmarkError(
            "stream did not include usage; the server must honor "
            "stream_options.include_usage"
        )

    prompt_tokens = _nonnegative_usage_integer(usage, "prompt_tokens")
    completion_tokens = _nonnegative_usage_integer(usage, "completion_tokens")
    total_value = usage.get("total_tokens")
    total_tokens = (
        None
        if total_value is None
        else _nonnegative_usage_integer(usage, "total_tokens")
    )
    return StreamingResponsePayload(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts) if reasoning_seen else None,
        finish_reason=finish_reason,
        response_id=response_id,
        client_ttft_seconds=client_ttft_seconds,
        sse_json_events=sse_json_events,
        sse_nonempty_text_events=sse_nonempty_text_events,
    )


def request_completion(
    configuration: BenchmarkConfiguration,
    *,
    phase: str,
    index: int,
) -> CompletionResult:
    document = request_document(configuration)
    body = json.dumps(document, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "text/event-stream" if configuration.stream else "application/json",
        "Content-Type": "application/json",
        "User-Agent": "exo-sglang-openai-benchmark/1",
    }
    if configuration.api_key is not None:
        headers["Authorization"] = f"Bearer {configuration.api_key}"
    request = urllib.request.Request(
        configuration.endpoint,
        data=body,
        headers=headers,
        method="POST",
    )

    started_at_utc = utc_now()
    started_at = time.perf_counter()
    payload: bytes | None = None
    streaming_payload: StreamingResponsePayload | None = None
    try:
        opened_response = cast(
            object,
            urllib.request.urlopen(
                request,
                timeout=configuration.timeout_seconds,
            ),
        )
        if configuration.stream:
            with cast(LineReadableResponse, opened_response) as response:
                streaming_payload = read_streaming_response(
                    response,
                    started_at=started_at,
                )
        else:
            with cast(ReadableResponse, opened_response) as response:
                payload = response.read()
    except urllib.error.HTTPError as error:
        raise BenchmarkError(
            f"request failed with HTTP {error.code}{_response_error_detail(error)}"
        ) from error
    except urllib.error.URLError as error:
        raise BenchmarkError(f"request failed: {error.reason}") from error
    except (TimeoutError, OSError) as error:
        raise BenchmarkError(f"request failed: {error}") from error
    wall_seconds = time.perf_counter() - started_at
    if wall_seconds <= 0.0:
        raise BenchmarkError("measured request wall time was not positive")

    if streaming_payload is not None:
        return CompletionResult(
            phase=phase,
            index=index,
            started_at_utc=started_at_utc,
            wall_seconds=wall_seconds,
            prompt_tokens=streaming_payload.prompt_tokens,
            completion_tokens=streaming_payload.completion_tokens,
            total_tokens=streaming_payload.total_tokens,
            content=streaming_payload.content,
            reasoning_content=streaming_payload.reasoning_content,
            finish_reason=streaming_payload.finish_reason,
            response_id=streaming_payload.response_id,
            client_ttft_seconds=streaming_payload.client_ttft_seconds,
            sse_json_events=streaming_payload.sse_json_events,
            sse_nonempty_text_events=(streaming_payload.sse_nonempty_text_events),
            server_metadata=None,
        )
    if payload is None:
        raise BenchmarkError("server response body was unexpectedly unavailable")

    try:
        decoded = cast(object, json.loads(payload.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError("server response was not valid UTF-8 JSON") from error
    document = _json_object(decoded, "server response")
    _raise_for_response_error(document)

    choices_value = document.get("choices")
    if not isinstance(choices_value, list) or not choices_value:
        raise BenchmarkError("server response.choices must be a nonempty list")
    choice = _json_object(cast(list[object], choices_value)[0], "response.choices[0]")
    message = _json_object(choice.get("message"), "response.choices[0].message")
    content = _message_text(
        message.get("content"), "response.choices[0].message.content"
    )
    reasoning_value = message.get("reasoning_content")
    reasoning_content = (
        None
        if reasoning_value is None
        else _message_text(
            reasoning_value,
            "response.choices[0].message.reasoning_content",
        )
    )

    usage = _json_object(document.get("usage"), "response.usage")
    prompt_tokens = _nonnegative_usage_integer(usage, "prompt_tokens")
    completion_tokens = _nonnegative_usage_integer(usage, "completion_tokens")
    total_value = usage.get("total_tokens")
    total_tokens = (
        None
        if total_value is None
        else _nonnegative_usage_integer(usage, "total_tokens")
    )

    return CompletionResult(
        phase=phase,
        index=index,
        started_at_utc=started_at_utc,
        wall_seconds=wall_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        content=content,
        reasoning_content=reasoning_content,
        finish_reason=_optional_string(choice, "finish_reason", "response.choices[0]"),
        response_id=_optional_string(document, "id", "response"),
        client_ttft_seconds=None,
        sse_json_events=None,
        sse_nonempty_text_events=None,
        server_metadata=_server_metadata(choice),
    )


def summarize(values: list[float | int]) -> JsonObject:
    if not values:
        raise BenchmarkError("cannot summarize an empty measurement list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_available(values: list[float | int | None]) -> JsonObject:
    available_values = [value for value in values if value is not None]
    if not available_values:
        return {
            "count": len(values),
            "available_count": 0,
            "missing_count": len(values),
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "available_count": len(available_values),
        "missing_count": len(values) - len(available_values),
        "mean": statistics.fmean(available_values),
        "median": statistics.median(available_values),
        "minimum": min(available_values),
        "maximum": max(available_values),
    }


def campaign_summary(
    *,
    campaign_id: str,
    configuration: BenchmarkConfiguration,
    results: list[CompletionResult],
) -> JsonObject:
    content_hashes = [result.content_sha256 for result in results]
    distinct_content_hashes = sorted(set(content_hashes))
    summary: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "summary",
        "campaign_id": campaign_id,
        "recorded_at_utc": utc_now(),
        "configuration": configuration.public_json(),
        "warmups_excluded_from_aggregates": True,
        "wall_seconds": summarize([result.wall_seconds for result in results]),
        "prompt_tokens": summarize([result.prompt_tokens for result in results]),
        "completion_tokens": summarize(
            [result.completion_tokens for result in results]
        ),
        "completion_tokens_per_second": summarize(
            [result.completion_tokens_per_second for result in results]
        ),
        "content_hash_consistency": {
            "all_identical": len(distinct_content_hashes) == 1,
            "distinct_count": len(distinct_content_hashes),
            "distinct_sha256": distinct_content_hashes,
        },
        "all_content_nonempty": all(result.content != "" for result in results),
    }
    if configuration.stream:
        summary.update(
            {
                "client_ttft_seconds": summarize_available(
                    [result.client_ttft_seconds for result in results]
                ),
                "post_first_token_seconds": summarize_available(
                    [result.post_first_token_seconds for result in results]
                ),
                "post_first_token_completion_tokens_per_second": (
                    summarize_available(
                        [
                            result.post_first_token_completion_tokens_per_second
                            for result in results
                        ]
                    )
                ),
                "full_wall_completion_tokens_per_second": summarize(
                    [result.completion_tokens_per_second for result in results]
                ),
                "sse_json_events": summarize_available(
                    [result.sse_json_events for result in results]
                ),
                "sse_nonempty_text_events": summarize_available(
                    [result.sse_nonempty_text_events for result in results]
                ),
            }
        )
    else:
        summary.update(
            {
                "server_reported_decode_throughput": summarize_available(
                    [
                        (
                            result.server_metadata.reported_decode_throughput
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_prefill_tokens_per_second": summarize_available(
                    [result.server_prefill_tokens_per_second for result in results]
                ),
                "server_scheduler_prefill_seconds": summarize_available(
                    [
                        (
                            result.server_metadata.scheduler_prefill_seconds
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_decode_tokens_per_second": summarize_available(
                    [result.server_decode_tokens_per_second for result in results]
                ),
                "server_scheduler_decode_seconds": summarize_available(
                    [
                        (
                            result.server_metadata.scheduler_decode_seconds
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_speculative_accept_rate": summarize_available(
                    [
                        (
                            result.server_metadata.speculative_accept_rate
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_speculative_accept_length": summarize_available(
                    [
                        (
                            result.server_metadata.speculative_accept_length
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_speculative_cap_length": summarize_available(
                    [
                        (
                            result.server_metadata.speculative_cap_length
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
                "server_speculative_verify_count": summarize_available(
                    [
                        (
                            result.server_metadata.speculative_verify_count
                            if result.server_metadata is not None
                            else None
                        )
                        for result in results
                    ]
                ),
            }
        )
    return summary


def _open_jsonl(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", newline="\n")
    except OSError as error:
        raise BenchmarkError(f"cannot open JSONL receipt {path}: {error}") from error


def _write_jsonl(stream: TextIO | None, document: JsonObject) -> None:
    if stream is None:
        return
    try:
        stream.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
    except OSError as error:
        raise BenchmarkError(f"cannot write JSONL receipt: {error}") from error


def _report_result(result: CompletionResult, total_in_phase: int) -> None:
    report = (
        f"{result.phase} {result.index}/{total_in_phase}: "
        f"wall={result.wall_seconds:.6f}s "
        f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
        f"rate={result.completion_tokens_per_second:.3f} tok/s "
        f"content_sha256={result.content_sha256}"
    )
    if result.client_ttft_seconds is not None:
        post_first_rate = result.post_first_token_completion_tokens_per_second
        formatted_post_first_rate = (
            "n/a" if post_first_rate is None else f"{post_first_rate:.3f} tok/s"
        )
        report += (
            f" ttft={result.client_ttft_seconds:.6f}s "
            f"post_first_rate={formatted_post_first_rate}"
        )
    if result.server_decode_tokens_per_second is not None:
        prefill_rate = result.server_prefill_tokens_per_second
        formatted_prefill_rate = (
            "n/a" if prefill_rate is None else f"{prefill_rate:.3f} tok/s"
        )
        report += (
            f" server_prefill_rate={formatted_prefill_rate} "
            f"server_decode_rate={result.server_decode_tokens_per_second:.3f} tok/s"
        )
    print(report)


def _show_content(result: CompletionResult) -> None:
    label = f"{result.phase} {result.index}"
    if result.reasoning_content is not None:
        print(f"--- {label} reasoning_content ---")
        print(result.reasoning_content)
    print(f"--- {label} content ---")
    print(result.content)
    print(f"--- end {label} ---")


def run(configuration: BenchmarkConfiguration) -> JsonObject:
    campaign_id = str(uuid.uuid4())
    output_stream = _open_jsonl(configuration.output_jsonl)
    measured_results: list[CompletionResult] = []
    try:
        for phase, count in (
            ("warmup", configuration.warmups),
            ("sample", configuration.samples),
        ):
            for index in range(1, count + 1):
                result = request_completion(
                    configuration,
                    phase=phase,
                    index=index,
                )
                _report_result(result, count)
                if configuration.show_content:
                    _show_content(result)
                _write_jsonl(
                    output_stream,
                    result.receipt(
                        campaign_id=campaign_id,
                        configuration=configuration,
                    ),
                )
                if phase == "sample":
                    measured_results.append(result)

        summary = campaign_summary(
            campaign_id=campaign_id,
            configuration=configuration,
            results=measured_results,
        )
        _write_jsonl(output_stream, summary)
        return summary
    finally:
        if output_stream is not None:
            output_stream.close()


def main(arguments: list[str] | None = None) -> int:
    parser = build_argument_parser()
    parsed_arguments = parser.parse_args(arguments)
    try:
        configuration = configuration_from_arguments(parsed_arguments)
        summary = run(configuration)
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
