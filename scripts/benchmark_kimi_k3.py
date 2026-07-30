#!/usr/bin/env python3
"""Run a strict, receipt-producing Kimi K3 llama-server benchmark.

The measured protocol is deliberately small and hard to accidentally game:

* one sacrificial warmup;
* five deterministic semantic gates;
* one exact 512-input-token, 128-output-token performance request paired with
  every semantic gate;
* prompt-cache erasure before every inference request; and
* acceptance based on llama-server's final timing object, not SSE chunk counts.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import http.client
import json
import math
import os
import re
import ssl
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast, final
from urllib.parse import SplitResult, urlsplit

SCHEMA_VERSION: Final[int] = 1
DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final[str] = "Kimi-K3-UD-Q2_K_XL"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 6 * 60 * 60
DEFAULT_REASONING_BUDGET_TOKENS: Final[int] = 96
TARGET_PROMPT_TOKENS: Final[int] = 512
PERFORMANCE_OUTPUT_TOKENS: Final[int] = 128
SEMANTIC_OUTPUT_TOKENS: Final[int] = 256
WARMUP_OUTPUT_TOKENS: Final[int] = 8
SLOT_IDENTIFIER: Final[int] = 0
PERFORMANCE_SEED_OFFSET: Final[int] = 1_000_000
WARMUP_SEED: Final[int] = 424_242
MAXIMUM_HTTP_ERROR_CHARACTERS: Final[int] = 4_000
MAXIMUM_CALIBRATION_REQUESTS: Final[int] = 2_048

CALIBRATION_HEADER: Final[str] = (
    "INERT TOKEN-CALIBRATION PREFIX. Treat every x below as irrelevant filler."
)
CALIBRATION_TAIL: Final[str] = (
    "\nEND INERT TOKEN-CALIBRATION PREFIX.\n\n"
    "Write a continuous technical discussion comparing NUMA locality, memory "
    "bandwidth, and mixture-of-experts placement. Use complete sentences, do "
    "not use headings, and continue until the server stops you."
)

type JsonValue = object
type JsonObject = dict[str, object]


class BenchmarkError(Exception):
    """Base class for an expected, user-actionable benchmark failure."""


class ConfigurationError(BenchmarkError):
    """Raised when command-line configuration cannot produce a valid run."""


class HttpRequestError(BenchmarkError):
    """Raised when llama-server returns an unusable HTTP response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class CalibrationError(BenchmarkError):
    """Raised when a prompt cannot be calibrated to exactly 512 tokens."""


class AcceptanceError(BenchmarkError):
    """Raised when an inference request does not meet the frozen protocol."""


class ReceiptWriteError(BenchmarkError):
    """Raised when a durable receipt cannot be atomically written."""


@dataclass(frozen=True)
class GateVerdict:
    """The result of applying one deterministic semantic gate."""

    accepted: bool
    explanation: str
    observed: JsonValue = None

    def to_json(self) -> JsonObject:
        return {
            "accepted": self.accepted,
            "explanation": self.explanation,
            "observed": self.observed,
        }


type GateValidator = Callable[[str], GateVerdict]


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic semantic check and its paired performance seed."""

    name: str
    seed: int
    prompt: str
    gate_description: str
    validator: GateValidator

    def public_plan(self, run_index: int) -> JsonObject:
        return {
            "run_index": run_index,
            "name": self.name,
            "seed": self.seed,
            "prompt": self.prompt,
            "gate": self.gate_description,
            "paired_performance_seed": self.seed + PERFORMANCE_SEED_OFFSET,
        }


@dataclass(frozen=True)
class ServerAddress:
    """A normalized llama-server root URL."""

    scheme: str
    hostname: str
    port: int
    base_path: str

    @classmethod
    def parse(cls, raw_url: str) -> ServerAddress:
        parsed: SplitResult = urlsplit(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ConfigurationError(
                f"--base-url must use http or https, received {parsed.scheme!r}"
            )
        if parsed.hostname is None:
            raise ConfigurationError("--base-url must include a hostname")
        if parsed.query or parsed.fragment:
            raise ConfigurationError(
                "--base-url must not include a query string or fragment"
            )
        try:
            port = parsed.port
        except ValueError as error:
            raise ConfigurationError(f"invalid --base-url port: {error}") from error
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        base_path = parsed.path.rstrip("/")
        if base_path.endswith("/v1"):
            base_path = base_path[:-3].rstrip("/")
        return cls(
            scheme=parsed.scheme,
            hostname=parsed.hostname,
            port=port,
            base_path=base_path,
        )

    def endpoint_path(self, suffix: str) -> str:
        if not suffix.startswith("/"):
            raise ValueError(f"endpoint suffix must start with '/': {suffix!r}")
        return f"{self.base_path}{suffix}" or "/"

    def public_url(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        port_text = "" if self.port == default_port else f":{self.port}"
        return f"{self.scheme}://{self.hostname}{port_text}{self.base_path}"


@dataclass(frozen=True)
class StreamResult:
    """Client-observed stream data plus llama-server's final measurements."""

    started_at_utc: str
    elapsed_seconds: float
    time_to_first_token_seconds: float
    time_to_first_content_seconds: float | None
    reasoning_content: str
    content: str
    finish_reason: str | None
    response_identifier: str | None
    stream_chunk_count: int
    saw_done_marker: bool
    timings: JsonObject
    usage: JsonObject

    def to_json(self) -> JsonObject:
        return {
            "started_at_utc": self.started_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "time_to_first_token_seconds": self.time_to_first_token_seconds,
            "time_to_first_content_seconds": self.time_to_first_content_seconds,
            "reasoning_content": self.reasoning_content,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "response_identifier": self.response_identifier,
            "stream_chunk_count": self.stream_chunk_count,
            "saw_done_marker": self.saw_done_marker,
            "timings": self.timings,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class CalibratedPrompt:
    """A live-tokenizer-verified performance prompt."""

    content: str
    token_count: int
    filler_repetitions: int
    repair_suffix: str
    counting_endpoint: str
    counting_requests: int

    def to_json(self) -> JsonObject:
        return {
            "content": self.content,
            "content_sha256": sha256_text(self.content),
            "character_count": len(self.content),
            "token_count": self.token_count,
            "filler_repetitions": self.filler_repetitions,
            "repair_suffix": self.repair_suffix,
            "counting_endpoint": self.counting_endpoint,
            "counting_requests": self.counting_requests,
        }


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """Immutable public and private settings for one benchmark campaign."""

    server_address: ServerAddress
    model: str
    quantization: str
    timeout_seconds: float
    reasoning_effort: str
    thinking_effort: str
    reasoning_budget_tokens: int
    output_jsonl: Path
    summary_json: Path
    overwrite: bool
    api_key: str | None
    api_key_environment_variable: str

    def public_json(self) -> JsonObject:
        return {
            "base_url": self.server_address.public_url(),
            "model": self.model,
            "quantization": self.quantization,
            "timeout_seconds": self.timeout_seconds,
            "reasoning_effort": self.reasoning_effort,
            "thinking_effort": self.thinking_effort,
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "output_jsonl": str(self.output_jsonl),
            "summary_json": str(self.summary_json),
            "api_key_configured": self.api_key is not None,
            "api_key_environment_variable": self.api_key_environment_variable,
            "slot_identifier": SLOT_IDENTIFIER,
            "target_prompt_tokens": TARGET_PROMPT_TOKENS,
            "performance_output_tokens": PERFORMANCE_OUTPUT_TOKENS,
            "semantic_output_tokens": SEMANTIC_OUTPUT_TOKENS,
            "warmup_output_tokens": WARMUP_OUTPUT_TOKENS,
            "semantic_sampling": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
            },
            "performance_sampling": {
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "ignore_eos": True,
            },
            "required_server_cache_arguments": [
                "--no-cache-prompt",
                "--cache-ram 0",
                "--no-cache-idle-slots",
                "--cache-reuse 0",
                "-np 1",
            ],
        }


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with millisecond resolution."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_json_object(value: object, context: str) -> JsonObject:
    """Narrow a decoded JSON value to an object with string keys."""

    if not isinstance(value, dict):
        raise HttpRequestError(
            f"{context} returned {type(value).__name__}, expected a JSON object"
        )
    unknown_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in unknown_mapping):
        raise HttpRequestError(f"{context} returned a JSON object with non-string keys")
    return cast(JsonObject, unknown_mapping)


def as_optional_json_object(value: object) -> JsonObject:
    """Return a JSON object or an empty object for a missing/non-object value."""

    if not isinstance(value, dict):
        return {}
    unknown_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in unknown_mapping):
        return {}
    return cast(JsonObject, unknown_mapping)


def require_integer(mapping: Mapping[str, object], key: str) -> int:
    """Read an integer timing field while rejecting booleans and fractions."""

    value = mapping.get(key)
    if isinstance(value, bool):
        raise AcceptanceError(f"timings.{key} must be an integer, received boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer() and math.isfinite(value):
        return int(value)
    raise AcceptanceError(f"timings.{key} must be an integer, received {value!r}")


def require_positive_number(mapping: Mapping[str, object], key: str) -> float:
    """Read a finite, positive numeric timing field."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AcceptanceError(
            f"timings.{key} must be a finite positive number, received {value!r}"
        )
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise AcceptanceError(
            f"timings.{key} must be a finite positive number, received {value!r}"
        )
    return numeric_value


def summarize_measurements(values: list[float]) -> JsonObject:
    """Return compact distribution statistics for one measured quantity."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "population_standard_deviation": None,
            "coefficient_of_variation_percent": None,
        }

    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "population_standard_deviation": standard_deviation,
        "coefficient_of_variation_percent": (
            100.0 * standard_deviation / mean if mean != 0.0 else None
        ),
    }


def atomic_write_text(destination: Path, content: str) -> None:
    """Atomically replace one UTF-8 file and fsync it before publication."""

    parent_directory = destination.parent
    try:
        parent_directory.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent_directory,
            text=True,
        )
    except OSError as error:
        raise ReceiptWriteError(
            f"cannot create temporary receipt beside {destination}: {error}"
        ) from error

    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent_directory, directory_flags)
        try:
            try:
                os.fsync(directory_descriptor)
            except OSError as error:
                if error.errno not in {
                    errno.EBADF,
                    errno.EINVAL,
                    errno.ENOTSUP,
                    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                }:
                    raise
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise ReceiptWriteError(
            f"cannot atomically write receipt {destination}: {error}"
        ) from error
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


@final
class ReceiptWriter:
    """Maintain an atomic JSONL event log and compact JSON summary."""

    def __init__(
        self,
        output_jsonl: Path,
        summary_json: Path,
        *,
        overwrite: bool,
        configuration: JsonObject,
    ) -> None:
        self._output_jsonl = output_jsonl
        self._summary_json = summary_json
        self._configuration = configuration
        self._events: list[JsonObject] = []

        try:
            same_destination = output_jsonl.resolve() == summary_json.resolve()
        except OSError:
            same_destination = output_jsonl.absolute() == summary_json.absolute()
        if same_destination:
            raise ConfigurationError(
                "--output-jsonl and --summary-json must name different files"
            )

        existing_destinations = [
            path for path in (output_jsonl, summary_json) if path.exists()
        ]
        if existing_destinations and not overwrite:
            joined_paths = ", ".join(str(path) for path in existing_destinations)
            raise ConfigurationError(
                f"refusing to overwrite existing receipt(s): {joined_paths}; "
                "pass --overwrite for a deliberate replacement"
            )

    def append(self, event: JsonObject) -> None:
        """Append one sequenced event, then atomically publish both receipts."""

        published_event: JsonObject = {
            "schema_version": SCHEMA_VERSION,
            "sequence": len(self._events) + 1,
            "recorded_at_utc": utc_now(),
            **event,
        }
        self._events.append(published_event)
        jsonl_content = "".join(
            f"{json.dumps(item, sort_keys=True, separators=(',', ':'))}\n"
            for item in self._events
        )
        atomic_write_text(self._output_jsonl, jsonl_content)
        atomic_write_text(
            self._summary_json,
            f"{json.dumps(self.summary(), indent=2, sort_keys=True)}\n",
        )

    def summary(self) -> JsonObject:
        """Build a compact, self-contained summary from accepted events."""

        status = "running"
        if self._events:
            final_event_type = self._events[-1].get("event")
            if final_event_type == "campaign_completed":
                status = "complete"
            elif final_event_type == "campaign_failed":
                status = "failed"

        semantic_events = [
            event
            for event in self._events
            if event.get("event") == "semantic_result"
            and event.get("accepted") is True
        ]
        performance_events = [
            event
            for event in self._events
            if event.get("event") == "performance_result"
            and event.get("accepted") is True
        ]
        semantic_indices: set[int] = set()
        for event in semantic_events:
            run_index = event.get("run_index")
            if isinstance(run_index, int):
                semantic_indices.add(run_index)
        performance_indices: set[int] = set()
        for event in performance_events:
            run_index = event.get("run_index")
            if isinstance(run_index, int):
                performance_indices.add(run_index)

        performance_measurements: list[JsonObject] = []
        prompt_rates: list[float] = []
        decode_rates: list[float] = []
        first_token_latencies: list[float] = []
        prompt_token_total = 0
        prompt_millisecond_total = 0.0
        decode_token_total = 0
        decode_millisecond_total = 0.0
        for event in performance_events:
            response = as_optional_json_object(event.get("response"))
            timings = as_optional_json_object(response.get("timings"))
            prompt_rate = timings.get("prompt_per_second")
            decode_rate = timings.get("predicted_per_second")
            first_token_latency = response.get("time_to_first_token_seconds")
            if isinstance(prompt_rate, int | float) and not isinstance(
                prompt_rate, bool
            ):
                prompt_rates.append(float(prompt_rate))
            if isinstance(decode_rate, int | float) and not isinstance(
                decode_rate, bool
            ):
                decode_rates.append(float(decode_rate))
            if isinstance(first_token_latency, int | float) and not isinstance(
                first_token_latency, bool
            ):
                first_token_latencies.append(float(first_token_latency))
            prompt_token_count = timings.get("prompt_n")
            prompt_milliseconds = timings.get("prompt_ms")
            if (
                isinstance(prompt_token_count, int)
                and not isinstance(prompt_token_count, bool)
                and prompt_token_count > 0
                and isinstance(prompt_milliseconds, int | float)
                and not isinstance(prompt_milliseconds, bool)
                and float(prompt_milliseconds) > 0.0
            ):
                prompt_token_total += prompt_token_count
                prompt_millisecond_total += float(prompt_milliseconds)
            decode_token_count = timings.get("predicted_n")
            decode_milliseconds = timings.get("predicted_ms")
            if (
                isinstance(decode_token_count, int)
                and not isinstance(decode_token_count, bool)
                and decode_token_count > 0
                and isinstance(decode_milliseconds, int | float)
                and not isinstance(decode_milliseconds, bool)
                and float(decode_milliseconds) > 0.0
            ):
                decode_token_total += decode_token_count
                decode_millisecond_total += float(decode_milliseconds)
            performance_measurements.append(
                {
                    "run_index": event.get("run_index"),
                    "case": event.get("case"),
                    "prompt_tokens": timings.get("prompt_n"),
                    "output_tokens": timings.get("predicted_n"),
                    "prompt_tokens_per_second": timings.get(
                        "prompt_per_second"
                    ),
                    "decode_tokens_per_second": timings.get(
                        "predicted_per_second"
                    ),
                    "time_to_first_token_seconds": first_token_latency,
                    "time_to_first_content_seconds": response.get(
                        "time_to_first_content_seconds"
                    ),
                    "elapsed_seconds": response.get("elapsed_seconds"),
                }
            )

        aggregate: JsonObject = {
            "prompt_tokens_per_second_mean": (
                statistics.fmean(prompt_rates) if prompt_rates else None
            ),
            "prompt_tokens_per_second_median": (
                statistics.median(prompt_rates) if prompt_rates else None
            ),
            "prompt_tokens_per_second_weighted": (
                1000.0 * prompt_token_total / prompt_millisecond_total
                if prompt_millisecond_total > 0.0
                else None
            ),
            "decode_tokens_per_second_mean": (
                statistics.fmean(decode_rates) if decode_rates else None
            ),
            "decode_tokens_per_second_median": (
                statistics.median(decode_rates) if decode_rates else None
            ),
            "decode_tokens_per_second_weighted": (
                1000.0 * decode_token_total / decode_millisecond_total
                if decode_millisecond_total > 0.0
                else None
            ),
            "time_to_first_token_seconds_mean": (
                statistics.fmean(first_token_latencies)
                if first_token_latencies
                else None
            ),
            "time_to_first_token_seconds_median": (
                statistics.median(first_token_latencies)
                if first_token_latencies
                else None
            ),
            "prompt_tokens_per_second_distribution": summarize_measurements(
                prompt_rates
            ),
            "decode_tokens_per_second_distribution": summarize_measurements(
                decode_rates
            ),
            "time_to_first_token_seconds_distribution": summarize_measurements(
                first_token_latencies
            ),
            "weighted_totals": {
                "prompt_tokens": prompt_token_total,
                "prompt_milliseconds": prompt_millisecond_total,
                "decode_tokens": decode_token_total,
                "decode_milliseconds": decode_millisecond_total,
            },
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "updated_at_utc": utc_now(),
            "configuration": self._configuration,
            "receipt_paths": {
                "jsonl": str(self._output_jsonl),
                "summary_json": str(self._summary_json),
            },
            "event_count": len(self._events),
            "accepted_semantic_runs": len(semantic_events),
            "accepted_performance_runs": len(performance_events),
            "accepted_pairs": len(semantic_indices & performance_indices),
            "performance_measurements": performance_measurements,
            "aggregate": aggregate,
        }


@final
class LlamaServerClient:
    """Minimal HTTP/S client for the llama-server endpoints used here."""

    def __init__(
        self,
        address: ServerAddress,
        timeout_seconds: float,
        api_key: str | None,
    ) -> None:
        self._address = address
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key

    def _connection(self) -> http.client.HTTPConnection:
        if self._address.scheme == "https":
            return http.client.HTTPSConnection(
                self._address.hostname,
                self._address.port,
                timeout=self._timeout_seconds,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(
            self._address.hostname,
            self._address.port,
            timeout=self._timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "exo-kimi-k3-benchmark/1",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def request_json(
        self,
        method: str,
        endpoint: str,
        payload: JsonObject | None = None,
    ) -> JsonValue:
        """Issue a bounded JSON request and decode a successful response."""

        encoded_payload = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        connection = self._connection()
        endpoint_path = self._address.endpoint_path(endpoint)
        try:
            connection.request(
                method,
                endpoint_path,
                body=encoded_payload,
                headers=self._headers(),
            )
            response = connection.getresponse()
            response_body = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            raise HttpRequestError(
                f"{method} {endpoint_path} failed: {error}"
            ) from error
        finally:
            connection.close()

        decoded_body = response_body.decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            bounded_body = decoded_body[:MAXIMUM_HTTP_ERROR_CHARACTERS]
            raise HttpRequestError(
                f"{method} {endpoint_path} returned HTTP {response.status}: "
                f"{bounded_body}",
                status=response.status,
            )
        try:
            return cast(JsonValue, json.loads(decoded_body))
        except json.JSONDecodeError as error:
            raise HttpRequestError(
                f"{method} {endpoint_path} returned invalid JSON: {error}"
            ) from error

    def request_object(
        self,
        method: str,
        endpoint: str,
        payload: JsonObject | None = None,
    ) -> JsonObject:
        return as_json_object(
            self.request_json(method, endpoint, payload),
            f"{method} {endpoint}",
        )

    def erase_slot(self) -> JsonObject:
        """Erase slot zero so no prior recurrent/KV state is reused."""

        return self.request_object(
            "POST",
            f"/slots/{SLOT_IDENTIFIER}?action=erase",
            {},
        )

    def stream_chat_completion(self, payload: JsonObject) -> StreamResult:
        """Stream one chat request and measure Kimi-aware client latencies."""

        encoded_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        connection = self._connection()
        endpoint_path = self._address.endpoint_path("/v1/chat/completions")
        started_at_utc = utc_now()
        started_at_monotonic = time.monotonic()
        try:
            try:
                connection.request(
                    "POST",
                    endpoint_path,
                    body=encoded_payload,
                    headers={
                        **self._headers(),
                        "Accept": "text/event-stream",
                    },
                )
                response = connection.getresponse()
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                raise HttpRequestError(
                    f"POST {endpoint_path} failed before streaming: {error}"
                ) from error

            if response.status < 200 or response.status >= 300:
                response_body = response.read().decode("utf-8", errors="replace")
                raise HttpRequestError(
                    f"POST {endpoint_path} returned HTTP {response.status}: "
                    f"{response_body[:MAXIMUM_HTTP_ERROR_CHARACTERS]}",
                    status=response.status,
                )

            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            first_token_seconds: float | None = None
            first_content_seconds: float | None = None
            finish_reason: str | None = None
            response_identifier: str | None = None
            timings: JsonObject = {}
            usage: JsonObject = {}
            chunk_count = 0
            saw_done_marker = False

            try:
                stream_items = iter_server_sent_event_data(response)
                for event_data in stream_items:
                    if event_data == "[DONE]":
                        saw_done_marker = True
                        break
                    try:
                        chunk_value = cast(object, json.loads(event_data))
                    except json.JSONDecodeError as error:
                        raise HttpRequestError(
                            "llama-server emitted invalid SSE JSON: "
                            f"{error}; data={event_data[:500]!r}"
                        ) from error
                    chunk = as_json_object(chunk_value, "SSE chunk")
                    chunk_count += 1
                    received_at = time.monotonic()

                    chunk_error = chunk.get("error")
                    if chunk_error is not None:
                        bounded_error = json.dumps(
                            chunk_error,
                            sort_keys=True,
                            separators=(",", ":"),
                        )[:MAXIMUM_HTTP_ERROR_CHARACTERS]
                        raise HttpRequestError(
                            f"llama-server emitted an SSE error: {bounded_error}"
                        )

                    chunk_identifier = chunk.get("id")
                    if isinstance(chunk_identifier, str):
                        response_identifier = chunk_identifier

                    chunk_timings = chunk.get("timings")
                    if isinstance(chunk_timings, dict):
                        timings = as_optional_json_object(
                            cast(dict[object, object], chunk_timings)
                        )
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = as_optional_json_object(
                            cast(dict[object, object], chunk_usage)
                        )

                    choices = chunk.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for choice in cast(list[object], choices):
                        if not isinstance(choice, dict):
                            continue
                        choice_mapping = cast(dict[str, object], choice)
                        choice_finish_reason = choice_mapping.get("finish_reason")
                        if isinstance(choice_finish_reason, str):
                            finish_reason = choice_finish_reason

                        delta = choice_mapping.get("delta")
                        if not isinstance(delta, dict):
                            continue
                        delta_mapping = cast(dict[str, object], delta)
                        reasoning_text = extract_delta_text(
                            delta_mapping.get("reasoning_content")
                        )
                        if not reasoning_text:
                            reasoning_text = extract_delta_text(
                                delta_mapping.get("reasoning")
                            )
                        content_text = extract_delta_text(
                            delta_mapping.get("content")
                        )

                        if (
                            reasoning_text or content_text
                        ) and first_token_seconds is None:
                            first_token_seconds = (
                                received_at - started_at_monotonic
                            )
                        if reasoning_text:
                            reasoning_parts.append(reasoning_text)
                        if content_text:
                            if first_content_seconds is None:
                                first_content_seconds = (
                                    received_at - started_at_monotonic
                                )
                            content_parts.append(content_text)
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                raise HttpRequestError(
                    f"POST {endpoint_path} stream failed: {error}"
                ) from error
        finally:
            connection.close()

        elapsed_seconds = time.monotonic() - started_at_monotonic
        if first_token_seconds is None:
            raise HttpRequestError(
                "stream ended without non-empty reasoning_content, reasoning, or "
                "content; Kimi-aware TTFT cannot be measured"
            )
        if not timings:
            raise HttpRequestError(
                "stream ended without llama-server's final timings object"
            )
        if not saw_done_marker and finish_reason is None:
            raise HttpRequestError(
                "stream closed without [DONE] or a finish_reason; response may be "
                "truncated"
            )

        return StreamResult(
            started_at_utc=started_at_utc,
            elapsed_seconds=elapsed_seconds,
            time_to_first_token_seconds=first_token_seconds,
            time_to_first_content_seconds=first_content_seconds,
            reasoning_content="".join(reasoning_parts),
            content="".join(content_parts),
            finish_reason=finish_reason,
            response_identifier=response_identifier,
            stream_chunk_count=chunk_count,
            saw_done_marker=saw_done_marker,
            timings=timings,
            usage=usage,
        )


def iter_server_sent_event_data(
    response: http.client.HTTPResponse,
) -> Iterator[str]:
    """Yield assembled ``data:`` fields from an HTTP SSE response."""

    data_lines: list[str] = []
    while True:
        raw_line = response.readline()
        if not raw_line:
            if data_lines:
                yield "\n".join(data_lines)
            return
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_value = line[5:]
        if data_value.startswith(" "):
            data_value = data_value[1:]
        data_lines.append(data_value)


def extract_delta_text(value: object) -> str:
    """Extract text from a normal string or a permissive reasoning object."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value_mapping = cast(dict[object, object], value)
        for key in ("content", "text"):
            nested_value = value_mapping.get(key)
            if isinstance(nested_value, str):
                return nested_value
    return ""


@final
class LiveTokenCounter:
    """Count chat-template input tokens using the live model tokenizer."""

    def __init__(
        self,
        client: LlamaServerClient,
        model: str,
        reasoning_effort: str,
        thinking_effort: str,
        reasoning_budget_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._thinking_effort = thinking_effort
        self._reasoning_budget_tokens = reasoning_budget_tokens
        self._mode: str | None = None
        self.request_count = 0

    @property
    def endpoint_description(self) -> str:
        if self._mode == "chat_input_tokens":
            return "/v1/chat/completions/input_tokens"
        if self._mode == "apply_template_tokenize":
            return "/apply-template + /tokenize"
        return "not-yet-selected"

    def count(self, content: str) -> int:
        self.request_count += 1
        if self.request_count > MAXIMUM_CALIBRATION_REQUESTS:
            raise CalibrationError(
                f"calibration exceeded {MAXIMUM_CALIBRATION_REQUESTS} token-count "
                "requests"
            )

        if self._mode in {None, "chat_input_tokens"}:
            try:
                response = self._client.request_object(
                    "POST",
                    "/v1/chat/completions/input_tokens",
                    self._chat_body(content),
                )
                self._mode = "chat_input_tokens"
                return read_input_token_count(
                    response,
                    "/v1/chat/completions/input_tokens",
                )
            except HttpRequestError as error:
                if self._mode == "chat_input_tokens" or error.status not in {
                    404,
                    405,
                    501,
                }:
                    raise
                self._mode = "apply_template_tokenize"

        template_response = self._client.request_object(
            "POST",
            "/apply-template",
            self._chat_body(content),
        )
        rendered_prompt = template_response.get("prompt")
        if not isinstance(rendered_prompt, str):
            raise CalibrationError(
                "/apply-template response did not contain a string prompt"
            )
        tokenize_response = self._client.request_object(
            "POST",
            "/tokenize",
            {
                "content": rendered_prompt,
                "add_special": False,
                "parse_special": True,
            },
        )
        tokens = tokenize_response.get("tokens")
        if not isinstance(tokens, list):
            raise CalibrationError(
                "/tokenize response did not contain a tokens array"
            )
        return len(cast(list[object], tokens))

    def rendered_prompt_receipt(
        self,
        content: str,
        *,
        expected_token_count: int | None = None,
    ) -> JsonObject:
        """Render and tokenize a prompt, returning only its hash and dimensions."""

        template_response = self._client.request_object(
            "POST",
            "/apply-template",
            self._chat_body(content),
        )
        rendered_prompt = template_response.get("prompt")
        if not isinstance(rendered_prompt, str):
            raise CalibrationError(
                "/apply-template response did not contain a string prompt"
            )
        tokenize_response = self._client.request_object(
            "POST",
            "/tokenize",
            {
                "content": rendered_prompt,
                "add_special": False,
                "parse_special": True,
            },
        )
        tokens = tokenize_response.get("tokens")
        if not isinstance(tokens, list):
            raise CalibrationError(
                "/tokenize response did not contain a tokens array"
            )
        token_count = len(cast(list[object], tokens))
        return {
            "rendered_prompt_sha256": sha256_text(rendered_prompt),
            "rendered_prompt_character_count": len(rendered_prompt),
            "rendered_prompt_token_count": token_count,
            "expected_token_count": expected_token_count,
            "matches_expected_token_count": (
                token_count == expected_token_count
                if expected_token_count is not None
                else None
            ),
        }

    def _chat_body(self, content: str) -> JsonObject:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "reasoning_effort": self._reasoning_effort,
            "reasoning_budget_tokens": self._reasoning_budget_tokens,
            "chat_template_kwargs": {
                "thinking_effort": self._thinking_effort
            },
        }


def read_input_token_count(response: Mapping[str, object], endpoint: str) -> int:
    """Validate an input-token-count endpoint response."""

    value = response.get("input_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(
            f"{endpoint} returned invalid input_tokens value {value!r}"
        )
    if value <= 0:
        raise CalibrationError(
            f"{endpoint} returned non-positive input_tokens value {value}"
        )
    return value


def calibration_content(filler_repetitions: int, repair_suffix: str = "") -> str:
    """Build an inert prompt whose filler count can be searched."""

    return (
        CALIBRATION_HEADER
        + (" x" * filler_repetitions)
        + repair_suffix
        + CALIBRATION_TAIL
    )


def calibrate_performance_prompt(
    counter: LiveTokenCounter,
    target_tokens: int,
) -> CalibratedPrompt:
    """Find a prompt whose live chat-template count exactly matches the target."""

    count_cache: dict[tuple[int, str], int] = {}

    def count_candidate(repetitions: int, repair_suffix: str = "") -> int:
        cache_key = (repetitions, repair_suffix)
        if cache_key not in count_cache:
            count_cache[cache_key] = counter.count(
                calibration_content(repetitions, repair_suffix)
            )
        return count_cache[cache_key]

    base_count = count_candidate(0)
    if base_count > target_tokens:
        raise CalibrationError(
            f"base performance prompt is already {base_count} tokens, above "
            f"target {target_tokens}"
        )
    if base_count == target_tokens:
        return calibrated_prompt_result(counter, 0, "", target_tokens)

    lower_repetitions = 0
    upper_repetitions = 1
    maximum_repetitions = target_tokens * 8
    while (
        count_candidate(upper_repetitions) < target_tokens
        and upper_repetitions < maximum_repetitions
    ):
        lower_repetitions = upper_repetitions
        upper_repetitions = min(
            upper_repetitions * 2,
            maximum_repetitions,
        )
    if count_candidate(upper_repetitions) < target_tokens:
        raise CalibrationError(
            "adding inert filler did not reach the target token count before "
            f"{maximum_repetitions} repetitions"
        )

    first_at_or_above = upper_repetitions
    last_below = lower_repetitions
    while last_below + 1 < first_at_or_above:
        midpoint = (last_below + first_at_or_above) // 2
        midpoint_count = count_candidate(midpoint)
        if midpoint_count < target_tokens:
            last_below = midpoint
        else:
            first_at_or_above = midpoint

    search_start = max(0, last_below - 64)
    search_end = min(maximum_repetitions, first_at_or_above + 64)
    below_target_candidates: list[tuple[int, int]] = []
    for repetitions in range(search_start, search_end + 1):
        candidate_count = count_candidate(repetitions)
        if candidate_count == target_tokens:
            return calibrated_prompt_result(
                counter,
                repetitions,
                "",
                target_tokens,
            )
        if candidate_count < target_tokens:
            below_target_candidates.append((candidate_count, repetitions))

    below_target_candidates.sort(reverse=True)
    repair_atoms = (" a", " x", " 0", ".", "a", "\n")
    for _, repetitions in below_target_candidates[:8]:
        for repair_atom in repair_atoms:
            for repair_count in range(1, 257):
                repair_suffix = repair_atom * repair_count
                candidate_count = count_candidate(repetitions, repair_suffix)
                if candidate_count == target_tokens:
                    return calibrated_prompt_result(
                        counter,
                        repetitions,
                        repair_suffix,
                        target_tokens,
                    )
                if candidate_count > target_tokens + 8:
                    break

    nearest_candidates = sorted(
        (
            (abs(token_count - target_tokens), repetitions, token_count)
            for (repetitions, repair_suffix), token_count in count_cache.items()
            if not repair_suffix
        )
    )
    nearest_description = ", ".join(
        f"repetitions={repetitions}:tokens={token_count}"
        for _, repetitions, token_count in nearest_candidates[:5]
    )
    raise CalibrationError(
        f"could not construct an exact {target_tokens}-token prompt; nearest "
        f"plain-filler candidates were {nearest_description}"
    )


def calibrated_prompt_result(
    counter: LiveTokenCounter,
    filler_repetitions: int,
    repair_suffix: str,
    target_tokens: int,
) -> CalibratedPrompt:
    """Construct a calibrated result after an exact count is observed."""

    return CalibratedPrompt(
        content=calibration_content(filler_repetitions, repair_suffix),
        token_count=target_tokens,
        filler_repetitions=filler_repetitions,
        repair_suffix=repair_suffix,
        counting_endpoint=counter.endpoint_description,
        counting_requests=counter.request_count,
    )


def common_chat_payload(
    configuration: BenchmarkConfiguration,
    *,
    content: str,
    seed: int,
    maximum_output_tokens: int,
) -> JsonObject:
    """Build fields shared by semantic and performance requests."""

    return {
        "model": configuration.model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": maximum_output_tokens,
        "seed": seed,
        "cache_prompt": False,
        "id_slot": SLOT_IDENTIFIER,
        "reasoning_effort": configuration.reasoning_effort,
        "reasoning_budget_tokens": configuration.reasoning_budget_tokens,
        "chat_template_kwargs": {
            "thinking_effort": configuration.thinking_effort
        },
    }


def semantic_payload(
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
) -> JsonObject:
    """Build a deterministic semantic-gate request."""

    return {
        **common_chat_payload(
            configuration,
            content=benchmark_case.prompt,
            seed=benchmark_case.seed,
            maximum_output_tokens=SEMANTIC_OUTPUT_TOKENS,
        ),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }


def performance_payload(
    configuration: BenchmarkConfiguration,
    calibrated_prompt: CalibratedPrompt,
    seed: int,
    *,
    output_tokens: int = PERFORMANCE_OUTPUT_TOKENS,
) -> JsonObject:
    """Build a fixed-token throughput request with EOS ignored."""

    return {
        **common_chat_payload(
            configuration,
            content=calibrated_prompt.content,
            seed=seed,
            maximum_output_tokens=output_tokens,
        ),
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "ignore_eos": True,
    }


def validate_cache_is_cold(result: StreamResult) -> list[str]:
    """Return acceptance errors for any observed prompt-cache reuse."""

    errors: list[str] = []
    try:
        cache_tokens = require_integer(result.timings, "cache_n")
        if cache_tokens != 0:
            errors.append(
                f"timings.cache_n is {cache_tokens}, expected exactly 0"
            )
    except AcceptanceError as error:
        errors.append(str(error))

    prompt_token_details = result.usage.get("prompt_tokens_details")
    if isinstance(prompt_token_details, dict):
        prompt_token_details_mapping = cast(
            dict[object, object],
            prompt_token_details,
        )
        cached_tokens = prompt_token_details_mapping.get("cached_tokens")
        if (
            cached_tokens is not None
            and (
                isinstance(cached_tokens, bool)
                or not isinstance(cached_tokens, int)
                or cached_tokens != 0
            )
        ):
            errors.append(
                "usage.prompt_tokens_details.cached_tokens is "
                f"{cached_tokens!r}, expected 0"
            )
    return errors


def validate_timing_rates(result: StreamResult) -> list[str]:
    """Return acceptance errors for missing/non-positive server timings."""

    errors: list[str] = []
    for timing_key in (
        "prompt_ms",
        "prompt_per_second",
        "predicted_ms",
        "predicted_per_second",
    ):
        try:
            require_positive_number(result.timings, timing_key)
        except AcceptanceError as error:
            errors.append(str(error))
    for count_key in ("prompt_n", "predicted_n"):
        try:
            count = require_integer(result.timings, count_key)
            if count <= 0:
                errors.append(f"timings.{count_key} is {count}, expected > 0")
        except AcceptanceError as error:
            errors.append(str(error))
    if not math.isfinite(result.time_to_first_token_seconds):
        errors.append("client TTFT is not finite")
    elif result.time_to_first_token_seconds < 0.0:
        errors.append("client TTFT is negative")
    if not math.isfinite(result.elapsed_seconds) or result.elapsed_seconds <= 0.0:
        errors.append("client end-to-end latency is not finite and positive")
    return errors


def validate_semantic_response(
    result: StreamResult,
    benchmark_case: BenchmarkCase,
) -> tuple[GateVerdict, list[str]]:
    """Apply cache, timing, and case-specific semantic acceptance checks."""

    errors = validate_cache_is_cold(result)
    errors.extend(validate_timing_rates(result))
    gate_verdict = benchmark_case.validator(result.content)
    if not gate_verdict.accepted:
        errors.append(f"semantic gate failed: {gate_verdict.explanation}")
    return gate_verdict, errors


def validate_performance_response(
    result: StreamResult,
    *,
    expected_prompt_tokens: int,
    expected_output_tokens: int,
) -> list[str]:
    """Apply exact-token and valid-timing checks to a throughput response."""

    errors = validate_cache_is_cold(result)
    errors.extend(validate_timing_rates(result))
    try:
        prompt_tokens = require_integer(result.timings, "prompt_n")
        if prompt_tokens != expected_prompt_tokens:
            errors.append(
                f"timings.prompt_n is {prompt_tokens}, expected exactly "
                f"{expected_prompt_tokens}"
            )
    except AcceptanceError:
        pass
    try:
        output_tokens = require_integer(result.timings, "predicted_n")
        if output_tokens != expected_output_tokens:
            errors.append(
                f"timings.predicted_n is {output_tokens}, expected exactly "
                f"{expected_output_tokens}"
            )
    except AcceptanceError:
        pass

    usage_prompt_tokens = result.usage.get("prompt_tokens")
    if (
        usage_prompt_tokens is not None
        and usage_prompt_tokens != expected_prompt_tokens
    ):
        errors.append(
            f"usage.prompt_tokens is {usage_prompt_tokens!r}, expected "
            f"{expected_prompt_tokens}"
        )
    usage_completion_tokens = result.usage.get("completion_tokens")
    if (
        usage_completion_tokens is not None
        and usage_completion_tokens != expected_output_tokens
    ):
        errors.append(
            f"usage.completion_tokens is {usage_completion_tokens!r}, expected "
            f"{expected_output_tokens}"
        )
    return errors


def final_nonempty_line(content: str) -> str:
    """Return the final non-empty output line, or an empty string."""

    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else ""


def validate_arithmetic(content: str) -> GateVerdict:
    """Validate the exact arithmetic marker for semantic case one."""

    observed_line = final_nonempty_line(content)
    match = re.fullmatch(r"FINAL\s*=\s*(-?\d+)", observed_line)
    if match is None:
        return GateVerdict(
            False,
            "final non-empty line must be FINAL=<integer>",
            observed_line,
        )
    observed_value = int(match.group(1))
    return GateVerdict(
        observed_value == 1080,
        "arithmetic result matches 1080"
        if observed_value == 1080
        else f"arithmetic result is {observed_value}, expected 1080",
        observed_value,
    )


def validate_geography(content: str) -> GateVerdict:
    """Validate the Paris/Seine marker for semantic case two."""

    observed_line = final_nonempty_line(content)
    match = re.fullmatch(
        r"FINAL\s*=\s*([^|]+?)\s*\|\s*([^|]+?)",
        observed_line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return GateVerdict(
            False,
            "final non-empty line must be FINAL=<capital>|<river>",
            observed_line,
        )
    capital = match.group(1).strip()
    river = match.group(2).strip()
    accepted = capital.casefold() == "paris" and river.casefold() == "seine"
    return GateVerdict(
        accepted,
        "capital and river match Paris and Seine"
        if accepted
        else f"observed capital={capital!r}, river={river!r}",
        {"capital": capital, "river": river},
    )


def strip_single_code_fence(content: str, language: str = "") -> str:
    """Remove one surrounding Markdown fence when the whole response uses it."""

    stripped = content.strip()
    language_pattern = re.escape(language) if language else r"[A-Za-z0-9_+-]*"
    match = re.fullmatch(
        rf"```{language_pattern}\s*\n?(.*?)\n?```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match is not None else stripped


def validate_json_transformation(content: str) -> GateVerdict:
    """Validate the exact JSON transformation for semantic case three."""

    candidate = strip_single_code_fence(content, "json")
    try:
        decoded = cast(object, json.loads(candidate))
    except json.JSONDecodeError as error:
        return GateVerdict(False, f"response is not valid JSON: {error}", candidate)
    expected = {
        "sorted": [1, 3, 5, 8],
        "sum": 17,
        "sum_is_even": False,
    }
    accepted = decoded == expected
    return GateVerdict(
        accepted,
        "JSON structure and values match exactly"
        if accepted
        else f"decoded JSON does not match {expected!r}",
        decoded,
    )


def literal_list_from_call(node: ast.AST) -> list[object] | None:
    """Read the sole literal-list argument from a target function call."""

    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "dedupe_keep_order" or len(node.args) != 1:
        return None
    try:
        value = cast(object, ast.literal_eval(node.args[0]))
    except (ValueError, TypeError, SyntaxError):
        return None
    return cast(list[object], value) if isinstance(value, list) else None


def expected_assertion_present(
    tree: ast.AST,
    input_values: list[object],
    expected_values: list[object],
) -> bool:
    """Check for one exact ``function(input) == expected`` assertion."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        comparison = node.test
        if not isinstance(comparison, ast.Compare):
            continue
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
            continue
        if len(comparison.comparators) != 1:
            continue
        observed_input = literal_list_from_call(comparison.left)
        try:
            observed_expected = cast(
                object,
                ast.literal_eval(comparison.comparators[0]),
            )
        except (ValueError, TypeError, SyntaxError):
            continue
        if observed_input == input_values and observed_expected == expected_values:
            return True
    return False


def validate_python_code(content: str) -> GateVerdict:
    """Statically validate the requested function and two exact assertions."""

    if final_nonempty_line(content) != "FINAL=CODE":
        return GateVerdict(
            False,
            "final non-empty line must be exactly FINAL=CODE",
            final_nonempty_line(content),
        )
    marker_position = content.rfind("FINAL=CODE")
    code_portion = content[:marker_position].strip()
    code_portion = strip_single_code_fence(code_portion, "python")
    try:
        tree = ast.parse(code_portion)
    except SyntaxError as error:
        return GateVerdict(False, f"Python code does not parse: {error}", None)

    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "dedupe_keep_order"
    ]
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    errors: list[str] = []
    if imports:
        errors.append("imports are forbidden")
    if len(functions) != 1:
        errors.append(
            "exactly one top-level dedupe_keep_order function is required"
        )
    elif len(functions[0].args.args) != 1:
        errors.append("dedupe_keep_order must have exactly one positional argument")
    if len(assertions) < 2:
        errors.append("at least two assertions are required")
    if not expected_assertion_present(tree, [3, 1, 3, 2], [3, 1, 2]):
        errors.append("the required non-empty assertion is missing")
    if not expected_assertion_present(tree, [], []):
        errors.append("the required empty-list assertion is missing")
    if functions and not any(
        isinstance(node, ast.Return) for node in ast.walk(functions[0])
    ):
        errors.append("dedupe_keep_order must contain a return statement")

    return GateVerdict(
        not errors,
        "Python function and assertions passed static validation"
        if not errors
        else "; ".join(errors),
        {
            "function_count": len(functions),
            "assertion_count": len(assertions),
            "import_count": len(imports),
            "code_sha256": sha256_text(code_portion),
        },
    )


def validate_ordering(content: str) -> GateVerdict:
    """Validate the unique ordering marker for semantic case five."""

    observed_line = final_nonempty_line(content)
    match = re.fullmatch(r"FINAL\s*=\s*(.+)", observed_line)
    if match is None:
        return GateVerdict(
            False,
            "final non-empty line must be FINAL=<name>><name>...",
            observed_line,
        )
    observed_order = [
        item.strip() for item in match.group(1).split(">") if item.strip()
    ]
    expected_order = ["Mira", "Noah", "Lin", "Omar", "Sora"]
    accepted = observed_order == expected_order
    return GateVerdict(
        accepted,
        "ordering constraints resolve to the expected unique order"
        if accepted
        else f"observed order {observed_order!r}, expected {expected_order!r}",
        observed_order,
    )


BENCHMARK_CASES: Final[tuple[BenchmarkCase, ...]] = (
    BenchmarkCase(
        name="arithmetic",
        seed=314_159,
        prompt=(
            "Compute (37 × 29) + (144 ÷ 12) - 5. Give a concise explanation. "
            "Your final non-empty line must be FINAL=<integer>."
        ),
        gate_description="Final marker must equal FINAL=1080.",
        validator=validate_arithmetic,
    ),
    BenchmarkCase(
        name="geography",
        seed=271_828,
        prompt=(
            "In one sentence, state the capital of France and the river that "
            "runs through it. Your final non-empty line must be "
            "FINAL=<capital>|<river>."
        ),
        gate_description="Final marker must identify Paris and the Seine.",
        validator=validate_geography,
    ),
    BenchmarkCase(
        name="json_transformation",
        seed=161_803,
        prompt=(
            "Deduplicate [8, 3, 5, 3, 1], sort ascending, and compute the sum. "
            "Return only one JSON object with exactly the keys sorted, sum, and "
            "sum_is_even. Do not use a Markdown fence."
        ),
        gate_description=(
            'Exact JSON value: {"sorted":[1,3,5,8],"sum":17,'
            '"sum_is_even":false}.'
        ),
        validator=validate_json_transformation,
    ),
    BenchmarkCase(
        name="python_code",
        seed=141_421,
        prompt=(
            "Write a Python function dedupe_keep_order(items) that removes "
            "duplicates while preserving first occurrence order. Use no imports. "
            "Include these two assertions exactly in meaning: "
            "assert dedupe_keep_order([3, 1, 3, 2]) == [3, 1, 2] and "
            "assert dedupe_keep_order([]) == []. Return plain text and do not use "
            "Markdown fences. End with a final non-empty line FINAL=CODE."
        ),
        gate_description=(
            "Parseable no-import Python, target function, two exact assertions, "
            "and FINAL=CODE."
        ),
        validator=validate_python_code,
    ),
    BenchmarkCase(
        name="ordering",
        seed=173_205,
        prompt=(
            "Order Mira, Noah, Lin, Omar, and Sora using all constraints: Mira "
            "is before Noah; Lin is after Noah but before Omar; Sora is after "
            "Omar. Briefly state the order, then make the final non-empty line "
            "FINAL=<names joined by > symbols>."
        ),
        gate_description="Final marker must equal Mira>Noah>Lin>Omar>Sora.",
        validator=validate_ordering,
    ),
)


def execute_stream_request(
    client: LlamaServerClient,
    receipt_writer: ReceiptWriter,
    *,
    event_prefix: str,
    run_index: int | None,
    case_name: str,
    payload: JsonObject,
    rendered_prompt: JsonObject,
) -> tuple[JsonObject, StreamResult]:
    """Erase the slot, stream a request, and durably record transport failures."""

    print(
        f"[kimi-k3] erasing slot {SLOT_IDENTIFIER} before {event_prefix} "
        f"{case_name}",
        file=sys.stderr,
        flush=True,
    )
    slot_erase_response = client.erase_slot()
    print(
        f"[kimi-k3] starting {event_prefix} {case_name}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = client.stream_chat_completion(payload)
    except BenchmarkError as error:
        receipt_writer.append(
            {
                "event": f"{event_prefix}_request_failed",
                "run_index": run_index,
                "case": case_name,
                "request": payload,
                "rendered_prompt": rendered_prompt,
                "request_sha256": sha256_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                ),
                "slot_erase_response": slot_erase_response,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    return slot_erase_response, result


def run_benchmark(configuration: BenchmarkConfiguration) -> JsonObject:
    """Execute one complete five-pair Kimi K3 benchmark campaign."""

    public_configuration = configuration.public_json()
    receipt_writer = ReceiptWriter(
        configuration.output_jsonl,
        configuration.summary_json,
        overwrite=configuration.overwrite,
        configuration=public_configuration,
    )
    receipt_writer.append(
        {
            "event": "campaign_started",
            "configuration": public_configuration,
            "semantic_cases": [
                benchmark_case.public_plan(run_index)
                for run_index, benchmark_case in enumerate(
                    BENCHMARK_CASES,
                    start=1,
                )
            ],
        }
    )

    client = LlamaServerClient(
        configuration.server_address,
        configuration.timeout_seconds,
        configuration.api_key,
    )
    try:
        health = client.request_json("GET", "/health")
        models = client.request_json("GET", "/v1/models")
        receipt_writer.append(
            {
                "event": "server_ready",
                "health": health,
                "models": models,
            }
        )

        print(
            "[kimi-k3] calibrating a live chat-template prompt to exactly "
            f"{TARGET_PROMPT_TOKENS} tokens",
            file=sys.stderr,
            flush=True,
        )
        token_counter = LiveTokenCounter(
            client,
            configuration.model,
            configuration.reasoning_effort,
            configuration.thinking_effort,
            configuration.reasoning_budget_tokens,
        )
        calibrated_prompt = calibrate_performance_prompt(
            token_counter,
            TARGET_PROMPT_TOKENS,
        )
        performance_rendered_prompt = token_counter.rendered_prompt_receipt(
            calibrated_prompt.content,
            expected_token_count=TARGET_PROMPT_TOKENS,
        )
        receipt_writer.append(
            {
                "event": "prompt_calibrated",
                "calibration": calibrated_prompt.to_json(),
                "rendered_prompt": performance_rendered_prompt,
            }
        )
        print(
            f"[kimi-k3] calibrated in {calibrated_prompt.counting_requests} "
            f"counting requests via {calibrated_prompt.counting_endpoint}",
            file=sys.stderr,
            flush=True,
        )

        warmup_request = performance_payload(
            configuration,
            calibrated_prompt,
            WARMUP_SEED,
            output_tokens=WARMUP_OUTPUT_TOKENS,
        )
        warmup_slot_erase, warmup_result = execute_stream_request(
            client,
            receipt_writer,
            event_prefix="warmup",
            run_index=None,
            case_name="sacrificial",
            payload=warmup_request,
            rendered_prompt=performance_rendered_prompt,
        )
        warmup_errors = validate_performance_response(
            warmup_result,
            expected_prompt_tokens=TARGET_PROMPT_TOKENS,
            expected_output_tokens=WARMUP_OUTPUT_TOKENS,
        )
        receipt_writer.append(
            {
                "event": "warmup_result",
                "case": "sacrificial",
                "accepted": not warmup_errors,
                "acceptance_errors": warmup_errors,
                "request": warmup_request,
                "rendered_prompt": performance_rendered_prompt,
                "slot_erase_response": warmup_slot_erase,
                "response": warmup_result.to_json(),
            }
        )
        if warmup_errors:
            raise AcceptanceError(
                "warmup violated benchmark invariants: "
                + "; ".join(warmup_errors)
            )

        for run_index, benchmark_case in enumerate(BENCHMARK_CASES, start=1):
            semantic_request = semantic_payload(configuration, benchmark_case)
            semantic_rendered_prompt = token_counter.rendered_prompt_receipt(
                benchmark_case.prompt
            )
            semantic_slot_erase, semantic_result = execute_stream_request(
                client,
                receipt_writer,
                event_prefix="semantic",
                run_index=run_index,
                case_name=benchmark_case.name,
                payload=semantic_request,
                rendered_prompt=semantic_rendered_prompt,
            )
            semantic_verdict, semantic_errors = validate_semantic_response(
                semantic_result,
                benchmark_case,
            )
            receipt_writer.append(
                {
                    "event": "semantic_result",
                    "run_index": run_index,
                    "case": benchmark_case.name,
                    "accepted": not semantic_errors,
                    "acceptance_errors": semantic_errors,
                    "gate": semantic_verdict.to_json(),
                    "request": semantic_request,
                    "rendered_prompt": semantic_rendered_prompt,
                    "slot_erase_response": semantic_slot_erase,
                    "response": semantic_result.to_json(),
                }
            )
            if semantic_errors:
                raise AcceptanceError(
                    f"semantic run {run_index} ({benchmark_case.name}) failed: "
                    + "; ".join(semantic_errors)
                )

            performance_request = performance_payload(
                configuration,
                calibrated_prompt,
                benchmark_case.seed + PERFORMANCE_SEED_OFFSET,
            )
            performance_slot_erase, performance_result = execute_stream_request(
                client,
                receipt_writer,
                event_prefix="performance",
                run_index=run_index,
                case_name=benchmark_case.name,
                payload=performance_request,
                rendered_prompt=performance_rendered_prompt,
            )
            performance_errors = validate_performance_response(
                performance_result,
                expected_prompt_tokens=TARGET_PROMPT_TOKENS,
                expected_output_tokens=PERFORMANCE_OUTPUT_TOKENS,
            )
            receipt_writer.append(
                {
                    "event": "performance_result",
                    "run_index": run_index,
                    "case": benchmark_case.name,
                    "accepted": not performance_errors,
                    "acceptance_errors": performance_errors,
                    "request": performance_request,
                    "rendered_prompt": performance_rendered_prompt,
                    "slot_erase_response": performance_slot_erase,
                    "response": performance_result.to_json(),
                }
            )
            if performance_errors:
                raise AcceptanceError(
                    f"performance run {run_index} ({benchmark_case.name}) "
                    "failed: "
                    + "; ".join(performance_errors)
                )

            prompt_rate = require_positive_number(
                performance_result.timings,
                "prompt_per_second",
            )
            decode_rate = require_positive_number(
                performance_result.timings,
                "predicted_per_second",
            )
            print(
                f"[kimi-k3] accepted pair {run_index}/5 "
                f"({benchmark_case.name}): prompt={prompt_rate:.3f} tok/s, "
                f"decode={decode_rate:.3f} tok/s, "
                f"TTFT={performance_result.time_to_first_token_seconds:.3f}s",
                file=sys.stderr,
                flush=True,
            )

        receipt_writer.append(
            {
                "event": "campaign_completed",
                "accepted_pairs": len(BENCHMARK_CASES),
            }
        )
        return receipt_writer.summary()
    except Exception as error:
        with suppress(ReceiptWriteError):
            receipt_writer.append(
                {
                    "event": "campaign_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        raise


def dry_run_plan(configuration: BenchmarkConfiguration) -> JsonObject:
    """Return the complete protocol plan without touching files or the network."""

    return {
        "dry_run": True,
        "configuration": configuration.public_json(),
        "live_actions_not_performed": [
            "GET /health",
            "GET /v1/models",
            "live input-token calibration to exactly 512 tokens",
            "slot erase before every inference request",
            "one sacrificial warmup",
            "five semantic requests",
            "five paired 512-input/128-output performance requests",
            "atomic JSONL and JSON receipt writes",
        ],
        "semantic_cases": [
            benchmark_case.public_plan(run_index)
            for run_index, benchmark_case in enumerate(BENCHMARK_CASES, start=1)
        ],
        "ttft_trigger_fields_in_order": [
            "delta.reasoning_content",
            "delta.reasoning",
            "delta.content",
        ],
        "acceptance_invariants": {
            "timings.cache_n": 0,
            "performance_timings.prompt_n": TARGET_PROMPT_TOKENS,
            "performance_timings.predicted_n": PERFORMANCE_OUTPUT_TOKENS,
            "positive_server_prompt_and_decode_rates": True,
            "all_five_semantic_gates": True,
        },
    }


def positive_timeout(value: str) -> float:
    """Parse a finite, positive timeout value for argparse."""

    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "timeout must be a number of seconds"
        ) from error
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than 0")
    return timeout


def nonnegative_integer(value: str) -> int:
    """Parse a non-negative integer for argparse."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a live Kimi K3 llama-server with five semantic gates and "
            "five paired, cache-cold 512-input/128-output performance requests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Recommended server cache flags: --no-cache-prompt --cache-ram 0 "
            "--no-cache-idle-slots --cache-reuse 0 -np 1. Example: "
            "python3 scripts/benchmark_kimi_k3.py --quant UD-Q2_K_XL "
            "--output-jsonl /mnt/sanic/kimi-q2.jsonl"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="llama-server root URL; a trailing /v1 is accepted and normalized",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model name/alias sent to the chat endpoint",
    )
    parser.add_argument(
        "--quant",
        required=True,
        help="quantization label written to every campaign receipt",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP connect/read timeout; the default is six hours",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="OpenAI-style reasoning_effort request value",
    )
    parser.add_argument(
        "--thinking-effort",
        default="high",
        help="Kimi K3 chat-template thinking_effort value",
    )
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=nonnegative_integer,
        default=DEFAULT_REASONING_BUDGET_TOKENS,
        help=(
            "per-request reasoning token budget, leaving semantic output space "
            "for final content"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("kimi-k3-benchmark.jsonl"),
        help="atomic append-style event receipt (rewritten safely per event)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help=(
            "atomic compact summary receipt; defaults to OUTPUT_JSONL with "
            ".summary.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="deliberately replace existing JSONL/summary receipts",
    )
    parser.add_argument(
        "--api-key-env",
        default="LLAMA_API_KEY",
        help="environment variable containing an optional bearer API key",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen protocol without network or filesystem writes",
    )
    return parser


def configuration_from_arguments(
    arguments: argparse.Namespace,
) -> BenchmarkConfiguration:
    """Validate argparse output and build the immutable configuration."""

    output_jsonl = cast(Path, arguments.output_jsonl)
    summary_argument = cast(Path | None, arguments.summary_json)
    summary_json = (
        summary_argument
        if summary_argument is not None
        else output_jsonl.with_suffix(".summary.json")
    )
    api_key_environment_variable = cast(str, arguments.api_key_env)
    api_key = os.environ.get(api_key_environment_variable)
    if api_key == "":
        api_key = None

    model = cast(str, arguments.model).strip()
    quantization = cast(str, arguments.quant).strip()
    reasoning_effort = cast(str, arguments.reasoning_effort).strip()
    thinking_effort = cast(str, arguments.thinking_effort).strip()
    if not model:
        raise ConfigurationError("--model must not be empty")
    if not quantization:
        raise ConfigurationError("--quant must not be empty")
    if not reasoning_effort:
        raise ConfigurationError("--reasoning-effort must not be empty")
    if not thinking_effort:
        raise ConfigurationError("--thinking-effort must not be empty")
    if not api_key_environment_variable:
        raise ConfigurationError("--api-key-env must not be empty")

    return BenchmarkConfiguration(
        server_address=ServerAddress.parse(cast(str, arguments.base_url)),
        model=model,
        quantization=quantization,
        timeout_seconds=cast(float, arguments.timeout_seconds),
        reasoning_effort=reasoning_effort,
        thinking_effort=thinking_effort,
        reasoning_budget_tokens=cast(int, arguments.reasoning_budget_tokens),
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        overwrite=cast(bool, arguments.overwrite),
        api_key=api_key,
        api_key_environment_variable=api_key_environment_variable,
    )


def main(arguments: list[str] | None = None) -> int:
    """Parse arguments, execute the campaign, and report the final summary."""

    parser = build_argument_parser()
    parsed_arguments = parser.parse_args(arguments)
    try:
        configuration = configuration_from_arguments(parsed_arguments)
        if cast(bool, parsed_arguments.dry_run):
            print(json.dumps(dry_run_plan(configuration), indent=2, sort_keys=True))
            return 0

        summary = run_benchmark(configuration)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BenchmarkError as error:
        print(
            f"benchmark_kimi_k3.py: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
