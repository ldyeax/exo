#!/usr/bin/env python3
"""Bounded native SGLang client for the pinned OLMoE EP experiments."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Final, Literal, cast, final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

OLMOE_VOCABULARY_SIZE: Final = 50_304
SSE_LINE_MAXIMUM_BYTES: Final = 1024 * 1024
SSE_EVENT_SLACK: Final = 8
SSE_LINES_PER_TOKEN_LIMIT: Final = 8
SSE_READ_CHUNK_BYTES: Final = 64 * 1024
SANITY_RESPONSE_MAXIMUM_BYTES: Final = 4 * 1024 * 1024


class OlmoeServingClientError(RuntimeError):
    """Raised when native SGLang returns incomplete or ambiguous evidence."""


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    if not token_ids or any(
        token_id < 0 or token_id >= OLMOE_VOCABULARY_SIZE for token_id in token_ids
    ):
        raise ValueError("token IDs must be inside the OLMoE vocabulary")
    return hashlib.sha256(_canonical_json(list(token_ids))).hexdigest()


@dataclass(frozen=True, slots=True)
class OlmoeSamplingParameters:
    max_new_tokens: int
    temperature: float
    ignore_eos: bool
    sampling_seed: int

    def __post_init__(self) -> None:
        if (
            self.max_new_tokens <= 0
            or not math.isfinite(self.temperature)
            or self.temperature != 0.0
        ):
            raise ValueError("OLMoE requests require bounded greedy sampling")

    def json_object(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True, slots=True)
class OlmoeNativeGenerateRequest:
    input_ids: tuple[int, ...]
    sampling_params: OlmoeSamplingParameters
    stream: bool
    return_logprob: bool = False
    log_metrics: bool = False

    def __post_init__(self) -> None:
        token_ids_sha256(self.input_ids)
        if self.return_logprob:
            raise ValueError("OLMoE benchmark requests must not return logprobs")

    def json_object(self) -> JsonObject:
        return {
            "input_ids": list(self.input_ids),
            "sampling_params": self.sampling_params.json_object(),
            "stream": self.stream,
            "return_logprob": self.return_logprob,
            "log_metrics": self.log_metrics,
        }


class _PermissiveModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


@final
class _LengthFinishReason(_PermissiveModel):
    type: Literal["length"]
    length: int = Field(gt=0)


@final
class _StreamMetaInfo(_PermissiveModel):
    prompt_tokens: int = Field(gt=0)
    completion_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    finish_reason: _LengthFinishReason | None = None


@final
class _StreamEvent(_PermissiveModel):
    output_ids: tuple[int, ...]
    meta_info: _StreamMetaInfo

    @model_validator(mode="after")
    def validate_output_ids(self) -> "_StreamEvent":
        if any(
            token_id < 0 or token_id >= OLMOE_VOCABULARY_SIZE
            for token_id in self.output_ids
        ):
            raise ValueError("stream output ID is outside the OLMoE vocabulary")
        return self


@final
class _SanityFinishReason(_PermissiveModel):
    type: Literal["stop", "length"]


@final
class _SanityMetaInfo(_PermissiveModel):
    prompt_tokens: int = Field(gt=0)
    completion_tokens: int = Field(gt=0)
    cached_tokens: int = Field(ge=0)
    finish_reason: _SanityFinishReason


@final
class _SanityResponse(_PermissiveModel):
    text: str
    output_ids: tuple[int, ...]
    meta_info: _SanityMetaInfo

    @model_validator(mode="after")
    def validate_output(self) -> "_SanityResponse":
        if (
            not self.output_ids
            or self.meta_info.completion_tokens != len(self.output_ids)
            or any(
                token_id < 0 or token_id >= OLMOE_VOCABULARY_SIZE
                for token_id in self.output_ids
            )
        ):
            raise ValueError("sanity output IDs are invalid")
        return self


@dataclass(frozen=True, slots=True)
class EndpointObservation:
    status_code: int
    response_sha256: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ServerInfoObservation:
    call: EndpointObservation
    response: JsonObject
    canonical_response_sha256: str


@dataclass(frozen=True, slots=True)
class SanityResponseObservation:
    text: str
    output_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    finish_reason: JsonObject
    total_client_seconds: float


@dataclass(frozen=True, slots=True)
class GenerateObservation:
    input_ids_sha256: str
    output_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    output_ids_sha256: str
    finish_reason_sha256: str
    stream_line_count: int
    stream_event_count: int
    output_bearing_event_count: int
    maximum_stream_line_bytes: int
    first_stream_event_output_tokens: int
    total_client_seconds: float
    client_observed_ttft_seconds: float
    client_observed_generation_window_seconds: float
    client_observed_decode_tokens_per_second: float


def _strict_json_object(contents: bytes, description: str) -> JsonObject:
    try:
        value = cast(object, json.loads(contents))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OlmoeServingClientError(f"{description} is not strict JSON") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise OlmoeServingClientError(f"{description} is not a JSON object")
    return cast(JsonObject, value)


def _iter_bounded_sse_lines(
    response: httpx.Response,
    require_within_deadline: Callable[[], None],
) -> Iterator[str]:
    pending = bytearray()
    try:
        for chunk in response.iter_bytes():
            require_within_deadline()
            view = memoryview(chunk)
            for offset in range(0, len(chunk), SSE_READ_CHUNK_BYTES):
                pending.extend(view[offset : offset + SSE_READ_CHUNK_BYTES])
                while (newline := pending.find(b"\n")) >= 0:
                    line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    if line.endswith(b"\r"):
                        line = line[:-1]
                    if len(line) > SSE_LINE_MAXIMUM_BYTES:
                        raise OlmoeServingClientError("SSE line is oversized")
                    yield line.decode("utf-8", errors="strict")
                if len(pending) > SSE_LINE_MAXIMUM_BYTES + 1:
                    raise OlmoeServingClientError("SSE line is oversized")
    except UnicodeDecodeError as error:
        raise OlmoeServingClientError("SSE response is not UTF-8") from error
    if pending:
        raise OlmoeServingClientError("SSE response has an unterminated line")


@final
class OlmoeNativeServingClient:
    """Synchronous HTTP client with bounded native SGLang response parsing."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        deadline_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout must be positive and finite")
        self._clock_ns = clock_ns
        self._deadline_clock_ns = deadline_clock_ns
        self._timeout_ns = int(timeout_seconds * 1_000_000_000)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"Accept-Encoding": "identity"},
        )

    def __enter__(self) -> "OlmoeNativeServingClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _endpoint(
        self, method: Literal["GET", "POST"], path: str
    ) -> tuple[EndpointObservation, bytes]:
        started = self._clock_ns()
        response = self._client.request(method, path)
        elapsed = (self._clock_ns() - started) / 1_000_000_000
        contents = response.content
        observation = EndpointObservation(
            status_code=response.status_code,
            response_sha256=hashlib.sha256(contents).hexdigest(),
            elapsed_seconds=elapsed,
        )
        if response.status_code != 200:
            raise OlmoeServingClientError(
                f"{method} {path} returned HTTP {response.status_code}"
            )
        return observation, contents

    def health_generate(self) -> EndpointObservation:
        return self._endpoint("GET", "/health_generate")[0]

    def flush_cache(self) -> EndpointObservation:
        return self._endpoint("POST", "/flush_cache")[0]

    def server_info(self) -> ServerInfoObservation:
        call, contents = self._endpoint("GET", "/server_info")
        response = _strict_json_object(contents, "server_info response")
        return ServerInfoObservation(
            call=call,
            response=response,
            canonical_response_sha256=hashlib.sha256(
                _canonical_json(response)
            ).hexdigest(),
        )

    def generate_sanity(
        self, request: OlmoeNativeGenerateRequest
    ) -> SanityResponseObservation:
        if request.stream:
            raise ValueError("sanity request must not stream")
        started = self._clock_ns()
        response = self._client.post("/generate", json=request.json_object())
        completed = self._clock_ns()
        if response.status_code != 200:
            raise OlmoeServingClientError(
                f"POST /generate sanity returned HTTP {response.status_code}"
            )
        if len(response.content) > SANITY_RESPONSE_MAXIMUM_BYTES:
            raise OlmoeServingClientError("sanity response is oversized")
        try:
            parsed = _SanityResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise OlmoeServingClientError("sanity response is invalid") from error
        if (
            parsed.meta_info.prompt_tokens != len(request.input_ids)
            or parsed.meta_info.cached_tokens != 0
            or len(parsed.output_ids) > request.sampling_params.max_new_tokens
            or (
                parsed.meta_info.finish_reason.type == "length"
                and len(parsed.output_ids) != request.sampling_params.max_new_tokens
            )
        ):
            raise OlmoeServingClientError(
                "sanity response prompt/cache/output count is not exact"
            )
        elapsed = (completed - started) / 1_000_000_000
        if elapsed <= 0.0 or completed - started > self._timeout_ns:
            raise OlmoeServingClientError("sanity response exceeded its time bound")
        return SanityResponseObservation(
            text=parsed.text,
            output_ids=parsed.output_ids,
            prompt_tokens=parsed.meta_info.prompt_tokens,
            completion_tokens=parsed.meta_info.completion_tokens,
            cached_tokens=parsed.meta_info.cached_tokens,
            finish_reason=cast(
                JsonObject, parsed.meta_info.finish_reason.model_dump(mode="json")
            ),
            total_client_seconds=elapsed,
        )

    def generate(self, request: OlmoeNativeGenerateRequest) -> GenerateObservation:
        if not request.stream:
            raise ValueError("benchmark request must stream")
        started = self._clock_ns()
        deadline = self._deadline_clock_ns() + self._timeout_ns

        def require_within_deadline() -> None:
            if self._deadline_clock_ns() > deadline:
                raise OlmoeServingClientError("generate stream exceeded its deadline")

        maximum_events = request.sampling_params.max_new_tokens + SSE_EVENT_SLACK
        maximum_lines = (
            request.sampling_params.max_new_tokens + 2
        ) * SSE_LINES_PER_TOKEN_LIMIT
        output_ids: list[int] = []
        line_count = 0
        event_count = 0
        output_event_count = 0
        maximum_line_bytes = 0
        first_output_ns: int | None = None
        last_output_ns: int | None = None
        first_event_tokens: int | None = None
        previous_completion = 0
        finish_reason: _LengthFinishReason | None = None
        saw_done = False
        with self._client.stream(
            "POST", "/generate", json=request.json_object()
        ) as response:
            if response.status_code != 200:
                raise OlmoeServingClientError(
                    f"POST /generate returned HTTP {response.status_code}"
                )
            if response.headers.get("content-encoding", "identity") != "identity":
                raise OlmoeServingClientError("generate response is compressed")
            for line in _iter_bounded_sse_lines(response, require_within_deadline):
                received = self._clock_ns()
                line_count += 1
                maximum_line_bytes = max(maximum_line_bytes, len(line.encode()))
                if line_count > maximum_lines:
                    raise OlmoeServingClientError("generate stream has too many lines")
                if saw_done:
                    if line:
                        raise OlmoeServingClientError("data follows SSE [DONE]")
                    continue
                if line == "data: [DONE]":
                    saw_done = True
                    continue
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    raise OlmoeServingClientError("generate SSE line is not data")
                try:
                    event = _StreamEvent.model_validate_json(
                        line.removeprefix("data: ")
                    )
                except ValidationError as error:
                    raise OlmoeServingClientError(
                        "generate SSE event is invalid"
                    ) from error
                event_count += 1
                if event_count > maximum_events:
                    raise OlmoeServingClientError("generate stream has too many events")
                meta = event.meta_info
                if (
                    meta.prompt_tokens != len(request.input_ids)
                    or meta.cached_tokens != 0
                ):
                    raise OlmoeServingClientError(
                        "generate stream prompt/cache count is not exact"
                    )
                if meta.completion_tokens < previous_completion:
                    raise OlmoeServingClientError("completion count moved backwards")
                if meta.finish_reason is not None:
                    if (
                        meta.finish_reason.length
                        != request.sampling_params.max_new_tokens
                        or meta.completion_tokens
                        != request.sampling_params.max_new_tokens
                    ):
                        raise OlmoeServingClientError("finish reason is not canonical")
                    finish_reason = meta.finish_reason
                if not event.output_ids:
                    if meta.completion_tokens != previous_completion:
                        raise OlmoeServingClientError(
                            "completion advanced without output IDs"
                        )
                    continue
                if meta.completion_tokens == len(event.output_ids):
                    if len(event.output_ids) <= len(output_ids) or event.output_ids[
                        : len(output_ids)
                    ] != tuple(output_ids):
                        raise OlmoeServingClientError(
                            "cumulative output does not extend its prefix"
                        )
                    output_ids = list(event.output_ids)
                elif meta.completion_tokens == len(output_ids) + len(event.output_ids):
                    output_ids.extend(event.output_ids)
                else:
                    raise OlmoeServingClientError(
                        "output IDs disagree with completion count"
                    )
                previous_completion = meta.completion_tokens
                output_event_count += 1
                if first_output_ns is None:
                    first_output_ns = received
                    first_event_tokens = len(event.output_ids)
                last_output_ns = received
        completed = self._clock_ns()
        if (
            not saw_done
            or finish_reason is None
            or len(output_ids) != request.sampling_params.max_new_tokens
            or previous_completion != len(output_ids)
            or first_output_ns is None
            or last_output_ns is None
            or first_event_tokens is None
            or output_event_count < 2
            or last_output_ns <= first_output_ns
            or completed <= started
            or completed - started > self._timeout_ns
        ):
            raise OlmoeServingClientError("generate stream lacks complete evidence")
        output_tuple = tuple(output_ids)
        generation_window = (last_output_ns - first_output_ns) / 1_000_000_000
        return GenerateObservation(
            input_ids_sha256=token_ids_sha256(request.input_ids),
            output_ids=output_tuple,
            prompt_tokens=len(request.input_ids),
            completion_tokens=len(output_tuple),
            cached_tokens=0,
            output_ids_sha256=token_ids_sha256(output_tuple),
            finish_reason_sha256=hashlib.sha256(
                _canonical_json(finish_reason.model_dump(mode="json"))
            ).hexdigest(),
            stream_line_count=line_count,
            stream_event_count=event_count,
            output_bearing_event_count=output_event_count,
            maximum_stream_line_bytes=maximum_line_bytes,
            first_stream_event_output_tokens=first_event_tokens,
            total_client_seconds=(completed - started) / 1_000_000_000,
            client_observed_ttft_seconds=(first_output_ns - started) / 1_000_000_000,
            client_observed_generation_window_seconds=generation_window,
            client_observed_decode_tokens_per_second=(
                len(output_tuple) - first_event_tokens
            )
            / generation_window,
        )
