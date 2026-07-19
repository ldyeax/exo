#!/usr/bin/env python3
"""Deterministic native SGLang client for GLM-4.7 serving benchmarks.

This module does not launch a server or acquire the benchmark lease. A lease
owner can inject its rank-zero endpoint, run the two canonical workloads, and
embed the returned evidence in a warm-serving receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, cast, final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from exo.shared.types.common import NodeId
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (
    GLM_4_7_FLASH_DECODE_INPUT_TOKENS,
    GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS,
    GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION,
    GLM_4_7_FLASH_PREFILL_INPUT_TOKENS,
    GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS,
    GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
    SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES,
    SGLANG_KT_SERVING_SSE_EVENT_SLACK,
    SGLANG_KT_SERVING_SSE_LINES_PER_TOKEN_LIMIT,
    WARM_SERVING_MINIMUM_SAMPLES,
    WARM_SERVING_MINIMUM_WARMUPS,
    ServingWorkloadKind,
    SglangKtServingInvocationEvidence,
    SglangKtServingServerInfoIdentity,
    SglangKtServingWorkloadEvidence,
    SglangKtServingWorkloadRequest,
    calculate_sglang_kt_length_finish_reason_sha256,
    calculate_sglang_kt_token_ids_sha256,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

GLM_4_7_FLASH_VOCABULARY_SIZE = 154_880
DETERMINISTIC_INPUT_ID_FLOOR = 100
TTFT_SEMANTICS = "client_stream_first_output_event_including_http_and_queue_v1"
SSE_READ_CHUNK_BYTES = 16 * 1024


class Glm47ServingClientError(RuntimeError):
    """Raised when native SGLang serving evidence is incomplete or ambiguous."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@final
class NativeSamplingParameters(_StrictModel):
    max_new_tokens: int = Field(gt=0)
    temperature: float
    ignore_eos: Literal[True]
    sampling_seed: int

    @model_validator(mode="after")
    def validate_greedy_request(self) -> "NativeSamplingParameters":
        if (
            not math.isfinite(self.temperature)
            or self.temperature != 0.0
            or self.sampling_seed != GLM_4_7_FLASH_SERVING_SAMPLING_SEED
        ):
            raise ValueError("native benchmark sampling parameters are not pinned")
        return self


@final
class NativeGenerateRequest(_StrictModel):
    input_ids: tuple[int, ...]
    sampling_params: NativeSamplingParameters
    stream: Literal[True]
    return_logprob: Literal[False]
    log_metrics: Literal[True]

    @model_validator(mode="after")
    def validate_input_ids(self) -> "NativeGenerateRequest":
        if not self.input_ids or any(
            token_id < 0 or token_id >= GLM_4_7_FLASH_VOCABULARY_SIZE
            for token_id in self.input_ids
        ):
            raise ValueError("native request input IDs are outside the GLM vocabulary")
        return self


@final
class _LengthFinishReason(_StrictModel):
    type: Literal["length"]
    length: int = Field(gt=0)


@final
class _NativeMetaInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    prompt_tokens: int = Field(gt=0)
    completion_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    finish_reason: _LengthFinishReason | None = None


@final
class _NativeGenerateEvent(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    output_ids: tuple[int, ...]
    meta_info: _NativeMetaInfo

    @model_validator(mode="after")
    def validate_output_ids(self) -> "_NativeGenerateEvent":
        if any(
            token_id < 0 or token_id >= GLM_4_7_FLASH_VOCABULARY_SIZE
            for token_id in self.output_ids
        ):
            raise ValueError(
                "native response output IDs are outside the GLM vocabulary"
            )
        return self


@final
class EndpointCallObservation(_StrictModel):
    status_code: int = Field(ge=100, le=599)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_seconds: float = Field(ge=0.0)


@final
class ServerInfoObservation(_StrictModel):
    call: EndpointCallObservation
    response: JsonObject
    canonical_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_canonical_response(self) -> "ServerInfoObservation":
        expected = hashlib.sha256(canonical_sglang_kt_json(self.response)).hexdigest()
        if self.canonical_response_sha256 != expected:
            raise ValueError("server_info canonical response SHA-256 does not match")
        return self


@final
class _PinnedSglangServerInfo(BaseModel):
    # /server_info contains many scheduler details; only these pinned launch
    # fields become receipt identity, while the raw response hash binds the rest.
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    version: Literal["0.0.0.dev0"]
    model_path: str
    tp_size: Literal[1]
    pp_size: Literal[1]
    nnodes: Literal[1]
    node_rank: Literal[0]
    disable_radix_cache: Literal[True]


def build_glm47_server_info_identity(
    observation: ServerInfoObservation,
    *,
    node_id: NodeId,
    host: str,
    port: int,
) -> SglangKtServingServerInfoIdentity:
    """Parse pinned rank-zero /server_info fields and bind the raw response."""

    if observation.call.status_code != 200:
        raise Glm47ServingClientError("server_info observation was not successful")
    expected_canonical_sha256 = hashlib.sha256(
        canonical_sglang_kt_json(observation.response)
    ).hexdigest()
    if observation.canonical_response_sha256 != expected_canonical_sha256:
        raise Glm47ServingClientError(
            "server_info canonical response hash is not bound"
        )
    try:
        parsed = _PinnedSglangServerInfo.model_validate(observation.response)
        identity = SglangKtServingServerInfoIdentity(
            node_id=node_id,
            host=host,
            port=port,
            canonical_response_sha256=expected_canonical_sha256,
            version=parsed.version,
            model_path=parsed.model_path,
            tp_size=parsed.tp_size,
            pp_size=parsed.pp_size,
            nnodes=parsed.nnodes,
            node_rank=parsed.node_rank,
            disable_radix_cache=parsed.disable_radix_cache,
        )
    except ValidationError as error:
        raise Glm47ServingClientError(
            "server_info response does not match the pinned local SGLang runtime"
        ) from error
    if parsed.version != GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION:
        raise Glm47ServingClientError("server_info version is not pinned")
    return identity


@final
class NativeGenerateObservation(_StrictModel):
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(gt=0)
    completion_tokens: int = Field(gt=0)
    cached_tokens: int = Field(ge=0)
    output_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stream_line_count: int = Field(gt=0)
    stream_event_count: int = Field(gt=0)
    output_bearing_event_count: int = Field(gt=0)
    maximum_stream_line_bytes: int = Field(gt=0)
    first_stream_event_output_tokens: int = Field(gt=0)
    total_client_seconds: float = Field(gt=0.0)
    client_observed_ttft_seconds: float = Field(gt=0.0)
    client_observed_generation_window_seconds: float = Field(gt=0.0)
    client_observed_decode_tokens_per_second: float = Field(gt=0.0)


@final
@dataclass(frozen=True)
class PreparedServingWorkload:
    native_request: NativeGenerateRequest
    receipt_request: SglangKtServingWorkloadRequest


def build_deterministic_glm47_input_ids(
    kind: ServingWorkloadKind,
) -> tuple[int, ...]:
    """Build stable valid token IDs without involving a mutable tokenizer."""

    token_count = (
        GLM_4_7_FLASH_PREFILL_INPUT_TOKENS
        if kind == "prefill"
        else GLM_4_7_FLASH_DECODE_INPUT_TOKENS
    )
    modulus = GLM_4_7_FLASH_VOCABULARY_SIZE - DETERMINISTIC_INPUT_ID_FLOOR
    token_ids: list[int] = []
    counter = 0
    while len(token_ids) < token_count:
        digest = hashlib.sha256(
            f"exo-glm47-serving-v1:{kind}:{counter}".encode()
        ).digest()
        for offset in range(0, len(digest), 4):
            value = int.from_bytes(digest[offset : offset + 4], "big")
            token_ids.append(DETERMINISTIC_INPUT_ID_FLOOR + value % modulus)
            if len(token_ids) == token_count:
                break
        counter += 1
    return tuple(token_ids)


def prepare_glm47_serving_workload(
    kind: ServingWorkloadKind,
) -> PreparedServingWorkload:
    input_ids = build_deterministic_glm47_input_ids(kind)
    max_new_tokens = (
        GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS
        if kind == "prefill"
        else GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS
    )
    native_request = NativeGenerateRequest(
        input_ids=input_ids,
        sampling_params=NativeSamplingParameters(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
            sampling_seed=GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
        ),
        stream=True,
        return_logprob=False,
        log_metrics=True,
    )
    return PreparedServingWorkload(
        native_request=native_request,
        receipt_request=SglangKtServingWorkloadRequest(
            kind=kind,
            input_token_count=len(input_ids),
            input_ids_sha256=calculate_sglang_kt_token_ids_sha256(input_ids),
            max_new_tokens=max_new_tokens,
            sampling_seed=GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
            temperature=0.0,
            ignore_eos=True,
            stream=True,
            return_logprob=False,
            log_metrics=True,
        ),
    )


def _strict_json_object(contents: bytes, description: str) -> JsonObject:
    try:
        parsed = parse_sglang_kt_strict_json(contents)
    except SglangKtReceiptFileError as error:
        raise Glm47ServingClientError(f"{description} is not strict JSON") from error
    if not isinstance(parsed, dict):
        raise Glm47ServingClientError(f"{description} is not a JSON object")
    parsed_mapping = cast(dict[object, object], parsed)
    if not all(isinstance(key, str) for key in parsed_mapping):
        raise Glm47ServingClientError(f"{description} has a non-string JSON key")
    return cast(JsonObject, parsed_mapping)


def _parse_sse_event(line: str) -> _NativeGenerateEvent | None:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data: "):
        raise Glm47ServingClientError("native generate stream contains a non-data line")
    payload = line.removeprefix("data: ")
    if payload == "[DONE]":
        return None
    try:
        return _NativeGenerateEvent.model_validate_json(payload)
    except ValidationError as error:
        raise Glm47ServingClientError(
            "native generate stream contains an invalid event"
        ) from error


def _iter_bounded_sse_lines(
    response: httpx.Response,
    require_within_deadline: Callable[[], None],
) -> Iterator[str]:
    """Decode one SSE response without buffering an unbounded line."""

    pending = bytearray()
    try:
        chunks = iter(response.iter_bytes())
        while True:
            require_within_deadline()
            try:
                chunk = next(chunks)
            except StopIteration:
                break
            require_within_deadline()
            chunk_view = memoryview(chunk)
            for offset in range(0, len(chunk), SSE_READ_CHUNK_BYTES):
                require_within_deadline()
                pending.extend(chunk_view[offset : offset + SSE_READ_CHUNK_BYTES])
                while (newline_index := pending.find(b"\n")) >= 0:
                    require_within_deadline()
                    encoded_line = bytes(pending[:newline_index])
                    del pending[: newline_index + 1]
                    if encoded_line.endswith(b"\r"):
                        encoded_line = encoded_line[:-1]
                    if len(encoded_line) > SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES:
                        raise Glm47ServingClientError(
                            "native generate stream contains an oversized line"
                        )
                    yield encoded_line.decode("utf-8", errors="strict")
                if len(pending) > SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES + 1:
                    raise Glm47ServingClientError(
                        "native generate stream contains an oversized line"
                    )
    except UnicodeDecodeError as error:
        raise Glm47ServingClientError(
            "native generate stream is not strict UTF-8"
        ) from error
    if pending:
        raise Glm47ServingClientError(
            "native generate stream ends with an unterminated line"
        )


@final
class Glm47NativeServingClient:
    """Small synchronous client whose measured boundary is the HTTP stream."""

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
            raise ValueError("request timeout must be positive and finite")
        self._clock_ns = clock_ns
        self._deadline_clock_ns = deadline_clock_ns
        self._timeout_ns = int(timeout_seconds * 1_000_000_000)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"Accept-Encoding": "identity"},
        )

    def __enter__(self) -> Glm47NativeServingClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _endpoint_call(
        self,
        method: Literal["GET", "POST"],
        path: str,
    ) -> tuple[EndpointCallObservation, bytes]:
        started_ns = self._clock_ns()
        response = self._client.request(method, path)
        completed_ns = self._clock_ns()
        contents = response.content
        observation = EndpointCallObservation(
            status_code=response.status_code,
            response_sha256=hashlib.sha256(contents).hexdigest(),
            elapsed_seconds=(completed_ns - started_ns) / 1_000_000_000,
        )
        if response.status_code != 200:
            raise Glm47ServingClientError(
                f"{method} {path} returned HTTP {response.status_code}"
            )
        return observation, contents

    def health_generate(self) -> EndpointCallObservation:
        observation, _ = self._endpoint_call("GET", "/health_generate")
        return observation

    def flush_cache(self) -> EndpointCallObservation:
        observation, _ = self._endpoint_call("POST", "/flush_cache")
        return observation

    def server_info(self) -> ServerInfoObservation:
        call, contents = self._endpoint_call("GET", "/server_info")
        response = _strict_json_object(contents, "server_info response")
        canonical_response_sha256 = hashlib.sha256(
            canonical_sglang_kt_json(response)
        ).hexdigest()
        return ServerInfoObservation(
            call=call,
            response=response,
            canonical_response_sha256=canonical_response_sha256,
        )

    def generate(
        self,
        request: NativeGenerateRequest,
    ) -> NativeGenerateObservation:
        started_ns = self._clock_ns()
        deadline_ns = self._deadline_clock_ns() + self._timeout_ns

        def require_within_deadline() -> None:
            if self._deadline_clock_ns() > deadline_ns:
                raise Glm47ServingClientError(
                    "native generate stream exceeded the client duration bound"
                )

        maximum_events = (
            request.sampling_params.max_new_tokens + SGLANG_KT_SERVING_SSE_EVENT_SLACK
        )
        maximum_lines = (
            request.sampling_params.max_new_tokens + 2
        ) * SGLANG_KT_SERVING_SSE_LINES_PER_TOKEN_LIMIT
        output_ids: list[int] = []
        seen_event_sha256: set[str] = set()
        stream_line_count = 0
        stream_event_count = 0
        output_bearing_event_count = 0
        maximum_stream_line_bytes = 0
        first_output_ns: int | None = None
        last_output_ns: int | None = None
        first_event_output_tokens: int | None = None
        prompt_tokens: int | None = None
        cached_tokens: int | None = None
        previous_completion_tokens = 0
        final_event: _NativeGenerateEvent | None = None
        saw_done = False
        with self._client.stream(
            "POST",
            "/generate",
            json=request.model_dump(mode="json"),
        ) as response:
            if response.status_code != 200:
                raise Glm47ServingClientError(
                    f"POST /generate returned HTTP {response.status_code}"
                )
            if response.headers.get("content-encoding", "identity") != "identity":
                raise Glm47ServingClientError(
                    "native generate stream uses an unpinned content encoding"
                )
            for line in _iter_bounded_sse_lines(response, require_within_deadline):
                received_ns = self._clock_ns()
                stream_line_count += 1
                line_bytes = len(line.encode("utf-8"))
                maximum_stream_line_bytes = max(
                    maximum_stream_line_bytes,
                    line_bytes,
                )
                if received_ns - started_ns > self._timeout_ns:
                    raise Glm47ServingClientError(
                        "native generate stream exceeded the client duration bound"
                    )
                if stream_line_count > maximum_lines:
                    raise Glm47ServingClientError(
                        "native generate stream exceeded its line-count bound"
                    )
                if line_bytes > SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES:
                    raise Glm47ServingClientError(
                        "native generate stream contains an oversized line"
                    )
                if saw_done:
                    if line:
                        raise Glm47ServingClientError(
                            "native generate stream contains data after [DONE]"
                        )
                    continue
                if line == "data: [DONE]":
                    saw_done = True
                    continue
                event = _parse_sse_event(line)
                if event is None:
                    continue

                stream_event_count += 1
                if stream_event_count > maximum_events:
                    raise Glm47ServingClientError(
                        "native generate stream exceeded its event-count bound"
                    )
                event_sha256 = hashlib.sha256(
                    canonical_sglang_kt_json(event.model_dump(mode="json"))
                ).hexdigest()
                if event_sha256 in seen_event_sha256:
                    raise Glm47ServingClientError(
                        "native generate stream contains a duplicate event"
                    )
                seen_event_sha256.add(event_sha256)

                meta = event.meta_info
                if meta.finish_reason is not None and (
                    meta.completion_tokens != request.sampling_params.max_new_tokens
                    or meta.finish_reason.length
                    != request.sampling_params.max_new_tokens
                ):
                    raise Glm47ServingClientError(
                        "native generate stream has a noncanonical finish reason"
                    )
                if prompt_tokens is None:
                    prompt_tokens = meta.prompt_tokens
                    cached_tokens = meta.cached_tokens
                elif (
                    meta.prompt_tokens != prompt_tokens
                    or meta.cached_tokens != cached_tokens
                ):
                    raise Glm47ServingClientError(
                        "native generate metadata changed within one stream"
                    )
                if meta.completion_tokens < previous_completion_tokens:
                    raise Glm47ServingClientError(
                        "native generate completion count moved backwards"
                    )
                if (
                    meta.completion_tokens > request.sampling_params.max_new_tokens
                    or len(event.output_ids) > request.sampling_params.max_new_tokens
                ):
                    raise Glm47ServingClientError(
                        "native generate completion exceeded the request bound"
                    )
                if not event.output_ids:
                    if meta.completion_tokens != previous_completion_tokens:
                        raise Glm47ServingClientError(
                            "native generate completion advanced without output IDs"
                        )
                    final_event = event
                    continue
                if meta.completion_tokens <= previous_completion_tokens:
                    raise Glm47ServingClientError(
                        "native generate output did not advance completion"
                    )

                if meta.completion_tokens == len(event.output_ids):
                    if len(event.output_ids) <= len(output_ids) or event.output_ids[
                        : len(output_ids)
                    ] != tuple(output_ids):
                        raise Glm47ServingClientError(
                            "native cumulative output does not extend its prior prefix"
                        )
                    output_ids = list(event.output_ids)
                elif meta.completion_tokens == len(output_ids) + len(event.output_ids):
                    output_ids.extend(event.output_ids)
                else:
                    raise Glm47ServingClientError(
                        "native output IDs do not match the cumulative completion count"
                    )
                if meta.completion_tokens != len(output_ids):
                    raise Glm47ServingClientError(
                        "native output progress does not match collected output IDs"
                    )

                output_bearing_event_count += 1
                if first_output_ns is None:
                    first_output_ns = received_ns
                    first_event_output_tokens = len(event.output_ids)
                last_output_ns = received_ns
                previous_completion_tokens = meta.completion_tokens
                final_event = event
        completed_ns = self._clock_ns()
        if completed_ns - started_ns > self._timeout_ns:
            raise Glm47ServingClientError(
                "native generate stream exceeded the client duration bound"
            )
        if not saw_done or final_event is None:
            raise Glm47ServingClientError("native generate stream is incomplete")
        final_meta = final_event.meta_info
        if (
            prompt_tokens is None
            or cached_tokens is None
            or first_output_ns is None
            or last_output_ns is None
            or first_event_output_tokens is None
            or final_meta.finish_reason is None
            or final_meta.completion_tokens != len(output_ids)
            or final_meta.completion_tokens != request.sampling_params.max_new_tokens
            or output_bearing_event_count < 2
            or last_output_ns <= first_output_ns
            or completed_ns <= started_ns
        ):
            raise Glm47ServingClientError(
                "native generate stream lacks complete timing or token evidence"
            )

        generation_window_seconds = (last_output_ns - first_output_ns) / 1_000_000_000
        return NativeGenerateObservation(
            input_ids_sha256=calculate_sglang_kt_token_ids_sha256(request.input_ids),
            prompt_tokens=prompt_tokens,
            completion_tokens=final_meta.completion_tokens,
            cached_tokens=cached_tokens,
            output_ids_sha256=calculate_sglang_kt_token_ids_sha256(tuple(output_ids)),
            finish_reason_sha256=(
                calculate_sglang_kt_length_finish_reason_sha256(
                    final_meta.finish_reason.length
                )
            ),
            stream_line_count=stream_line_count,
            stream_event_count=stream_event_count,
            output_bearing_event_count=output_bearing_event_count,
            maximum_stream_line_bytes=maximum_stream_line_bytes,
            first_stream_event_output_tokens=first_event_output_tokens,
            total_client_seconds=(completed_ns - started_ns) / 1_000_000_000,
            client_observed_ttft_seconds=(first_output_ns - started_ns) / 1_000_000_000,
            client_observed_generation_window_seconds=generation_window_seconds,
            client_observed_decode_tokens_per_second=(
                final_meta.completion_tokens - first_event_output_tokens
            )
            / generation_window_seconds,
        )


def _invocation_evidence(
    *,
    ordinal: int,
    flush: EndpointCallObservation,
    generate: NativeGenerateObservation,
) -> SglangKtServingInvocationEvidence:
    return SglangKtServingInvocationEvidence(
        ordinal=ordinal,
        input_ids_sha256=generate.input_ids_sha256,
        cache_flush_status_code=cast(Literal[200], flush.status_code),
        cache_flush_response_sha256=flush.response_sha256,
        prompt_tokens=generate.prompt_tokens,
        completion_tokens=generate.completion_tokens,
        cached_tokens=generate.cached_tokens,
        output_ids_sha256=generate.output_ids_sha256,
        finish_reason_sha256=generate.finish_reason_sha256,
        stream_line_count=generate.stream_line_count,
        stream_event_count=generate.stream_event_count,
        output_bearing_event_count=generate.output_bearing_event_count,
        maximum_stream_line_bytes=generate.maximum_stream_line_bytes,
        first_stream_event_output_tokens=generate.first_stream_event_output_tokens,
        total_client_seconds=generate.total_client_seconds,
        client_observed_ttft_seconds=generate.client_observed_ttft_seconds,
        client_observed_generation_window_seconds=(
            generate.client_observed_generation_window_seconds
        ),
        client_observed_decode_tokens_per_second=(
            generate.client_observed_decode_tokens_per_second
        ),
        ttft_semantics=TTFT_SEMANTICS,
    )


def run_glm47_serving_invocation(
    client: Glm47NativeServingClient,
    workload: PreparedServingWorkload,
    ordinal: int,
) -> SglangKtServingInvocationEvidence:
    """Flush the radix cache and collect one bounded native invocation."""

    return _invocation_evidence(
        ordinal=ordinal,
        flush=client.flush_cache(),
        generate=client.generate(workload.native_request),
    )


def run_glm47_serving_workload(
    client: Glm47NativeServingClient,
    workload: PreparedServingWorkload,
    *,
    warmup_count: int = WARM_SERVING_MINIMUM_WARMUPS,
    sample_count: int = WARM_SERVING_MINIMUM_SAMPLES,
) -> SglangKtServingWorkloadEvidence:
    """Collect one workload without collecting external JIT cache manifests.

    A performance receipt harness can interleave calls to
    ``run_glm47_serving_invocation`` so it can hash owned JIT cache directories
    after the penultimate warmup, final warmup, and measurement phases.
    """

    if warmup_count < WARM_SERVING_MINIMUM_WARMUPS:
        raise ValueError("at least two warmups are required")
    if sample_count < WARM_SERVING_MINIMUM_SAMPLES:
        raise ValueError("at least three measurement samples are required")

    warmups = tuple(
        run_glm47_serving_invocation(client, workload, ordinal)
        for ordinal in range(1, warmup_count + 1)
    )
    samples = tuple(
        run_glm47_serving_invocation(client, workload, ordinal)
        for ordinal in range(1, sample_count + 1)
    )
    return SglangKtServingWorkloadEvidence(
        request=workload.receipt_request,
        warmups=warmups,
        samples=samples,
    )


def run_required_glm47_serving_workloads(
    client: Glm47NativeServingClient,
    *,
    warmup_count: int = WARM_SERVING_MINIMUM_WARMUPS,
    sample_count: int = WARM_SERVING_MINIMUM_SAMPLES,
) -> tuple[SglangKtServingWorkloadEvidence, SglangKtServingWorkloadEvidence]:
    return (
        run_glm47_serving_workload(
            client,
            prepare_glm47_serving_workload("prefill"),
            warmup_count=warmup_count,
            sample_count=sample_count,
        ),
        run_glm47_serving_workload(
            client,
            prepare_glm47_serving_workload("decode"),
            warmup_count=warmup_count,
            sample_count=sample_count,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmups", type=int, default=WARM_SERVING_MINIMUM_WARMUPS)
    parser.add_argument("--samples", type=int, default=WARM_SERVING_MINIMUM_SAMPLES)
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_url = cast(str, args.base_url)
    timeout_seconds = cast(float, args.timeout_seconds)
    warmups = cast(int, args.warmups)
    samples = cast(int, args.samples)
    with Glm47NativeServingClient(
        base_url,
        timeout_seconds=timeout_seconds,
    ) as client:
        health = client.health_generate()
        server_info = client.server_info()
        workloads = run_required_glm47_serving_workloads(
            client,
            warmup_count=warmups,
            sample_count=samples,
        )
    payload = {
        "health_generate": health.model_dump(mode="json"),
        "server_info": server_info.model_dump(mode="json"),
        "workloads": [workload.model_dump(mode="json") for workload in workloads],
    }
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
