from __future__ import annotations

import json

import httpx
import pytest

from scripts.sglang_olmoe_serving_client import (
    LOGIT_PARITY_IO_TIMEOUT_MAXIMUM_SECONDS,
    LOGIT_PARITY_RESPONSE_MAXIMUM_BYTES,
    OLMOE_VOCABULARY_SIZE,
    OlmoeLogitParityRequest,
    OlmoeNativeGenerateRequest,
    OlmoeNativeServingClient,
    OlmoeSamplingParameters,
    OlmoeServingClientError,
)


def _request(*, stream: bool, maximum: int = 3) -> OlmoeNativeGenerateRequest:
    return OlmoeNativeGenerateRequest(
        input_ids=(101, 102, 103),
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=maximum,
            temperature=0.0,
            ignore_eos=stream,
            sampling_seed=20_260_720,
        ),
        stream=stream,
    )


def _sanity_transport(
    *,
    prompt_tokens: int = 3,
    cached_tokens: int = 0,
    output_ids: list[int] | None = None,
) -> httpx.MockTransport:
    ids = [201, 202] if output_ids is None else output_ids

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        return httpx.Response(
            200,
            json={
                "text": "42",
                "output_ids": ids,
                "meta_info": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(ids),
                    "cached_tokens": cached_tokens,
                    "finish_reason": {"type": "stop"},
                },
            },
        )

    return httpx.MockTransport(handler)


def test_sanity_requires_exact_prompt_cache_and_vocabulary() -> None:
    with OlmoeNativeServingClient(
        "http://test", timeout_seconds=10.0, transport=_sanity_transport()
    ) as client:
        observation = client.generate_sanity(_request(stream=False))

    assert observation.prompt_tokens == 3
    assert observation.cached_tokens == 0
    assert observation.output_ids == (201, 202)


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (_sanity_transport(prompt_tokens=2), "prompt/cache/output"),
        (_sanity_transport(cached_tokens=1), "prompt/cache/output"),
        (_sanity_transport(output_ids=[OLMOE_VOCABULARY_SIZE]), "invalid"),
    ],
)
def test_sanity_rejects_unbound_response(
    transport: httpx.MockTransport, message: str
) -> None:
    with (
        OlmoeNativeServingClient(
            "http://test", timeout_seconds=10.0, transport=transport
        ) as client,
        pytest.raises(OlmoeServingClientError, match=message),
    ):
        client.generate_sanity(_request(stream=False))


def _logit_parity_request() -> OlmoeLogitParityRequest:
    return OlmoeLogitParityRequest(
        input_ids=(101, 102, 103),
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=1,
            temperature=0.0,
            ignore_eos=True,
            sampling_seed=20_260_720,
        ),
        candidate_token_ids=(139, 1_769),
        top_logprobs_num=3,
    )


def test_logit_parity_candidate_count_is_bounded_independently_of_top_k() -> None:
    with pytest.raises(ValueError, match="bound"):
        OlmoeLogitParityRequest(
            input_ids=(101,),
            sampling_params=OlmoeSamplingParameters(
                max_new_tokens=1,
                temperature=0.0,
                ignore_eos=True,
                sampling_seed=20_260_720,
            ),
            candidate_token_ids=tuple(range(65)),
            top_logprobs_num=1,
        )


def _logit_parity_transport(
    *,
    top_logprobs: list[list[object]] | None = None,
    candidate_logprobs: list[list[object]] | None = None,
) -> httpx.MockTransport:
    top = (
        [[-0.4, 139, None], [-0.5, 1_769, None], [-1.0, 42, None]]
        if top_logprobs is None
        else top_logprobs
    )
    candidates = (
        [[-0.4, 139, None], [-0.5, 1_769, None]]
        if candidate_logprobs is None
        else candidate_logprobs
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        timeout = request.extensions["timeout"]
        assert isinstance(timeout, dict)
        assert timeout["read"] == LOGIT_PARITY_IO_TIMEOUT_MAXIMUM_SECONDS
        assert payload["stream"] is False
        assert payload["return_logprob"] is True
        assert payload["logprob_start_len"] == -1
        assert payload["top_logprobs_num"] == 3
        assert payload["token_ids_logprob"] == [139, 1_769]
        response_payload = {
            "text": " token",
            "output_ids": [139],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "length", "length": 1},
                "output_token_logprobs": [[-0.4, 139, None]],
                "output_top_logprobs": [top],
                "output_token_ids_logprobs": [candidates],
            },
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(response_payload).encode()),
        )

    return httpx.MockTransport(handler)


def test_logit_parity_captures_exact_candidate_and_top_k_evidence() -> None:
    with OlmoeNativeServingClient(
        "http://test", timeout_seconds=10.0, transport=_logit_parity_transport()
    ) as client:
        observation = client.generate_logit_parity(_logit_parity_request())

    assert observation.generated_token_id == 139
    assert observation.generated_token_logprob == -0.4
    assert [entry.token_id for entry in observation.candidate_logprobs] == [139, 1_769]
    assert [entry.token_id for entry in observation.top_logprobs] == [139, 1_769, 42]
    assert len(observation.response_sha256) == 64


def test_logit_parity_accepts_explicit_candidate_outside_top_k() -> None:
    transport = _logit_parity_transport(
        top_logprobs=[[-0.4, 139, None], [-0.6, 42, None], [-1.0, 43, None]],
        candidate_logprobs=[[-0.4, 139, None], [-8.0, 1_769, None]],
    )

    with OlmoeNativeServingClient(
        "http://test", timeout_seconds=10.0, transport=transport
    ) as client:
        observation = client.generate_logit_parity(_logit_parity_request())

    assert [entry.token_id for entry in observation.candidate_logprobs] == [139, 1_769]
    assert observation.candidate_logprobs[1].logprob == -8.0


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (
            _logit_parity_transport(
                top_logprobs=[[-0.4, 139, None], [-0.5, 1_769, None]]
            ),
            "exact request",
        ),
        (
            _logit_parity_transport(
                candidate_logprobs=[[-0.5, 1_769, None], [-0.4, 139, None]]
            ),
            "exact request",
        ),
        (
            _logit_parity_transport(
                candidate_logprobs=[[-8.0, 139, None], [-0.5, 1_769, None]]
            ),
            "exact request",
        ),
        (
            _logit_parity_transport(
                top_logprobs=[
                    [-0.5, 1_769, None],
                    [-0.4, 139, None],
                    [-1.0, 42, None],
                ]
            ),
            "exact request",
        ),
        (
            _logit_parity_transport(
                top_logprobs=[
                    [-0.3, 42, None],
                    [-0.4, 139, None],
                    [-1.0, 1_769, None],
                ]
            ),
            "exact request",
        ),
    ],
)
def test_logit_parity_rejects_incomplete_or_ambiguous_evidence(
    transport: httpx.MockTransport, message: str
) -> None:
    with (
        OlmoeNativeServingClient(
            "http://test", timeout_seconds=10.0, transport=transport
        ) as client,
        pytest.raises(OlmoeServingClientError, match=message),
    ):
        client.generate_logit_parity(_logit_parity_request())


def test_logit_parity_rejects_oversized_response_before_parsing() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            stream=httpx.ByteStream(b"x" * (LOGIT_PARITY_RESPONSE_MAXIMUM_BYTES + 1)),
        )
    )

    with (
        OlmoeNativeServingClient(
            "http://test", timeout_seconds=10.0, transport=transport
        ) as client,
        pytest.raises(OlmoeServingClientError, match="oversized"),
    ):
        client.generate_logit_parity(_logit_parity_request())


def test_logit_parity_enforces_absolute_stream_deadline() -> None:
    timestamps = iter((0, 0, 0, 11_000_000_000))

    with (
        OlmoeNativeServingClient(
            "http://test",
            timeout_seconds=10.0,
            transport=_logit_parity_transport(),
            deadline_clock_ns=lambda: next(timestamps, 11_000_000_000),
        ) as client,
        pytest.raises(OlmoeServingClientError, match="deadline"),
    ):
        client.generate_logit_parity(_logit_parity_request())


def test_logit_parity_rejects_compressed_response() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=httpx.ByteStream(b"not accepted"),
        )
    )

    with (
        OlmoeNativeServingClient(
            "http://test", timeout_seconds=10.0, transport=transport
        ) as client,
        pytest.raises(OlmoeServingClientError, match="compressed"),
    ):
        client.generate_logit_parity(_logit_parity_request())


def _stream_transport(
    *, prompt_tokens: int = 3, cached_tokens: int = 0, final_id: int = 203
) -> httpx.MockTransport:
    events = (
        {
            "output_ids": [201],
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "cached_tokens": cached_tokens,
                "finish_reason": None,
            },
        },
        {
            "output_ids": [201, 202],
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
                "cached_tokens": cached_tokens,
                "finish_reason": None,
            },
        },
        {
            "output_ids": [201, 202, final_id],
            "meta_info": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 3,
                "cached_tokens": cached_tokens,
                "finish_reason": {"type": "length", "length": 3},
            },
        },
    )
    contents = (
        "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n" for event in events
        )
        + "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=contents.encode(),
        )

    return httpx.MockTransport(handler)


def test_stream_accepts_complete_cumulative_token_evidence() -> None:
    with OlmoeNativeServingClient(
        "http://test", timeout_seconds=10.0, transport=_stream_transport()
    ) as client:
        observation = client.generate(_request(stream=True))

    assert observation.output_ids == (201, 202, 203)
    assert observation.prompt_tokens == 3
    assert observation.cached_tokens == 0
    assert observation.stream_event_count == 3
    assert observation.output_bearing_event_count == 3
    assert observation.client_observed_decode_tokens_per_second > 0.0


def test_stream_accepts_complete_delta_token_evidence() -> None:
    events = tuple(
        {
            "output_ids": [token_id],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": index,
                "cached_tokens": 0,
                "finish_reason": (
                    {"type": "length", "length": 3} if index == 3 else None
                ),
            },
        }
        for index, token_id in enumerate((201, 202, 203), start=1)
    )
    contents = (
        "".join(f"data: {json.dumps(event)}\n" for event in events) + "data: [DONE]\n"
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=contents.encode(),
        )
    )

    with OlmoeNativeServingClient(
        "http://test", timeout_seconds=10.0, transport=transport
    ) as client:
        observation = client.generate(_request(stream=True))

    assert observation.output_ids == (201, 202, 203)
    assert observation.output_bearing_event_count == 3


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (_stream_transport(prompt_tokens=2), "prompt/cache"),
        (_stream_transport(cached_tokens=1), "prompt/cache"),
        (_stream_transport(final_id=OLMOE_VOCABULARY_SIZE), "invalid"),
    ],
)
def test_stream_rejects_prompt_cache_or_vocabulary_drift(
    transport: httpx.MockTransport, message: str
) -> None:
    with (
        OlmoeNativeServingClient(
            "http://test", timeout_seconds=10.0, transport=transport
        ) as client,
        pytest.raises(OlmoeServingClientError, match=message),
    ):
        client.generate(_request(stream=True))
