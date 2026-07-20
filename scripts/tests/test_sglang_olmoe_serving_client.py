from __future__ import annotations

import json

import httpx
import pytest

from scripts.sglang_olmoe_serving_client import (
    OLMOE_VOCABULARY_SIZE,
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
