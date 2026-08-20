from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from scripts.benchmark_sglang_openai import (
    BenchmarkConfiguration,
    BenchmarkError,
    CompletionResult,
    ServerMetadata,
    campaign_summary,
    iter_sse_data,
    read_streaming_response,
    request_document,
)


class FakeLineResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    def __enter__(self) -> FakeLineResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def readline(self) -> bytes:
        return next(self._lines, b"")


def _configuration(*, stream: bool) -> BenchmarkConfiguration:
    return BenchmarkConfiguration(
        endpoint="http://127.0.0.1:30022/v1/chat/completions",
        model="Qwen/Qwen3.8-27B-FP8",
        prompt="Write a deterministic function.",
        max_tokens=512,
        warmups=1,
        samples=3,
        timeout_seconds=600.0,
        output_jsonl=Path("receipt.jsonl"),
        show_content=False,
        enable_thinking=False,
        stream=stream,
        api_key=None,
        api_key_environment_variable="OPENAI_API_KEY",
    )


def _sse_event(document: object) -> bytes:
    encoded = json.dumps(document, separators=(",", ":")).encode()
    return b"data: " + encoded + b"\n\n"


def _stream_lines(*events: bytes) -> list[bytes]:
    return b"".join(events).splitlines(keepends=True)


def test_request_documents_select_compatible_metadata_mode() -> None:
    nonstream = request_document(_configuration(stream=False))
    assert nonstream["stream"] is False
    assert nonstream["return_meta_info"] is True
    assert "stream_options" not in nonstream

    stream = request_document(_configuration(stream=True))
    assert stream["stream"] is True
    assert stream["stream_options"] == {"include_usage": True}
    assert "return_meta_info" not in stream


def test_iter_sse_data_handles_comments_and_multiline_data() -> None:
    response = FakeLineResponse(
        [
            b": keepalive\r\n",
            b"event: message\r\n",
            b"data: {\r\n",
            b'data: "ok": true}\r\n',
            b"\r\n",
        ]
    )

    assert list(iter_sse_data(response)) == [b'{\n"ok": true}']


def test_read_streaming_response_records_text_usage_and_ttft() -> None:
    response = FakeLineResponse(
        _stream_lines(
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"delta": {"role": "assistant"}}],
                }
            ),
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"delta": {"reasoning_content": "plan "}}],
                }
            ),
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"delta": {"content": "answer"}}],
                }
            ),
            _sse_event(
                cast(
                    object,
                    {
                        "id": "chatcmpl-1",
                        "choices": [{"delta": {}, "finish_reason": "length"}],
                    },
                )
            ),
            _sse_event(
                {
                    "id": "chatcmpl-1",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 5,
                        "total_tokens": 22,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )

    payload = read_streaming_response(
        response,
        started_at=10.0,
        clock=lambda: 12.5,
    )

    assert payload.prompt_tokens == 17
    assert payload.completion_tokens == 5
    assert payload.total_tokens == 22
    assert payload.content == "answer"
    assert payload.reasoning_content == "plan "
    assert payload.finish_reason == "length"
    assert payload.response_id == "chatcmpl-1"
    assert payload.client_ttft_seconds == 2.5
    assert payload.sse_json_events == 5
    assert payload.sse_nonempty_text_events == 2


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (_sse_event({"choices": [{"delta": {"content": "partial"}}]}),),
            "before the \\[DONE\\] sentinel",
        ),
        ((b"data: [DONE]\n\n",), "did not include usage"),
    ],
)
def test_read_streaming_response_rejects_incomplete_receipts(
    events: tuple[bytes, ...], message: str
) -> None:
    with pytest.raises(BenchmarkError, match=message):
        read_streaming_response(
            FakeLineResponse(_stream_lines(*events)),
            started_at=0.0,
            clock=lambda: 1.0,
        )


def test_receipts_and_summaries_distinguish_client_and_server_rates() -> None:
    metadata = ServerMetadata(
        raw={
            "decode_throughput": 180.0,
            "forward_entry_time": 100.0,
            "prefill_finished_time": 100.2,
            "request_finished_ts": 105.2,
        },
        e2e_latency_seconds=5.3,
        reported_decode_throughput=180.0,
        request_received_timestamp=99.9,
        api_server_dispatch_finish_timestamp=99.95,
        request_finished_timestamp=105.2,
        response_sent_to_client_timestamp=None,
        forward_entry_timestamp=100.0,
        prefill_finished_timestamp=100.2,
        speculative_accept_rate=0.75,
        speculative_accept_length=4.75,
        speculative_cap_length=5.0,
        speculative_verify_count=106,
    )
    nonstream_result = CompletionResult(
        phase="sample",
        index=1,
        started_at_utc="2026-08-15T12:00:00.000Z",
        wall_seconds=5.4,
        prompt_tokens=50,
        completion_tokens=501,
        total_tokens=551,
        content="answer",
        reasoning_content=None,
        finish_reason="length",
        response_id="chatcmpl-1",
        client_ttft_seconds=None,
        sse_json_events=None,
        sse_nonempty_text_events=None,
        server_metadata=metadata,
    )
    nonstream_configuration = _configuration(stream=False)
    nonstream_receipt = nonstream_result.receipt(
        campaign_id="campaign", configuration=nonstream_configuration
    )

    assert nonstream_receipt["server_reported_decode_throughput"] == 180.0
    assert math.isclose(
        cast(float, nonstream_receipt["server_prefill_tokens_per_second"]), 250.0
    )
    assert math.isclose(
        cast(float, nonstream_receipt["server_decode_tokens_per_second"]), 100.0
    )
    assert "client_ttft_seconds" not in nonstream_receipt

    stream_result = CompletionResult(
        phase="sample",
        index=1,
        started_at_utc="2026-08-15T12:00:00.000Z",
        wall_seconds=6.0,
        prompt_tokens=50,
        completion_tokens=501,
        total_tokens=551,
        content="answer",
        reasoning_content="plan",
        finish_reason="length",
        response_id="chatcmpl-2",
        client_ttft_seconds=1.0,
        sse_json_events=12,
        sse_nonempty_text_events=10,
        server_metadata=None,
    )
    stream_configuration = _configuration(stream=True)
    stream_receipt = stream_result.receipt(
        campaign_id="campaign", configuration=stream_configuration
    )
    assert math.isclose(
        cast(
            float,
            stream_receipt["post_first_token_completion_tokens_per_second"],
        ),
        100.0,
    )
    assert math.isclose(
        cast(float, stream_receipt["full_wall_completion_tokens_per_second"]),
        83.5,
    )
    assert "server_meta_info" not in stream_receipt

    stream_summary = campaign_summary(
        campaign_id="campaign",
        configuration=stream_configuration,
        results=[stream_result],
    )
    ttft_summary = stream_summary["client_ttft_seconds"]
    assert isinstance(ttft_summary, dict)
    assert ttft_summary["mean"] == 1.0
