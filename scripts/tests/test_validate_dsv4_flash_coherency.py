from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
from urllib.request import Request

import pytest

from scripts import validate_dsv4_flash_coherency as gate

VALID_REASONING = (
    "I found all four values across the context: MAPLE, RIVER, ORBIT, and CEDAR."
)


def _semantic_capture(
    *,
    content: str | None = None,
    reasoning: str = VALID_REASONING,
    finish_reason: str = "stop",
) -> gate.StreamCapture:
    return gate.StreamCapture(
        http_status=200,
        elapsed_seconds=1.25,
        time_to_first_output_seconds=0.75,
        content=(
            gate._canonical_json(gate.expected_facts()) if content is None else content
        ),
        reasoning_content=reasoning,
        finish_reason=finish_reason,
        tool_calls={},
        completion_tokens=48,
        event_count=4,
        saw_done=True,
        protocol_issue_codes=(),
    )


def _tool_capture(arguments: str | None = None) -> gate.StreamCapture:
    parts = gate.ToolCallParts(
        identifier="call_coherence",
        kind="function",
        name_fragments=["record_coherence_", "decision"],
        argument_fragments=[
            arguments
            if arguments is not None
            else gate._canonical_json(gate.expected_facts())
        ],
    )
    return gate.StreamCapture(
        http_status=200,
        elapsed_seconds=0.5,
        time_to_first_output_seconds=0.25,
        content="",
        reasoning_content="",
        finish_reason="tool_calls",
        tool_calls={0: parts},
        completion_tokens=24,
        event_count=3,
        saw_done=True,
        protocol_issue_codes=(),
    )


def _sse_line(value: object) -> bytes:
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return b"data: " + encoded + b"\n"


def _semantic_lines(content: str, reasoning: str) -> tuple[bytes, ...]:
    midpoint = len(content) // 2
    return (
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": reasoning[:30],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": reasoning[30:],
                            "content": content[:midpoint],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content[midpoint:]},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [],
                "usage": {"prompt_tokens": 8_700, "completion_tokens": 48},
            }
        ),
        b"data: [DONE]\n",
    )


def _tool_lines(arguments: str) -> tuple[bytes, ...]:
    midpoint = len(arguments) // 2
    return (
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_coherence",
                                    "type": "function",
                                    "function": {
                                        "name": "record_coherence_",
                                        "arguments": arguments[:midpoint],
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "decision",
                                        "arguments": arguments[midpoint:],
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    )


class FakeResponse:
    def __init__(
        self,
        *,
        body: bytes = b"",
        lines: tuple[bytes, ...] = (),
        status: int = 200,
    ) -> None:
        self.body = body
        self.lines = lines
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *unused: object) -> None:
        del unused

    def read(self) -> bytes:
        return self.body

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


class FakeUrlOpen:
    def __init__(
        self,
        *,
        corrupt_content: str | None = None,
        busy_flushes: int = 0,
    ) -> None:
        self.corrupt_content = corrupt_content
        self.busy_flushes = busy_flushes
        self.paths: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        del timeout
        path = urlsplit(request.full_url).path
        self.paths.append(path)
        if path == "/flush_cache":
            assert request.data == b"{}"
            if self.busy_flushes > 0:
                self.busy_flushes -= 1
                return FakeResponse(body=b'{"success":false}', status=400)
            return FakeResponse(body=b'{"success":true}')
        assert path == "/v1/chat/completions"
        assert request.data is not None
        payload = json.loads(request.data)
        assert isinstance(payload, dict)
        self.payloads.append(cast(dict[str, Any], payload))
        if "tools" in payload:
            return FakeResponse(
                lines=_tool_lines(gate._canonical_json(gate.expected_facts()))
            )
        content = (
            gate._canonical_json(gate.expected_facts())
            if self.corrupt_content is None
            else self.corrupt_content
        )
        return FakeResponse(lines=_semantic_lines(content, VALID_REASONING))


def test_prompt_is_realistic_multichunk_and_requires_thinking() -> None:
    agents_context = Path("AGENTS.md").read_text(encoding="utf-8")
    system_prompt = gate.build_system_prompt(agents_context)
    payload = gate.semantic_payload(
        model="deepseek-v4-flash",
        system_prompt=system_prompt,
        maximum_tokens=256,
    )

    assert len(system_prompt.encode("utf-8")) >= 40_000
    fact_offsets = [
        system_prompt.index("amber=MAPLE"),
        system_prompt.index("cobalt=RIVER"),
        system_prompt.index("jade=ORBIT"),
    ]
    assert fact_offsets[0] > 6_000
    assert fact_offsets[1] - fact_offsets[0] > 8_000
    assert fact_offsets[2] - fact_offsets[1] > 8_000
    assert "violet=CEDAR" not in system_prompt
    user_prompt = cast(list[dict[str, str]], payload["messages"])[1]["content"]
    assert "violet=CEDAR" in user_prompt
    assert payload["stream"] is True
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"thinking": True}
    assert gate.expected_facts() == {
        "amber": "MAPLE",
        "cobalt": "RIVER",
        "jade": "ORBIT",
        "violet": "CEDAR",
    }


def test_exact_semantic_answer_and_reasoning_are_accepted() -> None:
    report = gate.validate_semantic_capture(_semantic_capture())

    assert report["accepted"] is True
    assert report["issue_codes"] == []
    assert "content" not in report
    assert "reasoning_content" not in report


@pytest.mark.parametrize(
    ("mutate", "expected_issue"),
    [
        (lambda answer: answer + " garbage", "semantic_content_not_exact"),
        (lambda answer: answer + answer, "semantic_content_not_exact"),
        (lambda answer: " " + answer, "semantic_content_not_exact"),
        (
            lambda answer: answer.replace('"jade":"ORBIT"', '"jade":"WRONG"'),
            "semantic_content_mismatch",
        ),
    ],
)
def test_semantic_answer_rejects_trailing_repeated_and_wrong_text(
    mutate: Any,
    expected_issue: str,
) -> None:
    answer = gate._canonical_json(gate.expected_facts())
    report = gate.validate_semantic_capture(_semantic_capture(content=mutate(answer)))

    assert report["accepted"] is False
    assert expected_issue in report["issue_codes"]


def test_semantic_answer_requires_real_thinking_path_evidence() -> None:
    report = gate.validate_semantic_capture(
        _semantic_capture(
            reasoning="A generic thought with enough bytes but no result."
        )
    )

    assert report["accepted"] is False
    assert "reasoning_missing_semantic_evidence" in report["issue_codes"]


def test_forced_tool_call_requires_one_exact_semantic_object() -> None:
    valid_report = gate.validate_tool_capture(_tool_capture())
    duplicate_key_arguments = (
        '{"amber":"MAPLE","amber":"MAPLE","cobalt":"RIVER",'
        '"jade":"ORBIT","violet":"CEDAR"}'
    )
    duplicate_report = gate.validate_tool_capture(
        _tool_capture(duplicate_key_arguments)
    )
    trailing_report = gate.validate_tool_capture(
        _tool_capture(gate._canonical_json(gate.expected_facts()) + " garbage")
    )
    repeated_call = _tool_capture()
    repeated_report = gate.validate_tool_capture(
        replace(
            repeated_call,
            tool_calls={0: repeated_call.tool_calls[0], 1: gate.ToolCallParts()},
        )
    )

    assert valid_report["accepted"] is True
    assert duplicate_report["accepted"] is False
    assert "tool_arguments_invalid_json" in duplicate_report["issue_codes"]
    assert trailing_report["accepted"] is False
    assert "tool_arguments_invalid_json" in trailing_report["issue_codes"]
    assert repeated_report["accepted"] is False
    assert "expected_one_tool_call_at_index_zero" in repeated_report["issue_codes"]
    assert "tool_arguments" not in valid_report


def test_stream_collection_reassembles_fragments_and_flags_data_after_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = gate._canonical_json(gate.expected_facts())
    lines = _semantic_lines(content, VALID_REASONING) + (_sse_line({"choices": []}),)
    monkeypatch.setattr(
        gate.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(lines=lines),
    )

    capture = gate.collect_stream(
        url="http://127.0.0.1:30010/v1/chat/completions",
        payload={"stream": True},
        timeout_seconds=5.0,
    )

    assert capture.content == content
    assert capture.reasoning_content == VALID_REASONING
    assert capture.completion_tokens == 48
    assert "stream_data_after_done" in capture.protocol_issue_codes


def test_run_gate_flushes_every_request_and_receipt_never_contains_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_urlopen = FakeUrlOpen()
    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)

    report = gate.run_gate(
        completion_url="http://127.0.0.1:30010/v1/chat/completions",
        flush_url="http://127.0.0.1:30010/flush_cache",
        model="deepseek-v4-flash",
        agents_context="repository policy line\n" * 400,
        repetitions=2,
        maximum_tokens=512,
        tool_maximum_tokens=128,
        timeout_seconds=5.0,
        flush_timeout_seconds=2.0,
        validate_tool_call=True,
    )
    serialized_report = gate._canonical_json(report)

    assert report["coherent"] is True
    assert fake_urlopen.paths == [
        "/flush_cache",
        "/v1/chat/completions",
        "/flush_cache",
        "/v1/chat/completions",
        "/flush_cache",
        "/v1/chat/completions",
    ]
    assert len(cast(list[object], report["flushes"])) == 3
    assert all(payload["stream"] is True for payload in fake_urlopen.payloads)
    assert (
        len(
            cast(list[dict[str, str]], fake_urlopen.payloads[0]["messages"])[0][
                "content"
            ].encode("utf-8")
        )
        >= 40_000
    )
    assert VALID_REASONING not in serialized_report
    assert gate._canonical_json(gate.expected_facts()) not in serialized_report


def test_flush_cache_retries_transient_scheduler_busy_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_urlopen = FakeUrlOpen(busy_flushes=2)
    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gate.time, "sleep", lambda unused_seconds: None)

    report = gate.flush_cache(
        "http://127.0.0.1:30010/flush_cache",
        timeout_seconds=2.0,
    )

    assert report["http_status"] == 200
    assert report["busy_retries"] == 2
    assert fake_urlopen.paths == ["/flush_cache"] * 3


def test_main_failure_json_does_not_leak_corrupt_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_garbage = "PRIVATE_GENERATED_TRAILING_GARBAGE"
    fake_urlopen = FakeUrlOpen(corrupt_content=private_garbage)
    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text("repository policy line\n" * 50, encoding="utf-8")

    exit_code = gate.main(
        [
            "--agents-path",
            str(agents_path),
            "--repetitions",
            "2",
            "--timeout",
            "5",
            "--flush-timeout",
            "2",
        ]
    )
    output = capsys.readouterr().out
    parsed_output = json.loads(output)

    assert exit_code == 1
    assert parsed_output["coherent"] is False
    assert private_garbage not in output
    assert VALID_REASONING not in output
