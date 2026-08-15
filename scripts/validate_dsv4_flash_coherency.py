#!/usr/bin/env python3
"""Strict OpenCode-path semantic validation for DeepSeek V4 Flash.

The gate deliberately exercises the streaming OpenAI chat endpoint with a
repository-sized system prompt and thinking enabled.  It never writes model
text, reasoning text, or tool arguments to stdout: receipts contain only
lengths, hashes, timing, and stable validation issue codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

DEFAULT_COMPLETION_URL = "http://127.0.0.1:30010/v1/chat/completions"
DEFAULT_MODEL_PATH = Path("/tmp/dsv4-local-checkpoint-0731")
DEFAULT_AGENTS_PATH = Path(__file__).resolve().parents[1] / "AGENTS.md"
TOOL_NAME = "record_coherence_decision"
FOLLOWUP_RECEIPT = "coherence-followup-7f3a"
FOLLOWUP_FINAL_CONTENT = f"FOLLOWUP_COMPLETE:{FOLLOWUP_RECEIPT}"
MINIMUM_REASONING_BYTES = 24
CACHE_FLUSH_RETRY_INTERVAL_SECONDS = 0.1


class _HttpResponse(Protocol):
    """The small typed surface used from urllib's dynamically typed response."""

    status: int

    def read(self) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class ParsedArguments(argparse.Namespace):
    """Statically typed command-line arguments populated by argparse."""

    url: str
    flush_url: str | None
    model_path: Path
    agents_path: Path
    repetitions: int
    max_new_tokens: int
    tool_max_new_tokens: int
    timeout: float
    flush_timeout: float
    validate_tool_call: bool
    validate_followups: bool
    flush_cache_between_runs: bool


def _open_url(request: urllib.request.Request, timeout_seconds: float) -> _HttpResponse:
    return cast(
        _HttpResponse,
        urllib.request.urlopen(request, timeout=timeout_seconds),
    )


class GateError(RuntimeError):
    """A safe-to-report harness failure that never retains generated text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


COHERENCE_FACTS = (
    ("amber", "MAPLE"),
    ("cobalt", "RIVER"),
    ("jade", "ORBIT"),
    ("violet", "CEDAR"),
)


@dataclass(slots=True)
class ToolCallParts:
    identifier: str | None = None
    kind: str | None = None
    name_fragments: list[str] = field(default_factory=list)
    argument_fragments: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StreamCapture:
    http_status: int
    elapsed_seconds: float
    time_to_first_output_seconds: float | None
    content: str
    reasoning_content: str
    finish_reason: str | None
    tool_calls: dict[int, ToolCallParts]
    completion_tokens: int | None
    event_count: int
    saw_done: bool
    protocol_issue_codes: tuple[str, ...]


def expected_facts() -> dict[str, object]:
    return dict(COHERENCE_FACTS)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _workspace_inventory_rows() -> tuple[str, ...]:
    """Create deterministic, realistic passive context spanning prefill chunks."""

    areas = (
        "routing",
        "worker",
        "master",
        "events",
        "dashboard",
        "networking",
        "runtime",
        "telemetry",
    )
    checks = (
        "strict typing and focused async tests",
        "immutable event replay and ordering tests",
        "request-schema and cancellation tests",
        "resource ownership and cleanup tests",
    )
    return tuple(
        (
            f"{index:03d} src/exo/{areas[index % len(areas)]}/"
            f"component_{index:03d}.py is owned by team-{index % 11:02d}; "
            f"changes require {checks[index % len(checks)]}; its paired test is "
            f"src/exo/{areas[index % len(areas)]}/tests/"
            f"test_component_{index:03d}.py."
        )
        for index in range(192)
    )


def _render_fact(index: int) -> str:
    key, value = COHERENCE_FACTS[index]
    return f"<coherence_fact_{key}>{key}={value}</coherence_fact_{key}>"


def build_system_prompt(agents_context: str) -> str:
    """Build a realistic OpenCode system prompt without using private fixtures."""

    if not agents_context.strip():
        raise GateError("agents_context_empty")
    inventory = _workspace_inventory_rows()
    return (
        "You are a coding agent operating in a live repository through an "
        "OpenAI-compatible OpenCode client. Follow the repository instructions "
        "below. Treat them as authoritative context, do not quote or summarize "
        "them in your answer, and do not invent shell output. This request is a "
        "read-only inference health check: do not edit files, execute commands, "
        "or contact external systems. Use the enabled private thinking channel "
        "for the requested extraction, then obey the exact final-output "
        "contract.\n\n"
        "<repository_instructions>\n"
        f"{agents_context.rstrip()}\n"
        "</repository_instructions>\n\n"
        f"{_render_fact(0)}\n\n"
        "<passive_workspace_inventory_part_1>\n"
        "The following deterministic repository inventory is read-only context, "
        "similar in size and shape to file metadata supplied to a coding agent.\n"
        f"{'\n'.join(inventory[:64])}\n"
        "</passive_workspace_inventory_part_1>\n\n"
        f"{_render_fact(1)}\n\n"
        "<passive_workspace_inventory_part_2>\n"
        f"{'\n'.join(inventory[64:128])}\n"
        "</passive_workspace_inventory_part_2>\n\n"
        f"{_render_fact(2)}\n\n"
        "<passive_workspace_inventory_part_3>\n"
        f"{'\n'.join(inventory[128:])}\n"
        "</passive_workspace_inventory_part_3>\n\n"
        "For the health check, later user instructions about the facts and "
        "response shape are the active task. Repository guidance remains "
        "applicable where it does not conflict with that response shape."
    )


def build_extraction_task() -> str:
    return (
        "Copy four coherence facts without calculation, ranking, transformation, "
        "or inference. The amber, cobalt, and jade facts appear in three widely "
        "separated regions of the earlier system context. The fourth fact is here:\n"
        f"{_render_fact(3)}\n\n"
        "In private reasoning, briefly mention MAPLE, RIVER, ORBIT, and CEDAR once "
        "each to confirm that all four regions were read. Keep private reasoning "
        "below 40 words."
    )


def build_semantic_user_prompt() -> str:
    return (
        f"{build_extraction_task()}\n\n"
        "The final content must be one compact JSON object with keys in this "
        "exact order: amber, cobalt, jade, violet. Copy each corresponding value "
        "exactly. Emit no Markdown, prose, leading or trailing whitespace, or "
        "repeated answer."
    )


def build_tool_user_prompt() -> str:
    return (
        f"Copy the four coherence facts and call {TOOL_NAME} exactly once with "
        "those four key/value pairs. Do not emit prose.\n\n"
        f"{build_extraction_task()}"
    )


def semantic_payload(
    *,
    model: str,
    system_prompt: str,
    maximum_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_semantic_user_prompt()},
        ],
        "temperature": 0.0,
        "max_tokens": maximum_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": True},
    }


def tool_payload(
    *,
    model: str,
    system_prompt: str,
    maximum_tokens: int,
) -> dict[str, object]:
    facts = expected_facts()
    properties = {
        key: {"type": "string", "enum": [value]} for key, value in facts.items()
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_tool_user_prompt(),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": (
                        "Record the deterministic, side-effect-free coherence "
                        "facts. The harness does not execute this tool."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": maximum_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }


def followup_tool_payload(
    *,
    model: str,
    system_prompt: str,
    initial_assistant_content: str,
    maximum_tokens: int,
) -> dict[str, object]:
    """Build a realistic second-turn tool request retaining the first answer."""

    payload = tool_payload(
        model=model,
        system_prompt=system_prompt,
        maximum_tokens=maximum_tokens,
    )
    messages = cast(list[dict[str, object]], payload["messages"])
    initial_user = build_semantic_user_prompt()
    tool_user = cast(str, messages[-1]["content"])
    payload["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user},
        {"role": "assistant", "content": initial_assistant_content},
        {
            "role": "user",
            "content": (
                f"Follow up on the immediately preceding verified answer. {tool_user}"
            ),
        },
    ]
    return payload


def tool_result_followup_payload(
    *,
    model: str,
    system_prompt: str,
    initial_assistant_content: str,
    tool_capture: StreamCapture,
    maximum_tokens: int,
) -> dict[str, object]:
    """Continue the same conversation after injecting the validated tool result."""

    parts = tool_capture.tool_calls[0]
    assert parts.identifier is not None
    tool_name = "".join(parts.name_fragments)
    arguments = "".join(parts.argument_fragments)
    tool_result = _canonical_json({"accepted": True, "receipt": FOLLOWUP_RECEIPT})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_semantic_user_prompt()},
            {"role": "assistant", "content": initial_assistant_content},
            {
                "role": "user",
                "content": (
                    "Follow up on the immediately preceding verified answer. "
                    f"{build_tool_user_prompt()}"
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": parts.identifier,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": parts.identifier,
                "name": tool_name,
                "content": tool_result,
            },
            {
                "role": "user",
                "content": (
                    "Acknowledge the tool result by emitting exactly "
                    f"{FOLLOWUP_FINAL_CONTENT} with no Markdown, reasoning, "
                    "whitespace, or other text."
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": maximum_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }


def derive_flush_url(completion_url: str) -> str:
    parsed_url = urllib.parse.urlsplit(completion_url)
    return urllib.parse.urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, "/flush_cache", "timeout=30", "")
    )


def flush_cache(url: str, timeout_seconds: float) -> dict[str, object]:
    started_at = time.perf_counter()
    deadline = started_at + timeout_seconds
    busy_retries = 0
    while True:
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0:
            raise GateError("cache_flush_busy_timeout")
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _open_url(request, remaining_seconds) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            # SGLang rejects a flush with HTTP 400 while the just-completed
            # streaming request is still retiring in the scheduler. Retry only
            # that transient status; authentication, routing, and server errors
            # remain immediate hard failures.
            if error.code != 400:
                raise
            error.read()
            busy_retries += 1
        else:
            if status == 200:
                break
            if status != 400:
                raise GateError("cache_flush_non_200")
            busy_retries += 1

        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0:
            raise GateError("cache_flush_busy_timeout")
        time.sleep(min(CACHE_FLUSH_RETRY_INTERVAL_SECONDS, remaining_seconds))

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "http_status": status,
        "busy_retries": busy_retries,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "response_bytes": len(body),
        "response_sha256": hashlib.sha256(body).hexdigest(),
    }


def _iter_sse_data(response: Iterable[bytes]) -> Iterable[bytes]:
    for raw_line in response:
        line = raw_line.rstrip(b"\r\n")
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            raise GateError("stream_non_sse_line")
        yield line.removeprefix(b"data:").lstrip(b" ")


def _decode_event(encoded_event: bytes) -> dict[str, object]:
    try:
        value = cast(object, json.loads(encoded_event))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError("stream_event_invalid_json") from error
    if not isinstance(value, dict):
        raise GateError("stream_event_not_object")
    return cast(dict[str, object], value)


def _consume_tool_calls(
    raw_tool_calls: object,
    tool_calls: dict[int, ToolCallParts],
    issue_codes: set[str],
) -> None:
    if raw_tool_calls is None:
        return
    if not isinstance(raw_tool_calls, list):
        issue_codes.add("tool_calls_not_list")
        return
    for raw_call_value in cast(list[object], raw_tool_calls):
        if not isinstance(raw_call_value, dict):
            issue_codes.add("tool_call_not_object")
            continue
        raw_call = cast(dict[str, object], raw_call_value)
        raw_index = raw_call.get("index", 0)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            issue_codes.add("tool_call_index_invalid")
            continue
        parts = tool_calls.setdefault(raw_index, ToolCallParts())
        raw_identifier = raw_call.get("id")
        if raw_identifier is not None:
            if not isinstance(raw_identifier, str) or not raw_identifier:
                issue_codes.add("tool_call_id_invalid")
            elif parts.identifier not in (None, raw_identifier):
                issue_codes.add("tool_call_id_changed")
            else:
                parts.identifier = raw_identifier
        raw_kind = raw_call.get("type")
        if raw_kind is not None:
            if not isinstance(raw_kind, str):
                issue_codes.add("tool_call_type_not_string")
            elif parts.kind not in (None, raw_kind):
                issue_codes.add("tool_call_type_changed")
            else:
                parts.kind = raw_kind
        raw_function_value = raw_call.get("function")
        if raw_function_value is None:
            continue
        if not isinstance(raw_function_value, dict):
            issue_codes.add("tool_function_not_object")
            continue
        raw_function = cast(dict[str, object], raw_function_value)
        raw_name = raw_function.get("name")
        if isinstance(raw_name, str):
            parts.name_fragments.append(raw_name)
        elif raw_name is not None:
            issue_codes.add("tool_name_not_string")
        raw_arguments = raw_function.get("arguments")
        if isinstance(raw_arguments, str):
            parts.argument_fragments.append(raw_arguments)
        elif raw_arguments is not None:
            issue_codes.add("tool_arguments_not_string")


def collect_stream(
    *,
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
) -> StreamCapture:
    request_body = _canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    first_output_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, ToolCallParts] = {}
    issue_codes: set[str] = set()
    finish_reason: str | None = None
    completion_tokens: int | None = None
    event_count = 0
    saw_done = False

    with _open_url(request, timeout_seconds) as response:
        status = response.status
        if status != 200:
            raise GateError("chat_completion_non_200")
        for encoded_event in _iter_sse_data(response):
            received_at = time.perf_counter()
            if saw_done:
                issue_codes.add("stream_data_after_done")
                continue
            if encoded_event == b"[DONE]":
                saw_done = True
                continue
            event = _decode_event(encoded_event)
            event_count += 1
            raw_usage_value = event.get("usage")
            if isinstance(raw_usage_value, dict):
                raw_usage = cast(dict[str, object], raw_usage_value)
                raw_completion_tokens = raw_usage.get("completion_tokens")
                if isinstance(raw_completion_tokens, int) and not isinstance(
                    raw_completion_tokens, bool
                ):
                    completion_tokens = raw_completion_tokens
            raw_choices_value = event.get("choices")
            if not isinstance(raw_choices_value, list):
                issue_codes.add("choices_not_list")
                continue
            raw_choices = cast(list[object], raw_choices_value)
            if len(raw_choices) > 1:
                issue_codes.add("multiple_choices")
            for raw_choice_value in raw_choices:
                if not isinstance(raw_choice_value, dict):
                    issue_codes.add("choice_not_object")
                    continue
                raw_choice = cast(dict[str, object], raw_choice_value)
                if raw_choice.get("index") not in (None, 0):
                    issue_codes.add("choice_index_invalid")
                raw_finish_reason = raw_choice.get("finish_reason")
                if raw_finish_reason is not None:
                    if not isinstance(raw_finish_reason, str):
                        issue_codes.add("finish_reason_not_string")
                    elif finish_reason is not None:
                        issue_codes.add("finish_reason_repeated")
                    else:
                        finish_reason = raw_finish_reason
                raw_delta_value = raw_choice.get("delta")
                if not isinstance(raw_delta_value, dict):
                    issue_codes.add("delta_not_object")
                    continue
                raw_delta = cast(dict[str, object], raw_delta_value)
                for key, destination in (
                    ("content", content_parts),
                    ("reasoning_content", reasoning_parts),
                ):
                    raw_fragment = raw_delta.get(key)
                    if isinstance(raw_fragment, str):
                        if raw_fragment and first_output_at is None:
                            first_output_at = received_at
                        destination.append(raw_fragment)
                    elif raw_fragment is not None:
                        issue_codes.add(f"{key}_not_string")
                raw_tool_calls: object = raw_delta.get("tool_calls")
                if (
                    isinstance(raw_tool_calls, list)
                    and raw_tool_calls
                    and first_output_at is None
                ):
                    first_output_at = received_at
                _consume_tool_calls(
                    cast(object, raw_tool_calls), tool_calls, issue_codes
                )

    elapsed_seconds = time.perf_counter() - started_at
    if not saw_done:
        issue_codes.add("done_event_missing")
    return StreamCapture(
        http_status=status,
        elapsed_seconds=elapsed_seconds,
        time_to_first_output_seconds=(
            None if first_output_at is None else first_output_at - started_at
        ),
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        completion_tokens=completion_tokens,
        event_count=event_count,
        saw_done=saw_done,
        protocol_issue_codes=tuple(sorted(issue_codes)),
    )


def _base_capture_report(capture: StreamCapture) -> dict[str, object]:
    content_bytes = capture.content.encode("utf-8")
    reasoning_bytes = capture.reasoning_content.encode("utf-8")
    return {
        "http_status": capture.http_status,
        "elapsed_seconds": round(capture.elapsed_seconds, 6),
        "time_to_first_output_seconds": (
            None
            if capture.time_to_first_output_seconds is None
            else round(capture.time_to_first_output_seconds, 6)
        ),
        "completion_tokens": capture.completion_tokens,
        "event_count": capture.event_count,
        "finish_reason": capture.finish_reason,
        "saw_done": capture.saw_done,
        "content_bytes": len(content_bytes),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "reasoning_bytes": len(reasoning_bytes),
        "reasoning_sha256": hashlib.sha256(reasoning_bytes).hexdigest(),
    }


def validate_semantic_capture(capture: StreamCapture) -> dict[str, object]:
    expected = _canonical_json(expected_facts())
    issue_codes = set(capture.protocol_issue_codes)
    if capture.content != expected:
        if expected in capture.content:
            issue_codes.add("semantic_content_not_exact")
        else:
            issue_codes.add("semantic_content_mismatch")
    if capture.finish_reason != "stop":
        issue_codes.add("semantic_finish_reason_not_stop")
    if capture.tool_calls:
        issue_codes.add("semantic_unexpected_tool_call")
    reasoning_bytes = capture.reasoning_content.encode("utf-8")
    if len(reasoning_bytes) < MINIMUM_REASONING_BYTES:
        issue_codes.add("reasoning_too_short")
    reasoning_lower = capture.reasoning_content.lower()
    reasoning_evidence = tuple(value for _, value in COHERENCE_FACTS)
    if any(piece.lower() not in reasoning_lower for piece in reasoning_evidence):
        issue_codes.add("reasoning_missing_semantic_evidence")
    if "\ufffd" in capture.content or "\ufffd" in capture.reasoning_content:
        issue_codes.add("unicode_replacement_character_present")
    report = _base_capture_report(capture)
    report.update(
        {
            "accepted": not issue_codes,
            "issue_codes": sorted(issue_codes),
            "expected_content_sha256": _sha256_text(expected),
        }
    )
    return report


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _reject_non_finite_constant(unused: str) -> object:
    del unused
    raise ValueError


def validate_tool_capture(capture: StreamCapture) -> dict[str, object]:
    issue_codes = set(capture.protocol_issue_codes)
    arguments_text = ""
    if capture.content:
        issue_codes.add("tool_unexpected_content")
    if capture.reasoning_content:
        issue_codes.add("tool_unexpected_reasoning")
    if capture.finish_reason != "tool_calls":
        issue_codes.add("tool_finish_reason_not_tool_calls")
    if set(capture.tool_calls) != {0}:
        issue_codes.add("expected_one_tool_call_at_index_zero")
    else:
        parts = capture.tool_calls[0]
        if not parts.identifier:
            issue_codes.add("tool_call_id_missing")
        if parts.kind != "function":
            issue_codes.add("tool_call_type_invalid")
        if "".join(parts.name_fragments) != TOOL_NAME:
            issue_codes.add("tool_name_invalid")
        arguments_text = "".join(parts.argument_fragments)
        try:
            arguments = cast(
                object,
                json.loads(
                    arguments_text,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_non_finite_constant,
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            issue_codes.add("tool_arguments_invalid_json")
        else:
            if _canonical_json(arguments) != _canonical_json(expected_facts()):
                issue_codes.add("tool_arguments_semantic_mismatch")
    report = _base_capture_report(capture)
    arguments_bytes = arguments_text.encode("utf-8")
    report.update(
        {
            "accepted": not issue_codes,
            "issue_codes": sorted(issue_codes),
            "tool_arguments_bytes": len(arguments_bytes),
            "tool_arguments_sha256": hashlib.sha256(arguments_bytes).hexdigest(),
            "expected_arguments_sha256": _sha256_text(
                _canonical_json(expected_facts())
            ),
        }
    )
    return report


def validate_tool_result_followup_capture(
    capture: StreamCapture,
) -> dict[str, object]:
    """Require an exact normal assistant continuation after a tool result."""

    issue_codes = set(capture.protocol_issue_codes)
    if capture.content != FOLLOWUP_FINAL_CONTENT:
        if FOLLOWUP_FINAL_CONTENT in capture.content:
            issue_codes.add("followup_content_not_exact")
        else:
            issue_codes.add("followup_content_mismatch")
    if capture.finish_reason != "stop":
        issue_codes.add("followup_finish_reason_not_stop")
    if capture.reasoning_content:
        issue_codes.add("followup_unexpected_reasoning")
    if capture.tool_calls:
        issue_codes.add("followup_unexpected_tool_call")
    if "\ufffd" in capture.content or "\ufffd" in capture.reasoning_content:
        issue_codes.add("unicode_replacement_character_present")
    report = _base_capture_report(capture)
    report.update(
        {
            "accepted": not issue_codes,
            "issue_codes": sorted(issue_codes),
            "expected_content_sha256": _sha256_text(FOLLOWUP_FINAL_CONTENT),
        }
    )
    return report


def run_followup_sequence(
    *,
    completion_url: str,
    flush_url: str,
    model: str,
    system_prompt: str,
    semantic_maximum_tokens: int,
    tool_maximum_tokens: int,
    timeout_seconds: float,
    flush_timeout_seconds: float,
) -> dict[str, object]:
    """Run initial answer -> tool follow-up -> tool-result continuation."""

    flush_report = flush_cache(flush_url, flush_timeout_seconds)
    initial_payload = semantic_payload(
        model=model,
        system_prompt=system_prompt,
        maximum_tokens=semantic_maximum_tokens,
    )
    initial_capture = collect_stream(
        url=completion_url,
        payload=initial_payload,
        timeout_seconds=timeout_seconds,
    )
    initial_report = validate_semantic_capture(initial_capture)
    report: dict[str, object] = {
        "accepted": False,
        "cache_policy": "single_flush_before_sequence",
        "flush": flush_report,
        "initial_semantic": initial_report,
        "tool_followup": None,
        "tool_result_followup": None,
        "request_sha256": {
            "initial": _sha256_text(_canonical_json(initial_payload)),
            "tool_followup": None,
            "tool_result_followup": None,
        },
    }
    if initial_report["accepted"] is not True:
        return report

    tool_followup_payload = followup_tool_payload(
        model=model,
        system_prompt=system_prompt,
        initial_assistant_content=initial_capture.content,
        maximum_tokens=tool_maximum_tokens,
    )
    tool_capture = collect_stream(
        url=completion_url,
        payload=tool_followup_payload,
        timeout_seconds=timeout_seconds,
    )
    tool_report = validate_tool_capture(tool_capture)
    report["tool_followup"] = tool_report
    request_hashes = cast(dict[str, object], report["request_sha256"])
    request_hashes["tool_followup"] = _sha256_text(
        _canonical_json(tool_followup_payload)
    )
    if tool_report["accepted"] is not True:
        return report

    final_payload = tool_result_followup_payload(
        model=model,
        system_prompt=system_prompt,
        initial_assistant_content=initial_capture.content,
        tool_capture=tool_capture,
        maximum_tokens=tool_maximum_tokens,
    )
    final_capture = collect_stream(
        url=completion_url,
        payload=final_payload,
        timeout_seconds=timeout_seconds,
    )
    final_report = validate_tool_result_followup_capture(final_capture)
    report["tool_result_followup"] = final_report
    request_hashes["tool_result_followup"] = _sha256_text(
        _canonical_json(final_payload)
    )
    report["accepted"] = final_report["accepted"] is True
    return report


def run_gate(
    *,
    completion_url: str,
    flush_url: str,
    model: str,
    agents_context: str,
    repetitions: int,
    maximum_tokens: int,
    tool_maximum_tokens: int,
    timeout_seconds: float,
    flush_timeout_seconds: float,
    validate_tool_call: bool,
    validate_followups: bool = False,
) -> dict[str, object]:
    if repetitions < 2:
        raise GateError("repetitions_below_two")
    if maximum_tokens < 1 or tool_maximum_tokens < 1:
        raise GateError("maximum_tokens_not_positive")
    if timeout_seconds <= 0 or flush_timeout_seconds <= 0:
        raise GateError("timeout_not_positive")
    system_prompt = build_system_prompt(agents_context)
    semantic_request = semantic_payload(
        model=model,
        system_prompt=system_prompt,
        maximum_tokens=maximum_tokens,
    )
    semantic_runs: list[dict[str, object]] = []
    flushes: list[dict[str, object]] = []
    for _ in range(repetitions):
        flushes.append(flush_cache(flush_url, flush_timeout_seconds))
        semantic_runs.append(
            validate_semantic_capture(
                collect_stream(
                    url=completion_url,
                    payload=semantic_request,
                    timeout_seconds=timeout_seconds,
                )
            )
        )

    tool_report: dict[str, object] | None = None
    if validate_tool_call:
        flushes.append(flush_cache(flush_url, flush_timeout_seconds))
        tool_report = validate_tool_capture(
            collect_stream(
                url=completion_url,
                payload=tool_payload(
                    model=model,
                    system_prompt=system_prompt,
                    maximum_tokens=tool_maximum_tokens,
                ),
                timeout_seconds=timeout_seconds,
            )
        )

    followup_report: dict[str, object] | None = None
    if validate_followups:
        followup_report = run_followup_sequence(
            completion_url=completion_url,
            flush_url=flush_url,
            model=model,
            system_prompt=system_prompt,
            semantic_maximum_tokens=maximum_tokens,
            tool_maximum_tokens=tool_maximum_tokens,
            timeout_seconds=timeout_seconds,
            flush_timeout_seconds=flush_timeout_seconds,
        )

    content_hashes = {cast(str, run["content_sha256"]) for run in semantic_runs}
    deterministic = len(content_hashes) == 1
    semantics_accepted = all(cast(bool, run["accepted"]) for run in semantic_runs)
    tool_accepted = tool_report is None or cast(bool, tool_report["accepted"])
    followups_accepted = followup_report is None or cast(
        bool, followup_report["accepted"]
    )
    agents_bytes = agents_context.encode("utf-8")
    system_bytes = system_prompt.encode("utf-8")
    return {
        "schema_version": 2,
        "coherent": (
            semantics_accepted
            and deterministic
            and tool_accepted
            and followups_accepted
        ),
        "deterministic_final_content": deterministic,
        "cache_policy": "flush_before_every_request",
        "challenge": {
            "agents_context_bytes": len(agents_bytes),
            "agents_context_sha256": hashlib.sha256(agents_bytes).hexdigest(),
            "system_prompt_bytes": len(system_bytes),
            "system_prompt_sha256": hashlib.sha256(system_bytes).hexdigest(),
            "semantic_request_sha256": _sha256_text(_canonical_json(semantic_request)),
            "expected_facts_sha256": _sha256_text(_canonical_json(expected_facts())),
            "thinking_required": True,
        },
        "flushes": flushes,
        "semantic_runs": semantic_runs,
        "forced_tool_call": tool_report,
        "opencode_followup_sequence": followup_report,
    }


def parse_args(argv: Sequence[str] | None = None) -> ParsedArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate the DSV4 OpenCode streaming, thinking, and optional "
            "tool-call paths without printing model-generated text."
        )
    )
    parser.add_argument("--url", default=DEFAULT_COMPLETION_URL)
    parser.add_argument("--flush-url")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--agents-path", type=Path, default=DEFAULT_AGENTS_PATH)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tool-max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--flush-timeout", type=float, default=60.0)
    parser.add_argument("--validate-tool-call", action="store_true")
    parser.add_argument(
        "--validate-followups",
        action="store_true",
        help=(
            "run an OpenCode-shaped multi-turn tool call and tool-result "
            "continuation after the repeated basic gate"
        ),
    )
    parser.add_argument(
        "--flush-cache-between-runs",
        action="store_true",
        help=(
            "Compatibility flag; strict mode always flushes before every request, "
            "including the first."
        ),
    )
    return parser.parse_args(argv, namespace=ParsedArguments())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        agents_context = args.agents_path.read_text(encoding="utf-8")
        report = run_gate(
            completion_url=args.url,
            flush_url=args.flush_url or derive_flush_url(args.url),
            model=str(args.model_path),
            agents_context=agents_context,
            repetitions=args.repetitions,
            maximum_tokens=args.max_new_tokens,
            tool_maximum_tokens=args.tool_max_new_tokens,
            timeout_seconds=args.timeout,
            flush_timeout_seconds=args.flush_timeout,
            validate_tool_call=args.validate_tool_call,
            validate_followups=args.validate_followups,
        )
    except (OSError, urllib.error.URLError, GateError) as error:
        failure: dict[str, object] = {
            "schema_version": 2,
            "coherent": False,
            "error_type": type(error).__name__,
        }
        if isinstance(error, GateError):
            failure["error_code"] = error.code
        print(_canonical_json(failure))
        return 1
    print(_canonical_json(report))
    return 0 if cast(bool, report["coherent"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
