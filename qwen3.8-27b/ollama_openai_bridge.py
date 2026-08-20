#!/usr/bin/env python3
"""Expose an Ollama-compatible coding endpoint backed by SGLang OpenAI chat."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypeAlias, cast
from urllib.parse import urlparse

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


class BridgeError(RuntimeError):
    """An error that can be returned to the Ollama client."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BridgeConfiguration:
    backend_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    context_length: int
    max_output_tokens: int
    model_size_bytes: int


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _object(value: object, description: str) -> JsonObject:
    if not isinstance(value, dict):
        raise BridgeError(HTTPStatus.BAD_REQUEST, f"{description} must be an object")
    return cast(JsonObject, value)


def _array(value: object, description: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise BridgeError(HTTPStatus.BAD_REQUEST, f"{description} must be an array")
    return cast(list[JsonValue], value)


def _string(value: object, description: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "nonempty string" if not allow_empty else "string"
        raise BridgeError(HTTPStatus.BAD_REQUEST, f"{description} must be a {suffix}")
    return value


def _tool_call_identifier(index: int, name: str) -> str:
    digest = hashlib.sha256(f"{index}:{name}".encode()).hexdigest()[:20]
    return f"call_{digest}"


def _json_arguments(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, separators=(",", ":"))


def _json_array(values: Sequence[JsonValue]) -> list[JsonValue]:
    return list(values)


def _openai_messages(messages_value: object) -> list[JsonObject]:
    messages = _array(messages_value, "messages")
    converted: list[JsonObject] = []
    pending_calls_by_name: dict[str, list[str]] = {}
    call_number = 0

    for message_number, raw_message in enumerate(messages):
        message = _object(raw_message, f"messages[{message_number}]")
        role = _string(message.get("role"), f"messages[{message_number}].role")
        content = message.get("content", "")
        if not isinstance(content, (str, list)):
            raise BridgeError(
                HTTPStatus.BAD_REQUEST,
                f"messages[{message_number}].content must be a string or content array",
            )

        converted_message: JsonObject = {"role": role, "content": content}
        raw_tool_calls = message.get("tool_calls")
        if role == "assistant" and raw_tool_calls is not None:
            tool_calls = _array(
                raw_tool_calls, f"messages[{message_number}].tool_calls"
            )
            converted_calls: list[JsonObject] = []
            for tool_number, raw_tool_call in enumerate(tool_calls):
                tool_call = _object(
                    raw_tool_call,
                    f"messages[{message_number}].tool_calls[{tool_number}]",
                )
                function = _object(
                    tool_call.get("function"),
                    f"messages[{message_number}].tool_calls[{tool_number}].function",
                )
                name = _string(
                    function.get("name"),
                    (
                        f"messages[{message_number}].tool_calls[{tool_number}]"
                        ".function.name"
                    ),
                    allow_empty=False,
                )
                identifier = tool_call.get("id")
                if not isinstance(identifier, str) or not identifier:
                    identifier = _tool_call_identifier(call_number, name)
                call_number += 1
                pending_calls_by_name.setdefault(name, []).append(identifier)
                converted_calls.append(
                    {
                        "id": identifier,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _json_arguments(function.get("arguments", {})),
                        },
                    }
                )
            converted_message["tool_calls"] = _json_array(converted_calls)

        if role == "tool":
            identifier = message.get("tool_call_id")
            tool_name = message.get("tool_name")
            if not isinstance(identifier, str) or not identifier:
                if isinstance(tool_name, str) and pending_calls_by_name.get(tool_name):
                    identifier = pending_calls_by_name[tool_name].pop(0)
                else:
                    identifier = _tool_call_identifier(
                        call_number, str(tool_name or "tool")
                    )
                    call_number += 1
            converted_message["tool_call_id"] = identifier
            if isinstance(tool_name, str) and tool_name:
                converted_message["name"] = tool_name

        converted.append(converted_message)

    return converted


def build_openai_request(
    ollama_request: Mapping[str, JsonValue], configuration: BridgeConfiguration
) -> JsonObject:
    """Translate an Ollama chat request to SGLang's OpenAI-compatible schema."""

    requested_model = ollama_request.get("model", configuration.model)
    if requested_model != configuration.model:
        raise BridgeError(
            HTTPStatus.NOT_FOUND,
            f"model {requested_model!r} is not available; use {configuration.model!r}",
        )

    options_value = ollama_request.get("options", {})
    options = _object(options_value, "options") if options_value is not None else {}
    stream = ollama_request.get("stream", True)
    if not isinstance(stream, bool):
        raise BridgeError(HTTPStatus.BAD_REQUEST, "stream must be a boolean")

    document: JsonObject = {
        "model": configuration.model,
        "messages": _json_array(_openai_messages(ollama_request.get("messages"))),
        "stream": stream,
        "temperature": options.get("temperature", 0.2),
        "top_p": options.get("top_p", 0.95),
        "max_tokens": options.get("num_predict", configuration.max_output_tokens),
        "parallel_tool_calls": True,
    }

    option_names = {
        "top_k": "top_k",
        "min_p": "min_p",
        "seed": "seed",
        "stop": "stop",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repeat_penalty": "repetition_penalty",
    }
    for ollama_name, openai_name in option_names.items():
        if ollama_name in options:
            document[openai_name] = options[ollama_name]

    tools = ollama_request.get("tools")
    if tools is not None:
        document["tools"] = _array(tools, "tools")
        document["tool_choice"] = "auto"

    response_format = ollama_request.get("format")
    if response_format == "json":
        document["response_format"] = {"type": "json_object"}
    elif isinstance(response_format, dict):
        document["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_response",
                "strict": True,
                "schema": response_format,
            },
        }
    elif response_format is not None:
        raise BridgeError(
            HTTPStatus.BAD_REQUEST, "format must be 'json' or a JSON schema object"
        )

    think = ollama_request.get("think")
    if think is False:
        document["reasoning_effort"] = "none"
    elif think is True:
        document["reasoning_effort"] = "medium"
    elif isinstance(think, str):
        if think not in {"low", "medium", "high", "max"}:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST, f"unsupported think level {think!r}"
            )
        document["reasoning_effort"] = think

    if stream:
        document["stream_options"] = {"include_usage": True}

    return document


def _ollama_tool_calls(value: object) -> list[JsonObject]:
    if value is None:
        return []
    calls = _array(value, "OpenAI tool_calls")
    converted: list[JsonObject] = []
    for index, raw_call in enumerate(calls):
        call = _object(raw_call, f"OpenAI tool_calls[{index}]")
        function = _object(call.get("function"), f"OpenAI tool_calls[{index}].function")
        name = _string(
            function.get("name"),
            f"OpenAI tool_calls[{index}].function.name",
            allow_empty=False,
        )
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = cast(object, json.loads(raw_arguments))
            except json.JSONDecodeError as error:
                raise BridgeError(
                    HTTPStatus.BAD_GATEWAY,
                    f"backend returned invalid tool arguments for {name!r}: {error}",
                ) from error
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise BridgeError(
                HTTPStatus.BAD_GATEWAY,
                f"backend returned non-object tool arguments for {name!r}",
            )
        identifier = call.get("id")
        converted_call: JsonObject = {
            "type": "function",
            "function": {
                "index": index,
                "name": name,
                "arguments": arguments,
            },
        }
        if isinstance(identifier, str) and identifier:
            converted_call["id"] = identifier
        converted.append(converted_call)
    return converted


def convert_openai_response(
    openai_response: Mapping[str, JsonValue], *, elapsed_nanoseconds: int
) -> JsonObject:
    """Translate a nonstreaming OpenAI chat response to Ollama format."""

    choices = _array(openai_response.get("choices"), "backend choices")
    if not choices:
        raise BridgeError(HTTPStatus.BAD_GATEWAY, "backend response had no choices")
    choice = _object(choices[0], "backend choices[0]")
    message = _object(choice.get("message"), "backend choices[0].message")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise BridgeError(HTTPStatus.BAD_GATEWAY, "backend content was not text")

    ollama_message: JsonObject = {"role": "assistant", "content": content}
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        ollama_message["thinking"] = reasoning
    tool_calls = _ollama_tool_calls(message.get("tool_calls"))
    if tool_calls:
        ollama_message["tool_calls"] = _json_array(tool_calls)

    usage_value = openai_response.get("usage", {})
    usage = _object(usage_value, "backend usage") if usage_value is not None else {}
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "stop"

    return {
        "model": openai_response.get("model", ""),
        "created_at": utc_timestamp(),
        "message": ollama_message,
        "done": True,
        "done_reason": finish_reason,
        "total_duration": elapsed_nanoseconds,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def iter_sse_documents(response: Iterator[bytes]) -> Iterator[JsonObject | None]:
    """Yield JSON SSE documents; ``None`` is the OpenAI DONE sentinel."""

    data_lines: list[bytes] = []
    for raw_line in response:
        line = raw_line.rstrip(b"\r\n")
        if line:
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
            continue
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        data_lines.clear()
        if payload == b"[DONE]":
            yield None
            continue
        try:
            document = cast(object, json.loads(payload))
        except json.JSONDecodeError as error:
            raise BridgeError(
                HTTPStatus.BAD_GATEWAY, f"backend emitted invalid SSE JSON: {error}"
            ) from error
        yield _object(document, "backend SSE event")

    if data_lines:
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            yield None
        else:
            try:
                document = cast(object, json.loads(payload))
            except json.JSONDecodeError as error:
                raise BridgeError(
                    HTTPStatus.BAD_GATEWAY,
                    f"backend emitted invalid trailing SSE JSON: {error}",
                ) from error
            yield _object(document, "backend trailing SSE event")


@dataclass
class StreamingConversion:
    model: str
    started_nanoseconds: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: dict[int, JsonObject] = field(default_factory=dict)

    def consume(self, document: Mapping[str, JsonValue]) -> list[JsonObject]:
        model = document.get("model")
        if isinstance(model, str) and model:
            self.model = model
        usage_value = document.get("usage")
        if isinstance(usage_value, dict):
            usage = cast(JsonObject, usage_value)
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self.prompt_tokens = prompt_tokens
            if isinstance(completion_tokens, int):
                self.completion_tokens = completion_tokens

        events: list[JsonObject] = []
        choices_value = document.get("choices", [])
        if not isinstance(choices_value, list):
            raise BridgeError(
                HTTPStatus.BAD_GATEWAY, "backend stream choices was not an array"
            )
        for raw_choice in choices_value:
            choice = _object(raw_choice, "backend stream choice")
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                self.finish_reason = finish_reason
            delta_value = choice.get("delta", {})
            delta = _object(delta_value, "backend stream delta")
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            message: JsonObject = {"role": "assistant", "content": ""}
            has_output = False
            if isinstance(content, str) and content:
                message["content"] = content
                has_output = True
            if isinstance(reasoning, str) and reasoning:
                message["thinking"] = reasoning
                has_output = True
            if has_output:
                events.append(self._event(message, done=False))
            self._consume_tool_call_deltas(delta.get("tool_calls"))
        return events

    def _consume_tool_call_deltas(self, value: object) -> None:
        if value is None:
            return
        for fallback_index, raw_call in enumerate(_array(value, "stream tool_calls")):
            call = _object(raw_call, f"stream tool_calls[{fallback_index}]")
            index_value = call.get("index", fallback_index)
            index = index_value if isinstance(index_value, int) else fallback_index
            aggregate = self.tool_calls.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            identifier = call.get("id")
            if isinstance(identifier, str) and identifier:
                aggregate["id"] = identifier
            function_value = call.get("function")
            if not isinstance(function_value, dict):
                continue
            function = cast(JsonObject, function_value)
            aggregate_function = cast(JsonObject, aggregate["function"])
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str):
                aggregate_function["name"] = str(aggregate_function["name"]) + name
            if isinstance(arguments, str):
                aggregate_function["arguments"] = (
                    str(aggregate_function["arguments"]) + arguments
                )
            elif isinstance(arguments, dict):
                aggregate_function["arguments"] = json.dumps(
                    arguments, separators=(",", ":")
                )

    def finish(self) -> list[JsonObject]:
        events: list[JsonObject] = []
        if self.tool_calls:
            ordered_calls = [
                self.tool_calls[index] for index in sorted(self.tool_calls)
            ]
            events.append(
                self._event(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": _json_array(_ollama_tool_calls(ordered_calls)),
                    },
                    done=False,
                )
            )
        elapsed = time.monotonic_ns() - self.started_nanoseconds
        final = self._event(
            {"role": "assistant", "content": ""},
            done=True,
            done_reason=self.finish_reason,
        )
        final.update(
            {
                "total_duration": elapsed,
                "load_duration": 0,
                "prompt_eval_count": self.prompt_tokens,
                "eval_count": self.completion_tokens,
            }
        )
        events.append(final)
        return events

    def _event(
        self, message: JsonObject, *, done: bool, done_reason: str | None = None
    ) -> JsonObject:
        event: JsonObject = {
            "model": self.model,
            "created_at": utc_timestamp(),
            "message": message,
            "done": done,
        }
        if done_reason is not None:
            event["done_reason"] = done_reason
        return event


def model_digest(model: str) -> str:
    return "sha256:" + hashlib.sha256(model.encode()).hexdigest()


def tags_document(configuration: BridgeConfiguration) -> JsonObject:
    return {
        "models": [
            {
                "name": configuration.model,
                "model": configuration.model,
                "modified_at": utc_timestamp(),
                "size": configuration.model_size_bytes,
                "digest": model_digest(configuration.model),
                "details": {
                    "parent_model": "",
                    "format": "safetensors",
                    "family": "qwen3_5",
                    "families": ["qwen3_5"],
                    "parameter_size": "27B",
                    "quantization_level": "FP8_E4M3",
                },
            }
        ]
    }


def show_document(configuration: BridgeConfiguration) -> JsonObject:
    return {
        "license": "See the upstream Qwen model card",
        "modelfile": (
            f"FROM {configuration.model}\n"
            f"PARAMETER num_ctx {configuration.context_length}\n"
        ),
        "parameters": f"num_ctx {configuration.context_length}",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "safetensors",
            "family": "qwen3_5",
            "families": ["qwen3_5"],
            "parameter_size": "27B",
            "quantization_level": "FP8_E4M3",
        },
        "model_info": {
            "general.architecture": "qwen3_5",
            "general.name": configuration.model,
            "general.parameter_count": 27_000_000_000,
            "qwen3_5.context_length": configuration.context_length,
        },
        "capabilities": ["completion", "tools", "thinking"],
        "modified_at": utc_timestamp(),
    }


class OllamaBridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        configuration: BridgeConfiguration,
    ) -> None:
        self.configuration = configuration
        super().__init__(server_address, OllamaBridgeHandler)


class OllamaBridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "QwenOllamaBridge/1.0"

    @property
    def configuration(self) -> BridgeConfiguration:
        return cast(OllamaBridgeServer, self.server).configuration

    def do_HEAD(self) -> None:
        if self.path == "/":
            self._send_bytes(HTTPStatus.OK, b"", "text/plain; charset=utf-8")
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                b"Ollama is running",
                "text/plain; charset=utf-8",
            )
        elif self.path == "/api/version":
            self._send_json(HTTPStatus.OK, {"version": "0.12.0-sglang-bridge"})
        elif self.path == "/api/tags":
            self._send_json(HTTPStatus.OK, tags_document(self.configuration))
        elif self.path == "/api/ps":
            models = _array(
                tags_document(self.configuration)["models"], "discovery models"
            )
            running = dict(_object(models[0], "discovery model"))
            running["expires_at"] = (
                (datetime.now(UTC) + timedelta(days=3650))
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            running["size_vram"] = self.configuration.model_size_bytes
            running["context_length"] = self.configuration.context_length
            self._send_json(HTTPStatus.OK, {"models": [running]})
        elif self.path == "/health":
            self._proxy_health()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        try:
            document = self._read_json()
            if self.path == "/api/show":
                requested_model = document.get("model", self.configuration.model)
                if requested_model != self.configuration.model:
                    raise BridgeError(HTTPStatus.NOT_FOUND, "model not found")
                self._send_json(HTTPStatus.OK, show_document(self.configuration))
            elif self.path == "/api/chat":
                self._chat(document)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except BridgeError as error:
            self._send_error(error.status, str(error))
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_json(self) -> JsonObject:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise BridgeError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(length_header)
        except ValueError as error:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST, "invalid Content-Length"
            ) from error
        if length < 0 or length > 64 * 1024 * 1024:
            raise BridgeError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request is too large"
            )
        try:
            value = cast(object, json.loads(self.rfile.read(length)))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BridgeError(
                HTTPStatus.BAD_REQUEST, f"invalid JSON: {error}"
            ) from error
        return _object(value, "request body")

    def _backend_request(self, document: JsonObject) -> urllib.request.Request:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.configuration.api_key:
            headers["Authorization"] = f"Bearer {self.configuration.api_key}"
        return urllib.request.Request(
            self.configuration.backend_url,
            data=json.dumps(document, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )

    def _open_backend(self, document: JsonObject) -> HTTPResponse:
        try:
            return cast(
                HTTPResponse,
                urllib.request.urlopen(
                    self._backend_request(document),
                    timeout=self.configuration.timeout_seconds,
                ),
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise BridgeError(
                HTTPStatus.BAD_GATEWAY,
                f"SGLang returned HTTP {error.code}: {detail}",
            ) from error
        except urllib.error.URLError as error:
            raise BridgeError(
                HTTPStatus.BAD_GATEWAY, f"cannot reach SGLang: {error.reason}"
            ) from error

    def _chat(self, ollama_request: JsonObject) -> None:
        document = build_openai_request(ollama_request, self.configuration)
        if document["stream"] is True:
            self._stream_chat(document)
            return

        started = time.monotonic_ns()
        with self._open_backend(document) as response:
            try:
                backend_document = cast(object, json.load(response))
            except json.JSONDecodeError as error:
                raise BridgeError(
                    HTTPStatus.BAD_GATEWAY, f"backend returned invalid JSON: {error}"
                ) from error
        converted = convert_openai_response(
            _object(backend_document, "backend response"),
            elapsed_nanoseconds=time.monotonic_ns() - started,
        )
        self._send_json(HTTPStatus.OK, converted)

    def _stream_chat(self, document: JsonObject) -> None:
        response = self._open_backend(document)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        conversion = StreamingConversion(
            model=self.configuration.model,
            started_nanoseconds=time.monotonic_ns(),
        )
        saw_done = False
        try:
            with response:
                for backend_document in iter_sse_documents(response):
                    if backend_document is None:
                        saw_done = True
                        break
                    for event in conversion.consume(backend_document):
                        self._write_stream_event(event)
            if not saw_done:
                raise BridgeError(
                    HTTPStatus.BAD_GATEWAY, "backend stream ended before [DONE]"
                )
            for event in conversion.finish():
                self._write_stream_event(event)
        except BridgeError as error:
            self._write_stream_event({"error": str(error)})

    def _proxy_health(self) -> None:
        parsed = urlparse(self.configuration.backend_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        try:
            response = cast(
                HTTPResponse,
                urllib.request.urlopen(
                    health_url,
                    timeout=min(self.configuration.timeout_seconds, 5.0),
                ),
            )
            with response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            self._send_error(
                HTTPStatus.SERVICE_UNAVAILABLE, f"backend unhealthy: {error}"
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {"status": "ok", "model": self.configuration.model},
        )

    def _write_stream_event(self, document: JsonObject) -> None:
        self.wfile.write(json.dumps(document, separators=(",", ":")).encode() + b"\n")
        self.wfile.flush()

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _send_json(self, status: int, document: JsonObject) -> None:
        self._send_bytes(
            status,
            json.dumps(document, separators=(",", ":")).encode(),
            "application/json",
        )

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *arguments: object) -> None:  # noqa: A002
        sys.stderr.write(
            f"[{self.log_date_time_string()}] {self.client_address[0]} "
            f"{format % arguments}\n"
        )


@dataclass
class CommandLineArguments:
    listen_host: str = "127.0.0.1"
    listen_port: int = 11434
    backend_url: str = "http://127.0.0.1:30022/v1/chat/completions"
    model: str = "Qwen3.8-27B-FP8"
    api_key: str | None = None
    timeout_seconds: float = 3600.0
    context_length: int = 262_144
    max_output_tokens: int = 16_384
    model_size_bytes: int = 27_889_309_280


def parse_arguments(arguments: Sequence[str] | None = None) -> CommandLineArguments:
    parser = argparse.ArgumentParser(
        description="Expose SGLang OpenAI chat as a tool-capable Ollama endpoint."
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=11434)
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:30022/v1/chat/completions",
    )
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--context-length", type=int, default=262_144)
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--model-size-bytes", type=int, default=27_889_309_280)
    parsed = parser.parse_args(arguments, namespace=CommandLineArguments())
    if not 1 <= parsed.listen_port <= 65535:
        parser.error("--listen-port must be between 1 and 65535")
    if parsed.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if parsed.context_length <= 0 or parsed.max_output_tokens <= 0:
        parser.error("token limits must be positive")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    configuration = BridgeConfiguration(
        backend_url=parsed.backend_url,
        model=parsed.model,
        api_key=parsed.api_key,
        timeout_seconds=parsed.timeout_seconds,
        context_length=parsed.context_length,
        max_output_tokens=parsed.max_output_tokens,
        model_size_bytes=parsed.model_size_bytes,
    )
    server = OllamaBridgeServer((parsed.listen_host, parsed.listen_port), configuration)
    print(
        f"Ollama bridge listening on http://{parsed.listen_host}:{parsed.listen_port}; "
        f"backend={parsed.backend_url}; model={parsed.model}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
