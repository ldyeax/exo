from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def _load_bridge() -> ModuleType:
    source = Path(__file__).parents[1] / "ollama_openai_bridge.py"
    specification = importlib.util.spec_from_file_location(
        "qwen38_ollama_openai_bridge", source
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


bridge = _load_bridge()


def _configuration() -> Any:
    return bridge.BridgeConfiguration(
        backend_url="http://127.0.0.1:30022/v1/chat/completions",
        model="Qwen3.8-27B-FP8",
        api_key=None,
        timeout_seconds=60.0,
        context_length=262_144,
        max_output_tokens=16_384,
        model_size_bytes=27_889_309_280,
    )


def test_request_translation_preserves_tools_and_history() -> None:
    request = bridge.build_openai_request(
        {
            "model": "Qwen3.8-27B-FP8",
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 512, "top_k": 20},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read one file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "Read README.md"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "README.md"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "read_file",
                    "content": "contents",
                },
            ],
        },
        _configuration(),
    )

    assistant_call = request["messages"][1]["tool_calls"][0]
    assert assistant_call["function"]["arguments"] == '{"path":"README.md"}'
    assert request["messages"][2]["tool_call_id"] == assistant_call["id"]
    assert request["tools"][0]["function"]["name"] == "read_file"
    assert request["tool_choice"] == "auto"
    assert request["reasoning_effort"] == "none"
    assert request["max_tokens"] == 512
    assert request["top_k"] == 20


def test_nonstream_response_translation_preserves_reasoning_and_tool_call() -> None:
    response = bridge.convert_openai_response(
        {
            "model": "Qwen3.8-27B-FP8",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should inspect it.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        },
        elapsed_nanoseconds=123,
    )

    assert response["message"]["content"] == ""
    assert response["message"]["thinking"] == "I should inspect it."
    assert response["message"]["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {
            "index": 0,
            "name": "read_file",
            "arguments": {"path": "README.md"},
        },
    }
    assert response["done_reason"] == "tool_calls"
    assert response["prompt_eval_count"] == 40
    assert response["eval_count"] == 12


def test_stream_translation_aggregates_partial_tool_arguments() -> None:
    conversion = bridge.StreamingConversion(
        model="Qwen3.8-27B-FP8", started_nanoseconds=0
    )
    assert (
        conversion.consume(
            {
                "model": "Qwen3.8-27B-FP8",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_2",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        == []
    )
    assert (
        conversion.consume(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"README.md"}'}}
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 51, "completion_tokens": 9},
            }
        )
        == []
    )

    events = conversion.finish()
    assert events[0]["message"]["tool_calls"][0]["function"] == {
        "index": 0,
        "name": "read_file",
        "arguments": {"path": "README.md"},
    }
    assert events[-1]["done"] is True
    assert events[-1]["done_reason"] == "tool_calls"
    assert events[-1]["prompt_eval_count"] == 51
    assert events[-1]["eval_count"] == 9


def test_sse_decoder_handles_comments_multiline_data_and_done() -> None:
    response = io.BytesIO(b': ping\ndata: {\ndata: "choices": []}\n\ndata: [DONE]\n\n')
    assert list(bridge.iter_sse_documents(response)) == [{"choices": []}, None]


def test_discovery_advertises_agent_capabilities_and_full_context() -> None:
    configuration = _configuration()
    shown = bridge.show_document(configuration)
    assert shown["capabilities"] == ["completion", "tools", "thinking"]
    assert shown["model_info"]["qwen3_5.context_length"] == 262_144

    tags = bridge.tags_document(configuration)
    assert tags["models"][0]["name"] == "Qwen3.8-27B-FP8"
    assert tags["models"][0]["size"] == 27_889_309_280


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(bridge.BridgeError, match="not available") as raised:
        bridge.build_openai_request(
            {
                "model": "not-the-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            _configuration(),
        )
    assert cast(Any, raised.value).status == 404
