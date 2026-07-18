from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.utils_mlx import (
    EXO_CHAT_TEMPLATE_DATE,
    apply_chat_template,
    chat_template_date_override,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        _messages: Sequence[object],
        **kwargs: object,
    ) -> str:
        self.call_kwargs = kwargs
        return "rendered"


def _task() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        input=[
            InputMessage(
                role="user",
                content=InputMessageContent("Give a deterministic response."),
            )
        ],
    )


def test_chat_template_date_override_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EXO_CHAT_TEMPLATE_DATE, "18 Jul 2026")
    tokenizer = RecordingTokenizer()

    typed_tokenizer = cast(TokenizerWrapper, cast(object, tokenizer))
    assert apply_chat_template(typed_tokenizer, _task()) == "rendered"

    assert tokenizer.call_kwargs is not None
    assert tokenizer.call_kwargs["date_string"] == "18 Jul 2026"


def test_chat_template_date_override_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EXO_CHAT_TEMPLATE_DATE, raising=False)
    tokenizer = RecordingTokenizer()

    typed_tokenizer = cast(TokenizerWrapper, cast(object, tokenizer))
    assert apply_chat_template(typed_tokenizer, _task()) == "rendered"

    assert tokenizer.call_kwargs is not None
    assert "date_string" not in tokenizer.call_kwargs


@pytest.mark.parametrize("value", ["", "line one\nline two", "x" * 129])
def test_chat_template_date_override_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(EXO_CHAT_TEMPLATE_DATE, value)

    with pytest.raises(ValueError, match=EXO_CHAT_TEMPLATE_DATE):
        chat_template_date_override()
