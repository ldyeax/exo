from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Self, cast
from urllib.request import Request

import pytest

from scripts import benchmark_dsv4_flash_128k as benchmark


def _valid_report() -> str:
    return (
        "AURORA-73 is the selected profile because the controlled evidence is both "
        "measurable and reviewable. The baseline delivered 46 tokens per second "
        "under the same prompt shape, launch policy, sampling seed, and hardware "
        "conditions used for the candidate. The optimized trial reached 73 tokens "
        "per second, so the measured improvement was 27 tokens per second without "
        "changing the question being answered. Reproducibility requires multiple "
        "cache-flushed trials, alternating run order, stable clocks, and a durable "
        "receipt that another operator can inspect. Thermal headroom matters because "
        "a brief cool run can exaggerate sustained decode speed while hiding later "
        "throttling or power contention. Semantic validation is equally important: "
        "a fast stream of copied filler, malformed symbols, or irrelevant prose is "
        "not a successful inference result. Reviewers should compare medians and "
        "tail behavior, preserve exact input token counts, and record natural stop "
        "conditions. They should also examine the final portion of the answer for "
        "repetition, since coherent opening text cannot excuse garbage produced "
        "after a terminal token. This combination separates genuine kernel and "
        "scheduling gains from cache effects, measurement noise, and corrupted "
        "generation. The resulting claim is conservative, useful, and suitable for "
        "future comparison across launch configurations. Measurement before claims."
    )


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class RecordingEncoder:
    def __init__(self) -> None:
        self.contents: list[str] = []

    def __call__(self, messages: list[dict[str, str]], *, thinking_mode: str) -> str:
        assert thinking_mode == "chat"
        assert len(messages) == 1
        content = messages[0]["content"]
        self.contents.append(content)
        return f"<user>{content}<assistant>"


class FakeResponse:
    status = 200

    def __init__(self, lines: tuple[bytes, ...]) -> None:
        self.lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        del unused

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


class FakeUrlOpen:
    def __init__(self, lines: tuple[bytes, ...]) -> None:
        self.lines = lines
        self.payload: dict[str, Any] | None = None

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        assert isinstance(request.data, bytes)
        decoded = cast(object, json.loads(request.data))
        assert isinstance(decoded, dict)
        self.payload = cast(dict[str, Any], decoded)
        return FakeResponse(self.lines)


def _sse_event(
    *,
    completion_tokens: int,
    prompt_tokens: int,
    output_ids: list[int],
    text: str,
    finish_reason: str | None = None,
    cached_tokens: int | None = None,
) -> bytes:
    metadata: dict[str, Any] = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }
    if finish_reason is not None:
        metadata["finish_reason"] = {"type": finish_reason}
    if cached_tokens is not None:
        metadata["cached_tokens"] = cached_tokens
    event = {"text": text, "output_ids": output_ids, "meta_info": metadata}
    return b"data: " + json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"


def _valid_stream(
    *,
    prompt_tokens: int,
    completion_tokens: int = 200,
    finish_reason: str = "stop",
    terminal_position: int | None = None,
    cached_tokens: int | None = None,
) -> tuple[bytes, ...]:
    report = _valid_report()
    output_ids = list(range(10_000, 10_000 + completion_tokens))
    if terminal_position is not None:
        output_ids[terminal_position] = 1
    first_count = 40
    midpoint = len(report) // 2
    return (
        _sse_event(
            completion_tokens=first_count,
            prompt_tokens=prompt_tokens,
            output_ids=output_ids[:first_count],
            text=report[:midpoint],
            cached_tokens=cached_tokens,
        ),
        _sse_event(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            output_ids=output_ids,
            text=report,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
        ),
        b"data: [DONE]\n",
    )


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter((10.0, 11.0, 13.0, 14.0))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(readings))


def test_exact_prompt_uses_varied_semantic_context_deterministically() -> None:
    tokenizer = CharacterTokenizer()
    first_encoder = RecordingEncoder()
    second_encoder = RecordingEncoder()

    first = benchmark.build_exact_prompt_ids(
        tokenizer, first_encoder, target_tokens=3_000
    )
    second = benchmark.build_exact_prompt_ids(
        tokenizer, second_encoder, target_tokens=3_000
    )

    assert len(first) == 3_000
    assert first == second
    generated_context = first_encoder.contents[-1]
    assert "Measurement note N00000" in generated_context
    assert "Measurement note N00008" in generated_context
    assert "AURORA-73" in generated_context
    assert " a a a a" not in generated_context


def test_exact_prompt_rejects_shape_too_small_for_semantic_task() -> None:
    with pytest.raises(ValueError, match="not larger than framing length"):
        benchmark.build_exact_prompt_ids(
            CharacterTokenizer(), RecordingEncoder(), target_tokens=40
        )


def test_semantic_gate_accepts_varied_grounded_prose() -> None:
    assessment = benchmark.assess_semantic_output(_valid_report())

    assert assessment.passed
    assert assessment.word_count >= benchmark.MINIMUM_OUTPUT_WORDS
    assert assessment.semantic_markers_present == len(
        benchmark.REQUIRED_SEMANTIC_MARKERS
    )


def test_semantic_gate_rejects_historical_filler_copy() -> None:
    assessment = benchmark.assess_semantic_output(" a" * 512)

    assert not assessment.passed
    assert "low_lexical_diversity" in assessment.issue_codes
    assert "dominant_repeated_word" in assessment.issue_codes
    assert "missing_required_ending" in assessment.issue_codes


def test_semantic_gate_rejects_garbage_after_valid_answer() -> None:
    assessment = benchmark.assess_semantic_output(_valid_report() + (" libcurl" * 160))

    assert not assessment.passed
    assert "missing_required_ending" in assessment.issue_codes
    assert "degenerate_output_tail" in assessment.issue_codes


def test_semantic_gate_rejects_prompt_copying_from_output_ids() -> None:
    assessment = benchmark.assess_semantic_output(
        _valid_report(),
        prompt_ids=list(range(1_000)),
        output_ids=list(range(200, 400)),
    )

    assert "prompt_copying" in assessment.issue_codes


def test_natural_stop_emits_validated_receipt_without_forcing_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = [7, 8, 9]
    fake_urlopen = FakeUrlOpen(
        _valid_stream(prompt_tokens=len(input_ids), cached_tokens=2)
    )
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    _install_clock(monkeypatch)

    receipt = benchmark.run_benchmark(
        "http://127.0.0.1:30010/generate",
        input_ids,
        512,
        30.0,
        terminal_token_ids=(1,),
        request_id="hotspot-phase-1",
    )

    assert fake_urlopen.payload is not None
    assert fake_urlopen.payload["rid"] == "hotspot-phase-1"
    assert receipt["request_id"] == "hotspot-phase-1"
    assert receipt["server_cached_tokens"] == 2
    sampling = cast(dict[str, Any], fake_urlopen.payload["sampling_params"])
    assert sampling["ignore_eos"] is False
    assert receipt["accepted"] is True
    assert receipt["performance_claim_eligible"] is True
    assert receipt["receipt_safety"] == "validated_natural_stop"
    assert receipt["completion_tokens"] == 200
    assert receipt["exact_requested_token_shape"] is False
    assert receipt["decode_tokens_per_second"] == 80.0
    assert cast(dict[str, Any], receipt["semantic_validation"])["passed"] is True
    assert "output_text" not in receipt


def test_forced_exact_shape_rejects_tokens_after_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = [7, 8, 9]
    fake_urlopen = FakeUrlOpen(
        _valid_stream(
            prompt_tokens=len(input_ids),
            completion_tokens=200,
            finish_reason="length",
            terminal_position=80,
        )
    )
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    _install_clock(monkeypatch)

    with pytest.raises(benchmark.BenchmarkValidationError) as raised:
        benchmark.run_benchmark(
            "http://127.0.0.1:30010/generate",
            input_ids,
            200,
            30.0,
            ignore_eos=True,
            terminal_token_ids=(1,),
        )

    assert "tokens_after_terminal_token" in raised.value.issue_codes
    assert fake_urlopen.payload is not None
    sampling = cast(dict[str, Any], fake_urlopen.payload["sampling_params"])
    assert sampling["ignore_eos"] is True


def test_forced_exact_shape_requires_an_eos_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = [7, 8, 9]
    fake_urlopen = FakeUrlOpen(
        _valid_stream(
            prompt_tokens=len(input_ids),
            completion_tokens=200,
            finish_reason="length",
        )
    )
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    _install_clock(monkeypatch)

    with pytest.raises(benchmark.BenchmarkValidationError) as raised:
        benchmark.run_benchmark(
            "http://127.0.0.1:30010/generate",
            input_ids,
            200,
            30.0,
            ignore_eos=True,
        )

    assert "forced_generation_eos_scan_unavailable" in raised.value.issue_codes


def test_forced_exact_shape_can_emit_a_labeled_validated_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = [7, 8, 9]
    fake_urlopen = FakeUrlOpen(
        _valid_stream(
            prompt_tokens=len(input_ids),
            completion_tokens=200,
            finish_reason="length",
        )
    )
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    _install_clock(monkeypatch)

    receipt = benchmark.run_benchmark(
        "http://127.0.0.1:30010/generate",
        input_ids,
        200,
        30.0,
        ignore_eos=True,
        terminal_token_ids=(1,),
    )

    assert receipt["exact_requested_token_shape"] is True
    assert receipt["receipt_safety"] == "validated_forced_length"
    assert receipt["terminal_token_scan"] == "verified"


def test_fixed_length_diagnostic_reports_token_hash_without_claiming_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = [7, 8, 9]
    fake_urlopen = FakeUrlOpen(
        _valid_stream(
            prompt_tokens=len(input_ids),
            completion_tokens=200,
            finish_reason="length",
            terminal_position=80,
        )
    )
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    _install_clock(monkeypatch)

    receipt = benchmark.run_benchmark(
        "http://127.0.0.1:30010/generate",
        input_ids,
        200,
        30.0,
        ignore_eos=True,
        terminal_token_ids=(1,),
        diagnostic_only=True,
    )

    assert receipt["accepted"] is True
    assert receipt["diagnostic_only"] is True
    assert receipt["performance_claim_eligible"] is False
    assert receipt["receipt_safety"] == "diagnostic_forced_length"
    assert receipt["terminal_token_scan"] == "observed_not_gated"
    assert receipt["terminal_token_count"] == 1
    assert receipt["output_token_ids_available"] is True
    assert isinstance(receipt["output_token_ids_sha256"], str)
    assert receipt["semantic_validation"]["enforced"] is False
