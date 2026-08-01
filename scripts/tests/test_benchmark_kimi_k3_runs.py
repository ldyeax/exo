from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import benchmark_kimi_k3


def _configuration(
    run_count: int,
    tmp_path: Path,
    *,
    speculative_n_max: int | None = None,
) -> benchmark_kimi_k3.BenchmarkConfiguration:
    return benchmark_kimi_k3.BenchmarkConfiguration(
        server_address=benchmark_kimi_k3.ServerAddress.parse("http://127.0.0.1:11434"),
        model="Kimi-K3-UD-Q2_K_XL",
        quantization="UD-Q2_K_XL",
        timeout_seconds=60.0,
        reasoning_effort="high",
        thinking_effort="high",
        reasoning_budget_tokens=96,
        run_count=run_count,
        speculative_n_max=speculative_n_max,
        output_jsonl=tmp_path / "benchmark.jsonl",
        summary_json=tmp_path / "summary.json",
        overwrite=False,
        api_key=None,
        api_key_environment_variable="LLAMA_API_KEY",
    )


def test_single_run_dry_plan_selects_one_complete_pair(tmp_path: Path) -> None:
    plan = benchmark_kimi_k3.dry_run_plan(_configuration(1, tmp_path))

    assert plan["configuration"]["run_count"] == 1
    assert len(plan["semantic_cases"]) == 1
    assert plan["semantic_cases"][0]["name"] == "arithmetic"
    assert plan["acceptance_invariants"]["all_selected_semantic_gates"] == 1
    assert "1 semantic request(s)" in plan["live_actions_not_performed"]


@pytest.mark.parametrize("value", ["0", "6", "-1", "not-an-integer"])
def test_run_count_rejects_values_outside_frozen_cases(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_kimi_k3.benchmark_run_count(value)


@pytest.mark.parametrize("width", [3, 5, 7])
def test_speculative_width_is_added_to_every_generation_payload(
    width: int,
    tmp_path: Path,
) -> None:
    configuration = _configuration(1, tmp_path, speculative_n_max=width)
    calibrated_prompt = benchmark_kimi_k3.CalibratedPrompt(
        content="calibrated",
        token_count=benchmark_kimi_k3.TARGET_PROMPT_TOKENS,
        filler_repetitions=1,
        repair_suffix="",
        counting_endpoint="/apply-template + /tokenize",
        counting_requests=1,
    )

    semantic_request = benchmark_kimi_k3.semantic_payload(
        configuration,
        benchmark_kimi_k3.BENCHMARK_CASES[0],
    )
    performance_request = benchmark_kimi_k3.performance_payload(
        configuration,
        calibrated_prompt,
        seed=1,
    )
    warmup_request = benchmark_kimi_k3.performance_payload(
        configuration,
        calibrated_prompt,
        seed=benchmark_kimi_k3.WARMUP_SEED,
        output_tokens=benchmark_kimi_k3.WARMUP_OUTPUT_TOKENS,
    )

    assert semantic_request["speculative.n_max"] == width
    assert performance_request["speculative.n_max"] == width
    assert warmup_request["speculative.n_max"] == width
    assert configuration.public_json()["speculative_n_max"] == width


def test_omitted_speculative_width_leaves_requests_unchanged(tmp_path: Path) -> None:
    configuration = _configuration(1, tmp_path)
    calibrated_prompt = benchmark_kimi_k3.CalibratedPrompt(
        content="calibrated",
        token_count=benchmark_kimi_k3.TARGET_PROMPT_TOKENS,
        filler_repetitions=1,
        repair_suffix="",
        counting_endpoint="/apply-template + /tokenize",
        counting_requests=1,
    )

    semantic_request = benchmark_kimi_k3.semantic_payload(
        configuration,
        benchmark_kimi_k3.BENCHMARK_CASES[0],
    )
    performance_request = benchmark_kimi_k3.performance_payload(
        configuration,
        calibrated_prompt,
        seed=1,
    )

    assert "speculative.n_max" not in semantic_request
    assert "speculative.n_max" not in performance_request
    assert "speculative_n_max" not in configuration.public_json()


@pytest.mark.parametrize("width", [3, 5, 7])
def test_cli_accepts_supported_speculative_widths(width: int) -> None:
    arguments = benchmark_kimi_k3.build_argument_parser().parse_args(
        [
            "--quant",
            "UD-Q2_K_XL",
            "--speculative-n-max",
            str(width),
        ]
    )

    configuration = benchmark_kimi_k3.configuration_from_arguments(arguments)

    assert configuration.speculative_n_max == width


@pytest.mark.parametrize("value", ["0", "1", "4", "6", "8", "not-an-integer"])
def test_speculative_width_rejects_values_outside_sweep(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_kimi_k3.speculative_candidate_width(value)
