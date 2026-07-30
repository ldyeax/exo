from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts import summarize_kimi_k3_telemetry as summary


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(record, separators=(',', ':'), sort_keys=True)}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _benchmark_records() -> list[dict[str, object]]:
    return [
        {
            "accepted": True,
            "case": f"case-{run_index}",
            "event": "performance_result",
            "run_index": run_index,
            "sequence": run_index,
            "response": {
                "elapsed_seconds": 5.0,
                "started_at_utc": (
                    f"1970-01-01T00:01:{10 + (run_index - 1) * 10:02d}Z"
                ),
            },
        }
        for run_index in range(1, 6)
    ]


def _telemetry_records(
    *,
    host: str,
    pid: int,
    omit_run_three_interior: bool,
) -> list[dict[str, object]]:
    seconds = [
        second
        for run_index in range(1, 6)
        for second in (
            69 + (run_index - 1) * 10,
            72 + (run_index - 1) * 10,
            76 + (run_index - 1) * 10,
        )
        if not (
            omit_run_three_interior
            and run_index == 3
            and second == 72 + (run_index - 1) * 10
        )
    ]
    return [
        {
            "host": host,
            "label": f"{host}-sampler",
            "pid": pid,
            "process_status": {"VmRSS": 1000 + index},
            "realtime_ns": second * summary.NANOSECONDS_PER_SECOND,
            "schema": summary.TELEMETRY_SCHEMA,
        }
        for index, second in enumerate(seconds)
    ]


def test_partial_coverage_never_interpolates_missing_source_window(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    dwagon = tmp_path / "dwagon.jsonl"
    fwuff = tmp_path / "fwuff.jsonl"
    output = tmp_path / "summary.json"
    _write_jsonl(benchmark, _benchmark_records())
    _write_jsonl(
        dwagon,
        _telemetry_records(
            host="dwagon",
            pid=100,
            omit_run_three_interior=False,
        ),
    )
    _write_jsonl(
        fwuff,
        _telemetry_records(
            host="fwuff",
            pid=200,
            omit_run_three_interior=True,
        ),
    )
    telemetry = (("dwagon", dwagon), ("fwuff", fwuff))

    try:
        summary.build_summary(benchmark, telemetry, output)
    except summary.SummaryError as error:
        assert "fwuff" in str(error)
        assert "no sample inside performance run 3" in str(error)
    else:
        raise AssertionError("strict coverage unexpectedly accepted a sample gap")

    result = summary.build_summary(
        benchmark,
        telemetry,
        output,
        allow_uncovered_source_windows=True,
    )

    assert result["schema"] == summary.PARTIAL_COVERAGE_SUMMARY_SCHEMA
    performance_runs_value = result["performance_runs"]
    assert isinstance(performance_runs_value, list)
    performance_runs = cast(list[object], performance_runs_value)
    run_three = summary.require_object(performance_runs[2], "run three")
    run_three_telemetry = summary.require_object(
        run_three["telemetry"],
        "run three telemetry",
    )
    unavailable = summary.require_object(
        run_three_telemetry["fwuff"],
        "fwuff run three",
    )
    assert unavailable["availability"] == "unavailable"
    assert unavailable["reason_code"] == "no_sample_inside_window"
    assert unavailable["interpolation_performed"] is False
    assert unavailable["metrics_computed"] is False
    gap = summary.require_object(unavailable["gap"], "fwuff gap")
    before = summary.require_object(gap["before"], "fwuff gap before")
    after = summary.require_object(gap["after"], "fwuff gap after")
    assert before["realtime_ns"] == 89 * summary.NANOSECONDS_PER_SECOND
    assert after["realtime_ns"] == 96 * summary.NANOSECONDS_PER_SECOND

    cluster_totals = summary.require_object(
        run_three["cluster_totals"],
        "run three cluster totals",
    )
    source_coverage = summary.require_object(
        cluster_totals["source_coverage"],
        "run three source coverage",
    )
    assert source_coverage["included_sources"] == ["dwagon"]
    assert source_coverage["excluded_sources"] == ["fwuff"]
    assert source_coverage["complete"] is False

    overall = summary.require_object(result["overall"], "overall")
    telemetry_summary = summary.require_object(
        overall["telemetry"],
        "overall telemetry",
    )
    fwuff_summary = summary.require_object(
        telemetry_summary["fwuff"],
        "overall fwuff",
    )
    coverage = summary.require_object(
        fwuff_summary["coverage"],
        "overall fwuff coverage",
    )
    assert coverage["covered_run_indices"] == [1, 2, 4, 5]
    assert coverage["unavailable_run_indices"] == [3]
    overall_cluster = summary.require_object(
        overall["cluster_totals"],
        "overall cluster",
    )
    cluster_coverage = summary.require_object(
        overall_cluster["source_window_coverage"],
        "overall cluster coverage",
    )
    assert cluster_coverage["expected_source_window_count"] == 10
    assert cluster_coverage["covered_source_window_count"] == 9
    assert cluster_coverage["coverage_fraction"] == 0.9
