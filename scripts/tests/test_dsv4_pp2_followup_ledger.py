from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import dsv4_pp2_followup_ledger as ledger
from scripts import transfer_dsv4_ep2_plan_to_pp2 as transfer
from scripts.tests.test_transfer_dsv4_ep2_plan_to_pp2 import (
    make_direct_coherency_receipt,
    make_direct_hotspot_confirmation,
    make_source_plan,
    sha256_file,
)


def prepare_bound_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "winner.pt"
    make_source_plan(source, source_gpu_experts_per_rank=14)
    confirmation = tmp_path / "split-history-confirm-hotspot.json"
    coherency = tmp_path / "split-history-confirm-coherency.json"
    make_direct_hotspot_confirmation(confirmation, source_plan=source)
    make_direct_coherency_receipt(coherency)
    transferred = transfer.stage_transferred_plan(
        source_ep2_plan=source,
        source_ep_confirmation_receipt=confirmation,
        source_ep_coherency_receipt=coherency,
        cache_root=tmp_path / "plans",
        expected_target_gpu_experts_per_layer=28,
    )
    return source, confirmation, coherency, transferred


def authorize(
    tmp_path: Path,
    *,
    source: Path,
    confirmation: Path,
    coherency: Path,
    transferred: Path,
    role: str,
    partition: str,
    async_depth: int,
    output_name: str,
    first_benchmark: Path | None = None,
) -> Path:
    return ledger.authorize_launch(
        ledger_path=tmp_path / "ledger.json",
        output_path=tmp_path / output_name,
        ep_confirmation_receipt=confirmation,
        ep_coherency_receipt=coherency,
        source_ep2_plan=source,
        transferred_plan=transferred,
        run_role=role,
        pipeline_layer_partition=partition,
        pp_async_batch_depth=async_depth,
        chunked_prefill_size=1024,
        first_benchmark_receipt=first_benchmark,
        native_artifact=transfer.QUALIFIED_NATIVE_ARTIFACT,
        native_artifact_sha256=transfer.QUALIFIED_NATIVE_ARTIFACT_SHA256,
    )


def write_first_benchmark(
    path: Path,
    *,
    transferred: Path,
    authorization: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "ok": True,
                "performance_claim_eligible": True,
                "configuration": {
                    "benchmark_provenance": {
                        "run_label": "pp2-transfer",
                        "expert_plan_path": str(transferred),
                        "expert_plan_sha256": sha256_file(transferred),
                    },
                    "launch_authorization": {
                        "authorization_receipt_path": str(authorization),
                        "authorization_receipt_sha256": sha256_file(authorization),
                        "ordinal": 1,
                        "run_role": "transfer",
                    },
                    "expected_chunked_prefill_size": 1024,
                    "expected_pp_async_batch_depth": 0,
                    "expected_cpuinfer_threads": 56,
                    "server_contract": {
                        "tp_size": 1,
                        "pp_size": 2,
                        "ep_size": 1,
                        "context_length": 524_288,
                        "max_total_tokens": 524_288,
                        "dsv4_oscar_int2_kv_storage": True,
                        "dsv4_oscar_int2_split_history": True,
                        "cuda_graph_backend_decode": "full",
                        "cuda_graph_bs_decode": [1, 2],
                    },
                },
                "native_generate": {
                    "request_count": 2,
                    "all_streams_complete": True,
                    "all_decode_timing_valid": True,
                    "all_naturally_terminated": True,
                    "concurrent_start_confirmed": True,
                    "concurrent_overlap_observed": True,
                },
                "openai_tool_calls": {
                    "request_count": 2,
                    "all_streams_complete": True,
                    "concurrent_start_confirmed": True,
                    "concurrent_overlap_observed": True,
                },
                "nvlink_traffic": {
                    "counter_deltas": {
                        f"counter-{index}": index + 1 for index in range(16)
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_ledger_allows_transfer_then_one_optimized_launch_only(
    tmp_path: Path,
) -> None:
    source, confirmation, coherency, transferred = prepare_bound_inputs(tmp_path)
    first_authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="transfer",
        partition="21,22",
        async_depth=0,
        output_name="authorization-transfer.json",
    )
    first_benchmark = tmp_path / "benchmark-transfer.json"
    write_first_benchmark(
        first_benchmark,
        transferred=transferred,
        authorization=first_authorization,
    )
    second_authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="optimized",
        partition="22,21",
        async_depth=0,
        output_name="authorization-optimized.json",
        first_benchmark=first_benchmark,
    )

    assert first_authorization.is_file()
    assert second_authorization.is_file()
    persisted = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert persisted["format"] == ledger.LEDGER_FORMAT
    assert persisted["maximum_model_launches"] == 2
    assert [launch["run_role"] for launch in persisted["launches"]] == [
        "transfer",
        "optimized",
    ]
    assert persisted["launches"][0]["configuration"]["context_length"] == 524_288
    assert (
        persisted["launches"][0]["configuration"]["physical_kv_cache_storage"]
        == "oscar-int2-asymmetric"
    )
    assert persisted["launches"][0]["configuration"]["oscar_split_history"] is True
    assert persisted["ep_winner"]["coherency_receipt_path"] == str(coherency)
    assert persisted["ep_winner"]["coherency_receipt_sha256"] == sha256_file(coherency)

    with pytest.raises(
        ledger.LaunchLedgerError,
        match="two permitted PP2 model launches are exhausted",
    ):
        authorize(
            tmp_path,
            source=source,
            confirmation=confirmation,
            coherency=coherency,
            transferred=transferred,
            role="optimized",
            partition="21,22",
            async_depth=1,
            output_name="authorization-third.json",
            first_benchmark=first_benchmark,
        )


@pytest.mark.parametrize(
    ("partition", "async_depth"),
    (("21,22", 0), ("22,21", 1)),
)
def test_optimized_launch_must_change_exactly_one_pp_only_knob(
    tmp_path: Path,
    partition: str,
    async_depth: int,
) -> None:
    source, confirmation, coherency, transferred = prepare_bound_inputs(tmp_path)
    first_authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="transfer",
        partition="21,22",
        async_depth=0,
        output_name="authorization-transfer.json",
    )
    first_benchmark = tmp_path / "benchmark-transfer.json"
    write_first_benchmark(
        first_benchmark,
        transferred=transferred,
        authorization=first_authorization,
    )

    with pytest.raises(
        ledger.LaunchLedgerError,
        match="change exactly one PP-only knob",
    ):
        authorize(
            tmp_path,
            source=source,
            confirmation=confirmation,
            coherency=coherency,
            transferred=transferred,
            role="optimized",
            partition=partition,
            async_depth=async_depth,
            output_name="authorization-optimized.json",
            first_benchmark=first_benchmark,
        )


def test_optimized_launch_accepts_complete_failed_first_benchmark(
    tmp_path: Path,
) -> None:
    source, confirmation, coherency, transferred = prepare_bound_inputs(tmp_path)
    first_authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="transfer",
        partition="21,22",
        async_depth=0,
        output_name="authorization-transfer.json",
    )
    first_benchmark = tmp_path / "benchmark-transfer.json"
    write_first_benchmark(
        first_benchmark,
        transferred=transferred,
        authorization=first_authorization,
    )
    receipt = json.loads(first_benchmark.read_text(encoding="utf-8"))
    receipt["ok"] = False
    receipt["performance_claim_eligible"] = False
    first_benchmark.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="optimized",
        partition="22,21",
        async_depth=0,
        output_name="authorization-optimized.json",
        first_benchmark=first_benchmark,
    )

    payload = json.loads(authorization.read_text(encoding="utf-8"))
    assert payload["first_benchmark_receipt"]["ok"] is False
    assert payload["first_benchmark_receipt"]["performance_claim_eligible"] is False


def test_optimized_launch_rejects_incomplete_failed_first_benchmark(
    tmp_path: Path,
) -> None:
    source, confirmation, coherency, transferred = prepare_bound_inputs(tmp_path)
    first_authorization = authorize(
        tmp_path,
        source=source,
        confirmation=confirmation,
        coherency=coherency,
        transferred=transferred,
        role="transfer",
        partition="21,22",
        async_depth=0,
        output_name="authorization-transfer.json",
    )
    first_benchmark = tmp_path / "benchmark-transfer.json"
    write_first_benchmark(
        first_benchmark,
        transferred=transferred,
        authorization=first_authorization,
    )
    receipt = json.loads(first_benchmark.read_text(encoding="utf-8"))
    receipt["ok"] = False
    receipt["performance_claim_eligible"] = False
    receipt.pop("nvlink_traffic")
    first_benchmark.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ledger.LaunchLedgerError, match="evidence is incomplete"):
        authorize(
            tmp_path,
            source=source,
            confirmation=confirmation,
            coherency=coherency,
            transferred=transferred,
            role="optimized",
            partition="22,21",
            async_depth=0,
            output_name="authorization-optimized.json",
            first_benchmark=first_benchmark,
        )
