from scripts.summarize_dsv4_kt_hybrid_timing import summarize


def _receipt(
    *,
    rank: int,
    layer: int,
    cpu_wait_ms: float,
    active_experts: int,
    pp_rank: int = 0,
) -> dict[str, object]:
    return {
        "format": "sglang_kt_hybrid_timing_v1",
        "tp_rank": rank,
        "pp_rank": pp_rank,
        "ep_rank": rank,
        "layer": layer,
        "step": 4,
        "num_tokens": 6,
        "start_monotonic_ns": 1_000 + layer * 100,
        "end_monotonic_ns": 1_050 + layer * 100 + rank,
        "cpu_wait_ms": cpu_wait_ms,
        "total_ms": cpu_wait_ms + 0.5,
        "cpu_active_experts": active_experts,
        "estimated_cpu_weight_stream_bytes": active_experts * 100,
        "numa_pages_by_node": {str(rank): 1000 + rank},
    }


def test_summary_pairs_layers_and_exposes_rank_tail() -> None:
    summary = summarize(
        [
            _receipt(rank=0, layer=0, cpu_wait_ms=2.0, active_experts=10),
            _receipt(rank=1, layer=0, cpu_wait_ms=4.0, active_experts=12),
            _receipt(rank=0, layer=1, cpu_wait_ms=3.0, active_experts=11),
            _receipt(rank=1, layer=1, cpu_wait_ms=1.0, active_experts=9),
        ]
    )

    assert summary["rank_count"] == 2
    assert summary["paired_layer_count"] == 2
    assert summary["mean_paired_cpu_wait_skew_ms"] == 2.0
    assert summary["worst_paired_cpu_wait_skew_ms"] == 2.0
    assert summary["mean_paired_active_expert_skew"] == 2.0
    assert summary["mean_paired_estimated_weight_byte_skew"] == 200.0
    assert summary["complete_cycle_count"] == 1
    assert summary["mean_cycle_critical_cpu_wait_ms"] == 7.0
    assert summary["pipeline_stage"] == [
        {
            "pp_rank": 0,
            "layer_count": 2,
            "layers": [0, 1],
            "complete_cycle_count": 1,
            "mean_cycle_critical_cpu_wait_ms": 7.0,
            "median_cycle_critical_cpu_wait_ms": 7.0,
            "max_cycle_critical_cpu_wait_ms": 7.0,
        }
    ]
    assert summary["latest_numa_pages_by_rank"] == {
        "pp0-tp0-ep0": {"0": 1000},
        "pp0-tp1-ep1": {"1": 1001},
    }


def test_summary_separates_disjoint_pp2_stage_critical_paths() -> None:
    summary = summarize(
        [
            _receipt(
                rank=0,
                pp_rank=0,
                layer=0,
                cpu_wait_ms=2.0,
                active_experts=10,
            ),
            _receipt(
                rank=0,
                pp_rank=0,
                layer=1,
                cpu_wait_ms=3.0,
                active_experts=11,
            ),
            _receipt(
                rank=0,
                pp_rank=1,
                layer=2,
                cpu_wait_ms=4.0,
                active_experts=12,
            ),
            _receipt(
                rank=0,
                pp_rank=1,
                layer=3,
                cpu_wait_ms=3.0,
                active_experts=9,
            ),
        ]
    )

    assert summary["rank_count"] == 2
    assert summary["paired_layer_count"] == 0
    assert summary["pipeline_stage"] == [
        {
            "pp_rank": 0,
            "layer_count": 2,
            "layers": [0, 1],
            "complete_cycle_count": 1,
            "mean_cycle_critical_cpu_wait_ms": 5.0,
            "median_cycle_critical_cpu_wait_ms": 5.0,
            "max_cycle_critical_cpu_wait_ms": 5.0,
        },
        {
            "pp_rank": 1,
            "layer_count": 2,
            "layers": [2, 3],
            "complete_cycle_count": 1,
            "mean_cycle_critical_cpu_wait_ms": 7.0,
            "median_cycle_critical_cpu_wait_ms": 7.0,
            "max_cycle_critical_cpu_wait_ms": 7.0,
        },
    ]
