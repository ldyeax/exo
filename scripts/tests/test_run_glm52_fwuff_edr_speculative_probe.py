from __future__ import annotations

from dataclasses import replace

import pytest

from scripts import run_glm52_fwuff_edr_speculative_probe as probe

PERFTEST_OUTPUT = """\
---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
#, usec
1, 3.60000
2, 3.70000
3, 3.80000
4, 4.90000
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 12288   10          3.60           4.90         3.70              3.75             0.40           4.90              4.90
---------------------------------------------------------------------------------------
"""


def observation() -> probe.HostObservation:
    return probe.HostObservation(
        hostname="dwagon",
        hca="mlx5_0",
        hca_bdf="0000:27:00.0",
        numa_node=0,
        netdevs=("ibs5",),
        state="4: ACTIVE",
        physical_state="5: LinkUp",
        rate="100 Gb/sec (4X EDR)",
        lid="0x7",
        perftest_path="/usr/bin/ib_write_lat",
        perftest_sha256="a" * 64,
        perftest_version="Version: 6.24",
        perftest_process_ids=(),
        counters={name: 10 for name in probe._COUNTERS},
    )


def test_transfer_contract_keeps_only_draft_messages_on_wire() -> None:
    contract = probe.transfer_contract()
    draft = contract["draft"]
    wire = contract["wire"]
    assert isinstance(draft, dict)
    assert isinstance(wire, dict)
    assert draft["draft_kv_and_tentative_state_remain_remote"] is True
    assert wire["feature_row_bytes"] == 12_288
    allowed = wire["allowed_payloads"]
    assert isinstance(allowed, list)
    assert [item["payload_bytes"] for item in allowed if isinstance(item, dict)] == [
        4,
        12_288,
        24_576,
        6_291_456,
        95_158_272,
    ]
    forbidden = wire["forbidden_payloads"]
    assert isinstance(forbidden, list)
    assert "target_kv_cache" in forbidden
    assert "full_vocabulary_logits" in forbidden
    next_admission = draft["next_admission_contract"]
    assert isinstance(next_admission, dict)
    assert next_admission["amx_artifact_logical_partition_slots"] == [0, 1]
    assert next_admission["physical_numa_node_map"] == [0, 0]
    assert next_admission["kt_cpuinfer_threads"] == 60
    selector = next_admission["standalone_weight_selector"]
    assert isinstance(selector, dict)
    assert selector["selected_tensor_count"] == 38
    exact_names = selector["exact_names"]
    assert isinstance(exact_names, list)
    assert exact_names[:3] == [
        "model.embed_tokens.weight",
        "lm_head.qweight",
        "lm_head.scales",
    ]
    assert "model.layers.78.self_attn.kv_b_proj.kc_qweight" in exact_names
    assert set(selector["required_shards"]) == {
        "model-00001-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    }


def test_remote_selector_adds_owned_header_and_excludes_routed_experts() -> None:
    weight_map: dict[str, object] = {
        name: (
            probe.REMOTE_DRAFT_HEADER_SHARD
            if name in probe.REMOTE_DRAFT_HEADER_WEIGHT_NAMES
            else probe.REMOTE_DRAFT_LAYER_SHARD
        )
        for name in probe.REMOTE_DRAFT_WEIGHT_NAMES
    }
    weight_map.update(
        {
            "model.layers.78.mlp.experts.0.gate_proj.weight": "expert.safetensors",
            "model.layers.77.eh_proj.weight": "other.safetensors",
        }
    )
    names, shards = probe.select_remote_draft_weight_names(weight_map)
    assert names == frozenset(probe.REMOTE_DRAFT_WEIGHT_NAMES)
    assert shards == frozenset(
        {probe.REMOTE_DRAFT_HEADER_SHARD, probe.REMOTE_DRAFT_LAYER_SHARD}
    )


def test_remote_selector_rejects_noncanonical_shards() -> None:
    weight_map: dict[str, object] = {
        name: (
            probe.REMOTE_DRAFT_HEADER_SHARD
            if name in probe.REMOTE_DRAFT_HEADER_WEIGHT_NAMES
            else probe.REMOTE_DRAFT_LAYER_SHARD
        )
        for name in probe.REMOTE_DRAFT_WEIGHT_NAMES
    }
    weight_map["model.layers.78.eh_proj.weight"] = "different.safetensors"
    with pytest.raises(probe.SpeculativeEdrProbeError, match="immutable shard pair"):
        probe.select_remote_draft_weight_names(weight_map)


def test_perftest_arguments_bind_rc_hca_size_iterations_and_port() -> None:
    assert probe.perftest_arguments(
        "/usr/bin/ib_write_lat",
        "mlx5_0",
        12_288,
        2_000,
        18_611,
        peer_ip="10.44.0.1",
    ) == (
        "/usr/bin/ib_write_lat",
        "10.44.0.1",
        "-d",
        "mlx5_0",
        "-i",
        "1",
        "-c",
        "RC",
        "-s",
        "12288",
        "-n",
        "2000",
        "-p",
        "18611",
        "-F",
        "-H",
    )


@pytest.mark.parametrize(
    ("payload_bytes", "iterations", "port"),
    ((0, 10, 18_610), (4, 4, 18_610), (4, 10, 80)),
)
def test_perftest_arguments_reject_invalid_bounds(
    payload_bytes: int,
    iterations: int,
    port: int,
) -> None:
    with pytest.raises(probe.SpeculativeEdrProbeError):
        probe.perftest_arguments(
            "/usr/bin/ib_write_lat",
            "mlx5_0",
            payload_bytes,
            iterations,
            port,
            peer_ip=None,
        )


def test_parse_perftest_latency_uses_histogram_quantiles() -> None:
    parsed = probe.parse_perftest_latency(
        PERFTEST_OUTPUT,
        expected_payload_bytes=12_288,
        expected_iterations=10,
    )
    assert parsed.payload_bytes == 12_288
    assert parsed.histogram_sample_count == 4
    assert parsed.histogram_p50_microseconds == 3.7
    assert parsed.histogram_p95_microseconds == 4.9
    assert parsed.histogram_p99_microseconds == 4.9
    assert parsed.reported_p99_microseconds == 4.9


def test_parse_perftest_latency_requires_exact_summary_identity() -> None:
    with pytest.raises(probe.SpeculativeEdrProbeError, match="exact summary row"):
        probe.parse_perftest_latency(
            PERFTEST_OUTPUT,
            expected_payload_bytes=24_576,
            expected_iterations=10,
        )


def test_host_validation_rejects_wrong_link_or_unowned_perftest() -> None:
    bad = replace(
        observation(),
        rate="40 Gb/sec (4X QDR)",
        perftest_process_ids=(123,),
    )
    with pytest.raises(probe.SpeculativeEdrProbeError) as captured:
        probe.validate_host_observation(
            bad,
            expected_hostname="dwagon",
            expected_hca="mlx5_0",
            expected_netdev="ibs5",
        )
    assert "rate='40 Gb/sec (4X QDR)'" in str(captured.value)
    assert "unowned ib_write_lat" in str(captured.value)


def test_counter_gate_requires_payload_and_zero_health_deltas() -> None:
    clean = {name: 0 for name in probe._COUNTERS}
    clean["port_xmit_data"] = 10
    clean["port_rcv_data"] = 10
    probe._validate_counter_deltas(clean, clean)

    unhealthy = dict(clean)
    unhealthy["port_rcv_errors"] = 1
    with pytest.raises(probe.SpeculativeEdrProbeError, match="health counters"):
        probe._validate_counter_deltas(clean, unhealthy)


def test_remote_observation_is_self_contained() -> None:
    source = probe._remote_observation_script("mlx5_0", "/usr/bin/ib_write_lat")
    assert "importlib" not in source
    assert "/root/exo" not in source
    assert "port_xmit_data" in source
    compile(source, "<remote-observation>", "exec")
