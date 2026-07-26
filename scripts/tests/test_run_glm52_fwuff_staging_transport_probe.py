from __future__ import annotations

import pytest

from scripts import run_glm52_fwuff_staging_transport_probe as probe


def test_payload_contract_is_narrow_and_includes_optional_prefix() -> None:
    assert [
        (item.message_kind, item.payload_bytes) for item in probe.PAYLOAD_PROBES
    ] == [
        ("PROPOSAL", 4),
        ("ADVANCE", 12_288),
        ("ADVANCE", 24_576),
        ("OPEN", 6_291_456),
    ]
    assert probe.PAYLOAD_PROBES[-1].transport_iterations == 64


def test_nearest_rank_and_latency_summary_report_requested_percentiles() -> None:
    summary = probe.latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["sample_count"] == 5
    assert summary["p50_microseconds"] == 3.0
    assert summary["p95_microseconds"] == 100.0
    assert summary["p99_microseconds"] == 100.0


@pytest.mark.parametrize("percentile", (0.0, -0.1, 1.1))
def test_nearest_rank_rejects_invalid_percentile(percentile: float) -> None:
    with pytest.raises(probe.StagingTransportProbeError):
        probe._nearest_rank([1.0], percentile)


def test_remote_sources_compile_without_repo_dependency() -> None:
    compile(probe.CUDA_WORKER_SOURCE, "<cuda-worker>", "exec")
    compile(probe.ZMQ_SERVER_SOURCE, "<zmq-server>", "exec")
    compile(probe.RAW_TCP_SERVER_SOURCE, "<raw-tcp-server>", "exec")
    network_source = probe._remote_netdev_script("ibs2")
    assert "/root/exo" not in network_source
    compile(network_source, "<network-worker>", "exec")


def test_remote_python_command_uses_encoded_source_and_exact_arguments() -> None:
    command = probe.remote_python_command(
        "/immutable/venv/bin/python",
        "print('ok')",
        ("--token", "a token"),
    )
    assert command.startswith("/immutable/venv/bin/python -c ")
    assert "print('ok')" not in command
    assert "--token 'a token'" in command


def test_make_payload_is_preallocated_and_sequence_is_in_place() -> None:
    payload = probe._make_payload(12_288)
    identity = id(payload)
    probe._put_sequence(payload, 0x01020304)
    assert id(payload) == identity
    assert payload[:4] == b"\x01\x02\x03\x04"
    assert payload[4] == 0xA5
    assert len(payload) == 12_288


def test_select_best_transport_prefers_measured_throughput() -> None:
    comparison = probe.select_best_transport(
        {"completed_messages_per_second": 100.0},
        {"completed_messages_per_second": 175.0},
    )
    assert comparison["throughput_winner"] == "two_slot"
    assert comparison["two_slot_throughput_speedup"] == 1.75


def test_select_best_transport_rejects_nonpositive_rates() -> None:
    with pytest.raises(probe.StagingTransportProbeError):
        probe.select_best_transport(
            {"completed_messages_per_second": 0.0},
            {"completed_messages_per_second": 1.0},
        )


def test_select_best_application_transport_can_split_latency_and_rate_winners() -> None:
    comparison = probe.select_best_application_transport(
        {
            "sync": {
                "completed_messages_per_second": 1_000.0,
                "round_trip_latency": {"p50_microseconds": 200.0},
            },
            "two_slot": {
                "completed_messages_per_second": 1_800.0,
                "completion_latency": {"p50_microseconds": 350.0},
            },
        }
    )
    assert comparison["throughput_winner"] == "two_slot"
    assert comparison["latency_winner"] == "sync"


def test_counter_gate_requires_payload_and_clean_health() -> None:
    clean = {name: 0 for name in probe._NETDEV_COUNTERS}
    clean["rx_bytes"] = 100
    clean["tx_bytes"] = 100
    probe._validate_counter_delta(host="fwuff", delta=clean)

    unhealthy = dict(clean)
    unhealthy["rx_errors"] = 1
    with pytest.raises(probe.StagingTransportProbeError, match="health"):
        probe._validate_counter_delta(host="fwuff", delta=unhealthy)


def test_remote_server_message_count_includes_warmup() -> None:
    payload = probe.PAYLOAD_PROBES[0]
    command = probe._remote_server_command(
        remote_runtime_python="/immutable/python",
        remote_edr_ip="10.44.0.2",
        port=18_740,
        mode="router",
        payload_bytes=payload.payload_bytes,
        message_count=(
            payload.transport_warmup_iterations + payload.transport_iterations
        ),
        token="owned-token",
        remote_application_cpu=119,
    )
    assert "--message-count 2100" in command
    assert "--bind-ip 10.44.0.2" in command
    assert "--mode router" in command
    assert "--application-cpu 119" in command


def test_raw_server_command_binds_edr_and_busy_poll_variant() -> None:
    command = probe._remote_raw_tcp_server_command(
        remote_runtime_python="/immutable/python",
        remote_edr_ip="10.44.0.2",
        port=18_750,
        payload_bytes=12_288,
        message_count=100,
        token="owned-raw-token",
        remote_application_cpu=119,
        busy_poll_microseconds=50,
    )
    assert "--bind-ip 10.44.0.2" in command
    assert "--payload-bytes 12288" in command
    assert "--busy-poll-microseconds 50" in command
