from __future__ import annotations

from scripts.qualify_dsv4_tp2_interconnect import (
    counter_deltas,
    parse_message_sizes,
    parse_nccl_transports,
    parse_nvlink_counters,
    parse_nvlink_status,
    parse_topology_relation,
)

TOPOLOGY = """
        GPU0    GPU1    CPU Affinity
GPU0     X      NV4     0-55
GPU1    NV4      X      56-111
"""

P2P_MATRIX = """
        GPU0    GPU1
GPU0     X       NS
GPU1     NS      X
"""

COUNTERS = """
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-a)
     Link 0: Data Tx: 100 KiB
     Link 0: Data Rx: 200 KiB
     Link 1: Data Tx: 300 KiB
     Link 1: Data Rx: 400 KiB
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-b)
     Link 0: Data Tx: 200 KiB
     Link 0: Data Rx: 100 KiB
     Link 1: Data Tx: 400 KiB
     Link 1: Data Rx: 300 KiB
"""


def test_parses_topology_and_records_the_p2p_matrix_disagreement() -> None:
    assert parse_topology_relation(TOPOLOGY, 0, 1) == "NV4"
    assert parse_topology_relation(TOPOLOGY, 1, 0) == "NV4"
    assert parse_topology_relation(P2P_MATRIX, 0, 1) == "NS"


def test_parses_per_link_status_and_payload_counter_deltas() -> None:
    status = parse_nvlink_status(
        """
GPU 0: NVIDIA GeForce RTX 3090
    Link 0: 14.062 GB/s
    Link 1: 14.062 GB/s
GPU 1: NVIDIA GeForce RTX 3090
    Link 0: 14.062 GB/s
    Link 1: 14.062 GB/s
"""
    )
    assert status == {0: {0: 14.062, 1: 14.062}, 1: {0: 14.062, 1: 14.062}}

    before = parse_nvlink_counters(COUNTERS)
    after = {key: value + 17 for key, value in before.items()}
    assert counter_deltas(before, after) == {key: 17 for key in before}


def test_nccl_transport_requires_directional_p2p_and_exposes_fallback() -> None:
    admitted = parse_nccl_transports(
        """
host NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/IPC/read
host NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/IPC/read
host NCCL INFO comm 0x1 rank 0 nranks 2 cudaDev 0 - Init COMPLETE
host NCCL INFO comm 0x2 rank 1 nranks 2 cudaDev 1 - Init COMPLETE
"""
    )
    assert admitted["p2p_directions"] == [[0, 1], [1, 0]]
    assert admitted["fallback_routes"] == []
    assert admitted["init_complete_count"] == 2

    fallback = parse_nccl_transports(
        "host NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct"
    )
    assert len(fallback["fallback_routes"]) == 1


def test_representative_message_sizes_are_mandatory() -> None:
    assert parse_message_sizes("49152,1048576") == (49_152, 1_048_576)
