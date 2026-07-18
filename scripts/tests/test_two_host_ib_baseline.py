from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Sequence
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from scripts import two_host_ib_baseline as baseline

HEALTH_COUNTERS = (
    "symbol_error",
    "link_downed",
    "link_error_recovery",
    "port_rcv_errors",
    "port_rcv_remote_physical_errors",
    "port_rcv_switch_relay_errors",
    "port_xmit_discards",
    "port_xmit_constraint_errors",
    "port_rcv_constraint_errors",
    "local_link_integrity_errors",
    "excessive_buffer_overrun_errors",
    "VL15_dropped",
)


def source_identity() -> baseline.SourceIdentity:
    return baseline.SourceIdentity(commit="1" * 40, dirty_file_hashes={})


def port(port_number: int, guid: str) -> baseline.PortIdentity:
    return baseline.PortIdentity(
        port=cast(Literal[1, 2], port_number),
        port_guid=guid,
        gid=f"fe80::{'0010:e000:0166:3a19' if port_number == 1 else '0010:e000:0166:3a1a'}",
    )


def host(
    name: str,
    *,
    transport: Literal["local", "ssh"],
    bdf: str,
    numa_node: int,
    artifact_hash: str,
) -> baseline.HostConfig:
    if name == "dwagon":
        ports = (
            port(1, "0010:e000:0166:3a19"),
            port(2, "0010:e000:0166:3a1a"),
        )
        node_guid = "0010:e000:0166:3a18"
        management_address = "192.168.40.24"
    else:
        ports = (
            baseline.PortIdentity(
                port=1,
                port_guid="e41d:2d03:004d:32e1",
                gid="fe80::e41d:2d03:004d:32e1",
            ),
            baseline.PortIdentity(
                port=2,
                port_guid="e41d:2d03:004d:32e2",
                gid="fe80::e41d:2d03:004d:32e2",
            ),
        )
        node_guid = "e41d:2d03:004d:32e0"
        management_address = "192.168.40.248"
    return baseline.HostConfig(
        name=name,
        management_address=management_address,
        transport=transport,
        ssh_target=None if transport == "local" else "fwuff",
        source_directory="/root/exo",
        source=source_identity(),
        preflight=baseline.HostPreflightPolicy(
            maximum_load_1m_per_online_cpu=1.0,
            minimum_available_memory_bytes=1,
            allowed_cpu_frequency_governors=("performance",),
        ),
        cpu_set=(0, 1),
        numa_nodes=(numa_node,),
        hca=baseline.HcaIdentity(
            device="mlx4_0",
            node_guid=node_guid,
            pci=baseline.PciIdentity(
                bdf=bdf,
                current_width=8,
                maximum_width=8,
                current_speed_gtps=8.0,
                maximum_speed_gtps=8.0,
                numa_node=numa_node,
            ),
            ports=ports,
        ),
        tools=baseline.HostTools(
            python="/root/exo/.venv/bin/python",
            harness_script="/root/exo/scripts/two_host_ib_baseline.py",
            numactl="/usr/bin/numactl",
            ib_write_bw="/usr/bin/ib_write_bw",
            ib_write_bw_sha256=artifact_hash,
            opensm="/usr/sbin/opensm" if transport == "local" else None,
        ),
    )


def config(result_directory: Path) -> baseline.BaselineConfig:
    artifact_hash = "a" * 64
    return baseline.BaselineConfig(
        schema_version=1,
        run_id=result_directory.name,
        namespace="ib-baseline-test",
        result_directory=str(result_directory),
        ssh=baseline.SshConfig(
            executable="/usr/bin/ssh",
            known_hosts_file="/root/.ssh/known_hosts",
            identity_file="/root/.ssh/id_ed25519",
            user="root",
            port=22,
            connect_timeout_seconds=5,
            server_alive_interval_seconds=10,
            server_alive_count_max=3,
        ),
        local_host=host(
            "dwagon",
            transport="local",
            bdf="0000:d8:00.0",
            numa_node=1,
            artifact_hash=artifact_hash,
        ),
        remote_host=host(
            "fwuff",
            transport="ssh",
            bdf="0000:16:00.0",
            numa_node=0,
            artifact_hash=artifact_hash,
        ),
        benchmark_artifact_revision=artifact_hash,
        benchmark=baseline.BenchmarkSpec(
            duration_seconds=5,
            margin_seconds=1,
            message_bytes=8 * 1024 * 1024,
            tx_depth=128,
            queue_pairs=1,
            mtu=4096,
            single_port_1_control_port=28515,
            single_port_2_control_port=28516,
            dual_port_control_port=28517,
        ),
        timeouts=baseline.TimeoutConfig(
            probe_seconds=5,
            opensm_start_seconds=2,
            rail_active_seconds=5,
            server_start_seconds=5,
            benchmark_seconds=15,
            cleanup_seconds=5,
            poll_seconds=0.01,
        ),
    )


def independent_config(result_directory: Path) -> baseline.BaselineConfig:
    raw = config(result_directory).model_dump(mode="json")
    raw["local_host"]["rail_cpu_bindings"] = {
        "port_1": [0],
        "port_2": [1],
    }
    raw["remote_host"]["rail_cpu_bindings"] = {
        "port_1": [0],
        "port_2": [1],
    }
    raw["local_host"]["hca"]["expected_health_counters"] = list(HEALTH_COUNTERS)
    raw["remote_host"]["hca"]["expected_health_counters"] = list(HEALTH_COUNTERS)
    raw["benchmark"]["independent_port_1_control_port"] = 28518
    raw["benchmark"]["independent_port_2_control_port"] = 28519
    return baseline.BaselineConfig.model_validate_json(json.dumps(raw))


def observation(
    host_config: baseline.HostConfig,
    *,
    counter: int = 100,
    unused_reserved_ports: tuple[int, ...] = (28515, 28516, 28517),
) -> baseline.HostObservation:
    observed_ports = tuple(
        baseline.PortObservation(
            port=item.port,
            port_guid=item.port_guid,
            gid=item.gid,
            state="4: ACTIVE",
            physical_state="5: LinkUp",
            rate=item.expected_rate,
            lid=1 if host_config.transport == "local" else 2,
            sm_lid=1,
            counters={
                "port_xmit_data": counter,
                "port_rcv_data": counter,
                "port_xmit_packets": counter,
                "port_rcv_packets": counter,
                **{name: 0 for name in host_config.hca.expected_health_counters},
            },
        )
        for item in host_config.hca.ports
    )
    return baseline.HostObservation(
        hostname=host_config.name,
        hca_device=host_config.hca.device,
        node_guid=host_config.hca.node_guid,
        pci_bdf=host_config.hca.pci.bdf,
        vendor_id=host_config.hca.pci.vendor_id,
        device_id=host_config.hca.pci.device_id,
        driver=host_config.hca.pci.driver,
        current_width=host_config.hca.pci.current_width,
        maximum_width=host_config.hca.pci.maximum_width,
        current_speed_gtps=host_config.hca.pci.current_speed_gtps,
        maximum_speed_gtps=host_config.hca.pci.maximum_speed_gtps,
        numa_node=host_config.hca.pci.numa_node,
        ports=cast(
            tuple[baseline.PortObservation, baseline.PortObservation], observed_ports
        ),
        ib_write_bw_sha256=host_config.tools.ib_write_bw_sha256,
        source=host_config.source,
        numa_cpu_sets={
            str(node): host_config.cpu_set for node in host_config.numa_nodes
        },
        online_cpu_count=8,
        load_average_1m=0.25,
        load_average_5m=0.2,
        load_average_15m=0.1,
        memory_total_bytes=1024 * 1024 * 1024,
        memory_available_bytes=512 * 1024 * 1024,
        memory_free_bytes=256 * 1024 * 1024,
        cpu_frequency_governors={
            str(cpu): "performance" for cpu in host_config.cpu_set
        },
        cpu_flags=("amx_bf16", "amx_int8", "amx_tile"),
        raid_sync_conflicts=(),
        gpu_bindings=(),
        unused_reserved_ports=unused_reserved_ports,
        conflicts=(),
    )


def result_directory(path: Path) -> baseline.ResultDirectory:
    path.mkdir()
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return baseline.ResultDirectory(path, descriptor)
    finally:
        os.close(descriptor)


def test_current_profile_binds_post_swap_x8_identities() -> None:
    profile = baseline.CURRENT_CX3_IDENTITIES
    assert profile["dwagon"]["pci"]["bdf"] == "0000:d8:00.0"  # type: ignore[index]
    assert profile["fwuff"]["pci"]["bdf"] == "0000:16:00.0"  # type: ignore[index]
    assert profile["dwagon"]["pci"]["current_width"] == 8  # type: ignore[index]
    assert profile["fwuff"]["pci"]["current_width"] == 8  # type: ignore[index]


def test_config_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    configured = config(tmp_path / "run")
    raw = configured.model_dump(mode="json")
    raw["remote_host"]["tools"]["ib_write_bw_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="artifact revision"):
        baseline.BaselineConfig.model_validate_json(json.dumps(raw))


def test_independent_config_requires_two_ports_and_disjoint_host_bindings(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    raw = configured.model_dump(mode="json")
    raw["benchmark"]["independent_port_1_control_port"] = 28518
    with pytest.raises(ValidationError, match="configured together"):
        baseline.BaselineConfig.model_validate_json(json.dumps(raw))

    raw = independent_config(tmp_path / "independent").model_dump(mode="json")
    raw["remote_host"]["rail_cpu_bindings"]["port_2"] = [0]
    with pytest.raises(ValidationError, match="must be disjoint"):
        baseline.BaselineConfig.model_validate_json(json.dumps(raw))


def test_legacy_config_keeps_three_ports_and_independent_config_appends_two(
    tmp_path: Path,
) -> None:
    assert config(tmp_path / "legacy").reserved_ports == (28515, 28516, 28517)
    assert independent_config(tmp_path / "independent").reserved_ports == (
        28515,
        28516,
        28517,
        28518,
        28519,
    )


def test_parse_single_port_output() -> None:
    rows = baseline.parse_ib_write_bw_output(
        "#bytes iterations peak average msg\n8388608 3758 0.00 28.56 0.000425\n"
    )
    assert len(rows) == 1
    assert rows[0].average_gigabits_per_second == 28.56
    assert rows[0].port is None


def test_parse_exact_native_dual_port_nine_column_output() -> None:
    rows = baseline.parse_ib_write_bw_output(
        "8388608 3758 0.00 28.02 0.000418 14.01 0.000209 14.01 0.000209\n"
    )
    assert [(row.port, row.average_gigabits_per_second) for row in rows] == [
        (None, 28.02),
        (1, 14.01),
        (2, 14.01),
    ]


def test_health_counter_deltas_allow_nonzero_baseline_and_reject_increase(
    tmp_path: Path,
) -> None:
    configured = independent_config(tmp_path / "run")
    before = observation(configured.remote_host, counter=100)
    before_ports = tuple(
        item.model_copy(
            update={
                "counters": {
                    **item.counters,
                    "link_downed": item.port,
                }
            }
        )
        for item in before.ports
    )
    before = before.model_copy(update={"ports": before_ports})
    after = observation(configured.remote_host, counter=110)
    after_ports = tuple(
        item.model_copy(
            update={
                "counters": {
                    **item.counters,
                    "link_downed": item.port,
                }
            }
        )
        for item in after.ports
    )
    after = after.model_copy(update={"ports": after_ports})
    deltas = baseline.counter_delta(before, after)
    port_1_deltas = cast(dict[str, baseline.JsonValue], deltas["1"])
    assert port_1_deltas["link_downed"] == 0
    assert port_1_deltas["port_xmit_data"] == 10

    increased_counters = dict(after.ports[0].counters)
    increased_counters["link_downed"] = 2
    increased_port = after.ports[0].model_copy(update={"counters": increased_counters})
    increased = after.model_copy(update={"ports": (increased_port, after.ports[1])})
    with pytest.raises(baseline.BaselineError, match="health counter link_downed"):
        baseline.counter_delta(before, increased)


def test_counter_reader_discovers_hw_counter_and_binds_required_set(
    tmp_path: Path,
) -> None:
    standard = tmp_path / "counters"
    hardware = tmp_path / "hw_counters"
    standard.mkdir()
    hardware.mkdir()
    (hardware / "symbol_error").write_text("7\n")
    assert (
        baseline.read_port_counter(
            (standard, hardware), "symbol_error", port=1, required=True
        )
        == 7
    )
    assert (
        baseline.read_port_counter(
            (standard, hardware), "link_downed", port=1, required=False
        )
        is None
    )
    with pytest.raises(baseline.BaselineError, match="required port 1 counter"):
        baseline.read_port_counter(
            (standard, hardware), "link_downed", port=1, required=True
        )


def test_perftest_commands_use_native_dual_port_and_management_address(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    server = baseline.perftest_command(
        configured,
        configured.remote_host,
        port=1,
        control_port=28517,
        dual_port=True,
        server=True,
    )
    assert server[:4] == (
        "/usr/bin/numactl",
        "--physcpubind=0,1",
        f"--membind={configured.remote_host.hca.pci.numa_node}",
        "/usr/bin/ib_write_bw",
    )
    client = baseline.perftest_command(
        configured,
        configured.local_host,
        port=1,
        control_port=28517,
        dual_port=True,
        server=False,
    )
    assert "--dualport" in server and "--report-per-port" in server
    assert server[-1] == "--report-per-port"
    assert client[-1] == "192.168.40.248"


def test_independent_commands_use_matching_per_rail_cpu_bindings(
    tmp_path: Path,
) -> None:
    configured = independent_config(tmp_path / "run")
    local_bindings = configured.local_host.rail_cpu_bindings
    remote_bindings = configured.remote_host.rail_cpu_bindings
    assert local_bindings is not None and remote_bindings is not None
    for port_number, control_port in zip(
        (cast(Literal[1, 2], 1), cast(Literal[1, 2], 2)),
        cast(tuple[int, int], configured.benchmark.independent_control_ports),
        strict=True,
    ):
        server = baseline.perftest_command(
            configured,
            configured.remote_host,
            port=port_number,
            control_port=control_port,
            dual_port=False,
            server=True,
            cpu_set=remote_bindings.for_port(port_number),
        )
        client = baseline.perftest_command(
            configured,
            configured.local_host,
            port=port_number,
            control_port=control_port,
            dual_port=False,
            server=False,
            cpu_set=local_bindings.for_port(port_number),
        )
        assert server[1] == f"--physcpubind={port_number - 1}"
        assert client[1] == f"--physcpubind={port_number - 1}"
        baseline.parse_remote_request(
            baseline.build_remote_request(
                configured.remote_host,
                server,
                "run:owner-token",
                configured.namespace,
                30.0,
                configured.reserved_ports,
            )
        )
    mismatched_server = baseline.perftest_command(
        configured,
        configured.remote_host,
        port=1,
        control_port=cast(
            tuple[int, int], configured.benchmark.independent_control_ports
        )[0],
        dual_port=False,
        server=True,
        cpu_set=remote_bindings.port_2,
    )
    with pytest.raises(baseline.BaselineError, match="exact bindings|rail CPU binding"):
        baseline.parse_remote_request(
            baseline.build_remote_request(
                configured.remote_host,
                mismatched_server,
                "run:owner-token",
                configured.namespace,
                30.0,
                configured.reserved_ports,
            )
        )


def test_remote_supervisor_request_round_trips_strict_json_config(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    command = baseline.perftest_command(
        configured,
        configured.remote_host,
        port=1,
        control_port=28515,
        dual_port=False,
        server=True,
    )
    request = baseline.build_remote_request(
        configured.remote_host,
        command,
        "run:owner-token",
        configured.namespace,
        30.0,
        configured.reserved_ports,
    )
    observed_host, observed_command, owner, namespace, timeout, reserved_ports = (
        baseline.parse_remote_request(request)
    )
    assert observed_host == configured.remote_host
    assert observed_command == command
    assert (owner, namespace, timeout) == (
        "run:owner-token",
        configured.namespace,
        30.0,
    )
    assert reserved_ports == configured.reserved_ports


def test_remote_supervisor_request_rejects_unreserved_command_port(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    command = baseline.perftest_command(
        configured,
        configured.remote_host,
        port=1,
        control_port=29999,
        dual_port=False,
        server=True,
    )
    request = baseline.build_remote_request(
        configured.remote_host,
        command,
        "run:owner-token",
        configured.namespace,
        30.0,
        configured.reserved_ports,
    )
    with pytest.raises(baseline.BaselineError, match="reserved control port"):
        baseline.parse_remote_request(request)


def test_host_probe_request_preserves_exact_reserved_ports(tmp_path: Path) -> None:
    configured = config(tmp_path / "run")
    request = baseline.HostProbeRequest(
        host=configured.remote_host, reserved_ports=configured.reserved_ports
    )
    observed = baseline.HostProbeRequest.model_validate_json(request.model_dump_json())
    assert observed.reserved_ports == (28515, 28516, 28517)
    skipped = baseline.HostProbeRequest(host=configured.remote_host, reserved_ports=())
    assert (
        baseline.HostProbeRequest.model_validate_json(
            skipped.model_dump_json()
        ).reserved_ports
        == ()
    )


def test_result_directory_rejects_path_replacement(tmp_path: Path) -> None:
    path = tmp_path / "run"
    results = result_directory(path)
    moved = tmp_path / "moved"
    path.rename(moved)
    path.mkdir()
    try:
        with pytest.raises(baseline.BaselineError, match="identity changed"):
            results.write_json(
                "runtime-metadata.json", {"schema_version": 1}, replace=True
            )
    finally:
        results.close()


def test_result_directory_writes_fragments_through_retained_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run"
    results = result_directory(path)
    try:
        results.write_json("runtime-metadata.json", {"schema_version": 1}, replace=True)
        results.write_json(
            "benchmark-result.json", {"schema_version": 1}, replace=False
        )
        assert json.loads((path / "runtime-metadata.json").read_text()) == {
            "schema_version": 1
        }
        assert json.loads((path / "benchmark-result.json").read_text()) == {
            "schema_version": 1
        }
    finally:
        results.close()


def test_process_conflicts_detects_perftest_without_matching_arguments(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "cmdline").write_bytes(b"/usr/bin/ib_write_bw\0")
    (proc / "123" / "environ").write_bytes(b"PATH=/usr/bin\0")
    assert "ib_write_bw" in baseline.process_conflicts(proc)[0]


def test_process_conflicts_allows_only_matching_owned_concurrent_sibling(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "cmdline").write_bytes(b"/usr/bin/ib_write_bw\0")
    (proc / "123" / "environ").write_bytes(
        b"EXO_BENCHMARK_NAMESPACE=ib-test\0EXO_BENCHMARK_OWNER_TOKEN=owner-123\0"
    )
    assert baseline.process_conflicts(proc) != ()
    assert (
        baseline.process_conflicts(proc, owned_identity=("ib-test", "owner-123")) == ()
    )
    assert baseline.process_conflicts(proc, owned_identity=("ib-test", "other")) != ()


@pytest.mark.parametrize(
    ("arguments", "environment", "expected_class"),
    [
        (("/usr/bin/ollama", "serve"), (), "exo_nccl_or_model_server"),
        (("/root/exo/.venv/bin/exo",), (), "exo_nccl_or_model_server"),
        (("python", "-m", "exo"), (), "exo_nccl_or_model_server"),
        (("/tmp/all_reduce_perf", "-b", "8"), (), "exo_nccl_or_model_server"),
        (
            ("python", "-m", "vllm.entrypoints.openai.api_server"),
            (),
            "exo_nccl_or_model_server",
        ),
        (
            (
                "/root/exo/.venv/bin/python",
                "-c",
                "from multiprocessing.spawn import spawn_main; "
                "spawn_main(tracker_fd=7, pipe_handle=11)",
                "--multiprocessing-fork",
            ),
            (),
            "exo_runner_or_python_multiprocessing_worker",
        ),
        (("/usr/bin/hf", "download", "org/model"), (), "heavy_storage"),
        (("/usr/bin/sha256sum", "/mnt/sanic/model"), (), "heavy_storage"),
        (("/usr/sbin/opensm",), (), "opensm_or_perftest"),
    ],
)
def test_process_classifier_covers_coordinate_conflict_classes(
    arguments: tuple[str, ...],
    environment: tuple[str, ...],
    expected_class: str,
) -> None:
    assert expected_class in baseline.classify_process_conflict(arguments, environment)


@pytest.mark.parametrize(
    "arguments",
    [
        (
            "/root/exo/.venv/bin/python",
            "/root/exo/scripts/two_host_ib_baseline.py",
            "remote-supervise",
        ),
        (
            "/usr/bin/ssh",
            "fwuff",
            "/root/exo/.venv/bin/python /root/exo/scripts/two_host_ib_baseline.py host-probe",
        ),
        ("python", "-c", "from multiprocessing.spawn import spawn_main; spawn_main()"),
    ],
)
def test_process_classifier_does_not_match_harness_or_generic_python(
    arguments: tuple[str, ...],
) -> None:
    assert baseline.classify_process_conflict(arguments) == ()


@pytest.mark.parametrize(
    "occupied_socket_type", [socket.SOCK_STREAM, socket.SOCK_DGRAM]
)
def test_reserved_port_probe_rejects_occupied_tcp_and_udp(
    occupied_socket_type: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    occupied = True
    observed_options: list[tuple[int, int, int]] = []

    class FakeSocket:
        def __init__(self, family: int, socket_type: int) -> None:
            self.family = family
            self.socket_type = socket_type

        def setsockopt(self, level: int, option: int, value: int) -> None:
            observed_options.append((level, option, value))

        def bind(self, address: tuple[str, int]) -> None:
            del address
            if (
                occupied
                and self.family == socket.AF_INET
                and self.socket_type == occupied_socket_type
            ):
                raise OSError("address already in use")

        def listen(self, backlog: int) -> None:
            del backlog

        def close(self) -> None:
            pass

    monkeypatch.setattr(baseline.socket, "socket", FakeSocket)
    with pytest.raises(baseline.PortConflictError, match="availability probe"):
        baseline.probe_reserved_ports_unused((28515,))
    occupied = False
    assert baseline.probe_reserved_ports_unused((28515,)) == (28515,)
    assert not any(
        level == socket.SOL_SOCKET and option == socket.SO_REUSEADDR
        for level, option, _ in observed_options
    )


def test_empty_reserved_port_request_skips_binding_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(ports: Sequence[int]) -> tuple[int, ...]:
        del ports
        raise AssertionError("empty request must not claim a port availability probe")

    monkeypatch.setattr(baseline, "probe_reserved_ports_unused", unexpected_probe)
    assert baseline.probe_requested_reserved_ports(()) == ()


def test_remote_host_probe_transports_empty_request_and_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = config(tmp_path / "run")
    results = result_directory(Path(configured.result_directory))
    requests: list[baseline.HostProbeRequest] = []

    def fake_run(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raw_request = kwargs.get("input")
        assert isinstance(raw_request, str)
        request = baseline.HostProbeRequest.model_validate_json(raw_request)
        requests.append(request)
        observed = observation(request.host, unused_reserved_ports=())
        return subprocess.CompletedProcess(
            tuple(command), 0, stdout=observed.model_dump_json(), stderr=""
        )

    monkeypatch.setattr(baseline.subprocess, "run", fake_run)
    try:
        effects = baseline.SystemEffects(configured, results)
        observed = effects.probe_remote(configured.remote_host, reserved_ports=())
    finally:
        results.close()
    assert [request.reserved_ports for request in requests] == [()]
    assert observed.unused_reserved_ports == ()


def test_remote_supervisor_rechecks_only_current_port_before_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = config(tmp_path / "run")
    command = baseline.perftest_command(
        configured,
        configured.remote_host,
        port=1,
        control_port=28515,
        dual_port=False,
        server=True,
    )
    request = baseline.build_remote_request(
        configured.remote_host,
        command,
        "run:owner-token",
        configured.namespace,
        30.0,
        configured.reserved_ports,
    )
    observed_ports: list[tuple[int, ...]] = []

    def no_conflicts(
        proc_root: Path = Path("/proc"),
        *,
        ignored_pids: Sequence[int] = (),
        owned_identity: tuple[str, str] | None = None,
    ) -> tuple[str, ...]:
        del proc_root, ignored_pids, owned_identity
        return ()

    def reject_ports(ports: Sequence[int]) -> tuple[int, ...]:
        observed_ports.append(tuple(ports))
        raise baseline.PortConflictError("occupied in test")

    monkeypatch.setattr(baseline, "process_conflicts", no_conflicts)
    monkeypatch.setattr(baseline, "probe_reserved_ports_unused", reject_ports)
    monkeypatch.setattr(
        baseline.sys,
        "stdin",
        StringIO(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"),
    )
    assert baseline.remote_supervise_main() == 70
    assert observed_ports == [(configured.benchmark.single_port_1_control_port,)]


def test_local_client_rechecks_only_current_port_before_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = config(tmp_path / "run")
    run_path = Path(configured.result_directory)
    results = result_directory(run_path)
    observed_ports: list[tuple[int, ...]] = []

    def reject_ports(ports: Sequence[int]) -> tuple[int, ...]:
        observed_ports.append(tuple(ports))
        raise baseline.PortConflictError("occupied in test")

    monkeypatch.setattr(baseline, "probe_reserved_ports_unused", reject_ports)
    command = baseline.perftest_command(
        configured,
        configured.local_host,
        port=2,
        control_port=configured.benchmark.single_port_2_control_port,
        dual_port=False,
        server=False,
    )
    try:
        effects = baseline.SystemEffects(configured, results)
        with pytest.raises(baseline.PortConflictError, match="occupied in test"):
            effects.start_local_client(command, "test-client", "run:owner-token")
    finally:
        results.close()
    assert observed_ports == [(configured.benchmark.single_port_2_control_port,)]


def test_remote_server_rejects_missing_current_port_before_transport(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    results = result_directory(Path(configured.result_directory))
    try:
        effects = baseline.SystemEffects(configured, results)
        with pytest.raises(baseline.BaselineError, match="numeric control port"):
            effects.start_remote_server(
                ("/usr/bin/ib_write_bw",),
                "test-server",
                "run:owner-token",
                30.0,
            )
    finally:
        results.close()


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (("--port=28515", "--port=28516"), "exactly one numeric control port"),
        (("--port=28x15",), "exactly one numeric control port"),
        (("--port=" + chr(0x661),), "exactly one numeric control port"),
        (("--port=28518",), "reserved control port"),
    ],
)
def test_control_port_parser_rejects_ambiguous_or_unreserved_values(
    arguments: tuple[str, ...], expected_error: str
) -> None:
    with pytest.raises(baseline.BaselineError, match=expected_error):
        baseline.reserved_control_port_from_command(
            ("/usr/bin/ib_write_bw", *arguments), (28515, 28516, 28517)
        )


def test_ssh_argv_ignores_mutable_configuration_and_pins_transport(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    assert baseline.build_ssh_argv(configured, "host-probe") == (
        "/usr/bin/ssh",
        "-F",
        "/dev/null",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=/root/.ssh/known_hosts",
        "-o",
        "IdentityFile=/root/.ssh/id_ed25519",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ProxyCommand=none",
        "-o",
        "ProxyJump=none",
        "-o",
        "CanonicalizeHostname=no",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "RequestTTY=no",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "PubkeyAuthentication=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "HostName=fwuff",
        "-l",
        "root",
        "-p",
        "22",
        "--",
        "fwuff",
        "/root/exo/.venv/bin/python /root/exo/scripts/two_host_ib_baseline.py host-probe",
    )


def test_host_config_requires_remote_harness_from_exact_source_tree(
    tmp_path: Path,
) -> None:
    configured = config(tmp_path / "run")
    raw = configured.remote_host.model_dump(mode="json")
    raw["tools"]["harness_script"] = "/tmp/two_host_ib_baseline.py"
    with pytest.raises(ValidationError, match="exact script"):
        baseline.HostConfig.model_validate_json(json.dumps(raw))


def test_stop_owned_ssh_transport_confirms_group_exit() -> None:
    owner_token = "test-owner"
    namespace = "test-namespace"
    process = subprocess.Popen(
        ("/usr/bin/sleep", "60"),
        text=True,
        env={
            "PATH": os.defpath,
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "EXO_BENCHMARK_NAMESPACE": namespace,
        },
        start_new_session=True,
    )
    process_group_id, start_ticks = baseline.process_identity(process.pid)
    receipt = baseline.OwnedProcess(
        host_name="dwagon",
        kind="test-ssh-transport",
        pid=process.pid,
        process_group_id=process_group_id,
        start_time_ticks=start_ticks,
        transport_pid=process.pid,
        namespace=namespace,
        owner_token=owner_token,
        log_path="/tmp/test.log",
    )
    cleanup = baseline.stop_owned_ssh_transport(process, receipt, 2.0)
    assert cleanup.ownership_verified is True
    assert cleanup.terminated is True
    assert process.poll() is not None


def test_stop_owned_ssh_transport_never_reports_a_survivor_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_token = "test-owner"
    namespace = "test-namespace"
    process = subprocess.Popen(
        ("/usr/bin/sleep", "60"),
        text=True,
        env={
            "PATH": os.defpath,
            "EXO_BENCHMARK_OWNER_TOKEN": owner_token,
            "EXO_BENCHMARK_NAMESPACE": namespace,
        },
        start_new_session=True,
    )
    process_group_id, start_ticks = baseline.process_identity(process.pid)
    receipt = baseline.OwnedProcess(
        host_name="dwagon",
        kind="test-ssh-transport",
        pid=process.pid,
        process_group_id=process_group_id,
        start_time_ticks=start_ticks,
        transport_pid=process.pid,
        namespace=namespace,
        owner_token=owner_token,
        log_path="/tmp/test.log",
    )
    real_killpg = os.killpg

    def leave_group_running(process_group: int, signal_number: int) -> None:
        del process_group, signal_number

    monkeypatch.setattr(baseline.os, "killpg", leave_group_running)
    try:
        cleanup = baseline.stop_owned_ssh_transport(process, receipt, 0.01)
        assert cleanup.ownership_verified is True
        assert cleanup.terminated is False
        assert cleanup.error == "SSH transport process group survived cleanup"
    finally:
        monkeypatch.undo()
        real_killpg(process_group_id, baseline.signal.SIGKILL)
        process.wait(timeout=2.0)


class FakeEffects:
    def __init__(self, configured: baseline.BaselineConfig) -> None:
        self.config = configured
        self.next_pid = 1000
        self.probe_requests: list[tuple[str, tuple[int, ...]]] = []
        self.events: list[str] = []

    def probe_local(
        self,
        host: baseline.HostConfig,
        *,
        reserved_ports: Sequence[int],
        ignored_pids: Sequence[int] = (),
    ) -> baseline.HostObservation:
        del ignored_pids
        normalized = tuple(reserved_ports)
        self.probe_requests.append((host.name, normalized))
        return observation(
            host,
            counter=self.next_pid,
            unused_reserved_ports=normalized,
        )

    def probe_remote(
        self, host: baseline.HostConfig, *, reserved_ports: Sequence[int]
    ) -> baseline.HostObservation:
        normalized = tuple(reserved_ports)
        self.probe_requests.append((host.name, normalized))
        return observation(
            host,
            counter=self.next_pid,
            unused_reserved_ports=normalized,
        )

    def receipt(self, host_name: str, kind: str) -> baseline.OwnedProcess:
        self.next_pid += 1
        return baseline.OwnedProcess(
            host_name=host_name,
            kind=kind,
            pid=self.next_pid,
            process_group_id=self.next_pid,
            start_time_ticks=self.next_pid * 10,
            transport_pid=self.next_pid,
            namespace=self.config.namespace,
            owner_token=self.owner_token,
            log_path=f"{self.config.result_directory}/{kind}.log",
        )

    owner_token: str = ""

    def start_opensm(
        self, port: baseline.PortIdentity, owner_token: str
    ) -> baseline.LocalHandle:
        self.owner_token = owner_token
        receipt = self.receipt(self.config.local_host.name, f"opensm-port-{port.port}")
        return baseline.LocalHandle(
            receipt,
            cast(subprocess.Popen[bytes], cast(object, None)),
            BytesIO(),
            f"opensm-{port.port}.log",
        )

    def start_remote_server(
        self,
        command: Sequence[str],
        kind: str,
        owner_token: str,
        timeout_seconds: float,
    ) -> baseline.RemoteHandle:
        self.owner_token = owner_token
        self.events.append(f"start-remote:{kind}")
        receipt = self.receipt(self.config.remote_host.name, kind)
        transport_receipt = self.receipt(
            self.config.local_host.name, f"{kind}-ssh-transport"
        )
        return baseline.RemoteHandle(
            receipt,
            transport_receipt,
            cast(subprocess.Popen[str], cast(object, None)),
            f"{kind}.log",
            {},
        )

    def start_local_client(
        self, command: Sequence[str], kind: str, owner_token: str
    ) -> baseline.LocalHandle:
        self.owner_token = owner_token
        self.events.append(f"start-local:{kind}")
        receipt = self.receipt(self.config.local_host.name, kind)
        return baseline.LocalHandle(
            receipt,
            cast(subprocess.Popen[bytes], cast(object, None)),
            BytesIO(),
            f"{kind}.log",
        )

    @staticmethod
    def cleanup(receipt: baseline.OwnedProcess) -> baseline.CleanupReceipt:
        return baseline.CleanupReceipt(
            receipt.host_name, receipt.kind, True, True, False
        )

    def wait_local(
        self, handle: baseline.LocalHandle, timeout_seconds: float
    ) -> tuple[int, str, baseline.CleanupReceipt]:
        self.events.append(f"wait-local:{handle.receipt.kind}")
        if "native-dual" in handle.receipt.kind:
            output = "8388608 3758 0.00 28.02 0.000418 14.01 0.000209 14.01 0.000209\n"
        elif "independent-concurrent-port-1" in handle.receipt.kind:
            output = "8388608 3758 0.00 20.00 0.000300\n"
        elif "independent-concurrent-port-2" in handle.receipt.kind:
            output = "8388608 3758 0.00 21.00 0.000310\n"
        else:
            output = "8388608 3758 0.00 28.56 0.000425\n"
        return 0, output, self.cleanup(handle.receipt)

    def wait_remote(
        self, handle: baseline.RemoteHandle, timeout_seconds: float
    ) -> tuple[int, str, baseline.CleanupReceipt]:
        self.events.append(f"wait-remote:{handle.receipt.kind}")
        return 0, "server complete", self.cleanup(handle.receipt)

    def stop_local(
        self, handle: baseline.LocalHandle, timeout_seconds: float
    ) -> baseline.CleanupReceipt:
        self.events.append(f"stop-local:{handle.receipt.kind}")
        return self.cleanup(handle.receipt)

    def stop_remote(
        self, handle: baseline.RemoteHandle, timeout_seconds: float
    ) -> baseline.CleanupReceipt:
        self.events.append(f"stop-remote:{handle.receipt.kind}")
        return self.cleanup(handle.receipt)


def test_run_harness_records_three_cases_and_confirms_all_cleanup(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-1"
    configured = config(run_path)
    results = result_directory(run_path)
    effects = FakeEffects(configured)
    try:
        outcome = baseline.run_harness(configured, effects, results)
        assert outcome["status"] == "completed"
        assert outcome["cleanup_succeeded"] is True
        cases = cast(list[baseline.JsonObject], cast(object, outcome["cases"]))
        assert [item["name"] for item in cases] == [
            "single-port-1",
            "single-port-2",
            "native-dual-port",
        ]
        owned_processes = cast(
            list[baseline.JsonObject], cast(object, outcome["owned_processes"])
        )
        assert len(owned_processes) == 8
        assert (
            json.loads((run_path / "benchmark-result.json").read_text())["reportable"]
            is True
        )
        assert effects.probe_requests[:2] == [
            (configured.local_host.name, configured.reserved_ports),
            (configured.remote_host.name, configured.reserved_ports),
        ]
        assert effects.probe_requests[2:] == [
            (host_name, ())
            for _ in range(8)
            for host_name in (
                configured.local_host.name,
                configured.remote_host.name,
            )
        ]
    finally:
        results.close()


def test_independent_concurrent_case_launches_both_pairs_and_aggregates_rows(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-independent"
    configured = independent_config(run_path)
    results = result_directory(run_path)
    effects = FakeEffects(configured)
    try:
        outcome = baseline.run_harness(configured, effects, results)
        assert outcome["status"] == "completed"
        assert outcome["reportable"] is True
        cases = cast(list[baseline.JsonObject], cast(object, outcome["cases"]))
        assert [item["name"] for item in cases] == [
            "single-port-1",
            "single-port-2",
            "native-dual-port",
            "independent-concurrent",
        ]
        concurrent = cases[-1]
        assert concurrent["aggregate_average_gigabits_per_second"] == 41.0
        rails = cast(list[baseline.JsonObject], cast(object, concurrent["rails"]))
        assert [rail["port"] for rail in rails] == [1, 2]
        assert [rail["maximum_average_gigabits_per_second"] for rail in rails] == [
            20.0,
            21.0,
        ]
        assert rails[0]["cpu_bindings"] == {"dwagon": [0], "fwuff": [0]}
        assert rails[1]["cpu_bindings"] == {"dwagon": [1], "fwuff": [1]}
        owned_processes = cast(
            list[baseline.JsonObject], cast(object, outcome["owned_processes"])
        )
        assert len(owned_processes) == 12
        independent_events = [
            event for event in effects.events if "independent-concurrent" in event
        ]
        assert independent_events == [
            "start-remote:independent-concurrent-port-1-server",
            "start-remote:independent-concurrent-port-2-server",
            "start-local:independent-concurrent-port-1-client",
            "start-local:independent-concurrent-port-2-client",
            "wait-local:independent-concurrent-port-1-client",
            "wait-local:independent-concurrent-port-2-client",
            "wait-remote:independent-concurrent-port-1-server",
            "wait-remote:independent-concurrent-port-2-server",
        ]
        current_ports = cast(
            tuple[int, int], configured.benchmark.independent_control_ports
        )
        assert effects.probe_requests.count(("dwagon", current_ports)) == 1
        assert effects.probe_requests.count(("fwuff", current_ports)) == 1
        counter_deltas = cast(
            dict[str, baseline.JsonValue], concurrent["counter_deltas"]
        )
        fwuff_deltas = cast(dict[str, baseline.JsonValue], counter_deltas["fwuff"])
        fwuff_port_1 = cast(dict[str, baseline.JsonValue], fwuff_deltas["1"])
        assert {name: fwuff_port_1[name] for name in HEALTH_COUNTERS} == {
            name: 0 for name in HEALTH_COUNTERS
        }
    finally:
        results.close()


class FailSecondIndependentClientEffects(FakeEffects):
    def start_local_client(
        self, command: Sequence[str], kind: str, owner_token: str
    ) -> baseline.LocalHandle:
        if kind == "independent-concurrent-port-2-client":
            self.events.append(f"start-local-failed:{kind}")
            raise baseline.BaselineError("synthetic second-client launch failure")
        return super().start_local_client(command, kind, owner_token)


class FailSecondIndependentServerEffects(FakeEffects):
    def start_remote_server(
        self,
        command: Sequence[str],
        kind: str,
        owner_token: str,
        timeout_seconds: float,
    ) -> baseline.RemoteHandle:
        if kind == "independent-concurrent-port-2-server":
            self.events.append(f"start-remote-failed:{kind}")
            raise baseline.BaselineError("synthetic second-server launch failure")
        return super().start_remote_server(command, kind, owner_token, timeout_seconds)


def test_independent_concurrent_second_server_failure_cleans_first_server(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-second-server-failure"
    configured = independent_config(run_path)
    results = result_directory(run_path)
    effects = FailSecondIndependentServerEffects(configured)
    try:
        outcome = baseline.run_harness(configured, effects, results)
        assert outcome["status"] == "benchmark_failed"
        assert outcome["reportable"] is False
        assert outcome["cleanup_succeeded"] is True
        assert [
            event
            for event in effects.events
            if event.startswith("stop-") and "independent-concurrent" in event
        ] == ["stop-remote:independent-concurrent-port-1-server"]
    finally:
        results.close()


def test_independent_concurrent_partial_launch_cleans_every_owned_process(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-partial-failure"
    configured = independent_config(run_path)
    results = result_directory(run_path)
    effects = FailSecondIndependentClientEffects(configured)
    try:
        outcome = baseline.run_harness(configured, effects, results)
        assert outcome["status"] == "benchmark_failed"
        assert outcome["reportable"] is False
        assert outcome["cleanup_succeeded"] is True
        independent_stops = [
            event
            for event in effects.events
            if event.startswith("stop-") and "independent-concurrent" in event
        ]
        assert independent_stops == [
            "stop-local:independent-concurrent-port-1-client",
            "stop-remote:independent-concurrent-port-2-server",
            "stop-remote:independent-concurrent-port-1-server",
        ]
    finally:
        results.close()


class UnconfirmedIndependentCleanupEffects(FakeEffects):
    def wait_local(
        self, handle: baseline.LocalHandle, timeout_seconds: float
    ) -> tuple[int, str, baseline.CleanupReceipt]:
        code, output, cleanup = super().wait_local(handle, timeout_seconds)
        if handle.receipt.kind == "independent-concurrent-port-2-client":
            cleanup = baseline.CleanupReceipt(
                cleanup.host_name,
                cleanup.kind,
                True,
                False,
                False,
                "synthetic survivor",
            )
        return code, output, cleanup


class HealthDeltaIndependentEffects(FakeEffects):
    def probe_local(
        self,
        host: baseline.HostConfig,
        *,
        reserved_ports: Sequence[int],
        ignored_pids: Sequence[int] = (),
    ) -> baseline.HostObservation:
        observed = super().probe_local(
            host,
            reserved_ports=reserved_ports,
            ignored_pids=ignored_pids,
        )
        if "wait-remote:independent-concurrent-port-2-server" not in self.events:
            return observed
        counters = dict(observed.ports[0].counters)
        counters["symbol_error"] = 1
        changed_port = observed.ports[0].model_copy(update={"counters": counters})
        return observed.model_copy(update={"ports": (changed_port, observed.ports[1])})


def test_independent_concurrent_unconfirmed_cleanup_is_not_reportable(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-cleanup-failure"
    configured = independent_config(run_path)
    results = result_directory(run_path)
    try:
        outcome = baseline.run_harness(
            configured, UnconfirmedIndependentCleanupEffects(configured), results
        )
        assert outcome["status"] == "cleanup_failed"
        assert outcome["cleanup_succeeded"] is False
        assert outcome["reportable"] is False
    finally:
        results.close()


def test_independent_concurrent_health_delta_is_not_reportable(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run-health-delta"
    configured = independent_config(run_path)
    results = result_directory(run_path)
    try:
        outcome = baseline.run_harness(
            configured, HealthDeltaIndependentEffects(configured), results
        )
        assert outcome["status"] == "benchmark_failed"
        assert outcome["cleanup_succeeded"] is True
        assert outcome["reportable"] is False
        assert "health counter symbol_error increased" in cast(str, outcome["error"])
    finally:
        results.close()


def test_validate_observation_rejects_x4_regression(tmp_path: Path) -> None:
    configured = config(tmp_path / "run")
    observed = observation(configured.remote_host).model_copy(
        update={"current_width": 4}
    )
    with pytest.raises(baseline.BaselineError, match="current_width differs"):
        baseline.validate_host_observation(
            observed,
            configured.remote_host,
            reserved_ports=configured.reserved_ports,
            require_active=True,
        )


@pytest.mark.parametrize(
    ("updates", "expected_error", "match"),
    [
        ({"load_average_1m": 9.0}, baseline.BaselineError, "load per online CPU"),
        ({"memory_available_bytes": 0}, baseline.BaselineError, "available memory"),
        (
            {"cpu_frequency_governors": {"0": "powersave", "1": "performance"}},
            baseline.BaselineError,
            "disallowed CPU frequency governors",
        ),
        (
            {"cpu_flags": ("amx_bf16", "amx_tile")},
            baseline.BaselineError,
            "missing required CPU flags",
        ),
        (
            {"raid_sync_conflicts": ("md0:resync",)},
            baseline.ProcessConflictError,
            "active RAID work",
        ),
        (
            {"gpu_bindings": ("cuda:0",)},
            baseline.BaselineError,
            "nonempty GPU bindings",
        ),
    ],
)
def test_validate_observation_enforces_coordinate_host_preflight(
    tmp_path: Path,
    updates: dict[str, object],
    expected_error: type[Exception],
    match: str,
) -> None:
    configured = config(tmp_path / "run")
    observed = observation(configured.remote_host).model_copy(update=updates)
    with pytest.raises(expected_error, match=match):
        baseline.validate_host_observation(
            observed,
            configured.remote_host,
            reserved_ports=configured.reserved_ports,
            require_active=True,
        )


def test_cpu_list_parser_and_numa_binding_validation(tmp_path: Path) -> None:
    assert baseline.parse_cpu_list("0-2,8,10-11") == (0, 1, 2, 8, 10, 11)
    configured = config(tmp_path / "run")
    observed = observation(configured.remote_host).model_copy(
        update={"numa_cpu_sets": {"0": (8, 9)}}
    )
    with pytest.raises(baseline.BaselineError, match="outside configured NUMA"):
        baseline.validate_host_observation(
            observed,
            configured.remote_host,
            reserved_ports=configured.reserved_ports,
            require_active=True,
        )


def test_static_metadata_binds_config_digest_and_both_hcas(tmp_path: Path) -> None:
    configured = config(tmp_path / "run")
    command = (
        "/root/exo/.venv/bin/python",
        "/root/exo/scripts/two_host_ib_baseline.py",
    )
    metadata = baseline.build_static_metadata(configured, command, "f" * 64)
    assert metadata["benchmark_contract"]["config_sha256"] == "f" * 64  # type: ignore[index]
    assert set(metadata["hca_bindings"]) == {"dwagon", "fwuff"}  # type: ignore[arg-type]
    assert len(metadata["hca_bindings"]["dwagon"]) == 2  # type: ignore[index]


def test_prepare_lease_metadata_passes_real_wrapper_validator(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    result_root.mkdir()
    configured = config(result_root / "prepared-run")
    observed_source = baseline.read_source_identity(Path("/root/exo"))
    configured = configured.model_copy(
        update={
            "local_host": configured.local_host.model_copy(
                update={"source": observed_source}
            ),
            "remote_host": configured.remote_host.model_copy(
                update={"source": observed_source}
            ),
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(configured.model_dump_json(indent=2))
    metadata_path = tmp_path / "metadata.json"
    prepared = baseline.prepare_lease_metadata(
        config_path=config_path,
        metadata_output=metadata_path,
        wrapper_python=Path("/root/exo/.venv/bin/python"),
        child_python=Path("/root/exo/.venv/bin/python"),
        benchmark_lease_script=Path("/root/exo/scripts/benchmark_lease.py"),
        harness_script=Path("/root/exo/scripts/two_host_ib_baseline.py"),
        owner="codex-ib-baseline",
        purpose="post-slot-swap x8 baseline",
        expected_duration_seconds=900.0,
        cleanup_grace_seconds=baseline.minimum_cleanup_grace_seconds(configured),
        heartbeat_seconds=30.0,
        lease_path=tmp_path / "lease.json",
        lock_path=tmp_path / "lock",
        result_root=result_root,
    )
    assert json.loads(metadata_path.read_text()) == prepared.metadata
    assert prepared.metadata["cpu_bindings"]["dwagon"]["memory_policy"] == "bind:1"  # type: ignore[index]
    assert prepared.metadata["cpu_bindings"]["fwuff"]["memory_policy"] == "bind:0"  # type: ignore[index]
    assert (
        prepared.benchmark_lease_argv[-len(prepared.child_argv) :]
        == prepared.child_argv
    )


def test_artifact_hash_helper_uses_full_file(tmp_path: Path) -> None:
    artifact = tmp_path / "ib_write_bw"
    artifact.write_bytes(b"pinned perftest")
    assert (
        baseline.file_sha256(artifact) == hashlib.sha256(b"pinned perftest").hexdigest()
    )
