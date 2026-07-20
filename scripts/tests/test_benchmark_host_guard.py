from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from scripts import benchmark_host_guard as guard

SHA_A = "a" * 64
SHA_B = "b" * 64
NOW_NS = 1_000_000_000_000
REMOTE_GIDS = (
    "fe80:0000:0000:0000:0000:0000:0000:0001",
    "fe80:0000:0000:0000:0000:0000:0000:0002",
)
LOCAL_GIDS = (
    "fe80:0000:0000:0000:0000:0000:0000:0011",
    "fe80:0000:0000:0000:0000:0000:0000:0012",
)


def file_binding(path: str, sha256: str = SHA_A) -> guard.FileIdentityBinding:
    return guard.FileIdentityBinding(
        path=path,
        resolved_path=path,
        sha256=sha256,
    )


def file_observation(
    binding: guard.FileIdentityBinding,
) -> guard.FileIdentityObservation:
    return guard.FileIdentityObservation(
        path=binding.path,
        resolved_path=binding.resolved_path,
        size_bytes=123,
        sha256=binding.sha256,
    )


def peer_binding(
    *,
    ssh_files: tuple[guard.FileIdentityBinding, ...] | None = None,
) -> guard.CoordinationPeerBinding:
    ssh, known_hosts, identity = ssh_files or (
        file_binding("/usr/bin/ssh"),
        file_binding("/etc/ssh/known_hosts"),
        file_binding("/root/.ssh/id_ed25519"),
    )
    opensm = file_binding("/usr/sbin/opensm", SHA_B)
    return guard.CoordinationPeerBinding(
        hostname="fwuff",
        ssh=guard.SshTransportBinding(
            executable=ssh,
            target="fwuff",
            user="root",
            port=22,
            known_hosts_file=known_hosts,
            identity_file=identity,
            connect_timeout_seconds=5,
            server_alive_interval_seconds=5,
            server_alive_count_max=2,
        ),
        remote_probe=guard.RemoteProbeIdentityBinding(
            python=file_binding("/usr/bin/python3"),
            script=file_binding("/opt/exo/benchmark_host_guard.py"),
        ),
        tools=guard.HostToolBindings(
            nvidia_smi=file_binding("/usr/bin/nvidia-smi"),
            systemctl=file_binding("/usr/bin/systemctl"),
        ),
        cpu_memory=guard.CpuMemoryBinding(
            minimum_online_cpu_count=4,
            numa_cpu_sets={"0": (0, 1), "1": (2, 3)},
            minimum_total_memory_bytes=128 * 1024**3,
        ),
        gpus=(
            guard.GpuBinding(
                uuid="GPU-11111111-2222-3333-4444-555555555555",
                pci_bus_id="00000000:01:00.0",
                name="NVIDIA GeForce RTX 3090",
                memory_total_bytes=24 * 1024**3,
            ),
        ),
        hca=guard.HcaBinding(
            device="mlx4_0",
            node_guid="e41d:2d03:004d:32e1",
            ports=(
                guard.HcaPortBinding(
                    port=1,
                    gid_index=0,
                    gid=REMOTE_GIDS[0],
                    expected_rate="40 Gb/sec (4X QDR)",
                    health_counter_maximums={"link_downed": 0},
                    idle_data_counter_maximum_deltas={
                        "port_rcv_data": 64,
                        "port_rcv_packets": 8,
                        "port_xmit_data": 64,
                        "port_xmit_packets": 8,
                    },
                ),
                guard.HcaPortBinding(
                    port=2,
                    gid_index=0,
                    gid=REMOTE_GIDS[1],
                    expected_rate="40 Gb/sec (4X QDR)",
                    health_counter_maximums={"link_downed": 0},
                    idle_data_counter_maximum_deltas={
                        "port_rcv_data": 64,
                        "port_rcv_packets": 8,
                        "port_xmit_data": 64,
                        "port_xmit_packets": 8,
                    },
                ),
            ),
        ),
        opensm_units=(
            guard.OpenSmUnitBinding(
                unit="opensm-port1.service",
                port=1,
                guid="0xe41d2d03004d32e1",
                executable=opensm,
                argv=(opensm.path, "--guid", "0xe41d2d03004d32e1"),
                version="OpenSM 3.3.24",
            ),
            guard.OpenSmUnitBinding(
                unit="opensm-port2.service",
                port=2,
                guid="0xe41d2d03004d32e2",
                executable=opensm,
                argv=(opensm.path, "--guid", "0xe41d2d03004d32e2"),
                version="OpenSM 3.3.24",
            ),
        ),
        reserved_ports=(62274, 62275),
        policy=guard.IdlePeerPolicy(
            maximum_load_1m_per_online_cpu=0.5,
            minimum_available_memory_bytes=64 * 1024**3,
            maximum_gpu_memory_used_bytes=128 * 1024**2,
            maximum_gpu_utilization_percent=2,
            maximum_gpu_memory_utilization_percent=2,
            maximum_gpu_temperature_celsius=70,
            maximum_clock_skew_ns=1_000_000_000,
        ),
    )


def fabric_binding() -> guard.CrossHostFabricBinding:
    managers = peer_binding().opensm_units
    return guard.CrossHostFabricBinding(
        rails=(
            guard.FabricRailBinding(
                local_port=1,
                remote_port=1,
                local_gid=LOCAL_GIDS[0],
                remote_gid=REMOTE_GIDS[0],
                rate="40 Gb/sec (4X QDR)",
                subnet_manager_host="remote",
                subnet_manager_unit=managers[0].unit,
                subnet_manager_guid=managers[0].guid,
                subnet_manager_argv=managers[0].argv,
            ),
            guard.FabricRailBinding(
                local_port=2,
                remote_port=2,
                local_gid=LOCAL_GIDS[1],
                remote_gid=REMOTE_GIDS[1],
                rate="40 Gb/sec (4X QDR)",
                subnet_manager_host="remote",
                subnet_manager_unit=managers[1].unit,
                subnet_manager_guid=managers[1].guid,
                subnet_manager_argv=managers[1].argv,
            ),
        )
    )


def config(
    binding: guard.CoordinationPeerBinding | None = None,
) -> guard.HostGuardConfig:
    return guard.HostGuardConfig(
        peer_role="idle_nonparticipant",
        peer=binding or peer_binding(),
        cross_host_fabric=fabric_binding(),
    )


def port_counters(
    *,
    link_downed: int = 0,
    data_offset: int = 0,
) -> dict[str, int]:
    return {
        "port_rcv_data": 100 + data_offset,
        "port_rcv_packets": 20 + data_offset,
        "port_xmit_data": 200 + data_offset,
        "port_xmit_packets": 40 + data_offset,
        "link_downed": link_downed,
    }


def hca_observation(
    binding: guard.HcaBinding,
    *,
    gids: tuple[str, str] = REMOTE_GIDS,
    lids: tuple[int, int] = (2, 4),
    sm_lids: tuple[int, int] = (2, 4),
    link_downed: int = 0,
    data_offset: int = 0,
) -> guard.HcaObservation:
    return guard.HcaObservation(
        device=binding.device,
        node_guid=binding.node_guid,
        ports=(
            guard.HcaPortObservation(
                port=1,
                gid=gids[0],
                state="4: ACTIVE",
                physical_state="5: LinkUp",
                rate="40 Gb/sec (4X QDR)",
                lid=lids[0],
                sm_lid=sm_lids[0],
                counter_device=binding.device,
                counter_port=1,
                counters=port_counters(
                    link_downed=link_downed, data_offset=data_offset
                ),
            ),
            guard.HcaPortObservation(
                port=2,
                gid=gids[1],
                state="4: ACTIVE",
                physical_state="5: LinkUp",
                rate="40 Gb/sec (4X QDR)",
                lid=lids[1],
                sm_lid=sm_lids[1],
                counter_device=binding.device,
                counter_port=2,
                counters=port_counters(
                    link_downed=link_downed, data_offset=data_offset
                ),
            ),
        ),
    )


def host_observation(
    binding: guard.CoordinationPeerBinding | None = None,
) -> guard.HostObservation:
    expected = binding or peer_binding()
    opensm = tuple(
        guard.OpenSmUnitObservation(
            unit=unit.unit,
            port=unit.port,
            guid=unit.guid,
            load_state="loaded",
            active_state="active",
            sub_state="running",
            main_pid=700 + unit.port,
            systemd_start_monotonic_us=1_000 + unit.port,
            process_start_time_ticks=2_000 + unit.port,
            argv=unit.argv,
            executable=file_observation(unit.executable),
            version=unit.version,
        )
        for unit in expected.opensm_units
    )
    return guard.HostObservation(
        hostname=expected.hostname,
        boot_id="11111111-2222-4333-8444-555555555555",
        probe_python=file_observation(expected.remote_probe.python),
        probe_script=file_observation(expected.remote_probe.script),
        nvidia_smi=file_observation(expected.tools.nvidia_smi),
        systemctl=file_observation(expected.tools.systemctl),
        cpu_memory=guard.CpuMemoryObservation(
            online_cpus=(0, 1, 2, 3),
            numa_cpu_sets={"0": (0, 1), "1": (2, 3)},
            load_average_1m=0.5,
            load_average_5m=0.4,
            load_average_15m=0.3,
            memory_total_bytes=256 * 1024**3,
            memory_available_bytes=200 * 1024**3,
            memory_free_bytes=180 * 1024**3,
            common_cpu_flags=("amx_bf16", "amx_int8", "amx_tile"),
        ),
        gpus=(
            guard.GpuTelemetryObservation(
                uuid=expected.gpus[0].uuid,
                pci_bus_id=expected.gpus[0].pci_bus_id,
                name=expected.gpus[0].name,
                memory_total_bytes=expected.gpus[0].memory_total_bytes,
                memory_used_bytes=32 * 1024**2,
                gpu_utilization_percent=0,
                memory_utilization_percent=0,
                temperature_celsius=35,
                power_draw_watts=20.0,
            ),
        ),
        gpu_processes=(),
        conflicting_processes=(),
        unsafe_profiler_modules=(
            guard.KernelModuleObservation(
                name="sep5",
                size_bytes=1024,
                reference_count=0,
                dependencies=(),
                state="Live",
            ),
        ),
        raid_sync_conflicts=(),
        unused_reserved_ports=expected.reserved_ports,
        hca=hca_observation(expected.hca),
        opensm_units=(opensm[0], opensm[1]),
    )


def snapshot(
    phase: Literal["preflight", "postflight"],
    *,
    binding: guard.CoordinationPeerBinding | None = None,
    contract: guard.HostGuardConfig | None = None,
    observation: guard.HostObservation | None = None,
    nonce_digit: str,
    offset_ns: int,
) -> guard.HostGuardSnapshot:
    expected = contract.peer if contract is not None else (binding or peer_binding())
    complete_config = contract or config(expected)
    request = guard.build_remote_probe_request(
        complete_config,
        now_unix_ns=NOW_NS + offset_ns,
        nonce=nonce_digit * 64,
    )
    receipt = guard.build_remote_probe_receipt(
        request,
        observation or host_observation(expected),
        observed_at_unix_ns=NOW_NS + offset_ns + 1,
    )
    return guard.build_host_guard_snapshot(
        phase,
        receipt,
        collected_at_unix_ns=NOW_NS + offset_ns + 2,
    )


def replace_observation(
    observation: guard.HostObservation,
    **updates: object,
) -> guard.HostObservation:
    return observation.model_copy(update=updates)


def test_host_guard_config_binds_idle_nonparticipant_and_fabric() -> None:
    value = config()
    assert value.peer_role == "idle_nonparticipant"
    assert value.cross_host_fabric is not None
    assert guard.calculate_host_guard_config_sha256(
        value
    ) == guard.calculate_host_guard_sha256(value)
    assert len(guard.calculate_host_guard_config_sha256(value)) == 64


def test_host_guard_config_is_strict_and_forbids_extra_fields() -> None:
    payload = config().model_dump(mode="json")
    payload["unreviewed"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        guard.HostGuardConfig.model_validate(payload)
    payload = config().model_dump(mode="json")
    payload["peer_role"] = "participant"
    with pytest.raises(ValidationError):
        guard.HostGuardConfig.model_validate(payload)


def test_host_guard_config_rejects_fabric_that_disagrees_with_peer() -> None:
    payload = fabric_binding().model_dump(mode="json")
    payload["rails"][0]["remote_gid"] = "fe80::99"
    mismatched = guard.CrossHostFabricBinding.model_validate_json(json.dumps(payload))
    with pytest.raises(ValidationError, match="remote endpoint differs"):
        guard.HostGuardConfig(peer=peer_binding(), cross_host_fabric=mismatched)


def test_complete_config_sha_binds_fabric_mapping_and_manager_direction_everywhere() -> (
    None
):
    first = config()
    payload = first.model_dump(mode="json")
    payload["cross_host_fabric"]["rails"][0]["subnet_manager_host"] = "local"
    second = guard.HostGuardConfig.model_validate_json(json.dumps(payload))
    assert first.peer == second.peer
    assert guard.calculate_host_guard_config_sha256(
        first
    ) != guard.calculate_host_guard_config_sha256(second)

    first_request = guard.build_remote_probe_request(
        first, now_unix_ns=NOW_NS, nonce="a" * 64
    )
    second_request = guard.build_remote_probe_request(
        second, now_unix_ns=NOW_NS, nonce="b" * 64
    )
    assert first_request.binding_sha256 == second_request.binding_sha256
    assert first_request.config_sha256 != second_request.config_sha256
    first_receipt = guard.build_remote_probe_receipt(
        first_request, host_observation(), observed_at_unix_ns=NOW_NS + 1
    )
    first_snapshot = guard.build_host_guard_snapshot(
        "preflight", first_receipt, collected_at_unix_ns=NOW_NS + 2
    )
    assert first_receipt.config_sha256 == first_request.config_sha256
    assert first_snapshot.config_sha256 == first_request.config_sha256

    comparison = guard.compare_snapshots(
        snapshot(
            "preflight",
            contract=first,
            nonce_digit="c",
            offset_ns=0,
        ),
        snapshot(
            "postflight",
            contract=second,
            nonce_digit="d",
            offset_ns=100,
        ),
    )
    assert not comparison.stable
    assert "complete host guard config changed between snapshots" in comparison.failures
    assert comparison.config_sha256 == guard.calculate_host_guard_config_sha256(first)


def test_canonical_json_round_trip_and_hash_are_deterministic() -> None:
    value = {"z": [3, True], "a": "ascii"}
    encoded = guard.canonical_host_guard_json(value)
    assert encoded == b'{"a":"ascii","z":[3,true]}\n'
    assert guard.parse_bounded_canonical_json(encoded) == value
    assert (
        guard.calculate_host_guard_sha256(value) == hashlib.sha256(encoded).hexdigest()
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1, "b":2}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"a":"\xc3\xa9"}\n',
        b'{"a":1}',
        b"[]\n",
        b"",
    ],
)
def test_canonical_json_rejects_ambiguous_or_noncanonical_payloads(
    payload: bytes,
) -> None:
    with pytest.raises(guard.HostGuardError):
        guard.parse_bounded_canonical_json(payload)


def test_request_and_receipt_bind_nonce_binding_and_canonical_digests() -> None:
    binding = peer_binding()
    request = guard.build_remote_probe_request(
        binding,
        now_unix_ns=NOW_NS,
        nonce="c" * 64,
    )
    receipt = guard.build_remote_probe_receipt(
        request,
        host_observation(binding),
        observed_at_unix_ns=NOW_NS + 1,
    )
    guard.verify_remote_probe_receipt(
        receipt,
        request,
        binding,
        now_unix_ns=NOW_NS + 2,
    )
    assert request.binding_sha256 == guard.calculate_coordination_peer_binding_sha256(
        binding
    )
    assert receipt.nonce == request.nonce


@pytest.mark.parametrize(
    "field", ["nonce", "request_sha256", "binding_sha256", "config_sha256"]
)
def test_receipt_verification_rejects_tampered_binding_fields(field: str) -> None:
    binding = peer_binding()
    request = guard.build_remote_probe_request(
        binding,
        now_unix_ns=NOW_NS,
        nonce="d" * 64,
    )
    receipt = guard.build_remote_probe_receipt(
        request,
        host_observation(binding),
        observed_at_unix_ns=NOW_NS + 1,
    )
    tampered = receipt.model_copy(update={field: "e" * 64})
    with pytest.raises(guard.HostGuardError):
        guard.verify_remote_probe_receipt(
            tampered,
            request,
            binding,
            now_unix_ns=NOW_NS + 2,
        )


def test_request_model_rejects_digest_tampering() -> None:
    request = guard.build_remote_probe_request(
        peer_binding(),
        now_unix_ns=NOW_NS,
        nonce="f" * 64,
    )
    payload = request.model_dump(mode="json")
    payload["expires_at_unix_ns"] += 1
    with pytest.raises(ValidationError, match="request digest is invalid"):
        guard.RemoteProbeRequest.model_validate_json(json.dumps(payload))


def test_validate_idle_peer_accepts_loaded_but_unused_profiler_driver_and_opensm() -> (
    None
):
    binding = peer_binding()
    guard.validate_idle_peer_observation(host_observation(binding), binding)


def idle_failure_cases() -> list[
    tuple[str, Callable[[guard.HostObservation], guard.HostObservation]]
]:
    def busy_gpu(value: guard.HostObservation) -> guard.HostObservation:
        gpu = value.gpus[0].model_copy(update={"memory_used_bytes": 1024**3})
        return replace_observation(value, gpus=(gpu,))

    def missing_amx(value: guard.HostObservation) -> guard.HostObservation:
        cpu = value.cpu_memory.model_copy(update={"common_cpu_flags": ("amx_bf16",)})
        return replace_observation(value, cpu_memory=cpu)

    def bad_hca(value: guard.HostObservation) -> guard.HostObservation:
        port = value.hca.ports[0].model_copy(update={"state": "2: INIT"})
        hca = value.hca.model_copy(update={"ports": (port, value.hca.ports[1])})
        return replace_observation(value, hca=hca)

    def bad_counter(value: guard.HostObservation) -> guard.HostObservation:
        counters = {**value.hca.ports[0].counters, "link_downed": 1}
        port = value.hca.ports[0].model_copy(update={"counters": counters})
        hca = value.hca.model_copy(update={"ports": (port, value.hca.ports[1])})
        return replace_observation(value, hca=hca)

    def stopped_opensm(value: guard.HostObservation) -> guard.HostObservation:
        unit = value.opensm_units[0].model_copy(update={"active_state": "inactive"})
        return replace_observation(value, opensm_units=(unit, value.opensm_units[1]))

    def active_sep(value: guard.HostObservation) -> guard.HostObservation:
        module = value.unsafe_profiler_modules[0].model_copy(
            update={"reference_count": 1}
        )
        return replace_observation(value, unsafe_profiler_modules=(module,))

    return [
        ("hostname", lambda value: replace_observation(value, hostname="other")),
        (
            "load",
            lambda value: replace_observation(
                value,
                cpu_memory=value.cpu_memory.model_copy(
                    update={"load_average_1m": 10.0}
                ),
            ),
        ),
        ("AMX", missing_amx),
        ("GPU", busy_gpu),
        (
            "GPU compute",
            lambda value: replace_observation(
                value,
                gpu_processes=(
                    guard.GpuProcessObservation(
                        gpu_uuid=value.gpus[0].uuid,
                        pid=900,
                        process_name="python",
                        used_memory_bytes=1024,
                    ),
                ),
            ),
        ),
        (
            "processes",
            lambda value: replace_observation(
                value,
                conflicting_processes=(
                    guard.ConflictingProcessObservation(
                        pid=901,
                        start_time_ticks=100,
                        classes=("storage",),
                        argv=("/usr/bin/rsync",),
                    ),
                ),
            ),
        ),
        (
            "RAID",
            lambda value: replace_observation(
                value, raid_sync_conflicts=("md0:resync",)
            ),
        ),
        (
            "reserved",
            lambda value: replace_observation(value, unused_reserved_ports=(62274,)),
        ),
        ("not ACTIVE", bad_hca),
        ("counter", bad_counter),
        ("OpenSM", stopped_opensm),
        ("unsafe profiler", active_sep),
    ]


@pytest.mark.parametrize(("message", "mutation"), idle_failure_cases())
def test_validate_idle_peer_fails_closed_for_resource_conflicts(
    message: str,
    mutation: Callable[[guard.HostObservation], guard.HostObservation],
) -> None:
    binding = peer_binding()
    with pytest.raises(guard.HostGuardError, match=message):
        guard.validate_idle_peer_observation(
            mutation(host_observation(binding)), binding
        )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("/usr/bin/ib_write_bw", "--dualport"), ("benchmark",)),
        (("/usr/bin/python", "/opt/two_host_ib_baseline.py"), ("benchmark",)),
        (("/usr/bin/uvicorn", "app:api"), ("model_server",)),
        (("/usr/bin/python", "-m", "sglang.launch_server"), ("model_server",)),
        (("/opt/intel/vtune/bin64/vtune", "-collect", "hotspots"), ("profiler",)),
        (("/usr/bin/rsync", "/source", "/destination"), ("storage",)),
        (("/usr/bin/hf", "download", "model"), ("storage",)),
        (("/usr/bin/python", "ordinary.py"), ()),
    ],
)
def test_classify_process_covers_benchmark_model_profiler_and_storage(
    argv: tuple[str, ...],
    expected: tuple[guard.ProcessClass, ...],
) -> None:
    assert guard.classify_process(argv) == expected


@pytest.mark.parametrize(
    ("script_name", "expected"),
    sorted(guard.WORKFLOW_SCRIPT_CLASSES.items()),
)
def test_classify_process_covers_every_real_exo_workflow_script(
    script_name: str,
    expected: guard.ProcessClass,
) -> None:
    script_path = Path(__file__).parents[1] / script_name
    assert script_path.is_file(), (
        f"workflow inventory names missing script {script_path}"
    )
    assert guard.classify_process((sys.executable, str(script_path))) == (expected,)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("/usr/bin/ibv_rc_pingpong",), ("benchmark",)),
        (("/usr/bin/iperf3", "-c", "192.0.2.1"), ("benchmark",)),
        (("/usr/bin/scp", "model", "fwuff:/mnt/sanic/model"), ("storage",)),
        (("/usr/bin/cp", "/mnt/sanic/model/x", "/var/lib/exo/models/x"), ("storage",)),
        (
            ("/usr/bin/curl", "-o", "/var/lib/exo/models/x", "https://hf.co/x"),
            ("storage",),
        ),
        (("/usr/bin/sha256sum", "/mnt/sanic/model/x.safetensors"), ("storage",)),
        (("/usr/sbin/opensm", "--guid", "0xe41d2d03004d32e1"), ()),
        (("/usr/bin/systemctl", "show", "opensm-port1.service"), ()),
        (("/usr/bin/sha256sum", "/etc/hosts"), ()),
        (("/usr/bin/curl", "http://127.0.0.1:52415/health"), ()),
        (("/usr/bin/cp", "/tmp/a", "/tmp/b"), ()),
        (("/usr/bin/cat", "/root/exo/scripts/two_host_model_stage.py"), ()),
    ],
)
def test_classify_process_covers_transfer_tools_without_benign_false_positives(
    argv: tuple[str, ...],
    expected: tuple[guard.ProcessClass, ...],
) -> None:
    assert guard.classify_process(argv) == expected


def test_collect_unsafe_profiler_modules_records_loaded_idle_drivers(
    tmp_path: Path,
) -> None:
    modules = tmp_path / "modules"
    modules.write_text(
        "ordinary 4096 2 dependency, Live 0x0\n"
        "sep5 8192 0 - Live 0x0\n"
        "pax 1024 0 helper,other, Live 0x0\n"
    )
    observed = guard.collect_unsafe_profiler_modules(
        ("pax", "sep5"),
        proc_modules_path=modules,
    )
    assert tuple(module.name for module in observed) == ("pax", "sep5")
    assert observed[0].dependencies == ("helper", "other")
    assert observed[1].reference_count == 0


def test_collect_unsafe_profiler_modules_rejects_malformed_input(
    tmp_path: Path,
) -> None:
    modules = tmp_path / "modules"
    modules.write_text("sep5 malformed\n")
    with pytest.raises(guard.HostGuardError, match="malformed"):
        guard.collect_unsafe_profiler_modules(
            ("pax", "sep5"), proc_modules_path=modules
        )


def test_compare_snapshots_accepts_stable_peer_with_fresh_requests() -> None:
    preflight = snapshot("preflight", nonce_digit="1", offset_ns=0)
    postflight = snapshot("postflight", nonce_digit="2", offset_ns=100)
    comparison = guard.compare_snapshots(preflight, postflight)
    assert comparison.stable
    assert comparison.failures == ()
    assert comparison.health_counter_deltas == {
        "port1.link_downed": 0,
        "port2.link_downed": 0,
    }
    assert set(comparison.data_counter_deltas) == {
        f"port{port}.{counter}"
        for port in (1, 2)
        for counter in guard.DATA_COUNTER_NAMES
    }
    assert set(comparison.data_counter_deltas.values()) == {0}


def test_compare_snapshots_allows_only_configured_opensm_background_traffic() -> None:
    binding = peer_binding()
    before = host_observation(binding)
    after = replace_observation(
        before,
        hca=hca_observation(binding.hca, data_offset=4),
    )
    comparison = guard.compare_snapshots(
        snapshot(
            "preflight",
            binding=binding,
            observation=before,
            nonce_digit="a",
            offset_ns=0,
        ),
        snapshot(
            "postflight",
            binding=binding,
            observation=after,
            nonce_digit="b",
            offset_ns=100,
        ),
    )
    assert comparison.stable
    assert set(comparison.data_counter_deltas.values()) == {4}
    assert comparison.data_counter_maximum_deltas["port1.port_rcv_packets"] == 8
    assert comparison.data_counter_maximum_deltas["port2.port_xmit_data"] == 64


def test_compare_snapshots_rejects_substantial_hidden_infiniband_payload() -> None:
    binding = peer_binding()
    before = host_observation(binding)
    after = replace_observation(
        before,
        hca=hca_observation(binding.hca, data_offset=100),
    )
    comparison = guard.compare_snapshots(
        snapshot(
            "preflight",
            binding=binding,
            observation=before,
            nonce_digit="c",
            offset_ns=0,
        ),
        snapshot(
            "postflight",
            binding=binding,
            observation=after,
            nonce_digit="d",
            offset_ns=100,
        ),
    )
    assert not comparison.stable
    assert any("idle OpenSM tolerance" in failure for failure in comparison.failures)


def test_compare_snapshots_rejects_data_counter_reset() -> None:
    binding = peer_binding()
    before = replace_observation(
        host_observation(binding),
        hca=hca_observation(binding.hca, data_offset=10),
    )
    after = replace_observation(
        host_observation(binding),
        hca=hca_observation(binding.hca, data_offset=0),
    )
    comparison = guard.compare_snapshots(
        snapshot(
            "preflight",
            binding=binding,
            observation=before,
            nonce_digit="e",
            offset_ns=0,
        ),
        snapshot(
            "postflight",
            binding=binding,
            observation=after,
            nonce_digit="f",
            offset_ns=100,
        ),
    )
    assert not comparison.stable
    assert any("data counter reset" in failure for failure in comparison.failures)


def test_compare_snapshots_rejects_boot_or_opensm_identity_change() -> None:
    before = host_observation()
    changed_unit = before.opensm_units[0].model_copy(update={"main_pid": 999})
    after = replace_observation(
        before,
        boot_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        opensm_units=(changed_unit, before.opensm_units[1]),
    )
    comparison = guard.compare_snapshots(
        snapshot("preflight", observation=before, nonce_digit="3", offset_ns=0),
        snapshot("postflight", observation=after, nonce_digit="4", offset_ns=100),
    )
    assert not comparison.stable
    assert comparison.failures == (
        "peer hardware, tool, or OpenSM process identity changed",
    )


@pytest.mark.parametrize(
    ("before_count", "after_count", "failure"), [(0, 1, "increased"), (1, 0, "reset")]
)
def test_compare_snapshots_rejects_hca_health_counter_change(
    before_count: int,
    after_count: int,
    failure: str,
) -> None:
    binding = peer_binding()
    before = replace_observation(
        host_observation(binding),
        hca=hca_observation(binding.hca, link_downed=before_count),
    )
    after = replace_observation(
        host_observation(binding),
        hca=hca_observation(binding.hca, link_downed=after_count),
    )
    comparison = guard.compare_snapshots(
        snapshot(
            "preflight",
            binding=binding,
            observation=before,
            nonce_digit="5",
            offset_ns=0,
        ),
        snapshot(
            "postflight",
            binding=binding,
            observation=after,
            nonce_digit="6",
            offset_ns=100,
        ),
    )
    assert not comparison.stable
    assert any(failure in item for item in comparison.failures)


def test_compare_snapshots_rejects_request_reuse_and_reverse_time() -> None:
    preflight = snapshot("preflight", nonce_digit="7", offset_ns=100)
    postflight = guard.build_host_guard_snapshot(
        "postflight",
        preflight.remote_receipt,
        collected_at_unix_ns=preflight.collected_at_unix_ns - 1,
    )
    comparison = guard.compare_snapshots(preflight, postflight)
    assert not comparison.stable
    assert "postflight was not collected after preflight" in comparison.failures
    assert "preflight and postflight reused one remote request" in comparison.failures


def local_observation() -> guard.HostObservation:
    remote = host_observation()
    local_hca_binding = peer_binding().hca.model_copy(
        update={"node_guid": "e41d:2d03:004d:aaaa"}
    )
    return replace_observation(
        remote,
        hostname="dwagon",
        hca=hca_observation(
            local_hca_binding,
            gids=LOCAL_GIDS,
            lids=(1, 3),
            sm_lids=(2, 4),
        ),
    )


def test_validate_cross_host_fabric_accepts_exact_two_remote_managed_rails() -> None:
    guard.validate_cross_host_fabric(
        local_observation(),
        host_observation(),
        fabric_binding(),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "gid",
        "rate",
        "sm_lid",
        "state",
        "counter_source",
        "manager_unit",
        "manager_guid",
        "manager_argv",
        "manager_port",
    ],
)
def test_validate_cross_host_fabric_rejects_endpoint_drift(mutation: str) -> None:
    remote = host_observation()
    changed_opensm = remote.opensm_units
    port_updates: dict[str, object]
    if mutation == "gid":
        port_updates = {"gid": "fe80::99"}
    elif mutation == "rate":
        port_updates = {"rate": "20 Gb/sec"}
    elif mutation == "sm_lid":
        port_updates = {"sm_lid": 3}
    elif mutation == "state":
        port_updates = {"state": "2: INIT"}
    elif mutation == "counter_source":
        port_updates = {"counter_port": 2}
    else:
        port_updates = {}
        manager_updates: dict[str, object]
        if mutation == "manager_unit":
            manager_updates = {"unit": "other-port1.service"}
        elif mutation == "manager_guid":
            manager_updates = {"guid": "0xaaaaaaaaaaaaaaaa"}
        elif mutation == "manager_argv":
            manager_updates = {
                "argv": (
                    "/usr/sbin/opensm",
                    "--guid",
                    "0xaaaaaaaaaaaaaaaa",
                )
            }
        else:
            manager_updates = {"port": 2}
        changed_manager = remote.opensm_units[0].model_copy(update=manager_updates)
        changed_opensm = (changed_manager, remote.opensm_units[1])
    port = remote.hca.ports[0].model_copy(update=port_updates)
    changed_hca = remote.hca.model_copy(update={"ports": (port, remote.hca.ports[1])})
    with pytest.raises(guard.HostGuardError, match="cross-host fabric"):
        guard.validate_cross_host_fabric(
            local_observation(),
            replace_observation(
                remote,
                hca=changed_hca,
                opensm_units=changed_opensm,
            ),
            fabric_binding(),
        )


def test_host_guard_config_rejects_remote_manager_metadata_mismatch() -> None:
    payload = config().model_dump(mode="json")
    payload["cross_host_fabric"]["rails"][0]["subnet_manager_unit"] = (
        "other-port1.service"
    )
    with pytest.raises(ValidationError, match="remote manager differs"):
        guard.HostGuardConfig.model_validate_json(json.dumps(payload))


def test_opensm_binding_rejects_guid_argv_mismatch() -> None:
    payload = peer_binding().opensm_units[0].model_dump(mode="json")
    payload["guid"] = "0xaaaaaaaaaaaaaaaa"
    with pytest.raises(ValidationError, match="argv does not contain"):
        guard.OpenSmUnitBinding.model_validate_json(json.dumps(payload))


def test_collect_remote_snapshot_uses_pinned_ssh_and_verifies_wire_receipt(
    tmp_path: Path,
) -> None:
    local_files: list[guard.FileIdentityBinding] = []
    for name in ("ssh", "known_hosts", "identity"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        local_files.append(
            guard.FileIdentityBinding(
                path=str(path),
                resolved_path=str(path),
                sha256=hashlib.sha256(name.encode()).hexdigest(),
            )
        )
    binding = peer_binding(ssh_files=tuple(local_files))
    calls: list[tuple[str, ...]] = []
    inherited_descriptors: list[tuple[int, ...]] = []

    def runner(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> guard.CommandResult:
        del timeout_seconds, maximum_stdout_bytes, maximum_stderr_bytes, environment
        calls.append(command)
        inherited_descriptors.append(pass_fds)
        request = guard.RemoteProbeRequest.model_validate_json(
            guard.canonical_host_guard_json(
                guard.parse_bounded_canonical_json(input_bytes)
            )
        )
        receipt = guard.build_remote_probe_receipt(
            request,
            host_observation(binding),
            observed_at_unix_ns=NOW_NS + 1,
        )
        return guard.CommandResult(
            return_code=0,
            stdout=guard.canonical_host_guard_json(receipt),
            stderr=b"",
        )

    times = iter((NOW_NS, NOW_NS + 2))
    result = guard.collect_remote_snapshot(
        config(binding),
        "preflight",
        runner=runner,
        now_unix_ns=lambda: next(times),
        nonce_factory=lambda: "8" * 64,
    )
    assert result.phase == "preflight"
    assert result.observation.hostname == "fwuff"
    assert len(calls) == 1
    assert calls[0][0].startswith("/proc/self/fd/")
    assert len(inherited_descriptors[0]) == 3
    assert tuple(sorted(inherited_descriptors[0])) == inherited_descriptors[0]
    descriptor_paths = {
        f"/proc/self/fd/{descriptor}" for descriptor in inherited_descriptors[0]
    }
    assert calls[0][0] in descriptor_paths
    assert any(
        argument.startswith("UserKnownHostsFile=")
        and argument.partition("=")[2] in descriptor_paths
        for argument in calls[0]
    )
    assert any(
        argument.startswith("IdentityFile=")
        and argument.partition("=")[2] in descriptor_paths
        for argument in calls[0]
    )
    assert "ClearAllForwardings=yes" in calls[0]
    assert calls[0][-1] == "remote-probe"
    assert calls[0][-3:-1] == (
        binding.remote_probe.python.resolved_path,
        binding.remote_probe.script.resolved_path,
    )


@pytest.mark.parametrize(
    ("return_code", "stdout", "stderr", "message"),
    [
        (1, b"", b"failure", "exited 1"),
        (0, b"{}\n", b"", "output is invalid"),
        (0, b"", b"warning", "unexpected stderr"),
    ],
)
def test_collect_remote_snapshot_fails_closed_on_transport_output(
    tmp_path: Path,
    return_code: int,
    stdout: bytes,
    stderr: bytes,
    message: str,
) -> None:
    files: list[guard.FileIdentityBinding] = []
    for name in ("ssh", "known_hosts", "identity"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(
            guard.FileIdentityBinding(
                path=str(path),
                resolved_path=str(path),
                sha256=hashlib.sha256(name.encode()).hexdigest(),
            )
        )
    binding = peer_binding(ssh_files=tuple(files))

    def runner(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> guard.CommandResult:
        del command, input_bytes, timeout_seconds, maximum_stdout_bytes
        del maximum_stderr_bytes, environment, pass_fds
        return guard.CommandResult(
            return_code=return_code, stdout=stdout, stderr=stderr
        )

    with pytest.raises(guard.HostGuardTransportError, match=message):
        guard.collect_remote_snapshot(
            binding,
            "preflight",
            runner=runner,
            now_unix_ns=lambda: NOW_NS,
            nonce_factory=lambda: "9" * 64,
        )


def test_observe_file_identity_binds_content_and_resolved_path(tmp_path: Path) -> None:
    path = tmp_path / "tool"
    path.write_bytes(b"bound")
    binding = guard.FileIdentityBinding(
        path=str(path),
        resolved_path=str(path),
        sha256=hashlib.sha256(b"bound").hexdigest(),
    )
    assert guard.observe_file_identity(binding).size_bytes == 5
    path.write_bytes(b"changed")
    with pytest.raises(guard.HostGuardError, match="digest changed"):
        guard.observe_file_identity(binding)


def executable_binding(path: Path) -> guard.FileIdentityBinding:
    resolved = path.resolve(strict=True)
    return guard.FileIdentityBinding(
        path=str(path),
        resolved_path=str(resolved),
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
    )


def test_run_bound_executable_executes_verified_open_inode() -> None:
    binding = executable_binding(Path(sys.executable))
    result = guard.run_bound_executable(
        binding,
        ("-c", "print('fd-pinned')"),
        runner=guard.run_bounded_command,
    )
    assert result.return_code == 0
    assert result.stdout == b"fd-pinned\n"
    assert result.stderr == b""


def test_run_bound_executable_rejects_path_replacement_during_execution(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tool"
    tool.write_bytes(b"verified-tool")
    binding = executable_binding(tool)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement-tool")

    def replacing_runner(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> guard.CommandResult:
        del input_bytes, timeout_seconds, maximum_stdout_bytes
        del maximum_stderr_bytes, environment
        assert len(pass_fds) == 1
        assert command[0] == f"/proc/self/fd/{pass_fds[0]}"
        assert Path(command[0]).read_bytes() == b"verified-tool"
        replacement.replace(tool)
        assert Path(command[0]).read_bytes() == b"verified-tool"
        return guard.CommandResult(return_code=0, stdout=b"", stderr=b"")

    with pytest.raises(guard.HostGuardError, match="path was replaced"):
        guard.run_bound_executable(binding, (), runner=replacing_runner)


def test_run_bound_executable_checks_replacement_even_when_runner_fails(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tool"
    tool.write_bytes(b"verified-tool")
    binding = executable_binding(tool)

    def failing_replacing_runner(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> guard.CommandResult:
        del command, input_bytes, timeout_seconds, maximum_stdout_bytes
        del maximum_stderr_bytes, environment, pass_fds
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"replacement-tool")
        replacement.replace(tool)
        raise guard.HostGuardTransportError("simulated runner failure")

    with pytest.raises(guard.HostGuardError, match="path was replaced"):
        guard.run_bound_executable(binding, (), runner=failing_replacing_runner)


def test_collect_remote_snapshot_rejects_ssh_identity_replacement(
    tmp_path: Path,
) -> None:
    local_files: list[guard.FileIdentityBinding] = []
    paths: list[Path] = []
    for name in ("ssh", "known_hosts", "identity"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)
        local_files.append(executable_binding(path))
    binding = peer_binding(ssh_files=tuple(local_files))

    def replacing_runner(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
        environment: Mapping[str, str],
        pass_fds: tuple[int, ...] = (),
    ) -> guard.CommandResult:
        del command, input_bytes, timeout_seconds, maximum_stdout_bytes
        del maximum_stderr_bytes, environment, pass_fds
        replacement = tmp_path / "new-identity"
        replacement.write_bytes(b"new identity")
        replacement.replace(paths[2])
        return guard.CommandResult(return_code=1, stdout=b"", stderr=b"failed")

    with pytest.raises(guard.HostGuardError, match="path was replaced"):
        guard.collect_remote_snapshot(
            config(binding),
            "preflight",
            runner=replacing_runner,
            now_unix_ns=lambda: NOW_NS,
            nonce_factory=lambda: "e" * 64,
        )


def run_real_bounded_command(
    code: str,
    *,
    timeout_seconds: float = 0.5,
    maximum_stdout_bytes: int = 4096,
    maximum_stderr_bytes: int = 4096,
) -> guard.CommandResult:
    return guard.run_bounded_command(
        (sys.executable, "-c", code),
        input_bytes=b"",
        timeout_seconds=timeout_seconds,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
        environment=guard.sanitized_command_environment(),
    )


def assert_pid_is_reaped(pid_path: Path) -> None:
    assert pid_path.is_file()
    pid = int(pid_path.read_text())
    deadline = time.monotonic() + 3.0
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{pid}").exists(), f"process {pid} survived cleanup"


def test_run_bounded_command_times_out_and_confirms_cleanup(tmp_path: Path) -> None:
    pid_path = tmp_path / "leader.pid"
    code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    with pytest.raises(guard.HostGuardTransportError, match="timed out"):
        run_real_bounded_command(code, timeout_seconds=0.3)
    assert_pid_is_reaped(pid_path)


@pytest.mark.parametrize(("descriptor", "message"), [(1, "stdout"), (2, "stderr")])
def test_run_bounded_command_rejects_oversized_output(
    descriptor: int,
    message: str,
) -> None:
    code = f"import os; os.write({descriptor}, b'x' * 8192)"
    with pytest.raises(guard.HostGuardTransportError, match=f"{message} exceeded"):
        run_real_bounded_command(
            code,
            maximum_stdout_bytes=128,
            maximum_stderr_bytes=128,
        )


def descendant_launcher_code(
    pid_path: Path,
    *,
    ignore_sigterm: bool,
    inherit_pipes: bool,
    parent_waits: bool,
) -> str:
    child_prefix = (
        "import os,signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_sigterm else "")
        + f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    redirection = (
        ""
        if inherit_pipes
        else ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
    )
    tail = "time.sleep(60)" if parent_waits else "time.sleep(0.1)"
    return (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child_prefix!r}]{redirection}); "
        f"deadline=time.monotonic()+2; p=Path({str(pid_path)!r}); "
        "\nwhile not p.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        f"{tail}"
    )


def test_run_bounded_command_kills_descendant_holding_pipes(tmp_path: Path) -> None:
    pid_path = tmp_path / "pipe-child.pid"
    with pytest.raises(guard.HostGuardTransportError, match="timed out"):
        run_real_bounded_command(
            descendant_launcher_code(
                pid_path,
                ignore_sigterm=False,
                inherit_pipes=True,
                parent_waits=False,
            ),
            timeout_seconds=0.4,
        )
    assert_pid_is_reaped(pid_path)


def test_run_bounded_command_sigkills_sigterm_resistant_descendant(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "resistant-child.pid"
    with pytest.raises(guard.HostGuardTransportError, match="timed out"):
        run_real_bounded_command(
            descendant_launcher_code(
                pid_path,
                ignore_sigterm=True,
                inherit_pipes=True,
                parent_waits=True,
            ),
            timeout_seconds=0.4,
        )
    assert_pid_is_reaped(pid_path)


def test_run_bounded_command_cleans_background_group_after_leader_exits(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "background-child.pid"
    result = run_real_bounded_command(
        descendant_launcher_code(
            pid_path,
            ignore_sigterm=False,
            inherit_pipes=False,
            parent_waits=False,
        ),
        timeout_seconds=2.0,
    )
    assert result.return_code == 0
    assert_pid_is_reaped(pid_path)


def test_models_reject_nonfinite_load_and_noncanonical_ordering() -> None:
    payload = peer_binding().policy.model_dump(mode="json")
    payload["maximum_load_1m_per_online_cpu"] = float("nan")
    with pytest.raises(ValidationError):
        guard.IdlePeerPolicy.model_validate(payload)
    payload = host_observation().model_dump(mode="json")
    payload["unsafe_profiler_modules"] = [
        {
            "name": "sep5",
            "size_bytes": 1,
            "reference_count": 0,
            "dependencies": [],
            "state": "Live",
        },
        {
            "name": "pax",
            "size_bytes": 1,
            "reference_count": 0,
            "dependencies": [],
            "state": "Live",
        },
    ]
    with pytest.raises(ValidationError, match="sorted and unique"):
        guard.HostObservation.model_validate_json(json.dumps(payload))
