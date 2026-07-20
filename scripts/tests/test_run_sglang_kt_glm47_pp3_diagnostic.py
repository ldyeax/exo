from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import scripts.run_sglang_kt_glm47_pp3_diagnostic as pp3
from scripts.two_host_mlx_nccl_poc import (
    HcaCounterSnapshot,
    HcaPortCounterSnapshot,
)


def make_config(tmp_path: Path) -> pp3.Pp3DiagnosticConfig:
    return pp3.Pp3DiagnosticConfig(
        run_id="glm47-pp3-test",
        result_directory=tmp_path / "result",
        dwagon_runtime_python="/runtime/dwagon/bin/python",
        fwuff_runtime_python="/runtime/fwuff/bin/python",
        dwagon_model_path=pp3.DEFAULT_DWAGON_MODEL_PATH,
        fwuff_model_path=pp3.DEFAULT_FWUFF_MODEL_PATH,
        local_source_directory="/source/dwagon",
        remote_source_directory="/source/fwuff",
        ssh_target="fwuff",
        dwagon_ip="192.168.40.24",
        fwuff_ip="192.168.40.248",
        dwagon_socket_interface="ens13f0np0",
        fwuff_socket_interface="ens17f0",
        distributed_port=62400,
        stage_ports=(62410, 62411, 62412),
        hca_devices=("mlx4_0:1", "mlx4_0:2"),
        dwagon_stage_placement=pp3.DWAGON_STAGE_PLACEMENT_PIPELINE_ORDER,
        resident_gpu_experts=4,
        readiness_timeout_seconds=60.0,
        request_timeout_seconds=900.0,
        cleanup_timeout_seconds=30.0,
        warmup_count=2,
        sample_count=3,
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_builds_exact_three_stage_hardware_plan(tmp_path: Path) -> None:
    specs = pp3.build_pp3_process_specs(make_config(tmp_path))

    assert tuple(spec.pipeline_rank for spec in specs) == (0, 1, 2)
    assert tuple(spec.node_id for spec in specs) == (
        pp3.DWAGON_NODE_ID,
        pp3.DWAGON_NODE_ID,
        pp3.FWUFF_NODE_ID,
    )
    assert tuple(spec.cpu_cores for spec in specs) == (
        tuple(range(56)),
        tuple(range(56, 112)),
        tuple(range(60)),
    )
    assert tuple(spec.memory_nodes for spec in specs) == ((0,), (1,), (0,))
    assert tuple(spec.gpu_uuid for spec in specs) == (
        pp3.DWAGON_STAGE_ZERO_GPU,
        pp3.DWAGON_STAGE_ONE_GPU,
        pp3.FWUFF_STAGE_TWO_GPU,
    )
    assert tuple((spec.start_layer, spec.end_layer) for spec in specs) == (
        (0, 16),
        (16, 32),
        (32, 47),
    )
    assert tuple(spec.executable for spec in specs) == (
        "/runtime/dwagon/bin/python",
        "/runtime/dwagon/bin/python",
        "/runtime/fwuff/bin/python",
    )
    assert tuple(spec.model_path for spec in specs) == (
        pp3.DEFAULT_DWAGON_MODEL_PATH,
        pp3.DEFAULT_DWAGON_MODEL_PATH,
        pp3.DEFAULT_FWUFF_MODEL_PATH,
    )
    for rank, spec in enumerate(specs):
        assert argument_value(spec.arguments, "--pp-size") == "3"
        assert argument_value(spec.arguments, "--tp-size") == "1"
        assert argument_value(spec.arguments, "--nnodes") == "3"
        assert argument_value(spec.arguments, "--node-rank") == str(rank)
        assert argument_value(spec.arguments, "--dist-init-addr") == (
            "192.168.40.24:62400"
        )
        assert argument_value(spec.arguments, "--kt-cpuinfer") == str(
            len(spec.cpu_cores)
        )
        assert "--disable-cuda-graph" in spec.arguments
        assert "--disable-radix-cache" in spec.arguments


def test_swaps_dwagon_hardware_without_changing_pipeline_ranges(
    tmp_path: Path,
) -> None:
    config = replace(
        make_config(tmp_path),
        dwagon_stage_placement=pp3.DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL,
    )

    specs = pp3.build_pp3_process_specs(config)
    plan_receipt = specs[0].plan.model_dump(mode="json")

    assert tuple(spec.cpu_cores for spec in specs) == (
        tuple(range(56, 112)),
        tuple(range(56)),
        tuple(range(60)),
    )
    assert tuple(spec.memory_nodes for spec in specs) == ((1,), (0,), (0,))
    assert tuple(spec.gpu_uuid for spec in specs) == (
        pp3.DWAGON_STAGE_ONE_GPU,
        pp3.DWAGON_STAGE_ZERO_GPU,
        pp3.FWUFF_STAGE_TWO_GPU,
    )
    assert tuple((spec.start_layer, spec.end_layer) for spec in specs) == (
        (0, 16),
        (16, 32),
        (32, 47),
    )
    assert plan_receipt["stages"][1]["memory_nodes"] == [0]
    assert plan_receipt["stages"][1]["gpu_uuid"] == pp3.DWAGON_STAGE_ZERO_GPU
    assert pp3._configuration_receipt(config) == {
        "dwagon_stage_placement": pp3.DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL,
    }


def test_cli_selects_cross_host_hca_local_placement(tmp_path: Path) -> None:
    arguments = pp3._parser().parse_args(
        (
            "--run-id",
            "swapped-placement",
            "--result-directory",
            str(tmp_path / "result"),
            "--dwagon-runtime-python",
            "/runtime/dwagon/bin/python",
            "--fwuff-runtime-python",
            "/runtime/fwuff/bin/python",
            "--dwagon-stage-placement",
            pp3.DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL,
        )
    )

    config = pp3._config_from_arguments(arguments)

    assert (
        config.dwagon_stage_placement == pp3.DWAGON_STAGE_PLACEMENT_CROSS_HOST_HCA_LOCAL
    )


def test_builds_numactl_commands_and_clean_nccl_environment(tmp_path: Path) -> None:
    specs = pp3.build_pp3_process_specs(make_config(tmp_path))
    parent = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "NCCL_NET": "Socket",
        "NCCL_SOCKET_IFNAME": "wrong0",
        "SGLANG_TORCH_PROFILER": "1",
        "VTUNE_PROFILER_DIR": "/unsafe",
        "PYTHONPATH": "/foreign",
    }

    for spec in specs:
        command = pp3.build_cpu_bound_sglang_kt_command(spec)
        socket_interface = (
            "ens17f0" if spec.node_id == pp3.FWUFF_NODE_ID else "ens13f0np0"
        )
        environment = pp3.build_stage_environment(
            spec,
            "owner-token",
            socket_interface,
            parent,
        )
        assert command[:5] == (
            "/usr/bin/numactl",
            "--physcpubind",
            ",".join(str(cpu) for cpu in spec.cpu_cores),
            "--membind",
            str(spec.memory_nodes[0]),
        )
        assert environment["CUDA_VISIBLE_DEVICES"] == spec.gpu_uuid
        assert environment["NCCL_NET"] == "IB"
        assert environment["NCCL_IB_HCA"] == "=mlx4_0:1,mlx4_0:2"
        assert environment["NCCL_IB_MERGE_NICS"] == "1"
        assert environment["GLOO_SOCKET_IFNAME"] == socket_interface
        assert environment["NCCL_SOCKET_IFNAME"] == socket_interface
        assert environment["NCCL_DEBUG"] == "INFO"
        assert environment["NCCL_GIN_ENABLE"] == "0"
        assert environment["SGLANG_PP_LAYER_PARTITION"] == "16,16,15"
        assert "SGLANG_TORCH_PROFILER" not in environment
        assert "VTUNE_PROFILER_DIR" not in environment
        assert "PYTHONPATH" not in environment


def test_remote_ownership_namespace_is_a_launch_argument(tmp_path: Path) -> None:
    spec = pp3.build_pp3_process_specs(make_config(tmp_path))[2]
    namespace = pp3.stage_ownership_namespace(spec)

    assert namespace == "62412"
    assert namespace in spec.arguments
    assert "EXO_SUPERVISOR_READY" in pp3.REMOTE_PROCESS_LAUNCH_SUPERVISOR_PROGRAM
    assert "ownership_verified" in pp3.REMOTE_PROCESS_STOP_PROGRAM


def test_local_start_identity_failure_terminates_unregistered_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        make_config(tmp_path),
        local_source_directory=str(tmp_path),
        cleanup_timeout_seconds=1.0,
    )
    spec = pp3.build_pp3_process_specs(config)[0]
    started_pids: list[int] = []

    def fail_identity(pid: int) -> tuple[int, int]:
        started_pids.append(pid)
        raise OSError("injected stat failure")

    monkeypatch.setattr(
        pp3,
        "build_cpu_bound_sglang_kt_command",
        lambda _spec: (sys.executable, "-c", "import time; time.sleep(60)"),
    )
    monkeypatch.setattr(pp3, "_read_process_identity", fail_identity)
    log_path = tmp_path / "local-start-failure.log"

    with (
        log_path.open("x", encoding="utf-8") as log_file,
        pytest.raises(OSError, match="injected stat failure"),
    ):
        pp3._start_local_stage(spec, config, "owner-token", log_file)

    assert len(started_pids) == 1
    assert pp3._unregistered_process_group_members(started_pids[0]) == ()


def test_remote_pump_failure_hands_registered_stage_to_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    spec = pp3.build_pp3_process_specs(config)[2]
    owner_token = "owner-token"
    namespace = pp3.stage_ownership_namespace(spec)
    receipt = {
        "pid": 4321,
        "process_group_id": 4321,
        "start_time_ticks": 12345,
        "owner_token": owner_token,
        "namespace": namespace,
    }
    markers = iter(("EXO_SUPERVISOR_READY", "EXO_OWNER " + json.dumps(receipt)))

    class StubRemoteProcess:
        pid = 9876
        stdin = io.StringIO()
        stdout = io.StringIO()

    process = StubRemoteProcess()
    cleaned: list[pp3._RunningStage] = []

    monkeypatch.setattr(pp3.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(pp3, "_read_marker", lambda *_args: next(markers))
    monkeypatch.setattr(
        pp3.threading.Thread,
        "start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("injected pump failure")),
    )

    def record_cleanup(
        running: pp3._RunningStage,
        _config: pp3.Pp3DiagnosticConfig,
    ) -> pp3.ProcessCleanupReceipt:
        cleaned.append(running)
        return pp3.ProcessCleanupReceipt(
            host_name="fwuff",
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    monkeypatch.setattr(pp3, "stop_stage", record_cleanup)
    log_path = tmp_path / "remote-pump-failure.log"

    with (
        log_path.open("x", encoding="utf-8") as log_file,
        pytest.raises(RuntimeError, match="injected pump failure"),
    ):
        pp3._start_remote_stage(spec, config, owner_token, log_file)

    assert len(cleaned) == 1
    assert cleaned[0].owned.pid == 4321
    assert cleaned[0].owned.owner_token == owner_token


def _start_owned_group_with_sanitized_child(
    owner_token: str,
    namespace: str,
) -> subprocess.Popen[str]:
    program = (
        "import os, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], env={'PATH': os.environ['PATH']}); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    environment = dict(os.environ)
    environment["EXO_BENCHMARK_OWNER_TOKEN"] = owner_token
    leader = subprocess.Popen(
        (sys.executable, "-c", program, namespace),
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    assert int(leader.stdout.readline()) > 0
    return leader


def test_local_ownership_accepts_descendants_in_the_owned_session() -> None:
    owner_token = "owner-token"
    namespace = "62410"
    leader = _start_owned_group_with_sanitized_child(owner_token, namespace)
    try:
        process_group_id, start_time_ticks = pp3._read_process_identity(leader.pid)
        owned = pp3.OwnedStageProcess(
            rank=0,
            host_name="dwagon",
            pid=leader.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=namespace,
            remote=False,
            transport_pid=leader.pid,
            log_path="/tmp/rank-0.log",
        )

        ownership_matches, members = pp3._local_group_ownership(owned)

        assert ownership_matches is True
        assert len(members) == 2
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5.0)


def test_local_ownership_fails_closed_when_live_leader_metadata_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_token = "owner-token"
    namespace = "62410"
    leader = _start_owned_group_with_sanitized_child(owner_token, namespace)
    try:
        process_group_id, start_time_ticks = pp3._read_process_identity(leader.pid)
        owned = pp3.OwnedStageProcess(
            rank=0,
            host_name="dwagon",
            pid=leader.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=namespace,
            remote=False,
            transport_pid=leader.pid,
            log_path="/tmp/rank-0.log",
        )
        original_read_bytes = Path.read_bytes

        def deny_leader_environment(path: Path) -> bytes:
            if path == Path(f"/proc/{leader.pid}/environ"):
                raise PermissionError("injected unreadable live leader")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", deny_leader_environment)

        ownership_matches, members = pp3._local_group_ownership(owned)

        assert ownership_matches is False
        assert leader.pid in members
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5.0)


def test_local_ownership_fails_closed_when_proc_stat_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_token = "owner-token"
    namespace = "62410"
    leader = _start_owned_group_with_sanitized_child(owner_token, namespace)
    try:
        process_group_id, start_time_ticks = pp3._read_process_identity(leader.pid)
        owned = pp3.OwnedStageProcess(
            rank=0,
            host_name="dwagon",
            pid=leader.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            owner_token=owner_token,
            ownership_namespace=namespace,
            remote=False,
            transport_pid=leader.pid,
            log_path="/tmp/rank-0.log",
        )
        original_read_text = Path.read_text

        def deny_leader_stat(
            path: Path,
            encoding: str | None = None,
            errors: str | None = None,
        ) -> str:
            if path == Path(f"/proc/{leader.pid}/stat"):
                raise PermissionError("injected unreadable live leader stat")
            return original_read_text(path, encoding=encoding, errors=errors)

        monkeypatch.setattr(Path, "read_text", deny_leader_stat)

        ownership_matches, members = pp3._local_group_ownership(owned)

        assert ownership_matches is False
        assert members == (leader.pid,)
    finally:
        os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5.0)


def test_remote_stop_accepts_descendants_in_the_owned_session() -> None:
    owner_token = "owner-token"
    namespace = "62412"
    leader = _start_owned_group_with_sanitized_child(owner_token, namespace)
    process_group_id, start_time_ticks = pp3._read_process_identity(leader.pid)
    receipt = {
        "host_name": "fwuff",
        "pid": leader.pid,
        "process_group_id": process_group_id,
        "start_time_ticks": start_time_ticks,
        "owner_token": owner_token,
        "namespace": namespace,
    }
    try:
        completed = subprocess.run(
            (
                sys.executable,
                "-c",
                pp3.REMOTE_PROCESS_STOP_PROGRAM,
                json.dumps(receipt),
                "2",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        cleanup = json.loads(completed.stdout)

        assert cleanup["ownership_verified"] is True
        assert cleanup["terminated"] is True, cleanup
    finally:
        if leader.poll() is None:
            os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5.0)


def test_remote_stop_program_fails_closed_on_live_metadata_read_errors() -> None:
    program = pp3.REMOTE_PROCESS_STOP_PROGRAM

    assert "except (FileNotFoundError, ProcessLookupError):" in program
    assert "cannot enumerate process group ownership" in program
    assert "cannot verify live owned process metadata" in program


_COUNTER_NAMES = (
    "VL15_dropped",
    "excessive_buffer_overrun_errors",
    "link_downed",
    "link_error_recovery",
    "local_link_integrity_errors",
    "port_rcv_constraint_errors",
    "port_rcv_data",
    "port_rcv_errors",
    "port_rcv_packets",
    "port_rcv_remote_physical_errors",
    "port_rcv_switch_relay_errors",
    "port_xmit_constraint_errors",
    "port_xmit_data",
    "port_xmit_discards",
    "port_xmit_packets",
    "symbol_error",
)


def hca_snapshot(host_name: str, offset: int) -> HcaCounterSnapshot:
    return HcaCounterSnapshot(
        schema_version=1,
        host_name=host_name,
        counter_source="sysfs_class_infiniband",
        captured_at_unix_seconds=1000.0 + offset,
        captured_at_monotonic_seconds=2000.0 + offset,
        ports=tuple(
            HcaPortCounterSnapshot(
                rail_id=f"rail-{port}",
                device="mlx4_0",
                port=port,
                state="4: ACTIVE",
                physical_state="5: LinkUp",
                rate="40 Gb/sec (4X QDR)",
                counters={name: 100 + offset for name in _COUNTER_NAMES},
            )
            for port in (1, 2)
        ),
    )


def test_hca_deltas_bind_each_host_and_convert_data_octets() -> None:
    before = {
        "dwagon": hca_snapshot("dwagon", 0),
        "fwuff": hca_snapshot("fwuff", 0),
    }
    after = {
        "dwagon": hca_snapshot("dwagon", 11),
        "fwuff": hca_snapshot("fwuff", 11),
    }

    deltas = pp3.calculate_hca_deltas(before, after)

    for host_name in ("dwagon", "fwuff"):
        host = cast(dict[str, object], deltas[host_name])
        for rail_id in ("rail-1", "rail-2"):
            rail = cast(dict[str, object], host[rail_id])
            assert rail["received_payload_bytes"] == 44
            assert rail["transmitted_payload_bytes"] == 44
            assert cast(dict[str, int], rail["counter_deltas"])["symbol_error"] == 11


def test_readiness_probes_every_logical_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = pp3.build_pp3_process_specs(make_config(tmp_path))
    seen: list[str] = []

    class Observation:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"status_code": 200}

    class Client:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 2.0
            seen.append(base_url)

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def health_generate(self) -> Observation:
            return Observation()

    monkeypatch.setattr(pp3, "Glm47NativeServingClient", Client)
    monkeypatch.setattr(pp3, "all_stages_alive", lambda _running: True)
    observations = pp3.wait_for_all_stages(specs, (), 1.0)

    assert seen == [
        "http://192.168.40.24:62410",
        "http://192.168.40.24:62411",
        "http://192.168.40.248:62412",
    ]
    assert observations == (
        {"status_code": 200},
        {"status_code": 200},
        {"status_code": 200},
    )
