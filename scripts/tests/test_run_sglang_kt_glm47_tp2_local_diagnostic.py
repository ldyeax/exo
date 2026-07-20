from __future__ import annotations

import hashlib
import json
import os
import signal
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import scripts.run_sglang_kt_glm47_tp2_local_diagnostic as tp2
from scripts import run_sglang_kt_glm47_pp3_diagnostic as pipeline
from scripts.sglang_kt_glm47_serving_client import (
    EndpointCallObservation,
    ServerInfoObservation,
)
from scripts.two_host_mlx_nccl_poc import ProcessCleanupReceipt


def make_config(tmp_path: Path) -> tp2.Tp2LocalDiagnosticConfig:
    return tp2.Tp2LocalDiagnosticConfig(
        run_id="glm47-tp2-local-test",
        result_directory=tmp_path / "result",
        dwagon_runtime_python="/runtime/dwagon/bin/python",
        dwagon_runtime_install_receipt=tmp_path / "runtime/install-receipt.json",
        dwagon_runtime_install_receipt_sha256="0" * 64,
        dwagon_model_path=pipeline.DEFAULT_DWAGON_MODEL_PATH,
        local_source_directory="/source/dwagon",
        dwagon_ip="192.168.40.24",
        dwagon_socket_interface="ens13f0np0",
        distributed_port=62600,
        service_port=62610,
        static_memory_fraction=0.9,
        resident_gpu_experts=40,
        readiness_timeout_seconds=60.0,
        request_timeout_seconds=900.0,
        cleanup_timeout_seconds=30.0,
        warmup_count=2,
        sample_count=3,
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def deterministic_telemetry(
    phase: str,
    stage_cpu_bindings: tuple[tp2.StageCpuCoreBinding, ...],
    expected_gpu_uuids: tuple[str, ...],
) -> tp2.pp2._HostTelemetrySnapshot:
    assert stage_cpu_bindings == (
        (0, tuple(range(56))),
        (1, tuple(range(56, 112))),
    )
    assert expected_gpu_uuids == tp2.DWAGON_GPU_UUIDS
    return tp2.pp2._HostTelemetrySnapshot(
        phase=phase,
        monotonic_nanoseconds=1,
        cpu_package_energy=(),
        evidence={
            "phase": phase,
            "observed_at_utc": "2026-07-20T12:00:00+00:00",
            "monotonic_nanoseconds": 1,
            "collection_elapsed_seconds": 0.001,
            "status": "complete",
            "cpu_stage_frequency": [],
            "cpu_package_temperature": [],
            "cpu_package_energy": [],
            "gpu": {"devices": []},
            "failures": [],
        },
    )


def test_builds_exact_single_parent_tp2_command_and_receipt(tmp_path: Path) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    assert spec.pipeline_parallel_size == 1
    assert spec.tensor_parallel_size == 2
    assert spec.ordered_gpu_uuids == (
        pipeline.DWAGON_STAGE_ZERO_GPU,
        pipeline.DWAGON_STAGE_ONE_GPU,
    )
    assert spec.cpu_cores == tuple(range(112))
    assert spec.memory_nodes == (0, 1)
    assert spec.cpu_infer_threads == 112
    assert spec.threadpool_count == 2
    assert spec.resident_gpu_experts == 40
    assert argument_value(spec.command, "--pp-size") == "1"
    assert argument_value(spec.command, "--tp-size") == "2"
    assert argument_value(spec.command, "--nnodes") == "1"
    assert argument_value(spec.command, "--node-rank") == "0"
    assert argument_value(spec.command, "--kt-cpuinfer") == "112"
    assert argument_value(spec.command, "--kt-threadpool-count") == "2"
    assert spec.command[spec.command.index("--kt-numa-nodes") + 1 :][0:2] == (
        "0",
        "1",
    )
    assert argument_value(spec.command, "--kt-num-gpu-experts") == "40"
    assert dict(spec.environment)["CUDA_VISIBLE_DEVICES"] == ",".join(
        spec.ordered_gpu_uuids
    )

    receipt = spec.receipt()
    parallelism = cast(dict[str, object], receipt["parallelism"])
    experts = cast(dict[str, object], receipt["experts"])
    gpu_workers = cast(list[dict[str, object]], receipt["gpu_workers"])
    assert parallelism == {
        "parent_process_count": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 2,
        "node_count": 1,
        "node_rank": 0,
    }
    assert experts["count_semantics"] == "per_layer_global_logical_expert_count"
    assert experts["resident_gpu_experts"] == 40
    launch_intent = cast(dict[str, object], receipt["hybrid_execution_launch_intent"])
    assert launch_intent == {
        "evidence_scope": "launch_intent_not_observed_execution",
        "cpuinfer_owner_tensor_parallel_rank": 0,
        "cpuinfer_result_scope": "full_cpu_expert_result",
        "cpuinfer_result_contribution_count": 1,
        "cpuinfer_merge_order": (
            "contributed_exactly_once_before_tensor_parallel_all_reduce"
        ),
        "resident_expert_gpu_weight_partition": (
            "tensor_parallel_half_shard_per_gpu_for_each_resident_expert"
        ),
    }
    assert [worker["tensor_parallel_rank"] for worker in gpu_workers] == [0, 1]
    assert [worker["matching_numa_node"] for worker in gpu_workers] == [0, 1]
    assert [worker["numa_local_physical_cpu_ids"] for worker in gpu_workers] == [
        list(range(56)),
        list(range(56, 112)),
    ]
    assert all("matching_physical_cpu_ids" not in worker for worker in gpu_workers)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("ordered_gpu_uuids", tuple(reversed(tp2.DWAGON_GPU_UUIDS))),
        ("cpu_cores", tuple(range(111))),
        ("memory_nodes", (1, 0)),
        ("cpu_infer_threads", 111),
        ("threadpool_count", 1),
        ("pipeline_parallel_size", 2),
        ("tensor_parallel_size", 1),
    ),
)
def test_process_spec_rejects_topology_substitution(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    with pytest.raises(ValueError, match="pinned dwagon topology"):
        replace(spec, **{field: replacement})


def test_process_spec_rejects_model_path_substitution(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), dwagon_model_path="/models/substitute")

    with pytest.raises(ValueError, match="pinned GLM-4.7 model path"):
        tp2.build_tp2_local_process_spec(config)


@pytest.mark.parametrize("resident_gpu_experts", (0, 45))
def test_process_spec_rejects_out_of_range_resident_gpu_experts(
    tmp_path: Path,
    resident_gpu_experts: int,
) -> None:
    config = replace(
        make_config(tmp_path),
        resident_gpu_experts=resident_gpu_experts,
    )

    with pytest.raises(ValueError, match="resident GPU experts"):
        tp2.build_tp2_local_process_spec(config)


def make_server_info_observation(
    *,
    tp_size: int = 2,
    kt_cpuinfer: int = 112,
    kt_num_gpu_experts: int = 40,
    dist_init_addr: str = "192.168.40.24:62600",
    mem_fraction_static: float = 0.9,
    status_code: int = 200,
) -> ServerInfoObservation:
    response: tp2.JsonObject = {
        "version": tp2.GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION,
        "model_path": pipeline.DEFAULT_DWAGON_MODEL_PATH,
        "kt_weight_path": pipeline.DEFAULT_DWAGON_MODEL_PATH,
        "tp_size": tp_size,
        "pp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
        "dist_init_addr": dist_init_addr,
        "kt_method": "BF16",
        "kt_cpuinfer": kt_cpuinfer,
        "kt_threadpool_count": 2,
        "kt_numa_nodes": [0, 1],
        "kt_num_gpu_experts": kt_num_gpu_experts,
        "kt_max_deferred_experts_per_token": 0,
        "kt_expert_placement_strategy": "uniform",
        "mem_fraction_static": mem_fraction_static,
        "attention_backend": "flashinfer",
        "kv_cache_dtype": "bfloat16",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_shared_experts_fusion": True,
        "chunked_prefill_size": tp2.GLM_4_7_FLASH_CHUNKED_PREFILL_SIZE,
        "context_length": tp2.GLM_4_7_FLASH_CONTEXT_LENGTH,
        "max_total_tokens": tp2.GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
        "max_running_requests": 1,
        "served_model_name": "GLM-4.7-Flash",
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "trust_remote_code": True,
        "scheduler_detail": "raw-field-is-preserved",
    }
    return ServerInfoObservation(
        call=EndpointCallObservation(
            status_code=status_code,
            response_sha256="1" * 64,
            elapsed_seconds=0.1,
        ),
        response=response,
        canonical_response_sha256=hashlib.sha256(
            tp2.canonical_sglang_kt_json(response)
        ).hexdigest(),
    )


def test_server_info_gate_accepts_and_preserves_exact_raw_tp2_response(
    tmp_path: Path,
) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))
    observation = make_server_info_observation()

    evidence = tp2.validate_tp2_server_info(observation, spec)

    identity = cast(dict[str, object], evidence["validated_identity"])
    raw = cast(dict[str, object], evidence["raw_observation"])
    response = cast(dict[str, object], raw["response"])
    assert identity["tp_size"] == 2
    assert identity["pp_size"] == 1
    assert identity["version"] == tp2.GLM_4_7_FLASH_PINNED_SGLANG_SERVER_VERSION
    assert response["scheduler_detail"] == "raw-field-is-preserved"


def test_e44_is_bound_through_command_receipts_and_server_info(
    tmp_path: Path,
) -> None:
    config = replace(
        make_config(tmp_path),
        resident_gpu_experts=44,
        static_memory_fraction=0.95,
    )
    spec = tp2.build_tp2_local_process_spec(config)

    assert spec.resident_gpu_experts == 44
    assert argument_value(spec.command, "--kt-num-gpu-experts") == "44"
    assert argument_value(spec.command, "--mem-fraction-static") == "0.95"
    process_receipt = spec.receipt()
    experts = cast(dict[str, object], process_receipt["experts"])
    assert experts["resident_gpu_experts"] == 44
    assert process_receipt["static_memory_fraction"] == 0.95

    server_info = tp2.validate_tp2_server_info(
        make_server_info_observation(
            kt_num_gpu_experts=44,
            mem_fraction_static=0.95,
        ),
        spec,
    )

    validated = cast(dict[str, object], server_info["validated_identity"])
    assert validated["kt_num_gpu_experts"] == 44
    assert validated["mem_fraction_static"] == 0.95
    configuration = tp2._configuration_receipt(config, spec)
    assert configuration["resident_gpu_experts"] == 44
    assert configuration["static_memory_fraction"] == 0.95
    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="kt_num_gpu_experts"):
        tp2.validate_tp2_server_info(make_server_info_observation(), spec)


def test_server_info_gate_rejects_tp1_response(tmp_path: Path) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="tp_size"):
        tp2.validate_tp2_server_info(
            make_server_info_observation(tp_size=1),
            spec,
        )


def test_server_info_gate_rejects_wrong_cpuinfer_allocation(tmp_path: Path) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="kt_cpuinfer"):
        tp2.validate_tp2_server_info(
            make_server_info_observation(kt_cpuinfer=56),
            spec,
        )


def test_server_info_gate_rejects_wrong_distributed_coordinator(
    tmp_path: Path,
) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="dist_init_addr"):
        tp2.validate_tp2_server_info(
            make_server_info_observation(dist_init_addr="127.0.0.1:62600"),
            spec,
        )


def test_launch_preflight_rejects_an_occupied_service_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OccupiedSocket:
        def bind(self, _endpoint: tuple[str, int]) -> None:
            raise OSError("address already in use")

        def close(self) -> None:
            return None

    monkeypatch.setattr(tp2.socket, "socket", lambda *_args: OccupiedSocket())
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="service endpoint"):
        tp2.require_launch_ports_available(spec)


def test_launch_pending_journal_exists_before_popen_and_is_immediately_updated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    config.result_directory.mkdir()
    spec = tp2.build_tp2_local_process_spec(config)
    observed_pending: dict[str, object] = {}

    class CreatedProcess:
        pid = 7331

    def popen(*_args: object, **_kwargs: object) -> CreatedProcess:
        journal_path = tp2._ownership_journal_path(config)
        assert journal_path.stat().st_mode & 0o777 == 0o600
        pending = cast(dict[str, object], json.loads(journal_path.read_text()))
        observed_pending.update(pending)
        return CreatedProcess()

    monkeypatch.setattr(tp2, "require_launch_ports_available", lambda _spec: None)
    monkeypatch.setattr(tp2.subprocess, "Popen", popen)
    monkeypatch.setattr(tp2, "_read_process_identity", lambda _pid: (7331, 12345))

    running = tp2.start_local_parent(spec, config, "pre-popen-owner-token")

    assert observed_pending["status"] == "launch_pending_before_popen"
    assert observed_pending["started_parent_process_count"] == 0
    launch_intent = cast(dict[str, object], observed_pending["launch_intent"])
    assert launch_intent["owner_token"] == "pre-popen-owner-token"
    assert (
        launch_intent["owner_token_sha256"]
        == hashlib.sha256(b"pre-popen-owner-token").hexdigest()
    )
    assert launch_intent["service_endpoint"] == str(spec.service_endpoint)
    assert launch_intent["distributed_coordinator"] == str(spec.distributed_coordinator)
    recovery = cast(dict[str, object], observed_pending["recovery"])
    assert "/proc/[0-9]*/environ" in cast(str, recovery["process_discovery"])
    updated = cast(
        dict[str, object],
        json.loads(tp2._ownership_journal_path(config).read_text()),
    )
    assert updated["status"] == "partial_start_registration_pending"
    process = cast(list[dict[str, object]], updated["processes"])[0]
    assert process["pid"] == 7331
    assert process["owner_token"] == "pre-popen-owner-token"
    assert running.launch_evidence["popen_returned"] is True
    running.log_file.close()
    tp2._clear_ownership_journal(config)


def test_ready_listener_is_bound_to_owned_group_session_and_token(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), dwagon_ip="127.0.0.1")
    spec = tp2.build_tp2_local_process_spec(config)
    proc_root = tmp_path / "proc"
    (proc_root / "net").mkdir(parents=True)
    listener_inode = "987654"
    listener_token = tp2._ipv4_proc_listener_token(spec.service_endpoint)
    (proc_root / "net/tcp").write_text(
        "  sl  local_address rem_address st tx_queue tr tm->when retrnsmt uid timeout inode\n"
        f"   0: {listener_token} 00000000:0000 0A 00000000:00000000 "
        f"00:00000000 00000000 0 0 {listener_inode}\n"
    )
    process_root = proc_root / "4242"
    (process_root / "fd").mkdir(parents=True)
    (process_root / "stat").write_text(
        "4242 (sglang) S 1 4242 4242 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 123 0\n"
    )
    (process_root / "environ").write_bytes(b"EXO_BENCHMARK_OWNER_TOKEN=owned-token\0")
    os.symlink(f"socket:[{listener_inode}]", process_root / "fd/7")
    owned = pipeline.OwnedStageProcess(
        rank=0,
        host_name="dwagon",
        pid=4242,
        process_group_id=4242,
        start_time_ticks=123,
        owner_token="owned-token",
        ownership_namespace=str(config.service_port),
        remote=False,
        transport_pid=4242,
        log_path="/tmp/rank-0.log",
    )
    running = SimpleNamespace(owned=owned)

    evidence = tp2.verify_owned_service_listener(
        spec,
        cast(tp2.RunningParent, running),
        proc_root=proc_root,
    )

    assert evidence["listener_pids"] == [4242]
    assert evidence["owned_process_group_id"] == 4242
    distributed_absent = tp2.observe_distributed_coordinator_listener_ownership(
        spec,
        cast(tp2.RunningParent, running),
        proc_root=proc_root,
    )
    assert distributed_absent["status"] == "not_listening_after_tp_initialization"
    assert distributed_absent["persistent_listener_required"] is False

    distributed_inode = "987655"
    distributed_token = tp2._ipv4_proc_listener_token(spec.distributed_coordinator)
    with (proc_root / "net/tcp").open("a") as tcp_table:
        tcp_table.write(
            f"   1: {distributed_token} 00000000:0000 0A 00000000:00000000 "
            f"00:00000000 00000000 0 0 {distributed_inode}\n"
        )
    os.symlink(f"socket:[{distributed_inode}]", process_root / "fd/8")
    distributed_owned = tp2.observe_distributed_coordinator_listener_ownership(
        spec,
        cast(tp2.RunningParent, running),
        proc_root=proc_root,
    )
    assert distributed_owned["status"] == "listening_and_owned"
    assert distributed_owned["listener_pids"] == [4242]

    (process_root / "environ").write_bytes(b"EXO_BENCHMARK_OWNER_TOKEN=stale\0")
    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="not owned"):
        tp2.verify_owned_service_listener(
            spec,
            cast(tp2.RunningParent, running),
            proc_root=proc_root,
        )


def test_parent_command_binds_all_physical_cores_and_both_memory_nodes(
    tmp_path: Path,
) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))

    command = tp2.build_parent_command(spec)

    assert command[:5] == (
        "/usr/bin/numactl",
        "--physcpubind",
        ",".join(str(cpu) for cpu in range(112)),
        "--membind",
        "0,1",
    )
    assert command[5:] == spec.command


def test_parent_environment_is_local_p2p_eligible_and_rejects_inherited_tuning(
    tmp_path: Path,
) -> None:
    spec = tp2.build_tp2_local_process_spec(make_config(tmp_path))
    hostile_parent = {
        "HOME": "/root",
        "PATH": "/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": "GPU-substitute",
        "NCCL_NET": "IB",
        "NCCL_IB_HCA": "=mlx4_0:1,mlx4_0:2",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "SGLANG_PP_LAYER_PARTITION": "24,23",
    }

    environment = tp2.build_parent_environment(
        spec,
        "owner-token",
        "ens13f0np0",
        hostile_parent,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(tp2.DWAGON_GPU_UUIDS)
    assert environment["NCCL_DEBUG"] == "INFO"
    assert environment["NCCL_SOCKET_IFNAME"] == "ens13f0np0"
    assert environment["GLOO_SOCKET_IFNAME"] == "ens13f0np0"
    for forbidden in (
        "NCCL_NET",
        "NCCL_IB_HCA",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "SGLANG_PP_LAYER_PARTITION",
    ):
        assert forbidden not in environment


def test_runtime_and_model_admission_reuses_exact_pp2_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    observed: list[object] = []

    def verify(value: object) -> tp2.JsonObject:
        observed.append(value)
        return {"install_id": "verified"}

    monkeypatch.setattr(tp2.pp2, "_verify_runtime_and_model_contract", verify)

    assert tp2._verify_runtime_and_model_contract(config) == {"install_id": "verified"}
    assert observed == [config]


def test_parser_rejects_duplicate_ports_and_relative_runtime_paths(
    tmp_path: Path,
) -> None:
    common = (
        "--run-id",
        "test",
        "--result-directory",
        str(tmp_path / "result"),
        "--dwagon-runtime-python",
        "/runtime/python",
        "--dwagon-runtime-install-receipt",
        "/runtime/install-receipt.json",
        "--dwagon-runtime-install-receipt-sha256",
        "0" * 64,
    )
    default_config = tp2._config_from_arguments(tp2._parser().parse_args(common))
    assert default_config.static_memory_fraction == 0.9
    assert default_config.resident_gpu_experts == 40
    e44_config = tp2._config_from_arguments(
        tp2._parser().parse_args((*common, "--resident-gpu-experts", "44"))
    )
    assert e44_config.resident_gpu_experts == 44
    for invalid_resident_count in ("0", "45"):
        with pytest.raises(SystemExit):
            tp2._parser().parse_args(
                (*common, "--resident-gpu-experts", invalid_resident_count)
            )
    duplicate_ports = tp2._parser().parse_args(
        (*common, "--distributed-port", "62600", "--service-port", "62600")
    )
    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="distinct TCP ports"):
        tp2._config_from_arguments(duplicate_ports)

    relative_runtime = tp2._parser().parse_args(
        (
            *common,
            "--dwagon-runtime-python",
            "runtime/python",
        )
    )
    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="absolute paths"):
        tp2._config_from_arguments(relative_runtime)


class _Evidence:
    def __init__(self, payload: tp2.JsonObject) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str) -> tp2.JsonObject:
        assert mode == "json"
        return self.payload


class _Workload(_Evidence):
    def __init__(self, kind: str) -> None:
        super().__init__({"kind": kind})
        self.kind = kind


class _Client:
    def __init__(self, endpoint: str, *, timeout_seconds: float) -> None:
        assert endpoint == "http://192.168.40.24:62610"
        assert timeout_seconds == 900.0

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def server_info(self) -> ServerInfoObservation:
        return make_server_info_observation()


def fake_running(config: tp2.Tp2LocalDiagnosticConfig) -> SimpleNamespace:
    log_path = config.result_directory / "rank-0.log"
    log_path.write_text("EXO_TP2_LAUNCH fixture\n")
    owned = pipeline.OwnedStageProcess(
        rank=0,
        host_name="dwagon",
        pid=1000,
        process_group_id=1000,
        start_time_ticks=2000,
        owner_token="raw-owner-token",
        ownership_namespace=str(config.service_port),
        remote=False,
        transport_pid=1000,
        log_path=str(log_path),
    )
    return SimpleNamespace(
        owned=owned,
        process=SimpleNamespace(poll=lambda: None),
        launch_evidence=tp2._launch_evidence(
            ("/actual/runtime/python", "-m", "sglang.launch_server"),
            {
                "CUDA_VISIBLE_DEVICES": ",".join(tp2.DWAGON_GPU_UUIDS),
                "EXO_BENCHMARK_OWNER_TOKEN": "raw-owner-token",
                "LD_LIBRARY_PATH": "/captured/at/popen",
            },
            config.local_source_directory,
        ),
    )


def install_run_fakes(
    config: tp2.Tp2LocalDiagnosticConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    calls: list[str] = []

    def sanity(_client: _Client, model_path: str) -> _Evidence:
        assert model_path == pipeline.DEFAULT_DWAGON_MODEL_PATH
        calls.append("sanity")
        return _Evidence({"output_text": "EXO_SANITY_OK"})

    def workload(
        _client: _Client,
        kind: str,
        *,
        warmup_count: int,
        sample_count: int,
        phase_observer: Callable[[str], None],
    ) -> _Workload:
        assert warmup_count == 2
        assert sample_count == 3
        calls.append(kind)
        phase_observer("warmups_complete")
        phase_observer("samples_complete")
        return _Workload(kind)

    monkeypatch.setattr(
        tp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )
    monkeypatch.setattr(
        tp2.pp2, "_safe_collect_host_telemetry_snapshot", deterministic_telemetry
    )
    monkeypatch.setattr(tp2, "start_local_parent", lambda *_args: fake_running(config))
    monkeypatch.setattr(tp2, "wait_for_parent_readiness", lambda *_args: ({},))
    monkeypatch.setattr(
        tp2,
        "verify_owned_service_listener",
        lambda *_args: {"verification": "owned-fixture"},
    )
    monkeypatch.setattr(
        tp2,
        "observe_distributed_coordinator_listener_ownership",
        lambda *_args: {"status": "not-listening-fixture"},
    )
    monkeypatch.setattr(tp2, "Glm47NativeServingClient", _Client)
    monkeypatch.setattr(tp2, "run_glm47_serving_sanity", sanity)
    monkeypatch.setattr(tp2, "prepare_glm47_serving_workload", lambda kind: kind)
    monkeypatch.setattr(tp2, "run_glm47_serving_workload", workload)
    monkeypatch.setattr(
        tp2.pipeline,
        "summarize_workload",
        lambda observed: {"kind": observed.kind, "median_tokens_per_second": 1.0},
    )
    return calls


def test_run_publishes_truthful_tp2_receipt_and_cleans_owned_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    calls = install_run_fakes(config, monkeypatch)

    def stop(
        running: SimpleNamespace,
        cleanup_timeout_seconds: float,
    ) -> ProcessCleanupReceipt:
        assert running.owned.rank == 0
        assert cleanup_timeout_seconds == 30.0
        assert tp2._ownership_journal_path(config).is_file()
        calls.append("cleanup")
        return ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    monkeypatch.setattr(tp2, "stop_local_parent", stop)

    payload = tp2.run_diagnostic(config)

    assert payload["schema_version"] == 1
    assert payload["kind"] == "glm47_flash_pp1_tp2_local_engineering_diagnostic"
    assert payload["status"] == "passed"
    assert calls == ["sanity", "prefill", "decode", "cleanup"]
    assert payload["runtime_contract"] == {"install_id": "verified"}
    topology = cast(dict[str, object], payload["topology"])
    assert topology["parent_process_count"] == 1
    assert topology["gpu_worker_count"] == 2
    assert topology["pipeline_parallel_size"] == 1
    assert topology["tensor_parallel_size"] == 2
    configuration = cast(dict[str, object], payload["configuration"])
    assert configuration["ordered_gpu_uuids"] == list(tp2.DWAGON_GPU_UUIDS)
    assert configuration["physical_cpu_ids"] == list(range(112))
    assert configuration["memory_nodes"] == [0, 1]
    assert configuration["cpu_infer_threads"] == 112
    assert configuration["threadpool_count"] == 2
    assert configuration["resident_gpu_experts"] == 40
    assert configuration["resident_gpu_expert_count_semantics"] == (
        "per_layer_global_logical_expert_count"
    )
    assert configuration["hybrid_execution_launch_intent"] == {
        "evidence_scope": "launch_intent_not_observed_execution",
        "cpuinfer_owner_tensor_parallel_rank": 0,
        "cpuinfer_result_scope": "full_cpu_expert_result",
        "cpuinfer_result_contribution_count": 1,
        "cpuinfer_merge_order": (
            "contributed_exactly_once_before_tensor_parallel_all_reduce"
        ),
        "resident_expert_gpu_weight_partition": (
            "tensor_parallel_half_shard_per_gpu_for_each_resident_expert"
        ),
    }
    process_spec = cast(tp2.JsonObject, payload["process_spec"])
    assert payload["process_spec_sha256"] == tp2._canonical_sha256(process_spec)
    launch = cast(dict[str, object], payload["launch"])
    environment = cast(dict[str, object], launch["environment"])
    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(tp2.DWAGON_GPU_UUIDS)
    assert environment["EXO_BENCHMARK_OWNER_TOKEN"] == "<redacted>"
    assert environment["LD_LIBRARY_PATH"] == "/captured/at/popen"
    assert launch["evidence_origin"] == "captured_immediately_before_popen"
    assert launch["argv"] == [
        "/actual/runtime/python",
        "-m",
        "sglang.launch_server",
    ]
    assert launch["exact_environment_sha256"] == tp2._canonical_sha256(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(tp2.DWAGON_GPU_UUIDS),
            "EXO_BENCHMARK_OWNER_TOKEN": "raw-owner-token",
            "LD_LIBRARY_PATH": "/captured/at/popen",
        }
    )
    assert payload["readiness_ownership"] == {"verification": "owned-fixture"}
    assert payload["distributed_coordinator_ownership"] == {
        "status": "not-listening-fixture"
    }
    server_info = cast(dict[str, object], payload["server_info"])
    validated_server = cast(dict[str, object], server_info["validated_identity"])
    assert validated_server["tp_size"] == 2
    assert validated_server["kt_num_gpu_experts"] == 40
    host_telemetry = cast(dict[str, object], payload["host_telemetry"])
    telemetry_semantics = cast(
        dict[str, object], host_telemetry["cpu_binding_label_semantics"]
    )
    assert telemetry_semantics["source_field_name"] == "pipeline_rank"
    assert telemetry_semantics["meaning_in_this_tp2_receipt"] == (
        "cpuinfer_threadpool_index"
    )
    assert telemetry_semantics["bindings"] == [
        {
            "cpuinfer_threadpool_index": 0,
            "physical_cpu_ids": list(range(56)),
        },
        {
            "cpuinfer_threadpool_index": 1,
            "physical_cpu_ids": list(range(56, 112)),
        },
    ]
    assert "raw-owner-token" not in json.dumps(payload)
    assert payload["cleanup_complete"] is True
    assert not tp2._ownership_journal_path(config).exists()
    receipt_path = config.result_directory / "tp2-local-diagnostic-result.json"
    assert (
        json.loads(receipt_path.read_text())["receipt_content_sha256"]
        == (payload["receipt_content_sha256"])
    )


def test_incomplete_cleanup_retains_mode_0600_recovery_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_run_fakes(config, monkeypatch)
    monkeypatch.setattr(
        tp2,
        "wait_for_parent_readiness",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )
    monkeypatch.setattr(
        tp2,
        "stop_local_parent",
        lambda *_args: ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=False,
            terminated=False,
            forced=False,
        ),
    )

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="readiness failed"):
        tp2.run_diagnostic(config)

    journal_path = tp2._ownership_journal_path(config)
    assert journal_path.is_file()
    assert journal_path.stat().st_mode & 0o777 == 0o600
    journal = cast(dict[str, object], json.loads(journal_path.read_text()))
    processes = cast(list[dict[str, object]], journal["processes"])
    assert processes[0]["owner_token"] == "raw-owner-token"
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["status"] == "failed"
    assert receipt["cleanup_complete"] is False
    assert cast(dict[str, object], receipt["ownership_journal"])["retained"] is True


def test_managed_signal_handler_is_restored_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_run_fakes(config, monkeypatch)
    prior = signal.getsignal(signal.SIGTERM)

    def start(*_args: object) -> SimpleNamespace:
        running = fake_running(config)
        signal.raise_signal(signal.SIGTERM)
        return running

    monkeypatch.setattr(tp2, "start_local_parent", start)
    monkeypatch.setattr(
        tp2,
        "stop_local_parent",
        lambda *_args: ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        ),
    )

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="managed signal"):
        tp2.run_diagnostic(config)

    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["managed_signal"] == signal.SIGTERM
    assert receipt["cleanup_complete"] is True
    assert signal.getsignal(signal.SIGTERM) == prior


def install_popen_failure_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )
    monkeypatch.setattr(
        tp2.pp2,
        "_safe_collect_host_telemetry_snapshot",
        deterministic_telemetry,
    )
    monkeypatch.setattr(tp2, "require_launch_ports_available", lambda _spec: None)

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture Popen failure")

    monkeypatch.setattr(tp2.subprocess, "Popen", fail_popen)


def test_popen_failure_clears_prelaunch_journal_and_records_no_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_popen_failure_fakes(monkeypatch)

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="Popen failed"):
        tp2.run_diagnostic(config)

    assert not tp2._ownership_journal_path(config).exists()
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["started_parent_process_count"] == 0
    assert receipt["registered_parent_process_count"] == 0
    assert receipt["cleanup_complete"] is True
    launch_failure = cast(dict[str, object], receipt["parent_launch_failure"])
    assert launch_failure == {
        "popen_attempted": True,
        "popen_returned": False,
        "process_created": False,
        "journal_error": None,
    }
    journal = cast(dict[str, object], receipt["ownership_journal"])
    assert journal["created"] is True
    assert journal["cleared_before_receipt"] is True
    assert journal["clear_reason"] == "popen_failed_before_process_creation"
    assert journal["retained"] is False
    assert (
        "EXO_BENCHMARK_OWNER_TOKEN"
        in cast(dict[str, object], receipt["launch"])["environment"]
    )
    launch_environment = cast(
        dict[str, object], cast(dict[str, object], receipt["launch"])["environment"]
    )
    assert launch_environment["EXO_BENCHMARK_OWNER_TOKEN"] == "<redacted>"


def test_popen_failure_truthfully_retains_unclearable_prelaunch_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_popen_failure_fakes(monkeypatch)
    monkeypatch.setattr(
        tp2,
        "_clear_ownership_journal",
        lambda _config: (_ for _ in ()).throw(OSError("fixture clear failure")),
    )

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="Popen failed"):
        tp2.run_diagnostic(config)

    journal_path = tp2._ownership_journal_path(config)
    assert journal_path.is_file()
    assert journal_path.stat().st_mode & 0o777 == 0o600
    durable_journal = cast(dict[str, object], json.loads(journal_path.read_text()))
    assert durable_journal["status"] == "launch_failed_before_process_handle"
    assert "fixture Popen failure" in cast(str, durable_journal["failure"])
    launch_intent = cast(dict[str, object], durable_journal["launch_intent"])
    assert launch_intent["owner_token"]
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["cleanup_complete"] is True
    ownership_journal = cast(dict[str, object], receipt["ownership_journal"])
    assert ownership_journal["cleared_before_receipt"] is False
    assert ownership_journal["retained"] is True
    launch_failure = cast(dict[str, object], receipt["parent_launch_failure"])
    assert "fixture clear failure" in cast(str, launch_failure["journal_error"])
    assert launch_intent["owner_token"] not in json.dumps(receipt)


class _CreatedButUnregisteredProcess:
    pid = 7331


def install_partial_start_fakes(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_receipt: ProcessCleanupReceipt,
) -> None:
    monkeypatch.setattr(
        tp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )
    monkeypatch.setattr(
        tp2.pp2,
        "_safe_collect_host_telemetry_snapshot",
        deterministic_telemetry,
    )
    monkeypatch.setattr(tp2, "require_launch_ports_available", lambda _spec: None)
    monkeypatch.setattr(
        tp2.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _CreatedButUnregisteredProcess(),
    )
    monkeypatch.setattr(
        tp2,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(OSError("proc identity unavailable")),
    )
    monkeypatch.setattr(
        tp2.pipeline,
        "_terminate_unregistered_local_process",
        lambda *_args: cleanup_receipt,
    )


def test_unverified_partial_start_retains_recovery_journal_and_fails_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_partial_start_fakes(
        monkeypatch,
        ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=False,
            forced=True,
            error="process group survived SIGKILL",
        ),
    )

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="could not be registered"):
        tp2.run_diagnostic(config)

    journal_path = tp2._ownership_journal_path(config)
    assert journal_path.is_file()
    assert journal_path.stat().st_mode & 0o777 == 0o600
    journal = cast(dict[str, object], json.loads(journal_path.read_text()))
    assert journal["status"] == "partial_start_recovery_required"
    journal_process = cast(list[dict[str, object]], journal["processes"])[0]
    assert journal_process["pid"] == 7331
    assert journal_process["assumed_process_group_id"] == 7331
    assert journal_process["owner_token"]
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["status"] == "failed"
    assert receipt["started_parent_process_count"] == 1
    assert receipt["registered_parent_process_count"] == 0
    assert receipt["cleanup_complete"] is False
    assert cast(dict[str, object], receipt["ownership_journal"])["retained"] is True
    partial_start = cast(dict[str, object], receipt["partial_start"])
    assert partial_start["pid"] == 7331
    assert partial_start["identity_registered"] is False
    assert "owner_token" not in partial_start


def test_verified_partial_start_cleanup_is_recorded_and_clears_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    install_partial_start_fakes(
        monkeypatch,
        ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        ),
    )

    with pytest.raises(tp2.Tp2LocalDiagnosticError, match="could not be registered"):
        tp2.run_diagnostic(config)

    assert not tp2._ownership_journal_path(config).exists()
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "tp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["status"] == "failed"
    assert receipt["started_parent_process_count"] == 1
    assert receipt["registered_parent_process_count"] == 0
    assert receipt["cleanup_complete"] is True
    assert (
        cast(dict[str, object], receipt["ownership_journal"])["cleared_before_receipt"]
        is True
    )
    cleanup = cast(list[dict[str, object]], receipt["cleanup"])
    assert cleanup[0]["partial_start"] is True
    assert cleanup[0]["terminated"] is True
