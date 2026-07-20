from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from exo.shared.types.common import NodeId
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
)
from exo.worker.sglang_kt.native_glm47_parallelism import (
    DWAGON_RANK_ONE_CPUS,
    DWAGON_RANK_ZERO_CPUS,
    FWUFF_RANK_TWO_CPUS,
    Glm47NativeParallelism,
    build_dwagon_fwuff_native_glm47_plan,
    build_native_glm47_process_specs,
)
from scripts import run_sglang_kt_glm47_native_tp_ep_benchmark as native


def make_config(
    tmp_path: Path,
    *,
    mode: Glm47NativeParallelism = "tp3_ep1",
) -> native.NativeTpEpConfig:
    return native.NativeTpEpConfig(
        run_id="native-test",
        mode=mode,
        result_directory=tmp_path / "result",
        lock_path=tmp_path / "benchmark.lock",
        dwagon_runtime_python="/runtime/dwagon/bin/python",
        fwuff_runtime_python="/runtime/fwuff/bin/python",
        dwagon_model_path="/models/dwagon/glm47",
        fwuff_model_path="/models/fwuff/glm47",
        dwagon_sglang_source_directory="/source/dwagon/sglang",
        fwuff_sglang_source_directory="/source/fwuff/sglang",
        dwagon_repository_directory="/deploy/dwagon/exo",
        fwuff_repository_directory="/deploy/fwuff/exo",
        dwagon_model_contract="/deploy/dwagon/exo/contract.json",
        fwuff_model_contract="/deploy/fwuff/exo/contract.json",
        ssh_target="fwuff",
        dwagon_ip="192.168.40.24",
        fwuff_ip="192.168.40.248",
        dwagon_socket_interface="ens13f0np0",
        fwuff_socket_interface="ens17f0",
        distributed_port=30_000,
        rank_ports=(30_100, 30_101, 30_102),
        hca_devices=("mlx4_0:1", "mlx4_0:2"),
        readiness_timeout_seconds=30.0,
        request_timeout_seconds=20.0,
        cleanup_timeout_seconds=10.0,
    )


def make_plan(mode: Glm47NativeParallelism = "tp3_ep1"):
    return build_dwagon_fwuff_native_glm47_plan(
        mode,
        dwagon_runtime_python="/runtime/dwagon/bin/python",
        fwuff_runtime_python="/runtime/fwuff/bin/python",
        dwagon_sglang_source_directory="/source/dwagon/sglang",
        fwuff_sglang_source_directory="/source/fwuff/sglang",
        dwagon_model_path="/models/dwagon/glm47",
        fwuff_model_path="/models/fwuff/glm47",
        dwagon_ip="192.168.40.24",
        fwuff_ip="192.168.40.248",
        distributed_port=30_000,
        rank_ports=(30_100, 30_101, 30_102),
        hca_devices=("mlx4_0:1", "mlx4_0:2"),
        dwagon_socket_interface="ens13f0np0",
        fwuff_socket_interface="ens17f0",
    )


def test_run_parser_uses_non_ephemeral_fixed_ports_and_packaged_contracts(
    tmp_path: Path,
) -> None:
    arguments = native._parser().parse_args(
        (
            "run",
            "--run-id",
            "native-v1",
            "--mode",
            "tp3_ep1",
            "--result-directory",
            str(tmp_path / "result"),
            "--fwuff-runtime-python",
            "/runtime/fwuff/bin/python",
            "--fwuff-sglang-source-directory",
            "/source/fwuff/sglang",
            "--fwuff-repository-directory",
            "/deploy/fwuff/exo",
        )
    )

    config = native._run_config(arguments)

    assert config.distributed_port == 30_000
    assert config.rank_ports == (30_100, 30_101, 30_102)
    assert max((*config.rank_ports, config.distributed_port + 13)) < 32_768
    assert config.dwagon_model_contract.endswith(
        "/manifests/glm47_flash_bf16_7dd20894.json"
    )
    assert config.fwuff_model_contract == (
        "/deploy/fwuff/exo/src/exo/worker/sglang_kt/manifests/"
        "glm47_flash_bf16_7dd20894.json"
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ("--distributed-port", "49145"),
        ("--distributed-port", "30000", "--rank-zero-port", "30013"),
        ("--rank-one-port", "30100"),
    ),
)
def test_run_config_rejects_derived_or_service_port_collisions(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    parsed = native._parser().parse_args(
        (
            "run",
            "--run-id",
            "bad-ports",
            "--mode",
            "tp3_ep1",
            "--result-directory",
            str(tmp_path / "result"),
            "--fwuff-runtime-python",
            "/runtime/fwuff/bin/python",
            "--fwuff-sglang-source-directory",
            "/source/fwuff/sglang",
            "--fwuff-repository-directory",
            "/deploy/fwuff/exo",
            *arguments,
        )
    )
    with pytest.raises(native.NativeTpEpBenchmarkError):
        native._run_config(parsed)


def test_reserved_ports_cover_coordinator_offsets_on_both_hosts() -> None:
    local, remote = native._reserved_ports(make_plan())

    coordinator = (30_000, 30_001, 30_002, 30_003, 30_004, 30_005, 30_013)
    assert local == (*coordinator, 30_100, 30_101)
    assert remote == (*coordinator, 30_102)


@pytest.mark.parametrize(
    ("mode", "ep_size", "redundant"),
    (("tp3_ep1", "1", "0"), ("tp3_ep3", "3", "2")),
)
def test_lifecycle_adapter_preserves_native_command_environment_and_placement(
    tmp_path: Path,
    mode: Glm47NativeParallelism,
    ep_size: str,
    redundant: str,
) -> None:
    config = make_config(tmp_path, mode=mode)
    specs = build_native_glm47_process_specs(make_plan(mode))
    expected_cpus = (
        DWAGON_RANK_ZERO_CPUS,
        DWAGON_RANK_ONE_CPUS,
        FWUFF_RANK_TWO_CPUS,
    )

    for rank, spec in enumerate(specs):
        adapted = native._lifecycle_spec(spec)
        command = native.lifecycle.build_cpu_bound_sglang_kt_command(adapted)
        environment = native.lifecycle.build_stage_environment(
            adapted,
            "owner-token",
            spec.rank.socket_interface,
            {
                "PATH": "/usr/bin:/bin",
                "HOME": "/root",
                "NCCL_NET": "Socket",
                "SGLANG_TORCH_PROFILER": "1",
                "VTUNE_PROFILER_DIR": "/unsafe",
            },
        )
        assert command[:5] == (
            "/usr/bin/numactl",
            "--physcpubind",
            ",".join(str(core) for core in expected_cpus[rank]),
            "--membind",
            str(spec.rank.memory_nodes[0]),
        )
        assert spec.arguments[spec.arguments.index("--ep-size") + 1] == ep_size
        assert (
            spec.arguments[spec.arguments.index("--ep-num-redundant-experts") + 1]
            == redundant
        )
        assert environment["NCCL_NET"] == "IB"
        assert environment["NCCL_NET_GDR_LEVEL"] == "LOC"
        assert environment["NCCL_IB_HCA"] == "=mlx4_0:1,mlx4_0:2"
        assert environment["NCCL_IB_MERGE_NICS"] == "1"
        assert "SGLANG_TORCH_PROFILER" not in environment
        assert "VTUNE_PROFILER_DIR" not in environment

    lifecycle_config = native._lifecycle_config(config)
    assert lifecycle_config.local_source_directory == config.dwagon_repository_directory
    assert lifecycle_config.remote_source_directory == config.fwuff_repository_directory


def test_start_order_gates_each_nonzero_rank_before_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    specs = build_native_glm47_process_specs(make_plan())
    events: list[tuple[str, int, tuple[int, ...]]] = []

    def start(spec, _config, _owner_token):  # type: ignore[no-untyped-def]
        rank = int(spec.pipeline_rank)
        events.append(("start", rank, ()))
        return cast(
            native.lifecycle.RunningStage,
            SimpleNamespace(owned=SimpleNamespace(rank=rank)),
        )

    def gate(
        _config: native.NativeTpEpConfig,
        running,
        all_started,
        _latch,
    ):  # type: ignore[no-untyped-def]
        events.append(
            (
                "gate",
                int(running.owned.rank),
                tuple(int(stage.owned.rank) for stage in all_started),
            )
        )

    monkeypatch.setattr(native.lifecycle, "start_stage", start)
    monkeypatch.setattr(native, "wait_for_nonzero_port_admission", gate)
    running: list[native.lifecycle.RunningStage] = []

    native.start_native_stages(
        specs,
        config,
        native._lifecycle_config(config),
        "owner",
        native._CancellationLatch(),
        running,
    )

    assert events == [
        ("start", 2, ()),
        ("gate", 2, (2,)),
        ("start", 1, ()),
        ("gate", 1, (2, 1)),
        ("start", 0, ()),
    ]
    assert tuple(stage.owned.rank for stage in running) == (0, 1, 2)


def test_nonzero_gate_requires_exact_post_portargs_info_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "rank-2.log"
    log_path.write_text(
        "prefix server_args=ServerArgs(model_path='/models/fwuff/glm47') suffix\n"
    )
    running = cast(
        native.lifecycle.RunningStage,
        SimpleNamespace(owned=SimpleNamespace(rank=2, log_path=str(log_path))),
    )
    ownership_checks: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        native,
        "require_stages_owned",
        lambda _config, stages: ownership_checks.append(
            tuple(stage.owned.rank for stage in stages)
        ),
    )

    native.wait_for_nonzero_port_admission(
        make_config(tmp_path), running, (running,), native._CancellationLatch()
    )

    assert ownership_checks == [(2,), (2,)]


def test_server_info_must_match_native_tp_dp_ep_contract() -> None:
    plan = make_plan("tp3_ep3")
    response = {
        "model_path": str(plan.ranks[0].model_path),
        "tp_size": 3,
        "pp_size": 1,
        "dp_size": 3,
        "enable_dp_attention": True,
        "moe_dense_tp_size": 1,
        "ep_size": 3,
        "ep_num_redundant_experts": 2,
        "nnodes": 3,
        "node_rank": 0,
        "dist_init_addr": "192.168.40.24:30000",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "disable_shared_experts_fusion": True,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "max_running_requests": 3,
        "max_total_tokens": 2048,
        "mem_fraction_static": 0.92,
        "context_length": 2048,
        "chunked_prefill_size": 341,
    }

    native.validate_server_info(response, plan)
    response["ep_size"] = 1
    with pytest.raises(native.NativeTpEpBenchmarkError, match="server_info"):
        native.validate_server_info(response, plan)


class _ContractFile(SimpleNamespace):
    path: str
    role: str
    size_bytes: int
    sha256: str


def _write(path: Path, contents: bytes) -> _ContractFile:
    path.write_bytes(contents)
    return _ContractFile(
        path=path.name,
        role="tokenizer_config",
        size_bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def test_fast_model_inspection_hashes_runtime_metadata_but_not_weight_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    config_file = _write(model / "config.json", b'{"model_type":"glm4_moe_lite"}\n')
    config_file.role = "config"
    shard_contents = b"weight-payload"
    shard = _write(model / "model-00001-of-00001.safetensors", shard_contents)
    shard.role = "weight_shard"
    tokenizer = _write(model / "tokenizer_config.json", b'{"chat_template":"x"}\n')
    index_contents = json.dumps(
        {
            "metadata": {"total_size": len(shard_contents)},
            "weight_map": {"model.weight": shard.path},
        },
        separators=(",", ":"),
    ).encode()
    index_file = _write(model / "model.safetensors.index.json", index_contents)
    index_file.role = "safetensors_index"
    contract = SimpleNamespace(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        ktransformers_method="BF16",
        weight_map_entries=1,
        files=(config_file, index_file, tokenizer, shard),
    )
    loaded = SimpleNamespace(
        path=str(tmp_path / "contract.json"),
        receipt_sha256="1" * 64,
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        contract=contract,
    )
    monkeypatch.setattr(native, "require_immutable_model_snapshot", lambda _path: None)
    monkeypatch.setattr(
        native,
        "load_sglang_kt_model_contract",
        lambda *_args, **_kwargs: loaded,
    )

    evidence = native.inspect_fast_immutable_model(
        model.resolve(), (tmp_path / "contract.json").resolve()
    )

    assert evidence.weight_payload_rehashed is False
    assert evidence.shard_count == 1
    assert evidence.physical_weight_bytes == len(shard_contents)
    assert evidence.config_sha256 == config_file.sha256
    assert evidence.index_sha256 == index_file.sha256


def test_fast_model_inspection_rejects_changed_shard_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    config_file = _write(model / "config.json", b"{}")
    config_file.role = "config"
    shard = _write(model / "model.safetensors", b"changed")
    shard.role = "weight_shard"
    shard.size_bytes += 1
    index_contents = (
        b'{"metadata":{"total_size":7},"weight_map":{"x":"model.safetensors"}}'
    )
    index_file = _write(model / "model.safetensors.index.json", index_contents)
    index_file.role = "safetensors_index"
    contract = SimpleNamespace(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        ktransformers_method="BF16",
        weight_map_entries=1,
        files=(config_file, index_file, shard),
    )
    monkeypatch.setattr(native, "require_immutable_model_snapshot", lambda _path: None)
    monkeypatch.setattr(
        native,
        "load_sglang_kt_model_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            path=str(tmp_path / "contract.json"),
            receipt_sha256="1" * 64,
            contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
            contract=contract,
        ),
    )

    with pytest.raises(native.NativeTpEpBenchmarkError, match="metadata changed"):
        native.inspect_fast_immutable_model(
            model.resolve(), (tmp_path / "contract.json").resolve()
        )


def test_port_check_uses_exclusive_wildcard_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binds: list[tuple[str, int]] = []
    closes: list[bool] = []

    class Candidate:
        def bind(self, endpoint: tuple[str, int]) -> None:
            binds.append(endpoint)
            if endpoint[1] == 4000:
                raise OSError("occupied")

        def close(self) -> None:
            closes.append(True)

    monkeypatch.setattr(native.socket, "socket", lambda *_arguments: Candidate())
    with pytest.raises(native.NativeTpEpBenchmarkError, match="unavailable"):
        native._check_ports_clear("dwagon", "0.0.0.0", (4000,))

    evidence = native._check_ports_clear("dwagon", "0.0.0.0", (4001,))
    assert evidence.bind_ip == "0.0.0.0"
    assert evidence.ports == (4001,)
    assert binds == [("0.0.0.0", 4000), ("0.0.0.0", 4001)]
    assert closes == [True, True]


def _healthy_hca_deltas() -> dict[str, object]:
    counters = {name: 0 for name in native._HEALTH_COUNTER_NAMES}
    rail = {
        "counter_deltas": counters,
        "received_payload_bytes": 2 * 1024 * 1024,
        "transmitted_payload_bytes": 2 * 1024 * 1024,
    }
    return {
        "dwagon": {"rail-1": rail, "rail-2": rail},
        "fwuff": {"rail-1": rail, "rail-2": rail},
    }


def test_hca_validation_requires_clean_payload_on_both_rails_and_hosts() -> None:
    evidence = native.validate_hca_evidence(_healthy_hca_deltas())
    assert evidence["dual_rail_payload_observed"] is True

    broken = _healthy_hca_deltas()
    cast(dict[str, object], cast(dict[str, object], broken["fwuff"])["rail-2"])[
        "received_payload_bytes"
    ] = 0
    cast(dict[str, object], cast(dict[str, object], broken["fwuff"])["rail-2"])[
        "transmitted_payload_bytes"
    ] = 0
    with pytest.raises(native.NativeTpEpBenchmarkError, match="dual-rail"):
        native.validate_hca_evidence(broken)


def test_nccl_log_validation_requires_dual_rail_ib_and_rejects_socket(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config.result_directory.mkdir()
    admitted = "\n".join(
        (
            "NCCL INFO NCCL_IB_HCA set to =mlx4_0:1,mlx4_0:2",
            "NCCL INFO NET/IB : Using [0]mlx4_0:1/IB [1]mlx4_0:2/IB [RO]",
            "NCCL INFO NET/IB : Made virtual device [2] "
            "name=mlx4_0+mlx4_0 speed=80000 ndevs=2",
            "NCCL INFO NET/IB : GPU Direct RDMA Disabled for HCA 0 'mlx4_0'",
            "NCCL INFO NET/IB : GPU Direct RDMA Disabled for HCA 1 'mlx4_0'",
            "NCCL INFO Channel 00/0 [send] via NET/IB/2",
        )
    )
    for rank in range(3):
        (config.result_directory / f"rank-{rank}.log").write_text(admitted)

    evidence = native.validate_nccl_log_transport(config)
    assert evidence["merged_dual_rail"] is True

    (config.result_directory / "rank-2.log").write_text(
        admitted + "\nNCCL INFO NET/Socket : Using ens17f0\n"
    )
    with pytest.raises(native.NativeTpEpBenchmarkError, match="socket_fallback=True"):
        native.validate_nccl_log_transport(config)


def test_performance_comparability_requires_complete_admitted_evidence() -> None:
    admitted = {
        "failure": None,
        "cleanup_complete": True,
        "sanity": {"passed": True},
        "workloads": [{"kind": "prefill"}, {"kind": "decode"}],
        "hca_validation": {"dual_rail_payload_observed": True},
        "nccl_transport": {
            "transport": "NCCL/IB",
            "socket_fallback_absent": True,
        },
    }

    assert native.performance_evidence_is_comparable(**admitted) is True
    assert (
        native.performance_evidence_is_comparable(
            **{**admitted, "failure": RuntimeError("failed")}
        )
        is False
    )
    assert (
        native.performance_evidence_is_comparable(
            **{
                **admitted,
                "nccl_transport": {
                    "transport": "NCCL/IB",
                    "socket_fallback_absent": False,
                },
            }
        )
        is False
    )


def test_remote_controller_command_is_clean_and_binds_deployed_source(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    command = native._remote_controller_command(config, ("check-ports",))

    assert command[:2] == ("/usr/bin/env", "-i")
    assert "PYTHONSAFEPATH=1" in command
    assert "PYTHONPATH=/deploy/fwuff/exo/src:/deploy/fwuff/exo" in command
    assert command[-2:] == (
        "/deploy/fwuff/exo/scripts/run_sglang_kt_glm47_native_tp_ep_benchmark.py",
        "check-ports",
    )


def test_owned_receipt_parser_is_exact_and_strict() -> None:
    receipt = {
        "rank": 2,
        "host_name": "fwuff",
        "pid": 100,
        "process_group_id": 100,
        "start_time_ticks": 200,
        "owner_token": "owner",
        "ownership_namespace": "30102",
        "remote": True,
        "transport_pid": 300,
        "log_path": "/result/rank-2.log",
    }
    assert native._owned_from_json(json.dumps(receipt)).remote is True

    receipt["remote"] = "true"
    with pytest.raises(native.NativeTpEpBenchmarkError, match="remote flag"):
        native._owned_from_json(json.dumps(receipt))


def test_inspection_binding_rejects_controller_source_drift() -> None:
    plan = make_plan()
    local = SimpleNamespace(
        node_id=NodeId("dwagon"),
        controller_source_sha256="1" * 64,
        native_contract_source_sha256="2" * 64,
    )
    remote = SimpleNamespace(
        node_id=NodeId("fwuff"),
        controller_source_sha256="3" * 64,
        native_contract_source_sha256="2" * 64,
    )
    with pytest.raises(native.NativeTpEpBenchmarkError, match="do not match"):
        native._inspection_bindings(plan, local, remote)  # type: ignore[arg-type]
