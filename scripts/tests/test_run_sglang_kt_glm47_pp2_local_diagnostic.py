from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import stat
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import scripts.run_sglang_kt_glm47_pp2_local_diagnostic as pp2
from scripts import run_sglang_kt_glm47_pp3_diagnostic as pipeline
from scripts.two_host_mlx_nccl_poc import ProcessCleanupReceipt


def make_config(tmp_path: Path) -> pp2.Pp2LocalDiagnosticConfig:
    return pp2.Pp2LocalDiagnosticConfig(
        run_id="glm47-pp2-local-test",
        result_directory=tmp_path / "result",
        dwagon_runtime_python="/runtime/dwagon/bin/python",
        dwagon_runtime_install_receipt=tmp_path / "runtime/install-receipt.json",
        dwagon_runtime_install_receipt_sha256="0" * 64,
        dwagon_model_path=pipeline.DEFAULT_DWAGON_MODEL_PATH,
        local_source_directory="/source/dwagon",
        dwagon_ip="192.168.40.24",
        dwagon_socket_interface="ens13f0np0",
        distributed_port=62500,
        stage_ports=(62510, 62511),
        pipeline_layer_partition=(24, 23),
        resident_gpu_experts=40,
        readiness_timeout_seconds=60.0,
        request_timeout_seconds=900.0,
        cleanup_timeout_seconds=30.0,
        warmup_count=2,
        sample_count=3,
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_digest(contents: bytes) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def create_installed_distribution(
    site_packages: Path,
    distribution: str,
    version: str,
) -> pp2.runtime_install.InstalledDistribution:
    stem = distribution.replace("-", "_")
    module_path = site_packages / f"{stem}.py"
    module_contents = f'IDENTITY = "{distribution}"\n'.encode()
    module_path.write_bytes(module_contents)
    metadata_directory = site_packages / f"{stem}-{version}.dist-info"
    metadata_directory.mkdir()
    metadata_path = metadata_directory / "METADATA"
    metadata_contents = (
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n"
    ).encode()
    metadata_path.write_bytes(metadata_contents)
    record_path = metadata_directory / "RECORD"
    record_path.write_text(
        "\n".join(
            (
                f"{module_path.relative_to(site_packages)},"
                f"sha256={record_digest(module_contents)},{len(module_contents)}",
                f"{metadata_path.relative_to(site_packages)},"
                f"sha256={record_digest(metadata_contents)},{len(metadata_contents)}",
                f"{record_path.relative_to(site_packages)},,",
            )
        )
        + "\n"
    )
    return pp2.runtime_install.InstalledDistribution(
        distribution=distribution,
        version=version,
        metadata_path=str(metadata_path),
        record_sha256=file_sha256(record_path),
    )


def create_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    pp2.Pp2LocalDiagnosticConfig,
    Path,
    Path,
    pp2.runtime_install.RuntimeInstallPlan,
]:
    install_id = "1" * 64
    build_id = "2" * 64
    runtime_root = tmp_path / "runtime"
    install_root = runtime_root / "overlay" / "dwagon" / install_id
    site_packages = install_root / "venv/lib/python3.12/site-packages"
    runtime_python = install_root / "venv/bin/python"
    site_packages.mkdir(parents=True)
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    resolved_python = Path(sys.executable).resolve(strict=True)
    runtime_python.symlink_to(resolved_python)

    base_root = runtime_root / "base"
    base_site_packages = base_root / "site-packages"
    base_site_packages.mkdir(parents=True)
    pth_path = site_packages / pp2.runtime_install.BASE_RUNTIME_PTH_NAME
    pth_path.write_text(f"{base_site_packages}\n")
    installed = tuple(
        create_installed_distribution(site_packages, distribution, "0.6.3.post1")
        for distribution in ("kt-kernel", "ktransformers", "sglang-kt")
    )

    build_root = runtime_root / "build" / "dwagon" / build_id
    wheel_root = build_root / "wheels"
    wheel_root.mkdir(parents=True)
    build_receipt = build_root / "build-receipt.json"
    write_json(build_receipt, {"schema_version": 2, "status": "fixture"})
    wheels: list[pp2.runtime_install.RuntimeWheel] = []
    for distribution in ("kt-kernel", "ktransformers", "sglang-kt"):
        wheel_path = wheel_root / f"{distribution}.whl"
        wheel_path.write_bytes(f"wheel:{distribution}\n".encode())
        wheels.append(
            pp2.runtime_install.RuntimeWheel(
                distribution=distribution,
                version="0.6.3.post1",
                filename=wheel_path.name,
                path=str(wheel_path),
                size_bytes=wheel_path.stat().st_size,
                sha256=file_sha256(wheel_path),
            )
        )
    build = pp2.runtime_install.RuntimeBuildObservation(
        receipt_path=str(build_receipt),
        receipt_sha256=file_sha256(build_receipt),
        build_id=build_id,
        builder_sha256="3" * 64,
        host_profile="dwagon",
        ktransformers_revision=pp2.GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        sglang_revision=pp2.GLM_4_7_FLASH_SGLANG_REVISION,
        package_version="0.6.3.post1",
        wheels=tuple(wheels),
    )
    base_runtime = pp2.runtime_install.BaseRuntimeObservation(
        python_path=str(resolved_python),
        resolved_python_path=str(resolved_python),
        python_sha256=file_sha256(resolved_python),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.0",
        python_implementation="CPython",
        python_soabi="cpython-test",
        prefix=str(base_root),
        base_prefix=str(resolved_python.parent),
        site_packages=str(base_site_packages),
        pip_version="fixture",
        pip_freeze=("fixture==1",),
        pip_freeze_sha256=hashlib.sha256(b"fixture==1\n").hexdigest(),
    )
    layout = pp2.runtime_install.RuntimeInstallLayout(
        output_root=str(runtime_root / "overlay"),
        install_root=str(install_root),
        venv=str(install_root / "venv"),
        python=str(runtime_python),
        site_packages=str(site_packages),
        base_runtime_pth=str(pth_path),
        base_runtime_pth_sha256=file_sha256(pth_path),
        logs=str(install_root / "logs"),
        receipt=str(install_root / "install-receipt.json"),
    )
    plan = pp2.runtime_install.RuntimeInstallPlan(
        install_id=install_id,
        installer_sha256="4" * 64,
        build=build,
        base_runtime=base_runtime,
        layout=layout,
        environment=(("PYTHONDONTWRITEBYTECODE", "1"),),
        commands=((str(runtime_python), "-I", "-c", "pass"),),
    )
    install_receipt = Path(layout.receipt)
    receipt_payload = {
        "schema_version": 1,
        "status": "install_complete",
        **asdict(plan),
        "installed_distributions": [asdict(value) for value in installed],
        "completed_at_utc": "2026-07-20T08:32:29+00:00",
    }
    write_json(install_receipt, receipt_payload)

    def plan_runtime_install(
        selected_build_receipt: Path,
        selected_base_python: Path,
        selected_base_site_packages: Path,
        selected_output_root: Path,
    ) -> pp2.runtime_install.RuntimeInstallPlan:
        assert selected_build_receipt == build_receipt
        assert selected_base_python == resolved_python
        assert selected_base_site_packages == base_site_packages
        assert selected_output_root == runtime_root / "overlay"
        return plan

    monkeypatch.setattr(
        pp2.runtime_install,
        "plan_runtime_install",
        plan_runtime_install,
    )
    monkeypatch.setattr(
        pp2,
        "_verify_pinned_model_contract",
        lambda _path: {"path": pipeline.DEFAULT_DWAGON_MODEL_PATH},
    )
    config = replace(
        make_config(tmp_path),
        dwagon_runtime_python=str(runtime_python),
        dwagon_runtime_install_receipt=install_receipt,
        dwagon_runtime_install_receipt_sha256=file_sha256(install_receipt),
    )
    return config, install_receipt, build_receipt, plan


def fake_running_stage(
    spec: pp2.SglangKtProcessLaunchSpec,
    config: pp2.Pp2LocalDiagnosticConfig,
    owner_token: str,
) -> SimpleNamespace:
    rank = spec.pipeline_rank
    log_path = config.result_directory / f"rank-{rank}.log"
    log_path.write_text(f"rank {rank}\n")
    return SimpleNamespace(
        owned=pipeline.OwnedStageProcess(
            rank=rank,
            host_name="dwagon",
            pid=1000 + rank,
            process_group_id=1000 + rank,
            start_time_ticks=2000 + rank,
            owner_token=owner_token,
            ownership_namespace=str(62510 + rank),
            remote=False,
            transport_pid=1000 + rank,
            log_path=str(log_path),
        )
    )


def test_builds_exact_two_stage_numa_matched_local_plan(tmp_path: Path) -> None:
    specs = pp2.build_pp2_local_process_specs(make_config(tmp_path))

    assert tuple(spec.pipeline_rank for spec in specs) == (0, 1)
    assert tuple(spec.node_id for spec in specs) == (pipeline.DWAGON_NODE_ID,) * 2
    assert tuple(spec.cpu_cores for spec in specs) == (
        tuple(range(56)),
        tuple(range(56, 112)),
    )
    assert tuple(spec.memory_nodes for spec in specs) == ((0,), (1,))
    assert tuple(spec.gpu_uuid for spec in specs) == (
        pipeline.DWAGON_STAGE_ZERO_GPU,
        pipeline.DWAGON_STAGE_ONE_GPU,
    )
    assert tuple((spec.start_layer, spec.end_layer) for spec in specs) == (
        (0, 24),
        (24, 47),
    )
    assert tuple(spec.stage.resident_gpu_experts for spec in specs) == (40, 40)
    assert all(not spec.hca_devices for spec in specs)
    for rank, spec in enumerate(specs):
        assert argument_value(spec.arguments, "--pp-size") == "2"
        assert argument_value(spec.arguments, "--tp-size") == "1"
        assert argument_value(spec.arguments, "--nnodes") == "2"
        assert argument_value(spec.arguments, "--node-rank") == str(rank)
        assert argument_value(spec.arguments, "--kt-cpuinfer") == "56"
        assert argument_value(spec.arguments, "--kt-num-gpu-experts") == "40"
        assert dict(spec.environment)["SGLANG_PP_LAYER_PARTITION"] == "24,23"


def test_local_environment_does_not_force_infiniband_or_disable_p2p(
    tmp_path: Path,
) -> None:
    specs = pp2.build_pp2_local_process_specs(make_config(tmp_path))
    hostile_parent = {
        "HOME": "/root",
        "PATH": "/usr/bin:/bin",
        "NCCL_NET": "IB",
        "NCCL_IB_HCA": "=mlx4_0:1,mlx4_0:2",
        "NCCL_IB_MERGE_NICS": "1",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
    }

    for spec in specs:
        environment = pipeline.build_stage_environment(
            spec,
            "owner-token",
            "ens13f0np0",
            hostile_parent,
        )

        assert environment["NCCL_DEBUG"] == "INFO"
        assert environment["NCCL_SOCKET_IFNAME"] == "ens13f0np0"
        assert environment["GLOO_SOCKET_IFNAME"] == "ens13f0np0"
        for forbidden in (
            "NCCL_NET",
            "NCCL_IB_HCA",
            "NCCL_IB_MERGE_NICS",
            "NCCL_NET_GDR_LEVEL",
            "NCCL_P2P_DISABLE",
            "NCCL_SHM_DISABLE",
        ):
            assert forbidden not in environment


def test_runtime_contract_accepts_exact_content_addressed_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, install_receipt, build_receipt, plan = create_runtime_contract(
        tmp_path, monkeypatch
    )

    observed = pp2._verify_runtime_and_model_contract(config)

    assert cast(dict[str, object], observed["install_receipt"])["sha256"] == (
        file_sha256(install_receipt)
    )
    assert cast(dict[str, object], observed["build_receipt"])["sha256"] == (
        file_sha256(build_receipt)
    )
    assert observed["install_id"] == plan.install_id
    assert observed["sglang_revision"] == pp2.GLM_4_7_FLASH_SGLANG_REVISION
    assert observed["ktransformers_revision"] == (
        pp2.GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    )
    assert cast(dict[str, object], observed["runtime_python"])["path"] == (
        config.dwagon_runtime_python
    )
    assert set(
        cast(dict[str, object], observed["installed_distribution_file_counts"])
    ) == {"kt-kernel", "ktransformers", "sglang-kt"}


def test_runtime_contract_rejects_stale_revision_with_valid_receipt_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, install_receipt, _build_receipt, _plan = create_runtime_contract(
        tmp_path, monkeypatch
    )
    payload = cast(dict[str, object], json.loads(install_receipt.read_text()))
    build = cast(dict[str, object], payload["build"])
    build["sglang_revision"] = "5" * 40
    write_json(install_receipt, payload)
    config = replace(
        config,
        dwagon_runtime_install_receipt_sha256=file_sha256(install_receipt),
    )

    with pytest.raises(
        pp2.Pp2LocalDiagnosticError,
        match="revisions differ",
    ):
        pp2._verify_runtime_and_model_contract(config)


def test_runtime_contract_rejects_tampered_receipt_and_build_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, install_receipt, build_receipt, _plan = create_runtime_contract(
        tmp_path, monkeypatch
    )
    install_receipt.write_bytes(install_receipt.read_bytes() + b" ")
    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="required binding"):
        pp2._verify_runtime_and_model_contract(config)

    install_receipt.write_bytes(install_receipt.read_bytes()[:-1])
    build_receipt.write_bytes(build_receipt.read_bytes() + b" ")
    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="build receipt content"):
        pp2._verify_runtime_and_model_contract(config)


def test_runtime_contract_rejects_python_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _install_receipt, _build_receipt, _plan = create_runtime_contract(
        tmp_path, monkeypatch
    )
    substituted = replace(config, dwagon_runtime_python=str(Path(sys.executable)))

    with pytest.raises(
        pp2.Pp2LocalDiagnosticError,
        match="selected Python/install paths",
    ):
        pp2._verify_runtime_and_model_contract(substituted)


def test_bad_runtime_binding_fails_before_result_creation_or_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _install_receipt, _build_receipt, _plan = create_runtime_contract(
        tmp_path, monkeypatch
    )
    config = replace(
        config,
        dwagon_runtime_install_receipt_sha256="f" * 64,
    )
    monkeypatch.setattr(
        pipeline,
        "start_local_stage",
        lambda *_args: pytest.fail("runtime admission must precede process launch"),
    )

    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="required binding"):
        pp2.run_diagnostic(config)

    assert not config.result_directory.exists()


def test_model_path_substitution_fails_before_result_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(make_config(tmp_path), dwagon_model_path="/substituted/model")
    monkeypatch.setattr(
        pipeline,
        "start_local_stage",
        lambda *_args: pytest.fail("model admission must precede process launch"),
    )

    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="pinned.*contract path"):
        pp2.run_diagnostic(config)

    assert not config.result_directory.exists()


def test_cli_defaults_to_safe_e40_and_supports_bounded_ab_options(
    tmp_path: Path,
) -> None:
    parser = pp2._parser()
    required = (
        "--run-id",
        "local-pp2",
        "--result-directory",
        str(tmp_path / "result"),
        "--dwagon-runtime-python",
        "/runtime/dwagon/bin/python",
        "--dwagon-runtime-install-receipt",
        str(tmp_path / "runtime/install-receipt.json"),
        "--dwagon-runtime-install-receipt-sha256",
        "0" * 64,
    )

    default_config = pp2._config_from_arguments(parser.parse_args(required))
    tuned_config = pp2._config_from_arguments(
        parser.parse_args(
            (
                *required,
                "--pipeline-layer-partition",
                "23,24",
                "--resident-gpu-experts",
                "44",
            )
        )
    )

    assert default_config.pipeline_layer_partition == (24, 23)
    assert default_config.resident_gpu_experts == 40
    assert tuned_config.pipeline_layer_partition == (23, 24)
    assert tuned_config.resident_gpu_experts == 44
    assert "fwuff" not in parser.format_help().lower()
    assert "hca" not in parser.format_help().lower()


def test_cli_rejects_e45_and_unapproved_partition(tmp_path: Path) -> None:
    parser = pp2._parser()
    required = (
        "--run-id",
        "local-pp2",
        "--result-directory",
        str(tmp_path / "result"),
        "--dwagon-runtime-python",
        "/runtime/dwagon/bin/python",
        "--dwagon-runtime-install-receipt",
        str(tmp_path / "runtime/install-receipt.json"),
        "--dwagon-runtime-install-receipt-sha256",
        "0" * 64,
    )

    arguments = parser.parse_args((*required, "--resident-gpu-experts", "45"))
    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="between 1 and 44"):
        pp2._config_from_arguments(arguments)
    with pytest.raises(SystemExit):
        parser.parse_args((*required, "--pipeline-layer-partition", "22,25"))


def test_receipt_is_local_pp2_and_runs_canonical_semantic_workloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    calls: list[object] = []
    journal_snapshots: list[dict[str, object]] = []

    class Evidence:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return self.payload

    class Workload(Evidence):
        def __init__(self, kind: str) -> None:
            input_tokens, output_tokens = (
                (1024, 32) if kind == "prefill" else (128, 128)
            )
            super().__init__({"kind": kind})
            self.request = SimpleNamespace(
                kind=kind,
                input_token_count=input_tokens,
                max_new_tokens=output_tokens,
            )
            self.samples = tuple(
                SimpleNamespace(
                    client_observed_decode_tokens_per_second=6.0 + ordinal,
                    client_observed_ttft_seconds=1.0,
                    completion_tokens=output_tokens,
                    total_client_seconds=2.0,
                )
                for ordinal in range(3)
            )

    class Client:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            assert base_url == "http://192.168.40.24:62510"
            assert timeout_seconds == 900.0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def server_info(self) -> Evidence:
            return Evidence({"pp_size": 2, "tp_size": 1})

    def start_local_stage(
        spec: pp2.SglangKtProcessLaunchSpec,
        stage_config: pp2.Pp2LocalDiagnosticConfig,
        owner_token: str,
    ) -> SimpleNamespace:
        rank = spec.pipeline_rank
        if rank == 1:
            journal_path = pp2._ownership_journal_path(stage_config)
            assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
            journal_snapshots.append(
                cast(dict[str, object], json.loads(journal_path.read_text()))
            )
        log_path = stage_config.result_directory / f"rank-{rank}.log"
        log_path.write_text(f"rank {rank}\n")
        owned = pipeline.OwnedStageProcess(
            rank=rank,
            host_name="dwagon",
            pid=1000 + rank,
            process_group_id=1000 + rank,
            start_time_ticks=2000 + rank,
            owner_token=owner_token,
            ownership_namespace=str(62510 + rank),
            remote=False,
            transport_pid=1000 + rank,
            log_path=str(log_path),
        )
        return SimpleNamespace(owned=owned)

    def stop_local_stage(
        running: SimpleNamespace,
        cleanup_timeout_seconds: float,
    ) -> ProcessCleanupReceipt:
        assert cleanup_timeout_seconds == 30.0
        journal_path = pp2._ownership_journal_path(config)
        assert journal_path.is_file()
        if running.owned.rank == 1:
            journal_snapshots.append(
                cast(dict[str, object], json.loads(journal_path.read_text()))
            )
        calls.append(("cleanup", running.owned.rank))
        return ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    def run_sanity(_client: Client, model_path: str) -> Evidence:
        assert model_path == pipeline.DEFAULT_DWAGON_MODEL_PATH
        calls.append("sanity")
        return Evidence({"output_text": "EXO_SANITY_OK"})

    def run_workload(
        _client: Client,
        kind: str,
        *,
        warmup_count: int,
        sample_count: int,
    ) -> Workload:
        assert warmup_count == 2
        assert sample_count == 3
        calls.append(kind)
        return Workload(kind)

    monkeypatch.setattr(pipeline, "start_local_stage", start_local_stage)
    monkeypatch.setattr(pipeline, "stop_local_stage", stop_local_stage)
    monkeypatch.setattr(pipeline, "wait_for_all_stages", lambda *_args: ({}, {}))
    monkeypatch.setattr(pipeline, "all_stages_alive", lambda _running: True)
    monkeypatch.setattr(pp2, "Glm47NativeServingClient", Client)
    monkeypatch.setattr(pp2, "run_glm47_serving_sanity", run_sanity)
    monkeypatch.setattr(pp2, "prepare_glm47_serving_workload", lambda kind: kind)
    monkeypatch.setattr(pp2, "run_glm47_serving_workload", run_workload)
    monkeypatch.setattr(
        pp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )

    payload = pp2.run_diagnostic(config)

    assert payload["kind"] == "glm47_flash_pp2_local_engineering_diagnostic"
    assert payload["status"] == "passed"
    assert payload["cleanup_complete"] is True
    assert payload["configuration"] == {
        "host_name": "dwagon",
        "scope": "single_host",
        "pipeline_parallel_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_layer_partition": [24, 23],
        "resident_gpu_experts_per_stage": 40,
        "nccl_transport_policy": "automatic_local_p2p_nvlink_allowed",
    }
    assert cast(dict[str, object], payload["topology"])["scope"] == "dwagon_local"
    assert "hca_counters" not in payload
    assert "fwuff" not in json.dumps(payload).lower()
    assert calls == ["sanity", "prefill", "decode", ("cleanup", 1), ("cleanup", 0)]
    assert [snapshot["started_rank_count"] for snapshot in journal_snapshots] == [
        1,
        2,
    ]
    first_process = cast(list[dict[str, object]], journal_snapshots[0]["processes"])[0]
    assert first_process["owner_token"]
    assert not pp2._ownership_journal_path(config).exists()
    assert payload["planned_rank_count"] == 2
    assert payload["started_rank_count"] == 2
    assert payload["all_planned_stages_started"] is True
    assert payload["managed_signal"] is None
    assert payload["runtime_contract"] == {"install_id": "verified"}
    assert payload["ownership_journal"] == {
        "path": str(pp2._ownership_journal_path(config)),
        "created": True,
        "cleared_after_verified_cleanup": True,
        "retained": False,
    }
    receipt_path = config.result_directory / "pp2-local-diagnostic-result.json"
    assert (
        json.loads(receipt_path.read_text())["receipt_content_sha256"]
        == payload["receipt_content_sha256"]
    )


def test_rank_one_startup_failure_cleans_started_rank_without_false_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    cleaned: list[int] = []
    monkeypatch.setattr(
        pp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )

    def start(
        spec: pp2.SglangKtProcessLaunchSpec,
        stage_config: pp2.Pp2LocalDiagnosticConfig,
        owner_token: str,
    ) -> SimpleNamespace:
        if spec.pipeline_rank == 1:
            raise RuntimeError("rank one startup failed")
        return fake_running_stage(spec, stage_config, owner_token)

    def stop(
        running: SimpleNamespace,
        _timeout: float,
    ) -> ProcessCleanupReceipt:
        cleaned.append(running.owned.rank)
        return ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    monkeypatch.setattr(pipeline, "start_local_stage", start)
    monkeypatch.setattr(pipeline, "stop_local_stage", stop)

    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="startup failed"):
        pp2.run_diagnostic(config)

    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "pp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert cleaned == [0]
    assert receipt["status"] == "failed"
    assert receipt["planned_rank_count"] == 2
    assert receipt["started_rank_count"] == 1
    assert receipt["all_planned_stages_started"] is False
    assert receipt["cleanup_complete"] is True
    assert not pp2._ownership_journal_path(config).exists()


def test_managed_signal_during_start_is_raised_after_journal_and_cleans_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    cleaned: list[int] = []
    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(
        pp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )

    def start(
        spec: pp2.SglangKtProcessLaunchSpec,
        stage_config: pp2.Pp2LocalDiagnosticConfig,
        owner_token: str,
    ) -> SimpleNamespace:
        running = fake_running_stage(spec, stage_config, owner_token)
        os.kill(os.getpid(), signal.SIGTERM)
        return running

    def stop(
        running: SimpleNamespace,
        _timeout: float,
    ) -> ProcessCleanupReceipt:
        journal = pp2._ownership_journal_path(config)
        assert journal.is_file()
        assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        cleaned.append(running.owned.rank)
        return ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    monkeypatch.setattr(pipeline, "start_local_stage", start)
    monkeypatch.setattr(pipeline, "stop_local_stage", stop)

    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="managed signal"):
        pp2.run_diagnostic(config)

    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "pp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert cleaned == [0]
    assert receipt["managed_signal"] == signal.SIGTERM
    assert receipt["started_rank_count"] == 1
    assert receipt["cleanup_complete"] is True
    assert not pp2._ownership_journal_path(config).exists()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_incomplete_cleanup_retains_recovery_journal_with_raw_owner_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        pp2,
        "_verify_runtime_and_model_contract",
        lambda _config: {"install_id": "verified"},
    )
    monkeypatch.setattr(pipeline, "start_local_stage", fake_running_stage)
    monkeypatch.setattr(
        pipeline,
        "wait_for_all_stages",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )

    def stop(
        running: SimpleNamespace,
        _timeout: float,
    ) -> ProcessCleanupReceipt:
        complete = running.owned.rank == 0
        return ProcessCleanupReceipt(
            host_name="dwagon",
            ownership_verified=complete,
            terminated=complete,
            forced=False,
        )

    monkeypatch.setattr(pipeline, "stop_local_stage", stop)

    with pytest.raises(pp2.Pp2LocalDiagnosticError, match="readiness failed"):
        pp2.run_diagnostic(config)

    journal_path = pp2._ownership_journal_path(config)
    assert journal_path.is_file()
    assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
    journal = cast(dict[str, object], json.loads(journal_path.read_text()))
    processes = cast(list[dict[str, object]], journal["processes"])
    assert journal["started_rank_count"] == 2
    assert len(processes) == 2
    assert all(process["owner_token"] for process in processes)
    receipt = cast(
        dict[str, object],
        json.loads(
            (config.result_directory / "pp2-local-diagnostic-result.json").read_text()
        ),
    )
    assert receipt["cleanup_complete"] is False
    assert cast(dict[str, object], receipt["ownership_journal"])["retained"] is True
