from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from scripts import run_sglang_kt_glm47_moe_tuning as harness
from scripts import tune_sglang_kt_glm47_moe as tuner

COMMIT = "b" * 40
SGLANG_REVISION = "c" * 40
KTRANSFORMERS_REVISION = "d" * 40
GPU_UUID = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
GPU_BDF = "00000000:01:00.0"
GPU_NAME = "NVIDIA GeForce RTX 3090"
GID = "fe80:0000:0000:0000:0210:e000:0166:3a19"
INSTALL_ID = "1" * 64
BUILD_ID = "2" * 64
SHA = "a" * 64


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object, *, pretty: bool = True) -> bytes:
    return harness._canonical_json_bytes(value, pretty=pretty)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def executable_binding(path: Path) -> dict[str, object]:
    current = path
    chain: list[str] = []
    for _ in range(harness.MAXIMUM_SYMLINK_HOPS + 1):
        if not current.is_symlink():
            return {
                "path": str(path),
                "sha256": file_sha256(current),
                "symlink_chain": chain,
            }
        target = os.readlink(current)
        chain.append(target)
        target_path = Path(target)
        current = target_path if target_path.is_absolute() else current.parent / target
    raise AssertionError("test executable symlink chain is unexpectedly long")


def record_hash(contents: bytes) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
    )


def write_distribution(
    site_packages: Path, distribution: str, import_name: str
) -> dict[str, object]:
    version = "0.6.3.post1"
    package = site_packages / import_name
    package.mkdir(parents=True, exist_ok=True)
    payload = f"IDENTITY = {distribution!r}\n".encode()
    (package / "__init__.py").write_bytes(payload)
    info = site_packages / f"{import_name}-{version}.dist-info"
    info.mkdir()
    metadata = f"Name: {distribution}\nVersion: {version}\n".encode()
    (info / "METADATA").write_bytes(metadata)
    payload_name = f"{import_name}/__init__.py"
    metadata_name = f"{info.name}/METADATA"
    record_name = f"{info.name}/RECORD"
    record = (
        f"{payload_name},sha256={record_hash(payload)},{len(payload)}\r\n"
        f"{metadata_name},sha256={record_hash(metadata)},{len(metadata)}\r\n"
        f"{record_name},,\r\n"
    ).encode()
    (info / "RECORD").write_bytes(record)
    return {
        "distribution": distribution,
        "metadata_path": str(info / "METADATA"),
        "record_sha256": hashlib.sha256(record).hexdigest(),
        "version": version,
    }


def create_runtime_receipts(tmp_path: Path) -> dict[str, object]:
    install_root = tmp_path / "runtime" / INSTALL_ID
    runtime_bin = install_root / "venv" / "bin"
    site_packages = install_root / "venv" / "lib" / "python3.12" / "site-packages"
    runtime_bin.mkdir(parents=True, exist_ok=True)
    site_packages.mkdir(parents=True, exist_ok=True)
    runtime_python = runtime_bin / "python"
    python_intermediate = runtime_bin / "python3.12"
    if not runtime_python.exists() and not runtime_python.is_symlink():
        runtime_python.symlink_to("python3.12")
    if not python_intermediate.exists() and not python_intermediate.is_symlink():
        python_intermediate.symlink_to(Path(sys.executable).resolve())

    installed_distributions = [
        write_distribution(site_packages, "kt-kernel", "kt_kernel"),
        write_distribution(site_packages, "ktransformers", "ktransformers"),
        write_distribution(site_packages, "sglang-kt", "sglang_kt"),
    ]
    build_root = tmp_path / "build" / BUILD_ID
    wheels_root = build_root / "wheels"
    wheels_root.mkdir(parents=True)
    wheel_specs = (
        ("kt-kernel", "kt_kernel.whl"),
        ("ktransformers", "ktransformers.whl"),
        ("sglang-kt", "sglang_kt.whl"),
    )
    wheel_values: list[dict[str, object]] = []
    install_wheels: list[dict[str, object]] = []
    for distribution, filename in wheel_specs:
        path = wheels_root / filename
        path.write_bytes(f"wheel:{distribution}\n".encode())
        digest = file_sha256(path)
        wheel_values.append(
            {
                "distribution": distribution,
                "path": f"wheels/{filename}",
                "sha256": digest,
            }
        )
        install_wheels.append(
            {
                "distribution": distribution,
                "path": str(path),
                "sha256": digest,
            }
        )

    build_receipt = build_root / "build-receipt.json"
    write_json(
        build_receipt,
        {
            "schema_version": 2,
            "status": "wheel_build_complete",
            "build_id": BUILD_ID,
            "source": {
                "sglang_revision": SGLANG_REVISION,
                "ktransformers_revision": KTRANSFORMERS_REVISION,
                "package_version": "0.6.3.post1",
            },
            "layout": {"receipt": str(build_receipt)},
            "runtime_wheels": wheel_values,
        },
    )
    build_digest = file_sha256(build_receipt)
    freeze = ["torch==2.9.1", "triton==3.5.1"]
    freeze_digest = hashlib.sha256(("\n".join(freeze) + "\n").encode()).hexdigest()
    install_receipt = install_root / "install-receipt.json"
    write_json(
        install_receipt,
        {
            "schema_version": 1,
            "status": "install_complete",
            "install_id": INSTALL_ID,
            "layout": {
                "install_root": str(install_root),
                "python": str(runtime_python),
                "receipt": str(install_receipt),
            },
            "base_runtime": {
                "python_sha256": file_sha256(Path(sys.executable).resolve()),
                "pip_freeze": freeze,
                "pip_freeze_sha256": freeze_digest,
            },
            "build": {
                "build_id": BUILD_ID,
                "receipt_path": str(build_receipt),
                "receipt_sha256": build_digest,
                "sglang_revision": SGLANG_REVISION,
                "ktransformers_revision": KTRANSFORMERS_REVISION,
                "package_version": "0.6.3.post1",
                "wheels": install_wheels,
            },
            "installed_distributions": installed_distributions,
        },
    )
    embedded_digest = "3" * 64
    extension_digest = "4" * 64
    kernel_receipt = tmp_path / "runtime" / "kernel-validation.json"
    write_json(
        kernel_receipt,
        {
            "schema_version": 1,
            "status": "passed",
            "failures": [],
            "profiler": "none",
            "capabilities": ["kt_bf16_amx_executed_v1"],
            "config": {
                "build_receipt_path": str(build_receipt),
                "gpu_uuid": GPU_UUID,
            },
            "provenance": {
                "verified": True,
                "build_id": BUILD_ID,
                "receipt_path": str(build_receipt),
                "receipt_sha256": build_digest,
                "sglang_revision": SGLANG_REVISION,
                "ktransformers_revision": KTRANSFORMERS_REVISION,
                "kt_extension_sha256": extension_digest,
                "wheels": install_wheels,
                "embedded_provenance": [
                    {
                        "distribution": "kt-kernel",
                        "sglang_revision": SGLANG_REVISION,
                        "ktransformers_revision": KTRANSFORMERS_REVISION,
                        "sha256": embedded_digest,
                    },
                    {
                        "distribution": "sglang-kt",
                        "sglang_revision": SGLANG_REVISION,
                        "ktransformers_revision": KTRANSFORMERS_REVISION,
                        "sha256": embedded_digest,
                    },
                ],
            },
            "runtime_identity": {
                "torch_module_version": "2.9.1+cu128",
                "torch_cuda_version": "12.8",
            },
            "cuda": {
                "gpu_uuid": GPU_UUID,
                "gpu_name": GPU_NAME,
                "compute_capability": [8, 6],
                "torch_cuda_version": "12.8",
            },
        },
    )
    return {
        "runtime_python": executable_binding(runtime_python),
        "runtime_receipts": {
            "install": {
                "path": str(install_receipt),
                "sha256": file_sha256(install_receipt),
            },
            "build": {"path": str(build_receipt), "sha256": build_digest},
            "kernel_validation": {
                "path": str(kernel_receipt),
                "sha256": file_sha256(kernel_receipt),
            },
        },
    }


def config_payload(tmp_path: Path, *, run_id: str | None = None) -> dict[str, object]:
    selected_run_id = run_id or f"glm47-tune-test-{uuid.uuid4().hex}"
    runtime = create_runtime_receipts(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    numactl = Path(sys.executable).resolve()
    return {
        "schema_version": 3,
        "run_id": selected_run_id,
        "namespace": f"exo-{selected_run_id}",
        "profiler": "none",
        "result_directory": str(tmp_path / "results" / selected_run_id),
        "source_repository": str(Path(harness.__file__).resolve().parents[1]),
        "source_deployment_root": str(tmp_path / "deployments" / selected_run_id),
        **runtime,
        "numactl_executable": {"path": str(numactl), "sha256": file_sha256(numactl)},
        "contextual_model": {
            "model_id": "zai-org/GLM-4.7-Flash",
            "revision": COMMIT,
            "path": str(model_path),
        },
        "host": {
            "hostname": "dwagon",
            "gpu": {
                "uuid": GPU_UUID,
                "pci_address": GPU_BDF,
                "expected_name": GPU_NAME,
            },
            "cpu_cores": [0, 1, 2, 3],
            "memory_nodes": [0],
            "hca_bindings": [{"device": "mlx4_0", "port": 1, "gid": GID}],
        },
        "reserved_ports": [62191],
        "tuning": {
            "resident_experts": 4,
            "warmup_iterations": 3,
            "measurement_iterations": 5,
            "independent_samples": 3,
            "search_profile": "quick",
            "seed": 20260719,
            "expected_sglang_revision": SGLANG_REVISION,
            "expected_ktransformers_revision": KTRANSFORMERS_REVISION,
            "expected_torch_version": "2.9.1+cu128",
            "expected_triton_version": "3.5.1",
        },
        "timeouts": {"tuning_seconds": 120.0, "cleanup_seconds": 0.1},
    }


def make_config(tmp_path: Path) -> harness.TuningRunConfig:
    return harness.TuningRunConfig.model_validate_json(
        json.dumps(config_payload(tmp_path))
    )


@contextmanager
def held_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def test_tuning_defaults_include_large_batch_and_no_mutable_runtime_arguments(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    assert config.tuning.batch_sizes == (1, 8, 32, 128, 512, 1_024, 4_096)
    argv = harness._tuner_argv(config, tmp_path / "output")
    assert "--gpu-routes-per-token" not in argv
    assert not any(argument.startswith("--expected-") for argument in argv)
    assert argv[argv.index("--batch-sizes") + 1].endswith(",4096")


def test_tuning_rejects_too_few_cpu_experts_for_zero_resident_routes(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    tuning = cast(dict[str, object], payload["tuning"])
    tuning["resident_experts"] = 61
    with pytest.raises(ValueError, match="less than or equal to 60"):
        harness.TuningRunConfig.model_validate_json(json.dumps(payload))


def test_runtime_python_accepts_and_rechecks_real_multilink_chain(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    assert len(config.runtime_python.symlink_chain) == 2
    harness._verify_executable(config.runtime_python, "runtime Python")
    intermediate = Path(config.runtime_python.path).parent / "python3.12"
    intermediate.unlink()
    intermediate.symlink_to("/bin/false")
    with pytest.raises(harness.HarnessError, match="symlink target changed"):
        harness._verify_executable(config.runtime_python, "runtime Python")


def test_runtime_contract_detects_installed_file_drift(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    contract = harness._verified_runtime_contract(config)
    assert contract["install_id"] == INSTALL_ID
    install_root = Path(cast(str, contract["install_root"]))
    (
        install_root / "venv/lib/python3.12/site-packages/sglang_kt/__init__.py"
    ).write_text("changed\n", encoding="ascii")
    with pytest.raises(harness.HarnessError, match="installed runtime file"):
        harness._verified_runtime_contract(config)


def test_runtime_contract_binds_base_python_to_configured_executable(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    receipts = cast(dict[str, dict[str, object]], payload["runtime_receipts"])
    install_binding = receipts["install"]
    install_path = Path(cast(str, install_binding["path"]))
    install_receipt = cast(dict[str, object], json.loads(install_path.read_bytes()))
    base_runtime = cast(dict[str, object], install_receipt["base_runtime"])
    base_runtime["python_sha256"] = "f" * 64
    write_json(install_path, install_receipt)
    install_binding["sha256"] = file_sha256(install_path)
    config = harness.TuningRunConfig.model_validate_json(json.dumps(payload))
    with pytest.raises(harness.HarnessError, match="base runtime Python"):
        harness._verified_runtime_contract(config)


def test_contextual_model_binding_cannot_claim_snapshot_verification(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    contextual_model = cast(dict[str, object], payload["contextual_model"])
    contextual_model["verification"] = {
        "path": str(tmp_path / "arbitrary.json"),
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="verification"):
        harness.TuningRunConfig.model_validate_json(json.dumps(payload))


def test_load_config_rejects_oversized_input(tmp_path: Path) -> None:
    path = (tmp_path / "oversized.json").resolve()
    path.write_bytes(b" " * (harness.MAXIMUM_CONFIG_JSON_BYTES + 1))
    with pytest.raises(harness.HarnessError, match="bounded"):
        harness._load_config(path)


def test_prepare_lease_builds_read_only_capsule_and_valid_metadata(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    config_path = (tmp_path / "config.json").resolve()
    write_json(config_path, payload)
    metadata_path = (tmp_path / "metadata.json").resolve()
    preparation = harness.prepare_lease(
        config_path=config_path,
        metadata_output=metadata_path,
        owner="codex:/root",
        purpose="unit-test",
        expected_duration_seconds=300.0,
        cleanup_grace_seconds=30.0,
        heartbeat_seconds=5.0,
        lease_path=(tmp_path / "lease.json").resolve(),
        lock_path=(tmp_path / "lock").resolve(),
        result_root=(tmp_path / "results").resolve(),
    )
    deployment = Path(preparation.source_deployment_root)
    assert deployment.is_dir()
    assert deployment.stat().st_mode & 0o222 == 0
    metadata = cast(dict[str, object], json.loads(metadata_path.read_bytes()))
    contract = cast(dict[str, object], metadata["tuning_contract"])
    assert contract["candidate_only"] is True
    assert contract["runtime_receipts"] == payload["runtime_receipts"]
    assert contract["contextual_model_binding"] == payload["contextual_model"]
    assert contract["contextual_model_snapshot_consumed"] is False
    assert contract["contextual_model_binding_is_verification"] is False
    assert preparation.benchmark_lease_argv[
        -len(cast(list[str], metadata["command"])) :
    ] == tuple(cast(list[str], metadata["command"]))


def test_standard_lock_probe_requires_an_existing_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    with pytest.raises(harness.HarnessError, match="not held"):
        harness._assert_standard_lock_held(lock_path)
    with held_lock(lock_path):
        harness._assert_standard_lock_held(lock_path)


def test_find_conflicting_processes_reports_benchmark_and_storage_work(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "101").mkdir(parents=True)
    (proc_root / "101" / "cmdline").write_bytes(b"/usr/bin/ib_write_bw\0-d\0mlx4_0\0")
    (proc_root / "102").mkdir()
    (proc_root / "102" / "cmdline").write_bytes(b"hf\0download\0repo/model\0")
    conflicts = harness.find_conflicting_processes(
        proc_root, ignored_process_ids=frozenset()
    )
    assert [conflict["pid"] for conflict in conflicts] == [101, 102]


def test_profiler_preflight_records_loaded_idle_modules(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "self").mkdir(parents=True)
    (proc_root / "self" / "cmdline").write_bytes(b"python\0tuner.py\0")
    (proc_root / "modules").write_text(
        "sep5 1 0 - Live 0x0\npax 1 0 - Live 0x0\n",
        encoding="ascii",
    )

    evidence = harness._profiler_preflight(proc_root)

    assert evidence == {
        "mode": "none",
        "prohibited_modules_loaded": ["pax", "sep5"],
        "prohibited_profiler_processes": [],
    }


def test_profiler_preflight_rejects_active_driver_use(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "self").mkdir(parents=True)
    (proc_root / "self" / "cmdline").write_bytes(b"python\0tuner.py\0")
    (proc_root / "modules").write_text("sep5 1 0 - Live 0x0\n", encoding="ascii")
    process_directory = proc_root / "101"
    (process_directory / "fd").mkdir(parents=True)
    (process_directory / "cmdline").write_bytes(b"python\0worker.py\0")
    (process_directory / "fd" / "4").symlink_to("/dev/sep5")

    with pytest.raises(harness.HarnessError, match="profiler use is active"):
        harness._profiler_preflight(proc_root)


@dataclass
class AuthorizationFixture:
    config: harness.TuningRunConfig
    output: Path
    authorization: dict[str, object]
    lease_path: Path
    result_descriptor: int
    output_descriptor: int
    cache_descriptors: tuple[int, ...]
    read_descriptor: int


@contextmanager
def authorization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[AuthorizationFixture]:
    config = make_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    result_descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    output = result_path / harness.TUNING_OUTPUT_DIRECTORY_NAME
    output.mkdir()
    output_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    caches: dict[str, int] = {}
    for name in ("cuda-cache", "triton-cache", "tmp"):
        (result_path / name).mkdir()
        caches[name] = os.open(result_path / name, os.O_RDONLY | os.O_DIRECTORY)
    deployment = Path(config.source_deployment_root)
    tuner = deployment / harness.TUNER_RELATIVE_PATH
    tuner.parent.mkdir(parents=True)
    tuner.write_text("pass\n", encoding="ascii")
    source = deployment / harness.SOURCE_IDENTITY_FILENAME
    source.write_text("{}\n", encoding="ascii")
    tuning_config = deployment / harness.IMMUTABLE_CONFIG_FILENAME
    tuning_config.write_bytes(canonical_json(config.model_dump(mode="json")))
    preflight = result_path / harness.PREFLIGHT_FILENAME
    preflight.write_text("{}\n", encoding="ascii")
    telemetry = result_path / harness.TELEMETRY_BEFORE_FILENAME
    telemetry.write_text("{}\n", encoding="ascii")
    parent_pid = 90210
    parent_start = 123456
    owner_token = "e" * 64
    process_argv = [config.runtime_python.path, str(tuner), "--output-dir", str(output)]
    status = os.fstat(result_descriptor)
    output_status = os.fstat(output_descriptor)
    runtime_contract = harness._verified_runtime_contract(config)
    authorization: dict[str, object] = {
        "schema_version": 3,
        "authorization_kind": "active-benchmark-lease-inherited-pipe-v3",
        "lease_id": "lease-1",
        "run_id": config.run_id,
        "namespace": config.namespace,
        "lock_path": str(tmp_path / "lock"),
        "lease_path": str(tmp_path / "lease.json"),
        "result_directory": {
            "path": str(result_path),
            "device": status.st_dev,
            "inode": status.st_ino,
        },
        "harness_process": {"pid": parent_pid, "start_time_ticks": parent_start},
        "process_ownership": {
            "mode": "inherit-lease-child-process-group",
            "lease_child_pid": parent_pid,
            "process_group_id": parent_pid,
            "outer_wrapper_cleanup": "killpg",
        },
        "owner_token": owner_token,
        "launcher_argv": ["/usr/bin/numactl", *process_argv],
        "tuner_process_argv": process_argv,
        "tuning_output_directory": str(output),
        "output_directory_descriptor": output_descriptor,
        "output_directory_identity": {
            "device": output_status.st_dev,
            "inode": output_status.st_ino,
        },
        "cache_directories": {
            name: {
                "descriptor": descriptor,
                "identity": harness._directory_identity(descriptor),
            }
            for name, descriptor in caches.items()
        },
        "child_environment": {"inheritance_policy": "allowlist-v1"},
        "runtime_contract": runtime_contract,
        "receipt_bindings": {
            "runtime_python": config.runtime_python.model_dump(mode="json"),
            "runtime_install": config.runtime_receipts.install.model_dump(mode="json"),
            "runtime_build": config.runtime_receipts.build.model_dump(mode="json"),
            "kernel_validation": config.runtime_receipts.kernel_validation.model_dump(
                mode="json"
            ),
            "tuner_script": {"path": str(tuner), "sha256": file_sha256(tuner)},
            "source_identity": {"path": str(source), "sha256": file_sha256(source)},
            "tuning_config": {
                "path": str(tuning_config),
                "sha256": file_sha256(tuning_config),
            },
            "preflight": {"path": str(preflight), "sha256": file_sha256(preflight)},
            "telemetry_before": {
                "path": str(telemetry),
                "sha256": file_sha256(telemetry),
            },
        },
        "experiment_context": {
            "contextual_model": config.contextual_model.model_dump(mode="json"),
            "model_snapshot_consumed": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "binding_establishes_model_verification": False,
        },
        "created_at": "2026-07-19T12:00:00+00:00",
    }
    authorization_bytes = canonical_json(authorization, pretty=False)
    (result_path / harness.AUTHORIZATION_FILENAME).write_bytes(
        canonical_json(authorization)
    )
    (result_path / harness.AUTHORIZATION_FILENAME).chmod(0o444)
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, authorization_bytes)
    os.close(write_descriptor)
    lease_path = Path(cast(str, authorization["lease_path"]))
    write_json(
        lease_path,
        {
            "lease_id": "lease-1",
            "run_id": config.run_id,
            "exo_namespace": config.namespace,
            "result_directory": str(result_path),
            "child_pid": parent_pid,
            "child_cleanup_confirmation_required": True,
        },
    )
    monkeypatch.setenv("EXO_TESTS", "1")
    monkeypatch.setenv(harness.TUNER_AUTHORIZATION_FD_ENVIRONMENT, str(read_descriptor))
    monkeypatch.setenv(
        harness.TUNER_RESULT_DIRECTORY_FD_ENVIRONMENT, str(result_descriptor)
    )
    monkeypatch.setenv(
        harness.TUNER_OUTPUT_DIRECTORY_FD_ENVIRONMENT, str(output_descriptor)
    )
    monkeypatch.setenv(harness.TUNER_OWNER_TOKEN_ENVIRONMENT, owner_token)
    monkeypatch.setattr(harness.os, "getppid", lambda: parent_pid)
    monkeypatch.setattr(harness.os, "getpgrp", lambda: parent_pid)
    monkeypatch.setattr(
        harness,
        "_process_start_time_ticks",
        lambda process_id: parent_start if process_id == parent_pid else 1,
    )
    monkeypatch.setattr(harness, "_current_process_argv", lambda: process_argv)
    monkeypatch.setattr(harness, "_assert_standard_lock_held", lambda _path: None)
    fixture = AuthorizationFixture(
        config=config,
        output=output,
        authorization=authorization,
        lease_path=lease_path,
        result_descriptor=result_descriptor,
        output_descriptor=output_descriptor,
        cache_descriptors=tuple(caches.values()),
        read_descriptor=read_descriptor,
    )
    try:
        yield fixture
    finally:
        for descriptor in (
            read_descriptor,
            result_descriptor,
            output_descriptor,
            *caches.values(),
        ):
            with suppress(OSError):
                os.close(descriptor)


def test_validate_tuner_authorization_binds_runtime_lease_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with authorization_fixture(tmp_path, monkeypatch) as fixture:
        evidence = harness.validate_tuner_authorization(output_directory=fixture.output)
    assert evidence["run_id"] == fixture.config.run_id
    assert evidence["output_created_by_harness"] is True
    assert evidence["verified_authorization"] == fixture.authorization
    hashes = cast(dict[str, object], evidence["receipt_sha256"])
    assert {"runtime_install", "runtime_build", "kernel_validation"} <= set(hashes)


def test_validate_tuner_authorization_rejects_stored_payload_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with authorization_fixture(tmp_path, monkeypatch) as fixture:
        stored = Path(fixture.config.result_directory) / harness.AUTHORIZATION_FILENAME
        write_json(stored, {**fixture.authorization, "run_id": "tampered"})
        with pytest.raises(harness.HarnessError, match="differs from stored"):
            harness.validate_tuner_authorization(output_directory=fixture.output)


def test_validate_tuner_authorization_rejects_lease_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with authorization_fixture(tmp_path, monkeypatch) as fixture:
        original_read_json = harness._read_json
        lease_reads = 0

        def racing_read_json(
            path: Path,
            description: str,
            maximum_bytes: int = harness.MAXIMUM_RECEIPT_JSON_BYTES,
        ) -> harness.JsonObject:
            nonlocal lease_reads
            value = original_read_json(path, description, maximum_bytes)
            if path == fixture.lease_path:
                lease_reads += 1
                if lease_reads == 2:
                    value = {**value, "lease_id": "replacement"}
            return value

        monkeypatch.setattr(harness, "_read_json", racing_read_json)
        with pytest.raises(harness.HarnessError, match="changed during"):
            harness.validate_tuner_authorization(output_directory=fixture.output)


def test_validate_tuner_authorization_rejects_output_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with authorization_fixture(tmp_path, monkeypatch) as fixture:
        anchored = fixture.output.with_name("anchored-original")
        fixture.output.rename(anchored)
        fixture.output.mkdir()
        with pytest.raises(harness.HarnessError, match="fresh anchored directory"):
            harness.validate_tuner_authorization(output_directory=fixture.output)


class FakeProcess:
    pid = 43210

    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", 0)
        return self.returncode


@pytest.mark.parametrize(
    ("signal_number", "expected_timed_out"),
    ((signal.SIGINT, False), (None, True)),
)
def test_owned_tuner_records_signal_and_timeout_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_number: int | None,
    expected_timed_out: bool,
) -> None:
    config = make_config(tmp_path)
    result_dir = Path(config.result_directory)
    result_dir.mkdir(parents=True)
    result_descriptor = os.open(result_dir, os.O_RDONLY | os.O_DIRECTORY)
    fake = FakeProcess()
    popen_kwargs: dict[str, object] = {}

    def fake_popen(*_args: object, **kwargs: object) -> FakeProcess:
        popen_kwargs.update(kwargs)
        return fake

    monkeypatch.setattr(harness.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(harness, "_write_pipe_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        harness,
        "_authorization_payload",
        lambda *_args, **_kwargs: {"schema_version": 3},
    )
    monkeypatch.setattr(harness, "_verified_runtime_contract", lambda _config: {})
    monkeypatch.setattr(harness, "_process_start_time_ticks", lambda _pid: 10)
    monkeypatch.setattr(harness.os, "getpgrp", harness.os.getpid)
    monkeypatch.setattr(harness.os, "getpgid", lambda _pid: harness.os.getpid())
    monkeypatch.setattr(
        harness, "_discover_owned_processes", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        harness, "_terminate_owned_identities", lambda *_args, **_kwargs: True
    )
    if signal_number is None:
        moments = iter((0.0, 121.0, 121.0, 121.0, 121.0))
        monkeypatch.setattr(harness.time, "monotonic", lambda: next(moments, 121.0))
    state = harness.ManagedSignalState(first_signal_number=signal_number)
    try:
        outcome = harness._run_owned_tuner(
            config,
            lease={"lease_id": "lease"},
            result_descriptor=result_descriptor,
            result_dir=result_dir,
            lease_path=tmp_path / "lease",
            lock_path=tmp_path / "lock",
            owner_token="e" * 64,
            signal_state=state,
        )
        assert outcome.completion["timed_out"] is expected_timed_out
        assert outcome.cleanup_succeeded is True
        assert popen_kwargs["start_new_session"] is False
        os.close(outcome.output_descriptor)
    finally:
        os.close(result_descriptor)


def test_owned_tuner_prelaunch_failure_closes_all_setup_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    result_dir = Path(config.result_directory)
    result_dir.mkdir(parents=True)
    result_descriptor = os.open(result_dir, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(harness.os, "getpgrp", harness.os.getpid)
    monkeypatch.setattr(harness, "_verified_runtime_contract", lambda _config: {})

    def fail_authorization(*_args: object, **_kwargs: object) -> harness.JsonObject:
        raise harness.HarnessError("injected prelaunch failure")

    monkeypatch.setattr(harness, "_authorization_payload", fail_authorization)
    monkeypatch.setattr(
        harness.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
    )
    descriptors_before = set(os.listdir("/proc/self/fd"))
    try:
        outcome = harness._run_owned_tuner(
            config,
            lease={"lease_id": "lease"},
            result_descriptor=result_descriptor,
            result_dir=result_dir,
            lease_path=tmp_path / "lease",
            lock_path=tmp_path / "lock",
            owner_token="e" * 64,
            signal_state=harness.ManagedSignalState(),
        )
        assert outcome.cleanup_succeeded is True
        assert outcome.owned_processes == ()
        assert "injected prelaunch failure" in cast(str, outcome.error)
        os.close(outcome.output_descriptor)
        assert set(os.listdir("/proc/self/fd")) == descriptors_before
    finally:
        os.close(result_descriptor)


def test_tagged_process_discovery_rejects_pid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    token = uuid.uuid4().hex * 2
    proc_root = tmp_path / "proc"
    process_directory = proc_root / "101"
    process_directory.mkdir(parents=True)
    (process_directory / "environ").write_bytes(
        f"{harness.TUNER_OWNER_TOKEN_ENVIRONMENT}={token}\0".encode()
    )
    identities = iter(
        (
            ("S", 101, 101, 111),
            ("S", 202, 202, 222),
        )
    )
    monkeypatch.setattr(
        harness,
        "_process_stat_identity",
        lambda process_id: next(identities),
    )

    assert (
        harness._tagged_processes(
            token, config, tmp_path / "owned.log", proc_root=proc_root
        )
        == {}
    )


def test_tagged_descendant_cleanup_terminates_all_owned_processes(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    token = uuid.uuid4().hex * 2
    environment = {**os.environ, harness.TUNER_OWNER_TOKEN_ENVIRONMENT: token}
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,time; subprocess.Popen(['sleep','60']); time.sleep(60)",
        ],
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3
        processes: dict[tuple[int, int], harness.ProcessIdentity] = {}
        while time.monotonic() < deadline:
            processes = harness._tagged_processes(token, config, tmp_path / "owned.log")
            if len(processes) >= 2:
                break
            time.sleep(0.02)
        assert len(processes) >= 2
        assert harness._terminate_owned_identities(
            processes,
            1.0,
            discover_owned_processes=lambda: harness._tagged_processes(
                token, config, tmp_path / "owned.log"
            ),
        )
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=2)


def test_cleanup_rescans_descendant_spawned_by_term_handler(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    token = uuid.uuid4().hex * 2
    environment = {**os.environ, harness.TUNER_OWNER_TOKEN_ENVIRONMENT: token}
    script = (
        "import os,signal,subprocess,sys,time\n"
        "def stop(_signal, _frame):\n"
        " subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        " time.sleep(0.2)\n"
        " os._exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert parent.stdout is not None
        assert parent.stdout.readline().strip() == "ready"
        processes = harness._tagged_processes(token, config, tmp_path / "owned.log")
        assert len(processes) == 1
        assert harness._terminate_owned_identities(
            processes,
            1.0,
            discover_owned_processes=lambda: harness._tagged_processes(
                token, config, tmp_path / "owned.log"
            ),
        )
        assert len(processes) >= 2
        assert not any(harness._identity_alive(item) for item in processes.values())
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=2)


def test_cleanup_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    identity: harness.ProcessIdentity = {
        "host_name": "dwagon",
        "pid": 99,
        "process_group_id": 99,
        "start_time_ticks": 1,
        "transport_pid": 99,
        "namespace": "exo-test",
        "owner_token": "e" * 64,
        "log_path": "/tmp/test.log",
    }
    monkeypatch.setattr(harness, "_identity_alive", lambda _identity: True)
    monkeypatch.setattr(
        harness, "_signal_process_identity", lambda _identity, _signal: True
    )
    monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
    ticks = iter((0.0, 2.0, 2.0, 4.0, 4.0))
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks, 4.0))
    assert (
        harness._terminate_owned_identities(
            {(identity["pid"], identity["start_time_ticks"]): identity},
            1.0,
            discover_owned_processes=dict,
        )
        is False
    )


def kernel_config() -> dict[str, int]:
    return dict(tuner.build_rtx3090_search_space("quick")[0])


def numerical(*, scenario: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "relative_l1": 0.001,
        "max_absolute": 0.002,
        "repeat_exact": True,
    }
    if scenario is not None:
        value = {"scenario": scenario, **value}
    return value


def valid_stage_measurement(
    stage: str,
    strata: list[dict[str, object]],
    samples: int,
    kernel_configuration: dict[str, int],
) -> dict[str, object]:
    stratum_evidence: list[dict[str, object]] = []
    for stratum in strata:
        resident = cast(int, stratum["resident_routes_per_token"])
        stratum_evidence.append(
            {
                "name": stratum["name"],
                "resident_route_count": stratum["resident_route_count"],
                "resident_routes_per_token": resident,
                "probability_weight": stratum["probability_weight"],
                "sample_microseconds": [10.0 + resident] * samples,
                "fallback_before_microseconds": [20.0 + resident] * samples,
                "fallback_after_microseconds": [22.0 + resident] * samples,
                "numerical_evidence": numerical(),
            }
        )
    candidate = sum(
        cast(float, value["probability_weight"])
        * cast(list[float], value["sample_microseconds"])[0]
        for value in stratum_evidence
    )
    before = sum(
        cast(float, value["probability_weight"])
        * cast(list[float], value["fallback_before_microseconds"])[0]
        for value in stratum_evidence
    )
    after = sum(
        cast(float, value["probability_weight"])
        * cast(list[float], value["fallback_after_microseconds"])[0]
        for value in stratum_evidence
    )
    reference = (before + after) / 2
    return {
        "stage": stage,
        "config": kernel_configuration,
        "sample_microseconds": [candidate] * samples,
        "fallback_before_microseconds": [before] * samples,
        "fallback_after_microseconds": [after] * samples,
        "fallback_reference_microseconds": [reference] * samples,
        "relative_improvement_samples": [(reference - candidate) / reference] * samples,
        "stable_minimum_five_percent_improvement": (
            kernel_configuration != tuner._FALLBACK_CONFIG
        ),
        "numerical_evidence": [
            numerical(scenario=name)
            for name in ("uniform", "zero_resident", "mixed", "resident_skew")
        ],
        "timing_strata": stratum_evidence,
    }


def valid_anchor(batch_size: int, config: harness.TuningRunConfig) -> dict[str, object]:
    strata: list[dict[str, object]] = []
    numerator = batch_size * 4 * config.tuning.resident_experts
    lower_count, remainder = divmod(numerator, harness.GLM47_GLOBAL_EXPERTS)
    weighted_counts = (
        ((lower_count, 1.0),)
        if remainder == 0
        else (
            (lower_count, 1 - remainder / harness.GLM47_GLOBAL_EXPERTS),
            (lower_count + 1, remainder / harness.GLM47_GLOBAL_EXPERTS),
        )
    )
    for resident, probability in weighted_counts:
        strata.append(
            {
                "name": f"resident_total_{resident}",
                "resident_route_count": resident,
                "resident_routes_per_token": resident / batch_size,
                "probability_weight": probability,
                "global_route_sha256": SHA,
                "masked_route_sha256": SHA,
            }
        )
    candidate_records: list[dict[str, object]] = []
    for raw_kernel_config in tuner.build_rtx3090_search_space(
        config.tuning.search_profile
    ):
        candidate_config = dict(raw_kernel_config)
        gate = valid_stage_measurement(
            "gate_up",
            strata,
            config.tuning.independent_samples,
            candidate_config,
        )
        down = valid_stage_measurement(
            "down",
            strata,
            config.tuning.independent_samples,
            candidate_config,
        )
        candidate_records.append(
            {
                "config": candidate_config,
                "measurements": [gate, down],
                "rejections": [],
            }
        )
    selected_measurements = cast(
        list[dict[str, object]], candidate_records[0]["measurements"]
    )
    gate = selected_measurements[0]
    down = selected_measurements[1]
    route_scenarios = []
    for index, name in enumerate(
        ("uniform", "zero_resident", "mixed", "resident_skew")
    ):
        resident_count = min(index, 4) * batch_size
        route_scenarios.append(
            {
                "name": name,
                "global_route_sha256": SHA,
                "masked_route_sha256": SHA,
                "resident_route_count": resident_count,
                "masked_cpu_route_count": batch_size * 4 - resident_count,
                "resident_routes_per_token": resident_count / batch_size,
            }
        )
    return {
        "batch_size": batch_size,
        "route_scenarios": route_scenarios,
        "timing_route_strata": strata,
        "selected": {
            "shared_block_size_m": 16,
            "gate_up": gate,
            "down": down,
            "gate_up_retained_fallback": False,
            "down_retained_fallback": False,
        },
        "candidate_records": candidate_records,
    }


def write_valid_tuning_output(
    config: harness.TuningRunConfig,
) -> tuple[Path, int, int, dict[str, object]]:
    output = Path(config.result_directory) / harness.TUNING_OUTPUT_DIRECTORY_NAME
    config_directory = output / "configs/triton_3_5_1"
    config_directory.mkdir(parents=True)
    deployment_tuner = Path(config.source_deployment_root) / harness.TUNER_RELATIVE_PATH
    deployment_tuner.parent.mkdir(parents=True)
    deployment_tuner.write_bytes(Path(tuner.__file__).read_bytes())
    gate_name = "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090.json"
    down_name = "E=4,N=1536,device_name=NVIDIA_GeForce_RTX_3090_down.json"
    configs = {str(batch): kernel_config() for batch in config.tuning.batch_sizes}
    gate_contents = canonical_json(configs)
    down_contents = canonical_json(configs)
    (config_directory / gate_name).write_bytes(gate_contents)
    (config_directory / down_name).write_bytes(down_contents)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    runtime_contract = harness._verified_runtime_contract(config)
    identity = harness._directory_identity(descriptor)
    source_identity = (
        Path(config.source_deployment_root) / harness.SOURCE_IDENTITY_FILENAME
    )
    source_identity.write_text("{}\n", encoding="ascii")
    tuning_config = (
        Path(config.source_deployment_root) / harness.IMMUTABLE_CONFIG_FILENAME
    )
    tuning_config.write_text("{}\n", encoding="ascii")
    preflight = output.parent / harness.PREFLIGHT_FILENAME
    preflight.write_text("{}\n", encoding="ascii")
    telemetry = output.parent / harness.TELEMETRY_BEFORE_FILENAME
    telemetry.write_text("{}\n", encoding="ascii")
    contract_receipts = cast(dict[str, object], runtime_contract["receipt_bindings"])
    receipt_bindings: dict[str, object] = {
        "runtime_python": config.runtime_python.model_dump(mode="json"),
        **contract_receipts,
        "tuner_script": {
            "path": str(deployment_tuner),
            "sha256": file_sha256(deployment_tuner),
        },
        "source_identity": {
            "path": str(source_identity),
            "sha256": file_sha256(source_identity),
        },
        "tuning_config": {
            "path": str(tuning_config),
            "sha256": file_sha256(tuning_config),
        },
        "preflight": {"path": str(preflight), "sha256": file_sha256(preflight)},
        "telemetry_before": {
            "path": str(telemetry),
            "sha256": file_sha256(telemetry),
        },
    }
    result_status = output.parent.stat()
    process_argv = [config.runtime_python.path, str(deployment_tuner)]
    child_environment = {"inheritance_policy": "allowlist-v1"}
    parent_process = {"pid": 1234, "start_time_ticks": 5678}
    verified_authorization: dict[str, object] = {
        "schema_version": 3,
        "authorization_kind": "active-benchmark-lease-inherited-pipe-v3",
        "lease_id": "lease-1",
        "run_id": config.run_id,
        "namespace": config.namespace,
        "lock_path": "/var/lock/fwuffydwagon-benchmark.lock",
        "lease_path": "/var/lib/exo/coordination/benchmark-lease.json",
        "result_directory": {
            "path": str(output.parent),
            "device": result_status.st_dev,
            "inode": result_status.st_ino,
        },
        "harness_process": parent_process,
        "process_ownership": {
            "mode": "inherit-lease-child-process-group",
            "lease_child_pid": 1234,
            "process_group_id": 1234,
            "outer_wrapper_cleanup": "killpg",
        },
        "owner_token": "e" * 64,
        "launcher_argv": ["/usr/bin/numactl", *process_argv],
        "tuner_process_argv": process_argv,
        "tuning_output_directory": str(output),
        "output_directory_descriptor": descriptor,
        "output_directory_identity": identity,
        "cache_directories": {
            name: {"descriptor": index, "identity": {"device": 1, "inode": index}}
            for index, name in enumerate(("cuda-cache", "triton-cache", "tmp"), 100)
        },
        "child_environment": child_environment,
        "runtime_contract": runtime_contract,
        "receipt_bindings": receipt_bindings,
        "experiment_context": {
            "contextual_model": config.contextual_model.model_dump(mode="json"),
            "model_snapshot_consumed": False,
            "contextual_model_snapshot_weights_loaded": False,
            "synthetic_kernel_weights_generated": True,
            "binding_establishes_model_verification": False,
        },
        "created_at": "2026-07-19T12:00:00+00:00",
    }
    receipt_sha256 = {
        name: cast(dict[str, object], binding)["sha256"]
        for name, binding in receipt_bindings.items()
    }
    evidence: dict[str, object] = {
        "schema_version": 1,
        "authorization_sha256": hashlib.sha256(
            canonical_json(verified_authorization, pretty=False)
        ).hexdigest(),
        "lease_id": "lease-1",
        "run_id": config.run_id,
        "namespace": config.namespace,
        "result_directory": verified_authorization["result_directory"],
        "tuning_output_directory": str(output),
        "output_directory_descriptor": descriptor,
        "output_directory_identity": identity,
        "tuner_process_argv": process_argv,
        "launcher_argv": verified_authorization["launcher_argv"],
        "receipt_sha256": receipt_sha256,
        "child_environment": child_environment,
        "parent_process": parent_process,
        "lock_path": verified_authorization["lock_path"],
        "lease_path": verified_authorization["lease_path"],
        "output_created_by_harness": True,
        "runtime_contract": runtime_contract,
        "verified_authorization": verified_authorization,
    }
    authorization_path = output.parent / harness.AUTHORIZATION_FILENAME
    authorization_path.write_bytes(canonical_json(verified_authorization))
    authorization_path.chmod(0o444)
    manifest: dict[str, object] = {
        "schema_version": 3,
        "artifact_type": "glm47_sglang_kt_fused_moe_candidate_v3",
        "candidate": True,
        "deployment_admitted": False,
        "performance_comparable": False,
        "benchmark_scope": "synthetic_separate_stage_kernels_only",
        "production_path_reproduced": False,
        "limitations": [
            "alignment is prepared outside timed regions",
            "the pinned filtered activation and final reduction are not timed",
            "concurrent CPU AMX expert execution and resource contention are not reproduced",
            "resident-route strata preserve the uniform expected count but are synthetic",
            "stratum weights assume uniform global top-k routing, not a captured trace",
            "the untuned serving baseline profile does not consume this bundle",
            "the current serving_baseline profile strips SGLANG_* variables",
        ],
        "serving_admission_required": {
            "minimum_improvement": 0.03,
            "metric": "matched end-to-end serving performance",
            "baseline_target_profile": "GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE",
            "baseline_profile_is_untuned": True,
            "baseline_profile_consumes_this_bundle": False,
            "consumer_chain": "KTEP gpu_method.apply -> SGLang fused_moe config loader",
            "required_environment_binding": "SGLANG_MOE_CONFIG_DIR with exact config file SHA-256",
            "current_serving_baseline_strips_sglang_environment": True,
            "profile": "future tuned profile with exact config SHA-256 binding",
        },
        "tuner": {
            "path": str(deployment_tuner),
            "sha256": file_sha256(deployment_tuner),
        },
        "authorization": {
            "authorization_sha256": evidence["authorization_sha256"],
            "evidence": evidence,
            "evidence_sha256": hashlib.sha256(
                canonical_json(evidence, pretty=False)
            ).hexdigest(),
            "output_directory_identity": identity,
        },
        "output_contract": {
            "descriptor_anchored": True,
            "harness_created_empty_directory": True,
            "semantic_path": str(output),
            "identity": identity,
        },
        "shape": {"H": 2048, "N": 1536, "E": 4, "global_experts": 64, "top_k": 4},
        "workload": {
            "resident_experts": 4,
            "batch_sizes": list(config.tuning.batch_sizes),
            "global_experts": 64,
            "top_k": 4,
        },
        "route_contract": {
            "cpu_experts_are_masked_to": -1,
            "resident_global_expert_ids": [0, 1, 2, 3],
            "seed": config.tuning.seed,
            "timed_scenarios": "deterministic expected total resident-route count strata",
            "decode_actual_resident_gemm_timed": True,
            "correctness_scenarios": [
                "uniform",
                "zero_resident",
                "mixed",
                "resident_skew",
            ],
            "uniform_expected_resident_routes_per_token": 0.25,
        },
        "measurement_contract": {
            "batch_sizes": list(config.tuning.batch_sizes),
            "warmup_iterations": 3,
            "measurement_iterations": 5,
            "independent_samples": 3,
            "search_profile": "quick",
            "relative_l1_tolerance": 0.02,
            "max_absolute_tolerance": 0.02,
            "jit_compilation_excluded": True,
            "timing_source": "CUDA events",
            "candidate_order": "canonical fixed order",
            "drift_mitigation": "fallback-before/candidate/fallback-after",
            "minimum_stable_synthetic_improvement": 0.05,
            "selection_fallback": "retain deployed fallback unless every paired sample meets threshold",
        },
        "runtime_contract": runtime_contract,
        "runtime_parent_receipts": {
            name: {
                "sha256": cast(dict[str, object], binding)["sha256"],
                "binding": binding,
            }
            for name, binding in contract_receipts.items()
        },
        "runtime_observed": {
            "torch_version": "2.9.1+cu128",
            "cuda_version": "12.8",
            "triton_version": "3.5.1",
            "sglang_revision": SGLANG_REVISION,
            "ktransformers_revision": KTRANSFORMERS_REVISION,
            "device_name": GPU_NAME,
            "gpu_uuid": GPU_UUID,
        },
        "config_files": {
            "gate_up": {
                "relative_path": f"configs/triton_3_5_1/{gate_name}",
                "sha256": hashlib.sha256(gate_contents).hexdigest(),
                "shape": {"E": 4, "N": 1536},
                "batch_keys": list(config.tuning.batch_sizes),
            },
            "down": {
                "relative_path": f"configs/triton_3_5_1/{down_name}",
                "sha256": hashlib.sha256(down_contents).hexdigest(),
                "shape": {"E": 4, "N": 1536},
                "batch_keys": list(config.tuning.batch_sizes),
            },
        },
        "anchors": [valid_anchor(batch, config) for batch in config.tuning.batch_sizes],
    }
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    result_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    return output, result_descriptor, descriptor, manifest


def test_tuning_output_evidence_strictly_accepts_full_v3_candidate(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    _output, result_descriptor, descriptor, _manifest = write_valid_tuning_output(
        config
    )
    try:
        evidence = harness._tuning_output_evidence(
            config, result_descriptor, descriptor
        )
    finally:
        os.close(descriptor)
        os.close(result_descriptor)
    assert evidence["candidate_only"] is True
    assert evidence["performance_comparable"] is False
    gate = cast(dict[str, object], evidence["adoption_gate"])
    assert gate["minimum_improvement_percent"] == 3.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("candidate", False, "candidate"),
        ("deployment_admitted", True, "deployment_admitted"),
        ("artifact_type", "wrong", "artifact_type"),
    ),
)
def test_tuning_output_evidence_rejects_malformed_manifest(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    manifest[field] = value
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match=message):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_rejects_extra_tree_entry(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, _manifest = write_valid_tuning_output(config)
    (output / "unexpected.json").write_text("{}\n", encoding="ascii")
    try:
        with pytest.raises(harness.HarnessError, match="unexpected entries"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_rejects_self_rehashed_model_context_tamper(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    authorization = cast(dict[str, object], manifest["authorization"])
    evidence = cast(dict[str, object], authorization["evidence"])
    verified = cast(dict[str, object], evidence["verified_authorization"])
    experiment_context = cast(dict[str, object], verified["experiment_context"])
    contextual_model = cast(dict[str, object], experiment_context["contextual_model"])
    contextual_model["model_id"] = "attacker/self-rehashed"
    evidence["authorization_sha256"] = hashlib.sha256(
        canonical_json(verified, pretty=False)
    ).hexdigest()
    authorization["authorization_sha256"] = evidence["authorization_sha256"]
    authorization["evidence_sha256"] = hashlib.sha256(
        canonical_json(evidence, pretty=False)
    ).hexdigest()
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(
            harness.HarnessError, match="stored read-only authorization"
        ):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


@pytest.mark.parametrize("field", ("lease_id", "namespace", "result_directory"))
def test_tuning_output_evidence_rejects_self_rehashed_outer_authorization_tamper(
    tmp_path: Path, field: str
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    authorization = cast(dict[str, object], manifest["authorization"])
    evidence = cast(dict[str, object], authorization["evidence"])
    evidence[field] = "attacker/self-rehashed"
    authorization["evidence_sha256"] = hashlib.sha256(
        canonical_json(evidence, pretty=False)
    ).hexdigest()
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="authorization evidence"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_requires_read_only_stored_authorization(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    _output, result_descriptor, descriptor, _manifest = write_valid_tuning_output(
        config
    )
    authorization_path = Path(config.result_directory) / harness.AUTHORIZATION_FILENAME
    authorization_path.chmod(0o644)
    try:
        with pytest.raises(harness.HarnessError, match="bounded regular file"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered"))
def test_tuning_output_evidence_requires_full_canonical_candidate_order(
    tmp_path: Path, mutation: str
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    candidates = cast(list[dict[str, object]], anchors[0]["candidate_records"])
    if mutation == "missing":
        candidates.pop()
        message = "full search space"
    elif mutation == "extra":
        candidates.append(copy.deepcopy(candidates[-1]))
        message = "full search space"
    else:
        candidates[0], candidates[1] = candidates[1], candidates[0]
        message = "canonical search order"
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match=message):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_recomputes_stable_improvement(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    selected = cast(dict[str, object], anchors[0]["selected"])
    gate = cast(dict[str, object], selected["gate_up"])
    gate["stable_minimum_five_percent_improvement"] = False
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="stable-improvement"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_recomputes_retained_fallback(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    selected = cast(dict[str, object], anchors[0]["selected"])
    selected["gate_up_retained_fallback"] = True
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="fallback decision"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_requires_selected_candidate_measurement(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    selected = cast(dict[str, object], anchors[0]["selected"])
    selected_gate = copy.deepcopy(cast(dict[str, object], selected["gate_up"]))
    numerical_evidence = cast(
        list[dict[str, object]], selected_gate["numerical_evidence"]
    )
    numerical_evidence[0]["relative_l1"] = 0.0015
    selected["gate_up"] = selected_gate
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="not the admitted candidate"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_requires_deterministic_minimum_pair(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    anchor = anchors[0]
    candidates = cast(list[dict[str, object]], anchor["candidate_records"])
    faster_candidate = next(
        candidate
        for candidate in candidates
        if cast(dict[str, int], candidate["config"])["BLOCK_SIZE_M"] == 32
    )
    faster_measurements = cast(
        list[dict[str, object]], faster_candidate["measurements"]
    )
    for measurement in faster_measurements:
        samples = cast(list[float], measurement["sample_microseconds"])
        measurement["sample_microseconds"] = [sample / 2 for sample in samples]
        fallback = cast(list[float], measurement["fallback_reference_microseconds"])
        measurement["relative_improvement_samples"] = [
            (reference - sample / 2) / reference
            for sample, reference in zip(samples, fallback, strict=True)
        ]
        timing_strata = cast(list[dict[str, object]], measurement["timing_strata"])
        for stratum in timing_strata:
            stratum_samples = cast(list[float], stratum["sample_microseconds"])
            stratum["sample_microseconds"] = [sample / 2 for sample in stratum_samples]
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(
            harness.HarnessError, match="deterministic admitted minimum"
        ):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_binds_selected_config_to_emitted_file(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    config_files = cast(dict[str, dict[str, object]], manifest["config_files"])
    gate_binding = config_files["gate_up"]
    relative_path = cast(str, gate_binding["relative_path"])
    config_path = output / relative_path
    contents = cast(dict[str, dict[str, int]], json.loads(config_path.read_bytes()))
    contents[str(config.tuning.batch_sizes[0])]["num_warps"] = 8
    changed_contents = canonical_json(contents)
    config_path.write_bytes(changed_contents)
    gate_binding["sha256"] = hashlib.sha256(changed_contents).hexdigest()
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="emitted config file"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)


def test_tuning_output_evidence_enforces_declared_numerical_tolerance(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    output, result_descriptor, descriptor, manifest = write_valid_tuning_output(config)
    anchors = cast(list[dict[str, object]], manifest["anchors"])
    selected = cast(dict[str, object], anchors[0]["selected"])
    gate = cast(dict[str, object], selected["gate_up"])
    evidence = cast(list[dict[str, object]], gate["numerical_evidence"])
    evidence[0]["relative_l1"] = 0.021
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    try:
        with pytest.raises(harness.HarnessError, match="declared tolerance"):
            harness._tuning_output_evidence(config, result_descriptor, descriptor)
    finally:
        os.close(descriptor)
        os.close(result_descriptor)
