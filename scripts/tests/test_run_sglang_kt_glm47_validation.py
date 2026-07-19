from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from scripts import run_sglang_kt_glm47_validation as harness

GPU_UUID = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
GPU_BDF = "00000000:01:00.0"
GID = "fe80:0000:0000:0000:0210:e000:0166:3a19"
SHA = "a" * 64


def config_payload(
    tmp_path: Path,
    *,
    phase: str = "cpu_control",
    resident_gpu_experts: int = 0,
    runtime_path: str = "/runtime/venv/bin/python",
    runtime_sha256: str = SHA,
    runtime_chain: list[str] | None = None,
) -> dict[str, object]:
    run_id = f"glm47-test-{uuid.uuid4().hex}"
    deployment_root = tmp_path / "deployment"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "namespace": f"exo-{run_id}",
        "phase": phase,
        "profiler": "none",
        "hca_requirement": "metadata_only",
        "result_directory": str(tmp_path / "results" / run_id),
        "scratch_directory": f"/var/lib/exo/validation-scratch/{run_id}",
        "source": {
            "repository": "/root/exo",
            "deployment_root": str(deployment_root),
        },
        "runtime_python": {
            "path": runtime_path,
            "sha256": runtime_sha256,
            "symlink_chain": runtime_chain
            if runtime_chain is not None
            else ["python3.12", "/runtime/base/python3.12"],
        },
        "numactl_executable": "/usr/bin/numactl",
        "build_receipt": {
            "path": str(tmp_path / "build-receipt.json"),
            "sha256": SHA,
        },
        "model_path": str(tmp_path / "model"),
        "model_contract": {
            "path": str(
                deployment_root / "orchestrator" / harness.MODEL_CONTRACT_RELATIVE_PATH
            ),
            "sha256": harness.GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        },
        "host": {
            "hostname": "dwagon",
            "node_id": "dwagon",
            "gpu": {"uuid": GPU_UUID, "pci_address": GPU_BDF},
            "cpu_cores": [0, 1, 2, 3],
            "memory_nodes": [0],
            "threads_per_subpool": [4],
            "cpu_infer_threads": 4,
            "threadpool_count": 1,
            "hca_bindings": [
                {
                    "device": "mlx4_0",
                    "port": 1,
                    "gid_index": 0,
                    "gid": GID,
                }
            ],
        },
        "distributed_coordinator": {"ip": "192.0.2.10", "port": 29510},
        "service_endpoint": {"ip": "192.0.2.10", "port": 30100},
        "reserved_ports": [29510, 30100],
        "resident_gpu_experts": resident_gpu_experts,
        "timeouts": {
            "generator_seconds": 30.0,
            "kernel_seconds": 120.0,
            "model_seconds": 3600.0,
            "cleanup_seconds": 120.0,
        },
    }


def make_config(
    tmp_path: Path,
    *,
    phase: str = "cpu_control",
    resident_gpu_experts: int = 0,
) -> harness.ValidationConfig:
    return harness.ValidationConfig.model_validate_json(
        json.dumps(
            config_payload(
                tmp_path,
                phase=phase,
                resident_gpu_experts=resident_gpu_experts,
            )
        )
    )


def deployment(tmp_path: Path) -> harness.DeploymentIdentity:
    return harness.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="b" * 64,
        validator_sha256="c" * 64,
        validator_files=(),
        source=harness.SourceIdentity("d" * 40, {}),
    )


def command_outcome(
    name: str, *, return_code: int = 0, error: str | None = None
) -> harness.CommandOutcome:
    return harness.CommandOutcome(
        name=name,
        argv=("/mock/runtime-python", f"{name}.py"),
        return_code=return_code,
        stdout_name=f"{name}.stdout.log",
        stderr_name=f"{name}.stderr.log",
        elapsed_seconds=0.25,
        cleanup_succeeded=True,
        error=error,
    )


def open_results(config: harness.ValidationConfig) -> harness.ResultDirectory:
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return harness.ResultDirectory(result_path, descriptor)
    finally:
        os.close(descriptor)


def test_parse_command_json_accepts_one_clean_object(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    results = open_results(config)
    outcome = command_outcome("model-validator")
    try:
        with results.create_log(outcome.stdout_name) as stdout:
            stdout.write(b'{"schema_version":1,"status":"passed"}\n')

        assert harness._parse_command_json(results, outcome) == {
            "schema_version": 1,
            "status": "passed",
        }
    finally:
        results.close()


@pytest.mark.parametrize(
    "contents",
    (
        b'native diagnostic\n{"status":"passed"}\n',
        b'{"status":"passed"}\nnative diagnostic\n',
        b'{"status":"passed"}\n{"status":"passed"}\n',
    ),
)
def test_parse_command_json_rejects_contaminated_stream(
    tmp_path: Path,
    contents: bytes,
) -> None:
    config = make_config(tmp_path)
    results = open_results(config)
    outcome = command_outcome("model-validator")
    try:
        with results.create_log(outcome.stdout_name) as stdout:
            stdout.write(contents)

        with pytest.raises(harness.Glm47HarnessError, match="stdout is invalid JSON"):
            harness._parse_command_json(results, outcome)
    finally:
        results.close()


def fake_owned_cgroup(tmp_path: Path) -> harness.OwnedCgroup:
    parent = tmp_path / f"fake-cgroup-parent-{uuid.uuid4().hex}"
    child = parent / "validators-test"
    parent.mkdir(mode=0o700)
    child.mkdir(mode=0o700)
    (child / "cgroup.procs").write_bytes(b"")
    (child / "cgroup.events").write_bytes(b"populated 0\nfrozen 0\n")
    (child / "cgroup.kill").write_bytes(b"")
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = os.open(child, os.O_RDONLY | os.O_DIRECTORY)
    observed = os.fstat(descriptor)
    return harness.OwnedCgroup(
        path=child,
        parent_descriptor=parent_descriptor,
        descriptor=descriptor,
        device=observed.st_dev,
        inode=observed.st_ino,
        owner_uid=os.geteuid(),
        invocation_id="1" * 32,
        systemd_unit_name="exo-glm47-" + ("2" * 32) + ".service",
    )


def close_fake_owned_cgroup(owned: harness.OwnedCgroup) -> None:
    os.close(owned.descriptor)
    os.close(owned.parent_descriptor)


def install_run_validation_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    identity: harness.DeploymentIdentity,
    preflight: harness.JsonObject,
) -> None:
    owned_cgroup = fake_owned_cgroup(Path(identity.root).parent)

    def enable_child_subreaper() -> None:
        return None

    def collect_preflight(
        _config: harness.ValidationConfig,
    ) -> harness.JsonObject:
        return preflight

    def create_scratch(
        _config: harness.ValidationConfig,
    ) -> harness.OwnedScratchDirectory | None:
        return None

    def create_cgroup(
        _config: harness.ValidationConfig,
        _owner_token: str,
    ) -> harness.OwnedCgroup:
        return owned_cgroup

    def child_environment(
        _config: harness.ValidationConfig,
        _owner_token: str,
        _parent_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        return {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}

    def verify_runtime(_binding: harness.ExecutableBinding) -> None:
        return None

    def verify_artifact(_binding: harness.ArtifactBinding, _description: str) -> None:
        return None

    def load_identity(_root: Path) -> harness.DeploymentIdentity:
        return identity

    def cleanup_processes(
        _config: harness.ValidationConfig,
        _registry: harness.OwnershipRegistry,
        _default_log_path: str,
    ) -> bool:
        return True

    def cleanup_cgroup(
        observed: harness.OwnedCgroup,
        _timeout_seconds: float,
    ) -> bool:
        assert observed == owned_cgroup
        close_fake_owned_cgroup(observed)
        return True

    monkeypatch.setattr(harness, "_enable_child_subreaper", enable_child_subreaper)
    monkeypatch.setattr(harness, "collect_live_preflight", collect_preflight)
    monkeypatch.setattr(harness, "_create_scratch", create_scratch)
    monkeypatch.setattr(harness, "_create_owned_cgroup", create_cgroup)
    monkeypatch.setattr(harness, "build_child_environment", child_environment)
    monkeypatch.setattr(harness, "verify_runtime_python", verify_runtime)
    monkeypatch.setattr(harness, "_verify_artifact", verify_artifact)
    monkeypatch.setattr(harness, "load_deployment_identity", load_identity)
    monkeypatch.setattr(harness, "cleanup_all_owned_processes", cleanup_processes)
    monkeypatch.setattr(harness, "cleanup_owned_cgroup", cleanup_cgroup)


def command_names(result: harness.JsonObject) -> tuple[str, ...]:
    commands = cast(list[harness.JsonValue], result["commands"])
    return tuple(
        cast(str, cast(harness.JsonObject, command)["name"]) for command in commands
    )


def assert_persisted_result(
    config: harness.ValidationConfig,
    results: harness.ResultDirectory,
    result: harness.JsonObject,
) -> None:
    assert results.read_json(harness.BENCHMARK_RESULT_FILENAME) == result
    manifest = results.read_json(harness.CHILD_MANIFEST_FILENAME)
    assert manifest["status"] == result["status"]
    assert manifest["reportable"] == result["reportable"]
    assert manifest["model_checkpoint_verified"] == result["model_checkpoint_verified"]
    assert manifest["preflight"] == result["preflight"]
    assert manifest["pipeline"] == result["pipeline"]
    assert manifest["commands"] == result["commands"]
    assert manifest["config"] == config.model_dump(mode="json")
    runtime_metadata = results.read_json(harness.RUNTIME_METADATA_FILENAME)
    assert runtime_metadata["containment"] == result["containment"]


def test_import_is_inert_without_model_or_gpu_runtimes() -> None:
    repository = Path(__file__).resolve().parents[2]
    program = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(repository)!r})
sys.path.insert(0, {str(repository / "src")!r})

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        blocked = ('torch', 'sglang', 'ktransformers', 'kt_kernel', 'pynvml')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ModuleNotFoundError('heavy runtime import was attempted')
        return None

sys.meta_path.insert(0, BlockHeavyImports())
import scripts.run_sglang_kt_glm47_validation
"""
    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("phase", "residents"),
    (("kernel", 0), ("kernel", 4), ("cpu_control", 0), ("hybrid", 1), ("hybrid", 4)),
)
def test_config_accepts_exact_validation_phases(
    tmp_path: Path, phase: str, residents: int
) -> None:
    config = harness.ValidationConfig.model_validate_json(
        json.dumps(
            config_payload(tmp_path, phase=phase, resident_gpu_experts=residents)
        )
    )
    assert config.phase == phase
    assert config.resident_gpu_experts == residents


@pytest.mark.parametrize(
    ("phase", "residents"),
    (("cpu_control", 1), ("hybrid", 0)),
)
def test_config_rejects_phase_resident_mismatch(
    tmp_path: Path, phase: str, residents: int
) -> None:
    with pytest.raises(ValidationError, match="requires"):
        harness.ValidationConfig.model_validate_json(
            json.dumps(
                config_payload(tmp_path, phase=phase, resident_gpu_experts=residents)
            )
        )


def test_config_requires_exact_two_reserved_ports(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    payload["reserved_ports"] = [29510, 30100, 30101]
    with pytest.raises(ValidationError, match="reserved_ports"):
        harness.ValidationConfig.model_validate_json(json.dumps(payload))


def test_runtime_python_binds_explicit_symlink_chain(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    final = runtime_root / "python-base"
    final.write_bytes(b"runtime executable")
    final.chmod(0o755)
    intermediate = runtime_root / "python3.12"
    intermediate.symlink_to(final)
    lexical = runtime_root / "python"
    lexical.symlink_to("python3.12")
    binding = harness.ExecutableBinding(
        path=str(lexical),
        sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        symlink_chain=("python3.12", str(final)),
    )
    harness.verify_runtime_python(binding)
    intermediate.unlink()
    intermediate.symlink_to(runtime_root / "other")
    with pytest.raises(harness.Glm47HarnessError, match="symlink chain"):
        harness.verify_runtime_python(binding)


@pytest.mark.parametrize(
    "value",
    ("0000:16:00.0", "00000000:16:00.0", "00000000:16:00.0".upper()),
)
def test_gpu_pci_address_normalizes_nvml_and_linux_domains(value: str) -> None:
    binding = harness.GpuBinding(uuid=GPU_UUID, pci_address=value)
    assert binding.pci_address == "00000000:16:00.0"


def test_local_hca_policy_records_init_but_active_policy_rejects_it(
    tmp_path: Path,
) -> None:
    port = tmp_path / "mlx4_0/ports/1"
    (port / "gids").mkdir(parents=True)
    (port / "state").write_text("2: INIT\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    (port / "rate").write_text("40 Gb/sec (4X QDR)\n")
    (port / "gids/0").write_text(f"{GID}\n")
    binding = harness.HcaBinding(device="mlx4_0", port=1, gid_index=0, gid=GID)
    evidence = harness.collect_local_hca_evidence((binding,), "metadata_only", tmp_path)
    port_evidence = cast(harness.JsonObject, evidence[0])
    assert port_evidence["subnet_manager_active"] is False
    assert port_evidence["traffic_expected"] is False
    with pytest.raises(harness.Glm47HarnessError, match="LinkUp/GID"):
        harness.collect_local_hca_evidence((binding,), "active", tmp_path)


def test_loaded_sep_pax_are_recorded_but_not_treated_as_active_use() -> None:
    modules = "sep5 1 0 - Live 0x0\npax 1 0 - Live 0x0\n"
    harness.require_no_profiler_state(
        {"PATH": "/usr/bin"}, ("/runtime/python", "validator.py"), modules
    )
    assert harness.loaded_unsafe_profiler_modules(modules) == ("pax", "sep5")


@pytest.mark.parametrize(
    ("environment", "command"),
    (
        ({"VTUNE_PROFILER_DIR": "/tmp/profile"}, ("validator",)),
        ({"PATH": "/usr/bin"}, ("/opt/intel/amplxe-cl", "-collect")),
        ({"PATH": "/usr/bin"}, ("/usr/bin/sep5",)),
    ),
)
def test_profiler_controls_and_commands_fail_closed(
    environment: dict[str, str], command: tuple[str, ...]
) -> None:
    with pytest.raises(harness.Glm47HarnessError, match="profiler|SEP/PAX"):
        harness.require_no_profiler_state(environment, command, "")


def test_active_open_sep_device_is_rejected(tmp_path: Path) -> None:
    process = tmp_path / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0worker.py\0")
    (process / "fd" / "4").symlink_to("/dev/sep5_0")
    assert harness.find_active_profiler_use(tmp_path) == (123,)


def test_owned_process_discovery_fails_closed_on_unreadable_environment(
    tmp_path: Path,
) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "environ").mkdir()
    assert (
        harness._owned_token_processes(  # pyright: ignore[reportPrivateUsage]
            "owner-token", tmp_path
        )
        is None
    )


def test_child_environment_is_minimal_and_disables_bytecode(tmp_path: Path) -> None:
    environment = harness.build_child_environment(
        make_config(tmp_path),
        "owner-token",
        {"PATH": "/bin", "HOME": "/root", "NCCL_DEBUG": "TRACE"},
    )
    assert environment == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "EXO_BENCHMARK_OWNER_TOKEN": "owner-token",
        "EXO_NAMESPACE": environment["EXO_NAMESPACE"],
        "HOME": "/root",
        "PATH": "/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": environment["TMPDIR"],
    }
    assert "NCCL_DEBUG" not in environment


def fake_delegated_cgroup_tree(
    tmp_path: Path, config: harness.ValidationConfig
) -> tuple[Path, Path]:
    root = tmp_path / "cgroup-root"
    unit = (
        root / harness.SYSTEMD_SLICE / harness._systemd_unit_name(config)  # pyright: ignore[reportPrivateUsage]
    )
    (unit / harness.SYSTEMD_DELEGATE_SUBGROUP).mkdir(parents=True)
    (unit / "cgroup.procs").write_bytes(b"")
    proc_self_cgroup = tmp_path / "proc-self-cgroup"
    proc_self_cgroup.write_text(
        "0::/"
        f"{harness.SYSTEMD_SLICE}/"
        f"{harness._systemd_unit_name(config)}/"  # pyright: ignore[reportPrivateUsage]
        f"{harness.SYSTEMD_DELEGATE_SUBGROUP}\n"
    )
    return root, proc_self_cgroup


def test_systemd_cgroup_path_requires_exact_delegated_subgroup(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    _root, proc_self_cgroup = fake_delegated_cgroup_tree(tmp_path, config)
    observed = harness._systemd_cgroup_path(  # pyright: ignore[reportPrivateUsage]
        config, proc_self_cgroup
    )
    assert observed.name == harness.SYSTEMD_DELEGATE_SUBGROUP
    proc_self_cgroup.write_text("0::/system.slice/foreign.service/supervisor\n")
    with pytest.raises(harness.Glm47HarnessError, match="exact delegated"):
        harness._systemd_cgroup_path(  # pyright: ignore[reportPrivateUsage]
            config, proc_self_cgroup
        )


def test_create_owned_cgroup_uses_new_sibling_of_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    root, proc_self_cgroup = fake_delegated_cgroup_tree(tmp_path, config)

    def accept_controls(_owned: harness.OwnedCgroup) -> None:
        return None

    monkeypatch.setattr(harness, "_validate_owned_cgroup_controls", accept_controls)
    owner_token = f"{config.run_id}:owner"
    owned = harness._create_owned_cgroup(  # pyright: ignore[reportPrivateUsage]
        config,
        owner_token,
        cgroup_root=root,
        proc_self_cgroup=proc_self_cgroup,
        environment={"INVOCATION_ID": "3" * 32},
    )
    try:
        assert owned.path.parent.name == harness._systemd_unit_name(  # pyright: ignore[reportPrivateUsage]
            config
        )
        assert owned.path.parent / harness.SYSTEMD_DELEGATE_SUBGROUP != owned.path
        assert stat.S_IMODE(owned.path.stat().st_mode) == 0o700
        assert owned.invocation_id == "3" * 32
    finally:
        os.close(owned.descriptor)
        os.close(owned.parent_descriptor)
        owned.path.rmdir()


def test_create_owned_cgroup_rejects_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    root, proc_self_cgroup = fake_delegated_cgroup_tree(tmp_path, config)

    def accept_controls(_owned: harness.OwnedCgroup) -> None:
        return None

    monkeypatch.setattr(harness, "_validate_owned_cgroup_controls", accept_controls)
    owner_token = f"{config.run_id}:owner"
    child_name = "validators-" + hashlib.sha256(owner_token.encode()).hexdigest()[:32]
    unit = (
        root
        / harness.SYSTEMD_SLICE
        / harness._systemd_unit_name(  # pyright: ignore[reportPrivateUsage]
            config
        )
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (unit / child_name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(harness.Glm47HarnessError, match="new direct child"):
        harness._create_owned_cgroup(  # pyright: ignore[reportPrivateUsage]
            config,
            owner_token,
            cgroup_root=root,
            proc_self_cgroup=proc_self_cgroup,
            environment={"INVOCATION_ID": "4" * 32},
        )
    assert outside.is_dir()


def test_owned_cgroup_cleanup_rejects_replaced_path(tmp_path: Path) -> None:
    owned = fake_owned_cgroup(tmp_path)
    original = owned.path.with_name("renamed-original")
    owned.path.rename(original)
    outside = tmp_path / "outside-cgroup"
    outside.mkdir()
    owned.path.symlink_to(outside, target_is_directory=True)
    assert harness.cleanup_owned_cgroup(owned, 0.1) is False
    assert (original / "cgroup.kill").read_bytes() == b""
    assert outside.is_dir()


def test_preexec_cgroup_entry_failure_never_executes_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    results = harness.ResultDirectory(result_path, descriptor)
    os.close(descriptor)
    token = f"{config.run_id}:preexec-failure"
    registry = harness.OwnershipRegistry(config, results, token)
    owned = fake_owned_cgroup(tmp_path)
    marker = tmp_path / "validator-executed"

    def fail_entry(_descriptor: int) -> None:
        raise OSError("synthetic cgroup attach failure")

    monkeypatch.setattr(harness, "_enter_owned_cgroup", fail_entry)
    try:
        outcome = harness.run_owned_command(
            config,
            results,
            registry,
            owned,
            harness.SignalLatch(),
            name="preexec-failure",
            command=(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ),
            timeout_seconds=5,
            environment={
                "EXO_BENCHMARK_OWNER_TOKEN": token,
                "EXO_NAMESPACE": config.namespace,
                "PATH": "/usr/bin:/bin",
            },
        )
        assert outcome.error is not None
        assert "preexec_fn" in outcome.error
        assert not marker.exists()
    finally:
        close_fake_owned_cgroup(owned)
        results.close()


def test_commands_use_separate_immutable_deployments_and_exact_resources(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    identity = deployment(tmp_path)
    process_spec = tmp_path / "result" / "process.json"
    kernel = tmp_path / "result" / "kernel.json"
    model = tmp_path / "result" / "model.json"
    generator = harness.build_generator_command(config, identity, process_spec)
    kernel_command = harness.build_kernel_command(config, identity, kernel)
    model_command = harness.build_model_command(
        config,
        identity,
        process_spec=process_spec,
        process_spec_sha256="e" * 64,
        kernel_receipt=kernel,
        kernel_receipt_sha256="f" * 64,
        model_output=model,
    )
    assert generator[1] == str(
        tmp_path
        / "deployment/orchestrator/scripts/create_sglang_kt_glm47_validation_process_spec.py"
    )
    assert kernel_command[:5] == (
        "/usr/bin/numactl",
        "--physcpubind",
        "0,1,2,3",
        "--membind",
        "0",
    )
    assert (
        "deployment/orchestrator/scripts/validate_sglang_kt_runtime.py"
        in kernel_command[6]
    )
    assert (
        "deployment/validator/scripts/validate_sglang_kt_glm47_model.py"
        in model_command[1]
    )
    assert "PYTHONDONTWRITEBYTECODE" not in " ".join(generator)


def test_descriptor_anchored_scratch_cleanup_removes_ipc_files_dirs_and_links(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "scratch-root"
    parent.mkdir()
    scratch = parent / "run"
    scratch.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    os.mkfifo(scratch / "worker.fifo")
    assert harness._scratch_entry_is_unlinkable(  # pyright: ignore[reportPrivateUsage]
        stat.S_IFSOCK | 0o600
    )
    assert harness._scratch_entry_is_unlinkable(  # pyright: ignore[reportPrivateUsage]
        stat.S_IFIFO | 0o600
    )
    nested = scratch / "nested"
    nested.mkdir()
    (nested / "cache").write_text("temporary")
    (nested / "outside-link").symlink_to(outside)
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    observed = os.fstat(descriptor)
    owned = harness.OwnedScratchDirectory(
        scratch,
        parent_descriptor,
        descriptor,
        observed.st_dev,
        observed.st_ino,
    )
    assert harness.cleanup_owned_scratch(owned) is True
    assert not scratch.exists()
    assert outside.read_text() == "preserve"


def test_real_successful_subprocess_is_reaped_and_cleanup_confirms(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    results = harness.ResultDirectory(result_path, descriptor)
    os.close(descriptor)
    token = f"{config.run_id}:test"
    registry = harness.OwnershipRegistry(config, results, token)
    owned_cgroup = fake_owned_cgroup(tmp_path)
    environment = {
        "EXO_BENCHMARK_OWNER_TOKEN": token,
        "EXO_NAMESPACE": config.namespace,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin",
    }
    try:
        outcome = harness.run_owned_command(
            config,
            results,
            registry,
            owned_cgroup,
            harness.SignalLatch(),
            name="success",
            command=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys, time; "
                    f"sys.exit(91) if Path({str(owned_cgroup.path / 'cgroup.procs')!r})"
                    ".read_text() != '0\\n' else time.sleep(0.1)"
                ),
            ),
            timeout_seconds=5,
            environment=environment,
        )
        assert outcome.return_code == 0
        assert outcome.cleanup_succeeded is True
        assert len(registry.processes) == 1
        assert (owned_cgroup.path / "cgroup.procs").read_bytes() == b"0\n"
        assert (owned_cgroup.path / "cgroup.kill").read_bytes() == b"1\n"
    finally:
        close_fake_owned_cgroup(owned_cgroup)
        results.close()


def test_cleanup_kills_owned_process_when_receipt_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = config_payload(tmp_path)
    timeouts = payload["timeouts"]
    assert isinstance(timeouts, dict)
    timeouts["cleanup_seconds"] = 1.0
    config = harness.ValidationConfig.model_validate_json(json.dumps(payload))
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    results = harness.ResultDirectory(result_path, descriptor)
    os.close(descriptor)
    token = f"{config.run_id}:publication-failure"
    registry = harness.OwnershipRegistry(config, results, token)
    environment = {
        "EXO_BENCHMARK_OWNER_TOKEN": token,
        "EXO_NAMESPACE": config.namespace,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin",
    }
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import time; print('READY', flush=True); time.sleep(60)",
        ),
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"

    def fail_publication(_process: harness.OwnedProcess) -> None:
        raise OSError("synthetic receipt failure")

    monkeypatch.setattr(registry, "register", fail_publication)
    try:
        assert not harness.cleanup_all_owned_processes(
            config, registry, str(results.path_for("cleanup.stderr.log"))
        )
        assert not Path(f"/proc/{process.pid}").exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, 9)
        process.wait()
        results.close()


def test_timeout_cleanup_finds_and_reaps_nested_session(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    timeouts = payload["timeouts"]
    assert isinstance(timeouts, dict)
    timeouts["cleanup_seconds"] = 1.0
    config = harness.ValidationConfig.model_validate_json(json.dumps(payload))
    result_path = Path(config.result_directory)
    result_path.mkdir(parents=True)
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    results = harness.ResultDirectory(result_path, descriptor)
    os.close(descriptor)
    token = f"{config.run_id}:nested-test"
    registry = harness.OwnershipRegistry(config, results, token)
    owned_cgroup = fake_owned_cgroup(tmp_path)
    environment = {
        "EXO_BENCHMARK_OWNER_TOKEN": token,
        "EXO_NAMESPACE": config.namespace,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": "/usr/bin:/bin",
    }
    previous_subreaper_state = harness.child_subreaper_enabled()
    spawn_on_terminate = (
        "import signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: ("
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], start_new_session=True), "
        "time.sleep(0.05), sys.exit(0))); "
        "print('READY', flush=True); time.sleep(60)"
    )
    outer_program = (
        "import subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {spawn_on_terminate!r}], "
        "start_new_session=True, stdout=subprocess.PIPE, text=True); "
        "assert child.stdout is not None; "
        "assert child.stdout.readline().strip() == 'READY'; "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    try:
        harness.set_child_subreaper(True)
        outcome = harness.run_owned_command(
            config,
            results,
            registry,
            owned_cgroup,
            harness.SignalLatch(),
            name="nested-timeout",
            command=(
                sys.executable,
                "-c",
                outer_program,
            ),
            timeout_seconds=1.0,
            environment=environment,
        )
        assert outcome.error is not None
        assert "exceeded its timeout" in outcome.error
        nested_pid = int(results.read_bytes("nested-timeout.stdout.log").strip())
        assert Path(f"/proc/{nested_pid}").exists()
        assert harness.cleanup_all_owned_processes(
            config, registry, str(results.path_for("nested-timeout.stderr.log"))
        )
        assert len(registry.processes) >= 3
        assert not Path(f"/proc/{nested_pid}").exists()
    finally:
        try:
            harness.cleanup_all_owned_processes(
                config, registry, str(results.path_for("nested-timeout.stderr.log"))
            )
        finally:
            try:
                harness.set_child_subreaper(previous_subreaper_state)
            finally:
                try:
                    close_fake_owned_cgroup(owned_cgroup)
                finally:
                    results.close()


def test_run_validation_kernel_success_is_completed_but_not_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, phase="kernel")
    identity = deployment(tmp_path)
    preflight: harness.JsonObject = {
        "schema_version": 1,
        "status": "passed",
        "profiler": "none",
    }
    install_run_validation_prerequisites(monkeypatch, identity, preflight)

    def run_kernel_pipeline(
        _config: harness.ValidationConfig,
        _deployment: harness.DeploymentIdentity,
        _results: harness.ResultDirectory,
        _registry: harness.OwnershipRegistry,
        _owned_cgroup: harness.OwnedCgroup,
        _latch: harness.SignalLatch,
        _environment: Mapping[str, str],
        evidence: harness.JsonObject,
        outcomes: list[harness.CommandOutcome],
    ) -> None:
        evidence["generator"] = {
            "schema_version": 1,
            "process_spec_sha256": "1" * 64,
        }
        artifacts = cast(harness.JsonObject, evidence["artifacts"])
        artifacts["process-spec.json"] = "2" * 64
        artifacts["kernel-runtime-validation-receipt.json"] = "3" * 64
        outcomes.extend(
            (
                command_outcome("process-spec-generator"),
                command_outcome("kernel-validator"),
            )
        )

    monkeypatch.setattr(harness, "run_pipeline", run_kernel_pipeline)
    results = open_results(config)
    try:
        result = harness.run_validation(
            config, identity, results, harness.SignalLatch()
        )
        assert result["status"] == "completed"
        assert result["completed_normally"] is True
        assert result["cleanup_succeeded"] is True
        assert result["reportable"] is False
        assert result["model_checkpoint_verified"] is False
        assert result["preflight"] == preflight
        assert cast(harness.JsonObject, result["pipeline"])["model_validator"] is None
        assert command_names(result) == (
            "process-spec-generator",
            "kernel-validator",
        )
        assert_persisted_result(config, results, result)
    finally:
        results.close()


def test_run_validation_model_success_verifies_checkpoint_and_is_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, phase="cpu_control")
    identity = deployment(tmp_path)
    preflight: harness.JsonObject = {
        "schema_version": 1,
        "status": "passed",
        "profiler": "none",
    }
    install_run_validation_prerequisites(monkeypatch, identity, preflight)

    def run_model_pipeline(
        _config: harness.ValidationConfig,
        _deployment: harness.DeploymentIdentity,
        _results: harness.ResultDirectory,
        _registry: harness.OwnershipRegistry,
        _owned_cgroup: harness.OwnedCgroup,
        _latch: harness.SignalLatch,
        _environment: Mapping[str, str],
        evidence: harness.JsonObject,
        outcomes: list[harness.CommandOutcome],
    ) -> None:
        evidence["generator"] = {
            "schema_version": 1,
            "process_spec_sha256": "4" * 64,
        }
        evidence["model_validator"] = {
            "schema_version": 1,
            "status": "passed",
            "profiler": "none",
        }
        artifacts = cast(harness.JsonObject, evidence["artifacts"])
        artifacts["process-spec.json"] = "5" * 64
        artifacts["kernel-runtime-validation-receipt.json"] = "6" * 64
        artifacts["model-runtime-validation-receipt.json"] = "7" * 64
        outcomes.extend(
            (
                command_outcome("process-spec-generator"),
                command_outcome("kernel-validator"),
                command_outcome("model-validator"),
            )
        )

    monkeypatch.setattr(harness, "run_pipeline", run_model_pipeline)
    results = open_results(config)
    try:
        result = harness.run_validation(
            config, identity, results, harness.SignalLatch()
        )
        assert result["status"] == "completed"
        assert result["completed_normally"] is True
        assert result["cleanup_succeeded"] is True
        assert result["reportable"] is True
        assert result["model_checkpoint_verified"] is True
        assert result["preflight"] == preflight
        assert cast(harness.JsonObject, result["pipeline"])["model_validator"] == {
            "schema_version": 1,
            "status": "passed",
            "profiler": "none",
        }
        assert command_names(result) == (
            "process-spec-generator",
            "kernel-validator",
            "model-validator",
        )
        assert_persisted_result(config, results, result)
    finally:
        results.close()


def test_run_validation_cgroup_cleanup_failure_is_not_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, phase="cpu_control")
    identity = deployment(tmp_path)
    preflight: harness.JsonObject = {"schema_version": 1, "status": "passed"}
    install_run_validation_prerequisites(monkeypatch, identity, preflight)

    def run_model_pipeline(
        _config: harness.ValidationConfig,
        _deployment: harness.DeploymentIdentity,
        _results: harness.ResultDirectory,
        _registry: harness.OwnershipRegistry,
        _owned_cgroup: harness.OwnedCgroup,
        _latch: harness.SignalLatch,
        _environment: Mapping[str, str],
        evidence: harness.JsonObject,
        outcomes: list[harness.CommandOutcome],
    ) -> None:
        evidence["model_validator"] = {"schema_version": 1, "status": "passed"}
        outcomes.append(command_outcome("model-validator"))

    cleanup_called = False

    def fail_cgroup_cleanup(
        owned: harness.OwnedCgroup, _timeout_seconds: float
    ) -> bool:
        nonlocal cleanup_called
        cleanup_called = True
        close_fake_owned_cgroup(owned)
        return False

    monkeypatch.setattr(harness, "run_pipeline", run_model_pipeline)
    monkeypatch.setattr(harness, "cleanup_owned_cgroup", fail_cgroup_cleanup)
    results = open_results(config)
    try:
        result = harness.run_validation(
            config, identity, results, harness.SignalLatch()
        )
        assert cleanup_called is True
        assert result["status"] == "cleanup_failed"
        assert result["completed_normally"] is True
        assert result["cleanup_succeeded"] is False
        assert result["reportable"] is False
        assert result["model_checkpoint_verified"] is False
        assert result["cleanup_errors"] == [
            "owned validator cgroup cleanup was not proven complete"
        ]
        assert_persisted_result(config, results, result)
    finally:
        results.close()


def test_process_cleanup_exception_still_attempts_cgroup_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, phase="kernel")
    identity = deployment(tmp_path)
    preflight: harness.JsonObject = {"schema_version": 1, "status": "passed"}
    install_run_validation_prerequisites(monkeypatch, identity, preflight)

    def run_kernel_pipeline(
        _config: harness.ValidationConfig,
        _deployment: harness.DeploymentIdentity,
        _results: harness.ResultDirectory,
        _registry: harness.OwnershipRegistry,
        _owned_cgroup: harness.OwnedCgroup,
        _latch: harness.SignalLatch,
        _environment: Mapping[str, str],
        _evidence: harness.JsonObject,
        outcomes: list[harness.CommandOutcome],
    ) -> None:
        outcomes.append(command_outcome("kernel-validator"))

    def fail_process_cleanup(
        _config: harness.ValidationConfig,
        _registry: harness.OwnershipRegistry,
        _default_log_path: str,
    ) -> bool:
        raise OSError("synthetic process cleanup failure")

    cgroup_cleanup_called = False

    def confirm_cgroup_cleanup(
        owned: harness.OwnedCgroup, _timeout_seconds: float
    ) -> bool:
        nonlocal cgroup_cleanup_called
        cgroup_cleanup_called = True
        close_fake_owned_cgroup(owned)
        return True

    monkeypatch.setattr(harness, "run_pipeline", run_kernel_pipeline)
    monkeypatch.setattr(harness, "cleanup_all_owned_processes", fail_process_cleanup)
    monkeypatch.setattr(harness, "cleanup_owned_cgroup", confirm_cgroup_cleanup)
    results = open_results(config)
    try:
        result = harness.run_validation(
            config, identity, results, harness.SignalLatch()
        )
        assert cgroup_cleanup_called is True
        assert result["status"] == "cleanup_failed"
        assert result["cleanup_succeeded"] is False
        cleanup_errors = cast(list[harness.JsonValue], result["cleanup_errors"])
        assert cleanup_errors == [
            "owned process cleanup raised OSError: synthetic process cleanup failure"
        ]
        assert_persisted_result(config, results, result)
    finally:
        results.close()


def test_run_validation_mid_pipeline_failure_persists_partial_nonreportable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, phase="hybrid", resident_gpu_experts=1)
    identity = deployment(tmp_path)
    preflight: harness.JsonObject = {
        "schema_version": 1,
        "status": "passed",
        "profiler": "none",
    }
    install_run_validation_prerequisites(monkeypatch, identity, preflight)

    def fail_mid_pipeline(
        _config: harness.ValidationConfig,
        _deployment: harness.DeploymentIdentity,
        _results: harness.ResultDirectory,
        _registry: harness.OwnershipRegistry,
        _owned_cgroup: harness.OwnedCgroup,
        _latch: harness.SignalLatch,
        _environment: Mapping[str, str],
        evidence: harness.JsonObject,
        outcomes: list[harness.CommandOutcome],
    ) -> None:
        evidence["generator"] = {
            "schema_version": 1,
            "process_spec_sha256": "8" * 64,
        }
        artifacts = cast(harness.JsonObject, evidence["artifacts"])
        artifacts["process-spec.json"] = "9" * 64
        outcomes.extend(
            (
                command_outcome("process-spec-generator"),
                command_outcome(
                    "kernel-validator",
                    return_code=1,
                    error="synthetic command failure",
                ),
            )
        )
        raise harness.Glm47HarnessError("synthetic mid-pipeline failure")

    monkeypatch.setattr(harness, "run_pipeline", fail_mid_pipeline)
    results = open_results(config)
    try:
        result = harness.run_validation(
            config, identity, results, harness.SignalLatch()
        )
        assert result["status"] == "validation_failed"
        assert result["completed_normally"] is False
        assert result["cleanup_succeeded"] is True
        assert result["reportable"] is False
        assert result["model_checkpoint_verified"] is False
        assert result["error"] == ("Glm47HarnessError: synthetic mid-pipeline failure")
        assert result["preflight"] == preflight
        pipeline = cast(harness.JsonObject, result["pipeline"])
        assert pipeline["model_validator"] is None
        assert pipeline["artifacts"] == {"process-spec.json": "9" * 64}
        assert command_names(result) == (
            "process-spec-generator",
            "kernel-validator",
        )
        assert_persisted_result(config, results, result)
    finally:
        results.close()


def _make_runtime_chain(tmp_path: Path) -> tuple[Path, str, list[str]]:
    root = tmp_path / "runtime"
    root.mkdir()
    final = root / "base-python"
    final.write_bytes(b"fake executable")
    final.chmod(0o755)
    intermediate = root / "python3.12"
    intermediate.symlink_to(final)
    lexical = root / "python"
    lexical.symlink_to("python3.12")
    return (
        lexical,
        hashlib.sha256(final.read_bytes()).hexdigest(),
        [
            "python3.12",
            str(final),
        ],
    )


def test_prepare_lease_builds_exact_no_bytecode_deployments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lexical, runtime_sha, chain = _make_runtime_chain(tmp_path)
    build_receipt = tmp_path / "build-receipt.json"
    build_receipt.write_text("{}\n")
    model = tmp_path / "model"
    model.mkdir()
    result_root = tmp_path / "results"
    result_root.mkdir()
    deployment_root = tmp_path / "immutable-deployment"
    payload = config_payload(
        tmp_path,
        runtime_path=str(lexical),
        runtime_sha256=runtime_sha,
        runtime_chain=chain,
    )
    run_id = payload["run_id"]
    assert isinstance(run_id, str)
    payload["result_directory"] = str(result_root / run_id)
    payload["source"] = {
        "repository": str(Path(__file__).resolve().parents[2]),
        "deployment_root": str(deployment_root),
    }
    payload["model_contract"] = {
        "path": str(
            deployment_root / "orchestrator" / harness.MODEL_CONTRACT_RELATIVE_PATH
        ),
        "sha256": harness.GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    }
    payload["build_receipt"] = {
        "path": str(build_receipt),
        "sha256": hashlib.sha256(build_receipt.read_bytes()).hexdigest(),
    }
    payload["model_path"] = str(model)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    source_identity = harness.SourceIdentity("1" * 40, {})

    def stable_source_identity(_path: Path) -> harness.SourceIdentity:
        return source_identity

    monkeypatch.setattr(harness, "read_source_identity", stable_source_identity)
    metadata_path = tmp_path / "metadata.json"
    prepared = harness.prepare_lease(
        config_path=config_path,
        metadata_output=metadata_path,
        owner="codex:/root",
        purpose="glm47-cpu-control",
        expected_duration_seconds=7200,
        cleanup_grace_seconds=300,
        heartbeat_seconds=30,
        lease_path=tmp_path / "lease.json",
        lock_path=tmp_path / "lease.lock",
        result_root=result_root,
    )
    validator_root = deployment_root / "validator"
    observed_validator_files = {
        path.relative_to(validator_root)
        for path in validator_root.rglob("*")
        if path.is_file()
    }
    assert observed_validator_files == {
        Path(path) for path in harness.MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    }
    assert not tuple(deployment_root.rglob("__pycache__"))
    assert not tuple(deployment_root.rglob("*.pyc"))
    assert prepared.child_argv[:2] == (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
    )
    assert prepared.benchmark_lease_argv[:7] == (
        "/usr/bin/systemd-run",
        "--system",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
        "--service-type=exec",
    )
    assert f"--unit={prepared.systemd_unit_name}" in prepared.benchmark_lease_argv
    working_directory_argument = f"--working-directory={deployment_root}"
    assert working_directory_argument in prepared.benchmark_lease_argv
    assert "--property=Delegate=yes" in prepared.benchmark_lease_argv
    assert "--property=DelegateSubgroup=supervisor" in prepared.benchmark_lease_argv
    containment = cast(dict[str, object], prepared.metadata["containment_contract"])
    assert containment == {
        "schema": "systemd_delegated_cgroup_v1",
        "systemd_unit_name": prepared.systemd_unit_name,
        "systemd_slice": "system.slice",
        "delegate_subgroup": "supervisor",
        "validator_cgroup_layout": "delegated-sibling-v1",
        "attach_method": "preexec-cgroup.procs-v1",
        "cleanup_method": "cgroup.kill-v1",
    }
    command_separator = prepared.benchmark_lease_argv.index("--")
    assert prepared.benchmark_lease_argv.index(working_directory_argument) < (
        command_separator
    )
    assert prepared.benchmark_lease_argv[
        command_separator + 1 : command_separator + 3
    ] == (
        "/usr/bin/env",
        "PYTHONDONTWRITEBYTECODE=1",
    )
    deployed_harness = (
        deployment_root / "orchestrator/scripts/run_sglang_kt_glm47_validation.py"
    )
    help_result = subprocess.run(
        (sys.executable, str(deployed_harness), "--help"),
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert help_result.returncode == 0, help_result.stderr
    assert not tuple(deployment_root.rglob("__pycache__"))
    assert metadata_path.is_file()
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o444
    assert harness.load_deployment_identity(deployment_root) == prepared.deployment
