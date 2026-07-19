from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

import scripts.benchmark_lease as benchmark_lease_module
from scripts.benchmark_lease import (
    BenchmarkLease,
    LeaseBusyError,
    LeaseError,
    ManagedSignalState,
    StaleLeaseError,
    atomic_write_json,
    read_json_object,
    run_managed_command,
    run_with_lease,
    validate_ports,
    validate_run_metadata,
)


def make_metadata(
    *,
    generated_at: datetime | None = None,
    run_id: str = "test-run",
    namespace: str | None = None,
    reserved_ports: Sequence[int] = (53001, 53002),
    result_directory: Path = Path("/tmp/exo-test-results"),
    command: Sequence[str] = (sys.executable, "-c", "raise SystemExit(0)"),
) -> dict[str, object]:
    commit = "a" * 40
    return {
        "schema_version": benchmark_lease_module.METADATA_SCHEMA_VERSION,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "run_id": run_id,
        "namespace": namespace or f"namespace-{run_id}",
        "reserved_ports": list(reserved_ports),
        "result_directory": str(result_directory),
        "command": list(command),
        "git": {
            "commit": commit,
            "dirty": False,
            "dirty_file_hashes": {},
        },
        "hosts": ["dwagon", "fwuff"],
        "models": [
            {
                "model_id": "test/model",
                "revision": "b" * 40,
                "paths": {
                    "dwagon": "/models/test-model",
                    "fwuff": "/mnt/models/test-model",
                },
            }
        ],
        "gpu_bindings": {
            "dwagon": [{"uuid": "GPU-dwagon", "pci_address": "0000:01:00.0"}],
            "fwuff": [{"uuid": "GPU-fwuff", "pci_address": "0000:02:00.0"}],
        },
        "cpu_bindings": {
            "dwagon": {
                "cpu_set": "0-7",
                "numa_nodes": [0],
                "memory_policy": "bind:0",
            },
            "fwuff": {
                "cpu_set": "8-15",
                "numa_nodes": [1],
                "memory_policy": "bind:1",
            },
        },
        "hca_bindings": {
            "dwagon": [{"device": "mlx4_0", "port": 1, "ip_address": "10.0.0.1"}],
            "fwuff": [{"device": "mlx4_0", "port": 1, "ip_address": "10.0.0.2"}],
        },
        "source_deployments": {
            "dwagon": {
                "path": "/root/exo",
                "commit": commit,
                "dirty_file_hashes": {},
            },
            "fwuff": {
                "path": "/opt/exo",
                "commit": commit,
                "dirty_file_hashes": {},
            },
        },
        "owner_pids": {"dwagon": [], "fwuff": []},
    }


def make_local_metadata(
    *,
    generated_at: datetime | None = None,
    run_id: str = "test-run",
    namespace: str | None = None,
    reserved_ports: Sequence[int] = (53001, 53002),
    result_directory: Path = Path("/tmp/exo-test-results"),
    command: Sequence[str] = (sys.executable, "-c", "raise SystemExit(0)"),
) -> dict[str, object]:
    metadata = make_metadata(
        generated_at=generated_at,
        run_id=run_id,
        namespace=namespace,
        reserved_ports=reserved_ports,
        result_directory=result_directory,
        command=command,
    )
    metadata["hosts"] = ["dwagon"]
    models = cast(list[dict[str, object]], metadata["models"])
    cast(dict[str, object], models[0]["paths"]).pop("fwuff")
    for section_name in (
        "gpu_bindings",
        "cpu_bindings",
        "hca_bindings",
        "source_deployments",
        "owner_pids",
    ):
        cast(dict[str, object], metadata[section_name]).pop("fwuff")
    return metadata


def make_lease(
    tmp_path: Path,
    *,
    run_id: str = "test-run",
    metadata: dict[str, object] | None = None,
    allow_unconfirmed_cleanup_for_tests: bool = True,
    owner: str = "codex:test",
    purpose: str = "unit-test",
    cleanup_grace_seconds: float = 0.5,
) -> BenchmarkLease:
    namespace = f"namespace-{run_id}"
    ports = (53001, 53002)
    command = (sys.executable, "-c", "raise SystemExit(0)")
    result_directory = tmp_path / "results" / run_id
    bound_metadata = metadata or make_metadata()
    bound_metadata.update(
        {
            "run_id": run_id,
            "namespace": namespace,
            "reserved_ports": list(ports),
            "result_directory": str(result_directory),
            "command": list(command),
        }
    )
    lease = BenchmarkLease(
        lock_path=tmp_path / "benchmark.lock",
        lease_path=tmp_path / "lease.json",
        result_directory=result_directory,
        owner=owner,
        purpose=purpose,
        run_id=run_id,
        namespace=namespace,
        ports=ports,
        command=command,
        metadata=bound_metadata,
        heartbeat_seconds=0.01,
        expected_duration_seconds=60,
        cleanup_grace_seconds=cleanup_grace_seconds,
        _allow_unconfirmed_cleanup_for_tests=(allow_unconfirmed_cleanup_for_tests),
    )
    return lease


def set_lease_command(lease: BenchmarkLease, command: Sequence[str]) -> None:
    lease.command = tuple(command)
    lease.metadata["command"] = list(command)


def make_runtime_metadata(
    lease: BenchmarkLease,
    *,
    owner_token: str = "owner-token",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": lease.run_id,
        "namespace": lease.namespace,
        "owner_token": owner_token,
        "owned_processes": [
            {
                "host_name": "fwuff",
                "pid": 123,
                "process_group_id": 123,
                "start_time_ticks": 456,
                "owner_token": owner_token,
                "namespace": lease.namespace,
                "transport_pid": 789,
                "log_path": "/tmp/fwuff-exo.log",
            }
        ],
    }


def make_containment_metadata(run_id: str) -> dict[str, object]:
    namespace = f"namespace-{run_id}"
    metadata = make_metadata(run_id=run_id, namespace=namespace)
    binding = hashlib.sha256(
        f"glm47-validation-v1\0{run_id}\0{namespace}".encode()
    ).hexdigest()[:32]
    metadata["containment_contract"] = {
        "schema": "systemd_delegated_cgroup_v1",
        "systemd_unit_name": f"exo-glm47-{binding}.service",
        "systemd_slice": "system.slice",
        "delegate_subgroup": "supervisor",
        "validator_cgroup_layout": "delegated-sibling-v1",
        "attach_method": "preexec-cgroup.procs-v1",
        "cleanup_method": "cgroup.kill-v1",
    }
    return metadata


def make_runtime_containment(
    lease: BenchmarkLease,
    *,
    owner_token: str,
    invocation_id: str,
) -> dict[str, object]:
    contract = cast(dict[str, object], lease.metadata["containment_contract"])
    leaf = "validators-" + hashlib.sha256(owner_token.encode()).hexdigest()[:32]
    return {
        "schema_version": 1,
        "path": (f"/sys/fs/cgroup/system.slice/{contract['systemd_unit_name']}/{leaf}"),
        "device": 27,
        "inode": 12345,
        "owner_uid": os.geteuid(),
        "invocation_id": invocation_id,
        "systemd_unit_name": contract["systemd_unit_name"],
        "attach_method": contract["attach_method"],
        "cleanup_method": contract["cleanup_method"],
    }


def append_owned_process(
    runtime_metadata: dict[str, object], lease: BenchmarkLease
) -> None:
    owner_token = cast(str, runtime_metadata["owner_token"])
    processes = cast(list[dict[str, object]], runtime_metadata["owned_processes"])
    processes.append(
        {
            "host_name": "dwagon",
            "pid": 321,
            "process_group_id": 321,
            "start_time_ticks": 654,
            "owner_token": owner_token,
            "namespace": lease.namespace,
            "transport_pid": 987,
            "log_path": "/tmp/dwagon-exo.log",
        }
    )


def make_benchmark_result(
    lease: BenchmarkLease,
    *,
    cleanup_succeeded: bool,
    owned_processes: Sequence[object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": benchmark_lease_module.BENCHMARK_RESULT_SCHEMA_VERSION,
        "run_id": lease.run_id,
        "namespace": lease.namespace,
        "cleanup_succeeded": cleanup_succeeded,
    }
    if owned_processes is not None:
        result["owned_processes"] = list(owned_processes)
    return result


def write_benchmark_result(
    lease: BenchmarkLease,
    *,
    cleanup_succeeded: bool,
    owned_processes: Sequence[object] | None = None,
) -> None:
    atomic_write_json(
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME,
        make_benchmark_result(
            lease,
            cleanup_succeeded=cleanup_succeeded,
            owned_processes=owned_processes,
        ),
    )


def test_validate_ports_rejects_duplicates_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_ports(())
    with pytest.raises(ValueError, match="unique"):
        validate_ports((53001, 53001))
    with pytest.raises(ValueError, match="between"):
        validate_ports((0,))


def test_metadata_requires_complete_fresh_reproducibility_context() -> None:
    with pytest.raises(LeaseError, match="schema_version"):
        validate_run_metadata({})

    stale_metadata = make_metadata(
        generated_at=datetime.now(timezone.utc)
        - benchmark_lease_module.METADATA_MAX_AGE
        - timedelta(seconds=1)
    )
    with pytest.raises(LeaseError, match="older than"):
        validate_run_metadata(stale_metadata)

    incomplete_metadata = make_metadata()
    cast(dict[str, object], incomplete_metadata["hca_bindings"]).pop("fwuff")
    with pytest.raises(LeaseError, match="hca_bindings.*exactly"):
        validate_run_metadata(incomplete_metadata)

    mutable_revision_metadata = make_metadata()
    models = cast(list[dict[str, object]], mutable_revision_metadata["models"])
    models[0]["revision"] = "main"
    with pytest.raises(LeaseError, match="revision must be an exact commit hash"):
        validate_run_metadata(mutable_revision_metadata)

    abbreviated_commit_metadata = make_metadata()
    git_metadata = cast(dict[str, object], abbreviated_commit_metadata["git"])
    git_metadata["commit"] = "a" * 39
    with pytest.raises(LeaseError, match="exact commit hash"):
        validate_run_metadata(abbreviated_commit_metadata)

    dirty_mismatch_metadata = make_metadata()
    git_metadata = cast(dict[str, object], dirty_mismatch_metadata["git"])
    git_metadata["dirty_file_hashes"] = {"changed.py": "c" * 64}
    with pytest.raises(LeaseError, match="dirty must match"):
        validate_run_metadata(dirty_mismatch_metadata)

    invalid_digest_metadata = make_metadata()
    git_metadata = cast(dict[str, object], invalid_digest_metadata["git"])
    git_metadata["dirty"] = True
    git_metadata["dirty_file_hashes"] = {"changed.py": "c" * 63}
    with pytest.raises(LeaseError, match="SHA|digest"):
        validate_run_metadata(invalid_digest_metadata)

    mismatched_source_metadata = make_metadata()
    deployments = cast(
        dict[str, dict[str, object]], mismatched_source_metadata["source_deployments"]
    )
    deployments["fwuff"]["commit"] = "d" * 40
    with pytest.raises(LeaseError, match="must match git.commit"):
        validate_run_metadata(mismatched_source_metadata)

    mismatched_dirty_source_metadata = make_metadata()
    git_metadata = cast(dict[str, object], mismatched_dirty_source_metadata["git"])
    git_metadata["dirty"] = True
    git_metadata["dirty_file_hashes"] = {"changed.py": "c" * 64}
    deployments = cast(
        dict[str, dict[str, object]],
        mismatched_dirty_source_metadata["source_deployments"],
    )
    for deployment in deployments.values():
        deployment["dirty_file_hashes"] = {"other.py": "d" * 64}
    with pytest.raises(LeaseError, match="must match git.dirty_file_hashes"):
        validate_run_metadata(mismatched_dirty_source_metadata)


def test_metadata_validates_exact_optional_containment_contract() -> None:
    metadata = make_containment_metadata("containment-contract")
    assert validate_run_metadata(metadata) == metadata
    contract = cast(dict[str, object], metadata["containment_contract"])
    contract["cleanup_method"] = "unbound-cleanup"
    with pytest.raises(LeaseError, match="containment_contract"):
        validate_run_metadata(metadata)


def test_metadata_accepts_exact_raw_verbs_gid_as_hca_identity() -> None:
    metadata = make_metadata()
    bindings = cast(dict[str, list[dict[str, object]]], metadata["hca_bindings"])
    bindings["dwagon"] = [
        {"device": "mlx4_0", "port": 1, "gid": "fe80::10:e000:166:3a19"}
    ]

    validated = validate_run_metadata(metadata)

    validated_bindings = cast(
        dict[str, list[dict[str, object]]], validated["hca_bindings"]
    )
    assert validated_bindings["dwagon"][0]["gid"] == "fe80::10:e000:166:3a19"


@pytest.mark.parametrize(
    "binding",
    [
        {"device": "mlx4_0", "port": 1},
        {
            "device": "mlx4_0",
            "port": 1,
            "ip_address": "10.0.0.1",
            "gid": "fe80::1",
        },
        {"device": "mlx4_0", "port": 1, "gid": "not-a-gid"},
        {"device": "mlx4_0", "port": 1, "gid": "0.0.0.1"},
        {"device": "mlx4_0", "port": 1, "gid": "::"},
        {"device": "mlx4_0", "port": 1, "gid": "fe80::"},
        {"device": "mlx4_0", "port": 1, "ip_address": "not-an-address"},
    ],
)
def test_metadata_rejects_ambiguous_or_invalid_hca_identity(
    binding: dict[str, object],
) -> None:
    metadata = make_metadata()
    bindings = cast(dict[str, list[dict[str, object]]], metadata["hca_bindings"])
    bindings["dwagon"] = [binding]

    with pytest.raises(LeaseError, match="ip_address|gid|IP address"):
        validate_run_metadata(metadata)

    extra_host_metadata = make_metadata()
    gpu_bindings = cast(dict[str, object], extra_host_metadata["gpu_bindings"])
    gpu_bindings["unexpected"] = []
    with pytest.raises(LeaseError, match="gpu_bindings.*exactly"):
        validate_run_metadata(extra_host_metadata)


def test_owner_and_purpose_must_not_be_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owner"):
        make_lease(tmp_path, owner=" ")
    with pytest.raises(ValueError, match="purpose"):
        make_lease(tmp_path, purpose="")


def test_cleanup_grace_must_be_positive_and_finite(tmp_path: Path) -> None:
    for invalid_grace in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            make_lease(tmp_path, cleanup_grace_seconds=invalid_grace)


def test_run_metadata_is_bound_to_the_actual_lease(tmp_path: Path) -> None:
    mismatches: dict[str, object] = {
        "run_id": "different-run",
        "namespace": "different-namespace",
        "reserved_ports": [54001],
        "result_directory": "/tmp/different-results",
        "command": [sys.executable, "-c", "raise SystemExit(9)"],
    }
    for field_name, invalid_value in mismatches.items():
        lease = make_lease(tmp_path / field_name)
        lease.metadata[field_name] = invalid_value
        with pytest.raises(LeaseError, match=field_name):
            lease.acquire()
        assert not lease.lock_path.exists()


def test_lease_paths_must_be_absolute(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    with pytest.raises(ValueError, match="lock_path"):
        BenchmarkLease(
            lock_path=Path("relative.lock"),
            lease_path=lease.lease_path,
            result_directory=lease.result_directory,
            owner=lease.owner,
            purpose=lease.purpose,
            run_id=lease.run_id,
            namespace=lease.namespace,
            ports=lease.ports,
            command=lease.command,
            metadata=lease.metadata,
        )
    with pytest.raises(ValueError, match="lease_path"):
        BenchmarkLease(
            lock_path=lease.lock_path,
            lease_path=Path("relative-lease.json"),
            result_directory=lease.result_directory,
            owner=lease.owner,
            purpose=lease.purpose,
            run_id=lease.run_id,
            namespace=lease.namespace,
            ports=lease.ports,
            command=lease.command,
            metadata=lease.metadata,
        )
    with pytest.raises(ValueError, match="result_directory"):
        BenchmarkLease(
            lock_path=lease.lock_path,
            lease_path=lease.lease_path,
            result_directory=Path("relative-results"),
            owner=lease.owner,
            purpose=lease.purpose,
            run_id=lease.run_id,
            namespace=lease.namespace,
            ports=lease.ports,
            command=lease.command,
            metadata=lease.metadata,
        )


def test_result_directory_is_bound_to_run_id_and_lease_paths_are_distinct(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    with pytest.raises(ValueError, match="exact run_id"):
        BenchmarkLease(
            lock_path=lease.lock_path,
            lease_path=lease.lease_path,
            result_directory=lease.result_directory.parent / "wrong-run",
            owner=lease.owner,
            purpose=lease.purpose,
            run_id=lease.run_id,
            namespace=lease.namespace,
            ports=lease.ports,
            command=lease.command,
            metadata=lease.metadata,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        BenchmarkLease(
            lock_path=lease.lease_path,
            lease_path=lease.lease_path,
            result_directory=lease.result_directory,
            owner=lease.owner,
            purpose=lease.purpose,
            run_id=lease.run_id,
            namespace=lease.namespace,
            ports=lease.ports,
            command=lease.command,
            metadata=lease.metadata,
        )


def test_metadata_freshness_is_revalidated_at_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_at = datetime.now(timezone.utc)
    lease = make_lease(tmp_path, metadata=make_metadata(generated_at=generated_at))

    def stale_now() -> datetime:
        return (
            generated_at
            + benchmark_lease_module.METADATA_MAX_AGE
            + timedelta(seconds=1)
        )

    monkeypatch.setattr(benchmark_lease_module, "utc_now", stale_now)
    with pytest.raises(LeaseError, match="older than"):
        lease.acquire()
    assert not lease.lock_path.exists()


def test_stale_cli_metadata_is_rejected_before_lock_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = tmp_path / "stale-metadata.json"
    stale_metadata = make_metadata(
        generated_at=datetime.now(timezone.utc)
        - benchmark_lease_module.METADATA_MAX_AGE
        - timedelta(seconds=1)
    )
    atomic_write_json(metadata_path, stale_metadata)
    lock_path = tmp_path / "benchmark.lock"
    lease_path = tmp_path / "lease.json"

    return_code = benchmark_lease_module.main(
        (
            "--owner",
            "codex:test",
            "--purpose",
            "stale-metadata-test",
            "--run-id",
            "stale-metadata-test",
            "--namespace",
            "namespace-stale-metadata-test",
            "--port",
            "53001",
            "--metadata-json",
            str(metadata_path),
            "--lock-path",
            str(lock_path),
            "--lease-path",
            str(lease_path),
            "--result-root",
            str(tmp_path / "results"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        )
    )

    assert return_code == 75
    assert "older than" in capsys.readouterr().err
    assert not lock_path.exists()
    assert not lease_path.exists()
    assert not (tmp_path / "results").exists()


def test_cli_has_no_cleanup_confirmation_opt_out() -> None:
    assert (
        "local-only-no-cleanup-confirmation"
        not in benchmark_lease_module.build_parser().format_help()
    )


def test_private_cleanup_seam_is_unavailable_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXO_TESTS")
    with pytest.raises(ValueError, match="only under the test harness"):
        make_lease(tmp_path)


def test_lease_writes_metadata_and_removes_only_its_record(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    with lease:
        payload = read_json_object(lease.lease_path)
        assert payload["lease_id"] == lease.lease_id
        assert payload["ports"] == [53001, 53002]
        assert payload["exo_namespace"] == "namespace-test-run"
        assert payload["cleanup_grace_seconds"] == 0.5
    assert not lease.lease_path.exists()


def test_second_lease_reports_busy_owner(tmp_path: Path) -> None:
    first = make_lease(tmp_path, run_id="first")
    second = make_lease(tmp_path, run_id="second")
    with first, pytest.raises(LeaseBusyError, match="already held"):
        second.acquire()


def test_stale_metadata_is_never_overwritten(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    atomic_write_json(lease.lease_path, {"lease_id": "stale", "owner": "unknown"})
    with pytest.raises(StaleLeaseError, match="inspect it manually"):
        lease.acquire()
    assert json.loads(lease.lease_path.read_text())["lease_id"] == "stale"


def test_foreign_metadata_is_not_removed_during_release(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    atomic_write_json(lease.lease_path, {"lease_id": "foreign"})
    lease.release()
    assert json.loads(lease.lease_path.read_text())["lease_id"] == "foreign"


def test_missing_cleanup_result_fails_closed_by_default(tmp_path: Path) -> None:
    lease = make_lease(tmp_path, allow_unconfirmed_cleanup_for_tests=False)

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    lease_record = read_json_object(lease.lease_path)
    assert lease_record["manual_clearance_required"] is True
    fragment_errors = cast(dict[str, object], lease_record["fragment_errors"])
    assert "is required" in cast(
        str, fragment_errors[benchmark_lease_module.BENCHMARK_RESULT_FILENAME]
    )
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "cleanup_failed"
    assert manifest["cleanup_succeeded"] is False


def test_explicit_cleanup_confirmation_releases_default_lease(tmp_path: Path) -> None:
    lease = make_lease(tmp_path, allow_unconfirmed_cleanup_for_tests=False)
    result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    result_json = json.dumps(make_benchmark_result(lease, cleanup_succeeded=True))
    set_lease_command(
        lease,
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(result_path)!r}).write_text({result_json!r})",
        ),
    )

    assert run_with_lease(lease) == 0
    assert not lease.lease_path.exists()


def test_malformed_cleanup_result_preserves_tombstone_and_manifest(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path, allow_unconfirmed_cleanup_for_tests=False)
    result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    set_lease_command(
        lease,
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(result_path)!r}).write_text('{{')",
        ),
    )

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    assert read_json_object(lease.lease_path)["manual_clearance_required"] is True
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    fragment_errors = cast(dict[str, object], manifest["fragment_errors"])
    assert benchmark_lease_module.BENCHMARK_RESULT_FILENAME in fragment_errors
    assert manifest["cleanup_succeeded"] is False


def test_cleanup_result_io_error_preserves_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path, allow_unconfirmed_cleanup_for_tests=False)
    original_read = benchmark_lease_module.read_json_object
    original_result_read = benchmark_lease_module.read_json_object_at
    result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    set_lease_command(
        lease,
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(result_path)!r}).write_text('{{}}')",
        ),
    )

    def fail_result_read(directory_descriptor: int, filename: str) -> dict[str, object]:
        if filename == benchmark_lease_module.BENCHMARK_RESULT_FILENAME:
            raise OSError("simulated result read error")
        return original_result_read(directory_descriptor, filename)

    monkeypatch.setattr(benchmark_lease_module, "read_json_object_at", fail_result_read)
    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    lease_record = original_read(lease.lease_path)
    assert lease_record["manual_clearance_required"] is True
    manifest = original_read(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert "simulated result read error" in str(manifest["fragment_errors"])


def test_cleanup_failure_preserves_exclusion_record_and_blocks_next_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path, run_id="failed-cleanup")

    def process_group_still_exists(_process_id: int) -> bool:
        return True

    def cleanup_fails(
        _process: subprocess.Popen[bytes], *, grace_seconds: float
    ) -> bool:
        del grace_seconds
        return False

    monkeypatch.setattr(
        benchmark_lease_module, "process_group_exists", process_group_still_exists
    )
    monkeypatch.setattr(
        benchmark_lease_module,
        "terminate_owned_process_group",
        cleanup_fails,
    )

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    lease_record = read_json_object(lease.lease_path)
    assert lease_record["lease_id"] == lease.lease_id
    assert lease_record["cleanup_succeeded"] is False
    assert lease_record["manual_clearance_required"] is True
    assert "owned local process group remains" in cast(
        str, lease_record["cleanup_failure_reason"]
    )

    replacement = make_lease(tmp_path, run_id="replacement")
    with pytest.raises(StaleLeaseError, match="inspect it manually"):
        replacement.acquire()


def test_missing_tombstone_is_recreated_and_rechecked_before_unlock(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    lease.lease_path.unlink()

    lease.preserve_after_cleanup_failure("simulated unconfirmed cleanup")
    assert read_json_object(lease.lease_path)["manual_clearance_required"] is True

    lease.lease_path.unlink()
    lease.release()
    recreated = read_json_object(lease.lease_path)
    assert recreated["lease_id"] == lease.lease_id
    assert recreated["manual_clearance_required"] is True

    replacement = make_lease(tmp_path, run_id="replacement")
    with pytest.raises(StaleLeaseError):
        replacement.acquire()


def test_process_group_cleanup_exception_preserves_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)

    def process_group_still_exists(_process_id: int) -> bool:
        return True

    def cleanup_raises(
        _process: subprocess.Popen[bytes], *, grace_seconds: float
    ) -> bool:
        del grace_seconds
        raise OSError("simulated killpg error")

    monkeypatch.setattr(
        benchmark_lease_module, "process_group_exists", process_group_still_exists
    )
    monkeypatch.setattr(
        benchmark_lease_module, "terminate_owned_process_group", cleanup_raises
    )

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE
    lease_record = read_json_object(lease.lease_path)
    assert lease_record["manual_clearance_required"] is True
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert "simulated killpg error" in cast(str, manifest["error_message"])


def test_child_cleanup_failure_preserves_exclusion_record(tmp_path: Path) -> None:
    lease = make_lease(tmp_path, run_id="child-cleanup-failed")
    benchmark_result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    benchmark_result = make_benchmark_result(lease, cleanup_succeeded=False)
    set_lease_command(
        lease,
        (
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"Path({str(benchmark_result_path)!r}).write_text("
            f"{json.dumps(benchmark_result)!r})",
        ),
    )

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    lease_record = read_json_object(lease.lease_path)
    assert lease_record["manual_clearance_required"] is True
    assert "child reported cleanup failure" in cast(
        str, lease_record["cleanup_failure_reason"]
    )
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "cleanup_failed"
    assert manifest["cleanup_succeeded"] is False
    assert manifest["benchmark_result"] == benchmark_result


def test_child_fragments_are_preserved_and_folded_into_manifest(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_metadata = make_runtime_metadata(lease)
    owned_processes = cast(Sequence[object], runtime_metadata["owned_processes"])
    benchmark_result = make_benchmark_result(
        lease,
        cleanup_succeeded=True,
        owned_processes=owned_processes,
    )
    benchmark_result["samples"] = [{"latency_seconds": 1.25}]
    child_manifest = {"preflight": "passed", "sample_count": 1}
    benchmark_result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    runtime_metadata_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    manifest_path = lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    atomic_write_json(benchmark_result_path, benchmark_result)
    atomic_write_json(runtime_metadata_path, runtime_metadata)
    atomic_write_json(manifest_path, child_manifest)

    lease.update_heartbeat()
    lease_record = read_json_object(lease.lease_path)
    assert lease_record["runtime_metadata"] == runtime_metadata

    runtime_metadata_path.unlink()
    lease.update_heartbeat()
    lease_record = read_json_object(lease.lease_path)
    assert lease_record["runtime_metadata"] == runtime_metadata

    lease.write_manifest(status="completed", return_code=0, cleanup_succeeded=True)
    manifest = read_json_object(manifest_path)
    assert manifest["benchmark_result"] == benchmark_result
    assert manifest["runtime_metadata"] == runtime_metadata
    assert manifest["child_manifest"] == child_manifest
    assert read_json_object(benchmark_result_path) == benchmark_result
    assert not runtime_metadata_path.exists()
    lease.release()


def test_malformed_runtime_metadata_fails_closed_with_last_known_ownership(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path, allow_unconfirmed_cleanup_for_tests=False)
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    benchmark_result = make_benchmark_result(lease, cleanup_succeeded=True)
    set_lease_command(
        lease,
        (
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"Path({str(runtime_path)!r}).write_text('{{'); "
            f"Path({str(result_path)!r}).write_text("
            f"{json.dumps(benchmark_result)!r})",
        ),
    )

    with pytest.raises(LeaseError, match="heartbeat failed"):
        run_with_lease(lease)

    lease_record = read_json_object(lease.lease_path)
    assert lease_record["manual_clearance_required"] is True
    assert benchmark_lease_module.RUNTIME_METADATA_FILENAME in cast(
        dict[str, object], lease_record["fragment_errors"]
    )
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["benchmark_result"] == benchmark_result
    assert manifest["cleanup_succeeded"] is False


def test_runtime_metadata_rejects_mismatched_ownership_token(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_metadata = make_runtime_metadata(lease)
    processes = cast(list[dict[str, object]], runtime_metadata["owned_processes"])
    processes[0]["owner_token"] = "foreign-token"
    atomic_write_json(
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME,
        runtime_metadata,
    )

    with pytest.raises(LeaseError, match="runtime-metadata.json"):
        lease.update_heartbeat()
    lease.preserve_after_cleanup_failure("invalid runtime ownership metadata")
    lease.release()
    record = read_json_object(lease.lease_path)
    assert record["manual_clearance_required"] is True
    assert benchmark_lease_module.RUNTIME_METADATA_FILENAME in cast(
        dict[str, object], record["fragment_errors"]
    )


def test_special_file_json_inputs_fail_without_blocking(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    os.mkfifo(runtime_path)

    started = time.monotonic()
    with pytest.raises(LeaseError, match="runtime-metadata.json"):
        lease.update_heartbeat()
    assert time.monotonic() - started < 1.0
    lease.preserve_after_cleanup_failure("special-file runtime fragment")
    lease.release()

    standalone_fifo = tmp_path / "standalone.json"
    os.mkfifo(standalone_fifo)
    started = time.monotonic()
    with pytest.raises(LeaseError, match="not a regular file"):
        read_json_object(standalone_fifo)
    assert time.monotonic() - started < 1.0


def test_runtime_owner_token_cannot_change_after_first_snapshot(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()

    changed = make_runtime_metadata(lease, owner_token="replacement-token")
    atomic_write_json(runtime_path, changed)
    with pytest.raises(LeaseError, match="owner_token changed"):
        lease.update_heartbeat()
    lease.preserve_after_cleanup_failure("runtime owner token changed")
    lease.release()
    assert read_json_object(lease.lease_path)["runtime_metadata"] == initial


def test_runtime_process_identity_set_cannot_shrink_or_replay_stale_snapshot(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    expanded = make_runtime_metadata(lease)
    append_owned_process(expanded, lease)
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()
    atomic_write_json(runtime_path, expanded)
    lease.update_heartbeat()

    atomic_write_json(runtime_path, initial)
    with pytest.raises(LeaseError, match="only grow"):
        lease.update_heartbeat()
    lease.preserve_after_cleanup_failure("stale runtime snapshot replayed")
    lease.release()
    assert read_json_object(lease.lease_path)["runtime_metadata"] == expanded


def test_concurrent_stale_runtime_refresh_cannot_regress_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    expanded = make_runtime_metadata(lease)
    append_owned_process(expanded, lease)
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()

    original_read = benchmark_lease_module.read_json_object
    original_result_read = benchmark_lease_module.read_json_object_at
    first_read_started = threading.Event()
    allow_first_read_to_finish = threading.Event()
    runtime_read_count = 0
    runtime_read_count_lock = threading.Lock()

    def ordered_runtime_read(
        directory_descriptor: int, filename: str
    ) -> dict[str, object]:
        nonlocal runtime_read_count
        if filename != benchmark_lease_module.RUNTIME_METADATA_FILENAME:
            return original_result_read(directory_descriptor, filename)
        with runtime_read_count_lock:
            runtime_read_count += 1
            read_number = runtime_read_count
        if read_number == 1:
            first_read_started.set()
            assert allow_first_read_to_finish.wait(timeout=1)
            return expanded
        return initial

    monkeypatch.setattr(
        benchmark_lease_module, "read_json_object_at", ordered_runtime_read
    )
    errors: list[Exception] = []

    def refresh() -> None:
        try:
            lease.update_heartbeat()
        except Exception as error:
            errors.append(error)

    first_refresh = threading.Thread(target=refresh)
    second_refresh = threading.Thread(target=refresh)
    first_refresh.start()
    assert first_read_started.wait(timeout=1)
    second_refresh.start()
    allow_first_read_to_finish.set()
    first_refresh.join(timeout=1)
    second_refresh.join(timeout=1)

    assert not first_refresh.is_alive()
    assert not second_refresh.is_alive()
    assert len(errors) == 1
    assert "only grow" in cast(str, lease.cleanup_confirmation_error())
    assert original_read(lease.lease_path)["runtime_metadata"] == expanded
    lease.preserve_after_cleanup_failure("concurrent stale runtime refresh")
    lease.release()
    assert original_read(lease.lease_path)["runtime_metadata"] == expanded


def test_runtime_process_identity_metadata_cannot_mutate(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()

    mutated = make_runtime_metadata(lease)
    processes = cast(list[dict[str, object]], mutated["owned_processes"])
    processes[0]["log_path"] = "/tmp/reassigned.log"
    atomic_write_json(runtime_path, mutated)
    with pytest.raises(LeaseError, match="identity metadata changed"):
        lease.update_heartbeat()
    lease.preserve_after_cleanup_failure("runtime identity mutated")
    lease.release()
    assert read_json_object(lease.lease_path)["runtime_metadata"] == initial


def test_runtime_containment_may_bind_once_and_reconciles_with_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation_id = "1" * 32
    monkeypatch.setenv("INVOCATION_ID", invocation_id)
    run_id = "containment-valid"
    lease = make_lease(
        tmp_path,
        run_id=run_id,
        metadata=make_containment_metadata(run_id),
        allow_unconfirmed_cleanup_for_tests=False,
    )
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    initial["containment"] = None
    initial["owned_processes"] = []
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()

    active = make_runtime_metadata(lease)
    active["containment"] = make_runtime_containment(
        lease,
        owner_token=cast(str, active["owner_token"]),
        invocation_id=invocation_id,
    )
    atomic_write_json(runtime_path, active)
    lease.update_heartbeat()
    result = make_benchmark_result(
        lease,
        cleanup_succeeded=True,
        owned_processes=cast(Sequence[object], active["owned_processes"]),
    )
    result["containment"] = active["containment"]
    atomic_write_json(
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME,
        result,
    )
    assert lease.cleanup_succeeded(True) is True
    lease.release()
    assert not lease.lease_path.exists()


def test_runtime_containment_cannot_change_after_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation_id = "2" * 32
    monkeypatch.setenv("INVOCATION_ID", invocation_id)
    run_id = "containment-mutation"
    lease = make_lease(
        tmp_path,
        run_id=run_id,
        metadata=make_containment_metadata(run_id),
    )
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    initial = make_runtime_metadata(lease)
    initial["containment"] = make_runtime_containment(
        lease,
        owner_token=cast(str, initial["owner_token"]),
        invocation_id=invocation_id,
    )
    atomic_write_json(runtime_path, initial)
    lease.update_heartbeat()
    mutated = make_runtime_metadata(lease)
    mutated_containment = dict(initial["containment"])
    mutated_containment["inode"] = 54321
    mutated["containment"] = mutated_containment
    atomic_write_json(runtime_path, mutated)
    with pytest.raises(LeaseError, match="containment changed"):
        lease.update_heartbeat()
    lease.preserve_after_cleanup_failure("runtime containment mutated")
    lease.release()
    assert read_json_object(lease.lease_path)["runtime_metadata"] == initial


def test_containment_contract_rejects_missing_or_mismatched_final_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation_id = "3" * 32
    monkeypatch.setenv("INVOCATION_ID", invocation_id)
    for suffix, runtime_is_active in (("missing", False), ("mismatch", True)):
        run_id = f"containment-{suffix}"
        lease = make_lease(
            tmp_path / suffix,
            run_id=run_id,
            metadata=make_containment_metadata(run_id),
            allow_unconfirmed_cleanup_for_tests=False,
        )
        lease.acquire()
        runtime = make_runtime_metadata(lease)
        runtime["containment"] = (
            make_runtime_containment(
                lease,
                owner_token=cast(str, runtime["owner_token"]),
                invocation_id=invocation_id,
            )
            if runtime_is_active
            else None
        )
        if not runtime_is_active:
            runtime["owned_processes"] = []
        atomic_write_json(
            lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME,
            runtime,
        )
        lease.update_heartbeat()
        result = make_benchmark_result(
            lease,
            cleanup_succeeded=True,
            owned_processes=cast(Sequence[object], runtime["owned_processes"]),
        )
        if runtime_is_active:
            mismatched = dict(cast(dict[str, object], runtime["containment"]))
            mismatched["inode"] = 99999
            result["containment"] = mismatched
        else:
            result["containment"] = None
        atomic_write_json(
            lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME,
            result,
        )
        assert lease.cleanup_succeeded(True) is False
        error = cast(str, lease.cleanup_confirmation_error())
        assert "containment" in error
        lease.preserve_after_cleanup_failure(f"{suffix} containment evidence")
        lease.release()


def test_benchmark_result_is_bound_and_reconciles_owned_processes(
    tmp_path: Path,
) -> None:
    invalid_fragments = (
        {"schema_version": 2},
        {"run_id": "another-run"},
        {"namespace": "another-namespace"},
    )
    for index, overrides in enumerate(invalid_fragments):
        lease = make_lease(
            tmp_path / str(index), allow_unconfirmed_cleanup_for_tests=False
        )
        lease.acquire()
        result = make_benchmark_result(lease, cleanup_succeeded=True)
        result.update(overrides)
        atomic_write_json(
            lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME,
            result,
        )
        assert lease.cleanup_succeeded(True) is False
        lease.preserve_after_cleanup_failure("invalid benchmark result binding")
        lease.release()

    lease = make_lease(
        tmp_path / "missing-runtime", allow_unconfirmed_cleanup_for_tests=False
    )
    lease.acquire()
    owned_processes = cast(
        Sequence[object], make_runtime_metadata(lease)["owned_processes"]
    )
    write_benchmark_result(
        lease,
        cleanup_succeeded=True,
        owned_processes=owned_processes,
    )
    assert lease.cleanup_succeeded(True) is False
    assert "require runtime metadata" in cast(str, lease.cleanup_confirmation_error())
    lease.preserve_after_cleanup_failure("result omitted runtime metadata")
    lease.release()

    lease = make_lease(tmp_path / "owned", allow_unconfirmed_cleanup_for_tests=False)
    lease.acquire()
    runtime_metadata = make_runtime_metadata(lease)
    atomic_write_json(
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME,
        runtime_metadata,
    )
    lease.update_heartbeat()
    mismatched_processes = cast(
        list[dict[str, object]],
        make_runtime_metadata(lease)["owned_processes"],
    )
    mismatched_processes[0]["transport_pid"] = 999
    write_benchmark_result(
        lease,
        cleanup_succeeded=True,
        owned_processes=mismatched_processes,
    )
    assert lease.cleanup_succeeded(True) is False
    assert "do not match" in cast(str, lease.cleanup_confirmation_error())
    lease.preserve_after_cleanup_failure("result ownership did not reconcile")
    lease.release()

    lease = make_lease(
        tmp_path / "omitted-owned", allow_unconfirmed_cleanup_for_tests=False
    )
    lease.acquire()
    runtime_metadata = make_runtime_metadata(lease)
    atomic_write_json(
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME,
        runtime_metadata,
    )
    lease.update_heartbeat()
    write_benchmark_result(lease, cleanup_succeeded=True)
    assert lease.cleanup_succeeded(True) is False
    assert "do not match" in cast(str, lease.cleanup_confirmation_error())
    lease.preserve_after_cleanup_failure("result omitted observed ownership")
    lease.release()


def test_runtime_fault_tombstone_retains_last_valid_ownership(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    runtime_path = (
        lease.result_directory / benchmark_lease_module.RUNTIME_METADATA_FILENAME
    )
    runtime_metadata = make_runtime_metadata(lease)
    atomic_write_json(runtime_path, runtime_metadata)
    lease.update_heartbeat()

    runtime_path.write_text("{")
    with pytest.raises(LeaseError, match="runtime-metadata.json"):
        lease.update_heartbeat()
    assert lease.cleanup_succeeded(True) is False
    lease.preserve_after_cleanup_failure("runtime ownership metadata became invalid")
    lease.release()

    record = read_json_object(lease.lease_path)
    assert record["runtime_metadata"] == runtime_metadata
    assert benchmark_lease_module.RUNTIME_METADATA_FILENAME in cast(
        dict[str, object], record["fragment_errors"]
    )


def test_manifest_survives_malformed_child_manifest(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    manifest_path = lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    manifest_path.write_text("{")

    lease.write_manifest(status="completed", return_code=0, cleanup_succeeded=True)

    manifest = read_json_object(manifest_path)
    fragment_errors = cast(dict[str, object], manifest["fragment_errors"])
    assert benchmark_lease_module.MANIFEST_FILENAME in fragment_errors
    assert manifest["status"] == "completed"
    lease.release()


def test_command_result_is_recorded_in_manifest(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    set_lease_command(lease, (sys.executable, "-c", "raise SystemExit(7)"))
    assert run_with_lease(lease) == 7

    manifest = read_json_object(lease.result_directory / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["return_code"] == 7
    assert manifest["command_return_code"] == 7
    assert manifest["cleanup_succeeded"] is True
    assert manifest["cleanup_forced"] is False
    assert isinstance(manifest["child_pid"], int)
    assert not lease.lease_path.exists()


def test_invalid_identifier_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        make_lease(tmp_path, run_id="contains spaces")


def test_update_refuses_foreign_metadata(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    atomic_write_json(lease.lease_path, {"lease_id": "foreign"})
    with pytest.raises(LeaseError, match="ownership changed"):
        lease.update_heartbeat()
    lease.release()


def test_acquire_releases_lock_and_retains_run_directory_when_metadata_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path, run_id="failed")
    original_write = benchmark_lease_module.atomic_write_json

    def fail_write(_path: Path, _value: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(benchmark_lease_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="write failed"):
        lease.acquire()
    assert lease.result_directory.is_dir()

    monkeypatch.setattr(benchmark_lease_module, "atomic_write_json", original_write)
    replacement = make_lease(tmp_path, run_id="replacement")
    with replacement:
        assert replacement.lease_path.exists()


def test_acquire_rollback_never_removes_a_replacement_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path, run_id="rollback-race")
    moved_owned_directory = tmp_path / "moved-owned-result"
    replacement_identity: tuple[int, int] | None = None

    def replace_result_and_fail(_path: Path, _value: object) -> None:
        nonlocal replacement_identity
        lease.result_directory.rename(moved_owned_directory)
        lease.result_directory.mkdir()
        replacement_status = lease.result_directory.stat()
        replacement_identity = (
            replacement_status.st_dev,
            replacement_status.st_ino,
        )
        raise OSError("write failed after result replacement")

    monkeypatch.setattr(
        benchmark_lease_module, "atomic_write_json", replace_result_and_fail
    )
    with pytest.raises(OSError, match="write failed after result replacement"):
        lease.acquire()

    assert replacement_identity is not None
    observed_replacement = lease.result_directory.stat()
    assert (observed_replacement.st_dev, observed_replacement.st_ino) == (
        replacement_identity
    )
    assert moved_owned_directory.is_dir()
    assert lease._result_directory_descriptor is None


def test_existing_result_directory_is_never_reused(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.result_directory.mkdir(parents=True)
    existing_result = lease.result_directory / "existing.json"
    existing_result.write_text("{}\n")

    with pytest.raises(LeaseError, match="use a new run ID"):
        lease.acquire()

    existing_result.unlink()
    with pytest.raises(LeaseError, match="use a new run ID"):
        lease.acquire()

    lease.result_directory.rmdir()
    with lease:
        assert lease.lease_path.exists()


def test_result_directory_symlink_is_never_followed(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    outside_directory = tmp_path / "outside-result"
    outside_directory.mkdir()
    lease.result_directory.parent.mkdir(parents=True)
    lease.result_directory.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(LeaseError, match="use a new run ID"):
        lease.acquire()

    assert lease.result_directory.is_symlink()
    assert not tuple(outside_directory.iterdir())
    assert not lease.lease_path.exists()


def test_result_root_symlink_is_never_followed(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    outside_root = tmp_path / "outside-root"
    outside_root.mkdir()
    lease.result_directory.parent.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(LeaseError, match="without symlinks"):
        lease.acquire()

    assert not (outside_root / lease.run_id).exists()
    assert not lease.lease_path.exists()


def test_wrapper_output_stays_in_owned_directory_after_path_replacement(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    moved_owned_directory = tmp_path / "moved-owned-result"
    outside_directory = tmp_path / "outside-result"
    outside_directory.mkdir()
    lease.result_directory.rename(moved_owned_directory)
    lease.result_directory.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(LeaseError, match="identity failure"):
        lease.write_manifest(status="completed", return_code=0, cleanup_succeeded=True)

    manifest = read_json_object(
        moved_owned_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "wrapper_error"
    assert manifest["cleanup_succeeded"] is False
    assert not (outside_directory / benchmark_lease_module.MANIFEST_FILENAME).exists()
    assert lease.result_directory.is_symlink()
    lease.release()
    assert read_json_object(lease.lease_path)["manual_clearance_required"] is True


def test_child_inherits_trusted_result_directory_descriptor(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    marker_name = "child-descriptor-marker"
    child_script = (
        "import os; "
        "fd=int(os.environ['EXO_BENCHMARK_RESULT_DIRECTORY_FD']); "
        f"out=os.open({marker_name!r},os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=fd); "
        "os.write(out,b'owned'); os.close(out)"
    )
    set_lease_command(lease, (sys.executable, "-c", child_script))

    assert run_with_lease(lease) == 0

    assert (lease.result_directory / marker_name).read_bytes() == b"owned"


def test_heartbeat_failure_stops_command_and_records_wrapper_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    set_lease_command(lease, (sys.executable, "-c", "import time; time.sleep(60)"))
    original_update = lease.update_heartbeat
    update_count = 0

    def fail_after_initial_update() -> None:
        nonlocal update_count
        update_count += 1
        if update_count > 1:
            raise LeaseError("heartbeat write failed")
        original_update()

    monkeypatch.setattr(lease, "update_heartbeat", fail_after_initial_update)
    with pytest.raises(LeaseError, match="heartbeat failed"):
        run_with_lease(lease)

    manifest = read_json_object(lease.result_directory / "manifest.json")
    assert manifest["status"] == "wrapper_error"
    assert manifest["cleanup_succeeded"] is False
    assert manifest["cleanup_forced"] is True
    assert read_json_object(lease.lease_path)["manual_clearance_required"] is True


def test_late_heartbeat_error_forces_tombstone_after_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()

    def late_heartbeat_error() -> Exception:
        return LeaseError("late heartbeat finalization error")

    monkeypatch.setattr(lease, "stop_heartbeat", late_heartbeat_error)
    assert (
        run_managed_command(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE
    )
    lease.release()

    record = read_json_object(lease.lease_path)
    assert record["manual_clearance_required"] is True
    manifest = read_json_object(lease.result_directory / "manifest.json")
    assert manifest["cleanup_succeeded"] is False
    assert "late heartbeat finalization error" in cast(str, manifest["error_message"])


def test_interrupt_allows_delayed_cooperative_cleanup_without_sigkill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulated_old_grace_seconds = 0.05
    child_cleanup_delay_seconds = 0.2
    lease = make_lease(
        tmp_path,
        allow_unconfirmed_cleanup_for_tests=False,
        cleanup_grace_seconds=0.75,
    )
    result_path = (
        lease.result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    )
    ready_path = tmp_path / "child-ready"
    result_json = json.dumps(make_benchmark_result(lease, cleanup_succeeded=True))
    child_script = (
        "import signal, time\nfrom pathlib import Path\n"
        "def finish(_signal, _frame):\n"
        f" time.sleep({child_cleanup_delay_seconds!r})\n"
        f" Path({str(result_path)!r}).write_text({result_json!r})\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, finish)\n"
        f"Path({str(ready_path)!r}).write_text('ready')\n"
        "while True: time.sleep(0.05)"
    )
    set_lease_command(lease, (sys.executable, "-c", child_script))
    lease.acquire()
    signal_state = ManagedSignalState()

    def interrupt_when_ready() -> None:
        deadline = time.monotonic() + 2
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        signal_state.record(signal.SIGTERM)

    interrupt_thread = threading.Thread(target=interrupt_when_ready)
    sent_signals: list[int] = []
    original_killpg = os.killpg

    def record_killpg(process_group_id: int, signal_number: int) -> None:
        if signal_number != 0:
            sent_signals.append(signal_number)
        original_killpg(process_group_id, signal_number)

    monkeypatch.setattr(os, "killpg", record_killpg)
    interrupt_thread.start()
    started = time.monotonic()
    assert run_managed_command(lease, signal_state) == 128 + signal.SIGTERM
    elapsed = time.monotonic() - started
    interrupt_thread.join(timeout=1)
    lease.release()

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL not in sent_signals
    assert elapsed > simulated_old_grace_seconds
    assert not lease.lease_path.exists()
    manifest = read_json_object(lease.result_directory / "manifest.json")
    assert manifest["cleanup_succeeded"] is True


def test_lingering_owned_child_is_terminated_and_invalidates_run(
    tmp_path: Path,
) -> None:
    lease = make_lease(tmp_path)
    child_pid_path = tmp_path / "child.pid"
    child_script = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)']); "
        f"Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )
    set_lease_command(lease, (sys.executable, "-c", child_script))

    assert run_with_lease(lease) == benchmark_lease_module.FORCED_CLEANUP_RETURN_CODE

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()

    manifest = read_json_object(lease.result_directory / "manifest.json")
    assert manifest["status"] == "cleanup_forced"
    assert manifest["command_return_code"] == 0
    assert manifest["cleanup_succeeded"] is True
    assert manifest["cleanup_forced"] is True

    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_signal_during_popen_assignment_still_cleans_tracked_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    set_lease_command(lease, (sys.executable, "-c", "import time; time.sleep(60)"))
    lease.acquire()
    signal_state = ManagedSignalState()
    original_popen = benchmark_lease_module.subprocess.Popen

    def spawn_and_signal(
        command: Sequence[str],
        *,
        start_new_session: bool,
        env: Mapping[str, str],
        pass_fds: tuple[int, ...],
    ) -> subprocess.Popen[bytes]:
        signal_state.record(signal.SIGTERM)
        return original_popen(
            command,
            start_new_session=start_new_session,
            env=env,
            pass_fds=pass_fds,
        )

    monkeypatch.setattr(benchmark_lease_module.subprocess, "Popen", spawn_and_signal)
    assert run_managed_command(lease, signal_state) == 128 + signal.SIGTERM
    child_pid = lease.child_pid
    lease.release()

    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "interrupted"
    assert manifest["cleanup_succeeded"] is True


def test_signal_during_cleanup_cannot_bypass_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    signal_state = ManagedSignalState()

    def process_group_still_exists(_process_id: int) -> bool:
        return True

    def cleanup_and_signal(
        _process: subprocess.Popen[bytes], *, grace_seconds: float
    ) -> bool:
        del grace_seconds
        signal_state.record(signal.SIGTERM)
        return True

    monkeypatch.setattr(
        benchmark_lease_module, "process_group_exists", process_group_still_exists
    )
    monkeypatch.setattr(
        benchmark_lease_module, "terminate_owned_process_group", cleanup_and_signal
    )

    assert run_managed_command(lease, signal_state) == 128 + signal.SIGTERM
    lease.release()
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "interrupted"
    assert manifest["cleanup_forced"] is True


def test_signal_during_manifest_is_latched_and_manifest_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    lease.acquire()
    signal_state = ManagedSignalState()
    original_write_manifest = lease.write_manifest
    write_count = 0

    def write_manifest_and_signal(
        *,
        status: str,
        return_code: int | None,
        cleanup_succeeded: bool,
        cleanup_forced: bool = False,
        command_return_code: int | None = None,
        error_message: str | None = None,
    ) -> Path:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            signal_state.record(signal.SIGHUP)
        return original_write_manifest(
            status=status,
            return_code=return_code,
            cleanup_succeeded=cleanup_succeeded,
            cleanup_forced=cleanup_forced,
            command_return_code=command_return_code,
            error_message=error_message,
        )

    monkeypatch.setattr(lease, "write_manifest", write_manifest_and_signal)
    assert run_managed_command(lease, signal_state) == 128 + signal.SIGHUP
    lease.release()

    assert write_count == 2
    manifest = read_json_object(
        lease.result_directory / benchmark_lease_module.MANIFEST_FILENAME
    )
    assert manifest["status"] == "interrupted"
    assert manifest["return_code"] == 128 + signal.SIGHUP


def test_repeated_signals_keep_the_first_signal() -> None:
    signal_state = ManagedSignalState()
    signal_state.record(signal.SIGTERM)
    signal_state.record(signal.SIGHUP)
    assert signal_state.first_signal_number == signal.SIGTERM


def test_cli_rejects_result_symlink_installed_before_acquire(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = "result-symlink-race"
    namespace = "namespace-result-symlink-race"
    result_root = tmp_path / "results"
    result_directory = result_root / run_id
    outside_directory = tmp_path / "outside-result"
    child_marker = tmp_path / "child-ran"
    command = (
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(child_marker)!r}).touch()",
    )
    metadata_path = tmp_path / "metadata.json"
    atomic_write_json(
        metadata_path,
        make_local_metadata(
            run_id=run_id,
            namespace=namespace,
            reserved_ports=(53001, 53002),
            result_directory=result_directory,
            command=command,
        ),
    )
    result_root.mkdir()
    outside_directory.mkdir()
    result_directory.symlink_to(outside_directory, target_is_directory=True)
    lease_path = tmp_path / "lease.json"

    return_code = benchmark_lease_module.main(
        (
            "--owner",
            "codex:test",
            "--purpose",
            "result-symlink-race-test",
            "--run-id",
            run_id,
            "--namespace",
            namespace,
            "--port",
            "53001,53002",
            "--metadata-json",
            str(metadata_path),
            "--lock-path",
            str(tmp_path / "benchmark.lock"),
            "--lease-path",
            str(lease_path),
            "--result-root",
            str(result_root),
            "--",
            *command,
        )
    )

    assert return_code == 75
    assert "use a new run ID" in capsys.readouterr().err
    assert not child_marker.exists()
    assert not tuple(outside_directory.iterdir())
    assert not lease_path.exists()
    assert result_directory.is_symlink()


def test_sigterm_cleans_child_writes_manifest_and_releases_lease(
    tmp_path: Path,
) -> None:
    run_id = "signal-test"
    namespace = "namespace-signal-test"
    result_root = tmp_path / "results"
    result_directory = result_root / run_id
    result_path = result_directory / benchmark_lease_module.BENCHMARK_RESULT_FILENAME
    ready_path = tmp_path / "signal-child-ready"
    benchmark_result = {
        "schema_version": benchmark_lease_module.BENCHMARK_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "namespace": namespace,
        "cleanup_succeeded": True,
    }
    child_script = (
        "import signal, time\nfrom pathlib import Path\n"
        "def finish(_signal, _frame):\n"
        " time.sleep(0.2)\n"
        f" Path({str(result_path)!r}).write_text("
        f"{json.dumps(benchmark_result)!r})\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, finish)\n"
        f"Path({str(ready_path)!r}).write_text('ready')\n"
        "while True: time.sleep(0.05)"
    )
    child_command = (sys.executable, "-c", child_script)
    metadata_path = tmp_path / "metadata.json"
    atomic_write_json(
        metadata_path,
        make_local_metadata(
            run_id=run_id,
            namespace=namespace,
            reserved_ports=(53001, 53002),
            result_directory=result_directory,
            command=child_command,
        ),
    )
    lease_path = tmp_path / "lease.json"
    wrapper = subprocess.Popen(
        (
            sys.executable,
            str(Path(benchmark_lease_module.__file__)),
            "--owner",
            "codex:test",
            "--purpose",
            "signal-test",
            "--run-id",
            run_id,
            "--namespace",
            namespace,
            "--port",
            "53001,53002",
            "--metadata-json",
            str(metadata_path),
            "--heartbeat-seconds",
            "0.01",
            "--cleanup-grace-seconds",
            "0.75",
            "--lock-path",
            str(tmp_path / "benchmark.lock"),
            "--lease-path",
            str(lease_path),
            "--result-root",
            str(result_root),
            "--",
            *child_command,
        )
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if lease_path.exists():
                payload = read_json_object(lease_path)
                candidate_pid = payload.get("child_pid")
                if isinstance(candidate_pid, int):
                    child_pid = candidate_pid
                    break
            time.sleep(0.01)
        assert child_pid is not None

        ready_deadline = time.monotonic() + 2
        while not ready_path.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert ready_path.exists()

        status_fields = {
            key: value.strip()
            for key, value in (
                line.split(":", maxsplit=1)
                for line in Path(f"/proc/{wrapper.pid}/status").read_text().splitlines()
                if ":" in line
            )
        }
        blocked_signals = int(status_fields["SigBlk"], 16)
        assert blocked_signals & (1 << (signal.SIGTERM - 1)) == 0
        os.kill(wrapper.pid, signal.SIGTERM)
        assert wrapper.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)

    assert not lease_path.exists()
    assert child_pid is not None
    assert not Path(f"/proc/{child_pid}").exists()
    manifest = read_json_object(result_directory / "manifest.json")
    assert manifest["status"] == "interrupted"
    assert manifest["return_code"] == 128 + signal.SIGTERM
    assert manifest["cleanup_succeeded"] is True
