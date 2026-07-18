from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scripts.benchmark_lease as benchmark_lease_module
from scripts.benchmark_lease import (
    BenchmarkLease,
    LeaseBusyError,
    LeaseError,
    StaleLeaseError,
    atomic_write_json,
    read_json_object,
    run_with_lease,
    validate_ports,
)


def make_lease(tmp_path: Path, *, run_id: str = "test-run") -> BenchmarkLease:
    return BenchmarkLease(
        lock_path=tmp_path / "benchmark.lock",
        lease_path=tmp_path / "lease.json",
        result_directory=tmp_path / "results" / run_id,
        owner="codex:test",
        purpose="unit-test",
        run_id=run_id,
        namespace=f"namespace-{run_id}",
        ports=(53001, 53002),
        command=(sys.executable, "-c", "raise SystemExit(0)"),
        heartbeat_seconds=0.01,
        expected_duration_seconds=60,
    )


def test_validate_ports_rejects_duplicates_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_ports((53001, 53001))
    with pytest.raises(ValueError, match="between"):
        validate_ports((0,))


def test_lease_writes_metadata_and_removes_only_its_record(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    with lease:
        payload = read_json_object(lease.lease_path)
        assert payload["lease_id"] == lease.lease_id
        assert payload["ports"] == [53001, 53002]
        assert payload["exo_namespace"] == "namespace-test-run"
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


def test_command_result_is_recorded_in_manifest(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.command = (sys.executable, "-c", "raise SystemExit(7)")
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


def test_acquire_releases_lock_when_metadata_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path, run_id="failed")
    original_write = benchmark_lease_module.atomic_write_json

    def fail_write(_path: Path, _value: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(benchmark_lease_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="write failed"):
        lease.acquire()

    monkeypatch.setattr(benchmark_lease_module, "atomic_write_json", original_write)
    replacement = make_lease(tmp_path, run_id="replacement")
    with replacement:
        assert replacement.lease_path.exists()


def test_nonempty_result_directory_is_never_reused(tmp_path: Path) -> None:
    lease = make_lease(tmp_path)
    lease.result_directory.mkdir(parents=True)
    existing_result = lease.result_directory / "existing.json"
    existing_result.write_text("{}\n")

    with pytest.raises(LeaseError, match="use a new run ID"):
        lease.acquire()

    existing_result.unlink()
    with lease:
        assert lease.lease_path.exists()


def test_heartbeat_failure_stops_command_and_records_wrapper_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = make_lease(tmp_path)
    lease.command = (sys.executable, "-c", "import time; time.sleep(60)")
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
    assert manifest["cleanup_succeeded"] is True
    assert manifest["cleanup_forced"] is True
    assert not lease.lease_path.exists()


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
    lease.command = (sys.executable, "-c", child_script)

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


def test_sigterm_cleans_child_writes_manifest_and_releases_lease(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}\n")
    lease_path = tmp_path / "lease.json"
    result_root = tmp_path / "results"
    wrapper = subprocess.Popen(
        (
            sys.executable,
            str(Path(benchmark_lease_module.__file__)),
            "--owner",
            "codex:test",
            "--purpose",
            "signal-test",
            "--run-id",
            "signal-test",
            "--namespace",
            "namespace-signal-test",
            "--port",
            "53001,53002",
            "--metadata-json",
            str(metadata_path),
            "--heartbeat-seconds",
            "0.01",
            "--lock-path",
            str(tmp_path / "benchmark.lock"),
            "--lease-path",
            str(lease_path),
            "--result-root",
            str(result_root),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
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
    manifest = read_json_object(result_root / "signal-test" / "manifest.json")
    assert manifest["status"] == "interrupted"
    assert manifest["return_code"] == 128 + signal.SIGTERM
    assert manifest["cleanup_succeeded"] is True
