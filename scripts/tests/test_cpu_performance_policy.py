from __future__ import annotations

import json
import os
import select
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import cpu_performance_policy as policy_module
from scripts.cpu_performance_policy import (
    CpuPerformancePolicyError,
    CpuPerformancePolicySysfsRoots,
    CpuPerformancePolicyTransactionConfig,
    cpu_performance_policy,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="ascii")


def _fake_sysfs(tmp_path: Path) -> CpuPerformancePolicySysfsRoots:
    roots = CpuPerformancePolicySysfsRoots(
        cpu=tmp_path / "sys/devices/system/cpu",
        powercap=tmp_path / "sys/class/powercap",
        hwmon=tmp_path / "sys/class/hwmon",
    )
    _write(roots.cpu / "online", "0-3")
    for policy_number, cpu_list, governor, preference in (
        (0, "0-1", "powersave", "balance_performance"),
        (2, "2-3", "schedutil", "balance_power"),
    ):
        policy = roots.cpu / f"cpufreq/policy{policy_number}"
        _write(policy / "affected_cpus", cpu_list)
        _write(policy / "related_cpus", cpu_list)
        _write(policy / "scaling_driver", "intel_pstate")
        _write(policy / "scaling_available_governors", "performance powersave")
        _write(
            policy / "energy_performance_available_preferences",
            "default performance balance_performance balance_power power",
        )
        _write(policy / "scaling_governor", governor)
        _write(policy / "energy_performance_preference", preference)

    _write(roots.cpu / "intel_pstate/status", "active")
    _write(roots.cpu / "intel_pstate/no_turbo", "0")
    _write(roots.cpu / "intel_pstate/min_perf_pct", "25")
    _write(roots.cpu / "intel_pstate/max_perf_pct", "100")
    _write(roots.cpu / "cpufreq/boost", "1")
    for cpu_id in (0, 2):
        throttle = roots.cpu / f"cpu{cpu_id}/thermal_throttle"
        _write(throttle / "core_throttle_count", "5")
        _write(throttle / "package_throttle_count", "7")

    package = roots.powercap / "intel-rapl:0"
    _write(package / "name", "package-0")
    _write(package / "enabled", "1")
    _write(package / "constraint_0_name", "long_term")
    _write(package / "constraint_0_power_limit_uw", "250000000")
    _write(package / "constraint_0_time_window_us", "1000000")

    hwmon = roots.hwmon / "hwmon0"
    _write(hwmon / "name", "coretemp")
    _write(hwmon / "temp1_label", "Package id 0")
    _write(hwmon / "temp1_max", "85000")
    _write(hwmon / "temp1_crit", "95000")
    return roots


def _write_process_identity(proc_root: Path, pid: int, start_time: int) -> None:
    fields_three_through_twenty_two = ["S", *(["0"] * 18), str(start_time)]
    _write(
        proc_root / str(pid) / "stat",
        f"{pid} (test process) {' '.join(fields_three_through_twenty_two)}",
    )


def _fake_transaction(tmp_path: Path) -> CpuPerformancePolicyTransactionConfig:
    proc_root = tmp_path / "proc"
    boot_id_path = proc_root / "sys/kernel/random/boot_id"
    _write(boot_id_path, "11111111-2222-3333-4444-555555555555")
    _write_process_identity(proc_root, os.getpid(), 123456)
    return CpuPerformancePolicyTransactionConfig(
        lock_path=tmp_path / "run/cpu-policy.lock",
        journal_path=tmp_path / "run/cpu-policy.journal.json",
        proc_root=proc_root,
        boot_id_path=boot_id_path,
    )


def _read_pipe_with_timeout(descriptor: int, timeout_seconds: float) -> bytes:
    readable, _, _ = select.select([descriptor], [], [], timeout_seconds)
    if not readable:
        raise TimeoutError("child process did not signal before the test timeout")
    return os.read(descriptor, 1)


def _wait_for_successful_child(pid: int) -> None:
    waited_pid, status = os.waitpid(pid, 0)
    assert waited_pid == pid
    assert os.waitstatus_to_exitcode(status) == 0


def _fork_crashed_session(
    roots: CpuPerformancePolicySysfsRoots,
    transaction: CpuPerformancePolicyTransactionConfig,
) -> int:
    read_descriptor, write_descriptor = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_descriptor)
        try:
            _write_process_identity(
                transaction.proc_root,
                os.getpid(),
                1_000_000 + os.getpid(),
            )
            with cpu_performance_policy(
                roots=roots,
                transaction=transaction,
            ):
                os.write(write_descriptor, b"1")
                os._exit(0)
        except BaseException:
            os._exit(2)
    os.close(write_descriptor)
    try:
        assert _read_pipe_with_timeout(read_descriptor, 3.0) == b"1"
    finally:
        os.close(read_descriptor)
    _wait_for_successful_child(pid)
    return pid


def _fork_holding_session(
    roots: CpuPerformancePolicySysfsRoots,
    transaction: CpuPerformancePolicyTransactionConfig,
) -> tuple[int, int]:
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(release_write)
        try:
            _write_process_identity(
                transaction.proc_root,
                os.getpid(),
                2_000_000 + os.getpid(),
            )
            with cpu_performance_policy(
                roots=roots,
                transaction=transaction,
            ):
                os.write(ready_write, b"1")
                if os.read(release_read, 1) != b"1":
                    os._exit(3)
            os._exit(0)
        except BaseException:
            os._exit(2)
    os.close(ready_write)
    os.close(release_read)
    try:
        assert _read_pipe_with_timeout(ready_read, 3.0) == b"1"
    finally:
        os.close(ready_read)
    return pid, release_write


def _release_holding_session(pid: int, release_descriptor: int) -> None:
    try:
        os.write(release_descriptor, b"1")
    finally:
        os.close(release_descriptor)
    _wait_for_successful_child(pid)


def _policy_value(
    roots: CpuPerformancePolicySysfsRoots, policy: int, field: str
) -> str:
    return (roots.cpu / f"cpufreq/policy{policy}" / field).read_text().strip()


def test_applies_verifies_captures_and_restores_exact_state(tmp_path: Path) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)

    with cpu_performance_policy(roots=roots, transaction=transaction) as session:
        for policy in (0, 2):
            assert _policy_value(roots, policy, "scaling_governor") == "performance"
            assert (
                _policy_value(roots, policy, "energy_performance_preference")
                == "performance"
            )
        _write(
            roots.cpu / "cpu0/thermal_throttle/package_throttle_count",
            "8",
        )

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert (
        _policy_value(roots, 0, "energy_performance_preference")
        == "balance_performance"
    )
    assert _policy_value(roots, 2, "scaling_governor") == "schedutil"
    assert _policy_value(roots, 2, "energy_performance_preference") == "balance_power"
    assert session.evidence["lifecycle"] == "restored"
    assert session.evidence["application_verified"] is True
    assert session.evidence["restoration_verified"] is True
    serialization = session.evidence["serialization"]
    assert isinstance(serialization, dict)
    assert serialization["journal_verified"] is True
    assert serialization["journal_removed"] is True
    assert serialization["lock_released"] is True
    assert len(session.evidence["policies"]) == 2  # type: ignore[arg-type]
    snapshots = session.evidence["snapshots"]
    assert isinstance(snapshots, dict)
    performance_after = snapshots["performance_after"]
    assert isinstance(performance_after, dict)
    throttle = performance_after["representative_throttle_counters"]
    assert isinstance(throttle, list)
    first = throttle[0]
    assert isinstance(first, dict)
    values = first["values"]
    assert isinstance(values, dict)
    assert values["package_throttle_count"] == 8
    json.dumps(session.evidence)


def test_restores_after_managed_workload_exception(tmp_path: Path) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)

    with (
        pytest.raises(RuntimeError, match="workload failed"),
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        raise RuntimeError("workload failed")

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert _policy_value(roots, 2, "energy_performance_preference") == "balance_power"


def test_deduplicates_policy_symlinks_and_ignores_offline_policy(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    _write(roots.cpu / "online", "0-1")
    (roots.cpu / "cpufreq/policy9").symlink_to(
        roots.cpu / "cpufreq/policy0", target_is_directory=True
    )

    with cpu_performance_policy(roots=roots, transaction=transaction) as session:
        policies = session.evidence["policies"]
        assert isinstance(policies, list)
        assert len(policies) == 1
        policy = policies[0]
        assert isinstance(policy, dict)
        assert policy["name"] == "policy0"


def test_missing_mandatory_epp_fails_before_any_policy_write(tmp_path: Path) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    (roots.cpu / "cpufreq/policy2/energy_performance_preference").unlink()

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("session must not start")

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert raised.value.evidence["lifecycle"] == "failed"
    assert any(
        failure.operation == "snapshot_policy" for failure in raised.value.failures
    )


def test_partial_application_failure_restores_every_policy_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_write = policy_module._write_sysfs_value
    writes: list[tuple[str, str, str]] = []

    def controlled_write(path: Path, value: str) -> None:
        writes.append((path.parent.name, path.name, value))
        if (
            path.parent.name == "policy0"
            and path.name == "energy_performance_preference"
            and value == "performance"
        ):
            raise OSError("injected application failure")
        real_write(path, value)

    monkeypatch.setattr(policy_module, "_write_sysfs_value", controlled_write)

    with (
        pytest.raises(CpuPerformancePolicyError, match="application failed"),
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("session must not start")

    restoration_writes = writes[2:]
    assert restoration_writes == [
        ("policy0", "scaling_governor", "powersave"),
        ("policy0", "energy_performance_preference", "balance_performance"),
        ("policy2", "scaling_governor", "schedutil"),
        ("policy2", "energy_performance_preference", "balance_power"),
    ]
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert _policy_value(roots, 2, "scaling_governor") == "schedutil"


def test_restoration_attempts_all_fields_and_reports_combined_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_write = policy_module._write_sysfs_value
    restoration_attempts: list[tuple[str, str]] = []

    def controlled_write(path: Path, value: str) -> None:
        if value != "performance":
            restoration_attempts.append((path.parent.name, path.name))
        if (
            path.parent.name == "policy0"
            and path.name == "scaling_governor"
            and value == "powersave"
        ) or (
            path.parent.name == "policy2"
            and path.name == "energy_performance_preference"
            and value == "balance_power"
        ):
            raise OSError(f"injected restoration failure for {path.name}")
        real_write(path, value)

    monkeypatch.setattr(policy_module, "_write_sysfs_value", controlled_write)

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pass

    assert restoration_attempts == [
        ("policy0", "scaling_governor"),
        ("policy0", "energy_performance_preference"),
        ("policy2", "scaling_governor"),
        ("policy2", "energy_performance_preference"),
    ]
    restoration_failures = [
        failure
        for failure in raised.value.failures
        if failure.phase == "restoration"
        and failure.operation.startswith("write_and_verify")
    ]
    assert len(restoration_failures) == 2
    assert raised.value.evidence["restoration_verified"] is False
    json.dumps(raised.value.evidence)


def test_read_failure_for_discovered_rapl_limit_fails_before_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    target = roots.powercap / "intel-rapl:0/constraint_0_power_limit_uw"
    real_read = policy_module._read_sysfs_value

    def controlled_read(path: Path) -> str:
        if path == target:
            raise OSError("injected RAPL read failure")
        return real_read(path)

    monkeypatch.setattr(policy_module, "_read_sysfs_value", controlled_read)

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("session must not start")

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert any(
        failure.operation == "snapshot_rapl_limits" for failure in raised.value.failures
    )


def test_static_turbo_drift_invalidates_session_but_restores_policies(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)

    with (
        pytest.raises(CpuPerformancePolicyError, match="verification failed"),
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        _write(roots.cpu / "intel_pstate/no_turbo", "1")

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert _policy_value(roots, 2, "energy_performance_preference") == "balance_power"


def test_missing_optional_telemetry_roots_are_explicit_and_allowed(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    for path in (
        roots.cpu / "intel_pstate/status",
        roots.cpu / "intel_pstate/no_turbo",
        roots.cpu / "intel_pstate/min_perf_pct",
        roots.cpu / "intel_pstate/max_perf_pct",
        roots.cpu / "cpufreq/boost",
    ):
        path.unlink()
    (roots.cpu / "intel_pstate").rmdir()
    for path in sorted(roots.powercap.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    roots.powercap.rmdir()
    for path in sorted(roots.hwmon.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    roots.hwmon.rmdir()

    with cpu_performance_policy(roots=roots, transaction=transaction) as session:
        snapshot = session.evidence["snapshots"]
        assert isinstance(snapshot, dict)
        before = snapshot["before"]
        assert isinstance(before, dict)
        platform = before["intel_pstate_and_turbo"]
        assert isinstance(platform, dict)
        assert platform["intel_pstate_available"] is False
        rapl = before["rapl_limits"]
        assert isinstance(rapl, dict)
        assert rapl["available"] is False
        temperatures = before["temperature_thresholds"]
        assert isinstance(temperatures, dict)
        assert temperatures["available"] is False


def test_overlapping_session_is_rejected_without_mutating_holder_state(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    holder_pid, release_descriptor = _fork_holding_session(roots, transaction)
    try:
        assert _policy_value(roots, 0, "scaling_governor") == "performance"
        with (
            pytest.raises(CpuPerformancePolicyError) as raised,
            cpu_performance_policy(roots=roots, transaction=transaction),
        ):
            pytest.fail("overlapping session must not acquire the lock")
        assert any(
            failure.operation == "acquire_transaction_lock"
            and failure.error_type == "TimeoutError"
            for failure in raised.value.failures
        )
        assert _policy_value(roots, 0, "scaling_governor") == "performance"
    finally:
        _release_holding_session(holder_pid, release_descriptor)
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert not transaction.journal_path.exists()


def test_blocking_session_waits_for_holder_then_runs(tmp_path: Path) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    holder_pid, release_descriptor = _fork_holding_session(roots, transaction)
    waiter_read, waiter_write = os.pipe()
    waiter_pid = os.fork()
    if waiter_pid == 0:
        os.close(waiter_read)
        try:
            blocking_transaction = replace(
                transaction,
                lock_timeout_seconds=None,
            )
            _write_process_identity(
                transaction.proc_root,
                os.getpid(),
                3_000_000 + os.getpid(),
            )
            with cpu_performance_policy(
                roots=roots,
                transaction=blocking_transaction,
            ):
                os.write(waiter_write, b"1")
            os._exit(0)
        except BaseException:
            os._exit(2)
    os.close(waiter_write)
    try:
        readable, _, _ = select.select([waiter_read], [], [], 0.15)
        assert readable == []
        _release_holding_session(holder_pid, release_descriptor)
        assert _read_pipe_with_timeout(waiter_read, 3.0) == b"1"
    finally:
        os.close(waiter_read)
    _wait_for_successful_child(waiter_pid)
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert not transaction.journal_path.exists()


def test_stale_crash_journal_is_recovered_before_new_mutation(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    crashed_pid = _fork_crashed_session(roots, transaction)
    assert transaction.journal_path.is_file()
    assert _policy_value(roots, 0, "scaling_governor") == "performance"
    (transaction.proc_root / str(crashed_pid) / "stat").unlink()
    (transaction.proc_root / str(crashed_pid)).rmdir()

    with cpu_performance_policy(
        roots=roots,
        transaction=transaction,
    ) as session:
        recovery = session.evidence["stale_recovery"]
        assert isinstance(recovery, dict)
        assert recovery["journal_found"] is True
        assert recovery["performed"] is True
        assert recovery["restoration_verified"] is True
        assert recovery["journal_removed"] is True
        assert _policy_value(roots, 0, "scaling_governor") == "performance"

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert not transaction.journal_path.exists()


def test_live_journal_owner_is_not_recovered_even_after_lock_release(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    crashed_pid = _fork_crashed_session(roots, transaction)

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("journal with a live owner identity must not be recovered")
    assert any(
        failure.operation == "recover_abandoned_transaction"
        and "live process identity" in failure.message
        for failure in raised.value.failures
    )
    assert transaction.journal_path.is_file()
    assert _policy_value(roots, 0, "scaling_governor") == "performance"

    (transaction.proc_root / str(crashed_pid) / "stat").unlink()
    (transaction.proc_root / str(crashed_pid)).rmdir()
    with cpu_performance_policy(roots=roots, transaction=transaction):
        pass
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert not transaction.journal_path.exists()


def test_keyboard_interrupt_during_application_restores_and_removes_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_write = policy_module._write_sysfs_value
    interrupted = False

    def interrupt_after_governor_write(path: Path, value: str) -> None:
        nonlocal interrupted
        if value == "performance":
            assert transaction.journal_path.is_file()
        real_write(path, value)
        if (
            not interrupted
            and path.parent.name == "policy0"
            and path.name == "scaling_governor"
            and value == "performance"
        ):
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(
        policy_module,
        "_write_sysfs_value",
        interrupt_after_governor_write,
    )

    with (
        pytest.raises(KeyboardInterrupt),
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("interrupted application must not enter the workload")

    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert (
        _policy_value(roots, 0, "energy_performance_preference")
        == "balance_performance"
    )
    assert _policy_value(roots, 2, "scaling_governor") == "schedutil"
    assert not transaction.journal_path.exists()


def test_final_topology_drift_fails_closed_and_leaves_recoverable_journal(
    tmp_path: Path,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        _write(roots.cpu / "online", "0-2")

    assert any(
        failure.operation == "verify_policy_topology_identity"
        for failure in raised.value.failures
    )
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert _policy_value(roots, 2, "scaling_governor") == "schedutil"
    assert transaction.journal_path.is_file()

    _write(roots.cpu / "online", "0-3")
    with cpu_performance_policy(roots=roots, transaction=transaction) as session:
        recovery = session.evidence["stale_recovery"]
        assert isinstance(recovery, dict)
        assert recovery["performed"] is True
    assert not transaction.journal_path.exists()


def test_active_snapshot_rediscovery_rejects_topology_drift_before_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_write = policy_module._write_sysfs_value
    topology_changed = False

    def change_topology_after_last_application(path: Path, value: str) -> None:
        nonlocal topology_changed
        real_write(path, value)
        if (
            not topology_changed
            and path.parent.name == "policy2"
            and path.name == "energy_performance_preference"
            and value == "performance"
        ):
            topology_changed = True
            _write(roots.cpu / "online", "0-2")

    monkeypatch.setattr(
        policy_module,
        "_write_sysfs_value",
        change_topology_after_last_application,
    )

    with (
        pytest.raises(CpuPerformancePolicyError) as raised,
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("active topology drift must prevent workload entry")

    assert any(
        failure.phase == "active"
        and failure.operation == "verify_policy_topology_identity"
        for failure in raised.value.failures
    )
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert transaction.journal_path.is_file()

    monkeypatch.setattr(policy_module, "_write_sysfs_value", real_write)
    _write(roots.cpu / "online", "0-3")
    with cpu_performance_policy(roots=roots, transaction=transaction):
        pass
    assert not transaction.journal_path.exists()


def test_governor_application_failure_never_attempts_epp_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_write = policy_module._write_sysfs_value
    writes: list[tuple[str, str, str]] = []

    def fail_governor(path: Path, value: str) -> None:
        writes.append((path.parent.name, path.name, value))
        if (
            path.parent.name == "policy0"
            and path.name == "scaling_governor"
            and value == "performance"
        ):
            raise OSError("injected governor failure")
        real_write(path, value)

    monkeypatch.setattr(policy_module, "_write_sysfs_value", fail_governor)

    with (
        pytest.raises(CpuPerformancePolicyError, match="application failed"),
        cpu_performance_policy(roots=roots, transaction=transaction),
    ):
        pytest.fail("failed governor application must not enter the workload")

    assert ("policy0", "energy_performance_preference", "performance") not in writes
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert not transaction.journal_path.exists()


def test_link_then_keyboard_interrupt_adopts_and_removes_published_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _fake_sysfs(tmp_path)
    transaction = _fake_transaction(tmp_path)
    real_link = policy_module.os.link
    interrupted = False

    def link_then_interrupt(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal interrupted
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(policy_module.os, "link", link_then_interrupt)
    session = cpu_performance_policy(roots=roots, transaction=transaction)

    with (
        pytest.raises(
            CpuPerformancePolicyError,
            match="journal publication failed",
        ) as raised,
        session,
    ):
        pytest.fail("interrupted journal publication must not enter the workload")

    assert any(
        failure.operation == "publish_pre_mutation_journal"
        and failure.error_type == "KeyboardInterrupt"
        for failure in raised.value.failures
    )
    serialization = session.evidence["serialization"]
    assert isinstance(serialization, dict)
    assert serialization["journal_adopted_after_interruption"] is True
    assert serialization["journal_removed"] is True
    assert serialization["lock_released"] is True
    assert session.evidence["restoration_verified"] is True
    assert _policy_value(roots, 0, "scaling_governor") == "powersave"
    assert (
        _policy_value(roots, 0, "energy_performance_preference")
        == "balance_performance"
    )
    assert not transaction.journal_path.exists()
    assert not any(
        path.name.endswith(".tmp") for path in transaction.journal_path.parent.iterdir()
    )
