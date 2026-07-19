import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import anyio
import pytest
from anyio.abc import Process

from exo.shared.types.common import NodeId
from exo.worker.sglang_kt.launch_spec import (
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtPreflightPassed,
    SglangKtRankAdmissionBinding,
)
from exo.worker.sglang_kt.process_supervisor import (
    AnyioSglangKtManagedProcess,
    LocalSglangKtProcessGroupSupervisor,
    ProcessOutputStream,
    ProcessOwnershipHandoff,
    ProcessStarter,
    SglangKtDistributedAdmissionBarrierRequiredError,
    SglangKtManagedProcess,
    build_cpu_bound_sglang_kt_command,
    build_sglang_kt_process_environment,
)
from exo.worker.tests.unittests.test_sglang_kt_launch_spec import (
    PYTHON_EXECUTABLE,
    make_glm_4_7_flash_bf16_plan,
)
from exo.worker.tests.unittests.test_sglang_kt_preflight import make_specs


class FakeManagedProcess:
    def __init__(
        self,
        pid: int,
        *,
        ignore_terminate: bool = False,
        ignore_kill: bool = False,
        group_survives_leader_exit: bool = False,
        stdout_chunks: tuple[bytes, ...] = (),
        stderr_chunks: tuple[bytes, ...] = (),
    ) -> None:
        self._pid = pid
        self._returncode: int | None = None
        self._exited = anyio.Event()
        self._ignore_terminate = ignore_terminate
        self._ignore_kill = ignore_kill
        self._group_alive = True
        self._group_survives_leader_exit = group_survives_leader_exit
        self._chunks = {
            "stdout": stdout_chunks,
            "stderr": stderr_chunks,
        }
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def finish(self, returncode: int) -> None:
        if self._returncode is None:
            self._returncode = returncode
            if not self._group_survives_leader_exit:
                self._group_alive = False
            self._exited.set()

    def accept_kill(self) -> None:
        self._ignore_kill = False

    async def wait(self) -> int:
        await self._exited.wait()
        assert self._returncode is not None
        return self._returncode

    async def terminate_group(self) -> bool:
        self.terminate_calls += 1
        if not self._ignore_terminate:
            self._group_alive = False
            self.finish(-15)
        return True

    async def kill_group(self) -> bool:
        self.kill_calls += 1
        if not self._ignore_kill:
            self._group_alive = False
            self.finish(-9)
        return True

    async def group_is_alive(self) -> bool:
        return self._group_alive

    async def aclose(self) -> None:
        self.close_calls += 1

    async def iter_output(self, stream: ProcessOutputStream) -> AsyncIterator[bytes]:
        for chunk in self._chunks[stream]:
            yield chunk
        await self._exited.wait()


class CloseSensitiveFakeManagedProcess(FakeManagedProcess):
    def __init__(self, pid: int, chunks: tuple[bytes, ...]) -> None:
        super().__init__(pid, stdout_chunks=chunks, stderr_chunks=chunks)
        self.closed_while_draining = False

    async def iter_output(self, stream: ProcessOutputStream) -> AsyncIterator[bytes]:
        for chunk in self._chunks[stream]:
            await anyio.sleep(0.001)
            if self.close_calls:
                self.closed_while_draining = True
                raise anyio.ClosedResourceError
            yield chunk
        await self._exited.wait()


class TerminateFailingFakeManagedProcess(FakeManagedProcess):
    async def terminate_group(self) -> bool:
        self.terminate_calls += 1
        raise PermissionError("TERM denied")


class TerminateFailingAfterNaturalExitFakeManagedProcess(FakeManagedProcess):
    async def terminate_group(self) -> bool:
        self.terminate_calls += 1
        self.finish(0)
        raise PermissionError("TERM denied before natural exit")


class ExitsBeforeTermDeliveryFakeManagedProcess(FakeManagedProcess):
    async def terminate_group(self) -> bool:
        self.terminate_calls += 1
        self.finish(0)
        return False


class CloseFailsOnceFakeManagedProcess(FakeManagedProcess):
    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("close failed once")


def make_process_spec_only_preflight(
    specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> SglangKtPreflightPassed:
    return SglangKtPreflightPassed(
        process_specs=specs,
        admission_bindings=tuple(
            SglangKtRankAdmissionBinding(
                pipeline_rank=spec.pipeline_rank,
                process_spec_sha256=calculate_sglang_kt_process_launch_spec_sha256(
                    spec
                ),
            )
            for spec in specs
        ),
    )


def make_preflight() -> SglangKtPreflightPassed:
    return make_process_spec_only_preflight(make_specs()[:2])


async def ready_immediately(_spec: SglangKtProcessLaunchSpec) -> None:
    return None


async def discard_output(
    _spec: SglangKtProcessLaunchSpec,
    _stream: ProcessOutputStream,
    _chunk: bytes,
) -> None:
    return None


FakeProcessLauncher = Callable[
    [SglangKtProcessLaunchSpec], Awaitable[SglangKtManagedProcess]
]


def make_process_starter(launcher: FakeProcessLauncher) -> ProcessStarter:
    async def start_process(
        process_spec: SglangKtProcessLaunchSpec,
        transfer_ownership: ProcessOwnershipHandoff,
    ) -> None:
        transfer_ownership(await launcher(process_spec))

    return start_process


def test_builds_exact_cpu_bound_command_and_sanitized_environment() -> None:
    spec = make_specs()[0]

    command = build_cpu_bound_sglang_kt_command(spec)
    environment = build_sglang_kt_process_environment(
        spec,
        {
            "KEEP": "yes",
            "CUDA_VISIBLE_DEVICES": "stale",
            "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "stale",
        },
    )

    assert command == (
        "/usr/bin/taskset",
        "--cpu-list",
        ",".join(str(cpu_core) for cpu_core in spec.cpu_cores),
        "--",
        *spec.command,
    )
    assert environment["KEEP"] == "yes"
    assert environment["CUDA_VISIBLE_DEVICES"] == spec.gpu_uuid
    assert "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE" not in environment
    assert all(environment[name] == value for name, value in spec.environment)


def test_flash_environment_removes_stale_nccl_and_sglang_configuration() -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
    )

    environment = build_sglang_kt_process_environment(
        spec,
        {
            "KEEP": "yes",
            "CUDA_VISIBLE_DEVICES": "stale",
            "PYTORCH_ALLOC_CONF": "stale",
            "NCCL_NET": "IB",
            "NCCL_IB_HCA": "=mlx4_0:1",
            "NCCL_SOCKET_IFNAME": "ib0",
            "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "stale",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
            "SGLANG_KT_HYBRID_TIMING": "stale",
        },
    )

    assert environment["KEEP"] == "yes"
    assert all(not name.startswith("NCCL_") for name in environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == spec.gpu_uuid
    assert environment["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert environment["SGLANG_KT_HYBRID_TIMING"] == "1"
    assert tuple(name for name in environment if name.startswith("SGLANG_")) == (
        "SGLANG_KT_HYBRID_TIMING",
    )


def test_refuses_node_without_local_specs_and_invalid_timeouts() -> None:
    preflight = make_preflight()

    with pytest.raises(ValueError, match="no process assigned"):
        LocalSglangKtProcessGroupSupervisor(
            preflight=preflight,
            node_id=NodeId("absent"),
            readiness_probe=ready_immediately,
        )
    with pytest.raises(ValueError, match="readiness timeout"):
        LocalSglangKtProcessGroupSupervisor(
            preflight=preflight,
            node_id=NodeId("dwagon"),
            readiness_probe=ready_immediately,
            readiness_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="admission timeout"):
        LocalSglangKtProcessGroupSupervisor(
            preflight=preflight,
            node_id=NodeId("dwagon"),
            readiness_probe=ready_immediately,
            admission_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="output drain timeout"):
        LocalSglangKtProcessGroupSupervisor(
            preflight=preflight,
            node_id=NodeId("dwagon"),
            readiness_probe=ready_immediately,
            output_drain_timeout_seconds=0,
        )


async def test_starts_both_local_ranks_and_stops_idempotently() -> None:
    processes = {
        0: FakeManagedProcess(1000),
        1: FakeManagedProcess(1001),
    }
    launched_ranks: list[int] = []

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        pipeline_rank = spec.pipeline_rank
        launched_ranks.append(pipeline_rank)
        return processes[pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        assert supervisor.is_ready
        assert launched_ranks == [0, 1]
        await supervisor.stop()
        await supervisor.stop()

    assert supervisor.is_stopped
    assert not supervisor.is_ready
    assert supervisor.failure is None
    assert tuple(
        receipt.pipeline_rank for receipt in supervisor.stop_receipt.processes
    ) == (
        0,
        1,
    )
    assert all(
        receipt.stop_signal == "term" for receipt in supervisor.stop_receipt.processes
    )
    assert all(process.terminate_calls == 1 for process in processes.values())
    assert all(process.kill_calls == 0 for process in processes.values())
    assert all(process.close_calls == 1 for process in processes.values())

    with pytest.raises(RuntimeError, match="did not become ready"):
        await supervisor.wait_ready()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await supervisor.run()


async def test_admission_failure_prevents_every_process_start() -> None:
    launch_attempts: list[int] = []
    verified_ranks: list[tuple[int, ...]] = []

    async def reject_admission(
        specs: tuple[SglangKtProcessLaunchSpec, ...],
        bindings: tuple[SglangKtRankAdmissionBinding, ...],
    ) -> None:
        verified_ranks.append(tuple(spec.pipeline_rank for spec in specs))
        assert tuple(binding.pipeline_rank for binding in bindings) == (0, 1)
        raise RuntimeError("admission evidence changed")

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        launch_attempts.append(spec.pipeline_rank)
        return FakeManagedProcess(1100 + spec.pipeline_rank)

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        admission_verifier=reject_admission,
        output_sink=discard_output,
    )

    with pytest.raises(RuntimeError, match="admission evidence changed"):
        await supervisor.run()

    assert verified_ranks == [(0, 1)]
    assert launch_attempts == []
    assert supervisor.is_stopped
    assert supervisor.stop_receipt.processes == ()


async def test_distributed_launch_requires_a_post_verification_barrier() -> None:
    launch_attempts: list[int] = []

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        launch_attempts.append(spec.pipeline_rank)
        return FakeManagedProcess(1200 + spec.pipeline_rank)

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight(make_specs()),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
    )

    with pytest.raises(
        SglangKtDistributedAdmissionBarrierRequiredError,
        match="requires an admission barrier",
    ):
        await supervisor.run()

    assert launch_attempts == []
    assert supervisor.is_stopped
    assert supervisor.stop_receipt.processes == ()


async def test_stop_cancels_admission_before_any_process_start() -> None:
    admission_started = anyio.Event()
    admission_cancelled = anyio.Event()
    launch_attempts: list[int] = []

    async def block_admission(
        _specs: tuple[SglangKtProcessLaunchSpec, ...],
        _bindings: tuple[SglangKtRankAdmissionBinding, ...],
    ) -> None:
        admission_started.set()
        try:
            await anyio.sleep_forever()
        finally:
            admission_cancelled.set()

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        launch_attempts.append(spec.pipeline_rank)
        return FakeManagedProcess(1300 + spec.pipeline_rank)

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        admission_verifier=block_admission,
        output_sink=discard_output,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await admission_started.wait()
        with anyio.fail_after(0.5):
            await supervisor.stop()

    assert admission_cancelled.is_set()
    assert launch_attempts == []
    assert supervisor.is_stopped


async def test_admission_timeout_prevents_every_process_start() -> None:
    launch_attempts: list[int] = []

    async def block_admission(
        _specs: tuple[SglangKtProcessLaunchSpec, ...],
        _bindings: tuple[SglangKtRankAdmissionBinding, ...],
    ) -> None:
        await anyio.sleep_forever()

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        launch_attempts.append(spec.pipeline_rank)
        return FakeManagedProcess(1400 + spec.pipeline_rank)

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        admission_verifier=block_admission,
        admission_timeout_seconds=0.01,
        output_sink=discard_output,
    )

    with pytest.raises(TimeoutError):
        await supervisor.run()

    assert launch_attempts == []
    assert supervisor.is_stopped


async def test_partial_launch_failure_stops_owned_process() -> None:
    first_process = FakeManagedProcess(1100)
    launch_count = 0

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            raise RuntimeError("second spawn failed")
        return first_process

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
    )

    with pytest.raises(RuntimeError, match="second spawn failed") as error:
        await supervisor.run()

    assert "second spawn failed" in str(error.value)
    assert supervisor.is_stopped
    assert first_process.terminate_calls == 1
    assert first_process.close_calls == 1
    assert supervisor.stop_receipt.processes[0].stop_signal == "term"


async def test_early_process_exit_stops_its_sibling() -> None:
    processes = {
        0: FakeManagedProcess(1200, group_survives_leader_exit=True),
        1: FakeManagedProcess(1201),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        pipeline_rank = spec.pipeline_rank
        return processes[pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            processes[0].finish(7)
            await supervisor.wait_stopped()

    assert "returncode=7" in repr(error.value)
    assert supervisor.is_stopped
    assert processes[0].terminate_calls == 1
    assert processes[1].terminate_calls == 1
    assert "returncode=7" in (supervisor.failure or "")


async def test_readiness_timeout_stops_every_process() -> None:
    processes = {
        0: FakeManagedProcess(1300),
        1: FakeManagedProcess(1301),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        pipeline_rank = spec.pipeline_rank
        return processes[pipeline_rank]

    async def never_ready(_spec: SglangKtProcessLaunchSpec) -> None:
        await anyio.sleep_forever()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=never_ready,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        readiness_timeout_seconds=0.05,
    )

    with pytest.raises(TimeoutError):
        await supervisor.run()

    assert not supervisor.is_ready
    assert supervisor.is_stopped
    assert all(process.terminate_calls == 1 for process in processes.values())
    with pytest.raises(RuntimeError, match="did not become ready"):
        await supervisor.wait_ready()


async def test_stop_during_readiness_cleans_up_without_waiting_for_timeout() -> None:
    processes = {
        0: FakeManagedProcess(1350),
        1: FakeManagedProcess(1351),
    }
    launches_complete = anyio.Event()
    launch_count = 0

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            launches_complete.set()
        return processes[spec.pipeline_rank]

    async def never_ready(_spec: SglangKtProcessLaunchSpec) -> None:
        await anyio.sleep_forever()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=never_ready,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        readiness_timeout_seconds=60,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await launches_complete.wait()
        await supervisor.stop()

    assert supervisor.is_stopped
    assert supervisor.failure is None
    assert all(process.terminate_calls == 1 for process in processes.values())
    with pytest.raises(RuntimeError, match="did not become ready"):
        await supervisor.wait_ready()


async def test_launch_is_bounded_by_the_startup_timeout() -> None:
    async def never_launch(
        _spec: SglangKtProcessLaunchSpec,
    ) -> SglangKtManagedProcess:
        await anyio.sleep_forever()
        raise AssertionError("unreachable")

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(never_launch),
        output_sink=discard_output,
        readiness_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError) as error:
        await supervisor.run()

    assert "TimeoutError" in repr(error.value)
    assert supervisor.is_stopped
    assert supervisor.stop_receipt.processes == ()


async def test_cancellation_after_spawn_cannot_lose_process_ownership() -> None:
    process = FakeManagedProcess(1380)
    process_handed_off = anyio.Event()
    cancel_scope_ready = anyio.Event()
    run_cancel_scope: anyio.CancelScope | None = None

    async def start_process(
        _spec: SglangKtProcessLaunchSpec,
        transfer_ownership: ProcessOwnershipHandoff,
    ) -> None:
        transfer_ownership(process)
        process_handed_off.set()
        await anyio.sleep_forever()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=start_process,
        output_sink=discard_output,
    )

    async def run_in_cancel_scope() -> None:
        nonlocal run_cancel_scope
        with anyio.CancelScope() as cancel_scope:
            run_cancel_scope = cancel_scope
            cancel_scope_ready.set()
            await supervisor.run()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_in_cancel_scope)
        await cancel_scope_ready.wait()
        await process_handed_off.wait()
        assert run_cancel_scope is not None
        run_cancel_scope.cancel()
        await supervisor.wait_stopped()

    assert supervisor.is_stopped
    assert len(supervisor.stop_receipt.processes) == 1
    assert process.terminate_calls == 1
    assert process.close_calls == 1


async def test_timeout_cancels_unhanded_starter_cleanup_without_a_leak() -> None:
    process = FakeManagedProcess(1390)
    process_spawned = anyio.Event()

    async def start_process(
        _spec: SglangKtProcessLaunchSpec,
        _transfer_ownership: ProcessOwnershipHandoff,
    ) -> None:
        process_spawned.set()
        try:
            await anyio.sleep_forever()
        finally:
            await process.terminate_group()
            await process.aclose()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=start_process,
        output_sink=discard_output,
        readiness_timeout_seconds=0.02,
    )

    with pytest.raises(TimeoutError):
        await supervisor.run()

    assert process_spawned.is_set()
    assert process.terminate_calls == 1
    assert process.close_calls == 1
    assert supervisor.is_stopped
    assert supervisor.stop_receipt.processes == ()


async def test_default_starter_uses_open_process_cancellation_cleanup_before_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open_process = anyio.open_process
    opened_processes: list[Process] = []

    async def delayed_open_process(*_args: object, **_kwargs: object) -> Process:
        process = await real_open_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        opened_processes.append(process)
        try:
            await anyio.sleep(0.05)
        except BaseException:
            with anyio.CancelScope(shield=True):
                process.kill()
                await process.wait()
                await process.aclose()
            raise
        return process

    monkeypatch.setattr(anyio, "open_process", delayed_open_process)
    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        output_sink=discard_output,
        readiness_timeout_seconds=0.01,
        kill_timeout_seconds=0.2,
    )

    process_was_leaked = False
    try:
        with pytest.raises(TimeoutError):
            await supervisor.run()
        assert len(opened_processes) == 1
        process_was_leaked = opened_processes[0].returncode is None
    finally:
        for process in opened_processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
                await process.aclose()

    assert not process_was_leaked
    assert supervisor.is_stopped
    assert supervisor.stop_receipt.processes == ()


async def test_stop_cancels_an_inflight_process_starter_promptly() -> None:
    starter_running = anyio.Event()
    starter_cancelled = anyio.Event()

    async def start_process(
        _spec: SglangKtProcessLaunchSpec,
        _transfer_ownership: ProcessOwnershipHandoff,
    ) -> None:
        starter_running.set()
        try:
            await anyio.sleep_forever()
        finally:
            starter_cancelled.set()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=start_process,
        output_sink=discard_output,
        readiness_timeout_seconds=60,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await starter_running.wait()
        with anyio.fail_after(0.2):
            await supervisor.stop()

    assert starter_cancelled.is_set()
    assert supervisor.is_stopped
    assert supervisor.failure is None


async def test_escalates_the_complete_group_to_kill() -> None:
    processes = {
        0: FakeManagedProcess(1400, ignore_terminate=True),
        1: FakeManagedProcess(1401, ignore_terminate=True),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        pipeline_rank = spec.pipeline_rank
        return processes[pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        await supervisor.stop()

    assert all(process.terminate_calls == 1 for process in processes.values())
    assert all(process.kill_calls == 1 for process in processes.values())
    assert all(
        receipt.stop_signal == "kill" for receipt in supervisor.stop_receipt.processes
    )


async def test_escalates_only_process_groups_still_alive_after_term() -> None:
    processes = {
        0: FakeManagedProcess(1450),
        1: FakeManagedProcess(1451, ignore_terminate=True),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return processes[spec.pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        await supervisor.stop()

    receipts = {
        receipt.pipeline_rank: receipt for receipt in supervisor.stop_receipt.processes
    }
    assert processes[0].terminate_calls == 1
    assert processes[0].kill_calls == 0
    assert receipts[0].stop_signal == "term"
    assert processes[1].terminate_calls == 1
    assert processes[1].kill_calls == 1
    assert receipts[1].stop_signal == "kill"


async def test_signal_failure_on_one_rank_does_not_skip_other_ranks() -> None:
    processes = {
        0: TerminateFailingFakeManagedProcess(1470),
        1: FakeManagedProcess(1471),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return processes[spec.pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await supervisor.stop()

    assert "TERM denied" in repr(error.value)
    assert supervisor.is_stopped
    assert processes[0].terminate_calls == 1
    assert processes[0].kill_calls == 1
    assert processes[1].terminate_calls == 1
    assert processes[1].kill_calls == 0
    receipts = {
        receipt.pipeline_rank: receipt for receipt in supervisor.stop_receipt.processes
    }
    assert receipts[0].stop_signal == "kill"
    assert receipts[1].stop_signal == "term"
    assert "TERM denied" in (supervisor.stop_receipt.failure or "")


async def test_failed_term_is_preserved_when_process_exits_naturally() -> None:
    process = TerminateFailingAfterNaturalExitFakeManagedProcess(1480)

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await supervisor.stop()

    assert "TERM denied before natural exit" in repr(error.value)
    assert supervisor.is_stopped
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert supervisor.stop_receipt.processes[0].returncode == 0
    assert supervisor.stop_receipt.processes[0].stop_signal == "none"
    assert "TERM denied before natural exit" in (supervisor.stop_receipt.failure or "")


async def test_exit_race_does_not_claim_term_delivery() -> None:
    process = ExitsBeforeTermDeliveryFakeManagedProcess(1481)

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        await supervisor.stop()

    assert supervisor.is_stopped
    assert supervisor.failure is None
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert supervisor.stop_receipt.processes[0].returncode == 0
    assert supervisor.stop_receipt.processes[0].stop_signal == "none"


async def test_drains_high_volume_output_without_blocking_shutdown() -> None:
    chunks = tuple(f"line-{index}\n".encode() for index in range(5_000))
    processes = {
        0: FakeManagedProcess(1500, stdout_chunks=chunks, stderr_chunks=chunks),
        1: FakeManagedProcess(1501),
    }
    drained_chunks = 0

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        pipeline_rank = spec.pipeline_rank
        return processes[pipeline_rank]

    async def count_output(
        _spec: SglangKtProcessLaunchSpec,
        _stream: ProcessOutputStream,
        _chunk: bytes,
    ) -> None:
        nonlocal drained_chunks
        drained_chunks += 1

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=count_output,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        await supervisor.stop()

    assert drained_chunks == 10_000


async def test_closes_process_handle_only_after_output_drains_finish() -> None:
    chunks = tuple(f"line-{index}\n".encode() for index in range(50))
    process = CloseSensitiveFakeManagedProcess(1550, chunks)
    drained_chunks = 0

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    async def count_output(
        _spec: SglangKtProcessLaunchSpec,
        _stream: ProcessOutputStream,
        _chunk: bytes,
    ) -> None:
        nonlocal drained_chunks
        drained_chunks += 1

    one_process_preflight = make_process_spec_only_preflight((make_specs()[0],))
    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=one_process_preflight,
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=count_output,
        output_drain_timeout_seconds=1,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run)
        await supervisor.wait_ready()
        await supervisor.stop()

    assert drained_chunks == 100
    assert not process.closed_while_draining
    assert process.close_calls == 1


async def test_output_sink_failure_during_shutdown_is_propagated() -> None:
    process = FakeManagedProcess(1570, stdout_chunks=(b"late output\n",))
    sink_entered = anyio.Event()
    release_sink = anyio.Event()

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    async def fail_late(
        _spec: SglangKtProcessLaunchSpec,
        _stream: ProcessOutputStream,
        _chunk: bytes,
    ) -> None:
        sink_entered.set()
        await release_sink.wait()
        raise RuntimeError("late output sink failed")

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=fail_late,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await sink_entered.wait()
            task_group.start_soon(supervisor.stop)
            while supervisor.is_ready:
                await anyio.sleep(0)
            release_sink.set()

    assert "late output sink failed" in repr(error.value)
    assert supervisor.is_stopped
    assert "late output sink failed" in (supervisor.stop_receipt.failure or "")
    assert process.close_calls == 1


async def test_output_drain_timeout_is_reported_as_truncation() -> None:
    process = FakeManagedProcess(1580, stdout_chunks=(b"blocked output\n",))
    sink_entered = anyio.Event()

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    async def block_output(
        _spec: SglangKtProcessLaunchSpec,
        _stream: ProcessOutputStream,
        _chunk: bytes,
    ) -> None:
        sink_entered.set()
        await anyio.sleep_forever()

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=block_output,
        output_drain_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await sink_entered.wait()
            await supervisor.stop()

    assert "output drain timeout truncated streams" in repr(error.value)
    assert supervisor.is_stopped
    assert "rank=0:stdout" in (supervisor.stop_receipt.failure or "")
    assert process.close_calls == 1


async def test_cleanup_retry_preserves_kill_history_after_close_failure() -> None:
    process = CloseFailsOnceFakeManagedProcess(1590, ignore_terminate=True)

    async def launch(_spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return process

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_process_spec_only_preflight((make_specs()[0],)),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.1,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await supervisor.stop()

    assert "close failed once" in repr(error.value)
    assert not supervisor.is_stopped
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.close_calls == 1

    await supervisor.retry_failed_cleanup()

    assert supervisor.is_stopped
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.close_calls == 2
    assert supervisor.stop_receipt.processes[0].stop_signal == "kill"


async def test_does_not_release_ownership_when_kill_times_out() -> None:
    processes = {
        0: FakeManagedProcess(1600, ignore_terminate=True, ignore_kill=True),
        1: FakeManagedProcess(1601, ignore_terminate=True, ignore_kill=True),
    }

    async def launch(spec: SglangKtProcessLaunchSpec) -> SglangKtManagedProcess:
        return processes[spec.pipeline_rank]

    supervisor = LocalSglangKtProcessGroupSupervisor(
        preflight=make_preflight(),
        node_id=NodeId("dwagon"),
        readiness_probe=ready_immediately,
        process_starter=make_process_starter(launch),
        output_sink=discard_output,
        readiness_timeout_seconds=0.1,
        terminate_timeout_seconds=0.01,
        kill_timeout_seconds=0.01,
    )

    with pytest.raises(BaseExceptionGroup) as error:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(supervisor.run)
            await supervisor.wait_ready()
            await supervisor.stop()

    assert "TimeoutError" in repr(error.value)
    assert not supervisor.is_stopped
    assert all(process.terminate_calls == 1 for process in processes.values())
    assert all(process.kill_calls == 1 for process in processes.values())
    assert all(process.close_calls == 0 for process in processes.values())
    with pytest.raises(RuntimeError, match="did not stop"):
        await supervisor.wait_stopped()
    with pytest.raises(RuntimeError, match="has not stopped"):
        _ = supervisor.stop_receipt

    for process in processes.values():
        process.accept_kill()
    await supervisor.retry_failed_cleanup()

    assert supervisor.is_stopped
    assert all(process.terminate_calls == 2 for process in processes.values())
    assert all(process.kill_calls == 2 for process in processes.values())
    assert all(process.close_calls == 1 for process in processes.values())
    assert "TimeoutError" in (supervisor.stop_receipt.failure or "")


async def test_anyio_process_handle_signals_descendants_after_leader_exit(
    tmp_path: Path,
) -> None:
    child_started_path = tmp_path / "child-started"
    child_terminated_path = tmp_path / "child-terminated"
    child_code = (
        "import signal,time\n"
        "from pathlib import Path\n"
        f"started=Path({str(child_started_path)!r})\n"
        f"terminated=Path({str(child_terminated_path)!r})\n"
        "started.write_text('started')\n"
        "def stop(_signal, _frame):\n"
        "    terminated.write_text('terminated')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "subprocess.Popen(\n"
        f"    [sys.executable, '-c', {child_code!r}],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"started=Path({str(child_started_path)!r})\n"
        "while not started.exists():\n"
        "    time.sleep(0.01)\n"
    )
    process = await anyio.open_process(
        [sys.executable, "-c", parent_code],
        start_new_session=True,
    )
    managed_process = AnyioSglangKtManagedProcess(process)

    try:
        with anyio.fail_after(5):
            while not child_started_path.exists():
                await anyio.sleep(0.01)

        assert await managed_process.wait() == 0
        assert await managed_process.group_is_alive()
        await managed_process.terminate_group()
        with anyio.fail_after(5):
            while not child_terminated_path.exists():
                await anyio.sleep(0.01)
            while await managed_process.group_is_alive():
                await anyio.sleep(0.01)
    finally:
        if await managed_process.group_is_alive():
            await managed_process.kill_group()
        if managed_process.returncode is None:
            await managed_process.wait()
        await managed_process.aclose()

    assert child_terminated_path.read_text() == "terminated"
