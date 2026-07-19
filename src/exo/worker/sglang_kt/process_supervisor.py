import os
import signal
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, final

import anyio
from anyio import EndOfStream
from anyio.abc import Process, TaskGroup
from anyio.lowlevel import checkpoint_if_cancelled
from loguru import logger
from pydantic import model_validator

from exo.shared.types.common import NodeId
from exo.shared.types.worker.sglang_kt import ResourceIndex
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import SglangKtProcessLaunchSpec
from exo.worker.sglang_kt.preflight import SglangKtPreflightPassed

ProcessOutputStream = Literal["stdout", "stderr"]
ProcessStopSignal = Literal["none", "term", "kill"]


class SglangKtManagedProcess(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    def iter_output(self, stream: ProcessOutputStream) -> AsyncIterator[bytes]: ...

    async def wait(self) -> int: ...

    async def terminate_group(self) -> bool: ...

    async def kill_group(self) -> bool: ...

    async def group_is_alive(self) -> bool: ...

    async def aclose(self) -> None: ...


ProcessOwnershipHandoff = Callable[[SglangKtManagedProcess], None]
ProcessStarter = Callable[
    [SglangKtProcessLaunchSpec, ProcessOwnershipHandoff], Awaitable[None]
]
ReadinessProbe = Callable[[SglangKtProcessLaunchSpec], Awaitable[None]]
ProcessOutputSink = Callable[
    [SglangKtProcessLaunchSpec, ProcessOutputStream, bytes], Awaitable[None]
]


@final
class SglangKtProcessStopReceipt(FrozenModel):
    pipeline_rank: ResourceIndex
    pid: int
    returncode: int
    stop_signal: ProcessStopSignal


@final
class SglangKtLocalProcessGroupStopReceipt(FrozenModel):
    node_id: NodeId
    failure: str | None
    processes: tuple[SglangKtProcessStopReceipt, ...]

    @model_validator(mode="after")
    def validate_processes(self) -> "SglangKtLocalProcessGroupStopReceipt":
        ranks = tuple(process.pipeline_rank for process in self.processes)
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("process stop receipts must have sorted, unique ranks")
        return self


@final
class AnyioSglangKtManagedProcess:
    """Own one SGLang process session, including every descendant process."""

    def __init__(self, process: Process) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        return await self._process.wait()

    async def terminate_group(self) -> bool:
        return _signal_process_group(self.pid, signal.SIGTERM)

    async def kill_group(self) -> bool:
        return _signal_process_group(self.pid, signal.SIGKILL)

    async def group_is_alive(self) -> bool:
        return _process_group_is_alive(self.pid)

    async def aclose(self) -> None:
        await self._process.aclose()

    async def iter_output(self, stream: ProcessOutputStream) -> AsyncIterator[bytes]:
        receive_stream = (
            self._process.stdout if stream == "stdout" else self._process.stderr
        )
        if receive_stream is None:
            return
        while True:
            try:
                yield await receive_stream.receive()
            except EndOfStream:
                return


def build_sglang_kt_process_environment(
    process_spec: SglangKtProcessLaunchSpec,
    parent_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if parent_environment is None else parent_environment)
    for variable_name in tuple(environment):
        if variable_name.startswith(process_spec.unset_environment_variable_prefixes):
            environment.pop(variable_name)
    for variable_name in process_spec.unset_environment_variables:
        environment.pop(variable_name, None)
    environment.update(process_spec.environment)
    return environment


def build_cpu_bound_sglang_kt_command(
    process_spec: SglangKtProcessLaunchSpec,
) -> tuple[str, ...]:
    cpu_list = ",".join(str(cpu_core) for cpu_core in sorted(process_spec.cpu_cores))
    return ("/usr/bin/taskset", "--cpu-list", cpu_list, "--", *process_spec.command)


async def launch_sglang_kt_process(
    process_spec: SglangKtProcessLaunchSpec,
    transfer_ownership: ProcessOwnershipHandoff,
) -> None:
    """Start a process and transfer ownership before the next checkpoint.

    Until ``transfer_ownership`` returns, this function owns any process it
    creates. Any alternate starter must provide the same cancellation contract.
    """

    # The AnyIO backend owns cancellation cleanup until open_process returns.
    # The synchronous callback then transfers ownership before another checkpoint.
    process = await anyio.open_process(
        build_cpu_bound_sglang_kt_command(process_spec),
        env=build_sglang_kt_process_environment(process_spec),
        start_new_session=True,
    )
    transfer_ownership(AnyioSglangKtManagedProcess(process))
    await checkpoint_if_cancelled()


async def log_sglang_kt_process_output(
    process_spec: SglangKtProcessLaunchSpec,
    stream: ProcessOutputStream,
    chunk: bytes,
) -> None:
    message = chunk.decode(errors="replace").rstrip()
    if message:
        logger.info(f"SGLang-KT rank {process_spec.pipeline_rank} {stream}: {message}")


@dataclass
class _OutputDrain:
    pipeline_rank: ResourceIndex
    stream: ProcessOutputStream
    finished: anyio.Event
    cancel_scope: anyio.CancelScope | None = None
    cancel_requested: bool = False


@dataclass
class _OwnedProcess:
    spec: SglangKtProcessLaunchSpec
    process: SglangKtManagedProcess
    stop_signal: ProcessStopSignal = "none"
    output_drains: tuple[_OutputDrain, ...] = ()
    closed: bool = False


@dataclass
class _ProcessStartAttempt:
    spec: SglangKtProcessLaunchSpec
    ownership_transferred: anyio.Event
    finished: anyio.Event
    cancel_scope: anyio.CancelScope | None = None
    cancel_requested: bool = False
    error: BaseException | None = None


class SglangKtProcessExitedUnexpectedlyError(RuntimeError):
    def __init__(self, *, pipeline_rank: int, pid: int, returncode: int) -> None:
        super().__init__(
            "SGLang-KT process exited before group shutdown: "
            f"rank={pipeline_rank}, pid={pid}, returncode={returncode}"
        )


class SglangKtProcessStarterContractError(RuntimeError):
    pass


class SglangKtOutputDrainTruncatedError(RuntimeError):
    def __init__(self, output_drains: tuple[_OutputDrain, ...]) -> None:
        streams = ", ".join(
            f"rank={output_drain.pipeline_rank}:{output_drain.stream}"
            for output_drain in output_drains
        )
        super().__init__(f"SGLang-KT output drain timeout truncated streams: {streams}")


@final
class LocalSglangKtProcessGroupSupervisor:
    """Transactionally supervise the SGLang ranks assigned to one Exo node."""

    def __init__(
        self,
        *,
        preflight: SglangKtPreflightPassed,
        node_id: NodeId,
        readiness_probe: ReadinessProbe,
        process_starter: ProcessStarter = launch_sglang_kt_process,
        output_sink: ProcessOutputSink = log_sglang_kt_process_output,
        readiness_timeout_seconds: float = 300.0,
        terminate_timeout_seconds: float = 15.0,
        kill_timeout_seconds: float = 5.0,
        output_drain_timeout_seconds: float = 5.0,
    ) -> None:
        if readiness_timeout_seconds <= 0:
            raise ValueError("readiness timeout must be positive")
        if terminate_timeout_seconds <= 0:
            raise ValueError("terminate timeout must be positive")
        if kill_timeout_seconds <= 0:
            raise ValueError("kill timeout must be positive")
        if output_drain_timeout_seconds <= 0:
            raise ValueError("output drain timeout must be positive")

        local_specs = tuple(
            sorted(
                (
                    process_spec
                    for process_spec in preflight.process_specs
                    if process_spec.node_id == node_id
                ),
                key=lambda process_spec: process_spec.pipeline_rank,
            )
        )
        if not local_specs:
            raise ValueError("passed preflight has no process assigned to this node")
        local_ranks = tuple(process_spec.pipeline_rank for process_spec in local_specs)
        if len(set(local_ranks)) != len(local_ranks):
            raise ValueError("local SGLang-KT process ranks must be unique")

        self.node_id = node_id
        self.process_specs = local_specs
        self._readiness_probe = readiness_probe
        self._process_starter = process_starter
        self._output_sink = output_sink
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._kill_timeout_seconds = kill_timeout_seconds
        self._output_drain_timeout_seconds = output_drain_timeout_seconds

        self._owned_processes: list[_OwnedProcess] = []
        self._process_start_attempts: list[_ProcessStartAttempt] = []
        self._run_called = False
        self._stopping = False
        self._ever_ready = anyio.Event()
        self._ready_or_finished = anyio.Event()
        self._stop_requested = anyio.Event()
        self._background_failure = anyio.Event()
        self._finished = anyio.Event()
        self._stopped = anyio.Event()
        self._background_error: BaseException | None = None
        self._failure: str | None = None
        self._stop_receipt: SglangKtLocalProcessGroupStopReceipt | None = None
        self._retry_cleanup_lock = anyio.Lock()

    @property
    def is_ready(self) -> bool:
        return (
            self._ever_ready.is_set()
            and not self._stopping
            and not self._finished.is_set()
        )

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def failure(self) -> str | None:
        return self._failure

    @property
    def stop_receipt(self) -> SglangKtLocalProcessGroupStopReceipt:
        if self._stop_receipt is None:
            raise RuntimeError("SGLang-KT process group has not stopped")
        return self._stop_receipt

    async def run(self) -> None:
        if self._run_called:
            raise RuntimeError("SGLang-KT process group supervisor cannot be restarted")
        self._run_called = True

        run_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        process_groups_stopped = False
        handles_closed = False
        try:
            try:
                async with anyio.create_task_group() as task_group:
                    try:
                        became_ready = await self._launch_and_wait_for_readiness(
                            task_group
                        )
                        if became_ready:
                            self._ever_ready.set()
                            self._ready_or_finished.set()
                            await self._wait_for_stop_or_background_failure()
                            self._raise_background_failure()
                    except BaseException as error:
                        run_error = error
                        self._record_failure(error)
                    finally:
                        self._stopping = True
                        with anyio.CancelScope(shield=True):
                            try:
                                await self._settle_process_start_attempts()
                                shutdown_error = await self._terminate_owned_processes()
                                process_groups_stopped = True
                                if shutdown_error is not None:
                                    self._record_failure(shutdown_error)
                                    if run_error is None:
                                        run_error = shutdown_error
                                drain_error = await self._wait_for_output_drains()
                                if run_error is None:
                                    run_error = self._background_error or drain_error
                                if drain_error is not None:
                                    self._record_failure(drain_error)
                            except BaseException as error:
                                cleanup_error = error
                                self._record_failure(error)
                            finally:
                                task_group.cancel_scope.cancel()
            except BaseException as error:
                if run_error is None and cleanup_error is None:
                    run_error = error
                    self._record_failure(error)

            if process_groups_stopped and cleanup_error is None:
                with anyio.CancelScope(shield=True):
                    try:
                        await self._close_owned_process_handles()
                        handles_closed = True
                    except BaseException as error:
                        cleanup_error = error
                        self._record_failure(error)
        finally:
            self._stopping = True
            if process_groups_stopped and handles_closed and cleanup_error is None:
                try:
                    self._stop_receipt = self._build_stop_receipt()
                except BaseException as error:
                    cleanup_error = error
                    self._record_failure(error)
            if self._stop_receipt is not None:
                self._stopped.set()
            self._ready_or_finished.set()
            self._finished.set()

        if cleanup_error is not None:
            raise cleanup_error
        if run_error is not None:
            raise run_error

    async def _launch_and_wait_for_readiness(
        self,
        task_group: TaskGroup,
    ) -> bool:
        startup_deadline = anyio.current_time() + self._readiness_timeout_seconds
        for process_spec in self.process_specs:
            if self._stop_requested.is_set():
                return False
            self._raise_background_failure()
            self._raise_if_startup_timed_out(startup_deadline)

            start_attempt = _ProcessStartAttempt(
                spec=process_spec,
                ownership_transferred=anyio.Event(),
                finished=anyio.Event(),
            )
            self._process_start_attempts.append(start_attempt)
            task_group.start_soon(self._run_process_starter, task_group, start_attempt)
            if not await self._wait_for_process_ownership(
                start_attempt, startup_deadline
            ):
                return False

            await checkpoint_if_cancelled()
            if self._stop_requested.is_set():
                return False

        readiness_events = tuple(anyio.Event() for _ in self.process_specs)
        for process_spec, readiness_event in zip(
            self.process_specs, readiness_events, strict=True
        ):
            task_group.start_soon(self._probe_readiness, process_spec, readiness_event)

        while True:
            self._raise_background_failure()
            if self._stop_requested.is_set():
                return False
            if all(event.is_set() for event in readiness_events):
                return True
            self._raise_if_startup_timed_out(startup_deadline)
            await anyio.sleep(0.01)

    async def _run_process_starter(
        self,
        task_group: TaskGroup,
        start_attempt: _ProcessStartAttempt,
    ) -> None:
        def transfer_ownership(process: SglangKtManagedProcess) -> None:
            self._transfer_process_ownership(task_group, start_attempt, process)

        try:
            with anyio.CancelScope() as cancel_scope:
                start_attempt.cancel_scope = cancel_scope
                if start_attempt.cancel_requested:
                    cancel_scope.cancel()
                try:
                    await self._process_starter(start_attempt.spec, transfer_ownership)
                except Exception as error:
                    if start_attempt.ownership_transferred.is_set():
                        self._record_background_failure(error)
                    else:
                        start_attempt.error = error
        finally:
            if (
                not start_attempt.ownership_transferred.is_set()
                and not start_attempt.cancel_requested
                and start_attempt.error is None
            ):
                start_attempt.error = SglangKtProcessStarterContractError(
                    "SGLang-KT process starter returned without transferring "
                    f"ownership: rank={start_attempt.spec.pipeline_rank}"
                )
            start_attempt.finished.set()

    def _transfer_process_ownership(
        self,
        task_group: TaskGroup,
        start_attempt: _ProcessStartAttempt,
        process: SglangKtManagedProcess,
    ) -> None:
        if start_attempt.ownership_transferred.is_set():
            raise SglangKtProcessStarterContractError(
                "SGLang-KT process starter transferred ownership more than once: "
                f"rank={start_attempt.spec.pipeline_rank}"
            )
        owned_process = _OwnedProcess(spec=start_attempt.spec, process=process)
        self._owned_processes.append(owned_process)
        self._start_process_tasks(task_group, owned_process)
        start_attempt.ownership_transferred.set()

    async def _wait_for_process_ownership(
        self,
        start_attempt: _ProcessStartAttempt,
        startup_deadline: float,
    ) -> bool:
        while True:
            if start_attempt.ownership_transferred.is_set():
                return True
            if start_attempt.finished.is_set():
                if start_attempt.error is not None:
                    raise start_attempt.error
                if self._stop_requested.is_set():
                    return False
                raise SglangKtProcessStarterContractError(
                    "SGLang-KT process starter finished without an ownership result: "
                    f"rank={start_attempt.spec.pipeline_rank}"
                )
            self._raise_background_failure()
            if self._stop_requested.is_set():
                await self._cancel_and_wait_for_process_start(start_attempt)
                return False
            if anyio.current_time() >= startup_deadline:
                await self._cancel_and_wait_for_process_start(start_attempt)
                raise TimeoutError("SGLang-KT process group startup timed out")
            await anyio.sleep(0.01)

    async def _cancel_and_wait_for_process_start(
        self,
        start_attempt: _ProcessStartAttempt,
    ) -> None:
        self._cancel_process_start(start_attempt)
        with anyio.fail_after(self._kill_timeout_seconds):
            await start_attempt.finished.wait()

    @staticmethod
    def _cancel_process_start(start_attempt: _ProcessStartAttempt) -> None:
        start_attempt.cancel_requested = True
        if start_attempt.cancel_scope is not None:
            start_attempt.cancel_scope.cancel()

    async def _settle_process_start_attempts(self) -> None:
        unfinished_attempts = tuple(
            start_attempt
            for start_attempt in self._process_start_attempts
            if not start_attempt.finished.is_set()
        )
        for start_attempt in unfinished_attempts:
            self._cancel_process_start(start_attempt)
        if unfinished_attempts:
            with anyio.fail_after(self._kill_timeout_seconds):
                for start_attempt in unfinished_attempts:
                    await start_attempt.finished.wait()

    @staticmethod
    def _raise_if_startup_timed_out(startup_deadline: float) -> None:
        if anyio.current_time() >= startup_deadline:
            raise TimeoutError("SGLang-KT process group startup timed out")

    def _start_process_tasks(
        self,
        task_group: TaskGroup,
        owned_process: _OwnedProcess,
    ) -> None:
        stdout_drain = _OutputDrain(
            pipeline_rank=owned_process.spec.pipeline_rank,
            stream="stdout",
            finished=anyio.Event(),
        )
        stderr_drain = _OutputDrain(
            pipeline_rank=owned_process.spec.pipeline_rank,
            stream="stderr",
            finished=anyio.Event(),
        )
        owned_process.output_drains = (stdout_drain, stderr_drain)
        task_group.start_soon(
            self._drain_process_output, owned_process, "stdout", stdout_drain
        )
        task_group.start_soon(
            self._drain_process_output, owned_process, "stderr", stderr_drain
        )
        task_group.start_soon(self._watch_process, owned_process)

    async def _wait_for_stop_or_background_failure(self) -> None:
        while True:
            if self._background_failure.is_set() or self._stop_requested.is_set():
                return
            await anyio.sleep(0.01)

    def _raise_background_failure(self) -> None:
        if self._background_error is not None:
            raise self._background_error

    def _record_background_failure(self, error: BaseException) -> None:
        if self._background_error is None:
            self._background_error = error
            self._record_failure(error)
            self._background_failure.set()

    async def wait_ready(self) -> None:
        await self._ready_or_finished.wait()
        if not self.is_ready:
            detail = self._failure or "process group stopped before readiness"
            raise RuntimeError(
                f"SGLang-KT process group did not become ready: {detail}"
            )

    async def stop(self) -> None:
        if not self._run_called:
            raise RuntimeError("SGLang-KT process group has not been started")
        self._stopping = True
        self._stop_requested.set()
        await self.wait_stopped()

    async def wait_stopped(self) -> None:
        await self._finished.wait()
        if not self._stopped.is_set():
            detail = self._failure or "process cleanup did not complete"
            raise RuntimeError(f"SGLang-KT process group did not stop: {detail}")

    async def retry_failed_cleanup(self) -> None:
        """Explicitly retry retained process ownership after a failed shutdown."""

        if not self._run_called:
            raise RuntimeError("SGLang-KT process group has not been started")
        await self._finished.wait()
        async with self._retry_cleanup_lock:
            if self._stopped.is_set():
                return
            shutdown_error: BaseException | None = None
            try:
                with anyio.CancelScope(shield=True):
                    shutdown_error = await self._terminate_owned_processes()
                    await self._close_owned_process_handles()
                self._stop_receipt = self._build_stop_receipt()
            except BaseException as error:
                self._record_failure(error)
                raise
            self._stopped.set()
            if shutdown_error is not None:
                self._record_failure(shutdown_error)
                raise shutdown_error

    async def _probe_readiness(
        self,
        process_spec: SglangKtProcessLaunchSpec,
        readiness_event: anyio.Event,
    ) -> None:
        try:
            await self._readiness_probe(process_spec)
        except Exception as error:
            self._record_background_failure(error)
        else:
            readiness_event.set()

    async def _watch_process(self, owned_process: _OwnedProcess) -> None:
        returncode = await owned_process.process.wait()
        if not self._stopping:
            self._record_background_failure(
                SglangKtProcessExitedUnexpectedlyError(
                    pipeline_rank=owned_process.spec.pipeline_rank,
                    pid=owned_process.process.pid,
                    returncode=returncode,
                )
            )

    async def _drain_process_output(
        self,
        owned_process: _OwnedProcess,
        stream: ProcessOutputStream,
        output_drain: _OutputDrain,
    ) -> None:
        try:
            with anyio.CancelScope() as cancel_scope:
                output_drain.cancel_scope = cancel_scope
                if output_drain.cancel_requested:
                    cancel_scope.cancel()
                async for chunk in owned_process.process.iter_output(stream):
                    await self._output_sink(owned_process.spec, stream, chunk)
        except Exception as error:
            self._record_background_failure(error)
        finally:
            output_drain.finished.set()

    async def _terminate_owned_processes(self) -> BaseExceptionGroup | None:
        owned_processes = tuple(self._owned_processes)
        if not owned_processes:
            return None

        live_processes, observation_errors = await self._observe_live_process_groups(
            owned_processes
        )
        shutdown_errors = [*observation_errors]
        shutdown_errors.extend(
            await self._signal_process_groups(live_processes, "term")
        )

        term_completed = False
        try:
            with anyio.move_on_after(self._terminate_timeout_seconds) as cancel_scope:
                await self._wait_for_process_groups(owned_processes)
            term_completed = not cancel_scope.cancel_called
        except BaseException as error:
            shutdown_errors.append(error)
        if term_completed:
            return _build_shutdown_error(shutdown_errors)

        live_processes, observation_errors = await self._observe_live_process_groups(
            owned_processes
        )
        shutdown_errors.extend(observation_errors)
        shutdown_errors.extend(
            await self._signal_process_groups(live_processes, "kill")
        )
        try:
            with anyio.fail_after(self._kill_timeout_seconds):
                await self._wait_for_process_groups(owned_processes)
        except BaseException as error:
            shutdown_errors.append(error)
            raise BaseExceptionGroup(
                "SGLang-KT process-group shutdown failed",
                shutdown_errors,
            ) from error
        return _build_shutdown_error(shutdown_errors)

    @staticmethod
    async def _signal_process_groups(
        owned_processes: tuple[_OwnedProcess, ...],
        stop_signal: Literal["term", "kill"],
    ) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for owned_process in owned_processes:
            if stop_signal == "term":
                signal_group = owned_process.process.terminate_group
            else:
                signal_group = owned_process.process.kill_group
            try:
                signal_delivered = await signal_group()
            except BaseException as error:
                errors.append(error)
            else:
                if signal_delivered and (
                    stop_signal == "kill" or owned_process.stop_signal == "none"
                ):
                    owned_process.stop_signal = stop_signal
        return tuple(errors)

    @staticmethod
    async def _observe_live_process_groups(
        owned_processes: tuple[_OwnedProcess, ...],
    ) -> tuple[tuple[_OwnedProcess, ...], tuple[BaseException, ...]]:
        live_processes: list[_OwnedProcess] = []
        errors: list[BaseException] = []
        for owned_process in owned_processes:
            try:
                is_alive = await owned_process.process.group_is_alive()
            except BaseException as error:
                errors.append(error)
                live_processes.append(owned_process)
            else:
                if is_alive:
                    live_processes.append(owned_process)
        return tuple(live_processes), tuple(errors)

    async def _wait_for_output_drains(
        self,
    ) -> SglangKtOutputDrainTruncatedError | None:
        output_drains = tuple(
            output_drain
            for owned_process in self._owned_processes
            for output_drain in owned_process.output_drains
        )
        if not output_drains:
            return None

        with anyio.move_on_after(self._output_drain_timeout_seconds) as cancel_scope:
            await self._wait_for_output_drain_events(output_drains)
        if not cancel_scope.cancel_called:
            return None

        truncated_drains = tuple(
            output_drain
            for output_drain in output_drains
            if not output_drain.finished.is_set()
        )
        for output_drain in truncated_drains:
            output_drain.cancel_requested = True
            if output_drain.cancel_scope is not None:
                output_drain.cancel_scope.cancel()

        with anyio.fail_after(self._kill_timeout_seconds):
            await self._wait_for_output_drain_events(output_drains)
        return SglangKtOutputDrainTruncatedError(truncated_drains)

    async def _close_owned_process_handles(self) -> None:
        first_error: BaseException | None = None
        for owned_process in self._owned_processes:
            if owned_process.closed:
                continue
            try:
                await owned_process.process.aclose()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                owned_process.closed = True
        if first_error is not None:
            raise first_error

    @classmethod
    async def _wait_for_process_groups(
        cls,
        owned_processes: tuple[_OwnedProcess, ...],
    ) -> None:
        for owned_process in owned_processes:
            if owned_process.process.returncode is None:
                await owned_process.process.wait()

        while True:
            live_processes, errors = await cls._observe_live_process_groups(
                owned_processes
            )
            if errors:
                raise BaseExceptionGroup(
                    "SGLang-KT process-group liveness checks failed",
                    list(errors),
                )
            if not live_processes:
                return
            await anyio.sleep(0.01)

    @staticmethod
    async def _wait_for_output_drain_events(
        output_drains: tuple[_OutputDrain, ...],
    ) -> None:
        for output_drain in output_drains:
            await output_drain.finished.wait()

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = repr(error)

    def _build_stop_receipt(self) -> SglangKtLocalProcessGroupStopReceipt:
        process_receipts: list[SglangKtProcessStopReceipt] = []
        for owned_process in self._owned_processes:
            returncode = owned_process.process.returncode
            if returncode is None:
                raise RuntimeError(
                    "cannot release SGLang-KT process ownership before exit"
                )
            process_receipts.append(
                SglangKtProcessStopReceipt(
                    pipeline_rank=owned_process.spec.pipeline_rank,
                    pid=owned_process.process.pid,
                    returncode=returncode,
                    stop_signal=owned_process.stop_signal,
                )
            )
        return SglangKtLocalProcessGroupStopReceipt(
            node_id=self.node_id,
            failure=self._failure,
            processes=tuple(process_receipts),
        )


def _signal_process_group(pid: int, process_signal: signal.Signals) -> bool:
    try:
        os.killpg(pid, process_signal)
    except ProcessLookupError:
        return False
    return True


def _process_group_is_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _build_shutdown_error(
    shutdown_errors: list[BaseException],
) -> BaseExceptionGroup | None:
    if not shutdown_errors:
        return None
    return BaseExceptionGroup(
        "SGLang-KT process-group shutdown encountered errors",
        shutdown_errors,
    )
