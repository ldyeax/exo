import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import anyio
from anyio import fail_after, to_thread
from loguru import logger

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import (
    ModelSnapshotId,
    card_cache,
    model_snapshot_id,
)
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.compute_resources import ComputeResource
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor

RUNNER_SHUTDOWN_TIMEOUT_SECONDS = 3
RUNNER_TASK_START_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RunnerTaskStartFailure:
    runner_id: RunnerId
    task_status: TaskStatus
    error_message: str


def get_local_runner_ids_for_task(
    task: Task,
    instance: Instance,
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
) -> tuple[RunnerId, ...]:
    match task:
        case (
            ConnectToGroup(runner_id=runner_id)
            | LoadModel(runner_id=runner_id)
            | StartWarmup(runner_id=runner_id)
        ) if runner_id is not None:
            return (runner_id,) if runner_id in runners else ()
        case TextGeneration() | ImageEdits() | ImageGeneration():
            return tuple(
                runner_id
                for runner_id, runner in runners.items()
                if runner.bound_instance.instance.instance_id == instance.instance_id
                and task.task_id not in runner.completed
                and task.task_id not in runner.in_progress
                and isinstance(runner.status, (RunnerReady, RunnerRunning))
            )
        case _:
            runner_id = instance.shard_assignments.node_to_runner.get(node_id)
            return () if runner_id is None else (runner_id,)


async def start_local_runner_task(
    task: Task,
    instance: Instance,
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    timeout_seconds: float = RUNNER_TASK_START_TIMEOUT_SECONDS,
) -> tuple[RunnerTaskStartFailure, ...]:
    runner_ids = get_local_runner_ids_for_task(task, instance, node_id, runners)
    failures: list[RunnerTaskStartFailure] = []

    async def start_runner_task(runner_id: RunnerId) -> None:
        try:
            with fail_after(timeout_seconds):
                await runners[runner_id].start_task(task)
        except TimeoutError:
            failures.append(
                RunnerTaskStartFailure(
                    runner_id=runner_id,
                    task_status=TaskStatus.TimedOut,
                    error_message=(
                        f"Timed out after {timeout_seconds:g}s while starting "
                        f"{task.__class__.__name__}"
                    ),
                )
            )
        except Exception as error:
            failures.append(
                RunnerTaskStartFailure(
                    runner_id=runner_id,
                    task_status=TaskStatus.Failed,
                    error_message=(
                        f"{type(error).__qualname__} while starting "
                        f"{task.__class__.__name__}: {error}"
                    ),
                )
            )

    # Start every eligible local rank independently so one failed rank does not
    # cancel a sibling that has already acknowledged the same generation task.
    async with anyio.create_task_group() as task_group:
        for runner_id in runner_ids:
            task_group.start_soon(start_runner_task, runner_id)
    return tuple(sorted(failures, key=lambda failure: failure.runner_id))


def reset_runner_backoff_for_instance(
    runner_backoff: KeyedBackoff[RunnerId], instance: Instance
) -> None:
    for runner_id in instance.shard_assignments.runner_to_shard:
        runner_backoff.reset(runner_id)


def get_assigned_local_runner_ids(
    instance: Instance,
    node_id: NodeId,
    node_compute_resources: Mapping[NodeId, Sequence[ComputeResource]],
) -> tuple[RunnerId, ...]:
    assignments = instance.shard_assignments
    if not assignments.compute_resource_to_runner:
        runner_id = assignments.node_to_runner.get(node_id)
        return () if runner_id is None else (runner_id,)

    resource_owners = assignments.compute_resource_to_node
    if resource_owners:
        runner_ids = {
            runner_id
            for resource_id, runner_id in assignments.compute_resource_to_runner.items()
            if resource_owners.get(resource_id) == node_id
        }
    else:
        local_resource_ids = {
            resource.resource_id for resource in node_compute_resources.get(node_id, ())
        }
        runner_ids = {
            runner_id
            for resource_id, runner_id in assignments.compute_resource_to_runner.items()
            if resource_id in local_resource_ids
        }
    return tuple(
        sorted(
            runner_ids,
            key=lambda runner_id: assignments.runner_to_shard[runner_id].device_rank,
        )
    )


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[IndexedEvent],
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[Base64ImageHash, Base64Image] = {}

        self._download_backoff: KeyedBackoff[ModelSnapshotId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._runner_backoff: KeyedBackoff[RunnerId] = KeyedBackoff(base=0.5, cap=10.0)
        self._runner_lifecycle_lock = anyio.Lock()
        self._stopped: anyio.Event = anyio.Event()

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._reconcile_custom_cards)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                await self.event_sender.send(
                    NodeGatheredInfo(
                        node_id=self.node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=info,
                    )
                )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for indexed_event in events:
                await self._apply_indexed_event(indexed_event)
                event = indexed_event.event

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )
                    if (
                        len(self.input_chunk_buffer[cmd_id])
                        == self.input_chunk_counts[cmd_id]
                    ):
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in self.input_chunk_buffer[cmd_id].values():
                            per_image[chunk.image_index].append(chunk)
                        for chunks_for_image in per_image.values():
                            sorted_chunks = sorted(
                                chunks_for_image, key=lambda c: c.chunk_index
                            )
                            img = Base64Image("".join(c.data for c in sorted_chunks))
                            self.image_cache[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img

    async def _apply_indexed_event(self, indexed_event: IndexedEvent) -> None:
        event = indexed_event.event
        if isinstance(event, InstanceDeleted):
            shutdown_acknowledgements: tuple[RunnerId, ...] = ()
            async with self._runner_lifecycle_lock:
                deleted_instance = self.state.instances.get(event.instance_id)
                self.state = apply(self.state, event=indexed_event)
                self._instance_backoff.reset(event.instance_id)
                if deleted_instance is not None:
                    reset_runner_backoff_for_instance(
                        self._runner_backoff, deleted_instance
                    )
                    local_runner_ids = get_assigned_local_runner_ids(
                        deleted_instance,
                        self.node_id,
                        self.state.node_compute_resources,
                    )
                    shutdown_acknowledgements = tuple(
                        runner_id
                        for runner_id in local_runner_ids
                        if runner_id not in self.runners
                    )
            for runner_id in shutdown_acknowledgements:
                await self.event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id,
                        runner_status=RunnerShutdown(),
                    )
                )
        else:
            self.state = apply(self.state, event=indexed_event)

    async def _reconcile_custom_cards(self) -> None:
        while True:
            await anyio.sleep(1)
            target = dict(self.state.custom_model_cards)
            for model_id, card in target.items():
                if card_cache.get(model_id, card.revision) == card:
                    continue
                await card_cache.save(card)

            for card in await card_cache.list_all():
                if card.is_custom and target.get(card.model_id) != card:
                    await card_cache.pop(card.model_id, card.revision)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self.image_cache,
                self._instance_backoff,
                self._download_backoff,
                self.state.node_compute_resources,
                self._runner_backoff,
            )
            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                runner_id = task.bound_instance.bound_runner_id
                resource_bound = bool(
                    task.bound_instance.instance.shard_assignments.compute_resource_to_runner
                )
                attempts = (
                    self._runner_backoff.attempts(runner_id)
                    if resource_bound
                    else self._instance_backoff.attempts(iid)
                )
                if attempts >= EXO_MAX_INSTANCE_RETRIES:
                    logger.warning(
                        f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(instance_id=iid),
                        )
                    )
                    continue

            assert task.task_status
            if isinstance(task, CreateRunner):
                await self._start_planned_runner(task)
                continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            if task.task_id not in self.state.tasks:
                await self.event_sender.send(
                    TaskCreated(task_id=task.task_id, task=task)
                )

            # lets not kill the worker if a runner is unresponsive
            match task:
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(
                        model_snapshot_id(shard.model_card)
                    )

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id, shard.model_card
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners[runner_id]
                    graceful_shutdown_received = False
                    try:
                        with fail_after(RUNNER_SHUTDOWN_TIMEOUT_SECONDS):
                            await runner.start_task(task)
                            await runner.wait_for_shutdown_received()
                            graceful_shutdown_received = True
                    except (TimeoutError, anyio.BrokenResourceError):
                        if task.task_id not in runner.completed:
                            await self.event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id,
                                    task_status=TaskStatus.TimedOut,
                                    runner_id=runner_id,
                                )
                            )
                    finally:
                        runner.shutdown()

                    await runner.wait_for_stopped()
                    async with self._runner_lifecycle_lock:
                        if self.runners.get(runner_id) is runner:
                            del self.runners[runner_id]
                        else:
                            logger.warning(
                                f"Runner ownership changed before shutdown for {runner_id}"
                            )
                            continue

                    if (
                        not graceful_shutdown_received
                        and not runner.shutdown_was_forwarded()
                    ):
                        await self.event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerShutdown(),
                            )
                        )
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    await self.runners[runner_id].cancel_task(cancelled_task_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    resolved_images = [
                        self.image_cache[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_card = next(
                            (
                                shard.model_card
                                for shard in instance.shard_assignments.runner_to_shard.values()
                            ),
                            None,
                        )
                        if model_card is not None:
                            self._download_backoff.reset(model_snapshot_id(model_card))

                    await self._start_runner_task(task)
                case task:
                    await self._start_runner_task(task)

    async def _start_planned_runner(self, task: CreateRunner) -> bool:
        async with self._runner_lifecycle_lock:
            current_instance = self.state.instances.get(task.instance_id)
            runner_id = task.bound_instance.bound_runner_id
            if (
                current_instance is None
                or current_instance != task.bound_instance.instance
            ):
                logger.info(
                    f"Skipping stale CreateRunner for deleted instance {task.instance_id}"
                )
                return False
            assigned_runner_ids = get_assigned_local_runner_ids(
                current_instance,
                self.node_id,
                self.state.node_compute_resources,
            )
            if runner_id not in assigned_runner_ids or runner_id in self.runners:
                logger.info(f"Skipping stale CreateRunner for runner {runner_id}")
                return False
            bound_resource_ids = set(task.bound_instance.bound_compute_resource_ids)
            live_resource_ids = {
                resource.resource_id
                for resource in self.state.node_compute_resources.get(self.node_id, ())
            }
            if bound_resource_ids and not bound_resource_ids.issubset(
                live_resource_ids
            ):
                logger.info(
                    f"Skipping CreateRunner for unavailable compute resource on {runner_id}"
                )
                return False

            logger.info(f"Worker plan: {task.__class__.__name__}")
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))
            await self._create_supervisor(task)
            if current_instance.shard_assignments.compute_resource_to_runner:
                self._runner_backoff.record_attempt(runner_id)
            else:
                self._instance_backoff.record_attempt(task.instance_id)

        await self.event_sender.send(
            TaskStatusUpdated(
                task_id=task.task_id,
                task_status=TaskStatus.Complete,
            )
        )
        return True

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            failures = await start_local_runner_task(
                task,
                instance,
                self.node_id,
                self.runners,
            )
            for failure in failures:
                runner_status = RunnerFailed(
                    error_message=failure.error_message,
                    diagnostics=[],
                )
                runner = self.runners.get(failure.runner_id)
                if runner is not None:
                    runner.status = runner_status
                logger.error(
                    f"Runner {failure.runner_id} task start failed: "
                    f"{failure.error_message}"
                )
                await self.event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id,
                        task_status=failure.task_status,
                        runner_id=failure.runner_id,
                    )
                )
                await self.event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=failure.runner_id,
                        runner_status=runner_status,
                    )
                )

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
                api_port=self.api_port,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{self.api_port}")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{self.api_port}"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != self.api_port:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
