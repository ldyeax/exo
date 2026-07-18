import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeTimedOut,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TestEvent,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.profiling import (
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.topology import Connection, RDMAConnection
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerShutdown,
    RunnerStatus,
)
from exo.utils.info_gatherer.info_gatherer import (
    MacmonMetrics,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    MemoryUsage,
    MiscData,
    NodeBackends,
    NodeComputeResources,
    NodeConfig,
    NodeDiskUsage,
    NodeNetworkInterfaces,
    RdmaCtlStatus,
    StaticNodeInformation,
    ThunderboltBridgeInfo,
)


def _is_rdma_ctl_enabled(
    node_id: NodeId, node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus]
) -> bool:
    """A node is RDMA-capable only if rdma_ctl status has been observed as enabled.

    Missing entries default to ``False`` — if we have not yet observed (or the node
    cannot run) ``rdma_ctl``, it must not participate in an RDMA-backed instance.
    """
    status = node_rdma_ctl.get(node_id)
    return status is not None and status.enabled


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent()
            | ChunkGenerated()
            | TaskAcknowledged()
            | InputChunkReceived()
            | TracesCollected()
            | TracesMerged()
        ):  # Pass-through events that don't modify state
            return state
        case CustomModelCardAdded():
            return apply_custom_model_card_added(event, state)
        case CustomModelCardDeleted():
            return apply_custom_model_card_deleted(event, state)
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeGatheredInfo():
            return apply_node_gathered_info(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)
        case InstanceLinkCreated():
            return apply_instance_link_created(event, state)
        case InstanceLinkDeleted():
            return apply_instance_link_deleted(event, state)


def apply(state: State, event: IndexedEvent) -> State:
    # Just to test that events are only applied in correct order
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(
            f"Expected event {state.last_event_applied_idx + 1} but received {event.idx}"
        )
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})


def apply_node_download_progress(event: NodeDownloadProgress, state: State) -> State:
    """
    Update or add a node download progress to state.
    """
    dp = event.download_progress
    node_id = dp.node_id

    current = list(state.downloads.get(node_id, ()))

    replaced = False
    for i, existing_dp in enumerate(current):
        # TODO(ciaran): deduplicate by model_id for now. Will need to use
        # shard_metadata again when pipeline and tensor downloads differ.
        # For now this is fine
        if (
            existing_dp.shard_metadata.model_card.model_id
            == dp.shard_metadata.model_card.model_id
        ):
            current[i] = dp
            replaced = True
            break

    if not replaced:
        current.append(dp)

    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        node_id: current,
    }
    return state.model_copy(update={"downloads": new_downloads})


def apply_task_created(event: TaskCreated, state: State) -> State:
    existing_task = state.tasks.get(event.task_id)
    if existing_task is not None:
        if existing_task != event.task:
            raise ValueError(f"Task {event.task_id} was created with conflicting data")
        return state

    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    expected_runner_ids = _expected_task_runner_ids(event.task, state)
    new_task_runner_statuses = {
        **state.task_runner_statuses,
        event.task_id: {
            runner_id: event.task.task_status for runner_id in expected_runner_ids
        },
    }
    return state.model_copy(
        update={
            "tasks": new_tasks,
            "task_runner_statuses": new_task_runner_statuses,
        }
    )


def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {
        tid: task for tid, task in state.tasks.items() if tid != event.task_id
    }
    new_task_runner_statuses = {
        task_id: statuses
        for task_id, statuses in state.task_runner_statuses.items()
        if task_id != event.task_id
    }
    return state.model_copy(
        update={
            "tasks": new_tasks,
            "task_runner_statuses": new_task_runner_statuses,
        }
    )


_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.Complete,
        TaskStatus.TimedOut,
        TaskStatus.Failed,
        TaskStatus.Cancelled,
    }
)


def _expected_task_runner_ids(task: Task, state: State) -> tuple[RunnerId, ...]:
    instance = state.instances.get(task.instance_id)
    match task:
        case TextGeneration() | ImageGeneration() | ImageEdits():
            if instance is None:
                return ()
            return tuple(instance.shard_assignments.runner_to_shard)
        case CreateRunner(bound_instance=bound_instance):
            return (bound_instance.bound_runner_id,)
        case (
            ConnectToGroup(runner_id=runner_id)
            | LoadModel(runner_id=runner_id)
            | StartWarmup(runner_id=runner_id)
        ) if runner_id is not None:
            return (runner_id,)
        case CancelTask(runner_id=runner_id) | Shutdown(runner_id=runner_id):
            return (runner_id,)
        case _:
            return ()


def _advance_runner_task_status(
    current: TaskStatus, incoming: TaskStatus
) -> TaskStatus:
    if current in _TERMINAL_TASK_STATUSES:
        return current
    if incoming in _TERMINAL_TASK_STATUSES:
        return incoming
    if current == TaskStatus.Running and incoming == TaskStatus.Pending:
        return current
    return incoming


def _aggregate_runner_task_statuses(
    statuses: Mapping[RunnerId, TaskStatus],
) -> TaskStatus:
    values = tuple(statuses.values())
    for terminal_status in (
        TaskStatus.Failed,
        TaskStatus.TimedOut,
        TaskStatus.Cancelled,
    ):
        if terminal_status in values:
            return terminal_status
    if values and all(status == TaskStatus.Complete for status in values):
        return TaskStatus.Complete
    if any(status != TaskStatus.Pending for status in values):
        return TaskStatus.Running
    return TaskStatus.Pending


def apply_task_status_updated(event: TaskStatusUpdated, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    task = state.tasks[event.task_id]
    if event.runner_id is None:
        next_status = (
            task.task_status
            if task.task_status in _TERMINAL_TASK_STATUSES
            else event.task_status
        )
        update: dict[str, object] = {"task_status": next_status}
        if next_status != TaskStatus.Failed:
            update["error_type"] = None
            update["error_message"] = None
        updated_task = task.model_copy(update=update)
        legacy_tasks: Mapping[TaskId, Task] = {
            **state.tasks,
            event.task_id: updated_task,
        }
        return state.model_copy(update={"tasks": legacy_tasks})

    instance = state.instances.get(task.instance_id)
    expected_runner_ids = _expected_task_runner_ids(task, state)
    current_statuses = dict(state.task_runner_statuses.get(event.task_id, {}))
    assigned_runner_ids: set[RunnerId] = (
        set(instance.shard_assignments.runner_to_shard)
        if instance is not None
        else set()
    )
    known_runner_ids: set[RunnerId] = (
        set(expected_runner_ids) or assigned_runner_ids or set(current_statuses)
    )
    if event.runner_id not in known_runner_ids:
        raise ValueError(
            f"Runner {event.runner_id} is not expected for task {event.task_id}"
        )

    if not current_statuses and expected_runner_ids:
        current_statuses = {
            runner_id: TaskStatus.Pending for runner_id in expected_runner_ids
        }
    if expected_runner_ids and event.runner_id not in expected_runner_ids:
        raise ValueError(
            f"Runner {event.runner_id} is not expected for task {event.task_id}"
        )

    current_runner_status = current_statuses.get(event.runner_id, TaskStatus.Pending)
    current_statuses[event.runner_id] = _advance_runner_task_status(
        current_runner_status, event.task_status
    )
    aggregate_status = _aggregate_runner_task_statuses(current_statuses)
    next_status = (
        task.task_status
        if task.task_status in _TERMINAL_TASK_STATUSES
        else aggregate_status
    )
    update = {"task_status": next_status}
    if next_status != TaskStatus.Failed:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = task.model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    new_task_runner_statuses = {
        **state.task_runner_statuses,
        event.task_id: current_statuses,
    }
    return state.model_copy(
        update={
            "tasks": new_tasks,
            "task_runner_statuses": new_task_runner_statuses,
        }
    )


def apply_task_failed(event: TaskFailed, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    updated_task = state.tasks[event.task_id].model_copy(
        update={"error_type": event.error_type, "error_message": event.error_message}
    )
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_instance_created(event: InstanceCreated, state: State) -> State:
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {
        **state.instances,
        instance.instance_id: instance,
    }
    return state.model_copy(update={"instances": new_instances})


def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    new_instances: Mapping[InstanceId, Instance] = {
        iid: inst for iid, inst in state.instances.items() if iid != event.instance_id
    }
    new_links: dict[InstanceLinkId, InstanceLink] = {}
    for link_id, link in state.instance_links.items():
        prefill = [i for i in link.prefill_instances if i != event.instance_id]
        decode = [i for i in link.decode_instances if i != event.instance_id]
        if not prefill or not decode:
            continue
        if prefill == list(link.prefill_instances) and decode == list(
            link.decode_instances
        ):
            new_links[link_id] = link
        else:
            new_links[link_id] = link.model_copy(
                update={"prefill_instances": prefill, "decode_instances": decode}
            )
    return state.model_copy(
        update={"instances": new_instances, "instance_links": new_links}
    )


def apply_instance_link_created(event: InstanceLinkCreated, state: State) -> State:
    new_links: Mapping[InstanceLinkId, InstanceLink] = {
        **state.instance_links,
        event.link.link_id: event.link,
    }
    return state.model_copy(update={"instance_links": new_links})


def apply_instance_link_deleted(event: InstanceLinkDeleted, state: State) -> State:
    new_links: Mapping[InstanceLinkId, InstanceLink] = {
        lid: link for lid, link in state.instance_links.items() if lid != event.link_id
    }
    return state.model_copy(update={"instance_links": new_links})


def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    if isinstance(event.runner_status, RunnerShutdown):
        new_runners: Mapping[RunnerId, RunnerStatus] = {
            rid: rs for rid, rs in state.runners.items() if rid != event.runner_id
        }
        new_ports: Mapping[RunnerId, int] = {
            rid: p
            for rid, p in state.prefill_server_ports.items()
            if rid != event.runner_id
        }
        return state.model_copy(
            update={"runners": new_runners, "prefill_server_ports": new_ports}
        )
    new_runners = {
        **state.runners,
        event.runner_id: event.runner_status,
    }
    update: dict[str, object] = {"runners": new_runners}
    if (
        isinstance(event.runner_status, RunnerReady)
        and event.runner_status.prefill_server_port is not None
    ):
        update["prefill_server_ports"] = {
            **state.prefill_server_ports,
            event.runner_id: event.runner_status.prefill_server_port,
        }
    return state.model_copy(update=update)


def apply_node_timed_out(event: NodeTimedOut, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_node(event.node_id)
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    downloads = {
        key: value for key, value in state.downloads.items() if key != event.node_id
    }
    # Clean up all granular node mappings
    node_memory = {
        key: value for key, value in state.node_memory.items() if key != event.node_id
    }
    node_disk = {
        key: value for key, value in state.node_disk.items() if key != event.node_id
    }
    node_system = {
        key: value for key, value in state.node_system.items() if key != event.node_id
    }
    node_network = {
        key: value for key, value in state.node_network.items() if key != event.node_id
    }
    node_thunderbolt = {
        key: value
        for key, value in state.node_thunderbolt.items()
        if key != event.node_id
    }
    node_thunderbolt_bridge = {
        key: value
        for key, value in state.node_thunderbolt_bridge.items()
        if key != event.node_id
    }
    node_rdma_ctl = {
        key: value for key, value in state.node_rdma_ctl.items() if key != event.node_id
    }
    node_compute_resources = {
        key: value
        for key, value in state.node_compute_resources.items()
        if key != event.node_id
    }
    # Only recompute cycles if the leaving node had TB bridge enabled
    leaving_node_status = state.node_thunderbolt_bridge.get(event.node_id)
    leaving_node_had_tb_enabled = (
        leaving_node_status is not None and leaving_node_status.enabled
    )
    thunderbolt_bridge_cycles = (
        topology.get_thunderbolt_bridge_cycles(node_thunderbolt_bridge, node_network)
        if leaving_node_had_tb_enabled
        else [list(cycle) for cycle in state.thunderbolt_bridge_cycles]
    )
    return state.model_copy(
        update={
            "downloads": downloads,
            "topology": topology,
            "last_seen": last_seen,
            "node_memory": node_memory,
            "node_disk": node_disk,
            "node_system": node_system,
            "node_network": node_network,
            "node_thunderbolt": node_thunderbolt,
            "node_thunderbolt_bridge": node_thunderbolt_bridge,
            "node_rdma_ctl": node_rdma_ctl,
            "node_compute_resources": node_compute_resources,
            "thunderbolt_bridge_cycles": thunderbolt_bridge_cycles,
        }
    )


def apply_node_gathered_info(event: NodeGatheredInfo, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_node(event.node_id)
    info = event.info

    # Build update dict with only the mappings that change
    update: dict[str, object] = {
        "last_seen": {
            **state.last_seen,
            event.node_id: datetime.fromisoformat(event.when),
        },
        "topology": topology,
    }

    match info:
        case MacmonMetrics():
            update["node_system"] = {
                **state.node_system,
                event.node_id: info.system_profile,
            }
            update["node_memory"] = {**state.node_memory, event.node_id: info.memory}
        case MemoryUsage():
            update["node_memory"] = {**state.node_memory, event.node_id: info}
        case NodeDiskUsage():
            update["node_disk"] = {**state.node_disk, event.node_id: info.disk_usage}
        case NodeConfig():
            pass
        case MiscData():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={"friendly_name": info.friendly_name}
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case StaticNodeInformation():
            current_identity = state.node_identities.get(event.node_id, NodeIdentity())
            new_identity = current_identity.model_copy(
                update={
                    "model_id": info.model,
                    "chip_id": info.chip,
                    "os_version": info.os_version,
                    "os_build_version": info.os_build_version,
                }
            )
            update["node_identities"] = {
                **state.node_identities,
                event.node_id: new_identity,
            }
        case NodeNetworkInterfaces():
            update["node_network"] = {
                **state.node_network,
                event.node_id: NodeNetworkInfo(interfaces=info.ifaces),
            }
        case MacThunderboltIdentifiers():
            update["node_thunderbolt"] = {
                **state.node_thunderbolt,
                event.node_id: NodeThunderboltInfo(interfaces=info.idents),
            }
        case MacThunderboltConnections():
            conn_map = {
                tb_ident.domain_uuid: (nid, tb_ident.rdma_interface)
                for nid in state.node_thunderbolt
                for tb_ident in state.node_thunderbolt[nid].interfaces
            }
            source_is_rdma_enabled = _is_rdma_ctl_enabled(
                event.node_id, state.node_rdma_ctl
            )
            as_rdma_conns = [
                Connection(
                    source=event.node_id,
                    sink=conn_map[tb_conn.sink_uuid][0],
                    edge=RDMAConnection(
                        source_rdma_iface=conn_map[tb_conn.source_uuid][1],
                        sink_rdma_iface=conn_map[tb_conn.sink_uuid][1],
                    ),
                )
                for tb_conn in info.conns
                if tb_conn.source_uuid in conn_map
                if tb_conn.sink_uuid in conn_map
                if source_is_rdma_enabled
                and _is_rdma_ctl_enabled(
                    conn_map[tb_conn.sink_uuid][0], state.node_rdma_ctl
                )
            ]
            topology.replace_all_out_rdma_connections(event.node_id, as_rdma_conns)
        case ThunderboltBridgeInfo():
            new_tb_bridge: dict[NodeId, ThunderboltBridgeStatus] = {
                **state.node_thunderbolt_bridge,
                event.node_id: info.status,
            }
            update["node_thunderbolt_bridge"] = new_tb_bridge
            # Only recompute cycles if the enabled status changed
            old_status = state.node_thunderbolt_bridge.get(event.node_id)
            old_enabled = old_status.enabled if old_status else False
            new_enabled = info.status.enabled
            if old_enabled != new_enabled:
                update["thunderbolt_bridge_cycles"] = (
                    topology.get_thunderbolt_bridge_cycles(
                        new_tb_bridge, state.node_network
                    )
                )
        case RdmaCtlStatus():
            update["node_rdma_ctl"] = {
                **state.node_rdma_ctl,
                event.node_id: NodeRdmaCtlStatus(enabled=info.enabled),
            }
            # If RDMA just got disabled on this node, drop any RDMA edges touching it
            # so placement / topology consumers cannot pick a disabled node for an
            # RDMA-backed instance. (Edges will repopulate on the next
            # MacThunderboltConnections poll once both endpoints are enabled again.)
            if not info.enabled:
                topology.remove_all_rdma_connections_touching(event.node_id)
        case NodeBackends():
            update["node_backends"] = {
                **state.node_backends,
                event.node_id: info.backends,
            }
        case NodeComputeResources():
            update["node_compute_resources"] = {
                **state.node_compute_resources,
                event.node_id: info.resources,
            }

    return state.model_copy(update=update)


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_connection(event.conn)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_connection(event.conn)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})


def apply_custom_model_card_added(event: CustomModelCardAdded, state: State) -> State:
    new_cards: Mapping[ModelId, ModelCard] = {
        **state.custom_model_cards,
        event.model_card.model_id: event.model_card,
    }
    return state.model_copy(update={"custom_model_cards": new_cards})


def apply_custom_model_card_deleted(
    event: CustomModelCardDeleted, state: State
) -> State:
    new_cards: Mapping[ModelId, ModelCard] = {
        model_id: card
        for model_id, card in state.custom_model_cards.items()
        if model_id != event.model_id
    }
    return state.model_copy(update={"custom_model_cards": new_cards})
