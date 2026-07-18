import pytest

from exo.shared.apply import (
    apply_task_created,
    apply_task_deleted,
    apply_task_status_updated,
)
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import TaskCreated, TaskDeleted, TaskStatusUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import LoadModel, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _generation_state() -> tuple[State, TextGeneration, tuple[RunnerId, ...]]:
    runner_ids = tuple(RunnerId(f"runner-{rank}") for rank in range(3))
    node_ids = tuple(NodeId(f"node-{rank}") for rank in range(3))
    model_card = ModelCard(
        model_id=ModelId("rank-status-model"),
        revision="a" * 40,
        storage_size=Memory.from_mb(1),
        n_layers=3,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    shards = {
        runner_id: PipelineShardMetadata(
            model_card=model_card,
            device_rank=rank,
            world_size=3,
            start_layer=rank,
            end_layer=rank + 1,
            n_layers=3,
        )
        for rank, runner_id in enumerate(runner_ids)
    }
    instance = MlxRingInstance(
        instance_id=InstanceId("rank-status-instance"),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard=shards,
            node_to_runner=dict(zip(node_ids, runner_ids, strict=True)),
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    task = TextGeneration(
        task_id=TaskId("rank-status-task"),
        instance_id=instance.instance_id,
        command_id=CommandId("rank-status-command"),
        task_params=TextGenerationTaskParams(
            model=model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("test"))],
        ),
    )
    state = State(instances={instance.instance_id: instance})
    state = apply_task_created(
        TaskCreated(task_id=task.task_id, task=task),
        state,
    )
    return state, task, runner_ids


def _update(
    state: State,
    task_id: TaskId,
    runner_id: RunnerId,
    status: TaskStatus,
) -> State:
    return apply_task_status_updated(
        TaskStatusUpdated(
            task_id=task_id,
            task_status=status,
            runner_id=runner_id,
        ),
        state,
    )


def test_generation_completes_only_after_every_rank() -> None:
    state, task, runner_ids = _generation_state()

    assert state.task_runner_statuses[task.task_id] == {
        runner_id: TaskStatus.Pending for runner_id in runner_ids
    }

    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Running)
    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Pending)
    assert state.task_runner_statuses[task.task_id][runner_ids[0]] == TaskStatus.Running

    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Complete)
    state = _update(state, task.task_id, runner_ids[1], TaskStatus.Complete)
    assert state.tasks[task.task_id].task_status == TaskStatus.Running

    state = _update(state, task.task_id, runner_ids[2], TaskStatus.Complete)
    assert state.tasks[task.task_id].task_status == TaskStatus.Complete


@pytest.mark.parametrize(
    "terminal_status",
    [TaskStatus.Failed, TaskStatus.TimedOut, TaskStatus.Cancelled],
)
def test_rank_failure_or_cancellation_is_globally_sticky(
    terminal_status: TaskStatus,
) -> None:
    state, task, runner_ids = _generation_state()

    state = _update(state, task.task_id, runner_ids[0], terminal_status)
    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Complete)
    state = _update(state, task.task_id, runner_ids[1], TaskStatus.Complete)
    state = _update(state, task.task_id, runner_ids[2], TaskStatus.Complete)

    assert state.tasks[task.task_id].task_status == terminal_status
    assert state.task_runner_statuses[task.task_id][runner_ids[0]] == terminal_status


def test_global_completion_is_sticky_against_late_rank_failure() -> None:
    state, task, runner_ids = _generation_state()
    for runner_id in runner_ids:
        state = _update(state, task.task_id, runner_id, TaskStatus.Complete)

    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Failed)

    assert state.tasks[task.task_id].task_status == TaskStatus.Complete
    assert (
        state.task_runner_statuses[task.task_id][runner_ids[0]] == TaskStatus.Complete
    )


def test_scoped_status_rejects_unknown_or_untargeted_runner() -> None:
    state, task, runner_ids = _generation_state()

    with pytest.raises(ValueError, match="not expected"):
        _update(state, task.task_id, RunnerId("unknown"), TaskStatus.Complete)

    targeted = LoadModel(
        task_id=TaskId("targeted-load"),
        instance_id=task.instance_id,
        runner_id=runner_ids[0],
    )
    state = apply_task_created(
        TaskCreated(task_id=targeted.task_id, task=targeted),
        state,
    )
    with pytest.raises(ValueError, match="not expected"):
        _update(state, targeted.task_id, runner_ids[1], TaskStatus.Complete)


def test_targeted_and_legacy_unscoped_statuses_remain_compatible() -> None:
    state, task, runner_ids = _generation_state()
    targeted = LoadModel(
        task_id=TaskId("targeted-load"),
        instance_id=task.instance_id,
        runner_id=runner_ids[1],
    )
    state = apply_task_created(
        TaskCreated(task_id=targeted.task_id, task=targeted),
        state,
    )
    state = _update(state, targeted.task_id, runner_ids[1], TaskStatus.Complete)
    assert state.tasks[targeted.task_id].task_status == TaskStatus.Complete

    state = apply_task_status_updated(
        TaskStatusUpdated(
            task_id=task.task_id,
            task_status=TaskStatus.Complete,
        ),
        state,
    )
    state = apply_task_status_updated(
        TaskStatusUpdated(
            task_id=task.task_id,
            task_status=TaskStatus.Running,
        ),
        state,
    )
    assert state.tasks[task.task_id].task_status == TaskStatus.Complete


def test_duplicate_create_is_idempotent_and_delete_cleans_rank_statuses() -> None:
    state, task, runner_ids = _generation_state()
    created = TaskCreated(task_id=task.task_id, task=task)

    assert apply_task_created(created, state) is state
    state = _update(state, task.task_id, runner_ids[0], TaskStatus.Running)
    assert state.tasks[task.task_id].task_status == TaskStatus.Running
    assert apply_task_created(created, state) is state
    assert state.tasks[task.task_id].task_status == TaskStatus.Running

    with pytest.raises(ValueError, match="conflicting data"):
        apply_task_created(
            TaskCreated(
                task_id=task.task_id,
                task=task.model_copy(update={"instance_id": InstanceId("other")}),
            ),
            state,
        )

    state = apply_task_deleted(TaskDeleted(task_id=task.task_id), state)
    assert task.task_id not in state.tasks
    assert task.task_id not in state.task_runner_statuses
