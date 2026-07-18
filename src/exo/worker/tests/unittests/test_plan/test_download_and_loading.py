import exo.worker.plan as plan_mod
from exo.shared.models.model_cards import HuggingFaceRevision
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import LoadModel
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerIdle,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

REVISION: HuggingFaceRevision = "0123456789abcdef0123456789abcdef01234567"


def _with_revision(
    shard: ShardMetadata, revision: HuggingFaceRevision
) -> ShardMetadata:
    return shard.model_copy(
        update={
            "model_card": shard.model_card.model_copy(update={"revision": revision})
        }
    )


def test_plan_requests_download_when_waiting_and_shard_not_downloaded():
    """
    When a runner is waiting for a model and its shard is not in the
    local download_status map, plan() should emit DownloadModel.
    """

    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, plan_mod.DownloadModel)
    assert result.instance_id == INSTANCE_1_ID
    assert result.shard_metadata == shard


def test_plan_loads_model_when_all_shards_downloaded_and_waiting():
    """
    When all shards for an instance are DownloadCompleted (globally) and
    all runners are in waiting/loading/loaded states, plan() should emit
    LoadModel once.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}

    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id is None


def test_plan_does_not_request_download_when_shard_already_downloaded():
    """
    If the local shard already has a DownloadCompleted entry, plan()
    should not re-emit DownloadModel while global state is still catching up.
    """
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    # Global state shows shard is downloaded for NODE_A
    global_download_status: dict[NodeId, list[DownloadProgress]] = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert not isinstance(result, plan_mod.DownloadModel)


def test_wrong_revision_does_not_suppress_pinned_download() -> None:
    main_shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    pinned_shard = _with_revision(main_shard, REVISION)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: pinned_shard},
    )
    runner = FakeRunnerSupervisor(
        bound_instance=BoundInstance(
            instance=instance,
            bound_runner_id=RUNNER_1_ID,
            bound_node_id=NODE_A,
        ),
        status=RunnerIdle(),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={
            NODE_A: [
                DownloadCompleted(
                    shard_metadata=main_shard,
                    node_id=NODE_A,
                    total=Memory(),
                )
            ]
        },
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerIdle()},
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, plan_mod.DownloadModel)
    assert result.shard_metadata.model_card.revision == REVISION


def test_wrong_revision_does_not_authorize_pinned_load() -> None:
    main_shard_a = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    main_shard_b = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    pinned_shard_a = _with_revision(main_shard_a, REVISION)
    pinned_shard_b = _with_revision(main_shard_b, REVISION)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={
            RUNNER_1_ID: pinned_shard_a,
            RUNNER_2_ID: pinned_shard_b,
        },
    )
    runner = FakeRunnerSupervisor(
        bound_instance=BoundInstance(
            instance=instance,
            bound_runner_id=RUNNER_1_ID,
            bound_node_id=NODE_A,
        ),
        status=RunnerConnected(),
    )

    wrong_revision_result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={
            NODE_A: [
                DownloadCompleted(
                    shard_metadata=pinned_shard_a,
                    node_id=NODE_A,
                    total=Memory(),
                )
            ],
            NODE_B: [
                DownloadCompleted(
                    shard_metadata=main_shard_b,
                    node_id=NODE_B,
                    total=Memory(),
                )
            ],
        },
        instances={INSTANCE_1_ID: instance},
        all_runners={
            RUNNER_1_ID: RunnerConnected(),
            RUNNER_2_ID: RunnerConnected(),
        },
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )
    exact_revision_result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={
            NODE_A: [
                DownloadCompleted(
                    shard_metadata=pinned_shard_a,
                    node_id=NODE_A,
                    total=Memory(),
                )
            ],
            NODE_B: [
                DownloadCompleted(
                    shard_metadata=pinned_shard_b,
                    node_id=NODE_B,
                    total=Memory(),
                )
            ],
        },
        instances={INSTANCE_1_ID: instance},
        all_runners={
            RUNNER_1_ID: RunnerConnected(),
            RUNNER_2_ID: RunnerConnected(),
        },
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert wrong_revision_result is None
    assert isinstance(exact_revision_result, LoadModel)


def test_plan_does_not_load_model_until_all_shards_downloaded_globally():
    """
    LoadModel should not be emitted while some shards are still missing from
    the global_download_status.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is not None
