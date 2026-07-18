from exo.shared.apply import apply_node_download_progress
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.worker.tests.constants import MODEL_A_ID, MODEL_B_ID

REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_apply_node_download_progress():
    state = State()
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    event = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event]}


def test_apply_two_node_download_progress():
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0, world_size=2)
    event1 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard1,
        total=Memory(),
    )
    event2 = DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=shard2,
        total=Memory(),
    )
    state = State(downloads={NodeId("node-1"): [event1]})

    new_state = apply_node_download_progress(
        NodeDownloadProgress(download_progress=event2), state
    )

    assert new_state.downloads == {NodeId("node-1"): [event1, event2]}


def test_apply_download_progress_keeps_distinct_model_revisions() -> None:
    node_id = NodeId("node-1")
    main_shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    pinned_shard = main_shard.model_copy(
        update={
            "model_card": main_shard.model_card.model_copy(
                update={"revision": REVISION}
            )
        }
    )
    main_completed = DownloadCompleted(
        node_id=node_id,
        shard_metadata=main_shard,
        total=Memory.from_bytes(10),
    )
    pinned_completed = DownloadCompleted(
        node_id=node_id,
        shard_metadata=pinned_shard,
        total=Memory.from_bytes(20),
    )

    with_both = apply_node_download_progress(
        NodeDownloadProgress(download_progress=pinned_completed),
        State(downloads={node_id: [main_completed]}),
    )

    assert with_both.downloads[node_id] == [main_completed, pinned_completed]

    pinned_pending = DownloadPending(
        node_id=node_id,
        shard_metadata=pinned_shard,
        total=Memory.from_bytes(20),
    )
    updated = apply_node_download_progress(
        NodeDownloadProgress(download_progress=pinned_pending), with_both
    )

    assert updated.downloads[node_id] == [main_completed, pinned_pending]
