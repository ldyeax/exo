import httpx
import pytest

from exo.api.types.api import AddCustomModelParams
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.shards import PipelineShardMetadata
from scripts.download_model_to_cluster import (
    ensure_model_card_registered,
    fetch_topology_nodes,
    node_model_status,
)

MODEL_ID = ModelId("test-org/test-model")
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
NODE_ID = NodeId("node-1")


def _card(revision: str = REVISION) -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        revision=revision,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCpu],
    )


def _download_state(revision: str) -> dict[NodeId, list[DownloadProgress]]:
    card = _card(revision)
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    return {
        NODE_ID: [
            DownloadCompleted(
                node_id=NODE_ID,
                shard_metadata=shard,
                total=card.storage_size,
            )
        ]
    }


def test_node_model_status_requires_exact_revision() -> None:
    assert (
        node_model_status(_download_state(REVISION), NODE_ID, MODEL_ID, REVISION)
        == "completed"
    )
    assert (
        node_model_status(_download_state(OTHER_REVISION), NODE_ID, MODEL_ID, REVISION)
        == "not_present"
    )


@pytest.mark.asyncio
async def test_fetch_topology_nodes_parses_state_snapshot() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/state/topology"
        return httpx.Response(
            200,
            json={"nodes": [str(NODE_ID), "node-2"], "connections": {}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        nodes = await fetch_topology_nodes(client, "http://cluster")

    assert nodes == [NODE_ID, NodeId("node-2")]


@pytest.mark.asyncio
async def test_registration_posts_exact_revision_when_only_main_exists() -> None:
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": str(MODEL_ID),
                            "hugging_face_id": str(MODEL_ID),
                            "revision": "main",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"id": str(MODEL_ID), "revision": REVISION},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        await ensure_model_card_registered(client, "http://cluster", _card())

    assert [request.method for request in requests] == ["GET", "POST"]
    posted = AddCustomModelParams.model_validate_json(requests[1].content)
    assert posted == AddCustomModelParams(model_id=MODEL_ID, revision=REVISION)


@pytest.mark.asyncio
async def test_registration_skips_matching_exact_revision() -> None:
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": str(MODEL_ID),
                        "hugging_face_id": str(MODEL_ID),
                        "revision": REVISION,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        await ensure_model_card_registered(client, "http://cluster", _card())

    assert [request.method for request in requests] == ["GET"]


@pytest.mark.asyncio
async def test_registration_fails_closed_when_server_drops_revision() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"id": str(MODEL_ID)})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        with pytest.raises(RuntimeError, match="did not register"):
            await ensure_model_card_registered(client, "http://cluster", _card())
