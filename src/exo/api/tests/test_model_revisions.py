from unittest.mock import AsyncMock, patch

from exo.api.main import API
from exo.api.types.api import AddCustomModelParams
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import AddCustomModelCard, ForwarderCommand
from exo.shared.types.common import ModelId, NodeId, SystemId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.shards import PipelineShardMetadata

MODEL_ID = ModelId("test-org/test-model")
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _card() -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        revision=REVISION,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCpu],
    )


async def test_get_models_exposes_exact_revision() -> None:
    api = object.__new__(API)
    api.state = State()

    with patch.object(
        model_cards.card_cache, "list_all", AsyncMock(return_value=[_card()])
    ):
        result = await api.get_models()

    assert len(result.data) == 1
    assert result.data[0].id == MODEL_ID
    assert result.data[0].revision == REVISION


async def test_get_downloaded_models_requires_exact_revision() -> None:
    exact_card = _card()
    main_card = exact_card.model_copy(update={"revision": "main"})
    shard = PipelineShardMetadata(
        model_card=main_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=main_card.n_layers,
        n_layers=main_card.n_layers,
    )
    node_id = NodeId("node-1")
    api = object.__new__(API)
    api.state = State(
        downloads={
            node_id: [
                DownloadCompleted(
                    node_id=node_id,
                    shard_metadata=shard,
                    total=main_card.storage_size,
                )
            ]
        }
    )

    with patch.object(
        model_cards.card_cache,
        "list_all",
        AsyncMock(return_value=[exact_card, main_card]),
    ):
        result = await api.get_models(status="downloaded")

    assert [(model.id, model.revision) for model in result.data] == [(MODEL_ID, "main")]


async def test_add_custom_model_fetches_and_broadcasts_exact_revision() -> None:
    api = object.__new__(API)
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender = AsyncMock()
    card = _card()

    with (
        patch.object(
            ModelCard,
            "fetch_from_hf",
            AsyncMock(return_value=card),
        ) as fetch,
        patch.object(model_cards.card_cache, "add_to_memory") as add_to_memory,
    ):
        result = await api.add_custom_model(
            AddCustomModelParams(model_id=MODEL_ID, revision=REVISION)
        )

    fetch.assert_awaited_once_with(MODEL_ID, REVISION)
    add_to_memory.assert_called_once_with(card)
    api.command_sender.send.assert_awaited_once()
    forwarded = api.command_sender.send.await_args.args[0]
    assert isinstance(forwarded, ForwarderCommand)
    assert isinstance(forwarded.command, AddCustomModelCard)
    assert forwarded.command.model_card == card
    assert result.revision == REVISION
