from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

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
from exo.utils.channels import channel

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


async def test_ollama_tags_requires_exact_revision() -> None:
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
        AsyncMock(return_value=[exact_card]),
    ):
        result = await api.ollama_tags()

    assert result.models == []


async def test_missing_instance_notifies_when_only_wrong_revision_is_downloaded() -> (
    None
):
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

    with (
        patch.object(
            model_cards.card_cache,
            "list_all",
            AsyncMock(return_value=[exact_card]),
        ),
        patch.object(model_cards.card_cache, "get", return_value=exact_card),
        patch.object(
            api,
            "_trigger_notify_user_to_download_model",
            AsyncMock(),
        ) as notify,
        pytest.raises(HTTPException, match="No instance found"),
    ):
        await api._validate_model_has_instance(  # pyright: ignore[reportPrivateUsage]
            MODEL_ID
        )

    notify.assert_awaited_once_with(MODEL_ID)


async def test_add_custom_model_fetches_and_broadcasts_exact_revision() -> None:
    api = object.__new__(API)
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    command_sender, command_receiver = channel[ForwarderCommand]()
    api.command_sender = command_sender
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
    forwarded_commands = command_receiver.collect()
    assert len(forwarded_commands) == 1
    forwarded = forwarded_commands[0]
    assert isinstance(forwarded, ForwarderCommand)
    assert isinstance(forwarded.command, AddCustomModelCard)
    assert forwarded.command.model_card == card
    assert result.revision == REVISION
