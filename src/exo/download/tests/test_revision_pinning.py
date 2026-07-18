"""CPU-only tests for exact Hugging Face revision pinning."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import tomlkit
from anyio import Path as AsyncPath
from pydantic import ValidationError

from exo.download.download_utils import (
    ModelRevisionMismatchError,
    build_model_path,
    download_shard,
    migrate_hugging_face_local_dir_to_pinned_revision,
    resolve_model_dir,
)
from exo.download.impl_shard_downloader import (
    ResumableShardDownloader,
    build_full_shard,
)
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    VisionCardConfig,
    validate_hugging_face_revision,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata

MODEL_ID = ModelId("test-org/test-model")
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"

PINNED_NCCL_MODEL_CARDS: tuple[tuple[str, ModelId, str, int], ...] = (
    (
        "mlx-community--SmolLM2-135M-Instruct-8bit.toml",
        ModelId("mlx-community/SmolLM2-135M-Instruct-8bit"),
        "0f0d9b8218915bc34d401e1a340b8c049d300d5e",
        142955136,
    ),
    (
        "mlx-community--Llama-3.2-1B-Instruct-4bit.toml",
        ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        "08231374eeacb049a0eade7922910865b8fce912",
        695242752,
    ),
    (
        "mlx-community--Llama-3.2-3B-Instruct-4bit.toml",
        ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e",
        1807423488,
    ),
    (
        "mlx-community--Llama-3.1-8B-Instruct-4bit.toml",
        ModelId("mlx-community/Llama-3.1-8B-Instruct-4bit"),
        "90215b22ec18e72f623dde2ea7af4097025160e2",
        4517404672,
    ),
    (
        "mlx-community--gpt-oss-20b-MXFP4-Q8.toml",
        ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        "773a7da77e569019bb0fd17a554b263738d669a3",
        12076119168,
    ),
    (
        "mlx-community--GLM-4.7-Flash-4bit.toml",
        ModelId("mlx-community/GLM-4.7-Flash-4bit"),
        "1454cffb1a21737e162f508e5bc70be9def89276",
        16852202496,
    ),
    (
        "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit.toml",
        ModelId("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        "6e302ea604ad9ab206367e2c501d1571023e7b6d",
        17180913664,
    ),
    (
        "mlx-community--Qwen3.5-35B-A3B-4bit.toml",
        ModelId("mlx-community/Qwen3.5-35B-A3B-4bit"),
        "1e20fd8d42056f870933bf98ca6211024744f7ec",
        20391405152,
    ),
    (
        "mlx-community--Llama-3.3-70B-Instruct-4bit.toml",
        ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
        "de2dfaf56839b7d0e834157d2401dee02726874d",
        39688355840,
    ),
    (
        "mlx-community--Qwen3-Coder-Next-4bit.toml",
        ModelId("mlx-community/Qwen3-Coder-Next-4bit"),
        "7b9321eabb85ce79625cac3f61ea691e4ea984b5",
        44844060160,
    ),
)


def _load_builtin_card(filename: str) -> ModelCard:
    card_path = Path(str(model_cards._BUILTIN_CARD_DIRS[0])) / filename  # pyright: ignore[reportPrivateUsage]
    return ModelCard.model_validate(tomlkit.loads(card_path.read_text()))


def _card(revision: str = "main", vision: VisionCardConfig | None = None) -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        revision=revision,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCpu],
        vision=vision,
    )


def _complete_model(model_dir: Path) -> list[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    files = [
        Path("config.json"),
        Path("model.safetensors.index.json"),
        Path("model.safetensors"),
    ]
    (model_dir / files[0]).write_text('{"model_type":"test"}')
    (model_dir / files[1]).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"layer.weight": "model.safetensors"},
            }
        )
    )
    (model_dir / files[2]).write_bytes(b"weights")
    return files


def _write_hf_metadata(model_dir: Path, relative_path: Path, revision: str) -> None:
    metadata_path = (
        model_dir / ".cache/huggingface/download" / f"{relative_path}.metadata"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(f"{revision}\netag\n0.0\n")


def _snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@contextmanager
def _patch_model_dirs(models_dir: Path) -> Iterator[None]:
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
        patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
    ):
        yield


def test_model_card_defaults_to_main_and_requires_exact_commit() -> None:
    assert _card().revision == "main"

    with pytest.raises(ValidationError):
        _card("release-branch")
    with pytest.raises(ValidationError):
        _card(REVISION.upper())
    with pytest.raises(ValidationError):
        _card(REVISION[:-1])


def test_existing_builtin_cards_roundtrip_with_valid_revisions() -> None:
    loaded = 0
    with patch("exo.shared.models.model_cards.EXO_MODELS_DIRS", ()):
        for directory in model_cards._BUILTIN_CARD_DIRS:  # pyright: ignore[reportPrivateUsage]
            for card_path in Path(str(directory)).rglob("*.toml"):
                card = ModelCard.model_validate(tomlkit.loads(card_path.read_text()))
                assert validate_hugging_face_revision(card.revision) == card.revision
                loaded += 1
    assert loaded > 0


@pytest.mark.parametrize(
    ("filename", "model_id", "revision", "storage_size_bytes"),
    PINNED_NCCL_MODEL_CARDS,
)
def test_nccl_model_cards_are_pinned_to_verified_snapshots(
    filename: str,
    model_id: ModelId,
    revision: str,
    storage_size_bytes: int,
) -> None:
    card = _load_builtin_card(filename)

    assert card.model_id == model_id
    assert card.revision == revision
    assert card.storage_size.in_bytes == storage_size_bytes


def test_smol_lm_card_is_compatible_with_three_tensor_ranks() -> None:
    card = _load_builtin_card("mlx-community--SmolLM2-135M-Instruct-8bit.toml")

    assert card.family == "llama"
    assert card.supports_tensor
    assert card.n_layers == 30
    assert card.context_length == 8192
    assert card.hidden_size % 3 == 0
    assert card.num_key_value_heads is not None
    assert card.num_key_value_heads % 3 == 0


def test_llama32_3b_card_is_tp2_compatible_without_remote_code() -> None:
    card = _load_builtin_card("mlx-community--Llama-3.2-3B-Instruct-4bit.toml")

    assert card.family == "llama"
    assert card.supports_tensor
    assert card.hidden_size % 2 == 0
    assert card.num_key_value_heads is not None
    assert card.num_key_value_heads % 2 == 0
    assert not card.trust_remote_code


def test_pinned_qwen35_card_inherits_revision_for_vision_weights() -> None:
    filename = "mlx-community--Qwen3.5-35B-A3B-4bit.toml"
    card = _load_builtin_card(filename)

    assert card.vision is not None
    assert card.vision.image_token_id == 248056
    assert card.vision.model_type == "qwen3_5_moe"
    assert card.vision.weights_repo == str(card.model_id)
    assert card.vision.weights_revision == card.revision


def test_vision_revisions_are_explicit_and_same_repo_inherits_main_pin() -> None:
    same_repo = _card(
        REVISION,
        VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo=str(MODEL_ID),
        ),
    )
    assert same_repo.vision is not None
    assert same_repo.vision.weights_revision == REVISION

    sibling = _card(
        REVISION,
        VisionCardConfig(
            image_token_id=1,
            model_type="test",
            weights_repo="test-org/vision",
            weights_revision=OTHER_REVISION,
            processor_repo="test-org/processor",
            processor_revision=REVISION,
        ),
    )
    main_shard = PipelineShardMetadata(
        model_card=sibling,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    vision_shard = ResumableShardDownloader._build_vision_shard(  # pyright: ignore[reportPrivateUsage]
        main_shard
    )
    assert vision_shard.model_card.revision == OTHER_REVISION
    assert sibling.vision is not None
    assert sibling.vision.processor_revision == REVISION


async def test_revision_aware_card_cache_and_custom_filenames_do_not_collide(
    tmp_path: Path,
) -> None:
    cache = model_cards._CardCache()  # pyright: ignore[reportPrivateUsage]
    main_card = _card()
    pinned_card = _card(REVISION)

    with patch("exo.shared.models.model_cards._custom_cards_dir", tmp_path):
        await cache.save(main_card)
        await cache.save(pinned_card)

    assert cache.get(MODEL_ID) == pinned_card
    assert cache.get(MODEL_ID, "main") == main_card
    assert cache.get(MODEL_ID, REVISION) == pinned_card
    assert (tmp_path / f"{MODEL_ID.normalize()}.toml").is_file()
    pinned_path = tmp_path / f"{MODEL_ID.normalize()}--{REVISION}.toml"
    assert pinned_path.is_file()
    assert (await ModelCard.load_from_path(AsyncPath(pinned_path))).revision == REVISION

    with patch("exo.shared.models.model_cards.card_cache", cache):
        assert await ModelCard.load(MODEL_ID) == pinned_card
        assert await ModelCard.load(MODEL_ID, "main") == main_card
        assert await ModelCard.load(MODEL_ID, REVISION) == pinned_card


async def test_legacy_load_and_shard_builder_select_unique_pinned_card() -> None:
    pinned_card = _card(REVISION)
    cache = model_cards._CardCache()  # pyright: ignore[reportPrivateUsage]
    cache.add_to_memory(pinned_card)
    with patch("exo.shared.models.model_cards.card_cache", cache):
        assert await ModelCard.load(MODEL_ID) == pinned_card
        shard = await build_full_shard(MODEL_ID)

    assert shard.model_card.revision == REVISION


async def test_omitted_revision_prefers_one_pin_and_rejects_multiple_pins() -> None:
    main_card = _card()
    pinned_card = _card(REVISION)
    cache = model_cards._CardCache()  # pyright: ignore[reportPrivateUsage]
    cache.add_to_memory(main_card)
    cache.add_to_memory(pinned_card)
    with patch("exo.shared.models.model_cards.card_cache", cache):
        assert await ModelCard.load(MODEL_ID) == pinned_card
        assert await ModelCard.load(MODEL_ID, "main") == main_card
        assert await ModelCard.load(MODEL_ID, REVISION) == pinned_card

    ambiguous_cache = model_cards._CardCache()  # pyright: ignore[reportPrivateUsage]
    ambiguous_cache.add_to_memory(main_card)
    ambiguous_cache.add_to_memory(pinned_card)
    ambiguous_cache.add_to_memory(_card(OTHER_REVISION))
    with (
        patch("exo.shared.models.model_cards.card_cache", ambiguous_cache),
        pytest.raises(ValueError, match="Specify an exact revision"),
    ):
        await ModelCard.load(MODEL_ID)


async def test_main_path_is_unchanged_and_pin_gets_receipted_directory(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with _patch_model_dirs(models_dir):
        main_dir = await resolve_model_dir(MODEL_ID)
        pinned_dir = await resolve_model_dir(MODEL_ID, REVISION)

    assert main_dir == models_dir / MODEL_ID.normalize()
    assert pinned_dir == models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
    assert not (main_dir / ".exo-huggingface-revision.json").exists()
    assert json.loads((pinned_dir / ".exo-huggingface-revision.json").read_text()) == {
        "repo_id": str(MODEL_ID),
        "revision": REVISION,
    }


def test_pinned_lookup_never_reuses_legacy_main_directory(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _complete_model(models_dir / MODEL_ID.normalize())

    with _patch_model_dirs(models_dir):
        assert build_model_path(MODEL_ID, REVISION) == (
            models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
        )


async def test_nonempty_unverified_pinned_directory_fails_closed(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    pinned_dir = models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
    _complete_model(pinned_dir)

    with _patch_model_dirs(models_dir), pytest.raises(ModelRevisionMismatchError):
        await resolve_model_dir(MODEL_ID, REVISION)


async def test_verified_hf_local_dir_is_adopted_without_redownload(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    pinned_dir = models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
    for relative_path in _complete_model(pinned_dir):
        _write_hf_metadata(pinned_dir, relative_path, REVISION)

    with _patch_model_dirs(models_dir):
        resolved = await resolve_model_dir(MODEL_ID, REVISION)

    assert resolved == pinned_dir
    assert (pinned_dir / ".exo-huggingface-revision.json").is_file()


def test_verified_legacy_hf_local_dir_migrates_by_atomic_rename(
    tmp_path: Path,
) -> None:
    source = tmp_path / MODEL_ID.normalize()
    for relative_path in _complete_model(source):
        _write_hf_metadata(source, relative_path, REVISION)

    destination = migrate_hugging_face_local_dir_to_pinned_revision(
        tmp_path, MODEL_ID, REVISION
    )

    assert not source.exists()
    assert destination == tmp_path / f"{MODEL_ID.normalize()}--{REVISION}"
    assert (destination / ".exo-huggingface-revision.json").is_file()


def test_legacy_hf_migration_rejects_destination_collision_without_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / MODEL_ID.normalize()
    for relative_path in _complete_model(source):
        _write_hf_metadata(source, relative_path, REVISION)
    before = _snapshot_files(source)
    destination = tmp_path / f"{MODEL_ID.normalize()}--{REVISION}"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        migrate_hugging_face_local_dir_to_pinned_revision(tmp_path, MODEL_ID, REVISION)

    assert _snapshot_files(source) == before
    assert destination.is_dir()


@pytest.mark.parametrize("problem", ["missing", "mixed"])
def test_legacy_hf_migration_validation_failure_does_not_mutate_source(
    tmp_path: Path, problem: str
) -> None:
    source = tmp_path / MODEL_ID.normalize()
    files = _complete_model(source)
    for index, relative_path in enumerate(files):
        if problem == "missing" and index == 0:
            continue
        metadata_revision = (
            OTHER_REVISION if problem == "mixed" and index == 0 else REVISION
        )
        _write_hf_metadata(source, relative_path, metadata_revision)
    before = _snapshot_files(source)

    with pytest.raises(ModelRevisionMismatchError):
        migrate_hugging_face_local_dir_to_pinned_revision(tmp_path, MODEL_ID, REVISION)

    assert _snapshot_files(source) == before
    assert not (tmp_path / f"{MODEL_ID.normalize()}--{REVISION}").exists()


def test_main_lookup_rejects_pinned_receipt_in_legacy_directory(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    legacy_dir = models_dir / MODEL_ID.normalize()
    _complete_model(legacy_dir)
    (legacy_dir / ".exo-huggingface-revision.json").write_text(
        json.dumps({"repo_id": str(MODEL_ID), "revision": REVISION})
    )

    with _patch_model_dirs(models_dir), pytest.raises(ModelRevisionMismatchError):
        build_model_path(MODEL_ID)


@pytest.mark.parametrize("problem", ["missing", "mixed"])
async def test_hf_local_dir_adoption_rejects_missing_or_mixed_metadata(
    tmp_path: Path, problem: str
) -> None:
    models_dir = tmp_path / "models"
    pinned_dir = models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
    files = _complete_model(pinned_dir)
    for index, relative_path in enumerate(files):
        if problem == "missing" and index == 0:
            continue
        metadata_revision = (
            OTHER_REVISION if problem == "mixed" and index == 0 else REVISION
        )
        _write_hf_metadata(pinned_dir, relative_path, metadata_revision)

    with _patch_model_dirs(models_dir), pytest.raises(ModelRevisionMismatchError):
        await resolve_model_dir(MODEL_ID, REVISION)
    assert not (pinned_dir / ".exo-huggingface-revision.json").exists()


async def test_download_progress_and_file_listing_use_card_revision(
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    card = _card(REVISION)
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    on_progress = AsyncMock()

    with (
        _patch_model_dirs(models_dir),
        patch(
            "exo.download.download_utils.fetch_file_list_with_cache",
            new_callable=AsyncMock,
            return_value=[],
        ) as fetch_file_list,
    ):
        target_dir, progress = await download_shard(
            shard,
            on_progress,
            skip_download=True,
            allow_patterns=["*"],
        )

    assert target_dir == models_dir / f"{MODEL_ID.normalize()}--{REVISION}"
    assert progress.repo_revision == REVISION
    assert fetch_file_list.await_args is not None
    assert fetch_file_list.await_args.args[:2] == (MODEL_ID, REVISION)
