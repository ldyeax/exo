from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import scripts.stage_glm52_pp_checkpoint_views as views


@dataclass(frozen=True, slots=True)
class CheckpointFixture:
    model_source: Path
    ktransformers_source: Path
    destination: Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_shard(path: Path, marker: str) -> None:
    path.write_bytes(marker.encode())


def _make_fixture(tmp_path: Path) -> CheckpointFixture:
    model_source = tmp_path / "model"
    ktransformers_source = tmp_path / "ktransformers"
    model_source.mkdir()
    ktransformers_source.mkdir()
    configuration = {
        "num_hidden_layers": 6,
        "first_k_dense_replace": 2,
        "moe_layer_freq": 1,
        "n_routed_experts": 2,
    }
    _write_json(model_source / "config.json", configuration)
    _write_json(model_source / "tokenizer.json", {"model": "fixture"})
    _write_json(model_source / "tokenizer_config.json", {"fixture": True})
    (model_source / "chat_template.jinja").write_text(
        "{{ messages }}",
        encoding="utf-8",
    )

    model_weight_map = {
        "model.embed_tokens.weight": "model-globals.safetensors",
        "model.norm.weight": "model-globals.safetensors",
        "lm_head.weight": "model-globals.safetensors",
    }
    for layer in range(7):
        model_weight_map[f"model.layers.{layer}.input_layernorm.weight"] = (
            f"model-layer-{layer}.safetensors"
        )
        model_weight_map[f"model.layers.{layer}.self_attn.q_proj.weight"] = (
            f"model-layer-{layer}.safetensors"
        )
        _write_shard(
            model_source / f"model-layer-{layer}.safetensors",
            f"model-layer-{layer}",
        )
    _write_shard(model_source / "model-globals.safetensors", "globals")
    _write_json(
        model_source / views.INDEX_FILENAME,
        {"metadata": {"total_size": 1}, "weight_map": model_weight_map},
    )

    ktransformers_weight_map: dict[str, str] = {}
    for layer in range(2, 7):
        filename = f"expert-layer-{layer}.safetensors"
        for projection in ("up", "gate", "down"):
            for expert in range(2):
                for numa_node in range(2):
                    for value_kind in ("weight", "scale"):
                        key = (
                            f"blk.{layer}.ffn_{projection}_exps.{expert}.numa."
                            f"{numa_node}.{value_kind}"
                        )
                        ktransformers_weight_map[key] = filename
        ktransformers_weight_map[f"blk.{layer}.input_layernorm.weight"] = (
            "ktransformers-nonexpert.safetensors"
        )
        _write_shard(ktransformers_source / filename, filename)
    _write_shard(
        ktransformers_source / "ktransformers-nonexpert.safetensors",
        "nonexpert",
    )
    _write_json(
        ktransformers_source / views.INDEX_FILENAME,
        {"metadata": {"total_size": 1}, "weight_map": ktransformers_weight_map},
    )
    return CheckpointFixture(
        model_source=model_source,
        ktransformers_source=ktransformers_source,
        destination=tmp_path / "views",
    )


def _plan(
    fixture: CheckpointFixture,
    *,
    link_mode: views.LinkMode = "symlink",
) -> views.CheckpointViewSuitePlan:
    return views.build_checkpoint_view_plan(
        model_source=fixture.model_source,
        ktransformers_source=fixture.ktransformers_source,
        destination_root=fixture.destination,
        layer_partition=(2, 2, 2),
        link_mode=link_mode,
    )


def test_selects_pp_owned_weights_and_prequantized_expert_shards(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)

    plan = _plan(fixture)

    first, middle, last = plan.stages
    assert set(first.model_weight_map) == {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight",
    }
    assert first.ktransformers_weight_map == {}
    assert set(middle.model_weight_map) == {
        "model.layers.2.input_layernorm.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.3.input_layernorm.weight",
        "model.layers.3.self_attn.q_proj.weight",
    }
    assert {shard.name for shard in middle.ktransformers_shards} == {
        "expert-layer-2.safetensors",
        "expert-layer-3.safetensors",
    }
    assert "model.norm.weight" in last.model_weight_map
    assert "lm_head.weight" in last.model_weight_map
    assert "model.embed_tokens.weight" not in last.model_weight_map
    assert all("blk.6." not in name for name in last.ktransformers_weight_map)
    assert plan.ktransformers_numa_nodes == (0, 1)


def test_materializes_atomic_symlink_views_and_filtered_indexes(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    plan = _plan(fixture)

    suite_manifest = views.materialize_checkpoint_views(plan)

    assert suite_manifest == fixture.destination / views.SUITE_MANIFEST_FILENAME
    assert suite_manifest.is_file()
    rank_zero_model = fixture.destination / "rank-0/model"
    rank_one_kt = fixture.destination / "rank-1/ktransformers"
    assert (rank_zero_model / "tokenizer.json").is_symlink()
    assert (rank_zero_model / "model-globals.safetensors").is_symlink()
    assert (rank_one_kt / "expert-layer-2.safetensors").is_symlink()
    assert not (rank_one_kt / "ktransformers-nonexpert.safetensors").exists()

    rank_zero_index = cast(
        dict[str, object],
        json.loads(
            (rank_zero_model / views.INDEX_FILENAME).read_text(encoding="utf-8")
        ),
    )
    rank_zero_weight_map = cast(dict[str, str], rank_zero_index["weight_map"])
    assert "model.embed_tokens.weight" in rank_zero_weight_map
    assert "model.norm.weight" not in rank_zero_weight_map
    assert all("model.layers.2." not in name for name in rank_zero_weight_map)

    rank_two_index = cast(
        dict[str, object],
        json.loads(
            (fixture.destination / "rank-2/model" / views.INDEX_FILENAME).read_text(
                encoding="utf-8"
            )
        ),
    )
    rank_two_weight_map = cast(dict[str, str], rank_two_index["weight_map"])
    assert "model.norm.weight" in rank_two_weight_map
    assert "lm_head.weight" in rank_two_weight_map
    assert "model.embed_tokens.weight" not in rank_two_weight_map


def test_hardlink_mode_links_payloads_without_copying(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    plan = _plan(fixture, link_mode="hardlink")

    views.materialize_checkpoint_views(plan)

    linked = fixture.destination / "rank-1/ktransformers/expert-layer-2.safetensors"
    source = fixture.ktransformers_source / "expert-layer-2.safetensors"
    assert not linked.is_symlink()
    assert linked.samefile(source)


def test_dry_plan_does_not_create_destination(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    receipt = views.checkpoint_view_plan_receipt(_plan(fixture))

    assert receipt["status"] == "planned"
    assert not fixture.destination.exists()
    assert receipt["plan_sha256"]


def test_rejects_incomplete_prequantized_expert_layer(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    index_path = fixture.ktransformers_source / views.INDEX_FILENAME
    raw_index = cast(
        dict[str, object],
        json.loads(index_path.read_text(encoding="utf-8")),
    )
    weight_map = cast(dict[str, str], raw_index["weight_map"])
    del weight_map["blk.2.ffn_up_exps.0.numa.0.weight"]
    _write_json(index_path, raw_index)

    with pytest.raises(views.CheckpointViewError, match="layer 2 is incomplete"):
        _plan(fixture)


def test_rejects_unknown_global_weight_ownership(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    index_path = fixture.model_source / views.INDEX_FILENAME
    raw_index = cast(
        dict[str, object],
        json.loads(index_path.read_text(encoding="utf-8")),
    )
    weight_map = cast(dict[str, str], raw_index["weight_map"])
    weight_map["model.unassigned.weight"] = "model-globals.safetensors"
    _write_json(index_path, raw_index)

    with pytest.raises(
        views.CheckpointViewError,
        match="unsupported non-layer weights",
    ):
        _plan(fixture)
