#!/usr/bin/env python3
"""Create metadata-only, rank-specific GLM-5.2 PP checkpoint views.

The source BF16 and KTransformers checkpoints are never copied or rewritten.
Each rank view contains:

* a filtered Hugging Face safetensors index and links to the BF16 shards that
  contain that rank's decoder layers;
* embeddings only on the first rank, and final norm/LM head only on the last;
* a filtered KTransformers index and links to the already-quantized AMX expert
  shards for that rank's MoE layers; and
* the configuration and tokenizer assets needed to launch SGLang from the
  rank-specific model path.

Only JSON indexes and filesystem metadata are read. Safetensors payloads and
headers are not opened. The destination is published atomically and is never
replaced in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

sys.dont_write_bytecode = True

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type LinkMode = Literal["symlink", "hardlink"]

INDEX_FILENAME: Final = "model.safetensors.index.json"
SUITE_MANIFEST_FILENAME: Final = "checkpoint-view-suite.json"
STAGE_MANIFEST_FILENAME: Final = "stage-view-manifest.json"
SCHEMA_VERSION: Final = 1

_MODEL_LAYER_PATTERN: Final = re.compile(r"^model\.layers\.(\d+)\.")
_KT_EXPERT_PATTERN: Final = re.compile(
    r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\."
    r"(\d+)\.numa\.(\d+)\.(weight|scale)$"
)
_MODEL_GLOBAL_WEIGHTS: Final = frozenset(
    {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
)
_REQUIRED_MODEL_ASSETS: Final = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_OPTIONAL_MODEL_ASSETS: Final = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
)


class CheckpointViewError(RuntimeError):
    """Raised when a safe and complete rank view cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SafetensorsIndex:
    path: Path
    metadata: JsonObject
    weight_map: dict[str, str]
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceFileEvidence:
    name: str
    source_path: Path
    size_bytes: int
    modified_time_nanoseconds: int
    device: int
    inode: int
    metadata_identity_sha256: str


@dataclass(frozen=True, slots=True)
class StageViewPlan:
    pipeline_rank: int
    start_layer: int
    end_layer: int
    model_weight_map: dict[str, str]
    ktransformers_weight_map: dict[str, str]
    model_shards: tuple[SourceFileEvidence, ...]
    ktransformers_shards: tuple[SourceFileEvidence, ...]
    model_weight_map_sha256: str
    ktransformers_weight_map_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointViewSuitePlan:
    model_source: Path
    ktransformers_source: Path
    destination_root: Path
    link_mode: LinkMode
    layer_partition: tuple[int, ...]
    total_layers: int
    first_sparse_layer: int
    moe_layer_frequency: int
    routed_expert_count: int
    ktransformers_numa_nodes: tuple[int, ...]
    model_index: SafetensorsIndex
    ktransformers_index: SafetensorsIndex
    model_assets: tuple[SourceFileEvidence, ...]
    stages: tuple[StageViewPlan, ...]
    source_model_shard_bytes: int
    source_ktransformers_shard_bytes: int
    plan_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_shard_filename(raw_filename: str, *, index_path: Path) -> str:
    filename = Path(raw_filename)
    if (
        not raw_filename
        or filename.name != raw_filename
        or raw_filename in {".", ".."}
        or not raw_filename.endswith(".safetensors")
    ):
        raise CheckpointViewError(
            f"unsafe safetensors shard name {raw_filename!r} in {index_path}"
        )
    return raw_filename


def _load_index(source: Path) -> SafetensorsIndex:
    path = source / INDEX_FILENAME
    if not path.is_file():
        raise CheckpointViewError(f"missing safetensors index: {path}")
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointViewError(f"cannot read safetensors index {path}") from error
    if not isinstance(raw, dict):
        raise CheckpointViewError(f"safetensors index is not an object: {path}")
    raw_metadata = raw.get("metadata", {})
    raw_weight_map = raw.get("weight_map")
    if not isinstance(raw_metadata, dict) or not isinstance(raw_weight_map, dict):
        raise CheckpointViewError(
            f"safetensors index has invalid metadata or weight_map: {path}"
        )

    metadata: JsonObject = {}
    for raw_key, raw_value in raw_metadata.items():
        if not isinstance(raw_key, str):
            raise CheckpointViewError(f"index metadata key is not a string: {path}")
        metadata[raw_key] = cast(JsonValue, raw_value)

    weight_map: dict[str, str] = {}
    for raw_key, raw_filename in raw_weight_map.items():
        if not isinstance(raw_key, str) or not isinstance(raw_filename, str):
            raise CheckpointViewError(
                f"safetensors weight_map is not string-to-string: {path}"
            )
        weight_map[raw_key] = _safe_shard_filename(
            raw_filename,
            index_path=path,
        )
    if not weight_map:
        raise CheckpointViewError(f"safetensors weight_map is empty: {path}")
    return SafetensorsIndex(
        path=path,
        metadata=metadata,
        weight_map=weight_map,
        sha256=_sha256_file(path),
    )


def _load_model_configuration(model_source: Path) -> JsonObject:
    path = model_source / "config.json"
    if not path.is_file():
        raise CheckpointViewError(f"missing model configuration: {path}")
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointViewError(f"cannot read model configuration {path}") from error
    if not isinstance(raw, dict):
        raise CheckpointViewError(f"model configuration is not an object: {path}")
    return cast(JsonObject, raw)


def _required_positive_integer(
    configuration: Mapping[str, JsonValue],
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    raw_value = configuration.get(field)
    minimum = 0 if allow_zero else 1
    if (
        not isinstance(raw_value, int)
        or isinstance(raw_value, bool)
        or raw_value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CheckpointViewError(
            f"model configuration field {field} must be a {qualifier} integer"
        )
    return raw_value


def _source_file_evidence(source_path: Path, *, name: str) -> SourceFileEvidence:
    try:
        status = source_path.stat()
    except OSError as error:
        raise CheckpointViewError(
            f"cannot stat required source file {source_path}"
        ) from error
    if not source_path.is_file():
        raise CheckpointViewError(f"required source is not a file: {source_path}")
    identity: JsonObject = {
        "device": status.st_dev,
        "inode": status.st_ino,
        "modified_time_nanoseconds": status.st_mtime_ns,
        "name": name,
        "size_bytes": status.st_size,
    }
    return SourceFileEvidence(
        name=name,
        source_path=source_path,
        size_bytes=status.st_size,
        modified_time_nanoseconds=status.st_mtime_ns,
        device=status.st_dev,
        inode=status.st_ino,
        metadata_identity_sha256=_canonical_sha256(identity),
    )


def _collect_model_assets(model_source: Path) -> tuple[SourceFileEvidence, ...]:
    assets: list[SourceFileEvidence] = []
    for filename in _REQUIRED_MODEL_ASSETS:
        path = model_source / filename
        if not path.is_file():
            raise CheckpointViewError(f"missing required model asset: {path}")
        assets.append(_source_file_evidence(path, name=filename))
    for filename in _OPTIONAL_MODEL_ASSETS:
        path = model_source / filename
        if path.is_file():
            assets.append(_source_file_evidence(path, name=filename))
    return tuple(sorted(assets, key=lambda item: item.name))


def _layer_ranges(partition: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for layer_count in partition:
        end = start + layer_count
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _select_model_weight_map(
    index: SafetensorsIndex,
    *,
    start_layer: int,
    end_layer: int,
    first_rank: bool,
    last_rank: bool,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    observed_globals: set[str] = set()
    for weight_name, filename in index.weight_map.items():
        layer_match = _MODEL_LAYER_PATTERN.match(weight_name)
        if layer_match is not None:
            layer = int(layer_match.group(1))
            if start_layer <= layer < end_layer:
                selected[weight_name] = filename
            continue
        observed_globals.add(weight_name)
        if first_rank and weight_name == "model.embed_tokens.weight":
            selected[weight_name] = filename
        if last_rank and weight_name in {"model.norm.weight", "lm_head.weight"}:
            selected[weight_name] = filename

    unexpected_globals = observed_globals - _MODEL_GLOBAL_WEIGHTS
    if unexpected_globals:
        preview = ", ".join(sorted(unexpected_globals)[:5])
        raise CheckpointViewError(
            "base checkpoint has unsupported non-layer weights; refusing to "
            f"guess their PP ownership: {preview}"
        )
    expected_globals = set()
    if first_rank:
        expected_globals.add("model.embed_tokens.weight")
    if last_rank:
        expected_globals.update({"model.norm.weight", "lm_head.weight"})
    missing_globals = expected_globals - selected.keys()
    if missing_globals:
        raise CheckpointViewError(
            "base checkpoint is missing PP-global weights: "
            + ", ".join(sorted(missing_globals))
        )
    observed_layers = {
        int(match.group(1))
        for name in selected
        if (match := _MODEL_LAYER_PATTERN.match(name)) is not None
    }
    expected_layers = set(range(start_layer, end_layer))
    if observed_layers != expected_layers:
        missing = sorted(expected_layers - observed_layers)
        raise CheckpointViewError(
            f"base checkpoint has no indexed weights for local layers {missing}"
        )
    return dict(sorted(selected.items()))


def _derive_ktransformers_numa_nodes(
    index: SafetensorsIndex,
    *,
    first_sparse_layer: int,
) -> tuple[int, ...]:
    observed: set[int] = set()
    for weight_name in index.weight_map:
        match = _KT_EXPERT_PATTERN.fullmatch(weight_name)
        if match is not None and int(match.group(1)) == first_sparse_layer:
            observed.add(int(match.group(4)))
    nodes = tuple(sorted(observed))
    if not nodes or nodes != tuple(range(len(nodes))):
        raise CheckpointViewError(
            "KTransformers checkpoint NUMA shard IDs must be contiguous from zero"
        )
    return nodes


def _is_sparse_layer(
    layer: int,
    *,
    first_sparse_layer: int,
    moe_layer_frequency: int,
) -> bool:
    return layer >= first_sparse_layer and layer % moe_layer_frequency == 0


def _select_ktransformers_weight_map(
    index: SafetensorsIndex,
    *,
    start_layer: int,
    end_layer: int,
    first_sparse_layer: int,
    moe_layer_frequency: int,
    routed_expert_count: int,
    numa_nodes: Sequence[int],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for weight_name, filename in index.weight_map.items():
        match = _KT_EXPERT_PATTERN.fullmatch(weight_name)
        if match is None:
            continue
        layer = int(match.group(1))
        if start_layer <= layer < end_layer and _is_sparse_layer(
            layer,
            first_sparse_layer=first_sparse_layer,
            moe_layer_frequency=moe_layer_frequency,
        ):
            selected[weight_name] = filename

    for layer in range(start_layer, end_layer):
        if not _is_sparse_layer(
            layer,
            first_sparse_layer=first_sparse_layer,
            moe_layer_frequency=moe_layer_frequency,
        ):
            continue
        expected = {
            (
                f"blk.{layer}.ffn_{projection}_exps.{expert}.numa."
                f"{numa_node}.{value_kind}"
            )
            for projection in ("up", "gate", "down")
            for expert in range(routed_expert_count)
            for numa_node in numa_nodes
            for value_kind in ("weight", "scale")
        }
        missing = expected - selected.keys()
        if missing:
            preview = ", ".join(sorted(missing)[:3])
            raise CheckpointViewError(
                f"KTransformers layer {layer} is incomplete; missing {len(missing)} "
                f"indexed expert tensors, including {preview}"
            )
    return dict(sorted(selected.items()))


def _collect_shards(
    source: Path,
    weight_map: Mapping[str, str],
) -> tuple[SourceFileEvidence, ...]:
    return tuple(
        _source_file_evidence(source / filename, name=filename)
        for filename in sorted(set(weight_map.values()))
    )


def _source_shard_bytes(source: Path, index: SafetensorsIndex) -> int:
    return sum(
        _source_file_evidence(source / filename, name=filename).size_bytes
        for filename in sorted(set(index.weight_map.values()))
    )


def _plan_identity(
    *,
    model_source: Path,
    ktransformers_source: Path,
    link_mode: LinkMode,
    partition: Sequence[int],
    model_index: SafetensorsIndex,
    ktransformers_index: SafetensorsIndex,
    stages: Sequence[StageViewPlan],
) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_pp_checkpoint_view_plan",
        "model_source": str(model_source),
        "ktransformers_source": str(ktransformers_source),
        "link_mode": link_mode,
        "layer_partition": list(partition),
        "model_index_sha256": model_index.sha256,
        "ktransformers_index_sha256": ktransformers_index.sha256,
        "stages": [
            {
                "pipeline_rank": stage.pipeline_rank,
                "layer_range": [stage.start_layer, stage.end_layer],
                "model_weight_map_sha256": stage.model_weight_map_sha256,
                "ktransformers_weight_map_sha256": (
                    stage.ktransformers_weight_map_sha256
                ),
                "model_shards": [
                    shard.metadata_identity_sha256 for shard in stage.model_shards
                ],
                "ktransformers_shards": [
                    shard.metadata_identity_sha256
                    for shard in stage.ktransformers_shards
                ],
            }
            for stage in stages
        ],
    }


def build_checkpoint_view_plan(
    *,
    model_source: Path,
    ktransformers_source: Path,
    destination_root: Path,
    layer_partition: Sequence[int],
    link_mode: LinkMode,
) -> CheckpointViewSuitePlan:
    """Validate source metadata and construct an immutable PP view plan."""

    resolved_model_source = model_source.resolve(strict=True)
    resolved_ktransformers_source = ktransformers_source.resolve(strict=True)
    resolved_destination = destination_root.resolve(strict=False)
    if not resolved_model_source.is_dir():
        raise CheckpointViewError(
            f"model source is not a directory: {resolved_model_source}"
        )
    if not resolved_ktransformers_source.is_dir():
        raise CheckpointViewError(
            f"KTransformers source is not a directory: {resolved_ktransformers_source}"
        )
    if resolved_destination == Path("/"):
        raise CheckpointViewError("destination root cannot be the filesystem root")
    if resolved_destination.exists():
        raise CheckpointViewError(
            f"destination already exists and will not be replaced: {resolved_destination}"
        )
    if not resolved_destination.parent.is_dir():
        raise CheckpointViewError(
            f"destination parent does not exist: {resolved_destination.parent}"
        )
    if link_mode not in {"symlink", "hardlink"}:
        raise CheckpointViewError(f"unsupported link mode: {link_mode}")

    configuration = _load_model_configuration(resolved_model_source)
    total_layers = _required_positive_integer(configuration, "num_hidden_layers")
    first_sparse_layer = _required_positive_integer(
        configuration,
        "first_k_dense_replace",
        allow_zero=True,
    )
    moe_layer_frequency = _required_positive_integer(
        configuration,
        "moe_layer_freq",
    )
    routed_expert_count = _required_positive_integer(
        configuration,
        "n_routed_experts",
    )
    partition = tuple(layer_partition)
    if (
        not partition
        or any(
            not isinstance(layer_count, int)
            or isinstance(layer_count, bool)
            or layer_count <= 0
            for layer_count in partition
        )
        or sum(partition) != total_layers
    ):
        raise CheckpointViewError(
            "layer partition must contain positive counts that sum to "
            f"num_hidden_layers={total_layers}"
        )

    model_index = _load_index(resolved_model_source)
    ktransformers_index = _load_index(resolved_ktransformers_source)
    numa_nodes = _derive_ktransformers_numa_nodes(
        ktransformers_index,
        first_sparse_layer=first_sparse_layer,
    )
    model_assets = _collect_model_assets(resolved_model_source)
    ranges = _layer_ranges(partition)
    stages: list[StageViewPlan] = []
    for rank, (start_layer, end_layer) in enumerate(ranges):
        model_weight_map = _select_model_weight_map(
            model_index,
            start_layer=start_layer,
            end_layer=end_layer,
            first_rank=rank == 0,
            last_rank=rank == len(partition) - 1,
        )
        ktransformers_weight_map = _select_ktransformers_weight_map(
            ktransformers_index,
            start_layer=start_layer,
            end_layer=end_layer,
            first_sparse_layer=first_sparse_layer,
            moe_layer_frequency=moe_layer_frequency,
            routed_expert_count=routed_expert_count,
            numa_nodes=numa_nodes,
        )
        stages.append(
            StageViewPlan(
                pipeline_rank=rank,
                start_layer=start_layer,
                end_layer=end_layer,
                model_weight_map=model_weight_map,
                ktransformers_weight_map=ktransformers_weight_map,
                model_shards=_collect_shards(
                    resolved_model_source,
                    model_weight_map,
                ),
                ktransformers_shards=_collect_shards(
                    resolved_ktransformers_source,
                    ktransformers_weight_map,
                ),
                model_weight_map_sha256=_canonical_sha256(
                    cast(JsonObject, model_weight_map)
                ),
                ktransformers_weight_map_sha256=_canonical_sha256(
                    cast(JsonObject, ktransformers_weight_map)
                ),
            )
        )
    plan_identity = _plan_identity(
        model_source=resolved_model_source,
        ktransformers_source=resolved_ktransformers_source,
        link_mode=link_mode,
        partition=partition,
        model_index=model_index,
        ktransformers_index=ktransformers_index,
        stages=stages,
    )
    return CheckpointViewSuitePlan(
        model_source=resolved_model_source,
        ktransformers_source=resolved_ktransformers_source,
        destination_root=resolved_destination,
        link_mode=link_mode,
        layer_partition=partition,
        total_layers=total_layers,
        first_sparse_layer=first_sparse_layer,
        moe_layer_frequency=moe_layer_frequency,
        routed_expert_count=routed_expert_count,
        ktransformers_numa_nodes=numa_nodes,
        model_index=model_index,
        ktransformers_index=ktransformers_index,
        model_assets=model_assets,
        stages=tuple(stages),
        source_model_shard_bytes=_source_shard_bytes(
            resolved_model_source,
            model_index,
        ),
        source_ktransformers_shard_bytes=_source_shard_bytes(
            resolved_ktransformers_source,
            ktransformers_index,
        ),
        plan_sha256=_canonical_sha256(plan_identity),
    )


def _file_evidence_receipt(evidence: SourceFileEvidence) -> JsonObject:
    return {
        "name": evidence.name,
        "source_path": str(evidence.source_path),
        "size_bytes": evidence.size_bytes,
        "modified_time_nanoseconds": evidence.modified_time_nanoseconds,
        "device": evidence.device,
        "inode": evidence.inode,
        "metadata_identity_sha256": evidence.metadata_identity_sha256,
    }


def _stage_summary(stage: StageViewPlan) -> JsonObject:
    model_bytes = sum(shard.size_bytes for shard in stage.model_shards)
    ktransformers_bytes = sum(shard.size_bytes for shard in stage.ktransformers_shards)
    return {
        "pipeline_rank": stage.pipeline_rank,
        "layer_range": [stage.start_layer, stage.end_layer],
        "model_tensor_count": len(stage.model_weight_map),
        "model_shard_count": len(stage.model_shards),
        "model_linked_bytes": model_bytes,
        "model_weight_map_sha256": stage.model_weight_map_sha256,
        "ktransformers_tensor_count": len(stage.ktransformers_weight_map),
        "ktransformers_shard_count": len(stage.ktransformers_shards),
        "ktransformers_linked_bytes": ktransformers_bytes,
        "ktransformers_weight_map_sha256": (stage.ktransformers_weight_map_sha256),
        "rank_view_linked_bytes": model_bytes + ktransformers_bytes,
        "model_path": str(stage_path := Path(f"rank-{stage.pipeline_rank}") / "model"),
        "ktransformers_weight_path": str(stage_path.parent / "ktransformers"),
    }


def checkpoint_view_plan_receipt(plan: CheckpointViewSuitePlan) -> JsonObject:
    """Return a compact, serializable receipt suitable for dry-run output."""

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_pp_checkpoint_view_suite",
        "status": "planned",
        "destination_root": str(plan.destination_root),
        "link_mode": plan.link_mode,
        "layer_partition": list(plan.layer_partition),
        "total_layers": plan.total_layers,
        "first_sparse_layer": plan.first_sparse_layer,
        "moe_layer_frequency": plan.moe_layer_frequency,
        "routed_expert_count": plan.routed_expert_count,
        "ktransformers_numa_nodes": list(plan.ktransformers_numa_nodes),
        "source": {
            "model_path": str(plan.model_source),
            "model_index_sha256": plan.model_index.sha256,
            "model_shard_bytes": plan.source_model_shard_bytes,
            "ktransformers_weight_path": str(plan.ktransformers_source),
            "ktransformers_index_sha256": plan.ktransformers_index.sha256,
            "ktransformers_shard_bytes": (plan.source_ktransformers_shard_bytes),
        },
        "stages": [_stage_summary(stage) for stage in plan.stages],
        "plan_sha256": plan.plan_sha256,
        "hash_evidence": {
            "indexes": "sha256-content",
            "generated_weight_maps": "sha256-canonical-json",
            "linked_shards": "sha256-filesystem-metadata-identity-not-content",
        },
    }


def _write_json(path: Path, value: JsonObject) -> str:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    with path.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _create_link(
    destination: Path,
    source: Path,
    *,
    link_mode: LinkMode,
) -> None:
    if link_mode == "symlink":
        destination.symlink_to(source)
    else:
        os.link(source, destination, follow_symlinks=True)


def _filtered_index_receipt(
    *,
    source_index: SafetensorsIndex,
    weight_map: Mapping[str, str],
    shards: Sequence[SourceFileEvidence],
    stage: StageViewPlan,
    checkpoint_kind: Literal["model", "ktransformers"],
) -> JsonObject:
    metadata = dict(source_index.metadata)
    metadata["total_size"] = sum(shard.size_bytes for shard in shards)
    metadata["exo_checkpoint_view"] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "pipeline_rank": stage.pipeline_rank,
        "start_layer_inclusive": stage.start_layer,
        "end_layer_exclusive": stage.end_layer,
        "source_index_sha256": source_index.sha256,
    }
    return {
        "metadata": metadata,
        "weight_map": dict(weight_map),
    }


def _stage_manifest(
    plan: CheckpointViewSuitePlan,
    stage: StageViewPlan,
    *,
    model_index_sha256: str,
    ktransformers_index_sha256: str,
) -> JsonObject:
    summary = _stage_summary(stage)
    summary["model_path"] = str(
        plan.destination_root / f"rank-{stage.pipeline_rank}" / "model"
    )
    summary["ktransformers_weight_path"] = str(
        plan.destination_root / f"rank-{stage.pipeline_rank}" / "ktransformers"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_pp_checkpoint_stage_view",
        "created_at_utc": _utc_now(),
        "plan_sha256": plan.plan_sha256,
        "link_mode": plan.link_mode,
        "summary": summary,
        "source": {
            "model_path": str(plan.model_source),
            "model_index_path": str(plan.model_index.path),
            "model_index_sha256": plan.model_index.sha256,
            "ktransformers_weight_path": str(plan.ktransformers_source),
            "ktransformers_index_path": str(plan.ktransformers_index.path),
            "ktransformers_index_sha256": plan.ktransformers_index.sha256,
        },
        "generated_indexes": {
            "model_index_sha256": model_index_sha256,
            "ktransformers_index_sha256": ktransformers_index_sha256,
        },
        "model_assets": [
            {
                **_file_evidence_receipt(asset),
                "content_sha256": _sha256_file(asset.source_path),
            }
            for asset in plan.model_assets
        ],
        "model_shards": [_file_evidence_receipt(shard) for shard in stage.model_shards],
        "ktransformers_shards": [
            _file_evidence_receipt(shard) for shard in stage.ktransformers_shards
        ],
        "hash_evidence": {
            "indexes": "sha256-content",
            "model_assets": "sha256-content",
            "generated_weight_maps": "sha256-canonical-json",
            "linked_shards": "sha256-filesystem-metadata-identity-not-content",
            "linked_shard_payloads_read": False,
        },
    }


def materialize_checkpoint_views(plan: CheckpointViewSuitePlan) -> Path:
    """Atomically create every rank view and return the suite manifest path."""

    temporary_root = (
        plan.destination_root.parent
        / f".{plan.destination_root.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_root.mkdir(mode=0o700)
    try:
        stage_manifest_hashes: list[JsonValue] = []
        for stage in plan.stages:
            stage_root = temporary_root / f"rank-{stage.pipeline_rank}"
            model_root = stage_root / "model"
            ktransformers_root = stage_root / "ktransformers"
            model_root.mkdir(mode=0o700, parents=True)
            ktransformers_root.mkdir(mode=0o700)

            for asset in plan.model_assets:
                _create_link(
                    model_root / asset.name,
                    asset.source_path,
                    link_mode=plan.link_mode,
                )
            for shard in stage.model_shards:
                _create_link(
                    model_root / shard.name,
                    shard.source_path,
                    link_mode=plan.link_mode,
                )
            for shard in stage.ktransformers_shards:
                _create_link(
                    ktransformers_root / shard.name,
                    shard.source_path,
                    link_mode=plan.link_mode,
                )

            model_index_sha256 = _write_json(
                model_root / INDEX_FILENAME,
                _filtered_index_receipt(
                    source_index=plan.model_index,
                    weight_map=stage.model_weight_map,
                    shards=stage.model_shards,
                    stage=stage,
                    checkpoint_kind="model",
                ),
            )
            ktransformers_index_sha256 = _write_json(
                ktransformers_root / INDEX_FILENAME,
                _filtered_index_receipt(
                    source_index=plan.ktransformers_index,
                    weight_map=stage.ktransformers_weight_map,
                    shards=stage.ktransformers_shards,
                    stage=stage,
                    checkpoint_kind="ktransformers",
                ),
            )
            manifest_path = stage_root / STAGE_MANIFEST_FILENAME
            manifest_sha256 = _write_json(
                manifest_path,
                _stage_manifest(
                    plan,
                    stage,
                    model_index_sha256=model_index_sha256,
                    ktransformers_index_sha256=ktransformers_index_sha256,
                ),
            )
            stage_manifest_hashes.append(
                {
                    "pipeline_rank": stage.pipeline_rank,
                    "relative_path": str(
                        Path(f"rank-{stage.pipeline_rank}") / STAGE_MANIFEST_FILENAME
                    ),
                    "sha256": manifest_sha256,
                }
            )

        suite_manifest = checkpoint_view_plan_receipt(plan)
        suite_manifest["status"] = "materialized"
        suite_manifest["created_at_utc"] = _utc_now()
        suite_manifest["stage_manifests"] = stage_manifest_hashes
        _write_json(temporary_root / SUITE_MANIFEST_FILENAME, suite_manifest)
        os.replace(temporary_root, plan.destination_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return plan.destination_root / SUITE_MANIFEST_FILENAME


def _parse_partition(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "partition must be a comma-separated list of integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("partition counts must be positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--ktransformers-source", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--partition", type=_parse_partition, required=True)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink"),
        default="symlink",
        help="link source files without copying tensor payloads (default: symlink)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the complete plan summary without writing",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        plan = build_checkpoint_view_plan(
            model_source=cast(Path, arguments.model_source),
            ktransformers_source=cast(Path, arguments.ktransformers_source),
            destination_root=cast(Path, arguments.destination_root),
            layer_partition=cast(tuple[int, ...], arguments.partition),
            link_mode=cast(LinkMode, arguments.link_mode),
        )
        receipt = checkpoint_view_plan_receipt(plan)
        if cast(bool, arguments.dry_run):
            print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))
            return 0
        manifest_path = materialize_checkpoint_views(plan)
        print(manifest_path)
        return 0
    except (CheckpointViewError, OSError, ValueError) as error:
        print(f"GLM-5.2 checkpoint view staging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
