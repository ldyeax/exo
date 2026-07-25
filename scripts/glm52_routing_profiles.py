#!/usr/bin/env python3
"""Capture GLM-5.2 routes and materialize fail-closed low-concurrency profiles.

This tool has three public commands:

``capture``
    Start, stop, and dump SGLang's expert-distribution recorder around a fixed
    coding/agent prompt corpus.  The server must already be running with
    ``--expert-distribution-recorder-mode stat`` and
    ``--expert-distribution-recorder-buffer-size -1``.

``materialize``
    Validate one SGLang ``.pt`` recorder dump with ``torch.load(...,
    weights_only=True)`` in the pinned runtime, then create a content-addressed
    frequency-placement input.  The emitted scores preserve observed activation
    frequency and add a deterministic tie break.

``plan``
    Read the real GLM-5.2 safetensor headers and a passed TP2 receipt, calculate
    per-rank expert bytes and conservative resident budgets, and emit immutable
    2K/4K/8K launch profiles.  The canonical 16K-token c2 lane is deliberately
    rejected for residency when it cannot retain the configured VRAM floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast
from urllib.parse import urlparse

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))

import httpx  # noqa: E402

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type PlacementStrategy = Literal["uniform", "frequency"]

SCHEMA_VERSION: Final = 1
MODEL_LAYER_COUNT: Final = 78
FIRST_ROUTED_LAYER: Final = 3
LAST_ROUTED_LAYER: Final = 77
ROUTED_LAYER_COUNT: Final = LAST_ROUTED_LAYER - FIRST_ROUTED_LAYER + 1
EXPERT_COUNT: Final = 256
EXPERTS_PER_TOKEN: Final = 8
TOTAL_ROUTED_EXPERT_POSITIONS: Final = ROUTED_LAYER_COUNT * EXPERT_COUNT
TENSOR_PARALLEL_SIZE: Final = 2
CHUNK_SIZES: Final = (2_048, 4_096, 8_192)
CANONICAL_INPUT_TOKENS: Final = 7_744
RESIDENT_INPUT_TOKENS: Final = 3_968
OUTPUT_TOKENS: Final = 128
SCHEDULER_HEADROOM_TOKENS: Final = 256
ROUTING_CAPTURE_MAX_NEW_TOKENS: Final = 16
DEFAULT_VRAM_FLOOR_MIB: Final = 512
DEFAULT_ALLOCATOR_GUARD_MIB: Final = 256
DEFAULT_RUNTIME_PYTHON: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/"
    "14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455/"
    "venv/bin/python"
)
DEFAULT_MODEL_PATH: Final = "/mnt/sanic/glm52"
DEFAULT_SERVER_URL: Final = "http://192.168.40.24:62710"
DEFAULT_BENCHMARK_SCRIPT: Final = (
    Path(__file__).resolve().parent
    / "run_sglang_kt_glm52_tp2_local_benchmark.py"
)
_MAXIMUM_JSON_BYTES: Final = 32 * 1024 * 1024
_MAXIMUM_PT_BYTES: Final = 512 * 1024 * 1024
_SAFETENSORS_MAXIMUM_HEADER_BYTES: Final = 64 * 1024 * 1024
_SHA256_LENGTH: Final = 64
_RECORDER_GLOB: Final = "expert_distribution_recorder_*.pt"
_PROJECTION_NAMES: Final = ("down_proj", "gate_proj", "up_proj")


class Glm52RoutingProfileError(RuntimeError):
    """Raised when routing evidence or a profile contract is incomplete."""


@dataclass(frozen=True, slots=True)
class RepresentativePrompt:
    prompt_id: str
    category: Literal["coding", "agent"]
    text: str


@dataclass(frozen=True, slots=True)
class RecorderFileIdentity:
    name: str
    device: int
    inode: int
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True, slots=True)
class ExpertTensorInventory:
    model_config_sha256: str
    safetensors_index_sha256: str
    tensor_inventory_sha256: str
    hidden_size: int
    moe_intermediate_size: int
    dtype: str
    bytes_per_element: int
    full_expert_bytes: int
    per_tp_rank_expert_bytes: int
    routed_layer_count: int
    tensor_count: int


@dataclass(frozen=True, slots=True)
class BaselineCapacity:
    receipt_path: str
    receipt_sha256: str
    minimum_observed_free_vram_mib: int
    minimum_required_free_vram_mib: int
    token_capacity: int
    kvcache_gib_per_rank: float
    conservative_kvcache_bytes_per_token_per_rank: int


@dataclass(frozen=True, slots=True)
class ResidentBudget:
    target_token_capacity: int
    estimated_reclaimed_kvcache_bytes_per_rank: int
    estimated_free_vram_mib_before_residents: int
    vram_floor_mib: int
    allocator_guard_mib: int
    allocatable_resident_bytes_per_rank: int
    maximum_safe_total_resident_experts: int
    minimum_per_layer_resident_experts_total: int
    minimum_per_layer_profile_admitted: bool
    rejection_reason: str | None


_PROMPTS: Final[tuple[RepresentativePrompt, ...]] = (
    RepresentativePrompt(
        prompt_id="coding-python-race",
        category="coding",
        text=(
            "You are reviewing an asyncio Python service. A producer appends work "
            "items to a deque, sets an Event, and a consumer clears the Event after "
            "draining the deque. Under load, an item can remain queued forever. "
            "Explain the lost-wakeup interleaving, propose the smallest exact fix, "
            "and give a focused pytest-asyncio regression test. Preserve cancellation "
            "and do not replace the queue with polling."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-rust-protocol",
        category="coding",
        text=(
            "Design a backwards-compatible Rust wire-protocol change that adds a "
            "content hash and monotonic generation to an artifact message. The "
            "decoder must reject ambiguous legacy/new encodings, integer overflow, "
            "duplicate fields, and trailing bytes. Show the typed data model, parsing "
            "invariants, and property tests; avoid unwrap in the network boundary."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-cuda-overlap",
        category="coding",
        text=(
            "A CUDA inference layer stages activations to pinned host memory, starts "
            "CPU expert work, computes a GPU branch, and then merges outputs. Two "
            "requests may overlap. Review the ownership protocol needed for a shared "
            "staging buffer: leases, generations, stream events, exception cleanup, "
            "and stale-handle rejection. Give pseudocode and identify deadlocks."
        ),
    ),
    RepresentativePrompt(
        prompt_id="coding-sql-migration",
        category="coding",
        text=(
            "Plan an online PostgreSQL migration from a nullable text identifier to "
            "a non-null UUID primary key for a high-write table. Include shadow "
            "columns, deterministic backfill, dual writes, validation, index build, "
            "cutover, rollback, and observability. State which operations can lock "
            "and how an agent should prove each phase before advancing."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-incident",
        category="agent",
        text=(
            "Act as an infrastructure agent investigating intermittent distributed "
            "inference stalls. Evidence: GPU utilization alternates between 0 and "
            "95 percent, CPU memory bandwidth stays high, InfiniBand counters are "
            "clean, and only concurrency two stalls. Produce a ranked hypothesis "
            "tree, exact read-only checks, stopping conditions, and a minimal "
            "experiment that distinguishes scheduling serialization from transport."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-repository",
        category="agent",
        text=(
            "You inherit a dirty repository with unrelated user edits and a failing "
            "CI type check. Implement a narrowly scoped feature that touches Python "
            "and Rust without losing work. Describe how you inspect ownership, split "
            "parallel tasks, preserve the worktree, validate only relevant blockers, "
            "and produce a reviewable handoff with exact files and evidence."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-deployment",
        category="agent",
        text=(
            "Create an immutable two-host model deployment plan over five network "
            "paths with heterogeneous rates. Artifacts may land on disk, tmpfs, or "
            "remain streamed when capacity is insufficient. Require content hashes, "
            "resume safety, link identity checks, failover, cache eviction rules, and "
            "a receipt that proves which bytes each host consumed."
        ),
    ),
    RepresentativePrompt(
        prompt_id="agent-benchmark",
        category="agent",
        text=(
            "Design a realistic language-model benchmark for concurrency one and two "
            "that separates prompt processing, decode-window throughput, and "
            "end-to-end throughput. Reuse a semantic warm-up without flushing model "
            "state, pin prompts and sampling, detect serialized admission, and "
            "explain why rolling decode logs can exceed the final aggregate rate."
        ),
    ),
)


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, maximum_bytes: int | None = None) -> str:
    try:
        status = path.lstat()
    except OSError as error:
        raise Glm52RoutingProfileError(f"cannot stat {path}: {error}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise Glm52RoutingProfileError(f"{path} must be a regular non-symlink file")
    if maximum_bytes is not None and status.st_size > maximum_bytes:
        raise Glm52RoutingProfileError(
            f"{path} exceeds the {maximum_bytes}-byte safety bound"
        )
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise Glm52RoutingProfileError(f"cannot open {path}: {error}") from error
    try:
        if os.fstat(descriptor).st_ino != status.st_ino:
            raise Glm52RoutingProfileError(f"{path} changed while opening")
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> JsonObject:
    digest = _sha256_file(path, maximum_bytes=_MAXIMUM_JSON_BYTES)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm52RoutingProfileError(
            f"cannot read {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Glm52RoutingProfileError(f"{description} must be a JSON object")
    result = cast(JsonObject, value)
    if _sha256_bytes(raw) != digest:
        raise Glm52RoutingProfileError(f"{description} changed while reading")
    return result


def _write_immutable_json(path: Path, value: JsonObject) -> None:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
        )
    except FileExistsError as error:
        raise Glm52RoutingProfileError(
            f"refusing to overwrite immutable artifact {path}"
        ) from error
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Glm52RoutingProfileError(f"short write for {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def representative_prompt_manifest() -> JsonObject:
    prompts: list[JsonValue] = []
    for prompt in _PROMPTS:
        encoded = prompt.text.encode()
        prompts.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "utf8_bytes": len(encoded),
                "text_sha256": _sha256_bytes(encoded),
            }
        )
    manifest: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_representative_coding_agent_routing_corpus_v1",
        "prompt_count": len(_PROMPTS),
        "prompts": prompts,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": ROUTING_CAPTURE_MAX_NEW_TOKENS,
            "seed": 52_026,
        },
    }
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    return manifest


def _recorder_snapshot(directory: Path) -> dict[str, RecorderFileIdentity]:
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        raise Glm52RoutingProfileError(
            "recorder directory must be an existing absolute non-symlink directory"
        )
    result: dict[str, RecorderFileIdentity] = {}
    for path in sorted(directory.glob(_RECORDER_GLOB)):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise Glm52RoutingProfileError(
                f"recorder output is not a regular non-symlink file: {path}"
            )
        result[path.name] = RecorderFileIdentity(
            name=path.name,
            device=status.st_dev,
            inode=status.st_ino,
            size_bytes=status.st_size,
            modified_time_ns=status.st_mtime_ns,
        )
    return result


def _validate_server_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise Glm52RoutingProfileError(
            "server URL must be a plain http(s) origin without credentials or a path"
        )
    return raw_url.rstrip("/")


def _post_control(
    client: httpx.Client,
    server_url: str,
    operation: Literal["start", "stop", "dump"],
) -> JsonObject:
    endpoint = {
        "start": "start_expert_distribution_record",
        "stop": "stop_expert_distribution_record",
        "dump": "dump_expert_distribution_record",
    }[operation]
    response = client.post(f"{server_url}/{endpoint}")
    response.raise_for_status()
    if len(response.content) > 1024 * 1024:
        raise Glm52RoutingProfileError(f"{operation} recorder response is too large")
    return {
        "operation": operation,
        "status_code": response.status_code,
        "body_sha256": _sha256_bytes(response.content),
    }


def _validate_capture_server_info(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise Glm52RoutingProfileError("server_info response must be a JSON object")
    info = cast(JsonObject, value)
    required: Mapping[str, JsonValue] = {
        "pp_size": 1,
        "tp_size": 2,
        "kt_method": "AMXINT4",
        "kt_max_deferred_experts_per_token": 0,
        "kt_enable_dynamic_expert_update": False,
        "expert_distribution_recorder_mode": "stat",
        "expert_distribution_recorder_buffer_size": -1,
    }
    for name, expected in required.items():
        observed = info.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise Glm52RoutingProfileError(
                f"capture server field {name} is {observed!r}, expected {expected!r}"
            )
    return info


def _send_representative_prompts(
    client: httpx.Client,
    server_url: str,
) -> list[JsonValue]:
    observations: list[JsonValue] = []
    for prompt in _PROMPTS:
        payload: JsonObject = {
            "text": prompt.text,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": ROUTING_CAPTURE_MAX_NEW_TOKENS,
                "seed": 52_026,
            },
            "stream": False,
        }
        started = time.monotonic()
        response = client.post(f"{server_url}/generate", json=payload)
        response.raise_for_status()
        elapsed = time.monotonic() - started
        if len(response.content) > 16 * 1024 * 1024:
            raise Glm52RoutingProfileError(
                f"response for {prompt.prompt_id} exceeds the safety bound"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise Glm52RoutingProfileError(
                f"response for {prompt.prompt_id} is not JSON"
            ) from error
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise Glm52RoutingProfileError(
                f"response for {prompt.prompt_id} lacks generated text"
            )
        observations.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "request_sha256": _canonical_sha256(payload),
                "response_sha256": _sha256_bytes(response.content),
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
            }
        )
    return observations


def _wait_for_one_new_recorder_file(
    directory: Path,
    before: Mapping[str, RecorderFileIdentity],
    timeout_seconds: float,
) -> Path:
    deadline = time.monotonic() + timeout_seconds
    stable_identity: RecorderFileIdentity | None = None
    stable_observations = 0
    while time.monotonic() < deadline:
        current = _recorder_snapshot(directory)
        new_names = sorted(set(current) - set(before))
        if len(new_names) > 1:
            raise Glm52RoutingProfileError(
                f"recorder dump created multiple candidate files: {new_names}"
            )
        if len(new_names) == 1:
            identity = current[new_names[0]]
            if identity.size_bytes <= 0:
                stable_observations = 0
            elif identity == stable_identity:
                stable_observations += 1
                if stable_observations >= 2:
                    return directory / identity.name
            else:
                stable_identity = identity
                stable_observations = 1
        time.sleep(0.1)
    raise Glm52RoutingProfileError(
        "timed out waiting for exactly one stable expert-distribution dump"
    )


def _validate_count_matrix(
    counts: Sequence[Sequence[object]],
) -> tuple[tuple[int, ...], ...]:
    if len(counts) != MODEL_LAYER_COUNT:
        raise Glm52RoutingProfileError(
            f"routing matrix has {len(counts)} layers, expected {MODEL_LAYER_COUNT}"
        )
    normalized: list[tuple[int, ...]] = []
    routed_totals: list[int] = []
    for layer_index, row in enumerate(counts):
        if len(row) != EXPERT_COUNT:
            raise Glm52RoutingProfileError(
                f"routing layer {layer_index} has {len(row)} experts, "
                f"expected {EXPERT_COUNT}"
            )
        values: list[int] = []
        for expert_index, raw_value in enumerate(row):
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise Glm52RoutingProfileError(
                    f"routing count {layer_index}/{expert_index} is not an integer"
                )
            if raw_value < 0:
                raise Glm52RoutingProfileError(
                    f"routing count {layer_index}/{expert_index} is negative"
                )
            values.append(raw_value)
        total = sum(values)
        if layer_index < FIRST_ROUTED_LAYER:
            if total != 0:
                raise Glm52RoutingProfileError(
                    f"dense layer {layer_index} unexpectedly contains routes"
                )
        else:
            if total <= 0 or total % EXPERTS_PER_TOKEN != 0:
                raise Glm52RoutingProfileError(
                    f"routed layer {layer_index} has invalid total {total}"
                )
            routed_totals.append(total)
        normalized.append(tuple(values))
    if len(set(routed_totals)) != 1:
        raise Glm52RoutingProfileError(
            "routed layers do not contain the same number of token routes"
        )
    return tuple(normalized)


def _deterministic_placement_scores(
    counts: Sequence[Sequence[object]],
) -> tuple[tuple[int, ...], ...]:
    normalized = _validate_count_matrix(counts)
    multiplier = MODEL_LAYER_COUNT * EXPERT_COUNT + 1
    maximum_count = max(max(row) for row in normalized)
    if maximum_count > (2**63 - 1 - multiplier) // multiplier:
        raise Glm52RoutingProfileError("routing counts overflow signed int64 scores")
    scores: list[tuple[int, ...]] = []
    for layer_index, row in enumerate(normalized):
        score_row: list[int] = []
        for expert_index, count in enumerate(row):
            if layer_index < FIRST_ROUTED_LAYER:
                score_row.append(0)
                continue
            flat_index = layer_index * EXPERT_COUNT + expert_index
            deterministic_tie = MODEL_LAYER_COUNT * EXPERT_COUNT - flat_index
            score_row.append(count * multiplier + deterministic_tie)
        scores.append(tuple(score_row))
    return tuple(scores)


def _ranked_routed_experts(
    counts: Sequence[Sequence[object]],
) -> tuple[tuple[int, int, int], ...]:
    normalized = _validate_count_matrix(counts)
    positions = [
        (layer_index, expert_index, normalized[layer_index][expert_index])
        for layer_index in range(FIRST_ROUTED_LAYER, MODEL_LAYER_COUNT)
        for expert_index in range(EXPERT_COUNT)
    ]
    return tuple(sorted(positions, key=lambda item: (-item[2], item[0], item[1])))


def _materialize_with_torch(
    source_path: Path,
    output_directory: Path,
    prompt_manifest_sha256: str,
) -> JsonObject:
    try:
        import torch
    except ImportError as error:
        raise Glm52RoutingProfileError(
            "materialization must run with the pinned SGLang runtime Python"
        ) from error

    source_sha256 = _sha256_file(source_path, maximum_bytes=_MAXIMUM_PT_BYTES)
    try:
        loaded = torch.load(source_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Glm52RoutingProfileError(
            f"cannot safely load routing tensor {source_path}: {error}"
        ) from error
    if not isinstance(loaded, dict) or set(loaded) < {"logical_count"}:
        raise Glm52RoutingProfileError(
            "routing artifact must be a dict containing logical_count"
        )
    logical_count = loaded["logical_count"]
    if not isinstance(logical_count, torch.Tensor):
        raise Glm52RoutingProfileError("logical_count is not a tensor")
    if logical_count.device.type != "cpu":
        raise Glm52RoutingProfileError("logical_count was not loaded onto CPU")
    if logical_count.dtype not in {torch.int32, torch.int64}:
        raise Glm52RoutingProfileError(
            f"logical_count dtype {logical_count.dtype} is not int32/int64"
        )
    if (
        logical_count.dim() != 3
        or logical_count.shape[0] <= 0
        or logical_count.shape[0] > 65_536
        or tuple(logical_count.shape[1:]) != (MODEL_LAYER_COUNT, EXPERT_COUNT)
    ):
        raise Glm52RoutingProfileError(
            "logical_count must have shape [1..65536, 78, 256]"
        )
    if bool(torch.any(logical_count < 0).item()):
        raise Glm52RoutingProfileError("logical_count contains negative values")
    summed = logical_count.to(dtype=torch.int64).sum(dim=0)
    counts = _validate_count_matrix(cast(list[list[object]], summed.tolist()))
    scores = _deterministic_placement_scores(counts)
    ranked = _ranked_routed_experts(counts)

    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise Glm52RoutingProfileError("output directory must not be a symlink")
    temporary_path = output_directory / (
        f".glm52-routing-frequency-{source_sha256}.tmp.pt"
    )
    if temporary_path.exists() or temporary_path.is_symlink():
        raise Glm52RoutingProfileError(
            f"refusing to reuse temporary artifact {temporary_path}"
        )
    score_tensor = torch.tensor(scores, dtype=torch.int64).unsqueeze(0)
    torch.save({"logical_count": score_tensor}, temporary_path)
    placement_sha256 = _sha256_file(
        temporary_path,
        maximum_bytes=_MAXIMUM_PT_BYTES,
    )
    placement_path = (
        output_directory / f"glm52-routing-frequency-{placement_sha256}.pt"
    )
    if placement_path.exists() or placement_path.is_symlink():
        raise Glm52RoutingProfileError(
            f"refusing to overwrite placement artifact {placement_path}"
        )
    os.replace(temporary_path, placement_path)
    placement_path.chmod(0o400)

    layer_totals = [sum(row) for row in counts]
    receipt: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "kind": "glm52_frequency_placement_input_v1",
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "logical_count_shape": list(logical_count.shape),
            "logical_count_dtype": str(logical_count.dtype),
        },
        "model_contract": {
            "layer_count": MODEL_LAYER_COUNT,
            "expert_count": EXPERT_COUNT,
            "first_routed_layer": FIRST_ROUTED_LAYER,
            "last_routed_layer": LAST_ROUTED_LAYER,
            "routed_layer_count": ROUTED_LAYER_COUNT,
            "experts_per_token": EXPERTS_PER_TOKEN,
        },
        "capture_contract": {
            "prompt_manifest_sha256": prompt_manifest_sha256,
            "routes_per_routed_layer": layer_totals[FIRST_ROUTED_LAYER],
            "routed_tokens": (
                layer_totals[FIRST_ROUTED_LAYER] // EXPERTS_PER_TOKEN
            ),
            "activation_counts_sha256": _canonical_sha256(
                cast(JsonValue, [list(row) for row in counts])
            ),
        },
        "ranking": {
            "policy": (
                "descending activation count, then ascending layer, then "
                "ascending expert; encoded into distinct signed-int64 scores"
            ),
            "score_multiplier": MODEL_LAYER_COUNT * EXPERT_COUNT + 1,
            "ranked_positions_sha256": _canonical_sha256(
                cast(JsonValue, [list(item) for item in ranked])
            ),
            "top_64": [
                {
                    "layer": layer,
                    "expert": expert,
                    "activation_count": count,
                }
                for layer, expert, count in ranked[:64]
            ],
        },
        "artifact": {
            "path": str(placement_path),
            "sha256": placement_sha256,
            "logical_count_shape": [1, MODEL_LAYER_COUNT, EXPERT_COUNT],
            "logical_count_dtype": "torch.int64",
            "immutable_mode": "0400",
        },
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "torch_version": torch.__version__,
        },
    }
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    receipt_path = output_directory / (
        f"glm52-routing-frequency-{placement_sha256}.json"
    )
    _write_immutable_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def materialize_routing_artifact(
    *,
    runtime_python: Path,
    source_path: Path,
    output_directory: Path,
    prompt_manifest_sha256: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> JsonObject:
    if not runtime_python.is_absolute() or not source_path.is_absolute():
        raise Glm52RoutingProfileError(
            "runtime Python and routing source paths must be absolute"
        )
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise Glm52RoutingProfileError(
            f"runtime Python is missing or not executable: {runtime_python}"
        )
    if (
        len(prompt_manifest_sha256) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in prompt_manifest_sha256)
    ):
        raise Glm52RoutingProfileError("prompt manifest hash is not lowercase SHA-256")
    command = (
        str(runtime_python),
        str(Path(__file__).resolve()),
        "_torch-materialize",
        "--source",
        str(source_path),
        "--output-directory",
        str(output_directory),
        "--prompt-manifest-sha256",
        prompt_manifest_sha256,
    )
    completed = command_runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300.0,
    )
    if completed.returncode != 0:
        raise Glm52RoutingProfileError(
            "routing materializer failed in pinned runtime: "
            f"{completed.stderr[-4_096:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise Glm52RoutingProfileError(
            "routing materializer did not return JSON"
        ) from error
    if not isinstance(value, dict) or value.get("status") != "materialized":
        raise Glm52RoutingProfileError(
            "routing materializer did not return a completed receipt"
        )
    receipt = cast(JsonObject, value)
    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict):
        raise Glm52RoutingProfileError("materializer receipt lacks artifact identity")
    artifact_path = Path(cast(str, artifact.get("path")))
    artifact_sha256 = cast(str, artifact.get("sha256"))
    if _sha256_file(artifact_path, maximum_bytes=_MAXIMUM_PT_BYTES) != artifact_sha256:
        raise Glm52RoutingProfileError(
            "materialized placement hash differs after subprocess exit"
        )
    return receipt


def capture_routing(
    *,
    server_url: str,
    recorder_directory: Path,
    output_directory: Path,
    runtime_python: Path,
    timeout_seconds: float,
) -> JsonObject:
    server_url = _validate_server_url(server_url)
    prompt_manifest = representative_prompt_manifest()
    prompt_manifest_sha256 = cast(str, prompt_manifest["manifest_content_sha256"])
    control_operations: list[JsonValue] = []
    observations: list[JsonValue] = []
    source_path: Path | None = None
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(f"{server_url}/get_server_info")
        response.raise_for_status()
        server_info = _validate_capture_server_info(response.json())

        # Establish a known stopped-and-empty recorder generation.  The first
        # dump is deliberately excluded from the post-clear directory snapshot.
        control_operations.append(_post_control(client, server_url, "stop"))
        control_operations.append(_post_control(client, server_url, "dump"))
        time.sleep(0.2)
        before = _recorder_snapshot(recorder_directory)

        started = False
        try:
            control_operations.append(_post_control(client, server_url, "start"))
            started = True
            observations = _send_representative_prompts(client, server_url)
            control_operations.append(_post_control(client, server_url, "stop"))
            started = False
            control_operations.append(_post_control(client, server_url, "dump"))
        finally:
            if started:
                try:
                    control_operations.append(
                        _post_control(client, server_url, "stop")
                    )
                    control_operations.append(
                        _post_control(client, server_url, "dump")
                    )
                except httpx.HTTPError:
                    pass
        source_path = _wait_for_one_new_recorder_file(
            recorder_directory,
            before,
            timeout_seconds,
        )

    materialized = materialize_routing_artifact(
        runtime_python=runtime_python,
        source_path=source_path,
        output_directory=output_directory,
        prompt_manifest_sha256=prompt_manifest_sha256,
    )
    server_identity: JsonObject = {
        name: server_info.get(name)
        for name in (
            "version",
            "model_path",
            "kt_weight_path",
            "pp_size",
            "tp_size",
            "kt_method",
            "kt_max_deferred_experts_per_token",
            "kt_enable_dynamic_expert_update",
            "expert_distribution_recorder_mode",
            "expert_distribution_recorder_buffer_size",
        )
    }
    receipt: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "captured",
        "kind": "glm52_representative_routing_capture_v1",
        "server_url": server_url,
        "server_identity": server_identity,
        "prompt_manifest": prompt_manifest,
        "request_observations": observations,
        "recorder_control_operations": control_operations,
        "source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path, maximum_bytes=_MAXIMUM_PT_BYTES),
        },
        "materialized_frequency_input": {
            "receipt_path": materialized["receipt_path"],
            "receipt_content_sha256": materialized["receipt_content_sha256"],
            "artifact": materialized["artifact"],
        },
        "exactness": {
            "max_deferred_experts_per_token": 0,
            "dynamic_expert_update": False,
            "approximate_model_lane_excluded": True,
        },
    }
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    receipt_path = output_directory / (
        "glm52-routing-capture-"
        f"{cast(str, receipt['receipt_content_sha256'])}.json"
    )
    _write_immutable_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _read_safetensors_header(path: Path) -> JsonObject:
    try:
        status = path.lstat()
    except OSError as error:
        raise Glm52RoutingProfileError(
            f"cannot stat safetensors shard {path}: {error}"
        ) from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise Glm52RoutingProfileError(
            f"safetensors shard {path} must be a regular non-symlink file"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise Glm52RoutingProfileError(
            f"cannot open safetensors shard {path}: {error}"
        ) from error
    try:
        if os.fstat(descriptor).st_ino != status.st_ino:
            raise Glm52RoutingProfileError(
                f"safetensors shard {path} changed while opening"
            )
        raw_length = os.read(descriptor, 8)
        if len(raw_length) != 8:
            raise Glm52RoutingProfileError(
                f"safetensors shard {path} has a truncated header length"
            )
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 2 <= header_length <= _SAFETENSORS_MAXIMUM_HEADER_BYTES:
            raise Glm52RoutingProfileError(
                f"safetensors shard {path} has unsafe header length {header_length}"
            )
        blocks: list[bytes] = []
        remaining = header_length
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise Glm52RoutingProfileError(
                    f"safetensors shard {path} has a truncated header"
                )
            blocks.append(block)
            remaining -= len(block)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(blocks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Glm52RoutingProfileError(
            f"safetensors shard {path} header is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise Glm52RoutingProfileError(
            f"safetensors shard {path} header is not an object"
        )
    return cast(JsonObject, value)


def inspect_expert_tensor_inventory(model_path: Path) -> ExpertTensorInventory:
    if not model_path.is_absolute() or not model_path.is_dir() or model_path.is_symlink():
        raise Glm52RoutingProfileError(
            "model path must be an existing absolute non-symlink directory"
        )
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config = _read_json(config_path, "model config")
    index = _read_json(index_path, "safetensors index")
    required_config = {
        "num_hidden_layers": MODEL_LAYER_COUNT,
        "first_k_dense_replace": FIRST_ROUTED_LAYER,
        "n_routed_experts": EXPERT_COUNT,
        "num_experts_per_tok": EXPERTS_PER_TOKEN,
    }
    for name, expected in required_config.items():
        value = config.get(name)
        if type(value) is not int or value != expected:
            raise Glm52RoutingProfileError(
                f"model config {name} is {value!r}, expected {expected}"
            )
    hidden_size = config.get("hidden_size")
    intermediate_size = config.get("moe_intermediate_size")
    if (
        type(hidden_size) is not int
        or cast(int, hidden_size) <= 0
        or type(intermediate_size) is not int
        or cast(int, intermediate_size) <= 0
    ):
        raise Glm52RoutingProfileError(
            "model config lacks positive hidden/moe intermediate sizes"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise Glm52RoutingProfileError("safetensors index lacks a weight_map object")
    typed_weight_map = cast(dict[str, JsonValue], weight_map)

    needed: dict[str, str] = {}
    for layer in range(FIRST_ROUTED_LAYER, MODEL_LAYER_COUNT):
        for expert in range(EXPERT_COUNT):
            for projection in _PROJECTION_NAMES:
                name = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}.weight"
                )
                shard_name = typed_weight_map.get(name)
                if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                    raise Glm52RoutingProfileError(
                        f"safetensors index lacks a safe shard for {name}"
                    )
                needed[name] = shard_name

    headers: dict[str, JsonObject] = {}
    inventory_rows: list[JsonValue] = []
    per_expert_sizes: dict[tuple[int, int], int] = {}
    expected_shapes = {
        "down_proj": [cast(int, hidden_size), cast(int, intermediate_size)],
        "gate_proj": [cast(int, intermediate_size), cast(int, hidden_size)],
        "up_proj": [cast(int, intermediate_size), cast(int, hidden_size)],
    }
    for tensor_name, shard_name in sorted(needed.items()):
        if shard_name not in headers:
            headers[shard_name] = _read_safetensors_header(model_path / shard_name)
        metadata = headers[shard_name].get(tensor_name)
        if not isinstance(metadata, dict):
            raise Glm52RoutingProfileError(
                f"safetensors shard {shard_name} lacks {tensor_name}"
            )
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        projection = tensor_name.rsplit(".", 2)[1]
        if dtype != "BF16" or shape != expected_shapes[projection]:
            raise Glm52RoutingProfileError(
                f"{tensor_name} has dtype/shape {dtype!r}/{shape!r}, expected "
                f"BF16/{expected_shapes[projection]!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or cast(int, offsets[0]) < 0
            or cast(int, offsets[1]) <= cast(int, offsets[0])
        ):
            raise Glm52RoutingProfileError(
                f"{tensor_name} has invalid safetensors offsets"
            )
        byte_count = cast(int, offsets[1]) - cast(int, offsets[0])
        element_count = math.prod(cast(list[int], shape))
        if byte_count != element_count * 2:
            raise Glm52RoutingProfileError(
                f"{tensor_name} byte count does not match BF16 shape"
            )
        parts = tensor_name.split(".")
        key = (int(parts[2]), int(parts[5]))
        per_expert_sizes[key] = per_expert_sizes.get(key, 0) + byte_count
        inventory_rows.append(
            {
                "name": tensor_name,
                "shard": shard_name,
                "dtype": dtype,
                "shape": shape,
                "bytes": byte_count,
            }
        )
    if len(per_expert_sizes) != TOTAL_ROUTED_EXPERT_POSITIONS:
        raise Glm52RoutingProfileError("expert tensor inventory is incomplete")
    unique_sizes = set(per_expert_sizes.values())
    if len(unique_sizes) != 1:
        raise Glm52RoutingProfileError(
            f"expert tensor byte sizes differ: {sorted(unique_sizes)}"
        )
    full_expert_bytes = unique_sizes.pop()
    if full_expert_bytes % TENSOR_PARALLEL_SIZE:
        raise Glm52RoutingProfileError(
            "full expert bytes do not divide evenly across TP ranks"
        )
    return ExpertTensorInventory(
        model_config_sha256=_sha256_file(config_path),
        safetensors_index_sha256=_sha256_file(index_path),
        tensor_inventory_sha256=_canonical_sha256(inventory_rows),
        hidden_size=cast(int, hidden_size),
        moe_intermediate_size=cast(int, intermediate_size),
        dtype="BF16",
        bytes_per_element=2,
        full_expert_bytes=full_expert_bytes,
        per_tp_rank_expert_bytes=full_expert_bytes // TENSOR_PARALLEL_SIZE,
        routed_layer_count=ROUTED_LAYER_COUNT,
        tensor_count=len(inventory_rows),
    )


def read_baseline_capacity(
    receipt_path: Path,
    *,
    expected_vram_floor_mib: int,
) -> BaselineCapacity:
    receipt_sha256 = _sha256_file(
        receipt_path,
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    )
    receipt = _read_json(receipt_path, "TP2 benchmark receipt")
    if receipt.get("status") != "passed":
        raise Glm52RoutingProfileError("TP2 baseline receipt is not passed")
    topology = receipt.get("topology")
    if not isinstance(topology, dict):
        raise Glm52RoutingProfileError("TP2 receipt lacks topology")
    process_spec = receipt.get("process_spec")
    if not isinstance(process_spec, dict):
        raise Glm52RoutingProfileError("TP2 receipt lacks process spec")
    experts = process_spec.get("experts")
    if (
        topology.get("pipeline_parallel_size") != 1
        or topology.get("tensor_parallel_size") != TENSOR_PARALLEL_SIZE
        or topology.get("parent_process_count") != 1
        or not isinstance(experts, dict)
        or experts.get("resident_gpu_experts") != 0
        or experts.get("max_deferred_experts_per_token") != 0
    ):
        raise Glm52RoutingProfileError(
            "baseline is not an exact PP1/TP2/zero-resident/deferred-zero run"
        )
    capacity = receipt.get("capacity_and_vram")
    server_info = receipt.get("server_info")
    if not isinstance(capacity, dict) or not isinstance(server_info, dict):
        raise Glm52RoutingProfileError(
            "baseline receipt lacks capacity/server evidence"
        )
    snapshot = capacity.get("postreadiness_snapshot")
    gate = capacity.get("postreadiness_gate")
    if not isinstance(snapshot, dict) or not isinstance(gate, dict):
        raise Glm52RoutingProfileError(
            "baseline lacks post-readiness VRAM evidence"
        )
    devices = snapshot.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != TENSOR_PARALLEL_SIZE
        or not all(isinstance(device, dict) for device in devices)
    ):
        raise Glm52RoutingProfileError("baseline VRAM device set is invalid")
    free_values = [
        device.get("free_mib")
        for device in cast(list[dict[str, JsonValue]], devices)
    ]
    if not all(type(value) is int and cast(int, value) >= 0 for value in free_values):
        raise Glm52RoutingProfileError("baseline free-VRAM values are invalid")
    observed_floor = gate.get("minimum_free_vram_mib")
    if (
        type(observed_floor) is not int
        or observed_floor != expected_vram_floor_mib
    ):
        raise Glm52RoutingProfileError(
            f"baseline VRAM floor is {observed_floor!r}, expected "
            f"{expected_vram_floor_mib}"
        )
    token_capacity = server_info.get("max_total_num_tokens")
    internal_states = server_info.get("internal_states")
    if (
        type(token_capacity) is not int
        or cast(int, token_capacity) <= 0
        or not isinstance(internal_states, list)
        or not internal_states
        or not isinstance(internal_states[0], dict)
    ):
        raise Glm52RoutingProfileError("baseline token capacity is invalid")
    memory_usage = cast(dict[str, JsonValue], internal_states[0]).get("memory_usage")
    if not isinstance(memory_usage, dict):
        raise Glm52RoutingProfileError("baseline lacks scheduler memory usage")
    kvcache_gib = memory_usage.get("kvcache")
    recorded_capacity = memory_usage.get("token_capacity")
    if (
        type(kvcache_gib) not in {int, float}
        or not math.isfinite(float(cast(int | float, kvcache_gib)))
        or float(cast(int | float, kvcache_gib)) <= 0
        or recorded_capacity != token_capacity
    ):
        raise Glm52RoutingProfileError(
            "baseline KV-cache memory/capacity evidence is invalid"
        )
    # SGLang exposes memory GiB rounded to two decimals.  Subtract half a
    # display unit before deriving reclaimable bytes, so estimates never rely
    # on the optimistic side of that rounding interval.
    conservative_kvcache_gib = max(
        0.0,
        float(cast(int | float, kvcache_gib)) - 0.005,
    )
    bytes_per_token = math.floor(
        conservative_kvcache_gib * 1024**3 / cast(int, token_capacity)
    )
    if bytes_per_token <= 0:
        raise Glm52RoutingProfileError(
            "baseline conservative KV bytes per token is zero"
        )
    return BaselineCapacity(
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
        minimum_observed_free_vram_mib=min(cast(list[int], free_values)),
        minimum_required_free_vram_mib=cast(int, observed_floor),
        token_capacity=cast(int, token_capacity),
        kvcache_gib_per_rank=float(cast(int | float, kvcache_gib)),
        conservative_kvcache_bytes_per_token_per_rank=bytes_per_token,
    )


def calculate_resident_budget(
    baseline: BaselineCapacity,
    inventory: ExpertTensorInventory,
    *,
    target_token_capacity: int,
    allocator_guard_mib: int,
) -> ResidentBudget:
    if target_token_capacity <= 0 or target_token_capacity > baseline.token_capacity:
        raise Glm52RoutingProfileError(
            "target token capacity must be positive and no larger than baseline"
        )
    if allocator_guard_mib < 0:
        raise Glm52RoutingProfileError("allocator guard must not be negative")
    reclaimed = (
        baseline.token_capacity - target_token_capacity
    ) * baseline.conservative_kvcache_bytes_per_token_per_rank
    estimated_free_mib = baseline.minimum_observed_free_vram_mib + math.floor(
        reclaimed / 1024**2
    )
    allocatable = max(
        0,
        (
            estimated_free_mib
            - baseline.minimum_required_free_vram_mib
            - allocator_guard_mib
        )
        * 1024**2,
    )
    maximum = min(
        TOTAL_ROUTED_EXPERT_POSITIONS,
        allocatable // inventory.per_tp_rank_expert_bytes,
    )
    per_layer_minimum_total = ROUTED_LAYER_COUNT
    rejection_reason: str | None = None
    if maximum == 0:
        rejection_reason = (
            "no nonzero resident budget retains both the observed VRAM floor "
            "and allocator/measurement guard"
        )
    elif maximum < per_layer_minimum_total:
        rejection_reason = (
            "the minimum one-expert-per-routed-layer KT count does not fit; "
            "only a sparse global ratio budget is admissible"
        )
    return ResidentBudget(
        target_token_capacity=target_token_capacity,
        estimated_reclaimed_kvcache_bytes_per_rank=reclaimed,
        estimated_free_vram_mib_before_residents=estimated_free_mib,
        vram_floor_mib=baseline.minimum_required_free_vram_mib,
        allocator_guard_mib=allocator_guard_mib,
        allocatable_resident_bytes_per_rank=allocatable,
        maximum_safe_total_resident_experts=maximum,
        minimum_per_layer_resident_experts_total=per_layer_minimum_total,
        minimum_per_layer_profile_admitted=maximum >= per_layer_minimum_total,
        rejection_reason=rejection_reason,
    )


def _ratio_for_exact_total_budget(total_budget: int) -> str:
    if not 0 < total_budget <= TOTAL_ROUTED_EXPERT_POSITIONS:
        raise Glm52RoutingProfileError(
            "exact ratio budget must be within routed expert positions"
        )
    ratio = total_budget / TOTAL_ROUTED_EXPERT_POSITIONS
    for candidate in (
        ratio,
        math.nextafter(ratio, math.inf),
        math.nextafter(math.nextafter(ratio, math.inf), math.inf),
    ):
        if int(candidate * TOTAL_ROUTED_EXPERT_POSITIONS) == total_budget:
            return repr(candidate)
    raise Glm52RoutingProfileError(
        f"cannot encode exact KT ratio for {total_budget} experts"
    )


def _profile_arguments(
    *,
    profile_id: str,
    result_root: Path,
    input_tokens: int,
    chunk_size: int,
    resident_budget: int,
    placement_strategy: PlacementStrategy,
    placement_artifact: Path | None,
    placement_sha256: str | None,
) -> list[JsonValue]:
    maximum_total_tokens = (
        2 * (input_tokens + OUTPUT_TOKENS) + SCHEDULER_HEADROOM_TOKENS
    )
    arguments: list[JsonValue] = [
        "--run-id",
        profile_id,
        "--result-directory",
        str(result_root / profile_id),
        "--benchmark-input-tokens",
        input_tokens,
        "--benchmark-output-tokens",
        OUTPUT_TOKENS,
        "--max-total-tokens",
        maximum_total_tokens,
        "--chunked-prefill-size",
        chunk_size,
        "--resident-gpu-expert-budget-total",
        resident_budget,
        "--kt-expert-placement-strategy",
        placement_strategy,
    ]
    if resident_budget > 0:
        arguments.extend(
            [
                "--kt-gpu-experts-ratio",
                _ratio_for_exact_total_budget(resident_budget),
            ]
        )
    if placement_strategy == "frequency":
        if placement_artifact is None or placement_sha256 is None:
            raise Glm52RoutingProfileError(
                "frequency profile requires a placement artifact and hash"
            )
        arguments.extend(
            [
                "--init-expert-location",
                str(placement_artifact),
                "--init-expert-location-sha256",
                placement_sha256,
            ]
        )
    return arguments


def build_profile_plan(
    *,
    baseline: BaselineCapacity,
    inventory: ExpertTensorInventory,
    placement_artifact: Path,
    placement_sha256: str,
    placement_receipt_path: Path,
    placement_receipt_sha256: str,
    result_root: Path,
    allocator_guard_mib: int,
) -> JsonObject:
    if _sha256_file(placement_artifact, maximum_bytes=_MAXIMUM_PT_BYTES) != (
        placement_sha256
    ):
        raise Glm52RoutingProfileError("frequency placement artifact hash changed")
    if _sha256_file(
        placement_receipt_path,
        maximum_bytes=_MAXIMUM_JSON_BYTES,
    ) != placement_receipt_sha256:
        raise Glm52RoutingProfileError("frequency placement receipt hash changed")
    placement_receipt = _read_json(
        placement_receipt_path,
        "frequency placement receipt",
    )
    placement_identity = placement_receipt.get("artifact")
    model_contract = placement_receipt.get("model_contract")
    if (
        placement_receipt.get("status") != "materialized"
        or placement_receipt.get("kind") != "glm52_frequency_placement_input_v1"
        or not isinstance(placement_identity, dict)
        or placement_identity.get("path") != str(placement_artifact)
        or placement_identity.get("sha256") != placement_sha256
        or placement_identity.get("logical_count_shape")
        != [1, MODEL_LAYER_COUNT, EXPERT_COUNT]
        or not isinstance(model_contract, dict)
        or model_contract.get("layer_count") != MODEL_LAYER_COUNT
        or model_contract.get("expert_count") != EXPERT_COUNT
        or model_contract.get("first_routed_layer") != FIRST_ROUTED_LAYER
        or model_contract.get("last_routed_layer") != LAST_ROUTED_LAYER
    ):
        raise Glm52RoutingProfileError(
            "frequency placement receipt does not bind the GLM-5.2 artifact"
        )

    canonical_capacity = (
        2 * (CANONICAL_INPUT_TOKENS + OUTPUT_TOKENS)
        + SCHEDULER_HEADROOM_TOKENS
    )
    resident_capacity = (
        2 * (RESIDENT_INPUT_TOKENS + OUTPUT_TOKENS)
        + SCHEDULER_HEADROOM_TOKENS
    )
    if canonical_capacity != baseline.token_capacity:
        raise Glm52RoutingProfileError(
            f"baseline token capacity {baseline.token_capacity} is not canonical "
            f"{canonical_capacity}"
        )
    canonical_budget = calculate_resident_budget(
        baseline,
        inventory,
        target_token_capacity=canonical_capacity,
        allocator_guard_mib=allocator_guard_mib,
    )
    if canonical_budget.maximum_safe_total_resident_experts != 0:
        raise Glm52RoutingProfileError(
            "canonical 16K lane unexpectedly admits residents; review guard policy"
        )
    resident_budget = calculate_resident_budget(
        baseline,
        inventory,
        target_token_capacity=resident_capacity,
        allocator_guard_mib=allocator_guard_mib,
    )
    admitted_sparse_budget = resident_budget.maximum_safe_total_resident_experts
    if admitted_sparse_budget <= 0:
        raise Glm52RoutingProfileError(
            "short matched lane still cannot admit any resident experts"
        )

    profiles: list[JsonValue] = []
    for chunk_size in CHUNK_SIZES:
        profile_id = f"glm52-long-zero-resident-chunk{chunk_size}"
        arguments = _profile_arguments(
            profile_id=profile_id,
            result_root=result_root,
            input_tokens=CANONICAL_INPUT_TOKENS,
            chunk_size=chunk_size,
            resident_budget=0,
            placement_strategy="uniform",
            placement_artifact=None,
            placement_sha256=None,
        )
        profile: JsonObject = {
            "profile_id": profile_id,
            "lane": "canonical_long_zero_resident",
            "admitted": True,
            "benchmark_concurrencies": [1, 2],
            "input_tokens_per_request": CANONICAL_INPUT_TOKENS,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "maximum_total_tokens": canonical_capacity,
            "chunked_prefill_size": chunk_size,
            "total_resident_gpu_experts": 0,
            "resident_bytes_per_tp_rank": 0,
            "placement_strategy": "uniform",
            "max_deferred_experts_per_token": 0,
            "launch_arguments": arguments,
        }
        profile["profile_content_sha256"] = _canonical_sha256(profile)
        profiles.append(profile)

    for chunk_size in CHUNK_SIZES:
        for resident_count, strategy in (
            (0, "uniform"),
            (admitted_sparse_budget, "uniform"),
            (admitted_sparse_budget, "frequency"),
        ):
            typed_strategy = cast(PlacementStrategy, strategy)
            suffix = (
                "zero"
                if resident_count == 0
                else f"{strategy}-global{resident_count}"
            )
            profile_id = f"glm52-resident-ab-{suffix}-chunk{chunk_size}"
            arguments = _profile_arguments(
                profile_id=profile_id,
                result_root=result_root,
                input_tokens=RESIDENT_INPUT_TOKENS,
                chunk_size=chunk_size,
                resident_budget=resident_count,
                placement_strategy=typed_strategy,
                placement_artifact=(
                    placement_artifact if typed_strategy == "frequency" else None
                ),
                placement_sha256=(
                    placement_sha256 if typed_strategy == "frequency" else None
                ),
            )
            profile = {
                "profile_id": profile_id,
                "lane": "matched_short_resident_ab",
                "admitted": True,
                "benchmark_concurrencies": [1, 2],
                "input_tokens_per_request": RESIDENT_INPUT_TOKENS,
                "output_tokens_per_request": OUTPUT_TOKENS,
                "maximum_total_tokens": resident_capacity,
                "chunked_prefill_size": chunk_size,
                "total_resident_gpu_experts": resident_count,
                "kt_gpu_experts_ratio": (
                    None
                    if resident_count == 0
                    else _ratio_for_exact_total_budget(resident_count)
                ),
                "resident_bytes_per_tp_rank": (
                    resident_count * inventory.per_tp_rank_expert_bytes
                ),
                "placement_strategy": typed_strategy,
                "frequency_input": (
                    None
                    if typed_strategy != "frequency"
                    else {
                        "path": str(placement_artifact),
                        "sha256": placement_sha256,
                    }
                ),
                "max_deferred_experts_per_token": 0,
                "launch_arguments": arguments,
            }
            profile["profile_content_sha256"] = _canonical_sha256(profile)
            profiles.append(profile)

    rejected_profiles: list[JsonValue] = []
    for strategy in ("uniform", "frequency"):
        rejected_profiles.append(
            {
                "profile_id": f"glm52-long-resident-{strategy}-rejected",
                "lane": "canonical_long_resident",
                "admitted": False,
                "maximum_total_tokens": canonical_capacity,
                "placement_strategy": strategy,
                "requested_minimum_total_resident_experts": 1,
                "reason": canonical_budget.rejection_reason,
            }
        )

    plan: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "kind": "glm52_low_concurrency_routing_residency_profiles_v1",
        "model_contract": {
            "layer_count": MODEL_LAYER_COUNT,
            "first_routed_layer": FIRST_ROUTED_LAYER,
            "last_routed_layer": LAST_ROUTED_LAYER,
            "routed_layer_count": ROUTED_LAYER_COUNT,
            "experts_per_layer": EXPERT_COUNT,
            "experts_per_token": EXPERTS_PER_TOKEN,
            "total_routed_expert_positions": TOTAL_ROUTED_EXPERT_POSITIONS,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        },
        "expert_tensor_inventory": cast(JsonObject, asdict(inventory)),
        "baseline_capacity": cast(JsonObject, asdict(baseline)),
        "budget_policy": {
            "method": (
                "observed minimum post-readiness free VRAM plus conservative "
                "linear KV reclaim, minus observed floor and allocator guard"
            ),
            "allocator_guard_mib": allocator_guard_mib,
            "canonical_16k": cast(JsonObject, asdict(canonical_budget)),
            "matched_short": cast(JsonObject, asdict(resident_budget)),
            "admitted_sparse_global_budget": admitted_sparse_budget,
            "admitted_ratio": _ratio_for_exact_total_budget(
                admitted_sparse_budget
            ),
            "per_rank_resident_bytes": (
                admitted_sparse_budget * inventory.per_tp_rank_expert_bytes
            ),
        },
        "frequency_input": {
            "path": str(placement_artifact),
            "sha256": placement_sha256,
            "receipt_path": str(placement_receipt_path),
            "receipt_sha256": placement_receipt_sha256,
        },
        "profiles": profiles,
        "rejected_profiles": rejected_profiles,
        "exactness": {
            "max_deferred_experts_per_token": 0,
            "deferred_expert_lane": "excluded_as_approximate_model",
            "dynamic_expert_update": False,
            "semantic_coherency_warmup_required": True,
            "server_restart_between_profiles_required": True,
            "heavyweight_sweep_requested": False,
        },
        "known_runtime_constraints": [
            (
                "--kt-num-gpu-experts is per routed layer; value 1 means 75 "
                "resident experts and is not admitted by this VRAM evidence"
            ),
            (
                "KT uniform placement assigns a sub-layer-count remainder to "
                "the earliest routed layers rather than spacing it across depth"
            ),
            (
                "chunk size changes prefill scheduling but does not by itself "
                "create VRAM for the canonical 16K KV working set"
            ),
        ],
        "benchmark_script": str(DEFAULT_BENCHMARK_SCRIPT),
    }
    plan["receipt_content_sha256"] = _canonical_sha256(plan)
    return plan


def _positive_integer(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_integer(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _sha256_argument(raw: str) -> str:
    if len(raw) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in raw
    ):
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    capture_parser.add_argument(
        "--recorder-directory",
        type=Path,
        required=True,
    )
    capture_parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    capture_parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(DEFAULT_RUNTIME_PYTHON),
    )
    capture_parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=1_800.0,
    )

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--source", type=Path, required=True)
    materialize_parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    materialize_parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(DEFAULT_RUNTIME_PYTHON),
    )
    materialize_parser.add_argument(
        "--prompt-manifest-sha256",
        type=_sha256_argument,
        default=representative_prompt_manifest()["manifest_content_sha256"],
    )

    worker_parser = subparsers.add_parser("_torch-materialize")
    worker_parser.add_argument("--source", type=Path, required=True)
    worker_parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    worker_parser.add_argument(
        "--prompt-manifest-sha256",
        type=_sha256_argument,
        required=True,
    )

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(DEFAULT_MODEL_PATH),
    )
    plan_parser.add_argument(
        "--baseline-receipt",
        type=Path,
        required=True,
    )
    plan_parser.add_argument(
        "--frequency-input",
        type=Path,
        required=True,
    )
    plan_parser.add_argument(
        "--frequency-input-sha256",
        type=_sha256_argument,
        required=True,
    )
    plan_parser.add_argument(
        "--frequency-receipt",
        type=Path,
        required=True,
    )
    plan_parser.add_argument(
        "--frequency-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    plan_parser.add_argument("--result-root", type=Path, required=True)
    plan_parser.add_argument("--output-directory", type=Path, required=True)
    plan_parser.add_argument(
        "--vram-floor-mib",
        type=_positive_integer,
        default=DEFAULT_VRAM_FLOOR_MIB,
    )
    plan_parser.add_argument(
        "--allocator-guard-mib",
        type=_nonnegative_integer,
        default=DEFAULT_ALLOCATOR_GUARD_MIB,
    )
    return parser


def _absolute(path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise Glm52RoutingProfileError(f"{description} must be absolute")
    return path


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "capture":
            receipt = capture_routing(
                server_url=cast(str, arguments.server_url),
                recorder_directory=_absolute(
                    cast(Path, arguments.recorder_directory),
                    "recorder directory",
                ),
                output_directory=_absolute(
                    cast(Path, arguments.output_directory),
                    "output directory",
                ),
                runtime_python=_absolute(
                    cast(Path, arguments.runtime_python),
                    "runtime Python",
                ),
                timeout_seconds=cast(float, arguments.timeout_seconds),
            )
        elif arguments.command == "materialize":
            receipt = materialize_routing_artifact(
                runtime_python=_absolute(
                    cast(Path, arguments.runtime_python),
                    "runtime Python",
                ),
                source_path=_absolute(
                    cast(Path, arguments.source),
                    "routing source",
                ),
                output_directory=_absolute(
                    cast(Path, arguments.output_directory),
                    "output directory",
                ),
                prompt_manifest_sha256=cast(
                    str,
                    arguments.prompt_manifest_sha256,
                ),
            )
        elif arguments.command == "_torch-materialize":
            receipt = _materialize_with_torch(
                _absolute(cast(Path, arguments.source), "routing source"),
                _absolute(
                    cast(Path, arguments.output_directory),
                    "output directory",
                ),
                cast(str, arguments.prompt_manifest_sha256),
            )
        elif arguments.command == "plan":
            inventory = inspect_expert_tensor_inventory(
                _absolute(cast(Path, arguments.model_path), "model path")
            )
            baseline = read_baseline_capacity(
                _absolute(
                    cast(Path, arguments.baseline_receipt),
                    "baseline receipt",
                ),
                expected_vram_floor_mib=cast(int, arguments.vram_floor_mib),
            )
            plan = build_profile_plan(
                baseline=baseline,
                inventory=inventory,
                placement_artifact=_absolute(
                    cast(Path, arguments.frequency_input),
                    "frequency input",
                ),
                placement_sha256=cast(
                    str,
                    arguments.frequency_input_sha256,
                ),
                placement_receipt_path=_absolute(
                    cast(Path, arguments.frequency_receipt),
                    "frequency receipt",
                ),
                placement_receipt_sha256=cast(
                    str,
                    arguments.frequency_receipt_sha256,
                ),
                result_root=_absolute(
                    cast(Path, arguments.result_root),
                    "result root",
                ),
                allocator_guard_mib=cast(int, arguments.allocator_guard_mib),
            )
            output_directory = _absolute(
                cast(Path, arguments.output_directory),
                "output directory",
            )
            plan_path = output_directory / (
                "glm52-low-concurrency-profiles-"
                f"{cast(str, plan['receipt_content_sha256'])}.json"
            )
            _write_immutable_json(plan_path, plan)
            receipt = {**plan, "receipt_path": str(plan_path)}
        else:
            raise Glm52RoutingProfileError(
                f"unknown command {arguments.command!r}"
            )
    except (
        Glm52RoutingProfileError,
        httpx.HTTPError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"GLM-5.2 routing/profile operation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
