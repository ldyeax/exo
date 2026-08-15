#!/usr/bin/env python3
# ruff: noqa: E402
"""Drive and finalize real-model DSV4 OSCAR INT2 calibration capture.

This program never launches the model itself.  ``prepare`` creates a bounded
session/config, ``request`` serially arms each hashed train/heldout prompt and
sends it to a dedicated local server, and ``finalize`` converts strict TP2 raw
shards into the capture manifest consumed by
``dsv4_oscar_int2_calibration.py collect``.  No missing tensor is synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

import torch

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SGLANG_PYTHON: Final = REPO_ROOT / "vendor" / "sglang" / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))

# The local SGLang and repository roots must be inserted before importing the
# capture implementations below.
from sglang.srt.layers.attention.dsv4.oscar_int2_capture import (
    CONFIG_FORMAT,
    CONTROL_FORMAT,
    FORMAT_VERSION,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    LATENT_DIM,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    RAW_FORMAT,
)

from scripts.dsv4_oscar_int2_calibration import (
    EXPECTED_COMPRESSION_RATIOS,
    PROMPT_FORMAT,
    PROMPT_VERSION,
    sha256_file,
)
from scripts.dsv4_oscar_int2_capture import (
    CaptureTensorSet,
    Dsv4OscarCaptureWriter,
)

Split = Literal["train", "heldout"]
JsonObject = dict[str, Any]
KINDS: Final[tuple[str, ...]] = (
    "attention_query_nope",
    "swa_latent",
    "compressed_latent",
    "c4_scorer_query",
    "c4_scorer_key",
)
CALIBRATION_HISTORY_ROWS_PER_PROMPT: Final = 128


def _load_json(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(JsonObject, loaded)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _regular_absolute(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    return path.resolve()


def _prompt_document(path: Path) -> tuple[JsonObject, dict[str, Split]]:
    path = _regular_absolute(path, label="prompt manifest")
    document = _load_json(path)
    if (
        document.get("format") != PROMPT_FORMAT
        or document.get("format_version") != PROMPT_VERSION
    ):
        raise ValueError("incompatible DSV4 OSCAR calibration prompt manifest")
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("calibration prompt manifest contains no prompts")
    prompt_splits: dict[str, Split] = {}
    for raw_prompt in raw_prompts:
        if not isinstance(raw_prompt, dict):
            raise TypeError("each calibration prompt must be an object")
        prompt_id = raw_prompt.get("id")
        split = raw_prompt.get("split")
        text = raw_prompt.get("text")
        token_count = raw_prompt.get("token_count")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or prompt_id in prompt_splits
            or split not in ("train", "heldout")
            or not isinstance(text, str)
            or not text
            or not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count < 128
        ):
            raise ValueError(
                "each capture prompt needs a unique id, split, non-empty text, "
                "and at least 128 declared tokens"
            )
        prompt_splits[prompt_id] = cast(Split, split)
    if set(prompt_splits.values()) != {"train", "heldout"}:
        raise ValueError("capture prompts must contain train and heldout splits")
    return document, prompt_splits


def _default_row_limits(
    prompt_splits: Mapping[str, Split],
) -> dict[str, dict[str, int]]:
    counts = {
        split: sum(value == split for value in prompt_splits.values())
        for split in cast(tuple[Split, Split], ("train", "heldout"))
    }
    minimum_rows: dict[Split, int] = {"train": 32, "heldout": 16}
    per_split = {
        split: max(1, math.ceil(minimum_rows[split] / counts[split]))
        for split in cast(tuple[Split, Split], ("train", "heldout"))
    }
    limits = {
        kind: {split: per_split[split] for split in ("train", "heldout")}
        for kind in KINDS
    }
    # Query covariance receives one row for every attention head, so four
    # token positions per prompt already supply thousands of observations.
    # C128 history, by contrast, yields only one compressed row per 128-token
    # boundary. Retain up to 128 real history/key rows per prompt without
    # inflating the query or protected-SWA capture. The capture hook naturally
    # records fewer rows when a request contains fewer compression boundaries.
    for kind in ("compressed_latent", "c4_scorer_key"):
        limits[kind] = {
            "train": CALIBRATION_HISTORY_ROWS_PER_PROMPT,
            "heldout": CALIBRATION_HISTORY_ROWS_PER_PROMPT,
        }
    return limits


def projected_raw_tensor_bytes(
    prompt_splits: Mapping[str, Split], limits: Mapping[str, Mapping[str, int]]
) -> int:
    rows = {
        kind: {
            split: sum(value == split for value in prompt_splits.values())
            * limits[kind][split]
            for split in ("train", "heldout")
        }
        for kind in KINDS
    }
    # BF16 tensors only.  Attention Q is split 32+32 across TP2, so its total
    # payload is still the full 64-head shape.  The remaining domains are
    # replicated and written by rank zero only.
    total = 0
    for split in ("train", "heldout"):
        total += (
            NUM_LAYERS
            * rows["attention_query_nope"][split]
            * NUM_ATTENTION_HEADS
            * LATENT_DIM
            * 2
        )
        total += NUM_LAYERS * rows["swa_latent"][split] * LATENT_DIM * 2
        compressed_layers = sum(value != 0 for value in EXPECTED_COMPRESSION_RATIOS)
        c4_layers = sum(value == 4 for value in EXPECTED_COMPRESSION_RATIOS)
        total += compressed_layers * rows["compressed_latent"][split] * LATENT_DIM * 2
        total += (
            c4_layers
            * rows["c4_scorer_query"][split]
            * INDEX_HEADS
            * INDEX_HEAD_DIM
            * 2
        )
        total += c4_layers * rows["c4_scorer_key"][split] * INDEX_HEAD_DIM * 2
    return total


def _config_path(session_dir: Path) -> Path:
    return session_dir / "runtime_capture_config.json"


def _control_path(session_dir: Path) -> Path:
    return session_dir / "capture_control.json"


def prepare_session(
    *,
    session_dir: Path,
    prompt_manifest: Path,
    checkpoint_path: Path,
    checkpoint_fingerprint: Path,
    model_id: str,
    maximum_cpu_bytes: int,
) -> Path:
    if not session_dir.is_absolute():
        raise ValueError("session_dir must be absolute")
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError("capture session_dir must be absent or empty")
    session_dir.mkdir(parents=True, exist_ok=True)
    if session_dir.is_symlink():
        raise ValueError("capture session_dir cannot be a symlink")
    prompt_manifest = _regular_absolute(prompt_manifest, label="prompt manifest")
    checkpoint_fingerprint = _regular_absolute(
        checkpoint_fingerprint, label="checkpoint fingerprint"
    )
    if (
        not checkpoint_path.is_absolute()
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_dir()
    ):
        raise ValueError("checkpoint_path must be an absolute regular directory")
    if not model_id:
        raise ValueError("model_id cannot be empty")
    _, prompt_splits = _prompt_document(prompt_manifest)
    limits = _default_row_limits(prompt_splits)
    projected_bytes = projected_raw_tensor_bytes(prompt_splits, limits)
    if projected_bytes > maximum_cpu_bytes:
        raise ValueError(
            f"bounded capture needs at most {projected_bytes} tensor bytes, above "
            f"--maximum-cpu-bytes={maximum_cpu_bytes}"
        )
    session_id = hashlib.sha256(
        (
            f"{sha256_file(prompt_manifest)}:{sha256_file(checkpoint_fingerprint)}:"
            f"{secrets.token_hex(32)}"
        ).encode()
    ).hexdigest()
    control = {
        "format": CONTROL_FORMAT,
        "format_version": FORMAT_VERSION,
        "generation": 0,
        "state": "idle",
    }
    control_path = _control_path(session_dir)
    _atomic_json(control, control_path)
    config = {
        "format": CONFIG_FORMAT,
        "format_version": FORMAT_VERSION,
        "session_id": session_id,
        "session_dir": str(session_dir.resolve()),
        "control_path": str(control_path.resolve()),
        "expected_tp_size": 2,
        "prompt_manifest_path": str(prompt_manifest),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        "prompt_splits": prompt_splits,
        "maximum_rows_per_prompt": limits,
        "projected_raw_tensor_bytes": projected_bytes,
        "maximum_cpu_bytes": maximum_cpu_bytes,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_fingerprint_path": str(checkpoint_fingerprint),
        "checkpoint_fingerprint_sha256": sha256_file(checkpoint_fingerprint),
        "model_id": model_id,
    }
    path = _config_path(session_dir)
    _atomic_json(config, path)
    return path


def _load_session(session_dir: Path) -> tuple[Path, JsonObject, JsonObject]:
    if (
        not session_dir.is_absolute()
        or session_dir.is_symlink()
        or not session_dir.is_dir()
    ):
        raise ValueError("session_dir must be an absolute regular directory")
    config_path = _regular_absolute(_config_path(session_dir), label="runtime config")
    control_path = _regular_absolute(
        _control_path(session_dir), label="capture control"
    )
    config = _load_json(config_path)
    control = _load_json(control_path)
    if (
        config.get("format") != CONFIG_FORMAT
        or config.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("incompatible runtime capture session")
    if (
        control.get("format") != CONTROL_FORMAT
        or control.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("incompatible runtime capture control")
    if config.get("control_path") != str(control_path):
        raise ValueError("runtime config does not bind the session control file")
    return config_path, config, control


def set_control(
    *, session_dir: Path, prompt_id: str | None, split: Split | None
) -> None:
    _, config, control = _load_session(session_dir)
    generation = control.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise TypeError("capture control generation is invalid")
    if prompt_id is None:
        document: JsonObject = {
            "format": CONTROL_FORMAT,
            "format_version": FORMAT_VERSION,
            "generation": generation + 1,
            "state": "idle",
        }
    else:
        if control.get("state") != "idle":
            raise ValueError("capture is already armed; disarm before re-arming")
        prompt_splits = config.get("prompt_splits")
        if not isinstance(prompt_splits, dict) or prompt_splits.get(prompt_id) != split:
            raise ValueError("prompt id/split is not bound by the capture config")
        document = {
            "format": CONTROL_FORMAT,
            "format_version": FORMAT_VERSION,
            "generation": generation + 1,
            "state": "armed",
            "prompt_id": prompt_id,
            "split": split,
        }
    _atomic_json(document, _control_path(session_dir))


def _local_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("capture requests are restricted to a local HTTP server")
    return base_url.rstrip("/") + "/v1/chat/completions"


def _post_prompt(
    *, endpoint: str, model: str, text: str, timeout_seconds: float
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"capture request failed: HTTP {error.code}: {detail}"
        ) from error
    decoded = json.loads(payload)
    if not isinstance(decoded, dict) or decoded.get("error") is not None:
        raise RuntimeError("capture server returned an invalid/error response")
    return hashlib.sha256(payload).hexdigest()


def request_all_prompts(
    *, session_dir: Path, base_url: str, served_model: str, timeout_seconds: float
) -> Path:
    _, config, control = _load_session(session_dir)
    if control.get("state") != "idle":
        raise ValueError("capture control must be idle before request")
    prompt_path_value = config.get("prompt_manifest_path")
    if not isinstance(prompt_path_value, str):
        raise TypeError("runtime config has no prompt manifest path")
    prompt_path = _regular_absolute(Path(prompt_path_value), label="prompt manifest")
    if sha256_file(prompt_path) != config.get("prompt_manifest_sha256"):
        raise ValueError("prompt manifest changed after capture prepare")
    prompt_document, _ = _prompt_document(prompt_path)
    prompts = cast(list[JsonObject], prompt_document["prompts"])
    endpoint = _local_endpoint(base_url)
    receipts: list[JsonObject] = []
    for prompt in prompts:
        prompt_id = cast(str, prompt["id"])
        split = cast(Split, prompt["split"])
        set_control(session_dir=session_dir, prompt_id=prompt_id, split=split)
        try:
            response_sha = _post_prompt(
                endpoint=endpoint,
                model=served_model,
                text=cast(str, prompt["text"]),
                timeout_seconds=timeout_seconds,
            )
        finally:
            set_control(session_dir=session_dir, prompt_id=None, split=None)
        receipts.append(
            {"prompt_id": prompt_id, "split": split, "response_sha256": response_sha}
        )
    receipt = {
        "format": "dsv4-oscar-int2-runtime-capture-requests",
        "format_version": 1,
        "session_id": config.get("session_id"),
        "prompt_manifest_sha256": config.get("prompt_manifest_sha256"),
        "served_model": served_model,
        "requests": receipts,
    }
    path = session_dir / "request_receipt.json"
    if path.exists():
        raise FileExistsError("capture request receipt already exists")
    _atomic_json(receipt, path)
    return path


def _raw_path(
    session_dir: Path, rank: int, layer_id: int, split: Split, kind: str
) -> Path:
    return (
        session_dir
        / "raw"
        / f"rank_{rank:02d}"
        / f"layer_{layer_id:02d}"
        / split
        / f"{kind}.pt"
    )


def _load_raw(
    *,
    session_dir: Path,
    session_id: str,
    config_sha256: str,
    rank: int,
    layer_id: int,
    split: Split,
    kind: str,
    tail: tuple[int, ...],
    head_start: int | None,
    expected_prompt_ids: set[str],
) -> tuple[torch.Tensor, list[str], torch.Tensor, Mapping[str, object]]:
    path = _raw_path(session_dir, rank, layer_id, split, kind)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing real-model capture: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"raw capture must be an object: {path}")
    expected = {
        "format": RAW_FORMAT,
        "format_version": FORMAT_VERSION,
        "config_sha256": config_sha256,
        "session_id": session_id,
        "kind": kind,
        "layer_id": layer_id,
        "split": split,
        "tp_rank": rank,
        "tp_size": 2,
        "head_start": head_start,
        "tail": list(tail),
    }
    for key, value in expected.items():
        if loaded.get(key) != value:
            raise ValueError(f"raw capture metadata mismatch for {path}: {key}")
    tensor = loaded.get("tensor")
    row_prompt_ids = loaded.get("row_prompt_ids")
    priorities = loaded.get("priorities")
    seen = loaded.get("seen_rows_by_prompt")
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != len(tail) + 1
        or tuple(tensor.shape[1:]) != tail
        or tensor.shape[0] == 0
        or not bool(torch.isfinite(tensor).all())
        or not isinstance(row_prompt_ids, list)
        or len(row_prompt_ids) != tensor.shape[0]
        or not all(isinstance(value, str) for value in row_prompt_ids)
        or set(row_prompt_ids) != expected_prompt_ids
        or not isinstance(priorities, torch.Tensor)
        or priorities.dtype != torch.float64
        or priorities.shape != (tensor.shape[0],)
        or not isinstance(seen, dict)
        or set(seen) != expected_prompt_ids
        or not all(isinstance(value, int) and value > 0 for value in seen.values())
    ):
        raise ValueError(f"raw capture payload/provenance is incomplete: {path}")
    return tensor.contiguous(), cast(list[str], row_prompt_ids), priorities, seen


def _load_attention_query(
    *,
    session_dir: Path,
    session_id: str,
    config_sha256: str,
    layer_id: int,
    split: Split,
    expected_prompt_ids: set[str],
) -> torch.Tensor:
    shards = [
        _load_raw(
            session_dir=session_dir,
            session_id=session_id,
            config_sha256=config_sha256,
            rank=rank,
            layer_id=layer_id,
            split=split,
            kind="attention_query_nope",
            tail=(NUM_ATTENTION_HEADS // 2, LATENT_DIM),
            head_start=rank * (NUM_ATTENTION_HEADS // 2),
            expected_prompt_ids=expected_prompt_ids,
        )
        for rank in (0, 1)
    ]
    first_tensor, first_ids, first_priorities, first_seen = shards[0]
    second_tensor, second_ids, second_priorities, second_seen = shards[1]
    if (
        first_ids != second_ids
        or not torch.equal(first_priorities, second_priorities)
        or dict(first_seen) != dict(second_seen)
        or first_tensor.shape[0] != second_tensor.shape[0]
    ):
        raise ValueError(
            f"TP2 query shards sampled different token rows at layer {layer_id}/{split}"
        )
    query = torch.cat((first_tensor, second_tensor), dim=1).contiguous()
    if tuple(query.shape[1:]) != (NUM_ATTENTION_HEADS, LATENT_DIM):
        raise AssertionError("TP2 attention query gather did not produce 64 heads")
    return query


def finalize_capture(*, session_dir: Path, output_dir: Path) -> Path:
    config_path, config, control = _load_session(session_dir)
    if control.get("state") != "idle":
        raise ValueError("capture must be disarmed before finalize")
    if not output_dir.is_absolute() or output_dir.is_symlink():
        raise ValueError("capture output_dir must be an absolute non-symlink path")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("capture output_dir must be absent or empty")
    session_id = config.get("session_id")
    prompt_splits = config.get("prompt_splits")
    if not isinstance(session_id, str) or not isinstance(prompt_splits, dict):
        raise TypeError("runtime capture config is malformed")
    checkpoint_path = config.get("checkpoint_path")
    fingerprint_path = config.get("checkpoint_fingerprint_path")
    prompt_path = config.get("prompt_manifest_path")
    model_id = config.get("model_id")
    if not all(
        isinstance(value, str)
        for value in (checkpoint_path, fingerprint_path, prompt_path, model_id)
    ):
        raise ValueError("runtime capture provenance is incomplete")
    config_sha256 = sha256_file(config_path)
    if sha256_file(Path(cast(str, fingerprint_path))) != config.get(
        "checkpoint_fingerprint_sha256"
    ):
        raise ValueError("checkpoint fingerprint changed after capture prepare")
    if sha256_file(Path(cast(str, prompt_path))) != config.get(
        "prompt_manifest_sha256"
    ):
        raise ValueError("prompt manifest changed after capture prepare")
    writer = Dsv4OscarCaptureWriter(
        output_dir=output_dir,
        checkpoint_path=Path(cast(str, checkpoint_path)),
        checkpoint_fingerprint_path=Path(cast(str, fingerprint_path)),
        prompt_manifest_path=Path(cast(str, prompt_path)),
        model_id=cast(str, model_id),
    )
    for layer_id, compression_ratio in enumerate(EXPECTED_COMPRESSION_RATIOS):
        for split in cast(tuple[Split, Split], ("train", "heldout")):
            expected_prompt_ids = {
                cast(str, prompt_id)
                for prompt_id, value in prompt_splits.items()
                if value == split
            }
            query = _load_attention_query(
                session_dir=session_dir,
                session_id=session_id,
                config_sha256=config_sha256,
                layer_id=layer_id,
                split=split,
                expected_prompt_ids=expected_prompt_ids,
            )
            swa, _, _, _ = _load_raw(
                session_dir=session_dir,
                session_id=session_id,
                config_sha256=config_sha256,
                rank=0,
                layer_id=layer_id,
                split=split,
                kind="swa_latent",
                tail=(LATENT_DIM,),
                head_start=None,
                expected_prompt_ids=expected_prompt_ids,
            )
            compressed = None
            scorer_query = None
            scorer_key = None
            if compression_ratio:
                compressed, _, _, _ = _load_raw(
                    session_dir=session_dir,
                    session_id=session_id,
                    config_sha256=config_sha256,
                    rank=0,
                    layer_id=layer_id,
                    split=split,
                    kind="compressed_latent",
                    tail=(LATENT_DIM,),
                    head_start=None,
                    expected_prompt_ids=expected_prompt_ids,
                )
            if compression_ratio == 4:
                scorer_query, _, _, _ = _load_raw(
                    session_dir=session_dir,
                    session_id=session_id,
                    config_sha256=config_sha256,
                    rank=0,
                    layer_id=layer_id,
                    split=split,
                    kind="c4_scorer_query",
                    tail=(INDEX_HEADS, INDEX_HEAD_DIM),
                    head_start=0,
                    expected_prompt_ids=expected_prompt_ids,
                )
                scorer_key, _, _, _ = _load_raw(
                    session_dir=session_dir,
                    session_id=session_id,
                    config_sha256=config_sha256,
                    rank=0,
                    layer_id=layer_id,
                    split=split,
                    kind="c4_scorer_key",
                    tail=(INDEX_HEAD_DIM,),
                    head_start=None,
                    expected_prompt_ids=expected_prompt_ids,
                )
            writer.add_chunk(
                layer_id=layer_id,
                split=split,
                prompt_ids=tuple(sorted(expected_prompt_ids)),
                tensors=CaptureTensorSet(
                    attention_query_nope=query,
                    swa_latent=swa,
                    compressed_latent=compressed,
                    c4_scorer_query=scorer_query,
                    c4_scorer_key=scorer_key,
                ),
            )
    return writer.finalize()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--session-dir", type=Path, required=True)
    prepare.add_argument("--prompt-manifest", type=Path, required=True)
    prepare.add_argument("--checkpoint-path", type=Path, required=True)
    prepare.add_argument("--checkpoint-fingerprint", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--maximum-cpu-bytes", type=int, default=512 * 1024 * 1024)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--session-dir", type=Path, required=True)
    arm.add_argument("--prompt-id", required=True)
    arm.add_argument("--split", choices=("train", "heldout"), required=True)
    disarm = subparsers.add_parser("disarm")
    disarm.add_argument("--session-dir", type=Path, required=True)
    request = subparsers.add_parser("request")
    request.add_argument("--session-dir", type=Path, required=True)
    request.add_argument("--base-url", default="http://127.0.0.1:30010")
    request.add_argument("--served-model", required=True)
    request.add_argument("--timeout-seconds", type=float, default=600.0)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--session-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_session(
            session_dir=args.session_dir,
            prompt_manifest=args.prompt_manifest,
            checkpoint_path=args.checkpoint_path,
            checkpoint_fingerprint=args.checkpoint_fingerprint,
            model_id=args.model_id,
            maximum_cpu_bytes=args.maximum_cpu_bytes,
        )
        print(path)
    elif args.command == "arm":
        set_control(
            session_dir=args.session_dir,
            prompt_id=args.prompt_id,
            split=cast(Split, args.split),
        )
    elif args.command == "disarm":
        set_control(session_dir=args.session_dir, prompt_id=None, split=None)
    elif args.command == "request":
        print(
            request_all_prompts(
                session_dir=args.session_dir,
                base_url=args.base_url,
                served_model=args.served_model,
                timeout_seconds=args.timeout_seconds,
            )
        )
    elif args.command == "finalize":
        print(
            finalize_capture(session_dir=args.session_dir, output_dir=args.output_dir)
        )
    else:
        raise AssertionError(f"unhandled command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
