#!/usr/bin/env python3
"""Provenance-safe capture writer for DSV4 OSCAR INT2 calibration.

This module is intentionally independent of SGLang internals so a temporary
forward hook can import it without changing the cache implementation.  Tensor
semantics are strict:

* ``attention_query_nope`` is post-query-normalization and post-RoPE slicing,
  before OSCAR, shape ``[tokens, 64, 448]``.  RoPE affects only the excluded
  tail, so this is the actual score-side consumer of the shared latent.
* ``swa_latent`` is the post-KV-normalization shared K=V noPE payload before
  OSCAR, shape ``[tokens, 448]``.  Never pass a conventional K/V head tensor.
* ``compressed_latent`` is the C4/C128 compressor output after its norm and
  before OSCAR, noPE slice only, shape ``[rows, 448]``.
* C4 ``scorer_query`` is weighted, post-scorer-RoPE, pre-Hadamard Q with shape
  ``[tokens, 64, 128]``; ``scorer_key`` is post-compressor-norm and before the
  existing fixed Hadamard, shape ``[rows, 128]``.

The writer hashes every tensor and emits the exact manifest consumed by
``dsv4_oscar_int2_calibration.py collect``.  It does not launch a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import torch

from scripts.dsv4_oscar_int2_calibration import (
    CAPTURE_FORMAT,
    CAPTURE_VERSION,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    LATENT_DIM,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    _atomic_json_save,
    _atomic_torch_save,
    _prompt_provenance,
    sha256_file,
    validate_model_config,
    verify_checkpoint_fingerprint,
)

Split = Literal["train", "heldout"]


@final
@dataclass(frozen=True)
class CaptureTensorSet:
    attention_query_nope: torch.Tensor
    swa_latent: torch.Tensor
    compressed_latent: torch.Tensor | None = None
    c4_scorer_query: torch.Tensor | None = None
    c4_scorer_key: torch.Tensor | None = None


def _capture_tensor(
    tensor: torch.Tensor,
    *,
    tail: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != len(tail) + 1
        or tuple(tensor.shape[1:]) != tail
        or tensor.shape[0] == 0
    ):
        raise ValueError(f"{label} must have shape [rows,{','.join(map(str, tail))}]")
    cpu = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError(f"{label} contains non-finite values")
    return cpu


@final
class Dsv4OscarCaptureWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        checkpoint_path: Path,
        checkpoint_fingerprint_path: Path,
        prompt_manifest_path: Path,
        model_id: str,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_fingerprint_path = checkpoint_fingerprint_path.resolve()
        self.prompt_manifest_path = prompt_manifest_path.resolve()
        fingerprint = verify_checkpoint_fingerprint(
            self.checkpoint_fingerprint_path, self.checkpoint_path
        )
        checkpoint_sha256 = fingerprint.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str):
            raise TypeError("checkpoint fingerprint has no aggregate digest")
        self.geometry = validate_model_config(
            self.checkpoint_path / "config.json",
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
        )
        self.prompt_sha256 = sha256_file(self.prompt_manifest_path)
        self.prompt_splits, _ = _prompt_provenance(
            self.prompt_manifest_path, self.prompt_sha256
        )
        self._chunks: list[dict[str, object]] = []
        self._chunk_counts: dict[tuple[int, Split], int] = {}

    def _write_tensor(
        self,
        *,
        directory: Path,
        name: str,
        tensor: torch.Tensor,
        tail: tuple[int, ...],
        label: str,
    ) -> dict[str, object]:
        cpu = _capture_tensor(tensor, tail=tail, label=label)
        path = directory / f"{name}.pt"
        if path.exists():
            raise FileExistsError(f"capture tensor already exists: {path}")
        _atomic_torch_save(cpu, path)
        return {
            "path": path.relative_to(self.output_dir).as_posix(),
            "sha256": sha256_file(path),
            "shape": list(cpu.shape),
        }

    def add_chunk(
        self,
        *,
        layer_id: int,
        split: Split,
        prompt_ids: tuple[str, ...],
        tensors: CaptureTensorSet,
    ) -> None:
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError("layer_id must be in [0, 42]")
        if split not in ("train", "heldout"):
            raise ValueError("split must be train or heldout")
        if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("prompt_ids must be non-empty and unique")
        if any(self.prompt_splits.get(prompt_id) != split for prompt_id in prompt_ids):
            raise ValueError("capture prompt IDs do not match the requested split")
        compress_ratio = self.geometry.compression_ratios[layer_id]
        if (compress_ratio != 0) != (tensors.compressed_latent is not None):
            raise ValueError("compressed_latent is required exactly on C4/C128 layers")
        has_scorer = (
            tensors.c4_scorer_query is not None and tensors.c4_scorer_key is not None
        )
        has_partial_scorer = (
            tensors.c4_scorer_query is not None or tensors.c4_scorer_key is not None
        )
        if (compress_ratio == 4) != has_scorer or (
            has_partial_scorer and not has_scorer
        ):
            raise ValueError("C4 scorer Q and K are required exactly on C4 layers")
        key = (layer_id, split)
        chunk_index = self._chunk_counts.get(key, 0)
        self._chunk_counts[key] = chunk_index + 1
        directory = (
            self.output_dir
            / "tensors"
            / f"layer_{layer_id:02d}"
            / split
            / f"chunk_{chunk_index:05d}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        prefix = f"layer{layer_id}.{split}.chunk{chunk_index}"
        chunk: dict[str, object] = {
            "layer_id": layer_id,
            "split": split,
            "prompt_ids": list(prompt_ids),
            "attention_query_nope": self._write_tensor(
                directory=directory,
                name="attention_query_nope",
                tensor=tensors.attention_query_nope,
                tail=(NUM_ATTENTION_HEADS, LATENT_DIM),
                label=f"{prefix}.attention_query_nope",
            ),
            "swa_latent": self._write_tensor(
                directory=directory,
                name="swa_latent",
                tensor=tensors.swa_latent,
                tail=(LATENT_DIM,),
                label=f"{prefix}.swa_latent",
            ),
        }
        if tensors.compressed_latent is not None:
            chunk["compressed_latent"] = self._write_tensor(
                directory=directory,
                name="compressed_latent",
                tensor=tensors.compressed_latent,
                tail=(LATENT_DIM,),
                label=f"{prefix}.compressed_latent",
            )
        if tensors.c4_scorer_query is not None and tensors.c4_scorer_key is not None:
            chunk["c4_scorer_query"] = self._write_tensor(
                directory=directory,
                name="c4_scorer_query",
                tensor=tensors.c4_scorer_query,
                tail=(INDEX_HEADS, INDEX_HEAD_DIM),
                label=f"{prefix}.c4_scorer_query",
            )
            chunk["c4_scorer_key"] = self._write_tensor(
                directory=directory,
                name="c4_scorer_key",
                tensor=tensors.c4_scorer_key,
                tail=(INDEX_HEAD_DIM,),
                label=f"{prefix}.c4_scorer_key",
            )
        self._chunks.append(chunk)

    def finalize(self) -> Path:
        missing = [
            (layer_id, split)
            for layer_id in range(NUM_LAYERS)
            for split in ("train", "heldout")
            if (layer_id, split) not in self._chunk_counts
        ]
        if missing:
            layer_id, split = missing[0]
            raise ValueError(
                f"capture is incomplete: layer {layer_id} has no {split} chunk"
            )
        referenced_prompts = {
            prompt_id
            for chunk in self._chunks
            for prompt_id in chunk["prompt_ids"]
            if isinstance(prompt_id, str)
        }
        if referenced_prompts != set(self.prompt_splits):
            raise ValueError(
                "capture does not reference every hashed calibration prompt"
            )
        manifest = {
            "format": CAPTURE_FORMAT,
            "format_version": CAPTURE_VERSION,
            "model": {
                "model_id": self.geometry.model_id,
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_fingerprint_path": str(self.checkpoint_fingerprint_path),
                "checkpoint_fingerprint_sha256": sha256_file(
                    self.checkpoint_fingerprint_path
                ),
            },
            "prompts": {
                "path": str(self.prompt_manifest_path),
                "sha256": self.prompt_sha256,
            },
            "chunks": self._chunks,
        }
        path = self.output_dir / "capture_manifest.json"
        if path.exists():
            raise FileExistsError(f"capture manifest already exists: {path}")
        _atomic_json_save(manifest, path)
        return path
