from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import dsv4_oscar_int2_runtime_capture as runtime_capture


def _write_prompt_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "dsv4-oscar-calibration-prompts",
                "format_version": 1,
                "prompts": [
                    {
                        "id": "train-a",
                        "split": "train",
                        "text": "train prompt",
                        "token_count": 256,
                    },
                    {
                        "id": "heldout-a",
                        "split": "heldout",
                        "text": "heldout prompt",
                        "token_count": 256,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_prepare_arm_disarm_is_provenance_bound(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.json"
    fingerprint_path = tmp_path / "fingerprint.json"
    checkpoint_path = tmp_path / "checkpoint"
    session_dir = tmp_path / "session"
    _write_prompt_manifest(prompt_path)
    fingerprint_path.write_text("{}", encoding="utf-8")
    checkpoint_path.mkdir()
    config_path = runtime_capture.prepare_session(
        session_dir=session_dir,
        prompt_manifest=prompt_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_fingerprint=fingerprint_path.resolve(),
        model_id="deepseek-v4-flash",
        maximum_cpu_bytes=512 * 1024 * 1024,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["expected_tp_size"] == 2
    assert config["prompt_manifest_sha256"] == runtime_capture.sha256_file(prompt_path)
    assert config["projected_raw_tensor_bytes"] <= config["maximum_cpu_bytes"]

    runtime_capture.set_control(
        session_dir=session_dir.resolve(), prompt_id="train-a", split="train"
    )
    control_path = session_dir / "capture_control.json"
    assert json.loads(control_path.read_text(encoding="utf-8"))["state"] == "armed"
    runtime_capture.set_control(
        session_dir=session_dir.resolve(), prompt_id=None, split=None
    )
    assert json.loads(control_path.read_text(encoding="utf-8"))["state"] == "idle"


def test_prepare_rejects_unbounded_budget(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.json"
    fingerprint_path = tmp_path / "fingerprint.json"
    checkpoint_path = tmp_path / "checkpoint"
    _write_prompt_manifest(prompt_path)
    fingerprint_path.write_text("{}", encoding="utf-8")
    checkpoint_path.mkdir()
    with pytest.raises(ValueError, match="bounded capture needs"):
        runtime_capture.prepare_session(
            session_dir=(tmp_path / "session").resolve(),
            prompt_manifest=prompt_path.resolve(),
            checkpoint_path=checkpoint_path.resolve(),
            checkpoint_fingerprint=fingerprint_path.resolve(),
            model_id="deepseek-v4-flash",
            maximum_cpu_bytes=1,
        )


def test_capture_requests_are_local_only() -> None:
    with pytest.raises(ValueError, match="local HTTP server"):
        runtime_capture._local_endpoint("https://example.com")
    assert (
        runtime_capture._local_endpoint("http://127.0.0.1:30000")
        == "http://127.0.0.1:30000/v1/chat/completions"
    )


def test_prompt_manifest_requires_c128_observation_length(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.json"
    _write_prompt_manifest(prompt_path)
    document = json.loads(prompt_path.read_text(encoding="utf-8"))
    document["prompts"][0]["token_count"] = 127
    prompt_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 128"):
        runtime_capture._prompt_document(prompt_path.resolve())


def _write_query_shard(
    session_dir: Path, *, rank: int, priorities: torch.Tensor
) -> None:
    path = runtime_capture._raw_path(
        session_dir, rank, 0, "train", "attention_query_nope"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": runtime_capture.RAW_FORMAT,
            "format_version": 1,
            "config_sha256": "c" * 64,
            "session_id": "s" * 64,
            "kind": "attention_query_nope",
            "layer_id": 0,
            "split": "train",
            "tp_rank": rank,
            "tp_size": 2,
            "head_start": rank * 32,
            "tail": [32, 448],
            "tensor": torch.full((2, 32, 448), float(rank + 1), dtype=torch.bfloat16),
            "priorities": priorities,
            "row_prompt_ids": ["train-a", "train-a"],
            "seen_rows_by_prompt": {"train-a": 10},
            "generations": [1],
        },
        path,
    )


def test_finalizer_strictly_combines_aligned_tp2_head_shards(tmp_path: Path) -> None:
    priorities = torch.tensor([0.1, 0.2], dtype=torch.float64)
    _write_query_shard(tmp_path, rank=0, priorities=priorities)
    _write_query_shard(tmp_path, rank=1, priorities=priorities)
    query = runtime_capture._load_attention_query(
        session_dir=tmp_path,
        session_id="s" * 64,
        config_sha256="c" * 64,
        layer_id=0,
        split="train",
        expected_prompt_ids={"train-a"},
    )
    assert query.shape == (2, 64, 448)
    assert torch.equal(query[:, :32], torch.ones_like(query[:, :32]))
    assert torch.equal(query[:, 32:], torch.full_like(query[:, 32:], 2.0))

    _write_query_shard(
        tmp_path,
        rank=1,
        priorities=torch.tensor([0.1, 0.3], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="sampled different token rows"):
        runtime_capture._load_attention_query(
            session_dir=tmp_path,
            session_id="s" * 64,
            config_sha256="c" * 64,
            layer_id=0,
            split="train",
            expected_prompt_ids={"train-a"},
        )
