from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import materialize_dsv4_frozen_hybrid_plan as materializer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "materialize_dsv4_frozen_hybrid_plan.py"
MANIFEST = (
    REPOSITORY_ROOT
    / "scripts"
    / "data"
    / "dsv4_flash_opencode_g14_p28_frozen_plan.json"
)
EXPECTED_PLACEMENT_SEMANTICS_SHA256 = (
    "c86ba036d3a5076f8d6f8e0a0745c8df4fdd4e460aa45c558cb91c903c722836"
)


def test_repository_frozen_plan_is_an_exact_ep2_hybrid_cover() -> None:
    placement = materializer.load_frozen_placement(MANIFEST)

    assert placement.name == "dsv4-flash-opencode-g14-p28"
    assert placement.gpu_experts_per_rank == 14
    assert placement.cpu_experts_per_rank == 114
    assert placement.hot_prefix_count == 28
    assert placement.placement_semantics_sha256 == (EXPECTED_PLACEMENT_SEMANTICS_SHA256)
    assert tuple(placement.gpu_masks_by_rank.shape) == (2, 43, 256)
    assert placement.gpu_masks_by_rank.sum(dim=2).unique().tolist() == [14]
    for layer in range(43):
        layer_experts: list[int] = []
        for rank in range(2):
            layer_experts.extend(
                torch.where(placement.gpu_masks_by_rank[rank, layer])[0].tolist()
            )
            layer_experts.extend(placement.cpu_expert_ids_by_rank[rank][layer].tolist())
        assert sorted(layer_experts) == list(range(256))


def test_cli_materializes_a_weights_only_runtime_plan(tmp_path: Path) -> None:
    output_path = tmp_path / "g14-p28.pt"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["materialization"] == "created"
    assert receipt["output_sha256"] == materializer.sha256_file(output_path)
    assert receipt["placement_semantics_sha256"] == (
        EXPECTED_PLACEMENT_SEMANTICS_SHA256
    )
    plan = torch.load(output_path, map_location="cpu", weights_only=True)
    assert plan["format"] == materializer.PLAN_FORMAT
    assert plan["placement_semantics_sha256"] == (EXPECTED_PLACEMENT_SEMANTICS_SHA256)
    assert plan["gpu_rank_counts"].tolist() == [14, 14]
    assert plan["cpu_rank_counts"].tolist() == [114, 114]
    assert plan["frozen_placement_name"] == "dsv4-flash-opencode-g14-p28"

    original_sha256 = materializer.sha256_file(output_path)
    original_inode = output_path.stat().st_ino
    repeated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 0, repeated.stderr
    repeated_receipt = json.loads(repeated.stdout)
    assert repeated_receipt["materialization"] == "reused"
    assert repeated_receipt["output_sha256"] == original_sha256
    assert materializer.sha256_file(output_path) == original_sha256
    assert output_path.stat().st_ino == original_inode


def test_runtime_plan_serialization_is_byte_deterministic_across_paths(
    tmp_path: Path,
) -> None:
    placement = materializer.load_frozen_placement(MANIFEST)
    manifest_sha256 = materializer.sha256_file(MANIFEST)
    plan = materializer.build_runtime_plan(
        placement,
        manifest_path=MANIFEST,
        source_manifest_sha256=manifest_sha256,
    )
    first_output = tmp_path / "first-random-name.pt"
    second_output = tmp_path / "second-random-name.pt"

    assert materializer.write_runtime_plan(plan, first_output) == "created"
    assert materializer.write_runtime_plan(plan, second_output) == "created"

    assert materializer.sha256_file(first_output) == materializer.sha256_file(
        second_output
    )


def test_valid_existing_runtime_plan_is_reused_without_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    placement = materializer.load_frozen_placement(MANIFEST)
    plan = materializer.build_runtime_plan(
        placement,
        manifest_path=MANIFEST,
        source_manifest_sha256=materializer.sha256_file(MANIFEST),
    )
    output_path = tmp_path / "runtime-plan.pt"
    assert materializer.write_runtime_plan(plan, output_path) == "created"
    original_sha256 = materializer.sha256_file(output_path)
    original_inode = output_path.stat().st_ino

    def unexpected_save(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an equivalent immutable plan must not be rewritten")

    monkeypatch.setattr(torch, "save", unexpected_save)

    assert materializer.write_runtime_plan(plan, output_path) == "reused"
    assert materializer.sha256_file(output_path) == original_sha256
    assert output_path.stat().st_ino == original_inode


def test_mismatched_existing_runtime_plan_is_atomically_replaced(
    tmp_path: Path,
) -> None:
    placement = materializer.load_frozen_placement(MANIFEST)
    plan = materializer.build_runtime_plan(
        placement,
        manifest_path=MANIFEST,
        source_manifest_sha256=materializer.sha256_file(MANIFEST),
    )
    output_path = tmp_path / "runtime-plan.pt"
    mismatched_plan = dict(plan)
    mismatched_plan["frozen_placement_name"] = "stale-placement"
    with output_path.open("wb") as output_file:
        torch.save(mismatched_plan, output_file)

    assert materializer.write_runtime_plan(plan, output_path) == "replaced"

    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    assert loaded["frozen_placement_name"] == placement.name


def test_runtime_plan_output_symlink_is_rejected(tmp_path: Path) -> None:
    placement = materializer.load_frozen_placement(MANIFEST)
    plan = materializer.build_runtime_plan(
        placement,
        manifest_path=MANIFEST,
        source_manifest_sha256=materializer.sha256_file(MANIFEST),
    )
    symlink_target = tmp_path / "target.pt"
    symlink_target.write_bytes(b"not a plan")
    output_path = tmp_path / "runtime-plan.pt"
    output_path.symlink_to(symlink_target)

    with pytest.raises(ValueError, match="symlink plan"):
        materializer.write_runtime_plan(plan, output_path)

    assert symlink_target.read_bytes() == b"not a plan"


def test_manifest_mutation_during_load_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "frozen-plan.json"
    manifest_path.write_bytes(MANIFEST.read_bytes())
    output_path = tmp_path / "runtime-plan.pt"
    original_loader = materializer.load_frozen_placement

    def mutating_loader(path: Path) -> materializer.FrozenPlacement:
        placement = original_loader(path)
        path.write_bytes(path.read_bytes() + b" ")
        return placement

    monkeypatch.setattr(materializer, "load_frozen_placement", mutating_loader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_dsv4_frozen_hybrid_plan",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="changed while loading"):
        materializer.main()

    assert not output_path.exists()


def test_frozen_plan_rejects_a_gpu_expert_outside_rank_ownership(
    tmp_path: Path,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rank0_bitset = int(manifest["rank0_ownership_hex"][0], 16)
    foreign_expert = next(
        expert_id for expert_id in range(256) if not (rank0_bitset >> expert_id) & 1
    )
    manifest["gpu_expert_ids_by_rank"][0][0][0] = foreign_expert
    manifest["gpu_expert_ids_by_rank"][0][0].sort()
    modified_manifest = tmp_path / "wrong-owner.json"
    modified_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes rank ownership"):
        materializer.load_frozen_placement(modified_manifest)


def test_frozen_plan_rejects_semantic_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["placement_semantics_sha256"] = "0" * 64
    modified_manifest = tmp_path / "wrong-sha.json"
    modified_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        materializer.load_frozen_placement(modified_manifest)
