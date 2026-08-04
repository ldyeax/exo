#!/usr/bin/env python3
"""Refresh hash fields in the local DSV4 candidate-campaign manifest.

Run this only after the launcher, SGLang, KTransformers, plans, and receipts are
frozen for a campaign. Changing any hash intentionally invalidates old campaign
results because the controller binds every result to the whole manifest digest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

try:
    from scripts.run_dsv4_flash_candidate_campaign import (
        CampaignError,
        sha256_file,
        sha256_source_tree,
    )
except ModuleNotFoundError:
    from run_dsv4_flash_candidate_campaign import (
        CampaignError,
        sha256_file,
        sha256_source_tree,
    )


def _root_for_manifest(path: Path) -> Path:
    return path.resolve().parents[2] if path.parent.name == "data" else path.parent


def _resolved_path(value: object, *, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignError("manifest hash target has no path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _refresh_artifacts(value: object, *, root: Path) -> None:
    if not isinstance(value, list):
        raise CampaignError("manifest artifact collection must be a list")
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            raise CampaignError("manifest artifact entry must be an object")
        artifact = cast(dict[str, object], raw)
        path = _resolved_path(artifact.get("path"), root=root)
        kind = artifact.get("kind", "file")
        artifact["sha256"] = (
            sha256_source_tree(path) if kind == "source_tree" else sha256_file(path)
        )


def _refresh_offline_variants(value: object, *, root: Path) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise CampaignError("manifest offline_variants must be a list")
    for raw in cast(list[object], value):
        if not isinstance(raw, dict):
            raise CampaignError("manifest offline variant must be an object")
        variant = cast(dict[str, object], raw)
        plan = _resolved_path(variant.get("expected_plan"), root=root)
        receipt = _resolved_path(variant.get("receipt"), root=root)
        if plan.is_symlink() or not plan.is_file():
            raise CampaignError(f"offline variant plan is not a regular file: {plan}")
        if receipt.is_symlink() or not receipt.is_file():
            raise CampaignError(
                f"offline variant receipt is not a regular file: {receipt}"
            )
        variant["expected_plan_sha256"] = sha256_file(plan)
        variant["receipt_sha256"] = sha256_file(receipt)


def _refresh_oscar_contract(value: object, *, root: Path) -> None:
    if not isinstance(value, dict):
        raise CampaignError("manifest oscar_contract must be an object")
    contract = cast(dict[str, object], value)
    for field in (
        "calibration_artifact",
        "checkpoint_fingerprint",
        "admission_receipt",
    ):
        raw_artifact = contract.get(field)
        if not isinstance(raw_artifact, dict):
            raise CampaignError(f"manifest oscar_contract.{field} must be an object")
        artifact = cast(dict[str, object], raw_artifact)
        path = _resolved_path(artifact.get("path"), root=root)
        if path.is_symlink() or not path.is_file():
            raise CampaignError(f"OSCAR contract file is missing: {path}")
        artifact["sha256"] = sha256_file(path)


def _apply_oscar_path_overrides(
    manifest: dict[str, object],
    *,
    calibration_path: Path | None,
    checkpoint_fingerprint_path: Path | None,
    admission_receipt_path: Path | None,
) -> None:
    raw_paths = (
        calibration_path,
        checkpoint_fingerprint_path,
        admission_receipt_path,
    )
    if all(path is None for path in raw_paths):
        return
    if any(path is None for path in raw_paths):
        raise CampaignError("all three OSCAR path overrides must be supplied together")
    paths = cast(tuple[Path, Path, Path], raw_paths)
    for path in paths:
        if not path.is_absolute():
            raise CampaignError("OSCAR path overrides must be absolute")
        if path.is_symlink() or not path.is_file():
            raise CampaignError(f"OSCAR path override is not a regular file: {path}")

    raw_contract = manifest.get("oscar_contract")
    if not isinstance(raw_contract, dict):
        raise CampaignError("manifest oscar_contract must be an object")
    contract = cast(dict[str, object], raw_contract)
    fields = (
        "calibration_artifact",
        "checkpoint_fingerprint",
        "admission_receipt",
    )
    for field, path in zip(fields, paths, strict=True):
        raw_artifact = contract.get(field)
        if not isinstance(raw_artifact, dict):
            raise CampaignError(f"manifest oscar_contract.{field} must be an object")
        cast(dict[str, object], raw_artifact)["path"] = str(path)

    raw_environment = manifest.get("fixed_environment")
    if not isinstance(raw_environment, dict):
        raise CampaignError("manifest fixed_environment must be an object")
    environment = cast(dict[str, object], raw_environment)
    environment["DSV4_OSCAR_CALIBRATION_PATH"] = str(paths[0])
    environment["DSV4_OSCAR_CHECKPOINT_FINGERPRINT_PATH"] = str(paths[1])
    environment["DSV4_OSCAR_ADMISSION_RECEIPT_PATH"] = str(paths[2])


def refreshed_manifest(
    path: Path,
    *,
    oscar_calibration_path: Path | None = None,
    oscar_checkpoint_fingerprint_path: Path | None = None,
    oscar_admission_receipt_path: Path | None = None,
) -> dict[str, object]:
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise CampaignError("campaign manifest must be an object")
    manifest = cast(dict[str, object], loaded)
    root = _root_for_manifest(path)
    _apply_oscar_path_overrides(
        manifest,
        calibration_path=oscar_calibration_path,
        checkpoint_fingerprint_path=oscar_checkpoint_fingerprint_path,
        admission_receipt_path=oscar_admission_receipt_path,
    )
    launcher = _resolved_path(manifest.get("launcher"), root=root)
    manifest["launcher_sha256"] = sha256_file(launcher)
    _refresh_oscar_contract(manifest.get("oscar_contract"), root=root)
    _refresh_artifacts(manifest.get("source_artifacts"), root=root)
    _refresh_offline_variants(manifest.get("offline_variants"), root=root)
    raw_candidates = [manifest.get("baseline")]
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise CampaignError("campaign candidates must be a list")
    raw_candidates.extend(cast(list[object], candidates))
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise CampaignError("campaign candidate must be an object")
        candidate = cast(dict[str, object], raw_candidate)
        _refresh_artifacts(candidate.get("artifacts", []), root=root)
        plan = _resolved_path(candidate.get("expected_plan"), root=root)
        if plan.is_file():
            candidate["expected_plan_sha256"] = sha256_file(plan)
        elif candidate.get("plan_materialized_at_launch") is not True:
            raise CampaignError(f"prebuilt candidate plan is missing: {plan}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--oscar-calibration-path", type=Path)
    parser.add_argument("--oscar-checkpoint-fingerprint-path", type=Path)
    parser.add_argument("--oscar-admission-receipt-path", type=Path)
    arguments = parser.parse_args()
    refreshed = refreshed_manifest(
        arguments.manifest,
        oscar_calibration_path=arguments.oscar_calibration_path,
        oscar_checkpoint_fingerprint_path=(arguments.oscar_checkpoint_fingerprint_path),
        oscar_admission_receipt_path=arguments.oscar_admission_receipt_path,
    )
    serialized = json.dumps(refreshed, indent=2, sort_keys=True) + "\n"
    current = arguments.manifest.read_text(encoding="utf-8")
    changed = serialized != current
    if arguments.write and changed:
        temporary = arguments.manifest.with_suffix(
            arguments.manifest.suffix + f".{os.getpid()}.tmp"
        )
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, arguments.manifest)
    print(json.dumps({"changed": changed, "written": arguments.write and changed}))
    return int(changed and not arguments.write)


if __name__ == "__main__":
    raise SystemExit(main())
