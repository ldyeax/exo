from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.refresh_dsv4_candidate_campaign_manifest import refreshed_manifest
from scripts.run_dsv4_flash_candidate_campaign import CampaignError, sha256_file


def test_refresh_hashes_offline_plan_and_receipt(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    baseline_plan = tmp_path / "baseline.pt"
    offline_plan = tmp_path / "offline.pt"
    offline_receipt = tmp_path / "offline.receipt.json"
    oscar_calibration = tmp_path / "oscar.pt"
    oscar_fingerprint = tmp_path / "checkpoint-fingerprint.json"
    oscar_admission = tmp_path / "admission.json"
    for path, content in (
        (launcher, "#!/bin/sh\n"),
        (baseline_plan, "baseline"),
        (offline_plan, "offline-plan"),
        (offline_receipt, "{}\n"),
        (oscar_calibration, "oscar"),
        (oscar_fingerprint, "{}\n"),
        (oscar_admission, "{}\n"),
    ):
        path.write_text(content, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "launcher": launcher.name,
                "launcher_sha256": "0" * 64,
                "fixed_environment": {},
                "source_artifacts": [],
                "oscar_contract": {
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                    "calibration_artifact": {
                        "path": oscar_calibration.name,
                        "sha256": "0" * 64,
                    },
                    "checkpoint_fingerprint": {
                        "path": oscar_fingerprint.name,
                        "sha256": "0" * 64,
                    },
                    "admission_receipt": {
                        "path": oscar_admission.name,
                        "sha256": "0" * 64,
                    },
                },
                "offline_variants": [
                    {
                        "id": "offline",
                        "expected_plan": offline_plan.name,
                        "expected_plan_sha256": "0" * 64,
                        "receipt": offline_receipt.name,
                        "receipt_sha256": "0" * 64,
                    }
                ],
                "baseline": {
                    "expected_plan": baseline_plan.name,
                    "expected_plan_sha256": "0" * 64,
                    "plan_materialized_at_launch": False,
                    "artifacts": [],
                },
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    refreshed = refreshed_manifest(manifest_path)
    offline = cast(list[dict[str, object]], refreshed["offline_variants"])[0]

    assert offline["expected_plan_sha256"] == sha256_file(offline_plan)
    assert offline["receipt_sha256"] == sha256_file(offline_receipt)
    assert refreshed["launcher_sha256"] == sha256_file(launcher)
    contract = cast(dict[str, object], refreshed["oscar_contract"])
    calibration = cast(dict[str, object], contract["calibration_artifact"])
    fingerprint = cast(dict[str, object], contract["checkpoint_fingerprint"])
    admission = cast(dict[str, object], contract["admission_receipt"])
    assert calibration["sha256"] == sha256_file(oscar_calibration)
    assert fingerprint["sha256"] == sha256_file(oscar_fingerprint)
    assert admission["sha256"] == sha256_file(oscar_admission)

    override_calibration = tmp_path / "override-oscar.pt"
    override_fingerprint = tmp_path / "override-fingerprint.json"
    override_admission = tmp_path / "override-admission.json"
    override_calibration.write_text("new-oscar", encoding="utf-8")
    override_fingerprint.write_text('{"new":true}\n', encoding="utf-8")
    override_admission.write_text('{"new":true}\n', encoding="utf-8")
    overridden = refreshed_manifest(
        manifest_path,
        oscar_calibration_path=override_calibration.resolve(),
        oscar_checkpoint_fingerprint_path=override_fingerprint.resolve(),
        oscar_admission_receipt_path=override_admission.resolve(),
    )
    overridden_contract = cast(dict[str, object], overridden["oscar_contract"])
    overridden_calibration = cast(
        dict[str, object], overridden_contract["calibration_artifact"]
    )
    assert overridden_calibration == {
        "path": str(override_calibration.resolve()),
        "sha256": sha256_file(override_calibration),
    }
    environment = cast(dict[str, object], overridden["fixed_environment"])
    assert environment["DSV4_OSCAR_CALIBRATION_PATH"] == str(
        override_calibration.resolve()
    )
    assert environment["DSV4_OSCAR_CHECKPOINT_FINGERPRINT_PATH"] == str(
        override_fingerprint.resolve()
    )
    assert environment["DSV4_OSCAR_ADMISSION_RECEIPT_PATH"] == str(
        override_admission.resolve()
    )

    with pytest.raises(CampaignError, match="all three OSCAR path overrides"):
        refreshed_manifest(
            manifest_path,
            oscar_calibration_path=override_calibration.resolve(),
        )


def test_refresh_fails_closed_when_oscar_contract_file_is_missing(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "launcher.sh"
    plan = tmp_path / "plan.pt"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    plan.write_text("plan", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "launcher": launcher.name,
                "source_artifacts": [],
                "oscar_contract": {
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                    "calibration_artifact": {
                        "path": "missing-oscar.pt",
                        "sha256": "0" * 64,
                    },
                    "checkpoint_fingerprint": {
                        "path": "missing-fingerprint.json",
                        "sha256": "0" * 64,
                    },
                    "admission_receipt": {
                        "path": "missing-admission.json",
                        "sha256": "0" * 64,
                    },
                },
                "offline_variants": [],
                "baseline": {
                    "expected_plan": plan.name,
                    "artifacts": [],
                },
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="OSCAR contract file is missing"):
        refreshed_manifest(manifest_path)
