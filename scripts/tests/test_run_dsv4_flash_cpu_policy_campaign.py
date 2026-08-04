from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Self, cast

import pytest

from scripts import run_dsv4_flash_cpu_policy_campaign as wrapper


class _FakePolicySession:
    def __init__(self, *, restoration_verified: bool = True) -> None:
        self.restoration_verified = restoration_verified
        self.evidence: dict[str, object] = {
            "lifecycle": "new",
            "application_verified": False,
            "restoration_verified": False,
            "failures": [],
            "policies": [{"name": "policy0"}],
            "serialization": {
                "journal_published": False,
                "journal_removed": False,
                "lock_released": False,
                "lock_acquired": False,
            },
            "snapshots": {},
        }

    def __enter__(self) -> Self:
        self.evidence["lifecycle"] = "active"
        self.evidence["application_verified"] = True
        serialization = cast(dict[str, object], self.evidence["serialization"])
        serialization["journal_published"] = True
        serialization["lock_acquired"] = True
        snapshots = cast(dict[str, object], self.evidence["snapshots"])
        snapshots.update(
            {
                "before": {},
                "active": {},
                "performance_after": {},
            }
        )
        return self

    def __exit__(self, *_args: object) -> None:
        self.evidence["lifecycle"] = "restored"
        self.evidence["restoration_verified"] = self.restoration_verified
        serialization = cast(dict[str, object], self.evidence["serialization"])
        serialization.update(
            {
                "journal_published": False,
                "journal_removed": True,
                "lock_released": True,
                "lock_acquired": False,
            }
        )
        snapshots = cast(dict[str, object], self.evidence["snapshots"])
        snapshots["restored"] = {}


class _Completed:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _invocation(tmp_path: Path) -> wrapper.CampaignInvocation:
    python_path = tmp_path / "python"
    python_path.write_text("python", encoding="utf-8")
    os.chmod(python_path, 0o700)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return wrapper.CampaignInvocation(
        python=python_path,
        manifest=manifest_path,
        work_directory=tmp_path / "work",
        execute_next=True,
        candidate=None,
        stage=None,
    )


def test_policy_campaign_publishes_only_after_verified_restore(
    tmp_path: Path,
) -> None:
    session = _FakePolicySession()
    receipt_path = tmp_path / "receipt.json"
    observed_lifecycles: list[object] = []

    payload = wrapper.run_policy_campaign(
        _invocation(tmp_path),
        receipt_path,
        policy_factory=lambda: session,
        child_runner=lambda _command: (
            observed_lifecycles.append(session.evidence["lifecycle"])
            or _Completed(0)
        ),
    )

    assert observed_lifecycles == ["active"]
    assert payload["accepted"] is True
    assert session.evidence["lifecycle"] == "restored"
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["cpu_performance_policy"]["evidence"]["lifecycle"] == "restored"
    assert receipt_path.stat().st_mode & 0o777 == 0o600


def test_policy_campaign_records_child_failure_after_restore(tmp_path: Path) -> None:
    session = _FakePolicySession()
    receipt_path = tmp_path / "failed.json"

    with pytest.raises(wrapper.CpuPolicyCampaignError, match="restoration evidence"):
        wrapper.run_policy_campaign(
            _invocation(tmp_path),
            receipt_path,
            policy_factory=lambda: session,
            child_runner=lambda _command: _Completed(17),
        )

    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["accepted"] is False
    assert written["child_returncode"] == 17
    assert written["cpu_performance_policy"]["evidence"]["lifecycle"] == "restored"


def test_policy_campaign_fails_closed_on_unverified_restore(tmp_path: Path) -> None:
    session = _FakePolicySession(restoration_verified=False)
    receipt_path = tmp_path / "unverified.json"

    with pytest.raises(wrapper.CpuPolicyCampaignError, match="restoration evidence"):
        wrapper.run_policy_campaign(
            _invocation(tmp_path),
            receipt_path,
            policy_factory=lambda: session,
            child_runner=lambda _command: _Completed(0),
        )

    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["accepted"] is False
    assert written["cpu_performance_policy"]["evidence_sha256"] is None


def test_invocation_rejects_stage_without_explicit_candidate(tmp_path: Path) -> None:
    invocation = _invocation(tmp_path)
    invalid = wrapper.CampaignInvocation(
        python=invocation.python,
        manifest=invocation.manifest,
        work_directory=invocation.work_directory,
        execute_next=True,
        candidate=None,
        stage="screen",
    )

    with pytest.raises(wrapper.CpuPolicyCampaignError, match="stage"):
        invalid.command()
