from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

import scripts.sglang_kt_resident_cluster as resident


@dataclass(frozen=True, slots=True)
class FakeSpec:
    pipeline_rank: int
    node_id: str
    service_endpoint: str


@dataclass(frozen=True, slots=True)
class FakeOwnedStage:
    rank: int
    host_name: str
    pid: int
    process_group_id: int
    start_time_ticks: int
    owner_token: str
    ownership_namespace: str
    remote: bool


@dataclass(frozen=True, slots=True)
class FakeRunningStage:
    owned: FakeOwnedStage


@dataclass(frozen=True, slots=True)
class FakeCleanupReceipt:
    ownership_verified: bool
    terminated: bool
    forced: bool
    error: str | None


def _build_ownership(
    spec: FakeSpec,
    running: FakeRunningStage,
) -> resident.ResidentStageOwnership:
    return resident.ResidentStageOwnership.from_owned_stage(
        running.owned,
        node_id=spec.node_id,
        service_endpoint=spec.service_endpoint,
    )


def _cleanup_evidence(
    running: FakeRunningStage,
    cleanup: FakeCleanupReceipt,
) -> resident.ResidentStageCleanupEvidence:
    return resident.cleanup_evidence_from_receipt(running, cleanup)


def _make_lifecycle(
    tmp_path: Path,
    *,
    stopped_ranks: list[int],
    fail_start_rank: int | None = None,
    incomplete_cleanup_rank: int | None = None,
) -> resident.ResidentClusterLifecycle[
    FakeSpec,
    FakeRunningStage,
    FakeCleanupReceipt,
]:
    specs = (
        FakeSpec(0, "dwagon", "192.168.40.24:62510"),
        FakeSpec(1, "dwagon", "192.168.40.24:62511"),
        FakeSpec(2, "fwuff", "192.168.40.93:62512"),
    )

    def start_stage(spec: FakeSpec, owner_token: str) -> FakeRunningStage:
        if spec.pipeline_rank == fail_start_rank:
            raise RuntimeError(f"rank {spec.pipeline_rank} start failed")
        pid = 10_000 + spec.pipeline_rank
        return FakeRunningStage(
            FakeOwnedStage(
                rank=spec.pipeline_rank,
                host_name=spec.node_id,
                pid=pid,
                process_group_id=pid,
                start_time_ticks=20_000 + spec.pipeline_rank,
                owner_token=owner_token,
                ownership_namespace=str(62510 + spec.pipeline_rank),
                remote=spec.node_id == "fwuff",
            )
        )

    def stop_stage(running: FakeRunningStage) -> FakeCleanupReceipt:
        stopped_ranks.append(running.owned.rank)
        complete = running.owned.rank != incomplete_cleanup_rank
        return FakeCleanupReceipt(
            ownership_verified=complete,
            terminated=complete,
            forced=False,
            error=None if complete else "fixture cleanup failure",
        )

    return resident.ResidentClusterLifecycle(
        run_id="resident-fixture",
        receipt_path=(tmp_path / "resident-ownership.json").resolve(),
        launch_contract={
            "partition": [26, 28, 24],
            "model": "/mnt/sanic/glm52",
        },
        specs=specs,
        start_stage=start_stage,
        stop_stage=stop_stage,
        build_stage_ownership=_build_ownership,
        build_cleanup_evidence=_cleanup_evidence,
    )


def _successful_verifier(
    stage: resident.ResidentStageOwnership,
) -> resident.ResidentStageVerification:
    return resident.ResidentStageVerification(
        pipeline_rank=stage.pipeline_rank,
        ownership_verified=True,
        alive=True,
        process_group_members=(stage.pid,),
    )


def test_lifecycle_publishes_private_receipt_and_allows_exact_attachment(
    tmp_path: Path,
) -> None:
    stopped_ranks: list[int] = []
    lifecycle = _make_lifecycle(tmp_path, stopped_ranks=stopped_ranks)

    lifecycle.start()
    receipt_path = lifecycle.receipt_path
    attachment = resident.verify_resident_cluster_attachment(
        receipt_path,
        expected_launch_contract=lifecycle.launch_contract,
        stage_verifier=_successful_verifier,
    )

    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert attachment.receipt.status == "running"
    assert tuple(stage.pipeline_rank for stage in attachment.receipt.stages) == (
        0,
        1,
        2,
    )
    assert len(attachment.receipt.owner_token) >= 32

    cleanup = lifecycle.stop()

    assert stopped_ranks == [2, 1, 0]
    assert all(item.terminated for item in cleanup)
    assert not receipt_path.exists()
    terminal = receipt_path.with_name(f"{receipt_path.name}.stopped.json")
    assert terminal.is_file()
    terminal_receipt = resident.load_resident_cluster_ownership_receipt(terminal)
    assert terminal_receipt.status == "stopped"


def test_attachment_rejects_a_different_launch_contract(tmp_path: Path) -> None:
    lifecycle = _make_lifecycle(tmp_path, stopped_ranks=[])
    lifecycle.start()
    try:
        with pytest.raises(
            resident.ResidentClusterAttachmentError,
            match="launch contract does not match",
        ):
            resident.verify_resident_cluster_attachment(
                lifecycle.receipt_path,
                expected_launch_contract={
                    "partition": [30, 28, 20],
                    "model": "/mnt/sanic/glm52",
                },
                stage_verifier=_successful_verifier,
            )
    finally:
        lifecycle.stop()


def test_attachment_rejects_insecure_receipt_permissions(tmp_path: Path) -> None:
    lifecycle = _make_lifecycle(tmp_path, stopped_ranks=[])
    lifecycle.start()
    try:
        os.chmod(lifecycle.receipt_path, 0o644)
        with pytest.raises(
            resident.ResidentClusterAttachmentError,
            match="mode 0600",
        ):
            resident.load_resident_cluster_ownership_receipt(lifecycle.receipt_path)
    finally:
        os.chmod(lifecycle.receipt_path, 0o600)
        lifecycle.stop()


def test_attachment_requires_an_explicit_receipt(tmp_path: Path) -> None:
    with pytest.raises(
        resident.ResidentClusterAttachmentError,
        match="cannot open resident receipt",
    ):
        resident.verify_resident_cluster_attachment(
            (tmp_path / "not-published.json").resolve(),
            expected_launch_contract={"partition": [26, 28, 24]},
            stage_verifier=_successful_verifier,
        )


def test_partial_start_cleans_only_started_rank_in_reverse(tmp_path: Path) -> None:
    stopped_ranks: list[int] = []
    lifecycle = _make_lifecycle(
        tmp_path,
        stopped_ranks=stopped_ranks,
        fail_start_rank=1,
    )

    with pytest.raises(RuntimeError, match="rank 1 start failed"):
        lifecycle.start()

    assert stopped_ranks == [0]
    assert not lifecycle.receipt_path.exists()


def test_cleanup_failure_retains_active_capability_receipt(
    tmp_path: Path,
) -> None:
    lifecycle = _make_lifecycle(
        tmp_path,
        stopped_ranks=[],
        incomplete_cleanup_rank=1,
    )
    lifecycle.start()

    with pytest.raises(
        resident.ResidentClusterCleanupError,
        match="cleanup is incomplete",
    ):
        lifecycle.stop()

    assert lifecycle.receipt_path.is_file()
    failed_path = lifecycle.receipt_path.with_name(
        f"{lifecycle.receipt_path.name}.cleanup_failed.json"
    )
    assert failed_path.is_file()
    failed = resident.load_resident_cluster_ownership_receipt(failed_path)
    assert failed.status == "cleanup_failed"
    assert any(not item.terminated for item in failed.cleanup)
