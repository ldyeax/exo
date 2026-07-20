from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from scripts.prepare_sglang_kt_source import (
    MailPatch,
    SglangKtSourcePlan,
    SourcePreparationError,
    prepare_sglang_kt_source,
)

GIT_NAME = "Exo Source Test"
GIT_EMAIL = "exo-source-test@example.invalid"


@dataclass(frozen=True)
class SourceFixture:
    ktransformers_source: Path
    sglang_source: Path
    other_submodule_source: Path
    plan: SglangKtSourcePlan


def test_default_plan_matches_admitted_launch_revisions() -> None:
    plan = SglangKtSourcePlan.exo_default()

    assert plan.ktransformers_result_revision == (GLM_4_7_FLASH_KTRANSFORMERS_REVISION)
    assert plan.sglang_result_revision == GLM_4_7_FLASH_SGLANG_REVISION


def test_default_plan_declares_exact_hashed_followup_stack() -> None:
    plan = SglangKtSourcePlan.exo_default()

    assert tuple(
        (patch.path.name, patch.base_revision, patch.result_revision)
        for patch in plan.sglang_followup_patches
    ) == (
        (
            "0003-fix-shard-OLMoE-QK-RMSNorm-across-TP-ranks.patch",
            "3721d710102456b6bf849122e781129dc3f7d9c6",
            "da64717bb2e87f7ebc6e69768ba575c18454ab3c",
        ),
        (
            "0004-fix-allocate-GLM-Flash-LM-head-on-final-PP-rank.patch",
            "da64717bb2e87f7ebc6e69768ba575c18454ab3c",
            "7fea582043df06ebdde549ee3de602a3d11b96c6",
        ),
    )
    for patch in plan.sglang_followup_patches:
        assert hashlib.sha256(patch.path.read_bytes()).hexdigest() == patch.sha256


def run_git(
    repository: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        env=None if environment is None else {**os.environ, **environment},
    )
    return result.stdout.strip()


def initialize_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(
        ("git", "init", "--initial-branch=main", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    run_git(path, "config", "user.name", GIT_NAME)
    run_git(path, "config", "user.email", GIT_EMAIL)


def commit_all(repository: Path, message: str, timestamp: str) -> str:
    run_git(repository, "add", "--all")
    environment = {
        "GIT_AUTHOR_DATE": timestamp,
        "GIT_COMMITTER_DATE": timestamp,
    }
    run_git(repository, "commit", "-m", message, environment=environment)
    return run_git(repository, "rev-parse", "HEAD")


def write_mail_patch(repository: Path, revision: str, destination: Path) -> str:
    patch = run_git(repository, "format-patch", "-1", "--stdout", revision)
    destination.write_text(patch + "\n")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def make_source_fixture(tmp_path: Path) -> SourceFixture:
    sglang_remote = tmp_path / "sglang-remote"
    initialize_repository(sglang_remote)
    module_path = sglang_remote / "module.py"
    module_path.write_text("WRAPPER_REGISTERED = False\n")
    sglang_base_revision = commit_all(
        sglang_remote,
        "base: SGLang",
        "2026-01-01T00:00:00+00:00",
    )
    module_path.write_text("WRAPPER_REGISTERED = True\n")
    sglang_result_revision = commit_all(
        sglang_remote,
        "feat: register wrapper",
        "2026-01-01T00:01:00+00:00",
    )
    sglang_patch_path = tmp_path / "0001-sglang.patch"
    sglang_patch_sha256 = write_mail_patch(
        sglang_remote, sglang_result_revision, sglang_patch_path
    )
    run_git(sglang_remote, "checkout", "--detach", sglang_base_revision)

    other_remote = tmp_path / "other-remote"
    initialize_repository(other_remote)
    (other_remote / "dependency.py").write_text("CLEAN = True\n")
    commit_all(
        other_remote,
        "base: other dependency",
        "2026-01-01T00:01:30+00:00",
    )

    ktransformers_source = tmp_path / "ktransformers"
    initialize_repository(ktransformers_source)
    (ktransformers_source / "README.md").write_text("test KTransformers source\n")
    run_git(
        ktransformers_source,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(sglang_remote),
        "third_party/sglang",
    )
    run_git(
        ktransformers_source,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(other_remote),
        "third_party/other",
    )
    ktransformers_base_revision = commit_all(
        ktransformers_source,
        "base: KTransformers",
        "2026-01-01T00:02:00+00:00",
    )

    sglang_source = ktransformers_source / "third_party/sglang"
    run_git(sglang_source, "checkout", "--detach", sglang_result_revision)
    ktransformers_result_revision = commit_all(
        ktransformers_source,
        "build: pin wrapper registration",
        "2026-01-01T00:03:00+00:00",
    )
    ktransformers_patch_path = tmp_path / "0002-ktransformers.patch"
    ktransformers_patch_sha256 = write_mail_patch(
        ktransformers_source,
        ktransformers_result_revision,
        ktransformers_patch_path,
    )

    run_git(ktransformers_source, "reset", "--hard", ktransformers_base_revision)
    run_git(sglang_source, "checkout", "--detach", sglang_base_revision)
    plan = SglangKtSourcePlan(
        ktransformers_base_revision=ktransformers_base_revision,
        ktransformers_result_revision=ktransformers_result_revision,
        sglang_base_revision=sglang_base_revision,
        sglang_result_revision=sglang_result_revision,
        sglang_submodule_path=Path("third_party/sglang"),
        sglang_patch=MailPatch(
            path=sglang_patch_path,
            sha256=sglang_patch_sha256,
            committer_name=GIT_NAME,
            committer_email=GIT_EMAIL,
        ),
        ktransformers_patch=MailPatch(
            path=ktransformers_patch_path,
            sha256=ktransformers_patch_sha256,
            committer_name=GIT_NAME,
            committer_email=GIT_EMAIL,
        ),
    )
    return SourceFixture(
        ktransformers_source=ktransformers_source,
        sglang_source=sglang_source,
        other_submodule_source=ktransformers_source / "third_party/other",
        plan=plan,
    )


def test_prepares_exact_revisions_from_clean_bases_and_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)

    receipt = prepare_sglang_kt_source(
        fixture.ktransformers_source,
        fixture.plan,
    )
    second_receipt = prepare_sglang_kt_source(
        fixture.ktransformers_source,
        fixture.plan,
    )

    assert receipt == second_receipt
    assert receipt.ktransformers_revision == fixture.plan.ktransformers_result_revision
    assert receipt.sglang_revision == fixture.plan.sglang_result_revision
    assert run_git(fixture.ktransformers_source, "status", "--porcelain=v1") == ""
    assert run_git(fixture.sglang_source, "status", "--porcelain=v1") == ""
    assert (
        run_git(fixture.ktransformers_source, "rev-parse", "HEAD")
        == fixture.plan.ktransformers_result_revision
    )
    assert (
        run_git(fixture.sglang_source, "rev-parse", "HEAD")
        == fixture.plan.sglang_result_revision
    )


def test_applies_followup_patch_stack_after_integrated_gitlink(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)
    gitlink_revision = fixture.plan.sglang_result_revision
    run_git(fixture.sglang_source, "config", "user.name", GIT_NAME)
    run_git(fixture.sglang_source, "config", "user.email", GIT_EMAIL)

    module_path = fixture.sglang_source / "module.py"
    module_path.write_text("WRAPPER_REGISTERED = True\nTP_QK_RMSNORM_SHARDED = True\n")
    tp_qk_revision = commit_all(
        fixture.sglang_source,
        "fix: shard QK RMSNorm",
        "2026-01-01T00:04:00+00:00",
    )
    followup_patch_path = tmp_path / "0003-sglang-followup.patch"
    followup_patch_sha256 = write_mail_patch(
        fixture.sglang_source,
        tp_qk_revision,
        followup_patch_path,
    )

    module_path.write_text(
        "WRAPPER_REGISTERED = True\n"
        "TP_QK_RMSNORM_SHARDED = True\n"
        "LM_HEAD_ON_FINAL_PP_RANK = True\n"
    )
    final_revision = commit_all(
        fixture.sglang_source,
        "fix: allocate LM head on final PP rank",
        "2026-01-01T00:05:00+00:00",
    )
    final_patch_path = tmp_path / "0004-sglang-followup.patch"
    final_patch_sha256 = write_mail_patch(
        fixture.sglang_source,
        final_revision,
        final_patch_path,
    )
    run_git(fixture.sglang_source, "checkout", "--detach", gitlink_revision)
    run_git(fixture.sglang_source, "reflog", "expire", "--expire=now", "--all")
    run_git(fixture.sglang_source, "gc", "--prune=now")

    plan = replace(
        fixture.plan,
        sglang_result_revision=final_revision,
        sglang_followup_patches=(
            MailPatch(
                path=followup_patch_path,
                sha256=followup_patch_sha256,
                committer_name=GIT_NAME,
                committer_email=GIT_EMAIL,
                base_revision=gitlink_revision,
                result_revision=tp_qk_revision,
            ),
            MailPatch(
                path=final_patch_path,
                sha256=final_patch_sha256,
                committer_name=GIT_NAME,
                committer_email=GIT_EMAIL,
                base_revision=tp_qk_revision,
                result_revision=final_revision,
            ),
        ),
        sglang_gitlink_revision=gitlink_revision,
    )

    receipt = prepare_sglang_kt_source(fixture.ktransformers_source, plan)
    second_receipt = prepare_sglang_kt_source(fixture.ktransformers_source, plan)

    assert receipt == second_receipt
    assert receipt.sglang_revision == final_revision
    assert run_git(fixture.sglang_source, "rev-parse", "HEAD") == final_revision
    assert run_git(fixture.sglang_source, "status", "--porcelain=v1") == ""
    assert (
        run_git(fixture.ktransformers_source, "status", "--porcelain=v1")
        == "M third_party/sglang"
    )
    assert tuple(Path(patch["path"]).name for patch in receipt.patches) == (
        fixture.plan.sglang_patch.path.name,
        followup_patch_path.name,
        final_patch_path.name,
        fixture.plan.ktransformers_patch.path.name,
    )


def test_reconstructs_uninitialized_submodule_from_integrated_parent(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)
    run_git(
        fixture.ktransformers_source,
        "submodule",
        "deinit",
        "--force",
        str(fixture.plan.sglang_submodule_path),
    )
    shutil.rmtree(fixture.sglang_source)
    fixture.sglang_source.mkdir(parents=True)

    receipt = prepare_sglang_kt_source(
        fixture.ktransformers_source,
        fixture.plan,
    )

    assert receipt.sglang_revision == fixture.plan.sglang_result_revision
    assert run_git(fixture.ktransformers_source, "status", "--porcelain=v1") == ""
    assert run_git(fixture.sglang_source, "status", "--porcelain=v1") == ""


def test_reconstructs_missing_result_from_initialized_base_submodule(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)
    run_git(
        fixture.sglang_source, "checkout", "--detach", fixture.plan.sglang_base_revision
    )
    run_git(
        fixture.sglang_source,
        "update-ref",
        "-d",
        "refs/remotes/origin/main",
    )
    run_git(fixture.sglang_source, "reflog", "expire", "--expire=now", "--all")
    run_git(fixture.sglang_source, "gc", "--prune=now")
    assert (
        subprocess.run(
            (
                "git",
                "-C",
                str(fixture.sglang_source),
                "cat-file",
                "-e",
                f"{fixture.plan.sglang_result_revision}^{{commit}}",
            ),
            check=False,
            capture_output=True,
        ).returncode
        != 0
    )

    receipt = prepare_sglang_kt_source(
        fixture.ktransformers_source,
        fixture.plan,
    )

    assert receipt.sglang_revision == fixture.plan.sglang_result_revision
    assert run_git(fixture.ktransformers_source, "status", "--porcelain=v1") == ""
    assert run_git(fixture.sglang_source, "status", "--porcelain=v1") == ""


def test_rejects_modified_integrated_parent_before_repairing_submodule(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)
    run_git(
        fixture.sglang_source, "checkout", "--detach", fixture.plan.sglang_base_revision
    )
    (fixture.ktransformers_source / "README.md").write_text("modified\n")

    with pytest.raises(
        SourcePreparationError,
        match="KTransformers integration excluding its SGLang submodule is not clean",
    ):
        prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)

    assert (
        run_git(fixture.sglang_source, "rev-parse", "HEAD")
        == fixture.plan.sglang_base_revision
    )


def test_rejects_other_dirty_submodule_before_repairing_sglang(
    tmp_path: Path,
) -> None:
    fixture = make_source_fixture(tmp_path)
    prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)
    run_git(
        fixture.sglang_source, "checkout", "--detach", fixture.plan.sglang_base_revision
    )
    (fixture.other_submodule_source / "dependency.py").write_text("CLEAN = False\n")

    with pytest.raises(
        SourcePreparationError,
        match="KTransformers integration excluding its SGLang submodule is not clean",
    ):
        prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)

    assert (
        run_git(fixture.sglang_source, "rev-parse", "HEAD")
        == fixture.plan.sglang_base_revision
    )


def test_rejects_modified_base_before_applying_patches(tmp_path: Path) -> None:
    fixture = make_source_fixture(tmp_path)
    (fixture.ktransformers_source / "README.md").write_text("modified\n")

    with pytest.raises(SourcePreparationError, match="KTransformers base is not clean"):
        prepare_sglang_kt_source(fixture.ktransformers_source, fixture.plan)

    assert (
        run_git(fixture.ktransformers_source, "rev-parse", "HEAD")
        == fixture.plan.ktransformers_base_revision
    )


@pytest.mark.parametrize("corrupt_sglang_patch", (True, False))
def test_rejects_patch_digest_mismatch_before_committing(
    tmp_path: Path, corrupt_sglang_patch: bool
) -> None:
    fixture = make_source_fixture(tmp_path)
    if corrupt_sglang_patch:
        invalid_plan = replace(
            fixture.plan,
            sglang_patch=replace(fixture.plan.sglang_patch, sha256="0" * 64),
        )
    else:
        invalid_plan = replace(
            fixture.plan,
            ktransformers_patch=replace(
                fixture.plan.ktransformers_patch, sha256="0" * 64
            ),
        )

    with pytest.raises(SourcePreparationError, match="SHA-256"):
        prepare_sglang_kt_source(fixture.ktransformers_source, invalid_plan)

    assert (
        run_git(fixture.ktransformers_source, "rev-parse", "HEAD")
        == fixture.plan.ktransformers_base_revision
    )
