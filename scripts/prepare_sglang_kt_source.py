#!/usr/bin/env python3
"""Apply Exo's exact SGLang-KTransformers integration commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

KTRANSFORMERS_BASE_REVISION = "8e46e5896c3d993a1285052f2618f5a9f01882d4"
KTRANSFORMERS_RESULT_REVISION = "f9ca69648421f5774215c4da9cf711dccf54f49e"
SGLANG_BASE_REVISION = "5d6bef9f61637aaeaf047bf8209def2af3eaa83f"
SGLANG_RESULT_REVISION = "3721d710102456b6bf849122e781129dc3f7d9c6"
SGLANG_SUBMODULE_PATH = Path("third_party/sglang")
PATCH_DIRECTORY = Path(__file__).resolve().parent / "patches" / "sglang_kt"


class SourcePreparationError(RuntimeError):
    """Raised when source state cannot produce the exact admitted revisions."""


@dataclass(frozen=True)
class MailPatch:
    path: Path
    sha256: str
    committer_name: str
    committer_email: str


@dataclass(frozen=True)
class SglangKtSourcePlan:
    ktransformers_base_revision: str
    ktransformers_result_revision: str
    sglang_base_revision: str
    sglang_result_revision: str
    sglang_submodule_path: Path
    sglang_patch: MailPatch
    ktransformers_patch: MailPatch

    @classmethod
    def exo_default(cls) -> Self:
        return cls(
            ktransformers_base_revision=KTRANSFORMERS_BASE_REVISION,
            ktransformers_result_revision=KTRANSFORMERS_RESULT_REVISION,
            sglang_base_revision=SGLANG_BASE_REVISION,
            sglang_result_revision=SGLANG_RESULT_REVISION,
            sglang_submodule_path=SGLANG_SUBMODULE_PATH,
            sglang_patch=MailPatch(
                path=PATCH_DIRECTORY
                / "0001-feat-fail-closed-on-GLM-Flash-KT-coverage.patch",
                sha256=(
                    "90d7cebbe2ece4b4498a72f3c5c27d33417f56a80be094d6da93022852eae584"
                ),
                committer_name="jimm",
                committer_email="jimm@jimm.horse",
            ),
            ktransformers_patch=MailPatch(
                path=PATCH_DIRECTORY / "0002-build-pin-GLM-Flash-KT-registration.patch",
                sha256=(
                    "885f4188b9ae92df4ff396823b1e29c7784e10d8278f4e9fac19220fbea5abd7"
                ),
                committer_name="jimm",
                committer_email="jimm@jimm.horse",
            ),
        )


@dataclass(frozen=True)
class SglangKtSourceReceipt:
    schema_version: int
    ktransformers_source: str
    ktransformers_revision: str
    sglang_source: str
    sglang_revision: str
    patches: tuple[dict[str, str], ...]


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
        env=None if environment is None else {**os.environ, **environment},
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise SourcePreparationError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return result


def _git_output(repository: Path, *arguments: str) -> str:
    return _run_git(repository, arguments).stdout.strip()


def _head_revision(repository: Path) -> str:
    return _git_output(repository, "rev-parse", "HEAD").lower()


def _require_clean(repository: Path, description: str) -> None:
    status = _git_output(repository, "status", "--porcelain=v1")
    if status:
        raise SourcePreparationError(f"{description} is not clean: {status}")


def _require_clean_ignoring_submodule(
    repository: Path,
    submodule_name: str,
    description: str,
) -> None:
    status = _git_output(
        repository,
        "-c",
        f"submodule.{submodule_name}.ignore=all",
        "status",
        "--porcelain=v1",
    )
    if status:
        raise SourcePreparationError(f"{description} is not clean: {status}")


def _require_revision(
    repository: Path, expected_revision: str, description: str
) -> None:
    observed_revision = _head_revision(repository)
    if observed_revision != expected_revision:
        raise SourcePreparationError(
            f"{description} revision is {observed_revision}, expected {expected_revision}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_patch(patch: MailPatch) -> None:
    if not patch.path.is_file():
        raise SourcePreparationError(f"required mail patch is absent: {patch.path}")
    observed_digest = _sha256(patch.path)
    if observed_digest != patch.sha256:
        raise SourcePreparationError(
            f"mail patch {patch.path} SHA-256 is {observed_digest}, "
            f"expected {patch.sha256}"
        )


def _apply_mail_patch(
    repository: Path,
    patch: MailPatch,
    expected_revision: str,
    *,
    require_clean_result: bool = True,
) -> None:
    _require_patch(patch)
    original_revision = _head_revision(repository)
    environment = {
        "GIT_COMMITTER_NAME": patch.committer_name,
        "GIT_COMMITTER_EMAIL": patch.committer_email,
    }
    try:
        _run_git(
            repository,
            ("am", "--committer-date-is-author-date", str(patch.path)),
            environment=environment,
        )
        _require_revision(repository, expected_revision, f"patched source {repository}")
        if require_clean_result:
            _require_clean(repository, f"patched source {repository}")
    except SourcePreparationError:
        _run_git(repository, ("am", "--abort"), check=False)
        _run_git(repository, ("reset", "--hard", original_revision), check=False)
        raise


def _initialize_sglang_submodule(
    ktransformers_source: Path, plan: SglangKtSourcePlan
) -> Path:
    _run_git(
        ktransformers_source,
        (
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--checkout",
            str(plan.sglang_submodule_path),
        ),
    )
    sglang_source = (ktransformers_source / plan.sglang_submodule_path).resolve(
        strict=True
    )
    _require_revision(sglang_source, plan.sglang_base_revision, "SGLang base")
    _require_clean(sglang_source, "SGLang base")
    return sglang_source


def _submodule_url(ktransformers_source: Path, submodule_path: Path) -> str:
    submodule_name = str(submodule_path)
    configured_url = _run_git(
        ktransformers_source,
        ("config", "--get", f"submodule.{submodule_name}.url"),
        check=False,
    ).stdout.strip()
    if configured_url:
        return configured_url
    recorded_url = _run_git(
        ktransformers_source,
        (
            "config",
            "--file",
            str(ktransformers_source / ".gitmodules"),
            "--get",
            f"submodule.{submodule_name}.url",
        ),
        check=False,
    ).stdout.strip()
    if not recorded_url:
        raise SourcePreparationError(
            f"no URL is configured for submodule {submodule_name}"
        )
    return recorded_url


def _clone_sglang_base_for_integrated_parent(
    ktransformers_source: Path, plan: SglangKtSourcePlan
) -> Path:
    sglang_source = ktransformers_source / plan.sglang_submodule_path
    if sglang_source.exists() and any(sglang_source.iterdir()):
        raise SourcePreparationError(
            f"uninitialized SGLang submodule path is not empty: {sglang_source}"
        )
    sglang_source.parent.mkdir(parents=True, exist_ok=True)
    submodule_url = _submodule_url(ktransformers_source, plan.sglang_submodule_path)
    result = subprocess.run(
        (
            "git",
            "-c",
            "protocol.file.allow=always",
            "clone",
            "--no-checkout",
            submodule_url,
            str(sglang_source),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise SourcePreparationError(
            f"could not clone SGLang base from {submodule_url}: {detail}"
        )
    _run_git(sglang_source, ("checkout", "--detach", plan.sglang_base_revision))
    _require_clean(sglang_source, "SGLang base for integrated parent")
    _apply_mail_patch(
        sglang_source,
        plan.sglang_patch,
        plan.sglang_result_revision,
    )
    return sglang_source.resolve(strict=True)


def _is_repository_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    result = _run_git(
        path,
        ("rev-parse", "--show-toplevel"),
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        observed_root = Path(result.stdout.strip()).resolve(strict=True)
        expected_root = path.resolve(strict=True)
    except OSError:
        return False
    return observed_root == expected_root


def _require_integrated_source(
    ktransformers_source: Path, plan: SglangKtSourcePlan
) -> Path:
    _require_revision(
        ktransformers_source,
        plan.ktransformers_result_revision,
        "KTransformers integration",
    )
    _require_clean_ignoring_submodule(
        ktransformers_source,
        str(plan.sglang_submodule_path),
        "KTransformers integration excluding its SGLang submodule",
    )
    sglang_source = ktransformers_source / plan.sglang_submodule_path
    if not _is_repository_root(sglang_source):
        sglang_source = _clone_sglang_base_for_integrated_parent(
            ktransformers_source, plan
        )
    observed_sglang_revision = _head_revision(sglang_source)
    if observed_sglang_revision == plan.sglang_base_revision:
        _require_clean(sglang_source, "SGLang base for integrated parent")
        result_exists = _run_git(
            sglang_source,
            ("cat-file", "-e", f"{plan.sglang_result_revision}^{{commit}}"),
            check=False,
        )
        if result_exists.returncode == 0:
            _run_git(
                sglang_source,
                ("checkout", "--detach", plan.sglang_result_revision),
            )
        else:
            _apply_mail_patch(
                sglang_source,
                plan.sglang_patch,
                plan.sglang_result_revision,
            )
    _require_revision(sglang_source, plan.sglang_result_revision, "SGLang integration")
    _require_clean(sglang_source, "SGLang integration")
    _require_clean(ktransformers_source, "KTransformers integration")
    return sglang_source


def prepare_sglang_kt_source(
    ktransformers_source: Path,
    plan: SglangKtSourcePlan | None = None,
) -> SglangKtSourceReceipt:
    """Apply both exact mail patches and return a machine-readable receipt."""
    selected_plan = plan or SglangKtSourcePlan.exo_default()
    source = ktransformers_source.expanduser().resolve(strict=True)
    _require_patch(selected_plan.sglang_patch)
    _require_patch(selected_plan.ktransformers_patch)
    observed_revision = _head_revision(source)

    if observed_revision == selected_plan.ktransformers_result_revision:
        sglang_source = _require_integrated_source(source, selected_plan)
    elif observed_revision == selected_plan.ktransformers_base_revision:
        _require_clean(source, "KTransformers base")
        sglang_source = _initialize_sglang_submodule(source, selected_plan)
        _apply_mail_patch(
            sglang_source,
            selected_plan.sglang_patch,
            selected_plan.sglang_result_revision,
        )

        # The outer mail patch needs a clean base checkout. Keep the new commit
        # object locally, return the worktree to the old gitlink, apply the
        # gitlink commit, and then restore the integrated submodule checkout.
        _run_git(
            sglang_source,
            ("checkout", "--detach", selected_plan.sglang_base_revision),
        )
        _require_clean(source, "KTransformers base after SGLang patch creation")
        _apply_mail_patch(
            source,
            selected_plan.ktransformers_patch,
            selected_plan.ktransformers_result_revision,
            require_clean_result=False,
        )
        _run_git(
            sglang_source,
            ("checkout", "--detach", selected_plan.sglang_result_revision),
        )
        sglang_source = _require_integrated_source(source, selected_plan)
    else:
        raise SourcePreparationError(
            f"KTransformers revision is {observed_revision}; expected base "
            f"{selected_plan.ktransformers_base_revision} or integrated result "
            f"{selected_plan.ktransformers_result_revision}"
        )

    return SglangKtSourceReceipt(
        schema_version=1,
        ktransformers_source=str(source),
        ktransformers_revision=selected_plan.ktransformers_result_revision,
        sglang_source=str(sglang_source),
        sglang_revision=selected_plan.sglang_result_revision,
        patches=(
            {
                "path": str(selected_plan.sglang_patch.path),
                "sha256": selected_plan.sglang_patch.sha256,
            },
            {
                "path": str(selected_plan.ktransformers_patch.path),
                "sha256": selected_plan.ktransformers_patch.sha256,
            },
        ),
    )


class _CliArguments(argparse.Namespace):
    ktransformers_source: Path


def _parse_arguments() -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ktransformers-source",
        required=True,
        type=Path,
        help="Existing KTransformers checkout at the exact admitted base or result",
    )
    arguments = _CliArguments()
    parser.parse_args(namespace=arguments)
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    try:
        receipt = prepare_sglang_kt_source(arguments.ktransformers_source)
    except (OSError, SourcePreparationError) as error:
        raise SystemExit(f"source preparation failed: {error}") from error
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
