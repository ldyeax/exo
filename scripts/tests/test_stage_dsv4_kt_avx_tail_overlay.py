from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.stage_dsv4_kt_avx_tail_overlay as overlay
from scripts.stage_dsv4_kt_avx_tail_overlay import OverlayStageError, stage_overlay

EXTENSION_NAME = "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"


def test_default_candidate_is_exactly_pinned() -> None:
    assert (
        Path(
            "/var/lib/exo/experiments/dsv4-avx-tail23-nblock128/lib/kt_kernel/"
            + EXTENSION_NAME
        )
        == overlay.DEFAULT_CANDIDATE
    )
    assert overlay.REQUIRED_CANDIDATE_SHA256 == (
        "32cbe088f6263bfb02c08eb1d7279de1d800147f25bfab5fa293888b2cddc0df"
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, bytes, bytes]:
    package_directory = tmp_path / "installed" / "python"
    utilities = package_directory / "utils"
    bytecode = package_directory / "__pycache__"
    utilities.mkdir(parents=True)
    bytecode.mkdir()
    (package_directory / "__init__.py").write_text("VALUE = 'installed'\n")
    (package_directory / "experts.py").write_text("EXPERTS = 256\n")
    (utilities / "amx.py").write_text("AMX = True\n")
    (bytecode / "ignored.pyc").write_bytes(b"volatile bytecode")
    installed_extension = b"installed rollback extension"
    candidate_extension = b"validated avx-tail candidate"
    (package_directory / EXTENSION_NAME).write_bytes(installed_extension)
    (package_directory / EXTENSION_NAME).chmod(0o755)

    candidate = tmp_path / "experiment" / EXTENSION_NAME
    candidate.parent.mkdir()
    candidate.write_bytes(candidate_extension)
    candidate.chmod(0o644)
    return (
        package_directory,
        candidate,
        tmp_path / "cache",
        installed_extension,
        candidate_extension,
    )


def test_stages_full_importable_overlay_without_mutating_installed_package(
    tmp_path: Path,
) -> None:
    package, candidate, cache, installed_bytes, candidate_bytes = make_inputs(tmp_path)

    result = stage_overlay(
        candidate=candidate,
        expected_candidate_sha256=sha256_bytes(candidate_bytes),
        cache_root=cache,
        package_directory=package,
    )

    staged_package = result / "kt_kernel"
    assert result.parent == cache
    assert staged_package.is_dir()
    assert (staged_package / "__init__.py").read_text() == "VALUE = 'installed'\n"
    assert (staged_package / "experts.py").read_text() == "EXPERTS = 256\n"
    assert (staged_package / "utils/amx.py").read_text() == "AMX = True\n"
    assert not (staged_package / "__pycache__").exists()
    assert (staged_package / EXTENSION_NAME).read_bytes() == candidate_bytes
    assert stat_mode(staged_package / EXTENSION_NAME) == 0o755
    assert stat_mode(candidate) == 0o644
    assert (package / EXTENSION_NAME).read_bytes() == installed_bytes

    receipt = json.loads((result / "overlay.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["candidate_extension_sha256"] == sha256_bytes(candidate_bytes)
    assert receipt["source_extension_sha256"] == sha256_bytes(installed_bytes)
    assert receipt["extension_relative_path"] == EXTENSION_NAME
    assert len(receipt["source_package_manifest_sha256"]) == 64
    assert len(receipt["staged_package_manifest_sha256"]) == 64
    assert "python_source_path" not in receipt
    assert "python_source_sha256" not in receipt
    assert "source_python_sha256" not in receipt

    repeated = stage_overlay(
        candidate=candidate,
        expected_candidate_sha256=sha256_bytes(candidate_bytes),
        cache_root=cache,
        package_directory=package,
    )
    assert repeated == result
    assert not list(cache.glob("*.partial-*"))


def test_explicit_hash_pinned_python_module_is_staged_and_receipted(
    tmp_path: Path,
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    installed_python = b"COUNTERS = None\n"
    replacement_python = b"COUNTERS = 'persistent'\n"
    (package / "experts_base.py").write_bytes(installed_python)
    (package / "experts_base.py").chmod(0o640)
    python_source = tmp_path / "source" / "experts_base.py"
    python_source.parent.mkdir()
    python_source.write_bytes(replacement_python)
    python_source.chmod(0o444)

    default_overlay = stage_overlay(
        candidate=candidate,
        expected_candidate_sha256=sha256_bytes(candidate_bytes),
        cache_root=cache,
        package_directory=package,
    )
    result = stage_overlay(
        candidate=candidate,
        expected_candidate_sha256=sha256_bytes(candidate_bytes),
        cache_root=cache,
        package_directory=package,
        python_source=python_source,
        expected_python_sha256=sha256_bytes(replacement_python),
    )

    staged_python = result / "kt_kernel/experts_base.py"
    assert result != default_overlay
    assert staged_python.read_bytes() == replacement_python
    assert stat_mode(staged_python) == 0o640
    assert stat_mode(python_source) == 0o444
    assert (package / "experts_base.py").read_bytes() == installed_python
    assert (default_overlay / "kt_kernel/experts_base.py").read_bytes() == (
        installed_python
    )

    receipt = json.loads((result / "overlay.json").read_text())
    assert receipt["python_replacement_relative_path"] == "experts_base.py"
    assert receipt["python_source_path"] == str(python_source)
    assert receipt["python_source_sha256"] == sha256_bytes(replacement_python)
    assert receipt["source_python_sha256"] == sha256_bytes(installed_python)

    repeated = stage_overlay(
        candidate=candidate,
        expected_candidate_sha256=sha256_bytes(candidate_bytes),
        cache_root=cache,
        package_directory=package,
        python_source=python_source,
        expected_python_sha256=sha256_bytes(replacement_python),
    )
    assert repeated == result


@pytest.mark.parametrize(
    ("supply_source", "supply_digest"),
    ((True, False), (False, True)),
)
def test_python_source_and_digest_must_be_supplied_together(
    tmp_path: Path, supply_source: bool, supply_digest: bool
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    python_source = tmp_path / "experts_base.py"
    python_source.write_bytes(b"replacement")

    with pytest.raises(OverlayStageError, match="must be supplied together"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
            python_source=python_source if supply_source else None,
            expected_python_sha256=(
                sha256_bytes(b"replacement") if supply_digest else None
            ),
        )

    assert not cache.exists()


def test_rejects_wrong_python_source_hash_before_creating_cache(
    tmp_path: Path,
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    python_source = tmp_path / "experts_base.py"
    python_source.write_bytes(b"replacement")

    with pytest.raises(OverlayStageError, match="Python source SHA-256 mismatch"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
            python_source=python_source,
            expected_python_sha256="0" * 64,
        )

    assert not cache.exists()


def test_python_replacement_requires_installed_destination(tmp_path: Path) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    python_source = tmp_path / "experts_base.py"
    python_source.write_bytes(b"replacement")

    with pytest.raises(OverlayStageError, match="one regular experts_base.py"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
            python_source=python_source,
            expected_python_sha256=sha256_bytes(b"replacement"),
        )


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o7777


def test_cli_emits_only_overlay_root_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)

    status = overlay.main(
        [
            "--candidate",
            str(candidate),
            "--expected-sha256",
            sha256_bytes(candidate_bytes),
            "--cache-root",
            str(cache),
            "--package-dir",
            str(package),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    emitted = Path(captured.out.rstrip("\n"))
    assert emitted.parent == cache
    assert (emitted / "kt_kernel/__init__.py").is_file()


def test_rejects_wrong_candidate_hash_before_creating_cache(tmp_path: Path) -> None:
    package, candidate, cache, _, _ = make_inputs(tmp_path)

    with pytest.raises(OverlayStageError, match="candidate SHA-256 mismatch"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256="0" * 64,
            cache_root=cache,
            package_directory=package,
        )

    assert not cache.exists()


def test_rejects_missing_matching_installed_extension(tmp_path: Path) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    (package / EXTENSION_NAME).unlink()

    with pytest.raises(OverlayStageError, match="must contain one regular"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
        )


def test_existing_overlay_tamper_fails_closed_instead_of_repairing(
    tmp_path: Path,
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    arguments = {
        "candidate": candidate,
        "expected_candidate_sha256": sha256_bytes(candidate_bytes),
        "cache_root": cache,
        "package_directory": package,
    }
    result = stage_overlay(**arguments)
    tampered = result / "kt_kernel/experts.py"
    tampered.write_text("EXPERTS = 1\n")

    with pytest.raises(OverlayStageError, match="staged package verification failed"):
        stage_overlay(**arguments)

    assert tampered.read_text() == "EXPERTS = 1\n"


def test_existing_python_replacement_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    (package / "experts_base.py").write_bytes(b"installed")
    python_source = tmp_path / "experts_base.py"
    replacement_python = b"pinned replacement"
    python_source.write_bytes(replacement_python)

    def stage_test_overlay() -> Path:
        return stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
            python_source=python_source,
            expected_python_sha256=sha256_bytes(replacement_python),
        )

    result = stage_test_overlay()
    tampered = result / "kt_kernel/experts_base.py"
    tampered.write_bytes(b"tampered")

    with pytest.raises(OverlayStageError, match="staged package verification failed"):
        stage_test_overlay()

    assert tampered.read_bytes() == b"tampered"


def test_publication_failure_leaves_no_partial_or_published_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(overlay.os, "rename", fail_rename)
    with pytest.raises(OverlayStageError, match="cannot atomically publish"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
        )

    assert {path.name for path in cache.iterdir()} == {".stage.lock"}


def test_rejects_symlink_in_package_payload(tmp_path: Path) -> None:
    package, candidate, cache, _, candidate_bytes = make_inputs(tmp_path)
    os.symlink(package / "experts.py", package / "experts_alias.py")

    with pytest.raises(OverlayStageError, match="unsupported symlink"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
        )


def test_rejects_cache_below_installed_package_without_creating_it(
    tmp_path: Path,
) -> None:
    package, candidate, _, _, candidate_bytes = make_inputs(tmp_path)
    cache = package / "forbidden-cache"

    with pytest.raises(OverlayStageError, match="must not be the installed package"):
        stage_overlay(
            candidate=candidate,
            expected_candidate_sha256=sha256_bytes(candidate_bytes),
            cache_root=cache,
            package_directory=package,
        )

    assert not cache.exists()
