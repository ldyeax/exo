from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGER = REPOSITORY_ROOT / "scripts" / "stage_kimi_k3_gguf.sh"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _write_manifest(path: Path, quant: str, files: dict[str, bytes]) -> None:
    payload = {
        "schema_version": 1,
        "repository": "test/kimi-k3",
        "revision": REVISION,
        "quantizations": {
            quant: [
                [name, len(content), hashlib.sha256(content).hexdigest()]
                for name, content in sorted(files.items())
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_stager(
    *,
    manifest: Path,
    quant: str,
    source: Path,
    destination_root: Path,
    plan: bool = False,
    verify_complete_partial: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(STAGER),
        "--quant",
        quant,
        "--source",
        str(source),
        "--destination-root",
        str(destination_root),
        "--parallel-copies",
        "2",
        "--hash-workers",
        "2",
    ]
    if plan:
        command.append("--plan")
    if verify_complete_partial:
        command.append("--verify-complete-partial")
    environment = os.environ.copy()
    environment["KIMI_K3_GGUF_MANIFEST_FILE"] = str(manifest)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_stage_publishes_only_after_pinned_sha256_verification(
    tmp_path: Path,
) -> None:
    quant = "TEST-Q2"
    files = {
        "Kimi-K3-TEST-Q2-00001-of-00002.gguf": b"header-data",
        "Kimi-K3-TEST-Q2-00002-of-00002.gguf": b"expert-data" * 4096,
    }
    source = tmp_path / "source"
    source.mkdir()
    for name, content in files.items():
        (source / name).write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, files)
    destination_root = tmp_path / "destination"

    result = _run_stager(
        manifest=manifest,
        quant=quant,
        source=source,
        destination_root=destination_root,
    )

    assert result.returncode == 0, result.stderr
    published = destination_root / quant
    assert published.is_dir()
    assert not (destination_root / f".{quant}.exo-partial").exists()
    for name, content in files.items():
        assert (published / name).read_bytes() == content
    receipt = json.loads(
        (published / ".exo-kimi-k3-stage-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == 2
    assert receipt["pinned_manifest"]["repository"] == "test/kimi-k3"
    assert receipt["pinned_manifest"]["revision"] == REVISION
    assert "per-shard SHA-256" in receipt["verification"]


def test_stage_rejects_same_size_content_corruption_before_publication(
    tmp_path: Path,
) -> None:
    quant = "TEST-IQ2"
    file_name = "Kimi-K3-TEST-IQ2-00001-of-00001.gguf"
    expected_content = b"correct-content"
    source = tmp_path / "source"
    source.mkdir()
    (source / file_name).write_bytes(b"corrupt-content")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, {file_name: expected_content})
    destination_root = tmp_path / "destination"

    result = _run_stager(
        manifest=manifest,
        quant=quant,
        source=source,
        destination_root=destination_root,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not (destination_root / quant).exists()
    assert (destination_root / f".{quant}.exo-partial" / file_name).is_file()


def test_stage_rejects_source_that_does_not_match_pinned_names_and_sizes(
    tmp_path: Path,
) -> None:
    quant = "TEST-Q2"
    file_name = "Kimi-K3-TEST-Q2-00001-of-00001.gguf"
    source = tmp_path / "source"
    source.mkdir()
    (source / file_name).write_bytes(b"one-byte-too-long")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, {file_name: b"expected-content"})

    result = _run_stager(
        manifest=manifest,
        quant=quant,
        source=source,
        destination_root=tmp_path / "destination",
        plan=True,
    )

    assert result.returncode != 0
    assert "no source exactly matches the pinned" in result.stderr


def test_stage_plan_reports_pinned_repository_revision_and_hash_policy(
    tmp_path: Path,
) -> None:
    quant = "TEST-Q2"
    file_name = "Kimi-K3-TEST-Q2-00001-of-00001.gguf"
    content = b"small-model"
    source = tmp_path / "source"
    source.mkdir()
    (source / file_name).write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, {file_name: content})

    result = _run_stager(
        manifest=manifest,
        quant=quant,
        source=source,
        destination_root=tmp_path / "destination",
        plan=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["pinned_manifest"]["repository"] == "test/kimi-k3"
    assert plan["pinned_manifest"]["revision"] == REVISION
    assert plan["content_verification"]["algorithm"] == "sha256"
    assert plan["content_verification"]["timing"] == (
        "after copy and before atomic publication"
    )


def test_complete_partial_can_skip_redundant_rsync_but_not_sha256(
    tmp_path: Path,
) -> None:
    quant = "TEST-IQ2"
    file_name = "Kimi-K3-TEST-IQ2-00001-of-00001.gguf"
    content = b"already-complete-partial"
    source = tmp_path / "source"
    source.mkdir()
    (source / file_name).write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, {file_name: content})
    destination_root = tmp_path / "destination"
    partial = destination_root / f".{quant}.exo-partial"
    partial.mkdir(parents=True)
    (partial / file_name).write_bytes(content)

    result = _run_stager(
        manifest=manifest,
        quant=quant,
        source=source,
        destination_root=destination_root,
        verify_complete_partial=True,
    )

    assert result.returncode == 0, result.stderr
    assert "skipping redundant rsync" in result.stderr
    assert "verified sha256=" in result.stderr
    assert (destination_root / quant / file_name).read_bytes() == content


def test_running_stage_uses_immutable_unlinked_script_snapshot(
    tmp_path: Path,
) -> None:
    quant = "TEST-Q2"
    file_name = "Kimi-K3-TEST-Q2-00001-of-00001.gguf"
    content = b"small-model"
    source = tmp_path / "source"
    source.mkdir()
    (source / file_name).write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, quant, {file_name: content})

    mutable_stager = tmp_path / "stage_kimi_k3_gguf.sh"
    stager_source = STAGER.read_text(encoding="utf-8")
    insertion_point = (
        "done\n\n[[ -n ${selected_source} ]] ||\n"
        '  fail "no source exactly matches the pinned ${quant} manifest"\n'
    )
    assert insertion_point in stager_source
    padding = "# immutable-snapshot regression padding\n" * 65536
    mutable_stager.write_text(
        stager_source.replace(
            insertion_point,
            f"done\n\n{padding}\n[[ -n ${{selected_source}} ]] ||\n"
            '  fail "no source exactly matches the pinned ${quant} manifest"\n',
            1,
        ),
        encoding="utf-8",
    )
    mutable_stager.chmod(0o755)

    real_findmnt = shutil.which("findmnt")
    assert real_findmnt is not None
    fake_binary_directory = tmp_path / "fake-bin"
    fake_binary_directory.mkdir()
    blocked_marker = tmp_path / "findmnt-blocked"
    release_marker = tmp_path / "release-findmnt"
    fake_findmnt = fake_binary_directory / "findmnt"
    fake_findmnt.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'touch "${KIMI_STAGE_TEST_BLOCKED_MARKER}"\n'
        "while [[ ! -e ${KIMI_STAGE_TEST_RELEASE_MARKER} ]]; do\n"
        "  sleep 0.01\n"
        "done\n"
        f'exec "{real_findmnt}" "$@"\n',
        encoding="utf-8",
    )
    fake_findmnt.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_binary_directory}:{environment['PATH']}"
    environment["KIMI_K3_GGUF_MANIFEST_FILE"] = str(manifest)
    environment["KIMI_STAGE_TEST_BLOCKED_MARKER"] = str(blocked_marker)
    environment["KIMI_STAGE_TEST_RELEASE_MARKER"] = str(release_marker)
    process = subprocess.Popen(
        [
            str(mutable_stager),
            "--quant",
            quant,
            "--source",
            str(source),
            "--destination-root",
            str(tmp_path / "destination"),
            "--plan",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )

    try:
        deadline = time.monotonic() + 10
        while not blocked_marker.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)

        # Truncate the original inode in place while the child is blocked
        # before the second half of the script. The child must continue from
        # its already-open anonymous snapshot.
        mutable_stager.write_text(
            "#!/usr/bin/env bash\nexit 97\n",
            encoding="utf-8",
        )
        release_marker.touch()
        stdout, stderr = process.communicate(timeout=30)
    finally:
        release_marker.touch()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    assert process.returncode == 0, stderr
    plan = json.loads(stdout)
    assert plan["quant"] == quant
    assert plan["file_count"] == 1
