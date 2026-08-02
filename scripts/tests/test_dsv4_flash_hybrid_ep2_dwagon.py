from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "dsv4_flash_hybrid_ep2_dwagon.sh"


def launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DSV4_")
        and name
        not in {
            "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
        }
    }
    invocation_log = tmp_path / "python-invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'multi_stream=%s block_size=%s args=%s\\n' "
        '"${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP-<unset>}" '
        '"${DSV4_DSPARK_BLOCK_SIZE-<unset>}" "$*" '
        '>>"$DSV4_TEST_PYTHON_INVOCATIONS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment.update(
        {
            "DSV4_CACHE_ROOT": str(tmp_path / "cache"),
            "DSV4_PYTHON": str(fake_python),
            "DSV4_TEST_PYTHON_INVOCATIONS": str(invocation_log),
        }
    )
    return environment, invocation_log


def run_launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_launcher_rejects_gpu_arithmetic_expansion_before_evaluation(
    tmp_path: Path,
) -> None:
    environment, _ = launcher_environment(tmp_path)
    side_effect_path = tmp_path / "arithmetic-expanded"
    environment["DSV4_GPU_EXPERTS_PER_LAYER"] = f"array[$(touch {side_effect_path})]+12"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be a decimal integer" in result.stderr
    assert not side_effect_path.exists()


def test_launcher_rejects_prefix_before_cache_path_interpolation(
    tmp_path: Path,
) -> None:
    environment, _ = launcher_environment(tmp_path)
    environment["DSV4_HYBRID_GPU_SELECTION_STRATEGY"] = (
        "profile-hot-prefix-profile-fill"
    )
    environment["DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER"] = "../../escaped"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be a decimal integer" in result.stderr
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "cache").exists()


def test_launcher_uses_explicit_prebuilt_plan_without_overwriting_it(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    prebuilt_plan = tmp_path / "prebuilt-hybrid-plan.pt"
    original_contents = b"prebuilt-plan-sentinel"
    prebuilt_plan.write_bytes(original_contents)
    environment["DSV4_HYBRID_EXPERT_SHARD_PLAN"] = str(prebuilt_plan)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert prebuilt_plan.read_bytes() == original_contents
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "prepare_dsv4_flash_0731.py" in invocations[0]
    assert "build_dsv4_kt_hybrid_shard_plan.py" not in invocations[0]


def test_launcher_rejects_missing_explicit_prebuilt_plan(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    missing_plan = tmp_path / "missing-hybrid-plan.pt"
    environment["DSV4_HYBRID_EXPERT_SHARD_PLAN"] = str(missing_plan)

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "does not exist or is not readable" in result.stderr
    assert not invocation_log.exists()


def test_launcher_retains_default_plan_auto_build(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    assert "build_dsv4_kt_hybrid_shard_plan.py" in invocations[0]
    assert "--gpu-rank-counts 12,12" in invocations[0]
    assert "--cpu-rank-counts 116,116" in invocations[0]
    assert "--gpu-selection profile-hot-prefix-profile-fill" in invocations[0]
    assert "--profile-hot-prefix-experts-per-layer 12" in invocations[0]
    assert "--gpu-fill-profile" in invocations[0]
    assert "prepare_dsv4_flash_0731.py" in invocations[1]


def test_launcher_defaults_to_safe_single_stream_moe(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare_invocation = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert prepare_invocation.startswith("multi_stream=0 ")


def test_launcher_preserves_explicit_multi_stream_override(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["SGLANG_OPT_USE_MULTI_STREAM_OVERLAP"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare_invocation = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert prepare_invocation.startswith("multi_stream=1 ")


def test_launcher_rejects_invalid_multi_stream_value(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["SGLANG_OPT_USE_MULTI_STREAM_OVERLAP"] = "invalid"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be a boolean value" in result.stderr
    assert not invocation_log.exists()


def test_launcher_defaults_to_validated_dspark_block_size(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare_invocation = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert " block_size=6 " in prepare_invocation


def test_launcher_preserves_explicit_dspark_block_size(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DSPARK_BLOCK_SIZE"] = "5"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare_invocation = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert " block_size=5 " in prepare_invocation
