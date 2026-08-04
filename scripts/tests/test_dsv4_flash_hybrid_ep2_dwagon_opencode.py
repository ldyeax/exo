from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "dsv4_flash_hybrid_ep2_dwagon_opencode.sh"
DRAFT_SPARSE_PROFILE = (
    REPOSITORY_ROOT
    / "scripts"
    / "data"
    / "dsv4_flash_draft_distinct_decode_calls_sparse.json"
)
TARGET_SPARSE_PROFILE = (
    REPOSITORY_ROOT
    / "scripts"
    / "data"
    / "dsv4_flash_opencode_target_distinct_decode_calls_sparse.json"
)
FROZEN_TARGET_MANIFEST = (
    REPOSITORY_ROOT
    / "scripts"
    / "data"
    / "dsv4_flash_opencode_g14_p28_frozen_plan.json"
)
TARGET_OVERRIDE_VARIABLES = (
    "DSV4_EXPERT_RECORDER_PROFILES",
    "DSV4_HYBRID_EXPERT_PROFILE",
    "DSV4_HYBRID_EXPERT_SHARD_PLAN",
    "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
)
DRAFT_OVERRIDE_VARIABLES = (
    "DSV4_DRAFT_EXPERT_RECORDER_PROFILES",
    "DSV4_DRAFT_HYBRID_EXPERT_PROFILE",
    "DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
    "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
)


def launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DSV4_")
        and name
        not in {
            "KT_MXFP4_AMX_MIN_EXPERT_TOKENS",
            "KT_MXFP4_AVX_SCALE_FOLD_MODE",
            "KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS",
            "KT_SINGLE_NUMA_INLINE_DISPATCH",
            "KT_TASK_QUEUE_PIN_FIRST_CORE",
            "KT_WORKER_SPIN_US",
            "SGLANG_DSPARK_FP32_LM_HEAD",
            "SGLANG_DSPARK_OPT_MARKOV_W2_BF16",
            "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE",
            "SGLANG_DSV4_INT4_KV_STORAGE",
            "SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH",
            "SGLANG_DSV4_OSCAR_CALIBRATION_PATH",
            "SGLANG_DSV4_OSCAR_CAPTURE_CONFIG",
            "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE",
            "SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY",
            "SGLANG_DSV4_SM86_C128_BF16_STORAGE",
            "SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU",
            "SGLANG_KT_DRAFT_METHOD",
            "SGLANG_KT_DRAFT_WEIGHT_PATH",
            "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
            "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM",
            "SGLANG_V4_MXFP4_SMALL_ROW_ROUTING",
        }
    }
    invocation_log = tmp_path / "python-invocations.txt"
    overlay_root = tmp_path / "kt-avx-tail-overlay"
    (overlay_root / "kt_kernel").mkdir(parents=True)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1:-} == */stage_dsv4_kt_avx_tail_overlay.py ]]; then\n"
        '  printf \'stage=%s\\n\' "$*" >>"$DSV4_TEST_PYTHON_INVOCATIONS"\n'
        "  if [[ ${DSV4_TEST_STAGE_FAIL:-0} == 1 ]]; then\n"
        "    printf 'synthetic overlay staging failure\\n' >&2\n"
        "    exit 19\n"
        "  fi\n"
        "  printf '%s\\n' \"$DSV4_TEST_OVERLAY_ROOT\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ ${1:-} == */materialize_dsv4_frozen_hybrid_plan.py ]]; then\n"
        '  printf \'materialize=%s\\n\' "$*" >>"$DSV4_TEST_PYTHON_INVOCATIONS"\n'
        "  while (($#)); do\n"
        "    if [[ $1 == --output ]]; then\n"
        '      mkdir -p "${2%/*}"\n'
        '      : >"$2"\n'
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "  exit 0\n"
        "fi\n"
        "if [[ ${1:-} == */dsv4_oscar_int2_calibration.py && ${2:-} == admit ]]; then\n"
        '  printf \'admit=%s\\n\' "$*" >>"$DSV4_TEST_PYTHON_INVOCATIONS"\n'
        "  while (($#)); do\n"
        "    if [[ $1 == --output ]]; then\n"
        '      mkdir -p "${2%/*}"\n'
        '      printf \'{"format":"test-oscar-admission"}\\n\' >"$2"\n'
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "  exit 0\n"
        "fi\n"
        "printf 'args=%s|target_profile=%s|target_plan=%s|target_sglang_plan=%s|"
        "gpu=%s|prefix=%s|draft_profile=%s|draft_plan=%s|draft_sglang_plan=%s|"
        "bcg_capture=%s|bcg_eager=%s|multi_stream=%s|lm_fp32=%s|markov_bf16=%s|"
        "amx_min=%s|avx_min=%s|small_row=%s|small_gemm=%s|kt_source=%s|"
        "split_amx=%s|draft_method=%s|draft_weights=%s|oscar=%s|oscar_split=%s|"
        "oscar_path=%s|oscar_admission=%s|int4_kv=%s|int4_indexer=%s|"
        "c128_bf16=%s|task_queue_pin=%s|inline_dispatch=%s|"
        "scale_fold=%s|worker_spin=%s\\n' "
        '"$*" "${DSV4_HYBRID_EXPERT_PROFILE-}" '
        '"${DSV4_HYBRID_EXPERT_SHARD_PLAN-}" '
        '"${SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN-}" '
        '"${DSV4_GPU_EXPERTS_PER_LAYER-}" '
        '"${DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER-}" '
        '"${DSV4_DRAFT_HYBRID_EXPERT_PROFILE-}" '
        '"${DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN-}" '
        '"${SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN-}" '
        '"${SGLANG_DSV4_CAPTURE_ATTN_IN_BCG-}" '
        '"${SGLANG_DSV4_EAGER_ATTN_MODULE_IN_BCG-}" '
        '"${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP-}" '
        '"${SGLANG_DSPARK_FP32_LM_HEAD-}" '
        '"${SGLANG_DSPARK_OPT_MARKOV_W2_BF16-}" '
        '"${KT_MXFP4_AMX_MIN_EXPERT_TOKENS-}" '
        '"${KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS-}" '
        '"${SGLANG_V4_MXFP4_SMALL_ROW_ROUTING-}" '
        '"${SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM-}" '
        '"${DSV4_KTRANSFORMERS_SOURCE-}" '
        '"${SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU-}" '
        '"${SGLANG_KT_DRAFT_METHOD-}" '
        '"${SGLANG_KT_DRAFT_WEIGHT_PATH-}" '
        '"${SGLANG_DSV4_OSCAR_INT2_KV_STORAGE-}" '
        '"${SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY-}" '
        '"${SGLANG_DSV4_OSCAR_CALIBRATION_PATH-}" '
        '"${SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH-}" '
        '"${SGLANG_DSV4_INT4_KV_STORAGE-}" '
        '"${SGLANG_DSV4_INT4_C4_INDEXER_STORAGE-}" '
        '"${SGLANG_DSV4_SM86_C128_BF16_STORAGE-}" '
        '"${KT_TASK_QUEUE_PIN_FIRST_CORE-}" '
        '"${KT_SINGLE_NUMA_INLINE_DISPATCH-}" '
        '"${KT_MXFP4_AVX_SCALE_FOLD_MODE-}" '
        '"${KT_WORKER_SPIN_US-}" '
        '>>"$DSV4_TEST_PYTHON_INVOCATIONS"\n'
        "if [[ ${1:-} == */build_dsv4_kt_hybrid_shard_plan.py ]]; then\n"
        "  while (($#)); do\n"
        "    if [[ $1 == --output ]]; then\n"
        '      mkdir -p "${2%/*}"\n'
        "      printf 'fake hybrid plan' >\"$2\"\n"
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '24576\\n24576\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    cache_root = tmp_path / "cache"
    oscar_artifact = cache_root / "oscar-int2" / "dsv4-oscar-int2-calibration.pt"
    oscar_artifact.parent.mkdir(parents=True)
    oscar_artifact.write_bytes(b"synthetic launcher-only OSCAR artifact")
    (oscar_artifact.parent / "checkpoint-fingerprint.json").write_text(
        "{}\n", encoding="utf-8"
    )
    environment.update(
        {
            "DSV4_CACHE_ROOT": str(cache_root),
            "DSV4_PYTHON": str(fake_python),
            "DSV4_TEST_OVERLAY_ROOT": str(overlay_root),
            "DSV4_TEST_PYTHON_INVOCATIONS": str(invocation_log),
            "PATH": f"{tmp_path}:{environment['PATH']}",
        }
    )
    return environment, invocation_log


def run_launcher(
    environment: dict[str, str], arguments: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def draft_plan_builds(invocations: list[str]) -> list[str]:
    return [line for line in invocations if "--ordering-layer-indices" in line]


def target_plan_builds(invocations: list[str]) -> list[str]:
    return [
        line
        for line in invocations
        if "build_dsv4_kt_hybrid_shard_plan.py" in line
        and "--ordering-layer-indices" not in line
    ]


def frozen_target_materializations(invocations: list[str]) -> list[str]:
    return [line for line in invocations if line.startswith("materialize=")]


def test_opencode_launcher_defaults_to_requalified_breakable_prefill(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--cuda-graph-backend-prefill breakable" in launch
    assert "--context-length 524288 --max-total-tokens 524288" in launch
    assert "--swa-full-tokens-ratio 0.0048828125" in launch
    assert "--kv-cache-dtype fp8_e4m3" in launch
    assert "--chunked-prefill-size 1024 --max-prefill-tokens 1024" in launch
    assert "--max-running-requests 1" in launch
    assert "--cuda-graph-backend-decode full" in launch
    assert "--cuda-graph-max-bs-decode 1 --cuda-graph-bs-decode 1" in launch
    assert "--cuda-graph-max-bs-prefill 1024" in launch
    assert "--cuda-graph-bs-prefill 256 512 1024" in launch
    assert "--skip-server-warmup" not in launch
    assert "--warmups dsv4_opencode_2694" in launch
    assert "--enable-p2p-check" in launch
    assert "--pre-warm-nccl" in launch
    assert "--speculative-dspark-block-size 5" in launch
    assert "--speculative-dspark-fixed-verify-len 4" in launch
    assert "--disable-shared-experts-fusion" in launch
    assert "--model-path /tmp/dsv4-local-checkpoint-0731" in launch
    assert (
        "--kt-weight-path /tmp/dsv4-local-checkpoint-0731 --kt-method MXFP4" in launch
    )
    assert "|gpu=14|prefix=14|" in launch
    assert (
        "|bcg_capture=|bcg_eager=1|multi_stream=1|lm_fp32=0|"
        "markov_bf16=0|amx_min=5|avx_min=2|small_row=1|small_gemm=1|"
    ) in launch
    assert f"|kt_source={environment['DSV4_TEST_OVERLAY_ROOT']}" in launch
    assert "|split_amx=|draft_method=|draft_weights=" in launch
    expected_oscar_artifact = (
        Path(environment["DSV4_CACHE_ROOT"])
        / "oscar-int2"
        / "dsv4-oscar-int2-calibration.pt"
    )
    expected_oscar_admission = expected_oscar_artifact.parent / "admission.json"
    assert (
        f"|oscar=1|oscar_split=1|oscar_path={expected_oscar_artifact}|"
        f"oscar_admission={expected_oscar_admission}|int4_kv=0|"
        "int4_indexer=0|c128_bf16=0"
    ) in launch
    admission = [
        line
        for line in invocation_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("admit=")
    ]
    assert len(admission) == 1
    assert f"--artifact {expected_oscar_artifact}" in admission[0]
    assert "--checkpoint /tmp/dsv4-local-checkpoint-0731" in admission[0]
    assert "--model-id deepseek-ai/DeepSeek-V4-Flash" in admission[0]
    assert f"--output {expected_oscar_admission}" in admission[0]


@pytest.mark.parametrize(
    "variable",
    (
        "SGLANG_DSV4_INT4_KV_STORAGE",
        "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE",
        "SGLANG_DSV4_SM86_C128_BF16_STORAGE",
    ),
)
def test_opencode_launcher_rejects_non_oscar_kv_layouts(
    tmp_path: Path, variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment[variable] = "1"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "conflicts with mandatory OSCAR-INT2 KV storage" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_disabling_oscar_int2(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_INT2_KV_STORAGE"] = "0"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_disabling_oscar_split_history(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY"] = "0"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_unadmitted_model_id(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_OSCAR_MODEL_ID"] = "local/unadmitted-model"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert (
        "requires DSV4_OSCAR_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash"
        in result.stderr
    )
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_calibration_capture_config(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_CAPTURE_CONFIG"] = str(
        tmp_path / "runtime_capture_config.json"
    )

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "calibration-only" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_non_fp8_oscar_carrier(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_KV_CACHE_DTYPE"] = "bfloat16"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires DSV4_KV_CACHE_DTYPE=fp8_e4m3" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


@pytest.mark.parametrize(
    "variable",
    ("DSV4_CONTEXT_LENGTH", "DSV4_MAX_TOTAL_TOKENS"),
)
def test_opencode_launcher_rejects_non_524k_capacity(
    tmp_path: Path, variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment[variable] = "8192"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert f"requires {variable}=524288" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_disabling_decode_graphs(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DECODE_GRAPH_BACKEND"] = "disabled"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires decode CUDA graphs" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_disabling_speculative_graphs(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DISABLE_SPECULATIVE"] = "1"
    environment["DSV4_DSPARK_FIXED_VERIFY_LEN"] = ""

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires graph-backed DSpark speculation" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_eager_target_verification(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_TARGET_VERIFY_EAGER"] = "1"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires graph-backed target verification" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_missing_oscar_artifact(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_OSCAR_CALIBRATION_PATH"] = str(tmp_path / "missing.pt")

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "readable absolute, non-symlink calibration artifact" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_missing_checkpoint_fingerprint(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    fingerprint = (
        Path(environment["DSV4_CACHE_ROOT"])
        / "oscar-int2"
        / "checkpoint-fingerprint.json"
    )
    fingerprint.unlink()

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "non-symlink checkpoint fingerprint" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_symlinked_oscar_artifact(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    artifact = Path(environment["DSV4_CACHE_ROOT"]) / "oscar-int2" / "real.pt"
    artifact.write_bytes(b"real")
    symlink = tmp_path / "oscar.pt"
    symlink.symlink_to(artifact)
    environment["DSV4_OSCAR_CALIBRATION_PATH"] = str(symlink)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "readable absolute, non-symlink calibration artifact" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_explicit_amxint4_cpu_split_is_fail_closed(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    amx_weights = tmp_path / "amxint4"
    amx_weights.mkdir()
    (amx_weights / "experts.safetensors").touch()
    environment["DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH"] = str(amx_weights)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert f"--kt-weight-path {amx_weights} --kt-method AMXINT4" in launch
    assert (
        "|split_amx=1|draft_method=MXFP4|draft_weights=/tmp/dsv4-local-checkpoint-0731"
    ) in launch


def test_opencode_launcher_rejects_amxint4_split_without_artifact(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    empty_weights = tmp_path / "empty-amxint4"
    empty_weights.mkdir()
    environment["DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH"] = str(empty_weights)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "contains no top-level safetensors artifact" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_amxint4_split_with_hotspot(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    amx_weights = tmp_path / "amxint4"
    amx_weights.mkdir()
    (amx_weights / "experts.safetensors").touch()
    environment["DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH"] = str(amx_weights)
    environment["DSV4_HOTSPOT_EXPERT_CACHE"] = "1"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "incompatible with the MXFP4 hotspot cache" in result.stderr
    assert "-m sglang.launch_server" not in invocation_log.read_text(encoding="utf-8")


def test_opencode_launcher_preserves_explicit_dspark_sps_table(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    fine_tier_table = (
        REPOSITORY_ROOT / "scripts" / "data" / "dsv4_flash_tp2_finetiers_sps.json"
    )
    environment["DSV4_SPS_TABLE"] = str(fine_tier_table)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert f"--speculative-dspark-sps-table-path {fine_tier_table}" in launch
    assert "dsv4_flash_fwuff_sm86_sps.json" not in launch


def test_opencode_launcher_preserves_explicit_fixed_verify_tier(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DSPARK_FIXED_VERIFY_LEN"] = "3"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--speculative-dspark-fixed-verify-len 3" in launch
    assert "--speculative-dspark-fixed-verify-len 4" not in launch


def test_opencode_launcher_explicit_empty_fixed_tier_restores_adaptive_policy(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DSPARK_FIXED_VERIFY_LEN"] = ""

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--speculative-dspark-fixed-verify-len" not in launch


def test_opencode_launcher_rejects_fixed_tier_past_verify_window(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DSPARK_FIXED_VERIFY_LEN"] = "7"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "cannot exceed DSV4_DSPARK_BLOCK_SIZE + 1" in result.stderr
    assert "-m sglang.launch_server" not in invocation_log.read_text(encoding="utf-8")


def test_opencode_launcher_rejects_unsupported_shared_expert_fusion(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_SHARED_EXPERTS_FUSION"] = "enforced"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "unsupported by the KTransformers EP launcher" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_rejects_unknown_shared_expert_fusion_mode(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_SHARED_EXPERTS_FUSION"] = "maybe"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "must be disabled, enforced, or auto" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_opencode_launcher_allows_explicit_eager_prefill_fallback(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_PREFILL_GRAPH_BACKEND"] = "disabled"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--cuda-graph-backend-prefill disabled" in launch


def test_opencode_launcher_allows_explicit_server_warmup_opt_out(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_SKIP_SERVER_WARMUP"] = "1"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--skip-server-warmup" in launch


def test_opencode_launcher_allows_realistic_shape_warmup_opt_out(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_OPENCODE_WARMUPS"] = ""

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--warmups" not in launch


def test_opencode_launcher_rejects_invalid_server_warmup_setting(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_SKIP_SERVER_WARMUP"] = "sometimes"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "DSV4 boolean settings must be 0 or 1" in result.stderr
    assert "-m sglang.launch_server" not in invocation_log.read_text(encoding="utf-8")


def test_opencode_launcher_rejects_the_known_unsafe_breakable_boundary(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_EAGER_ATTN_MODULE_IN_BCG"] = "0"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "breakable DSV4 prefill graphs require" in result.stderr
    assert "-m sglang.launch_server" not in invocation_log.read_text(encoding="utf-8")


def test_opencode_launcher_rejects_attention_capture_in_breakable_prefill(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_CAPTURE_ATTN_IN_BCG"] = "1"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "breakable DSV4 prefill graphs require" in result.stderr
    assert "-m sglang.launch_server" not in invocation_log.read_text(encoding="utf-8")


def test_opencode_launcher_materializes_qualified_frozen_target_plan(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not target_plan_builds(invocations)
    materializations = frozen_target_materializations(invocations)
    assert len(materializations) == 1
    assert f"--manifest {FROZEN_TARGET_MANIFEST}" in materializations[0]
    frozen_plan = tmp_path / "cache" / "opencode-g14-p28-frozen.pt"
    assert f"--output {frozen_plan}" in materializations[0]
    assert (
        f"|target_plan={frozen_plan}|target_sglang_plan={frozen_plan}"
        in (invocations[-1])
    )


def test_opencode_launcher_preserves_explicit_target_profile(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    explicit_profile = tmp_path / "explicit-target-profile.json"
    environment["DSV4_HYBRID_EXPERT_PROFILE"] = str(explicit_profile)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    target_builds = target_plan_builds(invocations)
    assert len(target_builds) == 1
    assert f"--profile {explicit_profile}" in target_builds[0]
    assert "--gpu-rank-counts 14,14" in target_builds[0]
    assert "--cpu-rank-counts 114,114" in target_builds[0]
    assert not frozen_target_materializations(invocations)
    assert str(TARGET_SPARSE_PROFILE) not in "\n".join(invocations)


@pytest.mark.parametrize(
    "plan_variable",
    [
        "DSV4_HYBRID_EXPERT_SHARD_PLAN",
        "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
    ],
)
def test_opencode_launcher_preserves_explicit_target_plan(
    tmp_path: Path, plan_variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    explicit_plan = tmp_path / "explicit-target-plan.pt"
    explicit_plan.write_bytes(b"prebuilt-target-plan-sentinel")
    environment[plan_variable] = str(explicit_plan)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not target_plan_builds(invocations)
    assert not frozen_target_materializations(invocations)
    assert str(TARGET_SPARSE_PROFILE) not in "\n".join(invocations)
    prepare = invocations[-1]
    assert f"|target_plan={explicit_plan}|target_sglang_plan={explicit_plan}" in prepare


@pytest.mark.parametrize("override_variable", TARGET_OVERRIDE_VARIABLES)
def test_opencode_launcher_treats_explicit_empty_target_input_as_an_override(
    tmp_path: Path, override_variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment[override_variable] = ""

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert str(TARGET_SPARSE_PROFILE) not in "\n".join(invocations)


def test_opencode_launcher_stages_combined_cpu_overlay_by_default(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    stage_invocations = [line for line in invocations if line.startswith("stage=")]
    assert stage_invocations == [
        (
            "stage="
            f"{REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'} "
            "--candidate /var/lib/exo/experiments/"
            "dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/"
            "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so "
            "--expected-sha256 "
            "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043 "
            "--cache-root /var/lib/exo/cache/dsv4-cpu-optimized-serving-overlays"
        )
    ]
    assert f"|kt_source={environment['DSV4_TEST_OVERLAY_ROOT']}" in invocations[-1]
    assert (
        "|task_queue_pin=1|inline_dispatch=1|scale_fold=lut-v1|worker_spin=1000"
        in invocations[-1]
    )


def test_opencode_launcher_rejects_explicit_empty_ktransformers_source(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_KTRANSFORMERS_SOURCE"] = ""

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be a non-empty explicit path" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_stages_hash_pinned_cpu_optimized_opt_in(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    candidate = tmp_path / "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
    candidate.write_bytes(b"the fake stage process owns hash verification")
    cache_root = tmp_path / "cpu-optimized-cache"
    environment.update(
        {
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
            "DSV4_KT_CPU_OPTIMIZED_CANDIDATE": str(candidate),
            "DSV4_KT_CPU_OPTIMIZED_CACHE_ROOT": str(cache_root),
        }
    )

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    stage_invocations = [line for line in invocations if line.startswith("stage=")]
    assert stage_invocations == [
        (
            "stage="
            f"{REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'} "
            f"--candidate {candidate} "
            "--expected-sha256 "
            "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043 "
            f"--cache-root {cache_root}"
        )
    ]
    assert f"|kt_source={environment['DSV4_TEST_OVERLAY_ROOT']}" in invocations[-1]
    assert (
        "|task_queue_pin=1|inline_dispatch=1|scale_fold=lut-v1|worker_spin=1000"
        in invocations[-1]
    )
    assert "--kt-cpuinfer 56" in invocations[-1]


def test_opencode_launcher_rejects_rejected_cpu_optimized_72_thread_arm(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
            "DSV4_CPUINFER_THREADS": "72",
        }
    )

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires the qualified DSV4_CPUINFER_THREADS=56" in result.stderr
    assert not invocation_log.exists()


@pytest.mark.parametrize("thread_count", ["1", "55", "72", "73", "112"])
def test_opencode_launcher_rejects_unadmitted_cpu_optimized_thread_counts(
    tmp_path: Path, thread_count: str
) -> None:
    environment, _invocation_log = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
            "DSV4_CPUINFER_THREADS": thread_count,
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires the qualified DSV4_CPUINFER_THREADS=56" in result.stderr


@pytest.mark.parametrize("spin_microseconds", ["0", "1", "999", "1001"])
def test_opencode_launcher_rejects_unqualified_cpu_optimized_spin_policy(
    tmp_path: Path, spin_microseconds: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["KT_WORKER_SPIN_US"] = spin_microseconds

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires the qualified KT_WORKER_SPIN_US=1000" in result.stderr
    assert not invocation_log.exists()


@pytest.mark.parametrize("scale_fold_mode", ["off", "exponent-v1"])
def test_opencode_launcher_rejects_unqualified_cpu_optimized_scale_fold_mode(
    tmp_path: Path, scale_fold_mode: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["KT_MXFP4_AVX_SCALE_FOLD_MODE"] = scale_fold_mode

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires qualified KT_MXFP4_AVX_SCALE_FOLD_MODE=lut-v1" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_invalid_scale_fold_mode(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["KT_MXFP4_AVX_SCALE_FOLD_MODE"] = "lut-v2"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert (
        "KT_MXFP4_AVX_SCALE_FOLD_MODE must be off, lut-v1, or exponent-v1"
        in result.stderr
    )
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_unbound_nonoff_scale_fold_mode(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["KT_MXFP4_AVX_SCALE_FOLD_MODE"] = "lut-v1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires the hash-pinned CPU-optimized overlay" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_combined_and_legacy_avx_overlays(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_AVX_TAIL_OVERLAY"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "combined CPU-optimized and legacy AVX-tail overlays" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_cpu_optimized_with_explicit_source(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "1"
    environment["DSV4_KTRANSFORMERS_SOURCE"] = str(tmp_path / "explicit-source")

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with DSV4_KTRANSFORMERS_SOURCE" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_unbound_inline_dispatch_env(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["KT_SINGLE_NUMA_INLINE_DISPATCH"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires the hash-pinned CPU-optimized overlay" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_stages_hash_pinned_task_queue_pin_opt_in(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    candidate = tmp_path / "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
    candidate.write_bytes(b"the fake stage process owns hash verification")
    cache_root = tmp_path / "task-queue-pin-cache"
    environment.update(
        {
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "0",
            "DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY": "1",
            "DSV4_KT_TASK_QUEUE_PIN_CANDIDATE": str(candidate),
            "DSV4_KT_TASK_QUEUE_PIN_CACHE_ROOT": str(cache_root),
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    stage_invocations = [line for line in invocations if line.startswith("stage=")]
    assert stage_invocations == [
        (
            "stage="
            f"{REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'} "
            f"--candidate {candidate} "
            "--expected-sha256 "
            "cfe7aaf328f71fc50aac877b56ee474e07b0f06ecc8571736ba3cc1e8e3786dd "
            f"--cache-root {cache_root}"
        )
    ]
    assert f"|kt_source={environment['DSV4_TEST_OVERLAY_ROOT']}" in invocations[-1]
    assert "|task_queue_pin=1" in invocations[-1]


def test_opencode_launcher_rejects_task_queue_pin_with_explicit_source(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY"] = "1"
    environment["DSV4_KTRANSFORMERS_SOURCE"] = str(tmp_path / "explicit-source")

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with DSV4_KTRANSFORMERS_SOURCE" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_unbound_task_queue_pin_env(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["KT_TASK_QUEUE_PIN_FIRST_CORE"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires a hash-pinned task-queue-capable overlay" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_overlapping_kt_experiment_overlays(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY"] = "1"
    environment["DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_cpu_optimized_with_another_experiment(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "1"
    environment["DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_stages_hash_pinned_persistent_counter_opt_in(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    candidate = tmp_path / "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
    candidate.write_bytes(b"the fake stage process owns hash verification")
    python_source = tmp_path / "experts_base.py"
    python_source.write_bytes(b"the fake stage process owns Python hash verification")
    cache_root = tmp_path / "persistent-counter-cache"
    environment.update(
        {
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "0",
            "DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY": "1",
            "DSV4_KT_PERSISTENT_COUNTER_CANDIDATE": str(candidate),
            "DSV4_KT_PERSISTENT_COUNTER_PYTHON_SOURCE": str(python_source),
            "DSV4_KT_PERSISTENT_COUNTER_CACHE_ROOT": str(cache_root),
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    stage_invocations = [line for line in invocations if line.startswith("stage=")]
    assert stage_invocations == [
        (
            "stage="
            f"{REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'} "
            f"--candidate {candidate} "
            "--expected-sha256 "
            "a40796696a1181bb680379a94d80753345b8f4d979e865b9a49e06ee1c965373 "
            f"--python-source {python_source} "
            "--expected-python-sha256 "
            "b6aaa020bba9a326e2e9191791d79b9c88ff6f429c84ebae80a31df8e37257bf "
            f"--cache-root {cache_root}"
        )
    ]
    assert f"|kt_source={environment['DSV4_TEST_OVERLAY_ROOT']}" in invocations[-1]


def test_opencode_launcher_rejects_persistent_counter_with_explicit_source(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY"] = "1"
    environment["DSV4_KTRANSFORMERS_SOURCE"] = str(tmp_path / "explicit-source")

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with DSV4_KTRANSFORMERS_SOURCE" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_preserves_explicit_ktransformers_source(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    explicit_source = tmp_path / "explicit-ktransformers"
    explicit_source.mkdir()
    environment["DSV4_KTRANSFORMERS_SOURCE"] = str(explicit_source)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("stage=") for line in invocations)
    assert f"|kt_source={explicit_source}" in invocations[-1]


def test_opencode_launcher_allows_explicit_all_overlay_opt_out(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["DSV4_STAGE_KT_AVX_TAIL_OVERLAY"] = "0"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("stage=") for line in invocations)
    assert "|kt_source=|split_amx=|draft_method=|draft_weights=" in invocations[-1]


def test_opencode_launcher_keeps_hash_pinned_avx_tail_as_explicit_rollback(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "0"
    environment["DSV4_STAGE_KT_AVX_TAIL_OVERLAY"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert [line for line in invocations if line.startswith("stage=")] == [
        f"stage={REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'}"
    ]
    assert "|inline_dispatch=0|scale_fold=off|worker_spin=1000" in invocations[-1]


def test_opencode_launcher_fails_closed_when_default_overlay_staging_fails(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_TEST_STAGE_FAIL"] = "1"

    result = run_launcher(environment)

    assert result.returncode == 19
    assert "synthetic overlay staging failure" in result.stderr
    assert invocation_log.read_text(encoding="utf-8").splitlines() == [
        (
            "stage="
            f"{REPOSITORY_ROOT / 'scripts' / 'stage_dsv4_kt_avx_tail_overlay.py'} "
            "--candidate /var/lib/exo/experiments/"
            "dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/"
            "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so "
            "--expected-sha256 "
            "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043 "
            "--cache-root /var/lib/exo/cache/dsv4-cpu-optimized-serving-overlays"
        )
    ]


def test_opencode_launcher_rejects_invalid_overlay_staging_boolean(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_AVX_TAIL_OVERLAY"] = "sometimes"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_STAGE_KT_AVX_TAIL_OVERLAY must be a boolean value" in result.stderr
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_invalid_task_queue_overlay_boolean(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY"] = "sometimes"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert (
        "DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY must be a boolean value"
        in result.stderr
    )
    assert not invocation_log.exists()


def test_opencode_launcher_rejects_invalid_cpu_optimized_overlay_boolean(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"] = "sometimes"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert (
        "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY must be a boolean value"
        in result.stderr
    )
    assert not invocation_log.exists()


def test_opencode_launcher_preserves_explicit_qualified_setting_overrides(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_DSPARK_BLOCK_SIZE": "6",
            "DSV4_GPU_EXPERTS_PER_LAYER": "12",
            "DSV4_GPU_EXPERTS_MAX_PER_LAYER": "22",
            "DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER": "9",
            "KT_MXFP4_AMX_MIN_EXPERT_TOKENS": "8",
            "KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS": "4",
            "SGLANG_DSPARK_FP32_LM_HEAD": "1",
            "SGLANG_DSPARK_OPT_MARKOV_W2_BF16": "1",
            "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "0",
            "SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM": "0",
            "SGLANG_V4_MXFP4_SMALL_ROW_ROUTING": "0",
        }
    )

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--kt-num-gpu-experts 22" in launch
    assert "--speculative-dspark-block-size 6" in launch
    assert "|gpu=12|prefix=9|" in launch
    assert (
        "|multi_stream=0|lm_fp32=1|markov_bf16=1|amx_min=8|avx_min=4|"
        "small_row=0|small_gemm=0|"
    ) in launch


def test_opencode_launcher_builds_repo_owned_draft_hot_profile(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    draft_builds = draft_plan_builds(invocations)
    assert len(draft_builds) == 1
    assert f"--profile {DRAFT_SPARSE_PROFILE}" in draft_builds[0]
    assert "--gpu-rank-counts 14,14" in draft_builds[0]
    assert "--cpu-rank-counts 114,114" in draft_builds[0]
    assert "--gpu-selection profile-hot" in draft_builds[0]
    assert str(tmp_path / "cache" / "draft-hybrid-gpu14-ep2.pt") in draft_builds[0]
    assert "/tmp/dsv4-draft-hot-g12-ep2.pt" not in "\n".join(invocations)


def test_opencode_launcher_preserves_explicit_draft_profile(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    explicit_profile = tmp_path / "explicit-profile.json"
    environment["DSV4_DRAFT_HYBRID_EXPERT_PROFILE"] = str(explicit_profile)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    draft_builds = draft_plan_builds(invocations)
    assert len(draft_builds) == 1
    assert f"--profile {explicit_profile}" in draft_builds[0]
    assert str(DRAFT_SPARSE_PROFILE) not in "\n".join(invocations)


def test_opencode_launcher_preserves_explicit_draft_recorder(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_DRAFT_EXPERT_RECORDER_PROFILES"] = "rank0.pt rank1.pt"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert any("build_dsv4_decode_call_profile.py" in line for line in invocations)
    assert len(draft_plan_builds(invocations)) == 1
    assert str(DRAFT_SPARSE_PROFILE) not in "\n".join(invocations)


@pytest.mark.parametrize(
    "plan_variable",
    [
        "DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
        "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
    ],
)
def test_opencode_launcher_preserves_explicit_draft_plan(
    tmp_path: Path, plan_variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    explicit_plan = tmp_path / "explicit-plan.pt"
    explicit_plan.write_bytes(b"prebuilt-plan-sentinel")
    environment[plan_variable] = str(explicit_plan)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not draft_plan_builds(invocations)
    assert str(DRAFT_SPARSE_PROFILE) not in "\n".join(invocations)
    prepare = invocations[-1]
    assert (f"|draft_plan={explicit_plan}|draft_sglang_plan={explicit_plan}") in prepare


@pytest.mark.parametrize("override_variable", DRAFT_OVERRIDE_VARIABLES)
def test_opencode_launcher_treats_explicit_empty_draft_input_as_an_override(
    tmp_path: Path, override_variable: str
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment[override_variable] = ""

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert not draft_plan_builds(invocations)
    assert str(DRAFT_SPARSE_PROFILE) not in "\n".join(invocations)
