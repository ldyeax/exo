from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "dsv4_flash_hybrid_pp2_dwagon_opencode.sh"
BASE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "dsv4_flash_0731_tp2_dwagon.sh"
TARGET_SPARSE_PROFILE = (
    REPOSITORY_ROOT
    / "scripts"
    / "data"
    / "dsv4_flash_opencode_target_distinct_decode_calls_sparse.json"
)
PRODUCTION_CONTROLLER_ROOT = Path("/var/lib/exo/cache/dsv4-flash-hybrid-pp2-opencode")


def isolated_launcher(tmp_path: Path) -> tuple[Path, Path]:
    """Bake a test controller root into a launcher copy without an env bypass."""

    controller_root = tmp_path / "cache"
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    production_root_assignment = (
        f"readonly canonical_cache_root={PRODUCTION_CONTROLLER_ROOT}"
    )
    test_root_assignment = (
        f"readonly canonical_cache_root={shlex.quote(str(controller_root))}"
    )
    repository_assignment = 'repo_root="$(cd "${script_dir}/.." && pwd)"'
    test_repository_assignment = f"repo_root={shlex.quote(str(REPOSITORY_ROOT))}"
    assert launcher_source.count(production_root_assignment) == 1
    assert launcher_source.count(repository_assignment) == 1
    launcher_source = launcher_source.replace(
        production_root_assignment, test_root_assignment
    ).replace(repository_assignment, test_repository_assignment)
    launcher_copy = tmp_path / "dsv4_flash_hybrid_pp2_test_launcher.sh"
    launcher_copy.write_text(launcher_source, encoding="utf-8")
    launcher_copy.chmod(0o755)
    return launcher_copy, controller_root


def launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    launcher_copy, controller_root = isolated_launcher(tmp_path)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DSV4_")
        and name
        not in {
            "SGLANG_KT_CPU_EXPERT_SHARD_PLAN",
            "SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_KT_GPU_EXPERT_MASK_PLAN",
            "SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN",
            "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
            "SGLANG_PP_LAYER_PARTITION",
            "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE",
            "SGLANG_DSV4_INT4_KV_STORAGE",
            "SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH",
            "SGLANG_DSV4_OSCAR_CALIBRATION_PATH",
            "SGLANG_DSV4_OSCAR_CAPTURE_CONFIG",
            "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE",
            "SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY",
            "SGLANG_DSV4_SM86_C128_BF16_STORAGE",
            "KT_MXFP4_AMX_MIN_EXPERT_TOKENS",
            "KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS",
        }
    }
    invocation_log = tmp_path / "python-invocations.txt"
    staged_kt_source = tmp_path / "staged-kt-overlay"
    (staged_kt_source / "kt_kernel").mkdir(parents=True)
    transferred_pp2_plan = tmp_path / "transferred-pp2-winner.pt"
    transferred_pp2_plan.write_bytes(b"fake transferred PP2 plan")
    source_ep2_winner = tmp_path / "qualified-ep2-winner.pt"
    source_ep2_winner.write_bytes(b"fake qualified EP2 winner")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'args=%s|tp=%s|pp=%s|ep=%s|spec=%s|ctx=%s|max=%s|kv=%s|swa=%s|"
        "chunk=%s|prefill=%s|decode=%s|running=%s|micro=%s|async=%s|"
        "partition=%s|gpus=%s|cpuinfer=%s|pools=%s|numa=%s|experts=%s|"
        "max_experts=%s|plan=%s|"
        "kt_source=%s|model=%s|amx_min=%s|avx_min=%s|multi_stream=%s|"
        "small_row=%s|small_batch=%s|oscar=%s|split_history=%s|oscar_path=%s|oscar_admission=%s|"
        "c4_int4=%s|kv_int4=%s|c128_bf16=%s|task_pin=%s|inline=%s|"
        "scale_fold=%s|native_candidate=%s|native_sha=%s|run_role=%s|auth=%s\\n' "
        '"$*" "${DSV4_TENSOR_PARALLEL_SIZE-}" '
        '"${DSV4_PIPELINE_PARALLEL_SIZE-}" "${DSV4_EXPERT_PARALLEL_SIZE-}" '
        '"${DSV4_DISABLE_SPECULATIVE-}" "${DSV4_CONTEXT_LENGTH-}" '
        '"${DSV4_MAX_TOTAL_TOKENS-}" "${DSV4_KV_CACHE_DTYPE-}" '
        '"${DSV4_SWA_FULL_TOKENS_RATIO-}" "${DSV4_CHUNKED_PREFILL_SIZE-}" '
        '"${DSV4_PREFILL_GRAPH_BACKEND-}" '
        '"${DSV4_DECODE_GRAPH_BACKEND-}" "${DSV4_MAX_RUNNING_REQUESTS-}" '
        '"${DSV4_PP_MAX_MICRO_BATCH_SIZE-}" "${DSV4_PP_ASYNC_BATCH_DEPTH-}" '
        '"${DSV4_PIPELINE_LAYER_PARTITION-}" "${DSV4_CUDA_VISIBLE_DEVICES-}" '
        '"${DSV4_CPUINFER_THREADS-}" "${DSV4_KT_THREADPOOL_COUNT-}" '
        '"${DSV4_KT_NUMA_NODES-}" '
        '"${DSV4_GPU_EXPERTS_PER_LAYER-}" '
        '"${DSV4_GPU_EXPERTS_MAX_PER_LAYER-}" '
        '"${SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN-}" '
        '"${DSV4_KTRANSFORMERS_SOURCE-}" '
        '"${DSV4_MODEL_PATH-}" '
        '"${KT_MXFP4_AMX_MIN_EXPERT_TOKENS-}" '
        '"${KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS-}" '
        '"${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP-}" '
        '"${SGLANG_V4_MXFP4_SMALL_ROW_ROUTING-}" '
        '"${SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM-}" '
        '"${SGLANG_DSV4_OSCAR_INT2_KV_STORAGE-}" '
        '"${SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY-}" '
        '"${SGLANG_DSV4_OSCAR_CALIBRATION_PATH-}" '
        '"${SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH-}" '
        '"${SGLANG_DSV4_INT4_C4_INDEXER_STORAGE-}" '
        '"${SGLANG_DSV4_INT4_KV_STORAGE-}" '
        '"${SGLANG_DSV4_SM86_C128_BF16_STORAGE-}" '
        '"${KT_TASK_QUEUE_PIN_FIRST_CORE-}" '
        '"${KT_SINGLE_NUMA_INLINE_DISPATCH-}" '
        '"${KT_MXFP4_AVX_SCALE_FOLD_MODE-}" '
        '"${DSV4_KT_CPU_OPTIMIZED_CANDIDATE-}" '
        '"${DSV4_KT_CPU_OPTIMIZED_SHA256-}" '
        '"${DSV4_PP_RUN_ROLE-}" '
        '"${DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT-}" '
        '>>"$DSV4_TEST_PYTHON_INVOCATIONS"\n'
        "if [[ ${1:-} == */materialize_dsv4_frozen_hybrid_plan.py ]]; then\n"
        "  shift\n"
        "  output=\n"
        "  while (( $# > 0 )); do\n"
        "    if [[ $1 == --output && $# -ge 2 ]]; then output=$2; break; fi\n"
        "    shift\n"
        "  done\n"
        "  [[ -n $output ]]\n"
        '  mkdir -p "$(dirname "$output")"\n'
        "  printf 'fake frozen EP2 plan' >\"$output\"\n"
        "elif [[ ${1:-} == */transfer_dsv4_ep2_plan_to_pp2.py ]]; then\n"
        "  printf '%s\\n' \"$DSV4_TEST_PP_TRANSFERRED_PLAN\"\n"
        "elif [[ ${1:-} == */stage_dsv4_kt_avx_tail_overlay.py ]]; then\n"
        "  printf '%s\\n' \"$DSV4_TEST_KT_OVERLAY_SOURCE\"\n"
        "elif [[ ${1:-} == */dsv4_pp2_followup_ledger.py ]]; then\n"
        "  while (($#)); do\n"
        "    if [[ $1 == --output ]]; then\n"
        '      mkdir -p "${2%/*}"\n'
        '      printf \'{"format":"test-pp2-authorization"}\\n\' >"$2"\n'
        "      printf '%s\\n' \"$2\"\n"
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "elif [[ ${1:-} == */dsv4_oscar_int2_calibration.py && ${2:-} == admit ]]; then\n"
        "  while (($#)); do\n"
        "    if [[ $1 == --output ]]; then\n"
        '      mkdir -p "${2%/*}"\n'
        '      printf \'{"format":"test-oscar-admission"}\\n\' >"$2"\n'
        "      break\n"
        "    fi\n"
        "    shift\n"
        "  done\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    oscar_artifact = tmp_path / "dsv4-oscar-int2-calibration.pt"
    oscar_artifact.write_bytes(b"synthetic launcher-only OSCAR artifact")
    local_model = tmp_path / "local-checkpoint"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "checkpoint-fingerprint.json").write_text("{}\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_ep2_winner.read_bytes()).hexdigest()
    ep_confirmation = tmp_path / "final-ep-confirmation.json"
    ep_confirmation.write_text(
        json.dumps(
            {
                "format": "dsv4_candidate_campaign_result_v1",
                "stage": "confirm",
                "qualified": True,
                "shutdown_method": "sigterm",
                "residual_compute_pids": [],
                "coherency": {
                    "coherent": True,
                    "deterministic_final_content": True,
                    "forced_tool_call": {"accepted": True},
                    "semantic_runs": [{"accepted": True}, {"accepted": True}],
                },
                "environment": {
                    "DSV4_TENSOR_PARALLEL_SIZE": "2",
                    "DSV4_PIPELINE_PARALLEL_SIZE": "1",
                    "DSV4_EXPERT_PARALLEL_SIZE": "2",
                    "DSV4_CONTEXT_LENGTH": "524288",
                    "DSV4_MAX_TOTAL_TOKENS": "524288",
                    "DSV4_KV_CACHE_DTYPE": "fp8_e4m3",
                    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE": "1",
                    "SGLANG_DSV4_INT4_KV_STORAGE": "0",
                    "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE": "0",
                    "SGLANG_DSV4_SM86_C128_BF16_STORAGE": "0",
                    "DSV4_CPUINFER_THREADS": "56",
                    "KT_WORKER_SPIN_US": "1000",
                    "KT_TASK_QUEUE_PIN_FIRST_CORE": "1",
                    "KT_SINGLE_NUMA_INLINE_DISPATCH": "1",
                    "KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1",
                    "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
                    "DSV4_KT_CPU_OPTIMIZED_CANDIDATE": (
                        "/var/lib/exo/experiments/"
                        "dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/"
                        "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
                    ),
                },
                "kt_single_numa_inline_dispatch_server_proof": {
                    "kt_single_numa_inline_dispatch_all_workers_active": True,
                    "kt_single_numa_inline_dispatch_rank_coverage_valid": True,
                },
                "kt_mxfp4_avx_scale_fold_server_proof": {
                    "kt_mxfp4_avx_scale_fold_all_workers_active": True,
                    "kt_mxfp4_avx_scale_fold_rank_coverage_valid": True,
                    "kt_mxfp4_avx_scale_fold_expected_n_block": 128,
                    "kt_mxfp4_avx_scale_fold_requested_mode": "lut-v1",
                    "workers": [
                        {"telemetry": {"n_block": 128, "lut_hash": "06d1a83dbf20f545"}},
                        {"telemetry": {"n_block": 128, "lut_hash": "06d1a83dbf20f545"}},
                    ],
                },
                "oscar_contract": {
                    "expected_server_info": {
                        "dsv4_oscar_int2_kv_storage": True,
                        "dsv4_kv_storage_mode": (
                            "oscar_int2_asymmetric+protected_swa_bfloat16"
                        ),
                        "dsv4_c4_kv_bytes_per_token": 272,
                        "dsv4_c128_kv_bytes_per_token": 272,
                        "dsv4_c4_indexer_bytes_per_token": 40,
                        "dsv4_int4_kv_storage": False,
                        "dsv4_int4_c4_indexer_storage": False,
                        "dsv4_sm86_c128_bf16_storage": False,
                    }
                },
                "plan": {
                    "path": str(source_ep2_winner),
                    "sha256": source_sha256,
                    "placement_semantics_sha256": "a" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ep_coherency = tmp_path / "final-ep-coherency.json"
    ep_coherency.write_text("{}\n", encoding="utf-8")
    environment.update(
        {
            "PP2_TEST_LAUNCHER_PATH": str(launcher_copy),
            "DSV4_CACHE_ROOT": str(controller_root),
            "DSV4_PYTHON": str(fake_python),
            "DSV4_TEST_PYTHON_INVOCATIONS": str(invocation_log),
            "DSV4_TEST_KT_OVERLAY_SOURCE": str(staged_kt_source),
            "DSV4_TEST_PP_TRANSFERRED_PLAN": str(transferred_pp2_plan),
            "DSV4_OSCAR_CALIBRATION_PATH": str(oscar_artifact),
            "DSV4_MODEL_PATH": str(local_model),
            "DSV4_KT_WEIGHT_PATH": str(local_model),
            "DSV4_PP_EP2_WINNER_PLAN": str(source_ep2_winner),
            "DSV4_PP_EP_WINNER_RECEIPT": str(ep_confirmation),
            "DSV4_PP_EP_COHERENCY_RECEIPT": str(ep_coherency),
            "DSV4_PP_LAUNCH_LEDGER": str(
                controller_root / "pp2-two-launch-ledger.json"
            ),
            "DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT": str(
                controller_root / "authorization-transfer.json"
            ),
        }
    )
    return environment, invocation_log, fake_python


def run_launcher(
    environment: dict[str, str], arguments: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", environment["PP2_TEST_LAUNCHER_PATH"], *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def install_fake_launch_tools(tmp_path: Path, environment: dict[str, str]) -> None:
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    fake_ss = binary_directory / "ss"
    fake_ss.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_ss.chmod(0o755)
    fake_nvidia_smi = binary_directory / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '24000\\n24000\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_numactl = binary_directory / "numactl"
    fake_numactl.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nshift 2\nexec "$@"\n',
        encoding="utf-8",
    )
    fake_numactl.chmod(0o755)
    environment["PATH"] = f"{binary_directory}:{environment['PATH']}"


def test_pp2_launcher_prepares_concurrency_control_without_starting_model(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert "no model process was started" in result.stdout
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 3
    assert "transfer_dsv4_ep2_plan_to_pp2.py" in invocations[0]
    assert "--source-ep-confirmation-receipt" in invocations[0]
    assert "--source-ep-coherency-receipt" in invocations[0]
    assert "--expected-target-gpu-experts-per-layer 28" in invocations[0]
    assert not any("build_dsv4_kt_hybrid_shard_plan.py" in line for line in invocations)
    assert "stage_dsv4_kt_avx_tail_overlay.py" in invocations[1]
    assert (
        "--candidate /var/lib/exo/experiments/"
        "dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/"
        "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
    ) in invocations[1]
    assert (
        "--expected-sha256 "
        "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043"
    ) in invocations[1]
    prepare = invocations[2]
    assert "prepare_dsv4_flash_0731.py" in prepare
    assert "|tp=1|pp=2|ep=1|spec=1|ctx=524288|max=524288|kv=fp8_e4m3|" in prepare
    assert "|swa=0.0048828125|chunk=1024|prefill=disabled|decode=full|" in prepare
    assert "|running=2|micro=1|async=0|partition=21,22|gpus=0,1|" in prepare
    assert "|cpuinfer=56|pools=2|numa=0 1|experts=28|max_experts=28|plan=" in prepare
    assert f"|kt_source={tmp_path / 'staged-kt-overlay'}|" in prepare
    assert f"|model={environment['DSV4_MODEL_PATH']}|" in prepare
    assert "|amx_min=5|avx_min=2|multi_stream=0" in prepare
    assert (
        f"|small_row=1|small_batch=1|oscar=1|split_history=1|"
        f"oscar_path={environment['DSV4_OSCAR_CALIBRATION_PATH']}|"
        f"oscar_admission={tmp_path / 'admission.json'}|"
        "c4_int4=0|kv_int4=0|c128_bf16=0"
    ) in prepare
    assert "|task_pin=1|inline=1|scale_fold=lut-v1|" in prepare
    assert (
        "|native_sha=7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043|"
    ) in prepare


def test_pp2_launcher_requires_final_ep_confirmation_receipt(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    del environment["DSV4_PP_EP_WINNER_RECEIPT"]

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_PP_EP_WINNER_RECEIPT" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_rejects_mounted_model_checkpoint(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_MODEL_PATH"] = "/mnt/sanic/llm_models/DeepSeek-V4-Flash"
    environment["DSV4_KT_WEIGHT_PATH"] = environment["DSV4_MODEL_PATH"]

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires an absolute, local, non-symlink model checkpoint" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_rejects_distinct_kt_weight_path(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    different_weights = tmp_path / "different-weights"
    different_weights.mkdir()
    environment["DSV4_KT_WEIGHT_PATH"] = str(different_weights)

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_KT_WEIGHT_PATH to equal" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_requires_final_ep_coherency_receipt(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    del environment["DSV4_PP_EP_COHERENCY_RECEIPT"]

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_PP_EP_COHERENCY_RECEIPT" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_rejects_explicit_kernel_source_bypass(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    explicit_kt_source = tmp_path / "explicit-kt-source"
    explicit_kt_source.mkdir()
    environment["DSV4_KTRANSFORMERS_SOURCE"] = str(explicit_kt_source)

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "cannot bypass the receipt-bound PP2 native artifact" in result.stderr
    assert "prepare_dsv4_flash_0731.py" not in invocation_log.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "variable",
    (
        "SGLANG_DSV4_INT4_KV_STORAGE",
        "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE",
        "SGLANG_DSV4_SM86_C128_BF16_STORAGE",
    ),
)
def test_pp2_launcher_rejects_non_oscar_kv_layouts(
    tmp_path: Path, variable: str
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment[variable] = "1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with mandatory OSCAR-INT2 KV storage" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_disabling_oscar_int2(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_INT2_KV_STORAGE"] = "0"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_disabling_oscar_split_history(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY"] = "0"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "requires SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_non_fp8_oscar_carrier(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_KV_CACHE_DTYPE"] = "bfloat16"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires DSV4_KV_CACHE_DTYPE=fp8_e4m3" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_unadmitted_model_id(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_OSCAR_MODEL_ID"] = "local/unadmitted-model"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires DSV4_OSCAR_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_calibration_capture_config(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["SGLANG_DSV4_OSCAR_CAPTURE_CONFIG"] = str(
        tmp_path / "runtime_capture_config.json"
    )

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "calibration-only" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_missing_oscar_artifact(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_OSCAR_CALIBRATION_PATH"] = str(tmp_path / "missing.pt")

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "readable, absolute, non-symlink EP2 calibration artifact" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_rejects_explicit_target_plan_bypass(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    explicit_plan = tmp_path / "target-plan.pt"
    explicit_plan.write_bytes(b"prebuilt-target-plan")
    environment["DSV4_PP_HYBRID_EXPERT_SHARD_PLAN"] = str(explicit_plan)

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with DSV4_PP_HYBRID_EXPERT_SHARD_PLAN" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_transfers_exact_ep2_winner_without_static_profile(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    source_winner = tmp_path / "fivefold-ep2-g13-winner.pt"
    source_winner.write_bytes(b"fake source winner validated by transfer helper")
    transfer_cache = tmp_path / "transferred-plan-cache"
    environment.update(
        {
            "DSV4_PP_EP2_WINNER_PLAN": str(source_winner),
            "DSV4_PP_TRANSFERRED_PLAN_CACHE_ROOT": str(transfer_cache),
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 3
    transfer_invocation = invocations[0]
    assert "transfer_dsv4_ep2_plan_to_pp2.py" in transfer_invocation
    assert f"--source-ep2-plan {source_winner}" in transfer_invocation
    assert f"--cache-root {transfer_cache}" in transfer_invocation
    assert "--expected-target-gpu-experts-per-layer 28" in transfer_invocation
    assert not any("build_dsv4_kt_hybrid_shard_plan.py" in line for line in invocations)
    assert str(TARGET_SPARSE_PROFILE) not in "\n".join(invocations)
    prepare = invocations[-1]
    assert f"|plan={environment['DSV4_TEST_PP_TRANSFERRED_PLAN']}|" in prepare
    assert "|experts=28|max_experts=28|" in prepare


def test_pp2_launcher_rejects_ambiguous_winner_and_explicit_pp2_plan(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    source_winner = tmp_path / "winner.pt"
    source_winner.write_bytes(b"winner")
    explicit_plan = tmp_path / "explicit-pp2.pt"
    explicit_plan.write_bytes(b"explicit")
    environment.update(
        {
            "DSV4_PP_EP2_WINNER_PLAN": str(source_winner),
            "DSV4_PP_HYBRID_EXPERT_SHARD_PLAN": str(explicit_plan),
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with DSV4_PP_HYBRID_EXPERT_SHARD_PLAN" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_rejects_winner_with_profile_based_inputs(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    source_winner = tmp_path / "winner.pt"
    source_winner.write_bytes(b"winner")
    environment.update(
        {
            "DSV4_PP_EP2_WINNER_PLAN": str(source_winner),
            "DSV4_PP_HYBRID_EXPERT_PROFILE": str(tmp_path / "stale-profile.pt"),
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "conflicts with PP2 profile-based placement inputs" in result.stderr
    assert not invocation_log.exists()


def test_pp2_launcher_accepts_g14_winner_transfer_with_g28_budget(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    source_winner = tmp_path / "g14-winner.pt"
    source_winner.write_bytes(b"winner")
    environment.update(
        {
            "DSV4_PP_EP2_WINNER_PLAN": str(source_winner),
            "DSV4_GPU_EXPERTS_PER_LAYER": "28",
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    transfer_invocation = invocations[0]
    assert "--expected-target-gpu-experts-per-layer 28" in transfer_invocation
    assert "|experts=28|max_experts=28|" in invocations[-1]


def test_pp2_launcher_accepts_variable_winner_with_odd_union_ceiling(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    source_winner = tmp_path / "winner.pt"
    source_winner.write_bytes(b"winner")
    environment.update(
        {
            "DSV4_PP_EP2_WINNER_PLAN": str(source_winner),
            "DSV4_GPU_EXPERTS_PER_LAYER": "27",
            "DSV4_GPU_EXPERTS_MAX_PER_LAYER": "250",
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert "--expected-target-gpu-experts-per-layer 27" in invocations[0]
    assert "|experts=27|max_experts=27|" in invocations[-1]


def test_pp2_launch_command_captures_bs1_and_bs2_decode_graphs_and_disables_speculation(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)
    environment["DSV4_PP_RUN_ROLE"] = "transfer"

    result = subprocess.run(
        ["bash", environment["PP2_TEST_LAUNCHER_PATH"], "--launch"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    admission = [
        line for line in invocations if "dsv4_oscar_int2_calibration.py admit" in line
    ]
    assert len(admission) == 1
    assert f"--artifact {environment['DSV4_OSCAR_CALIBRATION_PATH']}" in admission[0]
    assert f"--checkpoint {environment['DSV4_MODEL_PATH']}" in admission[0]
    assert "--model-id deepseek-ai/DeepSeek-V4-Flash" in admission[0]
    assert f"--output {tmp_path / 'admission.json'}" in admission[0]
    authorizations = [
        line for line in invocations if "dsv4_pp2_followup_ledger.py" in line
    ]
    assert len(authorizations) == 1
    assert "--run-role transfer" in authorizations[0]
    assert "--ep-coherency-receipt" in authorizations[0]
    assert f"--ledger {environment['DSV4_PP_LAUNCH_LEDGER']}" in authorizations[0]
    assert "--pipeline-layer-partition 21,22" in authorizations[0]
    assert "--pp-async-batch-depth 0" in authorizations[0]
    launch = invocations[-1]
    assert "-m sglang.launch_server" in launch
    assert "--tensor-parallel-size 1 --pp-size 2 --pp-max-micro-batch-size 1" in launch
    assert "--pp-async-batch-depth" not in launch
    assert "--ep-size 1" in launch
    assert "--max-running-requests 2" in launch
    assert "--context-length 524288 --max-total-tokens 524288" in launch
    assert "--kv-cache-dtype fp8_e4m3" in launch
    assert "--swa-full-tokens-ratio 0.0048828125" in launch
    assert "--chunked-prefill-size 1024 --max-prefill-tokens 1024" in launch
    assert "--cuda-graph-backend-decode full" in launch
    assert "--cuda-graph-max-bs-decode 2 --cuda-graph-bs-decode 1 2" in launch
    assert "--cuda-graph-backend-prefill disabled" in launch
    assert "--cuda-graph-max-bs-prefill 1024" in launch
    assert "--cuda-graph-bs-prefill 256 512 1024" in launch
    assert "--warmups dsv4_opencode_2694" in launch
    assert "--skip-server-warmup" not in launch
    assert "--speculative-algorithm" not in launch
    assert "--disable-overlap-schedule" in launch
    assert "--enable-p2p-check" in launch
    assert "--pre-warm-nccl" in launch
    assert "--kt-cpuinfer 56 --kt-threadpool-count 2 --kt-numa-nodes 0 1" in launch
    assert "--served-model-name deepseek-v4-flash" in launch
    assert "--tool-call-parser deepseekv4" in launch
    assert f"|auth={environment['DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT']}" in launch


@pytest.mark.parametrize(
    ("variable_name", "rotated_name", "expected_error"),
    (
        (
            "DSV4_CACHE_ROOT",
            "rotated-cache",
            "cache/controller namespace is fixed",
        ),
        (
            "DSV4_PP_LAUNCH_LEDGER",
            "fresh-two-launch-ledger.json",
            "launch ledger is fixed",
        ),
        (
            "DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT",
            "fresh-authorization.json",
            "canonical controller namespace",
        ),
    ),
)
def test_pp2_launcher_rejects_controller_namespace_rotation_without_touching_ledger(
    tmp_path: Path,
    variable_name: str,
    rotated_name: str,
    expected_error: str,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    canonical_ledger = Path(environment["DSV4_PP_LAUNCH_LEDGER"])
    canonical_ledger.parent.mkdir(parents=True)
    exhausted_ledger = b'{"format":"test-exhausted-ledger","launches":[{},{}]}\n'
    canonical_ledger.write_bytes(exhausted_ledger)
    rotated_path = tmp_path / rotated_name
    environment[variable_name] = str(rotated_path)
    environment["DSV4_PP_RUN_ROLE"] = "transfer"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert canonical_ledger.read_bytes() == exhausted_ledger
    assert not rotated_path.exists()
    assert not invocation_log.exists()


def test_pp2_launcher_rejects_reusable_authorization_receipt(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT"] = str(
        tmp_path / "old-authorization.json"
    )
    environment["DSV4_PP_RUN_ROLE"] = "transfer"

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "private controller output and cannot be reused" in result.stderr
    assert not invocation_log.exists()


def test_pp2_private_shim_rejects_effective_context_cli_tampering(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)
    environment["DSV4_PP_RUN_ROLE"] = "transfer"
    fake_numactl = Path(environment["PATH"].split(":", maxsplit=1)[0]) / "numactl"
    fake_numactl.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "shift 2\n"
        'exec "$@" --context-length 1\n',
        encoding="utf-8",
    )
    fake_numactl.chmod(0o755)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires exactly one --context-length 524288" in result.stderr
    invocation = invocation_log.read_text(encoding="utf-8")
    assert "dsv4_pp2_followup_ledger.py" not in invocation
    assert "-m sglang.launch_server" not in invocation


def test_pp2_private_shim_rejects_effective_oscar_environment_tampering(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)
    environment["DSV4_PP_RUN_ROLE"] = "transfer"
    fake_numactl = Path(environment["PATH"].split(":", maxsplit=1)[0]) / "numactl"
    fake_numactl.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "shift 2\n"
        "export SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=0\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    fake_numactl.chmod(0o755)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "requires SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1" in result.stderr
    invocation = invocation_log.read_text(encoding="utf-8")
    assert "dsv4_pp2_followup_ledger.py" not in invocation
    assert "-m sglang.launch_server" not in invocation


def test_pp2_model_launch_requires_an_explicit_ledger_role(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 2
    assert "DSV4_PP_RUN_ROLE must be transfer or optimized" in result.stderr
    if invocation_log.exists():
        assert "-m sglang.launch_server" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_accepts_the_second_run_partition(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_PIPELINE_LAYER_PARTITION"] = "22,21"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "|partition=22,21|" in prepare


def test_pp2_launcher_accepts_async_depth_one_as_an_isolated_tuning_knob(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)
    environment["DSV4_PP_ASYNC_BATCH_DEPTH"] = "1"
    environment["DSV4_PP_RUN_ROLE"] = "transfer"

    result = subprocess.run(
        ["bash", environment["PP2_TEST_LAUNCHER_PATH"], "--launch"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "--pp-max-micro-batch-size 1 --pp-async-batch-depth 1" in launch


def test_pp2_launcher_rejects_explicit_72_thread_oscar_transfer(
    tmp_path: Path,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_CPUINFER_THREADS"] = "72"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be exactly 56" in result.stderr
    if invocation_log.exists():
        assert "prepare_dsv4_flash_0731.py" not in invocation_log.read_text(
            encoding="utf-8"
        )


@pytest.mark.parametrize("thread_count", ("0", "55", "73", "056", "many"))
def test_pp2_launcher_rejects_unadmitted_cpuinfer_thread_count(
    tmp_path: Path,
    thread_count: str,
) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_CPUINFER_THREADS"] = thread_count

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be exactly 56" in result.stderr
    if invocation_log.exists():
        assert "prepare_dsv4_flash_0731.py" not in invocation_log.read_text(
            encoding="utf-8"
        )


def test_pp2_launcher_forces_pp_incompatible_features_off(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_DISABLE_SPECULATIVE"] = "0"
    environment["DSV4_PREFILL_GRAPH_BACKEND"] = "breakable"

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    prepare = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "|spec=1|" in prepare
    assert "|chunk=1024|prefill=disabled|decode=full|" in prepare


def test_pp2_swa_reserve_admits_1024_but_not_2048_token_chunks() -> None:
    context_tokens = 524_288
    reserve_tokens = int(context_tokens * 0.0048828125)
    page_size = 256

    assert reserve_tokens == 2_560
    assert reserve_tokens >= page_size + 2 * 1_024
    assert reserve_tokens < page_size + 2 * 2_048


def test_pp2_launcher_rejects_an_invalid_layer_partition(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    environment["DSV4_PIPELINE_LAYER_PARTITION"] = "20,23"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "must be 21,22 or 22,21" in result.stderr
    assert not invocation_log.exists()


def test_pp2_python_shim_rejects_direct_caller_entry(tmp_path: Path) -> None:
    environment, invocation_log, fake_python = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_PP_OPENCODE_PYTHON_SHIM": "1",
            "DSV4_PP_OPENCODE_REAL_PYTHON": str(fake_python),
        }
    )

    result = subprocess.run(
        [
            "bash",
            environment["PP2_TEST_LAUNCHER_PATH"],
            "-u",
            "-m",
            "sglang.launch_server",
            "--host",
            "127.0.0.1",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "launcher-private" in result.stderr
    assert not invocation_log.exists()


def test_pp2_python_shim_rejects_claimed_private_mode_without_capability(
    tmp_path: Path,
) -> None:
    environment, invocation_log, fake_python = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_PP_OPENCODE_PYTHON_SHIM": "private-v1",
            "DSV4_PP_OPENCODE_REAL_PYTHON": str(fake_python),
        }
    )

    result = subprocess.run(
        [
            "bash",
            environment["PP2_TEST_LAUNCHER_PATH"],
            "-u",
            "-m",
            "sglang.launch_server",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "private inherited capability" in result.stderr
    assert not invocation_log.exists()


def test_pp2_python_shim_allows_shape_warmup_opt_out(tmp_path: Path) -> None:
    environment, invocation_log, _ = launcher_environment(tmp_path)
    install_fake_launch_tools(tmp_path, environment)
    environment.update(
        {
            "DSV4_PP_OPENCODE_WARMUPS": "",
            "DSV4_PP_RUN_ROLE": "transfer",
        }
    )

    result = run_launcher(environment, ("--launch",))

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert "-m sglang.launch_server" in launch
    assert "--warmups" not in launch


def test_base_launcher_rejects_pipeline_speculation_before_launch(
    tmp_path: Path,
) -> None:
    environment, invocation_log, fake_python = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_PYTHON": str(fake_python),
            "DSV4_TENSOR_PARALLEL_SIZE": "1",
            "DSV4_PIPELINE_PARALLEL_SIZE": "2",
            "DSV4_DISABLE_SPECULATIVE": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(BASE_LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "pipeline parallelism does not support speculative decoding" in result.stderr
    assert not invocation_log.exists()
