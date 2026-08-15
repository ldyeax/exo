#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"
default_python=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
real_python="${DSV4_OPENCODE_REAL_PYTHON:-${DSV4_PYTHON:-$default_python}}"

# The existing hybrid launcher owns the validated performance and placement
# settings. It invokes DSV4_PYTHON for both preparation utilities and the final
# SGLang server. Re-enter this script as a narrow Python shim so only the
# launch_server invocation receives the OpenAI-compatible agent-serving flags.
if [[ ${DSV4_OPENCODE_PYTHON_SHIM:-0} == 1 ]]; then
  if [[ $# -ge 3 && $1 == -u && $2 == -m && $3 == sglang.launch_server ]]; then
    opencode_warmup_args=()
    opencode_warmups="${DSV4_OPENCODE_WARMUPS-dsv4_opencode_2694}"
    if [[ -n $opencode_warmups ]]; then
      opencode_warmup_args=(--warmups "$opencode_warmups")
    fi
    exec "$real_python" "$@" \
      "${opencode_warmup_args[@]}" \
      --enable-p2p-check \
      --pre-warm-nccl \
      --served-model-name deepseek-v4-flash \
      --reasoning-parser deepseek-v4 \
      --tool-call-parser deepseekv4 \
      --default-chat-template-kwargs '{"thinking":true}'
  fi
  exec "$real_python" "$@"
fi

if [[ ! -x $real_python ]]; then
  echo "DeepSeek V4 Python interpreter is not executable: $real_python" >&2
  exit 1
fi

# Native candidates are distributed as hash-pinned extensions, not in-place
# replacements for the rollback-safe installed package. The qualified combined
# 56-worker inline-dispatch plus N128 scale-fold artifact is the direct-serving
# default; the older AVX-tail binary remains an explicit campaign rollback. A caller
# supplying its own complete KTransformers source keeps ownership unless it
# explicitly opts into an overlay.  Campaign stages always set both switches,
# so the rollback baseline remains the older hash-pinned AVX-tail package.
if [[ -v DSV4_KTRANSFORMERS_SOURCE && -z $DSV4_KTRANSFORMERS_SOURCE ]]; then
  echo "DSV4_KTRANSFORMERS_SOURCE must be a non-empty explicit path" >&2
  exit 2
fi
stage_kt_avx_tail_overlay="${DSV4_STAGE_KT_AVX_TAIL_OVERLAY:-0}"
stage_kt_persistent_counter_overlay="${DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY:-0}"
stage_kt_task_queue_pin_overlay="${DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY:-0}"
if [[ -v DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY ]]; then
  stage_kt_cpu_optimized_overlay="$DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY"
elif [[ -v DSV4_KTRANSFORMERS_SOURCE ]]; then
  stage_kt_cpu_optimized_overlay=0
else
  stage_kt_cpu_optimized_overlay=1
fi
case "${stage_kt_avx_tail_overlay,,}" in
true | 1 | yes | y)
  stage_kt_avx_tail_overlay=1
  ;;
false | 0 | no | n)
  stage_kt_avx_tail_overlay=0
  ;;
*)
  echo "DSV4_STAGE_KT_AVX_TAIL_OVERLAY must be a boolean value" >&2
  exit 2
  ;;
esac
case "${stage_kt_persistent_counter_overlay,,}" in
true | 1 | yes | y)
  stage_kt_persistent_counter_overlay=1
  ;;
false | 0 | no | n)
  stage_kt_persistent_counter_overlay=0
  ;;
*)
  echo "DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY must be a boolean value" >&2
  exit 2
  ;;
esac
case "${stage_kt_task_queue_pin_overlay,,}" in
true | 1 | yes | y)
  stage_kt_task_queue_pin_overlay=1
  ;;
false | 0 | no | n)
  stage_kt_task_queue_pin_overlay=0
  ;;
*)
  echo "DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY must be a boolean value" >&2
  exit 2
  ;;
esac
case "${stage_kt_cpu_optimized_overlay,,}" in
true | 1 | yes | y)
  stage_kt_cpu_optimized_overlay=1
  ;;
false | 0 | no | n)
  stage_kt_cpu_optimized_overlay=0
  ;;
*)
  echo "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY must be a boolean value" >&2
  exit 2
  ;;
esac

selected_kt_experiment_overlays=$((\
  stage_kt_task_queue_pin_overlay + \
  stage_kt_persistent_counter_overlay + \
  stage_kt_cpu_optimized_overlay))
if ((selected_kt_experiment_overlays > 1)); then
  echo "the KT task-queue-pin, persistent-counter, and CPU-optimized overlays are mutually exclusive" >&2
  exit 2
fi
if [[ $stage_kt_cpu_optimized_overlay == 1 &&
  $stage_kt_avx_tail_overlay == 1 ]]; then
  echo "the combined CPU-optimized and legacy AVX-tail overlays are mutually exclusive" >&2
  exit 2
fi
if [[ $stage_kt_task_queue_pin_overlay == 0 &&
  $stage_kt_cpu_optimized_overlay == 0 &&
  ${KT_TASK_QUEUE_PIN_FIRST_CORE:-0} == 1 ]]; then
  echo "KT_TASK_QUEUE_PIN_FIRST_CORE=1 requires a hash-pinned task-queue-capable overlay" >&2
  exit 2
fi
if [[ -v DSV4_KTRANSFORMERS_SOURCE &&
  $stage_kt_task_queue_pin_overlay == 1 ]]; then
  echo "DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY conflicts with DSV4_KTRANSFORMERS_SOURCE" >&2
  exit 2
fi
if [[ $stage_kt_cpu_optimized_overlay == 0 &&
  ${KT_SINGLE_NUMA_INLINE_DISPATCH:-0} == 1 ]]; then
  echo "KT_SINGLE_NUMA_INLINE_DISPATCH=1 requires the hash-pinned CPU-optimized overlay" >&2
  exit 2
fi
if [[ -v DSV4_KTRANSFORMERS_SOURCE &&
  $stage_kt_cpu_optimized_overlay == 1 ]]; then
  echo "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY conflicts with DSV4_KTRANSFORMERS_SOURCE" >&2
  exit 2
fi
if [[ $stage_kt_cpu_optimized_overlay == 1 ]]; then
  case "${DSV4_CPUINFER_THREADS:-56}" in
  56)
    export DSV4_CPUINFER_THREADS="${DSV4_CPUINFER_THREADS:-56}"
    ;;
  *)
    echo "the CPU-optimized overlay requires the qualified DSV4_CPUINFER_THREADS=56" >&2
    exit 2
    ;;
  esac
  if [[ ${KT_WORKER_SPIN_US:-1000} != 1000 ]]; then
    echo "the CPU-optimized overlay requires the qualified KT_WORKER_SPIN_US=1000" >&2
    exit 2
  fi
fi

if [[ -v KT_MXFP4_AVX_SCALE_FOLD_MODE ]]; then
  mxfp4_avx_scale_fold_mode="$KT_MXFP4_AVX_SCALE_FOLD_MODE"
elif [[ $stage_kt_cpu_optimized_overlay == 1 ]]; then
  mxfp4_avx_scale_fold_mode=lut-v1
else
  mxfp4_avx_scale_fold_mode=off
fi
case "$mxfp4_avx_scale_fold_mode" in
off | lut-v1 | exponent-v1) ;;
*)
  echo "KT_MXFP4_AVX_SCALE_FOLD_MODE must be off, lut-v1, or exponent-v1" >&2
  exit 2
  ;;
esac
if [[ $stage_kt_cpu_optimized_overlay == 1 &&
  $mxfp4_avx_scale_fold_mode != lut-v1 ]]; then
  echo "the CPU-optimized overlay requires qualified KT_MXFP4_AVX_SCALE_FOLD_MODE=lut-v1" >&2
  exit 2
fi
if [[ $stage_kt_cpu_optimized_overlay == 0 &&
  $mxfp4_avx_scale_fold_mode != off ]]; then
  echo "non-off KT_MXFP4_AVX_SCALE_FOLD_MODE requires the hash-pinned CPU-optimized overlay" >&2
  exit 2
fi
export KT_MXFP4_AVX_SCALE_FOLD_MODE="$mxfp4_avx_scale_fold_mode"

# Persistent fine-grained dispatch counters are a deliberately separate A/B
# candidate. Both its native extension and the Python module that exposes the
# counters are staged from explicit sources with non-overridable SHA-256 pins;
# changing either source path cannot bypass content verification. Selecting
# this experiment takes precedence over the default AVX-tail overlay, but an
# explicit KTransformers source and an experiment selection are rejected as
# ambiguous.
if [[ -v DSV4_KTRANSFORMERS_SOURCE &&
  $stage_kt_persistent_counter_overlay == 1 ]]; then
  echo "DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY conflicts with DSV4_KTRANSFORMERS_SOURCE" >&2
  exit 2
fi
if [[ $stage_kt_cpu_optimized_overlay == 1 ]]; then
  cpu_optimized_default_candidate=
  cpu_optimized_default_candidate+="/var/lib/exo/experiments/"
  cpu_optimized_default_candidate+="dsv4-cpu-inline-scale-lut-n128-v1/"
  cpu_optimized_default_candidate+="lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
  cpu_optimized_candidate="${DSV4_KT_CPU_OPTIMIZED_CANDIDATE:-$cpu_optimized_default_candidate}"
  cpu_optimized_cache_root="${DSV4_KT_CPU_OPTIMIZED_CACHE_ROOT:-/var/lib/exo/cache/dsv4-cpu-optimized-serving-overlays}"
  cpu_optimized_sha256=7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043
  kt_overlay_root="$("$real_python" \
    "${script_dir}/stage_dsv4_kt_avx_tail_overlay.py" \
    --candidate "$cpu_optimized_candidate" \
    --expected-sha256 "$cpu_optimized_sha256" \
    --cache-root "$cpu_optimized_cache_root")"
  if [[ -z $kt_overlay_root || $kt_overlay_root == *$'\n'* ||
    ! -d ${kt_overlay_root}/kt_kernel ]]; then
    echo "staged DSV4 KT CPU-optimized overlay is not an importable package root: ${kt_overlay_root:-<empty>}" >&2
    exit 1
  fi
  export DSV4_KTRANSFORMERS_SOURCE="$kt_overlay_root"
  export KT_TASK_QUEUE_PIN_FIRST_CORE=1
  export KT_SINGLE_NUMA_INLINE_DISPATCH=1
elif [[ $stage_kt_task_queue_pin_overlay == 1 ]]; then
  task_queue_pin_default_candidate=
  task_queue_pin_default_candidate+="/tmp/dsv4-task-queue-pin-build-static/"
  task_queue_pin_default_candidate+="lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
  task_queue_pin_candidate="${DSV4_KT_TASK_QUEUE_PIN_CANDIDATE:-$task_queue_pin_default_candidate}"
  task_queue_pin_cache_root="${DSV4_KT_TASK_QUEUE_PIN_CACHE_ROOT:-/tmp/dsv4-task-queue-pin-serving-overlays}"
  task_queue_pin_sha256=cfe7aaf328f71fc50aac877b56ee474e07b0f06ecc8571736ba3cc1e8e3786dd
  kt_overlay_root="$("$real_python" \
    "${script_dir}/stage_dsv4_kt_avx_tail_overlay.py" \
    --candidate "$task_queue_pin_candidate" \
    --expected-sha256 "$task_queue_pin_sha256" \
    --cache-root "$task_queue_pin_cache_root")"
  if [[ -z $kt_overlay_root || $kt_overlay_root == *$'\n'* ||
    ! -d ${kt_overlay_root}/kt_kernel ]]; then
    echo "staged DSV4 KT task-queue-pin overlay is not an importable package root: ${kt_overlay_root:-<empty>}" >&2
    exit 1
  fi
  export DSV4_KTRANSFORMERS_SOURCE="$kt_overlay_root"
  export KT_TASK_QUEUE_PIN_FIRST_CORE=1
elif [[ $stage_kt_persistent_counter_overlay == 1 ]]; then
  persistent_counter_default_candidate=
  persistent_counter_default_candidate+="/tmp/dsv4-kt-persistent-counter-a40796696a1181bb/"
  persistent_counter_default_candidate+="lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
  persistent_counter_candidate="${DSV4_KT_PERSISTENT_COUNTER_CANDIDATE:-$persistent_counter_default_candidate}"
  persistent_counter_cache_root="${DSV4_KT_PERSISTENT_COUNTER_CACHE_ROOT:-/tmp/dsv4-kt-persistent-counter-overlays}"
  persistent_counter_python_default_source="${repo_root}/vendor/ktransformers/kt-kernel/python/experts_base.py"
  persistent_counter_python_source="${DSV4_KT_PERSISTENT_COUNTER_PYTHON_SOURCE:-$persistent_counter_python_default_source}"
  persistent_counter_sha256=a40796696a1181bb680379a94d80753345b8f4d979e865b9a49e06ee1c965373
  persistent_counter_python_sha256=b6aaa020bba9a326e2e9191791d79b9c88ff6f429c84ebae80a31df8e37257bf
  kt_overlay_root="$("$real_python" \
    "${script_dir}/stage_dsv4_kt_avx_tail_overlay.py" \
    --candidate "$persistent_counter_candidate" \
    --expected-sha256 "$persistent_counter_sha256" \
    --python-source "$persistent_counter_python_source" \
    --expected-python-sha256 "$persistent_counter_python_sha256" \
    --cache-root "$persistent_counter_cache_root")"
  if [[ -z $kt_overlay_root || $kt_overlay_root == *$'\n'* ||
    ! -d ${kt_overlay_root}/kt_kernel ]]; then
    echo "staged DSV4 KT persistent-counter overlay is not an importable package root: ${kt_overlay_root:-<empty>}" >&2
    exit 1
  fi
  export DSV4_KTRANSFORMERS_SOURCE="$kt_overlay_root"
elif [[ ! -v DSV4_KTRANSFORMERS_SOURCE && $stage_kt_avx_tail_overlay == 1 ]]; then
  kt_overlay_root="$("$real_python" \
    "${script_dir}/stage_dsv4_kt_avx_tail_overlay.py")"
  if [[ -z $kt_overlay_root || $kt_overlay_root == *$'\n'* ||
    ! -d ${kt_overlay_root}/kt_kernel ]]; then
    echo "staged DSV4 KT AVX-tail overlay is not an importable package root: ${kt_overlay_root:-<empty>}" >&2
    exit 1
  fi
  export DSV4_KTRANSFORMERS_SOURCE="$kt_overlay_root"
fi
if [[ $stage_kt_task_queue_pin_overlay == 0 &&
  $stage_kt_cpu_optimized_overlay == 0 ]]; then
  export KT_TASK_QUEUE_PIN_FIRST_CORE=0
fi
if [[ $stage_kt_cpu_optimized_overlay == 0 ]]; then
  export KT_SINGLE_NUMA_INLINE_DISPATCH=0
fi

export DSV4_OPENCODE_PYTHON_SHIM=1
export DSV4_OPENCODE_REAL_PYTHON="$real_python"
export DSV4_PYTHON="$script_path"
export DSV4_MODEL_PATH="${DSV4_MODEL_PATH:-/tmp/dsv4-local-checkpoint-0731}"
experimental_amxint4_cpu=0
if [[ -v DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH ]]; then
  experimental_amxint4_cpu=1
  amxint4_cpu_weight_path="$DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH"
  if [[ -z $amxint4_cpu_weight_path || ${amxint4_cpu_weight_path:0:1} != / ||
    ! -d $amxint4_cpu_weight_path || ! -r $amxint4_cpu_weight_path ]]; then
    echo "DSV4_EXPERIMENTAL_AMXINT4_CPU_WEIGHT_PATH must be a readable absolute directory" >&2
    exit 2
  fi
  if ! compgen -G "${amxint4_cpu_weight_path}/*.safetensors" >/dev/null; then
    echo "experimental AMXINT4 CPU directory contains no top-level safetensors artifact" >&2
    exit 2
  fi
  export DSV4_KT_METHOD=AMXINT4
  export DSV4_KT_WEIGHT_PATH="$amxint4_cpu_weight_path"
  export SGLANG_DSV4_SPLIT_MXFP4_GPU_AMXINT4_CPU=1
  export SGLANG_KT_DRAFT_METHOD=MXFP4
  export SGLANG_KT_DRAFT_WEIGHT_PATH="$DSV4_MODEL_PATH"
fi
if [[ ${DSV4_CONTEXT_LENGTH:-524288} != 524288 ]]; then
  echo "admitted OSCAR serving requires DSV4_CONTEXT_LENGTH=524288" >&2
  exit 2
fi
if [[ ${DSV4_MAX_TOTAL_TOKENS:-524288} != 524288 ]]; then
  echo "admitted OSCAR serving requires DSV4_MAX_TOTAL_TOKENS=524288" >&2
  exit 2
fi
export DSV4_CONTEXT_LENGTH=524288
export DSV4_MAX_TOTAL_TOKENS=524288
export DSV4_KV_CACHE_DTYPE="${DSV4_KV_CACHE_DTYPE:-fp8_e4m3}"
# OSCAR-INT2 is the only compressed KV layout admitted by this launcher.  The
# fp8_e4m3 spelling is a raw-byte carrier required by SGLang's public CLI; on
# exact SM86 the physical history pages are calibrated asymmetric OSCAR INT2,
# while the protected SWA window remains rotated BF16.  Never silently fall
# back to generic FP8, symmetric INT4, selective BF16 C128, or an identity
# rotation: each would invalidate both the memory budget and quality result.
if [[ $DSV4_KV_CACHE_DTYPE != fp8_e4m3 ]]; then
  echo "OSCAR-INT2 requires DSV4_KV_CACHE_DTYPE=fp8_e4m3 as its raw-byte carrier" >&2
  exit 2
fi
case "${SGLANG_DSV4_OSCAR_INT2_KV_STORAGE:-1}" in
1 | true | TRUE | yes | YES | y | Y) ;;
*)
  echo "this launcher requires SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1" >&2
  exit 2
  ;;
esac
# The qualified SM86 serving path partitions Oscar history inside fixed,
# backend-owned FP32 workspaces and fuses deterministic reduction into the
# graph-replayed decode kernel.  It remains Oscar storage end to end; disabling
# it would restore the slower monolithic Oscar reader and invalidate the
# promoted performance receipt.
case "${SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY:-1}" in
1 | true | TRUE | yes | YES | y | Y) ;;
*)
  echo "this launcher requires SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1" >&2
  exit 2
  ;;
esac
if [[ -n ${SGLANG_DSV4_OSCAR_CAPTURE_CONFIG:-} ]]; then
  echo "SGLANG_DSV4_OSCAR_CAPTURE_CONFIG is calibration-only and conflicts with admitted OSCAR serving" >&2
  exit 2
fi
for non_oscar_setting in \
  SGLANG_DSV4_INT4_KV_STORAGE \
  SGLANG_DSV4_INT4_C4_INDEXER_STORAGE \
  SGLANG_DSV4_SM86_C128_BF16_STORAGE; do
  non_oscar_value="${!non_oscar_setting:-0}"
  case "$non_oscar_value" in
  0 | false | FALSE | no | NO | n | N | "") ;;
  *)
    echo "$non_oscar_setting conflicts with mandatory OSCAR-INT2 KV storage" >&2
    exit 2
    ;;
  esac
done
default_oscar_calibration="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-hybrid-ep2}/oscar-int2/dsv4-oscar-int2-calibration.pt"
if [[ -v DSV4_OSCAR_CALIBRATION_PATH &&
  -v SGLANG_DSV4_OSCAR_CALIBRATION_PATH &&
  $DSV4_OSCAR_CALIBRATION_PATH != "$SGLANG_DSV4_OSCAR_CALIBRATION_PATH" ]]; then
  echo "DSV4_OSCAR_CALIBRATION_PATH conflicts with SGLANG_DSV4_OSCAR_CALIBRATION_PATH" >&2
  exit 2
fi
oscar_calibration_path="${DSV4_OSCAR_CALIBRATION_PATH:-${SGLANG_DSV4_OSCAR_CALIBRATION_PATH:-$default_oscar_calibration}}"
if [[ -z $oscar_calibration_path || ${oscar_calibration_path:0:1} != / ||
  ! -f $oscar_calibration_path || ! -r $oscar_calibration_path ||
  -L $oscar_calibration_path ]]; then
  echo "OSCAR-INT2 requires a readable absolute, non-symlink calibration artifact: ${oscar_calibration_path:-<empty>}" >&2
  exit 2
fi
oscar_root="${oscar_calibration_path%/*}"
oscar_fingerprint_path="${DSV4_OSCAR_CHECKPOINT_FINGERPRINT_PATH:-${oscar_root}/checkpoint-fingerprint.json}"
oscar_admission_path="${DSV4_OSCAR_ADMISSION_RECEIPT_PATH:-${oscar_root}/admission.json}"
oscar_model_id="${DSV4_OSCAR_MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}"
if [[ $oscar_model_id != deepseek-ai/DeepSeek-V4-Flash ]]; then
  echo "OSCAR-INT2 serving requires DSV4_OSCAR_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash" >&2
  exit 2
fi
launch_requested=0
for launcher_argument in "$@"; do
  if [[ $launcher_argument == --launch ]]; then
    launch_requested=1
    break
  fi
done
if [[ $launch_requested == 1 ]]; then
  if [[ -z $oscar_fingerprint_path || ${oscar_fingerprint_path:0:1} != / ||
    ! -f $oscar_fingerprint_path || ! -r $oscar_fingerprint_path ||
    -L $oscar_fingerprint_path ]]; then
    echo "OSCAR-INT2 launch requires a readable absolute, non-symlink checkpoint fingerprint: ${oscar_fingerprint_path:-<empty>}" >&2
    exit 2
  fi
  if [[ -z $oscar_admission_path || ${oscar_admission_path:0:1} != / ||
    -L $oscar_admission_path || ! -d ${oscar_admission_path%/*} ||
    ! -w ${oscar_admission_path%/*} ]]; then
    echo "OSCAR-INT2 admission receipt must target a writable absolute non-symlink path: ${oscar_admission_path:-<empty>}" >&2
    exit 2
  fi
  "$real_python" "${script_dir}/dsv4_oscar_int2_calibration.py" admit \
    --artifact "$oscar_calibration_path" \
    --checkpoint "$DSV4_MODEL_PATH" \
    --checkpoint-fingerprint "$oscar_fingerprint_path" \
    --model-id "$oscar_model_id" \
    --output "$oscar_admission_path"
  if [[ ! -f $oscar_admission_path || ! -r $oscar_admission_path ||
    -L $oscar_admission_path ]]; then
    echo "OSCAR-INT2 model-bound admission did not produce a readable receipt" >&2
    exit 1
  fi
fi
export SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1
export SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1
export SGLANG_DSV4_OSCAR_CALIBRATION_PATH="$oscar_calibration_path"
export SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH="$oscar_admission_path"
export SGLANG_DSV4_INT4_KV_STORAGE=0
export SGLANG_DSV4_INT4_C4_INDEXER_STORAGE=0
export SGLANG_DSV4_SM86_C128_BF16_STORAGE=0
export DSV4_RAGGED_VERIFY_MODE="${DSV4_RAGGED_VERIFY_MODE:-compact}"
# Keep the qualified fine-grained verify graph keys and TP2 SPS table even
# though the fixed verify-4 serving policy normally selects only key 4.  The
# table remains the explicit rollback when DSV4_DSPARK_FIXED_VERIFY_LEN is set
# to an empty string, and the full decode graph is the measured SM86 backend.
export DSV4_FINE_RAGGED_VERIFY_TIERS="${DSV4_FINE_RAGGED_VERIFY_TIERS:-1}"
export DSV4_SPS_TABLE="${DSV4_SPS_TABLE:-${script_dir}/data/dsv4_flash_tp2_finetiers_sps.json}"
if [[ ${DSV4_DECODE_GRAPH_BACKEND:-full} == disabled ]]; then
  echo "admitted OSCAR serving requires decode CUDA graphs" >&2
  exit 2
fi
if [[ ${DSV4_DISABLE_SPECULATIVE:-0} != 0 ]]; then
  echo "the EP2 OpenCode configuration requires graph-backed DSpark speculation" >&2
  exit 2
fi
if [[ ${DSV4_TARGET_VERIFY_EAGER:-0} != 0 ]]; then
  echo "the EP2 OpenCode configuration requires graph-backed target verification" >&2
  exit 2
fi
export DSV4_DECODE_GRAPH_BACKEND="${DSV4_DECODE_GRAPH_BACKEND:-full}"
export DSV4_DISABLE_SPECULATIVE=0
export DSV4_TARGET_VERIFY_EAGER=0
# This launcher is intrinsically local TP2/EP2.  Make SGLang run its actual
# CUDA peer-access check instead of installing the trust-the-driver monkey
# patch, and remove NCCL's first-use initialization from the first user
# request.  NCCL_P2P_LEVEL=NVL remains pinned by the base launcher; model-run
# counter receipts separately prove that payload traffic reaches every link.
# At 524K, the parity launcher's proportional 0.15 SWA reserve would allocate
# 78,592 slots for a checkpoint whose sliding window is only 128 tokens. Keep
# 2,560 slots instead. The admission floor for the configured 1,024-token
# chunk is 256 + 2 * max(128, 1,024) = 2,304 slots, while a 2,048-token chunk
# would require 4,352 slots. The original 1,024-slot reserve silently clamped
# prefill chunks to 512 tokens and increased fresh-cache TTFT.
export DSV4_SWA_FULL_TOKENS_RATIO="${DSV4_SWA_FULL_TOKENS_RATIO:-0.0048828125}"
# Cap prefill chunks at 1,024 and capture the practical agent-serving tiers.
# The lower tiers cover fresh/cached extensions down to 128 real tokens; the
# 1,024 tier keeps the 2,694-token benchmark fully on the BCG path. Avoiding
# the much larger 2,048-token capture preserves several GiB at 524K capacity.
export DSV4_CHUNKED_PREFILL_SIZE="${DSV4_CHUNKED_PREFILL_SIZE:-1024}"
export DSV4_PREFILL_GRAPH_TIERS="${DSV4_PREFILL_GRAPH_TIERS:-256 512 1024}"
export DSV4_PREFILL_GRAPH_MAX="${DSV4_PREFILL_GRAPH_MAX:-1024}"
# Do not advertise the OpenCode endpoint as ready until SGLang has exercised a
# real prefill/decode request. This consumes the lazy JIT work which otherwise
# made the first agent request miss the TTFT gate. The generic launcher keeps
# its historical skip-by-default behavior, and an explicit caller value wins.
export DSV4_SKIP_SERVER_WARMUP="${DSV4_SKIP_SERVER_WARMUP:-0}"
# Prime the actual three-chunk agent-serving shape before the ASGI application
# finishes startup. Presence is the override contract: setting this variable
# to an empty string deliberately keeps only SGLang's generic HTTP warmup.
if [[ ! -v DSV4_OPENCODE_WARMUPS ]]; then
  export DSV4_OPENCODE_WARMUPS=dsv4_opencode_2694
fi
# Keep CUDA graphs enabled for prefill, but run the whole attention module eagerly
# inside breakable-graph replay.  The old partial-attention graph boundary let
# padded replay rows write KV slot 0; this repaired boundary has passed the cold,
# cache-flushed OpenCode semantic and forced-tool-call gates.  The explicit
# variables remain overridable so `disabled` is still available as a diagnostic
# fallback.
export DSV4_PREFILL_GRAPH_BACKEND="${DSV4_PREFILL_GRAPH_BACKEND:-breakable}"
export DSV4_EAGER_ATTN_MODULE_IN_BCG="${DSV4_EAGER_ATTN_MODULE_IN_BCG:-1}"
export DSV4_CAPTURE_ATTN_IN_BCG="${DSV4_CAPTURE_ATTN_IN_BCG:-0}"
# Three repeated five-phase qualifications selected the frozen g14-p28 target
# placement. Keep its 14 slots per rank coupled to the draft placement and to
# the fallback profile builder used by explicit geometry overrides.
export DSV4_GPU_EXPERTS_PER_LAYER="${DSV4_GPU_EXPERTS_PER_LAYER:-14}"
# The graph-safe live-swap controller remains opt-in because the qualified
# placement is now installed directly at startup. Enabling the controller
# retains immutable CPU shadows for all rank-owned target experts and exposes
# the idle-boundary /kt_expert_hotspot path for future experiments; it never
# changes the fixed 14-slot GPU allocation or captured tensor addresses.
export SGLANG_KT_HOTSPOT_EXPERT_CACHE="${DSV4_HOTSPOT_EXPERT_CACHE:-${SGLANG_KT_HOTSPOT_EXPERT_CACHE:-0}}"
if [[ $experimental_amxint4_cpu == 1 &&
  $SGLANG_KT_HOTSPOT_EXPERT_CACHE != 0 ]]; then
  echo "experimental AMXINT4 CPU split tier is incompatible with the MXFP4 hotspot cache" >&2
  exit 2
fi
if [[ ! -v DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER ]]; then
  export DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER="$DSV4_GPU_EXPERTS_PER_LAYER"
fi
# Materialize the qualified target placement from its compact, reviewable JSON
# manifest unless the caller supplied any recorder/profile/plan input. A custom
# geometry or profile-selection override falls back to the profile builder
# instead of silently applying a mismatched frozen plan.
if [[ ! -v DSV4_EXPERT_RECORDER_PROFILES &&
  ! -v DSV4_HYBRID_EXPERT_PROFILE &&
  ! -v DSV4_HYBRID_EXPERT_SHARD_PLAN &&
  ! -v SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN ]]; then
  use_frozen_target_plan=1
  if [[ $DSV4_GPU_EXPERTS_PER_LAYER != 14 ||
    $DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER != 14 ||
    -v DSV4_HYBRID_GPU_SELECTION_STRATEGY ||
    -v DSV4_HYBRID_EXPERT_FILL_PROFILE ]]; then
    use_frozen_target_plan=0
  fi
  if [[ $use_frozen_target_plan == 1 ]]; then
    frozen_target_manifest="${DSV4_OPENCODE_FROZEN_TARGET_MANIFEST:-${script_dir}/data/dsv4_flash_opencode_g14_p28_frozen_plan.json}"
    frozen_target_plan="${DSV4_OPENCODE_FROZEN_TARGET_PLAN:-${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-hybrid-ep2}/opencode-g14-p28-frozen.pt}"
    "$real_python" "${script_dir}/materialize_dsv4_frozen_hybrid_plan.py" \
      --manifest "$frozen_target_manifest" \
      --output "$frozen_target_plan"
    if [[ ! -f $frozen_target_plan || ! -r $frozen_target_plan ]]; then
      echo "materialized OpenCode target plan is not readable: $frozen_target_plan" >&2
      exit 1
    fi
    export DSV4_HYBRID_EXPERT_SHARD_PLAN="$frozen_target_plan"
  else
    export DSV4_HYBRID_EXPERT_PROFILE="${script_dir}/data/dsv4_flash_opencode_target_distinct_decode_calls_sparse.json"
  fi
fi
# The generic hybrid launcher accepts its DSV4 spelling as the authoritative
# prebuilt target plan. Mirror the runtime spelling when that is the only
# explicit plan so it is preserved rather than replaced during preparation.
if [[ ! -v DSV4_HYBRID_EXPERT_SHARD_PLAN &&
  -v SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN ]]; then
  export DSV4_HYBRID_EXPERT_SHARD_PLAN="$SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN"
fi
# Match the geometry this checkpoint was trained with.  The old block-8
# override produced attractive filler-token numbers, but it changed the
# non-causal draft block seen by every DSpark position and is not a valid
# default for the checkpoint's dspark_block_size=5.
export DSV4_DSPARK_BLOCK_SIZE="${DSV4_DSPARK_BLOCK_SIZE:-5}"
# Fixed tier 4 won the repeated cache-state sweep. Make it the serving
# baseline before warmup and the first OpenCode request, rather than relying on
# the diagnostic /set_internal_state endpoint after startup. An explicitly
# empty value is the rollback to confidence/SPS scheduling.
if [[ ! -v DSV4_DSPARK_FIXED_VERIFY_LEN ]]; then
  export DSV4_DSPARK_FIXED_VERIFY_LEN=4
fi
# Qualified decode settings: overlap the model-side MoE stream, retain the
# LM head in BF16, use the qualified FP32 Markov projection, and use AMX only
# at five routed rows while the pinned AVX kernel handles the common 2/3-row
# tail.
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP:-1}"
export SGLANG_DSPARK_FP32_LM_HEAD="${SGLANG_DSPARK_FP32_LM_HEAD:-0}"
export SGLANG_DSPARK_OPT_MARKOV_W2_BF16="${SGLANG_DSPARK_OPT_MARKOV_W2_BF16:-0}"
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS="${KT_MXFP4_AMX_MIN_EXPERT_TOKENS:-5}"
export KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS="${KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS:-2}"
# On SM86 the frozen target has 14 rank-local GPU experts, while variable-width
# campaigns admit 1 through 22.  For every admitted width with top-6 routes and
# one to six live verification rows, build padded top-8 routing metadata in one
# graph-safe Triton program. Other shapes and architectures fall back inside
# the kernel wrapper; explicit zero is the rollback switch.
export SGLANG_V4_MXFP4_SMALL_ROW_ROUTING="${SGLANG_V4_MXFP4_SMALL_ROW_ROUTING:-1}"
# The package heuristic estimates split-K before routed expert multiplicity is
# visible.  On both the frozen E14 plan and variable-width E22 ceiling, real
# checkpoint weights selected block-N 128 / split-K 2 / four stages across all
# one-to-six-row decode graphs.  The kernel-side gate additionally requires
# exact SM86, V4's two matrix signatures, padded top-8 routing, and at most 22
# local experts. With opt-in value 1, an incompatible triton_kernels API or
# target dispatch is fatal instead of silently restoring the package heuristic;
# explicit zero is the deliberate fallback. After warmup, /server_info reports
# TP-gathered selection counters and requires both local ranks for all-active.
export SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM="${SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM:-1}"
# The qualified TP2 path uses the graph-safe custom all-reduce v2 kernel.  The
# runtime retains its own architecture/shape guards, and zero is the explicit
# rollback for diagnostics.
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2="${SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2:-1}"
# Rebuild the validated draft-hot plan from its compact, reviewable source
# profile. An explicitly set recorder, profile, or either plan variable owns
# draft placement, including when a caller deliberately sets it to empty.
if [[ ! -v DSV4_DRAFT_EXPERT_RECORDER_PROFILES &&
  ! -v DSV4_DRAFT_HYBRID_EXPERT_PROFILE &&
  ! -v DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN &&
  ! -v SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN ]]; then
  export DSV4_DRAFT_HYBRID_EXPERT_PROFILE="${script_dir}/data/dsv4_flash_draft_distinct_decode_calls_sparse.json"
fi

exec "${repo_root}/scripts/dsv4_flash_hybrid_ep2_dwagon.sh" "$@"
