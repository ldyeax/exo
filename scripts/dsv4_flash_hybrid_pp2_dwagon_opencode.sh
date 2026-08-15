#!/usr/bin/env bash
set -euo pipefail

# Two-stage local pipeline intended for the post-TP/EP concurrency study.
# DSV4_PP_EP2_WINNER_PLAN transfers the exact per-layer union of an admitted
# fixed- or variable-width EP2 placement into the EP1 plan used independently
# by both pipeline stages. DSV4_GPU_EXPERTS_PER_LAYER names the widest target
# union layer (the loader admission ceiling), not a promise of uniform width.
# DSV4_PP_EP_WINNER_RECEIPT must be the clean, qualified final EP confirmation
# hotspot receipt that names that exact source plan and proves split-history on
# both EP workers plus traffic on all NVLink counters.
# DSV4_PP_EP_COHERENCY_RECEIPT binds the separate deterministic text/tool proof.
# Model
# launches additionally require DSV4_PP_RUN_ROLE=transfer for the first launch
# and =optimized plus DSV4_PP_FIRST_BENCHMARK_RECEIPT for the second. The
# canonical controller refuses a third authorization through this PP2
# entrypoint. It is campaign evidence, not a host-wide execution-control
# primitive for unrelated lower-level launchers.
# SGLang currently rejects pp_size > 1 unless overlap scheduling and every
# speculative algorithm are disabled. This launcher therefore disables DSpark
# and passes --disable-overlap-schedule explicitly. Target decode CUDA graphs
# stay on.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"
default_python=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
real_python="${DSV4_PP_OPENCODE_REAL_PYTHON:-${DSV4_PYTHON:-$default_python}}"

# The launch counter is a property of this qualification campaign, not of a
# caller-selected compilation-cache directory.  Keep these literal paths in
# the launcher so rotating environment variables cannot mint another two-run
# namespace.  In particular, these are the paths that already contain the two
# completed PP2 authorizations on dwagon; do not migrate or recreate them.
readonly canonical_cache_root=/var/lib/exo/cache/dsv4-flash-hybrid-pp2-opencode
readonly canonical_launch_ledger="${canonical_cache_root}/pp2-two-launch-ledger.json"
readonly canonical_transfer_authorization="${canonical_cache_root}/authorization-transfer.json"
readonly canonical_optimized_authorization="${canonical_cache_root}/authorization-optimized.json"
readonly canonical_capability_directory="${canonical_cache_root}/.private-python-shim"

if [[ -v DSV4_CACHE_ROOT && $DSV4_CACHE_ROOT != "$canonical_cache_root" ]]; then
  echo "PP2 cache/controller namespace is fixed at $canonical_cache_root" >&2
  exit 2
fi
if [[ -v DSV4_PP_LAUNCH_LEDGER &&
  $DSV4_PP_LAUNCH_LEDGER != "$canonical_launch_ledger" ]]; then
  echo "PP2 launch ledger is fixed at $canonical_launch_ledger" >&2
  exit 2
fi
if [[ -v DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT &&
  $DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT != "$canonical_transfer_authorization" &&
  $DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT != "$canonical_optimized_authorization" ]]; then
  echo "PP2 launch authorization output must remain in the canonical controller namespace" >&2
  exit 2
fi
if [[ ! -v DSV4_PP_OPENCODE_PYTHON_SHIM &&
  -v DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT ]]; then
  echo "DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT is private controller output and cannot be reused" >&2
  exit 2
fi
export DSV4_CACHE_ROOT="$canonical_cache_root"
export DSV4_PP_LAUNCH_LEDGER="$canonical_launch_ledger"

pp2_process_start_ticks() {
  awk '{print $22}' "/proc/$$/stat"
}

pp2_assert_private_shim_descriptor() {
  local descriptor="${DSV4_PP_PRIVATE_SHIM_FD:-}"
  if [[ ! $descriptor =~ ^[1-9][0-9]*$ ||
    ! -r /proc/$$/fd/$descriptor ]]; then
    echo "PP2 recursive Python entry requires its private inherited capability" >&2
    exit 2
  fi

  local descriptor_target
  descriptor_target="$(readlink -- "/proc/$$/fd/$descriptor")"
  if [[ $descriptor_target != "${canonical_capability_directory}/pp2-python-shim."*' (deleted)' ]]; then
    echo "PP2 recursive Python capability is not the launcher's private unlinked file" >&2
    exit 2
  fi

  local descriptor_uid descriptor_mode descriptor_type
  IFS='|' read -r descriptor_uid descriptor_mode descriptor_type < <(
    stat -Lc '%u|%a|%F' "/proc/$$/fd/$descriptor"
  )
  if [[ $descriptor_uid != "$(id -u)" || $descriptor_mode != 600 ||
  $descriptor_type != "regular file" ]]; then
    echo "PP2 recursive Python capability has unsafe ownership or permissions" >&2
    exit 2
  fi
}

pp2_consume_private_shim_capability() {
  pp2_assert_private_shim_descriptor
  local descriptor="$DSV4_PP_PRIVATE_SHIM_FD"
  local magic creator_pid process_start_ticks launcher_sha256 nonce extra
  if ! IFS=' ' read -r magic creator_pid process_start_ticks launcher_sha256 nonce \
    <&"$descriptor"; then
    echo "PP2 recursive Python capability is empty" >&2
    exit 2
  fi
  if IFS= read -r extra <&"$descriptor"; then
    echo "PP2 recursive Python capability contains trailing data" >&2
    exit 2
  fi

  local current_launcher_sha256
  read -r current_launcher_sha256 _ < <(sha256sum -- "$script_path")
  if [[ $magic != dsv4_pp2_private_python_shim_v1 ||
    $creator_pid != "$$" ||
    $process_start_ticks != "$(pp2_process_start_ticks)" ||
    $launcher_sha256 != "$current_launcher_sha256" ||
    ! $nonce =~ ^[0-9a-f]{64}$ ]]; then
    echo "PP2 recursive Python capability does not match this one-shot launch" >&2
    exit 2
  fi

  exec {descriptor}<&-
  unset DSV4_PP_PRIVATE_SHIM_FD DSV4_PP_OPENCODE_PYTHON_SHIM
}

pp2_require_environment_value() {
  local variable_name="$1"
  local expected_value="$2"
  if [[ ! -v $variable_name || ${!variable_name} != "$expected_value" ]]; then
    echo "PP2 private shim requires $variable_name=$expected_value" >&2
    exit 2
  fi
}

pp2_require_cli_scalar() {
  local option="$1"
  local expected_value="$2"
  local count=0 value= argument_index
  for ((argument_index = 0; argument_index < ${#pp2_effective_arguments[@]}; argument_index++)); do
    if [[ ${pp2_effective_arguments[$argument_index]} == "$option" ]]; then
      if ((argument_index + 1 >= ${#pp2_effective_arguments[@]})); then
        echo "PP2 private shim found $option without a value" >&2
        exit 2
      fi
      count=$((count + 1))
      value="${pp2_effective_arguments[$((argument_index + 1))]}"
    elif [[ ${pp2_effective_arguments[$argument_index]} == "$option="* ]]; then
      echo "PP2 private shim rejects equals-form or duplicate override for $option" >&2
      exit 2
    fi
  done
  if ((count != 1)) || [[ $value != "$expected_value" ]]; then
    echo "PP2 private shim requires exactly one $option $expected_value" >&2
    exit 2
  fi
}

pp2_require_cli_sequence() {
  local option="$1"
  shift
  local -a expected_values=("$@")
  local count=0 argument_index value_index
  local -a observed_values=()
  for ((argument_index = 0; argument_index < ${#pp2_effective_arguments[@]}; argument_index++)); do
    if [[ ${pp2_effective_arguments[$argument_index]} == "$option" ]]; then
      count=$((count + 1))
      observed_values=()
      for ((value_index = argument_index + 1; value_index < ${#pp2_effective_arguments[@]}; value_index++)); do
        if [[ ${pp2_effective_arguments[$value_index]} == --* ]]; then
          break
        fi
        observed_values+=("${pp2_effective_arguments[$value_index]}")
      done
    elif [[ ${pp2_effective_arguments[$argument_index]} == "$option="* ]]; then
      echo "PP2 private shim rejects equals-form or duplicate override for $option" >&2
      exit 2
    fi
  done
  if ((count != 1 || ${#observed_values[@]} != ${#expected_values[@]})); then
    echo "PP2 private shim requires one exact $option sequence" >&2
    exit 2
  fi
  for ((value_index = 0; value_index < ${#expected_values[@]}; value_index++)); do
    if [[ ${observed_values[$value_index]} != "${expected_values[$value_index]}" ]]; then
      echo "PP2 private shim requires one exact $option sequence" >&2
      exit 2
    fi
  done
}

pp2_require_cli_flag_once() {
  local option="$1"
  local count=0 argument
  for argument in "${pp2_effective_arguments[@]}"; do
    if [[ $argument == "$option" ]]; then
      count=$((count + 1))
    elif [[ $argument == "$option="* ]]; then
      echo "PP2 private shim rejects a valued form of $option" >&2
      exit 2
    fi
  done
  if ((count != 1)); then
    echo "PP2 private shim requires exactly one $option" >&2
    exit 2
  fi
}

pp2_forbid_cli_option() {
  local option="$1"
  local argument
  for argument in "${pp2_effective_arguments[@]}"; do
    if [[ $argument == "$option" || $argument == "$option="* ]]; then
      echo "PP2 private shim forbids $option" >&2
      exit 2
    fi
  done
}

# This follow-up is deliberately local-only.  The generic base launcher falls
# back to /mnt/sanic, which is an NFS export on dwagon; default this PP entrypoint
# to the validated local checkpoint copy so an omitted variable cannot put model
# or tokenizer reads back on fwuff.  Callers may still provide another explicit
# local path for a later persistent copy.
export DSV4_MODEL_PATH="${DSV4_MODEL_PATH:-/tmp/dsv4-local-checkpoint-0731}"
if [[ ${DSV4_MODEL_PATH:0:1} != / ||
  ! -d $DSV4_MODEL_PATH ||
  -L $DSV4_MODEL_PATH ||
  ! -f ${DSV4_MODEL_PATH}/config.json ||
  -L ${DSV4_MODEL_PATH}/config.json ]]; then
  echo "PP2 requires an absolute, local, non-symlink model checkpoint" >&2
  exit 2
fi
case "$DSV4_MODEL_PATH" in
/mnt | /mnt/*)
  echo "PP2 refuses model reads from /mnt; use the local dwagon checkpoint" >&2
  exit 2
  ;;
esac
export DSV4_KT_WEIGHT_PATH="${DSV4_KT_WEIGHT_PATH:-$DSV4_MODEL_PATH}"
if [[ $DSV4_KT_WEIGHT_PATH != "$DSV4_MODEL_PATH" ]]; then
  echo "PP2 requires DSV4_KT_WEIGHT_PATH to equal the local DSV4_MODEL_PATH" >&2
  exit 2
fi

# The base launcher needs a Python executable once for its preparation helper
# and once for launch_server.  Recursive entry is therefore a private protocol,
# not a public boolean mode: only the exact preparation helper may pass without
# consuming the inherited capability, and launch_server consumes it before the
# ledger can authorize anything.
if [[ -v DSV4_PP_OPENCODE_PYTHON_SHIM ]]; then
  if [[ $DSV4_PP_OPENCODE_PYTHON_SHIM != private-v1 ]]; then
    echo "DSV4_PP_OPENCODE_PYTHON_SHIM is launcher-private and cannot be selected by a caller" >&2
    exit 2
  fi
  pp2_assert_private_shim_descriptor

  if [[ $# -ge 1 && $1 == "${repo_root}/scripts/prepare_dsv4_flash_0731.py" ]]; then
    exec "$real_python" "$@"
  fi
  if [[ $# -lt 3 || $1 != -u || $2 != -m || $3 != sglang.launch_server ]]; then
    echo "PP2 private Python shim only accepts its exact preparation or launch_server invocation" >&2
    exit 2
  fi

  pp_opencode_warmup_args=()
  pp_opencode_warmups="${DSV4_PP_OPENCODE_WARMUPS-dsv4_opencode_2694}"
  if [[ -n $pp_opencode_warmups ]]; then
    pp_opencode_warmup_args=(--warmups "$pp_opencode_warmups")
  fi
  pp2_effective_arguments=(
    "$@"
    "${pp_opencode_warmup_args[@]}"
    --disable-overlap-schedule
    --enable-p2p-check
    --pre-warm-nccl
    --served-model-name deepseek-v4-flash
    --reasoning-parser deepseek-v4
    --tool-call-parser deepseekv4
    --default-chat-template-kwargs '{"thinking":true}'
  )

  # Consume first: neither a failed validation nor a failed ledger update leaves
  # a reusable entry capability behind.
  pp2_consume_private_shim_capability

  pp2_require_environment_value DSV4_CACHE_ROOT "$canonical_cache_root"
  pp2_require_environment_value DSV4_PP_LAUNCH_LEDGER "$canonical_launch_ledger"
  pp2_require_environment_value DSV4_TENSOR_PARALLEL_SIZE 1
  pp2_require_environment_value DSV4_PIPELINE_PARALLEL_SIZE 2
  pp2_require_environment_value DSV4_EXPERT_PARALLEL_SIZE 1
  pp2_require_environment_value DSV4_CUDA_VISIBLE_DEVICES 0,1
  pp2_require_environment_value CUDA_VISIBLE_DEVICES 0,1
  pp2_require_environment_value DSV4_DISABLE_SPECULATIVE 1
  pp2_require_environment_value DSV4_CONTEXT_LENGTH 524288
  pp2_require_environment_value DSV4_MAX_TOTAL_TOKENS 524288
  pp2_require_environment_value DSV4_KV_CACHE_DTYPE fp8_e4m3
  pp2_require_environment_value DSV4_DECODE_GRAPH_BACKEND full
  pp2_require_environment_value DSV4_DECODE_GRAPH_MAX_BATCH_SIZE 2
  pp2_require_environment_value DSV4_DECODE_GRAPH_BATCH_SIZES "1 2"
  pp2_require_environment_value DSV4_PREFILL_GRAPH_BACKEND disabled
  pp2_require_environment_value DSV4_PREFILL_GRAPH_MAX 1024
  pp2_require_environment_value DSV4_PREFILL_GRAPH_TIERS "256 512 1024"
  pp2_require_environment_value DSV4_MAX_RUNNING_REQUESTS 2
  pp2_require_environment_value DSV4_PP_MAX_MICRO_BATCH_SIZE 1
  pp2_require_environment_value SGLANG_DSV4_OSCAR_INT2_KV_STORAGE 1
  pp2_require_environment_value SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY 1
  pp2_require_environment_value SGLANG_DSV4_INT4_KV_STORAGE 0
  pp2_require_environment_value SGLANG_DSV4_INT4_C4_INDEXER_STORAGE 0
  pp2_require_environment_value SGLANG_DSV4_SM86_C128_BF16_STORAGE 0
  pp2_require_environment_value SGLANG_OPT_USE_MULTI_STREAM_OVERLAP 0
  if [[ -n ${SGLANG_DSV4_OSCAR_CAPTURE_CONFIG:-} ]]; then
    echo "PP2 private shim forbids OSCAR capture mode during serving" >&2
    exit 2
  fi
  if [[ ! -f ${SGLANG_DSV4_OSCAR_CALIBRATION_PATH:-} ||
    ! -r ${SGLANG_DSV4_OSCAR_CALIBRATION_PATH:-} ||
    -L ${SGLANG_DSV4_OSCAR_CALIBRATION_PATH:-} ||
    ! -f ${SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH:-} ||
    ! -r ${SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH:-} ||
    -L ${SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH:-} ]]; then
    echo "PP2 private shim requires readable non-symlink OSCAR calibration and admission receipts" >&2
    exit 2
  fi
  case "${DSV4_PIPELINE_LAYER_PARTITION:-}" in
  21,22 | 22,21) ;;
  *)
    echo "PP2 private shim found an invalid pipeline layer partition" >&2
    exit 2
    ;;
  esac
  pp2_require_environment_value SGLANG_PP_LAYER_PARTITION "$DSV4_PIPELINE_LAYER_PARTITION"
  case "${DSV4_PP_ASYNC_BATCH_DEPTH:-}" in
  0 | 1) ;;
  *)
    echo "PP2 private shim found an invalid async batch depth" >&2
    exit 2
    ;;
  esac
  case "${DSV4_CHUNKED_PREFILL_SIZE:-}" in
  512 | 1024) ;;
  *)
    echo "PP2 private shim found an invalid chunked prefill size" >&2
    exit 2
    ;;
  esac

  case "${DSV4_PP_RUN_ROLE:-}" in
  transfer)
    expected_launch_authorization="$canonical_transfer_authorization"
    ;;
  optimized)
    expected_launch_authorization="$canonical_optimized_authorization"
    ;;
  *)
    echo "DSV4_PP_RUN_ROLE must be transfer or optimized for a PP2 model launch" >&2
    exit 2
    ;;
  esac
  pp2_require_environment_value \
    DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT "$expected_launch_authorization"
  if [[ -v DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT ]]; then
    echo "PP2 private shim refuses a pre-existing reusable authorization receipt" >&2
    exit 2
  fi

  pp2_require_cli_scalar --tensor-parallel-size 1
  pp2_require_cli_scalar --pp-size 2
  pp2_require_cli_scalar --pp-max-micro-batch-size 1
  pp2_require_cli_scalar --ep-size 1
  pp2_require_cli_scalar --context-length 524288
  pp2_require_cli_scalar --max-total-tokens 524288
  pp2_require_cli_scalar --kv-cache-dtype fp8_e4m3
  pp2_require_cli_scalar --chunked-prefill-size "$DSV4_CHUNKED_PREFILL_SIZE"
  pp2_require_cli_scalar --max-prefill-tokens "$DSV4_CHUNKED_PREFILL_SIZE"
  pp2_require_cli_scalar --max-running-requests 2
  pp2_require_cli_scalar --cuda-graph-backend-decode full
  pp2_require_cli_scalar --cuda-graph-max-bs-decode 2
  pp2_require_cli_sequence --cuda-graph-bs-decode 1 2
  pp2_require_cli_scalar --cuda-graph-backend-prefill disabled
  pp2_require_cli_scalar --cuda-graph-max-bs-prefill 1024
  pp2_require_cli_sequence --cuda-graph-bs-prefill 256 512 1024
  pp2_require_cli_scalar --served-model-name deepseek-v4-flash
  pp2_require_cli_scalar --reasoning-parser deepseek-v4
  pp2_require_cli_scalar --tool-call-parser deepseekv4
  pp2_require_cli_scalar --default-chat-template-kwargs '{"thinking":true}'
  if [[ $DSV4_PP_ASYNC_BATCH_DEPTH == 0 ]]; then
    pp2_forbid_cli_option --pp-async-batch-depth
  else
    pp2_require_cli_scalar --pp-async-batch-depth 1
  fi
  if [[ -n $pp_opencode_warmups ]]; then
    pp2_require_cli_scalar --warmups "$pp_opencode_warmups"
  else
    pp2_forbid_cli_option --warmups
  fi
  pp2_require_cli_flag_once --disable-overlap-schedule
  pp2_require_cli_flag_once --enable-p2p-check
  pp2_require_cli_flag_once --pre-warm-nccl

  for forbidden_override in \
    --config \
    --tp-size \
    --pipeline-parallel-size \
    --expert-parallel-size \
    --ep \
    --disable-cuda-graph \
    --disable-decode-cuda-graph \
    --disable-prefill-cuda-graph \
    --enable-breakable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --enforce-piecewise-cuda-graph \
    --cuda-graph-config \
    --cuda-graph-max-bs \
    --cuda-graph-bs \
    --piecewise-cuda-graph-tokens \
    --enable-hisparse; do
    pp2_forbid_cli_option "$forbidden_override"
  done
  for effective_argument in "${pp2_effective_arguments[@]}"; do
    if [[ $effective_argument == --speculative-* ]]; then
      echo "PP2 private shim forbids speculative launch arguments" >&2
      exit 2
    fi
  done

  pp_first_benchmark_args=()
  if [[ -n ${DSV4_PP_FIRST_BENCHMARK_RECEIPT:-} ]]; then
    pp_first_benchmark_args=(
      --first-benchmark-receipt "$DSV4_PP_FIRST_BENCHMARK_RECEIPT"
    )
  fi
  pp_launch_authorization="$("$real_python" \
    "${script_dir}/dsv4_pp2_followup_ledger.py" \
    --ledger "$DSV4_PP_LAUNCH_LEDGER" \
    --output "$DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT" \
    --ep-confirmation-receipt "$DSV4_PP_EP_WINNER_RECEIPT" \
    --ep-coherency-receipt "$DSV4_PP_EP_COHERENCY_RECEIPT" \
    --source-ep2-plan "$DSV4_PP_EP2_WINNER_PLAN" \
    --transferred-plan "$SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN" \
    --run-role "$DSV4_PP_RUN_ROLE" \
    --pipeline-layer-partition "$DSV4_PIPELINE_LAYER_PARTITION" \
    --pp-async-batch-depth "$DSV4_PP_ASYNC_BATCH_DEPTH" \
    --chunked-prefill-size "$DSV4_CHUNKED_PREFILL_SIZE" \
    --native-artifact "$DSV4_KT_CPU_OPTIMIZED_CANDIDATE" \
    --native-artifact-sha256 "$DSV4_KT_CPU_OPTIMIZED_SHA256" \
    "${pp_first_benchmark_args[@]}")"
  if [[ -z $pp_launch_authorization ||
    $pp_launch_authorization != "$expected_launch_authorization" ||
    $pp_launch_authorization == *$'\n'* ||
    ! -f $pp_launch_authorization ||
    ! -r $pp_launch_authorization ||
    -L $pp_launch_authorization ]]; then
    echo "PP2 launch controller did not produce the canonical authorization receipt" >&2
    exit 1
  fi
  export DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT="$pp_launch_authorization"
  exec "$real_python" "${pp2_effective_arguments[@]}"
fi

if [[ ! -x $real_python ]]; then
  echo "DeepSeek V4 Python interpreter is not executable: $real_python" >&2
  exit 1
fi

cache_root="$canonical_cache_root"
gpu_experts_per_layer="${DSV4_GPU_EXPERTS_PER_LAYER:-28}"
pipeline_layer_partition="${DSV4_PIPELINE_LAYER_PARTITION:-21,22}"
source_ep2_winner_plan="${DSV4_PP_EP2_WINNER_PLAN-}"
source_ep_winner_receipt="${DSV4_PP_EP_WINNER_RECEIPT-}"
source_ep_coherency_receipt="${DSV4_PP_EP_COHERENCY_RECEIPT-}"
transferred_plan_cache_root="${DSV4_PP_TRANSFERRED_PLAN_CACHE_ROOT:-${cache_root}/transferred-ep2-winner-plans}"

if [[ -z $source_ep_winner_receipt ||
  ${source_ep_winner_receipt:0:1} != / ||
  ! -f $source_ep_winner_receipt ||
  ! -r $source_ep_winner_receipt ||
  -L $source_ep_winner_receipt ]]; then
  echo "DSV4_PP_EP_WINNER_RECEIPT must name the readable absolute non-symlink final EP confirmation" >&2
  exit 2
fi
if [[ -z $source_ep_coherency_receipt ||
  ${source_ep_coherency_receipt:0:1} != / ||
  ! -f $source_ep_coherency_receipt ||
  ! -r $source_ep_coherency_receipt ||
  -L $source_ep_coherency_receipt ]]; then
  echo "DSV4_PP_EP_COHERENCY_RECEIPT must name the readable absolute non-symlink final EP coherency proof" >&2
  exit 2
fi
if [[ -z $source_ep2_winner_plan ]]; then
  echo "DSV4_PP_EP2_WINNER_PLAN must name the plan bound by the final EP confirmation" >&2
  exit 2
fi

if [[ ! $gpu_experts_per_layer =~ ^(0|[1-9][0-9]{0,2})$ ]]; then
  echo "DSV4_GPU_EXPERTS_PER_LAYER must be a decimal integer between 0 and 256" >&2
  exit 2
fi
gpu_experts_per_layer=$((10#$gpu_experts_per_layer))
if ((gpu_experts_per_layer > 256)); then
  echo "DSV4_GPU_EXPERTS_PER_LAYER must be a decimal integer between 0 and 256" >&2
  exit 2
fi
case "$pipeline_layer_partition" in
21,22 | 22,21) ;;
*)
  echo "DSV4_PIPELINE_LAYER_PARTITION must be 21,22 or 22,21" >&2
  exit 2
  ;;
esac

if [[ -v DSV4_PP_HYBRID_EXPERT_SHARD_PLAN ]]; then
  echo "DSV4_PP_EP2_WINNER_PLAN conflicts with DSV4_PP_HYBRID_EXPERT_SHARD_PLAN" >&2
  exit 2
fi
if [[ -v DSV4_PP_HYBRID_EXPERT_PROFILE ||
  -v DSV4_PP_HYBRID_EXPERT_FILL_PROFILE ||
  -v DSV4_PP_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER ]]; then
  echo "DSV4_PP_EP2_WINNER_PLAN conflicts with PP2 profile-based placement inputs" >&2
  exit 2
fi

if [[ ! -f $source_ep2_winner_plan || ! -r $source_ep2_winner_plan ||
  -L $source_ep2_winner_plan ]]; then
  echo "EP2 winner plan is not a readable non-symlink file: $source_ep2_winner_plan" >&2
  exit 2
fi
shard_plan="$("$real_python" \
  "${repo_root}/scripts/transfer_dsv4_ep2_plan_to_pp2.py" \
  --source-ep2-plan "$source_ep2_winner_plan" \
  --source-ep-confirmation-receipt "$source_ep_winner_receipt" \
  --source-ep-coherency-receipt "$source_ep_coherency_receipt" \
  --cache-root "$transferred_plan_cache_root" \
  --expected-target-gpu-experts-per-layer "$gpu_experts_per_layer")"
if [[ -z $shard_plan || $shard_plan == *$'\n'* ||
  ! -f $shard_plan || ! -r $shard_plan || -L $shard_plan ]]; then
  echo "transferred PP2 winner plan is not a readable file: ${shard_plan:-<empty>}" >&2
  exit 1
fi

# Carry the confirmed native tuple verbatim. PP may change one pipeline-only
# knob on its second launch, but it must not turn that run into a CPU-kernel A/B.
qualified_native_artifact=/var/lib/exo/experiments/dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
qualified_native_sha256=7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043
if [[ -v DSV4_KTRANSFORMERS_SOURCE ]]; then
  echo "DSV4_KTRANSFORMERS_SOURCE cannot bypass the receipt-bound PP2 native artifact" >&2
  exit 2
fi
if [[ ${DSV4_KT_CPU_OPTIMIZED_CANDIDATE:-$qualified_native_artifact} != "$qualified_native_artifact" ]]; then
  echo "DSV4_KT_CPU_OPTIMIZED_CANDIDATE must match the confirmed PP2 artifact" >&2
  exit 2
fi
for legacy_overlay in \
  DSV4_STAGE_KT_AVX_TAIL_OVERLAY \
  DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY \
  DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY; do
  case "${!legacy_overlay:-0}" in
  0 | false | FALSE | no | NO | n | N | "") ;;
  *)
    echo "$legacy_overlay conflicts with the receipt-bound PP2 native artifact" >&2
    exit 2
    ;;
  esac
done
case "${DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY:-1}" in
1 | true | TRUE | yes | YES | y | Y) ;;
*)
  echo "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY must remain enabled for PP2" >&2
  exit 2
  ;;
esac
staged_kt_source="$("$real_python" \
  "${repo_root}/scripts/stage_dsv4_kt_avx_tail_overlay.py" \
  --candidate "$qualified_native_artifact" \
  --expected-sha256 "$qualified_native_sha256" \
  --cache-root "${DSV4_KT_CPU_OPTIMIZED_CACHE_ROOT:-/var/lib/exo/cache/dsv4-cpu-optimized-serving-overlays}")"
if [[ -z $staged_kt_source || $staged_kt_source == *$'\n'* ||
  ! -d ${staged_kt_source}/kt_kernel ]]; then
  echo "staged DSV4 KT CPU-optimized overlay is not importable: ${staged_kt_source:-<empty>}" >&2
  exit 1
fi
export DSV4_KTRANSFORMERS_SOURCE="$staged_kt_source"
export DSV4_KT_CPU_OPTIMIZED_CANDIDATE="$qualified_native_artifact"
export DSV4_KT_CPU_OPTIMIZED_SHA256="$qualified_native_sha256"
export DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY=1
export DSV4_STAGE_KT_AVX_TAIL_OVERLAY=0
export DSV4_STAGE_KT_TASK_QUEUE_PIN_OVERLAY=0
export DSV4_STAGE_KT_PERSISTENT_COUNTER_OVERLAY=0
export KT_TASK_QUEUE_PIN_FIRST_CORE=1
export KT_SINGLE_NUMA_INLINE_DISPATCH=1
export KT_MXFP4_AVX_SCALE_FOLD_MODE=lut-v1

export DSV4_PP_OPENCODE_REAL_PYTHON="$real_python"
export DSV4_LAUNCH_ENTRYPOINT="$script_path"
export DSV4_PP_EP_WINNER_RECEIPT="$source_ep_winner_receipt"
export DSV4_PP_EP_COHERENCY_RECEIPT="$source_ep_coherency_receipt"
export DSV4_PP_EP2_WINNER_PLAN="$source_ep2_winner_plan"

# TP1/EP1 per stage. The union of both EP2 rank placements over half the layers
# keeps approximately the same per-GPU expert-weight budget and exact global hot
# set as the admitted winner across every layer in the TP2/EP2 configuration.
export DSV4_TENSOR_PARALLEL_SIZE=1
export DSV4_PIPELINE_PARALLEL_SIZE=2
export DSV4_EXPERT_PARALLEL_SIZE=1
export DSV4_PIPELINE_LAYER_PARTITION="$pipeline_layer_partition"
export DSV4_CUDA_VISIBLE_DEVICES=0,1
export DSV4_GPU_EXPERTS_PER_LAYER="$gpu_experts_per_layer"
# The generic launcher accepts a distinct allocation ceiling for variable-width
# plans. Fail closed on stale inherited state by making it exactly the maximum
# union width already validated by the transfer helper.
export DSV4_GPU_EXPERTS_MAX_PER_LAYER="$gpu_experts_per_layer"
export DSV4_EXPERT_LOCATION_MODE=hybrid
export SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN="$shard_plan"
unset SGLANG_KT_CPU_EXPERT_SHARD_PLAN SGLANG_KT_GPU_EXPERT_MASK_PLAN
unset SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN
unset DSV4_DRAFT_HYBRID_EXPERT_PROFILE DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN

# The rank-local KT selector maps PP0 to socket 0 and PP1 to socket 1. Each
# process receives one admitted AMX pool after distributed rank discovery. The
# offline policy screen rejected 72 workers, so PP2 transfers only the qualified
# 56-thread policy.
cpuinfer_threads="${DSV4_CPUINFER_THREADS:-56}"
case "$cpuinfer_threads" in
56) ;;
*)
  echo "DSV4_CPUINFER_THREADS must be exactly 56 for the PP2 OSCAR study" >&2
  exit 2
  ;;
esac
export DSV4_CPUINFER_THREADS="$cpuinfer_threads"
export DSV4_KT_THREADPOOL_COUNT=2
export DSV4_KT_NUMA_NODES="0 1"
export DSV4_NUMACTL_NODES=0,1
export DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN=0
if [[ ${KT_WORKER_SPIN_US:-1000} != 1000 ]]; then
  echo "KT_WORKER_SPIN_US must remain 1000 for the confirmed PP2 tuple" >&2
  exit 2
fi
if [[ ${KT_AMX_FINE_GRAINED_DECODE:-1} != 1 ]]; then
  echo "KT_AMX_FINE_GRAINED_DECODE must remain enabled for the confirmed PP2 tuple" >&2
  exit 2
fi
export KT_WORKER_SPIN_US=1000
export KT_AMX_FINE_GRAINED_DECODE=1
# AMX wins from five routed rows upward on this host. The pinned AVX-tail
# candidate wins for the common two- and three-row tail below that cutoff.
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS="${KT_MXFP4_AMX_MIN_EXPERT_TOKENS:-5}"
export KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS="${KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS:-2}"
export SGLANG_V4_MXFP4_SMALL_ROW_ROUTING=1
export SGLANG_V4_MXFP4_SM86_SMALL_BATCH_GEMM=1
# PP2 must transfer the winning OSCAR cache contract as well as its expert
# placement.  Generic INT4 and selective-C128 prototypes are not compatible
# baselines for the required follow-up runs and cannot be enabled here.
export DSV4_KV_CACHE_DTYPE="${DSV4_KV_CACHE_DTYPE:-fp8_e4m3}"
# fp8_e4m3 is only the public SGLang raw-byte carrier.  The admitted physical
# history layout remains calibrated asymmetric OSCAR INT2 on both PP stages;
# reject rather than overwrite inherited generic FP8/BF16 cache selections.
if [[ $DSV4_KV_CACHE_DTYPE != fp8_e4m3 ]]; then
  echo "PP2 OSCAR-INT2 requires DSV4_KV_CACHE_DTYPE=fp8_e4m3 as its raw-byte carrier" >&2
  exit 2
fi
case "${SGLANG_DSV4_OSCAR_INT2_KV_STORAGE:-1}" in
1 | true | TRUE | yes | YES | y | Y) ;;
*)
  echo "the PP2 transfer requires SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1" >&2
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
default_oscar_calibration=/var/lib/exo/cache/dsv4-flash-hybrid-ep2/oscar-int2/dsv4-oscar-int2-calibration.pt
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
  echo "PP2 OSCAR-INT2 requires the readable, absolute, non-symlink EP2 calibration artifact: ${oscar_calibration_path:-<empty>}" >&2
  exit 2
fi
oscar_root="${oscar_calibration_path%/*}"
oscar_fingerprint_path="${DSV4_OSCAR_CHECKPOINT_FINGERPRINT_PATH:-${oscar_root}/checkpoint-fingerprint.json}"
oscar_admission_path="${DSV4_OSCAR_ADMISSION_RECEIPT_PATH:-${oscar_root}/admission.json}"
oscar_model_id="${DSV4_OSCAR_MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}"
if [[ $oscar_model_id != deepseek-ai/DeepSeek-V4-Flash ]]; then
  echo "PP2 OSCAR-INT2 requires DSV4_OSCAR_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash" >&2
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
    echo "PP2 OSCAR-INT2 requires the admitted EP2 checkpoint fingerprint: ${oscar_fingerprint_path:-<empty>}" >&2
    exit 2
  fi
  if [[ -z $oscar_admission_path || ${oscar_admission_path:0:1} != / ||
    -L $oscar_admission_path || ! -d ${oscar_admission_path%/*} ||
    ! -w ${oscar_admission_path%/*} ]]; then
    echo "PP2 OSCAR-INT2 admission receipt must target a writable absolute non-symlink path: ${oscar_admission_path:-<empty>}" >&2
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
    echo "PP2 OSCAR-INT2 model-bound admission did not produce a readable receipt" >&2
    exit 1
  fi
fi
export SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1
# Mandatory transfer of the confirmed SM86 split-history path. The arena is a
# fixed 4,210,688-byte FP32 accumulation workspace per PP process; each process
# owns its own address-stable allocation for decode graph capture/replay.
case "${SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY:-1}" in
1 | true | TRUE | yes | YES | y | Y) ;;
*)
  echo "the PP2 transfer requires SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1" >&2
  exit 2
  ;;
esac
export SGLANG_DSV4_OSCAR_INT2_SPLIT_HISTORY=1
export SGLANG_DSV4_OSCAR_CALIBRATION_PATH="$oscar_calibration_path"
export SGLANG_DSV4_OSCAR_ADMISSION_RECEIPT_PATH="$oscar_admission_path"
export SGLANG_DSV4_INT4_C4_INDEXER_STORAGE=0
export SGLANG_DSV4_INT4_KV_STORAGE=0
export SGLANG_DSV4_SM86_C128_BF16_STORAGE=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0

# One request per pipeline microbatch lets two simultaneous requests fill the
# two stages. Capture both scheduler batch shapes: SGLang can merge the two
# lanes into a BS2 decode batch even though PP microbatches remain size one.
export DSV4_DISABLE_SPECULATIVE=1
if [[ ${DSV4_MAX_RUNNING_REQUESTS:-2} != 2 ]]; then
  echo "DSV4_MAX_RUNNING_REQUESTS must remain 2 for the PP2 concurrency study" >&2
  exit 2
fi
if [[ ${DSV4_PP_MAX_MICRO_BATCH_SIZE:-1} != 1 ]]; then
  echo "DSV4_PP_MAX_MICRO_BATCH_SIZE must remain 1 for the PP2 concurrency study" >&2
  exit 2
fi
case "${DSV4_PP_ASYNC_BATCH_DEPTH:-0}" in
0 | 1) ;;
*)
  echo "DSV4_PP_ASYNC_BATCH_DEPTH must be 0 or 1" >&2
  exit 2
  ;;
esac
export DSV4_MAX_RUNNING_REQUESTS=2
export DSV4_PP_MAX_MICRO_BATCH_SIZE=1
export DSV4_PP_ASYNC_BATCH_DEPTH="${DSV4_PP_ASYNC_BATCH_DEPTH:-0}"
export DSV4_DECODE_GRAPH_BATCH_SIZES="1 2"
export DSV4_DECODE_GRAPH_MAX_BATCH_SIZE=2

export DSV4_CONTEXT_LENGTH=524288
export DSV4_MAX_TOTAL_TOKENS=524288
# 2,560 SWA slots cover the local admission floor for a 1,024-token chunk:
# page_size + 2 * max(sliding_window_size, chunk) = 256 + 2 * 1,024 = 2,304.
# The next larger 2,048-token chunk would require at least 4,352 SWA slots and
# does not fit this accepted OpenCode reserve.
export DSV4_SWA_FULL_TOKENS_RATIO=0.0048828125
# Transfer the accepted OpenCode chunk and tier ceiling even though PP keeps
# the prefill graph backend disabled.
case "${DSV4_CHUNKED_PREFILL_SIZE:-1024}" in
512 | 1024) ;;
*)
  echo "DSV4_CHUNKED_PREFILL_SIZE must be 512 or 1024 at the fixed SWA reserve" >&2
  exit 2
  ;;
esac
export DSV4_CHUNKED_PREFILL_SIZE="${DSV4_CHUNKED_PREFILL_SIZE:-1024}"
if [[ ${DSV4_PREFILL_GRAPH_TIERS:-256 512 1024} != "256 512 1024" ]]; then
  echo "DSV4_PREFILL_GRAPH_TIERS must remain exactly 256 512 1024 for PP2" >&2
  exit 2
fi
if [[ ${DSV4_PREFILL_GRAPH_MAX:-1024} != 1024 ]]; then
  echo "DSV4_PREFILL_GRAPH_MAX must remain exactly 1024 for PP2" >&2
  exit 2
fi
export DSV4_PREFILL_GRAPH_TIERS="256 512 1024"
export DSV4_PREFILL_GRAPH_MAX=1024
# Local ServerArgs rejects tc_piecewise prefill graphs for pp_size > 1. Its
# breakable backend is mechanically PP-capable, but DeepSeek V4 is explicitly
# auto-disabled because c4-indexer scratch remains pinned in the capture pool
# and OOMs. Full prefill capture is experimental. Keep prefill eager/chunked;
# this does not disable the BS1 target decode graph below.
export DSV4_PREFILL_GRAPH_BACKEND=disabled
export DSV4_DECODE_GRAPH_BACKEND=full
export DSV4_MEM_FRACTION_STATIC=0.90
export DSV4_DISABLE_RADIX_CACHE=0
# Carry the qualified agent-serving warmup into PP: the internal 2,694-token
# request primes the eager three-chunk prefill and BS1 decode graph before the
# two-lane endpoint advertises readiness. An explicit empty value opts out.
export DSV4_SKIP_SERVER_WARMUP="${DSV4_SKIP_SERVER_WARMUP:-0}"
if [[ ! -v DSV4_PP_OPENCODE_WARMUPS ]]; then
  export DSV4_PP_OPENCODE_WARMUPS=dsv4_opencode_2694
fi
export DSV4_CACHE_ROOT="$cache_root"

if [[ $launch_requested == 1 ]]; then
  case "${DSV4_PP_RUN_ROLE:-}" in
  transfer | optimized) ;;
  *)
    echo "DSV4_PP_RUN_ROLE must be transfer or optimized for a PP2 model launch" >&2
    exit 2
    ;;
  esac
  export DSV4_PP_RUN_ROLE
  case "$DSV4_PP_RUN_ROLE" in
  transfer)
    expected_launch_authorization="$canonical_transfer_authorization"
    ;;
  optimized)
    expected_launch_authorization="$canonical_optimized_authorization"
    ;;
  esac
  if [[ -v DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT &&
    $DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT != "$expected_launch_authorization" ]]; then
    echo "PP2 authorization output does not match run role $DSV4_PP_RUN_ROLE" >&2
    exit 2
  fi
  export DSV4_PP_LAUNCH_AUTHORIZATION_OUTPUT="$expected_launch_authorization"
  unset DSV4_PP_LAUNCH_AUTHORIZATION_RECEIPT

  mkdir -p -- "$canonical_capability_directory"
  chmod 700 -- "$canonical_capability_directory"
  private_shim_capability_path="$(
    mktemp "${canonical_capability_directory}/pp2-python-shim.XXXXXXXX"
  )"
  trap 'rm -f -- "$private_shim_capability_path"' EXIT
  chmod 600 -- "$private_shim_capability_path"
  read -r private_shim_launcher_sha256 _ < <(sha256sum -- "$script_path")
  private_shim_nonce="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  if [[ ! $private_shim_launcher_sha256 =~ ^[0-9a-f]{64}$ ||
    ! $private_shim_nonce =~ ^[0-9a-f]{64}$ ]]; then
    echo "could not create the PP2 private Python capability" >&2
    exit 1
  fi
  printf '%s %s %s %s %s\n' \
    dsv4_pp2_private_python_shim_v1 \
    "$$" \
    "$(pp2_process_start_ticks)" \
    "$private_shim_launcher_sha256" \
    "$private_shim_nonce" \
    >"$private_shim_capability_path"
  exec {private_shim_descriptor}<"$private_shim_capability_path"
  rm -f -- "$private_shim_capability_path"
  trap - EXIT
  export DSV4_PP_PRIVATE_SHIM_FD="$private_shim_descriptor"
  export DSV4_PP_OPENCODE_PYTHON_SHIM=private-v1
  export DSV4_PYTHON="$script_path"
else
  unset DSV4_PP_OPENCODE_PYTHON_SHIM DSV4_PP_PRIVATE_SHIM_FD
  export DSV4_PYTHON="$real_python"
fi

exec "${repo_root}/scripts/dsv4_flash_fwuff_parity.sh" "$@"
