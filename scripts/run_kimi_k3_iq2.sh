#!/usr/bin/env bash
set -euo pipefail

# Reproducible all-local Kimi K3 UD-IQ2_XXS launcher for dwagon.
# This is a receipt-bound legacy binary replay. Build new llama.cpp work from
# vendor/llama.cpp on the authoritative branch declared in AGENTS.md.
# The default MoE-aware fit keeps routed experts on CPU and uses the two
# RTX 3090s for dense tensors. Arguments after `--` are appended verbatim so
# an observed fit failure can be retried with the documented fixed placement.

readonly default_runtime_root=/var/lib/exo/sources/llama.cpp-kimik3-text-47c5bbdf
readonly default_model_path=/mnt/sanic-edr/llm_models/Kimi-K3-GGUF/UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf
readonly default_runtime_commit=47c5bbdfd5ab5e847098f791a2cb9c0c90fb7dbd
readonly default_model_bytes=711067773664

runtime_root=${KIMI_IQ2_RUNTIME_ROOT:-"${default_runtime_root}"}
server_binary=${KIMI_IQ2_SERVER_BINARY:-"${runtime_root}/build-kimik3-text/bin/llama-server"}
model_path=${KIMI_IQ2_MODEL_PATH:-"${default_model_path}"}
expected_runtime_commit=${KIMI_IQ2_EXPECTED_RUNTIME_COMMIT:-"${default_runtime_commit}"}
expected_model_bytes=${KIMI_IQ2_EXPECTED_MODEL_BYTES:-"${default_model_bytes}"}

device_list=${KIMI_IQ2_DEVICE_LIST:-CUDA0,CUDA1}
fit_target_mib=${KIMI_IQ2_FIT_TARGET_MIB:-3072}
processor_bind=${KIMI_IQ2_PROCESSOR_BIND:-0-111}
memory_nodes=${KIMI_IQ2_MEMORY_NODES:-all}
generation_threads=${KIMI_IQ2_THREADS:-112}
batch_threads=${KIMI_IQ2_BATCH_THREADS:-112}
minimum_host_available_mib=${KIMI_IQ2_MINIMUM_HOST_AVAILABLE_MIB:-710000}
minimum_gpu_free_mib=${KIMI_IQ2_MINIMUM_GPU_FREE_MIB:-23000}

context_size=${KIMI_IQ2_CONTEXT_SIZE:-32768}
batch_size=${KIMI_IQ2_BATCH_SIZE:-4096}
physical_batch_size=${KIMI_IQ2_UBATCH_SIZE:-512}
context_checkpoint_count=${KIMI_IQ2_CONTEXT_CHECKPOINTS:-1}
parallel_slots=${KIMI_IQ2_PARALLEL_SLOTS:-1}

listen_host=${KIMI_IQ2_LISTEN_HOST:-127.0.0.1}
listen_port=${KIMI_IQ2_LISTEN_PORT:-11434}
model_alias=${KIMI_IQ2_MODEL_ALIAS:-Kimi-K3-UD-IQ2_XXS}
thinking_effort=${KIMI_IQ2_THINKING_EFFORT:-high}
random_seed=${KIMI_IQ2_SEED:-3407}
temperature=${KIMI_IQ2_TEMPERATURE:-1.0}
top_probability=${KIMI_IQ2_TOP_P:-0.95}
top_token_count=${KIMI_IQ2_TOP_K:-50}
minimum_probability=${KIMI_IQ2_MIN_P:-0.0}
request_timeout_seconds=${KIMI_IQ2_TIMEOUT_SECONDS:-21600}
sse_ping_interval_seconds=${KIMI_IQ2_SSE_PING_INTERVAL_SECONDS:-15}
device_check_timeout_seconds=${KIMI_IQ2_DEVICE_CHECK_TIMEOUT_SECONDS:-15}
log_file=${KIMI_IQ2_LOG_FILE:-}
slot_save_path=${KIMI_IQ2_SLOT_SAVE_PATH:-/tmp/exo-kimi-k3-slot-state-iq2}

skip_model_check=${KIMI_IQ2_SKIP_MODEL_CHECK:-0}
skip_device_check=${KIMI_IQ2_SKIP_DEVICE_CHECK:-0}
allow_other_host=${KIMI_IQ2_ALLOW_OTHER_HOST:-0}

validate_only=0
print_command=0
extra_server_arguments=()

print_usage() {
  cat <<'USAGE'
Usage:
  scripts/run_kimi_k3_iq2.sh [--validate-only] [--print-command] [--] [LLAMA_ARGUMENT ...]

Modes:
  --validate-only  Check the pinned binary, complete model, host capacity, and
                   both local GPUs without loading the model.
  --print-command  Validate, then print the shell-escaped server command.
  --help           Show this help.

The default placement is MoE-aware auto-fit:
  --device CUDA0,CUDA1 --split-mode layer --fit on --fit-target 3072
  --load-mode none --ubatch-size 512 --no-op-offload

If auto-fit fails, append this exact deterministic fallback:
  -- --fit off --n-cpu-moe 93 --gpu-layers 63 --tensor-split 32,31

Set KIMI_IQ2_SKIP_MODEL_CHECK=1 only while the pinned transfer is incomplete.
Arguments after `--` are appended verbatim and can intentionally override the
frozen defaults after a real observed failure.
USAGE
}

log() {
  printf '[kimi-iq2] %s\n' "$*" >&2
}

fail() {
  printf '[kimi-iq2] error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name=$1
  command -v "${command_name}" >/dev/null ||
    fail "required command is unavailable: ${command_name}"
}

require_unsigned_integer() {
  local variable_name=$1
  local value=$2
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${variable_name} must be an unsigned integer, got: ${value}"
}

print_shell_command() {
  printf '%q ' "$@"
  printf '\n'
}

while (($# > 0)); do
  case "$1" in
  --validate-only)
    validate_only=1
    ;;
  --print-command)
    print_command=1
    ;;
  --help)
    print_usage
    exit 0
    ;;
  --)
    shift
    extra_server_arguments=("$@")
    break
    ;;
  *)
    printf 'Unknown argument: %s\n\n' "$1" >&2
    print_usage >&2
    exit 2
    ;;
  esac
  shift
done

for integer_setting in \
  "KIMI_IQ2_EXPECTED_MODEL_BYTES:${expected_model_bytes}" \
  "KIMI_IQ2_FIT_TARGET_MIB:${fit_target_mib}" \
  "KIMI_IQ2_THREADS:${generation_threads}" \
  "KIMI_IQ2_BATCH_THREADS:${batch_threads}" \
  "KIMI_IQ2_MINIMUM_HOST_AVAILABLE_MIB:${minimum_host_available_mib}" \
  "KIMI_IQ2_MINIMUM_GPU_FREE_MIB:${minimum_gpu_free_mib}" \
  "KIMI_IQ2_CONTEXT_SIZE:${context_size}" \
  "KIMI_IQ2_BATCH_SIZE:${batch_size}" \
  "KIMI_IQ2_UBATCH_SIZE:${physical_batch_size}" \
  "KIMI_IQ2_CONTEXT_CHECKPOINTS:${context_checkpoint_count}" \
  "KIMI_IQ2_PARALLEL_SLOTS:${parallel_slots}" \
  "KIMI_IQ2_LISTEN_PORT:${listen_port}" \
  "KIMI_IQ2_SEED:${random_seed}" \
  "KIMI_IQ2_TOP_K:${top_token_count}" \
  "KIMI_IQ2_TIMEOUT_SECONDS:${request_timeout_seconds}" \
  "KIMI_IQ2_SSE_PING_INTERVAL_SECONDS:${sse_ping_interval_seconds}" \
  "KIMI_IQ2_DEVICE_CHECK_TIMEOUT_SECONDS:${device_check_timeout_seconds}"; do
  require_unsigned_integer "${integer_setting%%:*}" "${integer_setting#*:}"
done

[[ ${expected_runtime_commit} =~ ^[0-9a-f]{40}$ ]] ||
  fail "KIMI_IQ2_EXPECTED_RUNTIME_COMMIT must be a full lowercase Git commit"
for floating_setting in \
  "KIMI_IQ2_TEMPERATURE:${temperature}" \
  "KIMI_IQ2_TOP_P:${top_probability}" \
  "KIMI_IQ2_MIN_P:${minimum_probability}"; do
  [[ ${floating_setting#*:} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    fail "${floating_setting%%:*} must be a nonnegative decimal, got: ${floating_setting#*:}"
done
for boolean_setting in \
  "KIMI_IQ2_SKIP_MODEL_CHECK:${skip_model_check}" \
  "KIMI_IQ2_SKIP_DEVICE_CHECK:${skip_device_check}" \
  "KIMI_IQ2_ALLOW_OTHER_HOST:${allow_other_host}"; do
  [[ ${boolean_setting#*:} =~ ^[01]$ ]] ||
    fail "${boolean_setting%%:*} must be 0 or 1, got: ${boolean_setting#*:}"
done

((fit_target_mib > 0)) ||
  fail "KIMI_IQ2_FIT_TARGET_MIB must be positive"
((listen_port > 0 && listen_port <= 65535)) ||
  fail "listen port must be between 1 and 65535"
((physical_batch_size <= batch_size)) ||
  fail "physical batch size cannot exceed logical batch size"
((batch_size <= context_size)) ||
  fail "logical batch size cannot exceed context size"
[[ ${thinking_effort} =~ ^(low|high|max)$ ]] ||
  fail "KIMI_IQ2_THINKING_EFFORT must be low, high, or max"
[[ ${device_list} == CUDA0,CUDA1 || ${device_list} == CUDA1,CUDA0 ]] ||
  fail "KIMI_IQ2_DEVICE_LIST must contain exactly CUDA0 and CUDA1"

for required_command in awk find grep hostname install numactl stat timeout; do
  require_command "${required_command}"
done

awk -v value="${top_probability}" 'BEGIN { exit !(value >= 0 && value <= 1) }' ||
  fail "KIMI_IQ2_TOP_P must be between 0 and 1"
awk -v value="${minimum_probability}" 'BEGIN { exit !(value >= 0 && value <= 1) }' ||
  fail "KIMI_IQ2_MIN_P must be between 0 and 1"

if [[ ${allow_other_host} != 1 ]]; then
  current_host=$(hostname --short)
  [[ ${current_host} == dwagon ]] ||
    fail "this topology must run on dwagon, not ${current_host}; set KIMI_IQ2_ALLOW_OTHER_HOST=1 to override"
fi

[[ ${slot_save_path} == /* ]] ||
  fail "KIMI_IQ2_SLOT_SAVE_PATH must be an absolute path"
install --directory --mode=0750 -- "${slot_save_path}" ||
  fail "could not create slot-save directory: ${slot_save_path}"
[[ -d ${slot_save_path} && -w ${slot_save_path} ]] ||
  fail "slot-save path is not writable: ${slot_save_path}"

[[ -x ${server_binary} ]] ||
  fail "pinned llama-server is not executable: ${server_binary}"
version_output=$("${server_binary}" --version 2>&1) ||
  fail "failed to query llama-server version"
expected_short_commit=${expected_runtime_commit:0:8}
[[ ${version_output} == *"(${expected_short_commit})"* ]] ||
  fail "runtime is not pinned commit ${expected_short_commit}: ${version_output//$'\n'/; }"

help_output=$("${server_binary}" --help 2>&1) ||
  fail "failed to query llama-server arguments"
for required_option in \
  --device \
  --split-mode \
  --fit \
  --fit-target \
  --n-cpu-moe \
  --no-op-offload \
  --load-mode \
  --ctx-checkpoints \
  --cache-ram \
  --no-cache-prompt \
  --no-warmup \
  --slot-save-path \
  --reasoning-format; do
  grep --fixed-strings --quiet -- "${required_option}" <<<"${help_output}" ||
    fail "pinned runtime does not advertise required option ${required_option}"
done

if [[ ${skip_model_check} != 1 ]]; then
  model_directory=${model_path%/*}
  model_file_name=${model_path##*/}
  [[ ${model_file_name} =~ ^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]] ||
    fail "model path is not the first file of a split GGUF: ${model_path}"
  shard_prefix=${BASH_REMATCH[1]}
  first_shard_number=$((10#${BASH_REMATCH[2]}))
  expected_shard_count=$((10#${BASH_REMATCH[3]}))
  ((first_shard_number == 1)) ||
    fail "KIMI_IQ2_MODEL_PATH must point to shard 00001"
  ((expected_shard_count == 16)) ||
    fail "pinned IQ2 model must have 16 shards, got ${expected_shard_count}"

  remaining_control=$(
    find "${model_directory}" -maxdepth 1 -type f -name '*.aria2' -print -quit
  )
  [[ -z ${remaining_control} ]] ||
    fail "model download still has an aria2 resume control: ${remaining_control}"

  actual_model_bytes=0
  missing_shards=()
  for ((shard_number = 1; shard_number <= expected_shard_count; shard_number++)); do
    printf -v shard_number_text '%05d' "${shard_number}"
    shard_path="${model_directory}/${shard_prefix}-${shard_number_text}-of-${BASH_REMATCH[3]}.gguf"
    if [[ ! -s ${shard_path} ]]; then
      missing_shards+=("${shard_path}")
      continue
    fi
    shard_bytes=$(stat --format=%s "${shard_path}")
    shard_allocated_bytes=$(($(stat --format=%b "${shard_path}") * 512))
    ((shard_allocated_bytes >= shard_bytes)) ||
      fail "model shard is sparse/incomplete: ${shard_path} has ${shard_bytes} logical bytes but ${shard_allocated_bytes} allocated bytes"
    actual_model_bytes=$((actual_model_bytes + shard_bytes))
  done
  ((${#missing_shards[@]} == 0)) ||
    fail "model is incomplete; ${#missing_shards[@]} of ${expected_shard_count} shards are missing or empty"
  ((actual_model_bytes == expected_model_bytes)) ||
    fail "model byte count is ${actual_model_bytes}, expected ${expected_model_bytes}"
else
  actual_model_bytes=0
  log "model completeness check skipped by KIMI_IQ2_SKIP_MODEL_CHECK=1"
fi

if ((minimum_host_available_mib > 0)); then
  available_memory_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  [[ ${available_memory_kib} =~ ^[0-9]+$ ]] ||
    fail "could not read MemAvailable from /proc/meminfo"
  available_memory_mib=$((available_memory_kib / 1024))
  ((available_memory_mib >= minimum_host_available_mib)) ||
    fail "dwagon has ${available_memory_mib} MiB available; require at least ${minimum_host_available_mib} MiB"
else
  available_memory_mib=0
fi

if [[ ${skip_device_check} != 1 ]]; then
  if ! device_listing=$(
    timeout "${device_check_timeout_seconds}" "${server_binary}" --list-devices 2>&1
  ); then
    fail "local CUDA device discovery failed"
  fi
  for gpu_name in CUDA0 CUDA1; do
    gpu_line=$(grep --fixed-strings --max-count=1 -- "${gpu_name}:" <<<"${device_listing}")
    [[ -n ${gpu_line} ]] ||
      fail "${gpu_name} was not enumerated: ${device_listing//$'\n'/; }"
    if ((minimum_gpu_free_mib > 0)); then
      if [[ ${gpu_line} =~ ,\ ([0-9]+)\ MiB\ free\) ]]; then
        gpu_free_mib=${BASH_REMATCH[1]}
        ((gpu_free_mib >= minimum_gpu_free_mib)) ||
          fail "${gpu_name} has ${gpu_free_mib} MiB free; require at least ${minimum_gpu_free_mib} MiB"
      else
        fail "could not parse free memory for ${gpu_name}: ${gpu_line}"
      fi
    fi
  done
else
  device_listing=
  log "live device check skipped by KIMI_IQ2_SKIP_DEVICE_CHECK=1"
fi

server_arguments=(
  --model "${model_path}"
  --alias "${model_alias}"
  --device "${device_list}"
  --split-mode layer
  --fit on
  --fit-target "${fit_target_mib}"
  --load-mode none
  --no-op-offload
  --numa numactl
  --threads "${generation_threads}"
  --threads-batch "${batch_threads}"
  --ctx-size "${context_size}"
  --batch-size "${batch_size}"
  --ubatch-size "${physical_batch_size}"
  --cache-type-k f16
  --cache-type-v f16
  --ctx-checkpoints "${context_checkpoint_count}"
  --cache-ram 0
  --no-cache-idle-slots
  --no-cache-prompt
  --cache-reuse 0
  --slot-save-path "${slot_save_path}"
  --parallel "${parallel_slots}"
  --no-cont-batching
  --flash-attn on
  --no-warmup
  --jinja
  --no-mmproj
  --reasoning on
  --reasoning-format deepseek
  --chat-template-kwargs "{\"thinking_effort\":\"${thinking_effort}\"}"
  --seed "${random_seed}"
  --temperature "${temperature}"
  --top-k "${top_token_count}"
  --top-p "${top_probability}"
  --min-p "${minimum_probability}"
  --perf
  --prio 2
  --prio-batch 2
  --host "${listen_host}"
  --port "${listen_port}"
  --timeout "${request_timeout_seconds}"
  --sse-ping-interval "${sse_ping_interval_seconds}"
  --metrics
  --slots
  --no-webui
  --log-colors off
  --log-timestamps
)

if [[ -n ${log_file} ]]; then
  server_arguments+=(--log-file "${log_file}")
fi
server_arguments+=("${extra_server_arguments[@]}")

if ((skip_device_check != 1 && (validate_only == 1 || print_command == 1))); then
  if ! parser_validation_output=$(
    timeout \
      "${device_check_timeout_seconds}" \
      "${server_binary}" \
      "${server_arguments[@]}" \
      --help \
      2>&1
  ); then
    fail "llama-server rejected the assembled arguments without loading the model: ${parser_validation_output//$'\n'/; }"
  fi
fi

main_command=(
  /usr/bin/numactl
  "--physcpubind=${processor_bind}"
  "--interleave=${memory_nodes}"
  "${server_binary}"
  "${server_arguments[@]}"
)

log "runtime=${expected_short_commit} binary=${server_binary}"
log "model=${model_path} expected_bytes=${expected_model_bytes} verified_bytes=${actual_model_bytes}"
log "default_placement=MoE-aware dense-only auto-fit devices=${device_list} target_mib=${fit_target_mib}"
log "experts=CPU via fit planner; op_offload=off to avoid short-prompt PCIe expert streaming"
if ((${#extra_server_arguments[@]} > 0)); then
  printf -v appended_arguments_text '%q ' "${extra_server_arguments[@]}"
  log "appended_server_arguments=${appended_arguments_text% }"
fi
log "numa=physical_cpus:${processor_bind} memory_interleave:${memory_nodes} threads:${generation_threads}/${batch_threads}"
log "context=${context_size} batch=${batch_size} ubatch=${physical_batch_size} slots=${parallel_slots}"
log "slot_save_path=${slot_save_path}; required for explicit cache-cold slot erasure"
log "api=http://${listen_host}:${listen_port} alias=${model_alias} thinking_effort=${thinking_effort}"
log "warmup=built-in-disabled; send one sacrificial real prompt before collecting five measured runs"
if ((available_memory_mib > 0)); then
  log "dwagon_available_memory_mib=${available_memory_mib}"
fi
if [[ -n ${device_listing} ]]; then
  while IFS= read -r device_line; do
    [[ ${device_line} == "  "* ]] && log "device=${device_line#  }"
  done <<<"${device_listing}"
fi

if ((validate_only == 1)); then
  log "validation completed without loading the model"
  exit 0
fi
if ((print_command == 1)); then
  print_shell_command "${main_command[@]}"
  exit 0
fi

exec "${main_command[@]}"
