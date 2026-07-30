#!/usr/bin/env bash
set -euo pipefail

# Reproducible Kimi K3 UD-Q2_K_XL launcher for:
#   dwagon: llama-server, two local RTX 3090s, interleaved host RAM
#   fwuff:  CPU-only llama.cpp RPC endpoint over the 100 Gb/s EDR IPoIB link
#
# This script deliberately does not start or stop the fwuff RPC process. Use
# --print-rpc-command to obtain the matching command, start it on fwuff, and
# then run this launcher on dwagon.

readonly default_runtime_root=/var/lib/exo/sources/llama.cpp-kimik3-text-47c5bbdf
readonly default_model_path=/mnt/sanic-edr/llm_models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf
readonly default_runtime_commit=47c5bbdfd5ab5e847098f791a2cb9c0c90fb7dbd
readonly default_model_bytes=861277858912

runtime_root=${KIMI_Q2_RUNTIME_ROOT:-"${default_runtime_root}"}
server_binary=${KIMI_Q2_SERVER_BINARY:-"${runtime_root}/build-kimik3-text/bin/llama-server"}
rpc_server_binary=${KIMI_Q2_RPC_SERVER_BINARY:-"${runtime_root}/build-kimik3-text/bin/ggml-rpc-server"}
model_path=${KIMI_Q2_MODEL_PATH:-"${default_model_path}"}
expected_runtime_commit=${KIMI_Q2_EXPECTED_RUNTIME_COMMIT:-"${default_runtime_commit}"}
expected_model_bytes=${KIMI_Q2_EXPECTED_MODEL_BYTES:-"${default_model_bytes}"}

rpc_endpoint=${KIMI_Q2_RPC_ENDPOINT:-10.44.0.2:50052}
# Commit 47c5bbdf names remote devices RPC0/RPC1 globally and shows the
# endpoint as their description. Newer documentation's RPC0[host:port] syntax
# does not apply to this pinned runtime.
rpc_device_name=${KIMI_Q2_RPC_DEVICE_NAME:-RPC0}
device_list=${KIMI_Q2_DEVICE_LIST:-"CUDA0,${rpc_device_name},CUDA1"}
# These are relative layer-count weights, not literal GiB limits. At sixteen
# offloaded layers, 2,11,3 maps blocks 78-79 to CUDA0, blocks 80-90 to fwuff's
# CPU, and blocks 91-92 plus output to CUDA1. This measured ordering keeps the
# fully touched output head local while reducing the serial RPC stage.
tensor_split=${KIMI_Q2_TENSOR_SPLIT:-2,11,3}
gpu_layer_count=${KIMI_Q2_GPU_LAYERS:-16}

processor_bind=${KIMI_Q2_PROCESSOR_BIND:-0-111}
memory_nodes=${KIMI_Q2_MEMORY_NODES:-all}
generation_threads=${KIMI_Q2_THREADS:-112}
batch_threads=${KIMI_Q2_BATCH_THREADS:-112}
minimum_host_available_mib=${KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB:-720000}
minimum_gpu_free_mib=${KIMI_Q2_MINIMUM_GPU_FREE_MIB:-23000}
minimum_rpc_reported_mib=${KIMI_Q2_MINIMUM_RPC_REPORTED_MIB:-200000}

context_size=${KIMI_Q2_CONTEXT_SIZE:-32768}
batch_size=${KIMI_Q2_BATCH_SIZE:-4096}
physical_batch_size=${KIMI_Q2_UBATCH_SIZE:-512}
context_checkpoint_count=${KIMI_Q2_CONTEXT_CHECKPOINTS:-1}
parallel_slots=${KIMI_Q2_PARALLEL_SLOTS:-1}

listen_host=${KIMI_Q2_LISTEN_HOST:-127.0.0.1}
listen_port=${KIMI_Q2_LISTEN_PORT:-11434}
model_alias=${KIMI_Q2_MODEL_ALIAS:-Kimi-K3-UD-Q2_K_XL}
thinking_effort=${KIMI_Q2_THINKING_EFFORT:-high}
random_seed=${KIMI_Q2_SEED:-3407}
temperature=${KIMI_Q2_TEMPERATURE:-1.0}
top_probability=${KIMI_Q2_TOP_P:-0.95}
top_token_count=${KIMI_Q2_TOP_K:-50}
minimum_probability=${KIMI_Q2_MIN_P:-0.0}
request_timeout_seconds=${KIMI_Q2_TIMEOUT_SECONDS:-21600}
sse_ping_interval_seconds=${KIMI_Q2_SSE_PING_INTERVAL_SECONDS:-15}
log_file=${KIMI_Q2_LOG_FILE:-}
slot_save_path=${KIMI_Q2_SLOT_SAVE_PATH:-/tmp/exo-kimi-k3-slot-state-q2}

rpc_threads=${KIMI_Q2_RPC_THREADS:-60}
rpc_processor_bind=${KIMI_Q2_RPC_PROCESSOR_BIND:-0-59}
rpc_memory_node=${KIMI_Q2_RPC_MEMORY_NODE:-0}
device_check_timeout_seconds=${KIMI_Q2_DEVICE_CHECK_TIMEOUT_SECONDS:-15}

skip_model_check=${KIMI_Q2_SKIP_MODEL_CHECK:-0}
skip_device_check=${KIMI_Q2_SKIP_DEVICE_CHECK:-0}
allow_other_host=${KIMI_Q2_ALLOW_OTHER_HOST:-0}

validate_only=0
print_command=0
print_rpc_command=0
extra_server_arguments=()

print_usage() {
  cat <<'USAGE'
Usage:
  scripts/run_kimi_k3_q2.sh [--validate-only] [--print-command] [--] [LLAMA_ARGUMENT ...]
  scripts/run_kimi_k3_q2.sh --print-rpc-command

Modes:
  --validate-only      Check the pinned binary, complete model, host capacity,
                       and live local/remote devices without loading the model.
  --print-command      Validate, then print the shell-escaped main command
                       instead of launching it.
  --print-rpc-command  Print the matching CPU-only command to run on fwuff.
  --help               Show this help.

Frequently useful environment overrides:
  KIMI_Q2_MODEL_PATH
  KIMI_Q2_RPC_ENDPOINT
  KIMI_Q2_DEVICE_LIST
  KIMI_Q2_TENSOR_SPLIT
  KIMI_Q2_GPU_LAYERS
  KIMI_Q2_CONTEXT_SIZE
  KIMI_Q2_BATCH_SIZE
  KIMI_Q2_UBATCH_SIZE
  KIMI_Q2_THREADS
  KIMI_Q2_BATCH_THREADS
  KIMI_Q2_LISTEN_HOST
  KIMI_Q2_LISTEN_PORT
  KIMI_Q2_THINKING_EFFORT
  KIMI_Q2_LOG_FILE
  KIMI_Q2_SLOT_SAVE_PATH

Set KIMI_Q2_SKIP_DEVICE_CHECK=1 only when printing a command before the fwuff
RPC endpoint is running. Set KIMI_Q2_SKIP_MODEL_CHECK=1 only while the pinned
model download is incomplete. Arguments after `--` are appended verbatim and
can intentionally override the frozen defaults.
USAGE
}

log() {
  printf '[kimi-q2] %s\n' "$*" >&2
}

fail() {
  printf '[kimi-q2] error: %s\n' "$*" >&2
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
    --print-rpc-command)
      print_rpc_command=1
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

[[ ${rpc_endpoint} =~ ^[^,:[:space:]]+:([0-9]+)$ ]] ||
  fail "KIMI_Q2_RPC_ENDPOINT must be HOST:PORT, got: ${rpc_endpoint}"
rpc_host=${rpc_endpoint%:*}
rpc_port=${rpc_endpoint##*:}

for integer_setting in \
  "KIMI_Q2_EXPECTED_MODEL_BYTES:${expected_model_bytes}" \
  "KIMI_Q2_GPU_LAYERS:${gpu_layer_count}" \
  "KIMI_Q2_THREADS:${generation_threads}" \
  "KIMI_Q2_BATCH_THREADS:${batch_threads}" \
  "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB:${minimum_host_available_mib}" \
  "KIMI_Q2_MINIMUM_GPU_FREE_MIB:${minimum_gpu_free_mib}" \
  "KIMI_Q2_MINIMUM_RPC_REPORTED_MIB:${minimum_rpc_reported_mib}" \
  "KIMI_Q2_CONTEXT_SIZE:${context_size}" \
  "KIMI_Q2_BATCH_SIZE:${batch_size}" \
  "KIMI_Q2_UBATCH_SIZE:${physical_batch_size}" \
  "KIMI_Q2_CONTEXT_CHECKPOINTS:${context_checkpoint_count}" \
  "KIMI_Q2_PARALLEL_SLOTS:${parallel_slots}" \
  "KIMI_Q2_LISTEN_PORT:${listen_port}" \
  "KIMI_Q2_SEED:${random_seed}" \
  "KIMI_Q2_TOP_K:${top_token_count}" \
  "KIMI_Q2_TIMEOUT_SECONDS:${request_timeout_seconds}" \
  "KIMI_Q2_SSE_PING_INTERVAL_SECONDS:${sse_ping_interval_seconds}" \
  "KIMI_Q2_RPC_THREADS:${rpc_threads}" \
  "KIMI_Q2_DEVICE_CHECK_TIMEOUT_SECONDS:${device_check_timeout_seconds}" \
  "RPC_PORT:${rpc_port}"; do
  require_unsigned_integer "${integer_setting%%:*}" "${integer_setting#*:}"
done

[[ ${expected_runtime_commit} =~ ^[0-9a-f]{40}$ ]] ||
  fail "KIMI_Q2_EXPECTED_RUNTIME_COMMIT must be a full lowercase Git commit"
for floating_setting in \
  "KIMI_Q2_TEMPERATURE:${temperature}" \
  "KIMI_Q2_TOP_P:${top_probability}" \
  "KIMI_Q2_MIN_P:${minimum_probability}"; do
  [[ ${floating_setting#*:} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    fail "${floating_setting%%:*} must be a nonnegative decimal, got: ${floating_setting#*:}"
done
for boolean_setting in \
  "KIMI_Q2_SKIP_MODEL_CHECK:${skip_model_check}" \
  "KIMI_Q2_SKIP_DEVICE_CHECK:${skip_device_check}" \
  "KIMI_Q2_ALLOW_OTHER_HOST:${allow_other_host}"; do
  [[ ${boolean_setting#*:} =~ ^[01]$ ]] ||
    fail "${boolean_setting%%:*} must be 0 or 1, got: ${boolean_setting#*:}"
done

((rpc_port > 0 && rpc_port <= 65535)) ||
  fail "RPC port must be between 1 and 65535"
((listen_port > 0 && listen_port <= 65535)) ||
  fail "listen port must be between 1 and 65535"
((physical_batch_size <= batch_size)) ||
  fail "physical batch size cannot exceed logical batch size"
((batch_size <= context_size)) ||
  fail "logical batch size cannot exceed context size"
[[ ${thinking_effort} =~ ^(low|high|max)$ ]] ||
  fail "KIMI_Q2_THINKING_EFFORT must be low, high, or max"
[[ ${tensor_split} =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] ||
  fail "KIMI_Q2_TENSOR_SPLIT must contain three integer proportions"
IFS=, read -r -a tensor_split_weights <<<"${tensor_split}"
for tensor_split_weight in "${tensor_split_weights[@]}"; do
  ((10#${tensor_split_weight} > 0)) ||
    fail "every KIMI_Q2_TENSOR_SPLIT proportion must be positive"
done
IFS=, read -r -a configured_devices <<<"${device_list}"
((${#configured_devices[@]} == 3)) ||
  fail "KIMI_Q2_DEVICE_LIST must contain exactly three devices"
for required_device in CUDA0 CUDA1 "${rpc_device_name}"; do
  device_was_configured=0
  for configured_device in "${configured_devices[@]}"; do
    if [[ ${configured_device} == "${required_device}" ]]; then
      device_was_configured=1
      break
    fi
  done
  ((device_was_configured == 1)) ||
    fail "device list must contain ${required_device}"
done

if ((print_rpc_command == 1)); then
  rpc_command=(
    /usr/bin/numactl
    "--physcpubind=${rpc_processor_bind}"
    "--membind=${rpc_memory_node}"
    "${rpc_server_binary}"
    --host "${rpc_host}"
    --port "${rpc_port}"
    --device CPU
    --threads "${rpc_threads}"
  )
  print_shell_command "${rpc_command[@]}"
  exit 0
fi

for required_command in awk find grep hostname install numactl stat timeout; do
  require_command "${required_command}"
done

awk -v value="${top_probability}" 'BEGIN { exit !(value >= 0 && value <= 1) }' ||
  fail "KIMI_Q2_TOP_P must be between 0 and 1"
awk -v value="${minimum_probability}" 'BEGIN { exit !(value >= 0 && value <= 1) }' ||
  fail "KIMI_Q2_MIN_P must be between 0 and 1"

if [[ ${allow_other_host} != 1 ]]; then
  current_host=$(hostname --short)
  [[ ${current_host} == dwagon ]] ||
    fail "this topology must run on dwagon, not ${current_host}; set KIMI_Q2_ALLOW_OTHER_HOST=1 to override"
fi

[[ ${slot_save_path} == /* ]] ||
  fail "KIMI_Q2_SLOT_SAVE_PATH must be an absolute path"
install --directory --mode=0750 -- "${slot_save_path}" ||
  fail "could not create slot-save directory: ${slot_save_path}"
[[ -d ${slot_save_path} && -w ${slot_save_path} ]] ||
  fail "slot-save path is not a writable directory: ${slot_save_path}"

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
  --rpc \
  --device \
  --tensor-split \
  --gpu-layers \
  --split-mode \
  --fit \
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
    fail "KIMI_Q2_MODEL_PATH must point to shard 00001"

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
    shard_allocated_bytes=$(( $(stat --format=%b "${shard_path}") * 512 ))
    ((shard_allocated_bytes >= shard_bytes)) ||
      fail "model shard is sparse/incomplete: ${shard_path} has ${shard_bytes} logical bytes but ${shard_allocated_bytes} allocated bytes"
    actual_model_bytes=$((actual_model_bytes + shard_bytes))
  done

  ((${#missing_shards[@]} == 0)) ||
    fail "model is incomplete; ${#missing_shards[@]} of ${expected_shard_count} shards are missing or empty"
  if ((expected_model_bytes > 0 && actual_model_bytes != expected_model_bytes)); then
    fail "model byte count is ${actual_model_bytes}, expected ${expected_model_bytes}; transfer may still be running"
  fi
else
  actual_model_bytes=0
  log "model completeness check skipped by KIMI_Q2_SKIP_MODEL_CHECK=1"
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
    timeout \
      "${device_check_timeout_seconds}" \
      "${server_binary}" \
      --rpc "${rpc_endpoint}" \
      --list-devices \
      2>&1
  ); then
    fail "device discovery failed; start fwuff RPC with: $("$0" --print-rpc-command)"
  fi

  for required_device in CUDA0 CUDA1 "${rpc_device_name}"; do
    grep --fixed-strings --quiet -- "${required_device}:" <<<"${device_listing}" ||
      fail "required device ${required_device} was not enumerated: ${device_listing//$'\n'/; }"
  done

  remote_device_line=$(grep --fixed-strings --max-count=1 -- "${rpc_device_name}:" <<<"${device_listing}")
  [[ ${remote_device_line} == *"${rpc_endpoint}"* ]] ||
    fail "${rpc_device_name} does not describe the expected endpoint ${rpc_endpoint}: ${remote_device_line}"
  if ((minimum_rpc_reported_mib > 0)); then
    if [[ ${remote_device_line} =~ \(([0-9]+)\ MiB, ]]; then
      rpc_reported_mib=${BASH_REMATCH[1]}
      ((rpc_reported_mib >= minimum_rpc_reported_mib)) ||
        fail "${rpc_device_name} reports only ${rpc_reported_mib} MiB; it is probably not fwuff's CPU device"
    else
      fail "could not parse reported capacity for ${rpc_device_name}: ${remote_device_line}"
    fi
  fi

  if ((minimum_gpu_free_mib > 0)); then
    for gpu_name in CUDA0 CUDA1; do
      gpu_line=$(grep --fixed-strings --max-count=1 -- "${gpu_name}:" <<<"${device_listing}")
      if [[ ${gpu_line} =~ ,\ ([0-9]+)\ MiB\ free\) ]]; then
        gpu_free_mib=${BASH_REMATCH[1]}
        ((gpu_free_mib >= minimum_gpu_free_mib)) ||
          fail "${gpu_name} has ${gpu_free_mib} MiB free; require at least ${minimum_gpu_free_mib} MiB"
      else
        fail "could not parse free memory for ${gpu_name}: ${gpu_line}"
      fi
    done
  fi
else
  device_listing=
  log "live device check skipped by KIMI_Q2_SKIP_DEVICE_CHECK=1"
fi

server_arguments=(
  --model "${model_path}"
  --alias "${model_alias}"
  --rpc "${rpc_endpoint}"
  --device "${device_list}"
  --split-mode layer
  --gpu-layers "${gpu_layer_count}"
  --tensor-split "${tensor_split}"
  --fit off
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
  # `--help` is deliberately last. The pinned parser validates every preceding
  # argument (including resolved RPC device names), then exits before loading
  # or mapping the model.
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
log "placement=layer gpu_layers=${gpu_layer_count} devices=${device_list} tensor_split=${tensor_split}"
log "capacity=tensor-split values are proportions; RPC CPU reports total RAM as free, so --fit is forced off"
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
