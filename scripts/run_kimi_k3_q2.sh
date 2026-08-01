#!/usr/bin/env bash
set -euo pipefail

# Reproducible Kimi K3 UD-Q2_K_XL launcher for:
#   dwagon: llama-server, two local RTX 3090s, interleaved host RAM
#   fwuff:  llama.cpp RPC endpoint over the 100 Gb/s EDR IPoIB link; the
#           target may use its CPU while K3 DSpark uses its otherwise-idle GPU
#
# This script deliberately does not start or stop the fwuff RPC process. Use
# --print-rpc-command to obtain the matching command, start it on fwuff, and
# then run this launcher on dwagon.

readonly default_runtime_root=/mnt/llm-models/exo/sources/llama.cpp-kimik3-stack-v7-kernels-20260730
readonly default_rpc_runtime_root=/mnt/sanic/exo/sources/llama.cpp-kimik3-stack-v7-kernels-20260730
# The dedicated EDR/NFS view sustains materially higher reads than dwagon's
# local model RAID on this host. The verified local stage remains available
# through KIMI_Q2_MODEL_PATH as a storage-independent fallback.
readonly default_model_path=/mnt/sanic-edr/exo/model-views/Kimi-K3-Q2-DSpark-cf6b8244/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf
readonly default_rpc_tensor_source_root=/mnt/sanic/exo/model-views/Kimi-K3-Q2-DSpark-cf6b8244
readonly default_runtime_commit=d29a524eeaf39155825d6f0ef373075fe585cb12
readonly default_model_bytes=861277858912
readonly prlimit_binary=/usr/bin/prlimit

topology_preset=${KIMI_Q2_TOPOLOGY_PRESET:-legacy}
case "${topology_preset}" in
legacy)
  preset_rpc_enabled=1
  preset_device_list=CUDA0,${KIMI_Q2_RPC_DEVICE_NAME:-RPC0},CUDA1
  preset_tensor_split=2,11,3
  preset_gpu_layer_count=16
  preset_minimum_host_available_mib=720000
  preset_minimum_gpu_free_mibs=23000,23000
  preset_required_cuda_architectures=86
  preset_require_cuda_architecture_evidence=0
  ;;
future-4gpu-rpc2)
  # Three RTX 3090s plus one RTX 5090 on dwagon, with two layers on fwuff.
  # Keeping the 5090 last leaves the output-side layers on the fastest GPU.
  preset_rpc_enabled=1
  preset_device_list=${KIMI_Q2_RPC_DEVICE_NAME:-RPC0},CUDA0,CUDA1,CUDA2,CUDA3
  preset_tensor_split=2,2,2,2,4
  preset_gpu_layer_count=12
  preset_minimum_host_available_mib=740000
  preset_minimum_gpu_free_mibs=23000,23000,23000,31500
  preset_required_cuda_architectures='86;120'
  preset_require_cuda_architecture_evidence=1
  ;;
future-4gpu-local)
  # Bold all-local placement for three RTX 3090s plus one RTX 5090.
  preset_rpc_enabled=0
  preset_device_list=CUDA0,CUDA1,CUDA2,CUDA3
  preset_tensor_split=2,2,2,4
  preset_gpu_layer_count=10
  # The raw host tensors leave little runtime margin even on an otherwise
  # idle 768-GiB host. Fail closed unless the operator has recovered nearly
  # all host RAM or deliberately overrides this gate.
  preset_minimum_host_available_mib=750000
  preset_minimum_gpu_free_mibs=23000,23000,23000,31500
  preset_required_cuda_architectures='86;120'
  preset_require_cuda_architecture_evidence=1
  ;;
future-5gpu-local-2080)
  # The modified 22 GiB RTX 2080 Ti is CUDA0; the RTX 5090 remains last.
  preset_rpc_enabled=0
  preset_device_list=CUDA0,CUDA1,CUDA2,CUDA3,CUDA4
  preset_tensor_split=2,2,2,2,4
  preset_gpu_layer_count=12
  preset_minimum_host_available_mib=740000
  preset_minimum_gpu_free_mibs=21000,23000,23000,23000,31500
  preset_required_cuda_architectures='75;86;120'
  preset_require_cuda_architecture_evidence=1
  ;;
*)
  printf '[kimi-q2] error: unknown KIMI_Q2_TOPOLOGY_PRESET: %s\n' "${topology_preset}" >&2
  exit 1
  ;;
esac

runtime_root=${KIMI_Q2_RUNTIME_ROOT:-"${default_runtime_root}"}
rpc_runtime_root=${KIMI_Q2_RPC_RUNTIME_ROOT:-"${default_rpc_runtime_root}"}
server_binary=${KIMI_Q2_SERVER_BINARY:-"${runtime_root}/build-kimik3-stack-v701/bin/llama-server"}
rpc_server_binary=${KIMI_Q2_RPC_SERVER_BINARY:-"${rpc_runtime_root}/build-kimik3-stack-v701/bin/ggml-rpc-server"}
model_path=${KIMI_Q2_MODEL_PATH:-"${default_model_path}"}
rpc_tensor_source_root=${KIMI_Q2_RPC_TENSOR_SOURCE_ROOT:-"${default_rpc_tensor_source_root}"}
rpc_tensor_source_mode=${KIMI_Q2_RPC_TENSOR_SOURCE_MODE:-same-backing-file}
rpc_tensor_source_max_files=${KIMI_Q2_RPC_TENSOR_SOURCE_MAX_FILES:-32}
expected_runtime_commit=${KIMI_Q2_EXPECTED_RUNTIME_COMMIT:-"${default_runtime_commit}"}
expected_model_bytes=${KIMI_Q2_EXPECTED_MODEL_BYTES:-"${default_model_bytes}"}

rpc_endpoint=${KIMI_Q2_RPC_ENDPOINT:-10.44.0.2:50052}
# This pinned runtime names remote devices RPC0/RPC1 globally and shows the
# endpoint as their description. RPC0[host:port] syntax does not apply here.
rpc_device_name=${KIMI_Q2_RPC_DEVICE_NAME:-RPC0}
rpc_draft_device_name=${KIMI_Q2_RPC_DRAFT_DEVICE_NAME:-}
rpc_enabled=${KIMI_Q2_RPC_ENABLED:-"${preset_rpc_enabled}"}
rpc_server_device=${KIMI_Q2_RPC_SERVER_DEVICE:-CPU}
IFS=, read -r -a rpc_server_devices <<<"${rpc_server_device}"
rpc_target_server_device=${rpc_server_devices[0]:-}
rpc_has_cuda_device=0
for _rpc_server_device in "${rpc_server_devices[@]}"; do
  if [[ ${_rpc_server_device} != CPU ]]; then
    rpc_has_cuda_device=1
  fi
done
rpc_disable_cuda_graphs=${KIMI_Q2_RPC_DISABLE_CUDA_GRAPHS:-}
if [[ -z ${rpc_disable_cuda_graphs} ]]; then
  if ((rpc_has_cuda_device == 0)); then
    rpc_disable_cuda_graphs=0
  else
    # This is required by the modified 22 GiB RTX 2080 Ti and is a safe
    # conservative default for any explicitly selected remote CUDA device.
    rpc_disable_cuda_graphs=1
  fi
fi
rpc_required_cuda_architectures=${KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES:-}
rpc_built_cuda_architectures=${KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES:-}
device_list=${KIMI_Q2_DEVICE_LIST:-"${preset_device_list}"}
# These are relative layer-count weights, not literal GiB limits. At sixteen
# offloaded layers, 2,11,3 maps blocks 78-79 to CUDA0, blocks 80-90 to fwuff's
# CPU, and blocks 91-92 plus output to CUDA1. This measured ordering keeps the
# fully touched output head local while reducing the serial RPC stage.
tensor_split=${KIMI_Q2_TENSOR_SPLIT:-"${preset_tensor_split}"}
gpu_layer_count=${KIMI_Q2_GPU_LAYERS:-"${preset_gpu_layer_count}"}
minimum_gpu_free_mibs_override=${KIMI_Q2_MINIMUM_GPU_FREE_MIBS:-}
minimum_gpu_free_mib_scalar=${KIMI_Q2_MINIMUM_GPU_FREE_MIB:-}
required_cuda_architectures=${KIMI_Q2_REQUIRED_CUDA_ARCHITECTURES:-"${preset_required_cuda_architectures}"}
built_cuda_architectures=${KIMI_Q2_BUILT_CUDA_ARCHITECTURES:-}
require_cuda_architecture_evidence=${KIMI_Q2_REQUIRE_CUDA_ARCHITECTURE_EVIDENCE:-"${preset_require_cuda_architecture_evidence}"}
no_host_model_weights=${KIMI_Q2_NO_HOST:-1}

processor_bind=${KIMI_Q2_PROCESSOR_BIND:-0-111}
memory_nodes=${KIMI_Q2_MEMORY_NODES:-all}
generation_threads=${KIMI_Q2_THREADS:-112}
batch_threads=${KIMI_Q2_BATCH_THREADS:-112}
minimum_host_available_mib=${KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB:-"${preset_minimum_host_available_mib}"}
if [[ ${rpc_target_server_device} == CPU ]]; then
  default_minimum_rpc_host_available_mib=200000
  default_minimum_rpc_device_free_mib=0
else
  default_minimum_rpc_host_available_mib=64000
  default_minimum_rpc_device_free_mib=21000
fi
minimum_rpc_host_available_mib=${KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB:-"${default_minimum_rpc_host_available_mib}"}
minimum_rpc_device_free_mib=${KIMI_Q2_MINIMUM_RPC_DEVICE_FREE_MIB:-"${default_minimum_rpc_device_free_mib}"}
if [[ -n ${rpc_draft_device_name} ]]; then
  # The CUDA RPC backend itself reserves about 5.97 GiB on fwuff's RTX 3090,
  # leaving 18,159 MiB device-reported free before loading the 4.78-GB draft.
  # This floor still leaves ample room for the draft weights, KV, and scratch.
  default_minimum_rpc_draft_device_free_mib=16000
else
  default_minimum_rpc_draft_device_free_mib=0
fi
minimum_rpc_draft_device_free_mib=${KIMI_Q2_MINIMUM_RPC_DRAFT_DEVICE_FREE_MIB:-"${default_minimum_rpc_draft_device_free_mib}"}
rpc_ssh_target=${KIMI_Q2_RPC_SSH_TARGET:-fwuff}
rpc_expected_hostname=${KIMI_Q2_RPC_EXPECTED_HOSTNAME:-fwuff}
rpc_headroom_timeout_seconds=${KIMI_Q2_RPC_HEADROOM_TIMEOUT_SECONDS:-15}

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
                       and configured live devices without loading the model.
  --print-command      Validate, then print the shell-escaped main command
                       instead of launching it.
  --print-rpc-command  Print the matching command to run on the RPC host.
  --help               Show this help.

Topology presets:
  legacy                    Current 2 x RTX 3090 + fwuff CPU placement (default)
  future-4gpu-rpc2          3 x RTX 3090 + RTX 5090; two layers on fwuff
  future-4gpu-local         Bold all-local 3 x RTX 3090 + RTX 5090 placement
  future-5gpu-local-2080    Local 22 GiB 2080 Ti + 3 x 3090 + 5090 placement

Frequently useful environment overrides:
  KIMI_Q2_TOPOLOGY_PRESET
  KIMI_Q2_MODEL_PATH
  KIMI_Q2_RUNTIME_ROOT
  KIMI_Q2_RPC_RUNTIME_ROOT
  KIMI_Q2_RPC_TENSOR_SOURCE_ROOT
  KIMI_Q2_RPC_TENSOR_SOURCE_MODE
  KIMI_Q2_RPC_TENSOR_SOURCE_MAX_FILES
  KIMI_Q2_RPC_ENABLED
  KIMI_Q2_RPC_ENDPOINT
  KIMI_Q2_RPC_DEVICE_NAME
  KIMI_Q2_RPC_DRAFT_DEVICE_NAME
  KIMI_Q2_RPC_SERVER_DEVICE
  KIMI_Q2_RPC_DISABLE_CUDA_GRAPHS
  KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES
  KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES
  KIMI_Q2_RPC_SSH_TARGET
  KIMI_Q2_RPC_EXPECTED_HOSTNAME
  KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB
  KIMI_Q2_MINIMUM_RPC_DEVICE_FREE_MIB
  KIMI_Q2_MINIMUM_RPC_DRAFT_DEVICE_FREE_MIB
  KIMI_Q2_DEVICE_LIST
  KIMI_Q2_TENSOR_SPLIT
  KIMI_Q2_GPU_LAYERS
  KIMI_Q2_MINIMUM_GPU_FREE_MIBS
  KIMI_Q2_REQUIRED_CUDA_ARCHITECTURES
  KIMI_Q2_BUILT_CUDA_ARCHITECTURES
  KIMI_Q2_REQUIRE_CUDA_ARCHITECTURE_EVIDENCE
  KIMI_Q2_NO_HOST
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
model download is incomplete. Arguments after `--` are appended verbatim, but
model-source, RPC, and device-placement options are frozen and cannot be
duplicated there.

KIMI_Q2_MINIMUM_GPU_FREE_MIBS is a CSV aligned with the non-RPC target devices.
KIMI_Q2_MINIMUM_GPU_FREE_MIB remains a backwards-compatible scalar applied to
all of them. KIMI_Q2_RPC_SERVER_DEVICE accepts either one device or the
target,draft pair CPU,CUDA0. For the pair, set
KIMI_Q2_RPC_DRAFT_DEVICE_NAME=RPC1; the launcher binds that name to
--spec-draft-device and requires 16,000 MiB free by default. Any RPC server
containing CUDA defaults to GGML_CUDA_DISABLE_GRAPHS=1, which the modified
22 GiB RTX 2080 Ti requires. It must also declare
KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES; the shared build cache or an explicit
KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES must prove those architectures. Future
Blackwell presets require a separate CUDA build containing sm_120; the 2080
preset also requires sm_75. Every active RPC client topology independently
queries KIMI_Q2_RPC_SSH_TARGET (fwuff by default) and requires effective remote
headroom of at least KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB. Effective headroom
is the smaller of /proc/meminfo MemAvailable and all finite cgroup-v2
memory.max-minus-memory.current limits in the SSH process's hierarchy.
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

rpc_memory_probe() {
  cat <<'REMOTE_PROBE'
set -eu

host_name=$(hostname --short)
mem_available_kib=$(awk '$1 == "MemAvailable:" { print $2; exit }' /proc/meminfo)
case "${mem_available_kib}" in
  ''|*[!0-9]*) exit 70 ;;
esac

cgroup_path=$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)
case "${cgroup_path}" in
  /*) ;;
  *) exit 71 ;;
esac
if [ "${cgroup_path}" = / ]; then
  cgroup_directory=/sys/fs/cgroup
else
  cgroup_directory=/sys/fs/cgroup${cgroup_path}
fi

cgroup_headroom_kib=-1
while :; do
  if [ ! -r "${cgroup_directory}/memory.max" ] ||
    [ ! -r "${cgroup_directory}/memory.current" ]; then
    # The cgroup-v2 mount root has no memory.max/current on some kernels;
    # system-wide MemAvailable is the effective root-level constraint.
    [ "${cgroup_directory}" = /sys/fs/cgroup ] && break
    exit 72
  fi
  memory_max=$(cat "${cgroup_directory}/memory.max")
  memory_current=$(cat "${cgroup_directory}/memory.current")
  case "${memory_current}" in
    ''|*[!0-9]*) exit 74 ;;
  esac
  case "${memory_max}" in
    max) ;;
    ''|*[!0-9]*) exit 75 ;;
    *)
      if [ "${memory_current}" -ge "${memory_max}" ]; then
        candidate_headroom_kib=0
      else
        candidate_headroom_kib=$(((memory_max - memory_current) / 1024))
      fi
      if [ "${cgroup_headroom_kib}" -lt 0 ] ||
        [ "${candidate_headroom_kib}" -lt "${cgroup_headroom_kib}" ]; then
        cgroup_headroom_kib=${candidate_headroom_kib}
      fi
      ;;
  esac

  [ "${cgroup_directory}" != /sys/fs/cgroup ] || break
  cgroup_directory=${cgroup_directory%/*}
  case "${cgroup_directory}" in
    /sys/fs/cgroup|/sys/fs/cgroup/*) ;;
    *) exit 76 ;;
  esac
done

effective_headroom_kib=${mem_available_kib}
if [ "${cgroup_headroom_kib}" -ge 0 ] &&
  [ "${cgroup_headroom_kib}" -lt "${effective_headroom_kib}" ]; then
  effective_headroom_kib=${cgroup_headroom_kib}
fi
printf '%s %s %s %s\n' \
  "${host_name}" \
  "${mem_available_kib}" \
  "${effective_headroom_kib}" \
  "${cgroup_headroom_kib}"
REMOTE_PROBE
}

query_rpc_host_headroom() {
  local probe_output
  if ! probe_output=$(
    timeout \
      "${rpc_headroom_timeout_seconds}" \
      ssh \
      -o BatchMode=yes \
      -o "ConnectTimeout=${rpc_headroom_timeout_seconds}" \
      -o ConnectionAttempts=1 \
      -- \
      "${rpc_ssh_target}" \
      "$(rpc_memory_probe)"
  ); then
    fail "could not query fail-closed RPC host headroom over SSH target ${rpc_ssh_target}"
  fi
  [[ ${probe_output} != *$'\n'* ]] ||
    fail "RPC host headroom probe returned unexpected multiline output"

  local reported_hostname
  local mem_available_kib
  local effective_headroom_kib
  local cgroup_headroom_kib
  local unexpected_output
  read -r \
    reported_hostname \
    mem_available_kib \
    effective_headroom_kib \
    cgroup_headroom_kib \
    unexpected_output <<<"${probe_output}"
  [[ -z ${unexpected_output} &&
    ${reported_hostname} == "${rpc_expected_hostname}" ]] ||
    fail "RPC SSH target ${rpc_ssh_target} reported unexpected host/output: ${probe_output}"
  [[ ${mem_available_kib} =~ ^[0-9]+$ &&
    ${effective_headroom_kib} =~ ^[0-9]+$ &&
    ${cgroup_headroom_kib} =~ ^(-1|[0-9]+)$ ]] ||
    fail "RPC host headroom probe returned invalid capacity values: ${probe_output}"
  ((effective_headroom_kib <= mem_available_kib)) ||
    fail "RPC host effective headroom exceeds MemAvailable: ${probe_output}"
  if ((cgroup_headroom_kib >= 0)); then
    ((effective_headroom_kib <= cgroup_headroom_kib)) ||
      fail "RPC host effective headroom exceeds cgroup headroom: ${probe_output}"
  fi

  rpc_host_mem_available_mib=$((mem_available_kib / 1024))
  rpc_host_effective_available_mib=$((effective_headroom_kib / 1024))
  if ((cgroup_headroom_kib >= 0)); then
    rpc_host_cgroup_headroom_mib=$((cgroup_headroom_kib / 1024))
  else
    rpc_host_cgroup_headroom_mib=-1
  fi
  ((rpc_host_effective_available_mib >= minimum_rpc_host_available_mib)) ||
    fail "${rpc_expected_hostname} effective available memory is ${rpc_host_effective_available_mib} MiB (MemAvailable ${rpc_host_mem_available_mib} MiB, cgroup headroom ${rpc_host_cgroup_headroom_mib} MiB); require at least ${minimum_rpc_host_available_mib} MiB"
}

device_line_for_name() {
  local requested_device_name=$1
  awk -v requested_device_name="${requested_device_name}" '
    {
      candidate = $0
      sub(/^[[:space:]]*/, "", candidate)
      separator = index(candidate, ":")
      if (separator > 0 &&
          substr(candidate, 1, separator - 1) == requested_device_name) {
        matches += 1
        matching_line = $0
      }
    }
    END {
      if (matches == 1) {
        print matching_line
        exit 0
      }
      exit 1
    }
  '
}

cuda_architectures_for_binary() {
  local binary_path=$1
  local binary_directory=${binary_path%/*}
  local build_directory=${binary_directory%/*}
  local cache_path=${build_directory}/CMakeCache.txt
  if [[ -r ${cache_path} ]]; then
    awk -F= '$1 ~ /^CMAKE_CUDA_ARCHITECTURES:/ { print $2; exit }' "${cache_path}"
  fi
}

validate_cuda_architecture_set() {
  local evidence_label=$1
  local built_architectures=$2
  local required_architectures=$3
  [[ ${built_architectures} =~ ^[0-9]+[a-z]?(-[a-z]+)?(\;[0-9]+[a-z]?(-[a-z]+)?)*$ ]] ||
    fail "${evidence_label} is not a concrete CMake CUDA architecture list: ${built_architectures}"
  local built_entries=()
  local required_entries=()
  IFS=';' read -r -a built_entries <<<"${built_architectures}"
  IFS=';' read -r -a required_entries <<<"${required_architectures}"
  local required_architecture
  local built_architecture
  local built_architecture_base
  local architecture_found
  for required_architecture in "${required_entries[@]}"; do
    architecture_found=0
    for built_architecture in "${built_entries[@]}"; do
      built_architecture_base=${built_architecture%%-*}
      built_architecture_base=${built_architecture_base%[a-z]}
      if [[ ${built_architecture_base} == "${required_architecture}" ]]; then
        architecture_found=1
        break
      fi
    done
    ((architecture_found == 1)) ||
      fail "${evidence_label} requires sm_${required_architecture}, but the build contains ${built_architectures}"
  done
}

validate_rpc_command_artifacts() {
  [[ -x ${prlimit_binary} ]] ||
    fail "core-dump limiter is not executable: ${prlimit_binary}"
  [[ -d ${rpc_runtime_root} ]] ||
    fail "RPC runtime root is not a directory visible from this host: ${rpc_runtime_root}"
  [[ -x ${rpc_server_binary} ]] ||
    fail "RPC server binary is not executable: ${rpc_server_binary}"

  require_command git
  local rpc_source_commit
  rpc_source_commit=$(
    git -C "${rpc_runtime_root}" rev-parse --verify 'HEAD^{commit}' 2>/dev/null
  ) ||
    fail "could not verify the RPC runtime source commit under ${rpc_runtime_root}"
  [[ ${rpc_source_commit} == "${expected_runtime_commit}" ]] ||
    fail "RPC runtime source is ${rpc_source_commit}, expected ${expected_runtime_commit}"

  local rpc_help_output
  rpc_help_output=$("${rpc_server_binary}" --help 2>&1) ||
    fail "failed to query RPC server arguments"
  local required_rpc_option
  for required_rpc_option in --tensor-source-root --tensor-source-max-files; do
    grep --fixed-strings --quiet -- "${required_rpc_option}" <<<"${rpc_help_output}" ||
      fail "pinned RPC runtime does not advertise required option ${required_rpc_option}"
  done

  if ((rpc_has_cuda_device == 1)); then
    require_command awk
    [[ -n ${rpc_required_cuda_architectures} ]] ||
      fail "RPC server CUDA device list ${rpc_server_device} requires KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES"
    [[ ${rpc_required_cuda_architectures} =~ ^[0-9]+(\;[0-9]+)*$ ]] ||
      fail "KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES must be a semicolon-delimited SM list"
    if [[ -z ${rpc_built_cuda_architectures} ]]; then
      rpc_built_cuda_architectures=$(cuda_architectures_for_binary "${rpc_server_binary}")
    fi
    [[ -n ${rpc_built_cuda_architectures} ]] ||
      fail "cannot verify CUDA architectures for non-CPU RPC binary ${rpc_server_binary}"
    validate_cuda_architecture_set \
      "RPC CUDA architecture evidence" \
      "${rpc_built_cuda_architectures}" \
      "${rpc_required_cuda_architectures}"
  fi
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

validate_extra_server_arguments() {
  local argument
  local option
  for argument in "${extra_server_arguments[@]}"; do
    option=${argument%%=*}
    option=${option//_/-}
    if [[ ${option} == --spec-draft-device &&
      -n ${rpc_draft_device_name} ]]; then
      fail "argument after -- duplicates KIMI_Q2_RPC_DRAFT_DEVICE_NAME"
    fi
    case "${option}" in
    -m | --model | -mu | --model-url | -hf | -hfr | --hf-repo | -hff | --hf-file | \
      --models-dir | --models-preset | --models-autoload | --no-models-autoload | \
      --rpc | --rpc-tensor-source | --rpc-tensor-source-mode | \
      -dev | --device | -ts | --tensor-split | -ngl | --gpu-layers | --n-gpu-layers | \
      -sm | --split-mode | -lm | --load-mode | --fit | --main-gpu)
      fail "argument after -- duplicates frozen model/RPC/topology option ${option}"
      ;;
    esac
  done
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
validate_extra_server_arguments

if [[ -v KIMI_Q2_MINIMUM_RPC_REPORTED_MIB ]]; then
  fail "KIMI_Q2_MINIMUM_RPC_REPORTED_MIB was removed because RPC physical capacity is not free memory; configure KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB instead"
fi

for integer_setting in \
  "KIMI_Q2_EXPECTED_MODEL_BYTES:${expected_model_bytes}" \
  "KIMI_Q2_GPU_LAYERS:${gpu_layer_count}" \
  "KIMI_Q2_THREADS:${generation_threads}" \
  "KIMI_Q2_BATCH_THREADS:${batch_threads}" \
  "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB:${minimum_host_available_mib}" \
  "KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB:${minimum_rpc_host_available_mib}" \
  "KIMI_Q2_MINIMUM_RPC_DEVICE_FREE_MIB:${minimum_rpc_device_free_mib}" \
  "KIMI_Q2_MINIMUM_RPC_DRAFT_DEVICE_FREE_MIB:${minimum_rpc_draft_device_free_mib}" \
  "KIMI_Q2_RPC_TENSOR_SOURCE_MAX_FILES:${rpc_tensor_source_max_files}" \
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
  "KIMI_Q2_RPC_HEADROOM_TIMEOUT_SECONDS:${rpc_headroom_timeout_seconds}" \
  "KIMI_Q2_DEVICE_CHECK_TIMEOUT_SECONDS:${device_check_timeout_seconds}"; do
  require_unsigned_integer "${integer_setting%%:*}" "${integer_setting#*:}"
done
((rpc_tensor_source_max_files > 0)) ||
  fail "KIMI_Q2_RPC_TENSOR_SOURCE_MAX_FILES must be positive"
[[ ${rpc_tensor_source_mode} =~ ^(same-backing-file|fallback)$ ]] ||
  fail "KIMI_Q2_RPC_TENSOR_SOURCE_MODE must be same-backing-file or fallback"

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
  "KIMI_Q2_RPC_ENABLED:${rpc_enabled}" \
  "KIMI_Q2_RPC_DISABLE_CUDA_GRAPHS:${rpc_disable_cuda_graphs}" \
  "KIMI_Q2_REQUIRE_CUDA_ARCHITECTURE_EVIDENCE:${require_cuda_architecture_evidence}" \
  "KIMI_Q2_NO_HOST:${no_host_model_weights}" \
  "KIMI_Q2_SKIP_MODEL_CHECK:${skip_model_check}" \
  "KIMI_Q2_SKIP_DEVICE_CHECK:${skip_device_check}" \
  "KIMI_Q2_ALLOW_OTHER_HOST:${allow_other_host}"; do
  [[ ${boolean_setting#*:} =~ ^[01]$ ]] ||
    fail "${boolean_setting%%:*} must be 0 or 1, got: ${boolean_setting#*:}"
done

rpc_host=
rpc_port=0
if [[ ${rpc_enabled} == 1 ]]; then
  [[ ${rpc_endpoint} =~ ^[^,:[:space:]]+:([0-9]+)$ ]] ||
    fail "KIMI_Q2_RPC_ENDPOINT must be HOST:PORT, got: ${rpc_endpoint}"
  rpc_host=${rpc_endpoint%:*}
  rpc_port=${rpc_endpoint##*:}
  require_unsigned_integer RPC_PORT "${rpc_port}"
  ((rpc_port > 0 && rpc_port <= 65535)) ||
    fail "RPC port must be between 1 and 65535"
  [[ ${rpc_ssh_target} =~ ^([a-zA-Z0-9._-]+@)?[a-zA-Z0-9._-]+$ &&
    ${rpc_ssh_target} != -* ]] ||
    fail "KIMI_Q2_RPC_SSH_TARGET must be a simple SSH host or user@host"
  [[ ${rpc_expected_hostname} =~ ^[a-zA-Z0-9._-]+$ &&
    ${rpc_expected_hostname} != -* ]] ||
    fail "KIMI_Q2_RPC_EXPECTED_HOSTNAME must be a simple hostname"
  ((minimum_rpc_host_available_mib > 0)) ||
    fail "active RPC topology requires positive KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB"
  ((rpc_headroom_timeout_seconds > 0)) ||
    fail "active RPC topology requires positive KIMI_Q2_RPC_HEADROOM_TIMEOUT_SECONDS"
  if [[ ${rpc_target_server_device} == CPU ]]; then
    ((minimum_rpc_device_free_mib == 0)) ||
      fail "CPU RPC capacity must use KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB, not RPC device-reported memory"
  fi
fi
((listen_port > 0 && listen_port <= 65535)) ||
  fail "listen port must be between 1 and 65535"
((physical_batch_size <= batch_size)) ||
  fail "physical batch size cannot exceed logical batch size"
((batch_size <= context_size)) ||
  fail "logical batch size cannot exceed context size"
[[ ${thinking_effort} =~ ^(low|high|max)$ ]] ||
  fail "KIMI_Q2_THINKING_EFFORT must be low, high, or max"
[[ ${tensor_split} =~ ^[0-9]+(,[0-9]+)*$ ]] ||
  fail "KIMI_Q2_TENSOR_SPLIT must be a CSV of integer proportions"
IFS=, read -r -a tensor_split_weights <<<"${tensor_split}"
tensor_split_sum=0
for tensor_split_weight in "${tensor_split_weights[@]}"; do
  ((10#${tensor_split_weight} > 0)) ||
    fail "every KIMI_Q2_TENSOR_SPLIT proportion must be positive"
  tensor_split_sum=$((tensor_split_sum + 10#${tensor_split_weight}))
done
[[ ${device_list} =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] ||
  fail "KIMI_Q2_DEVICE_LIST must be a CSV of nonempty device names without whitespace"
IFS=, read -r -a configured_devices <<<"${device_list}"
((${#configured_devices[@]} == ${#tensor_split_weights[@]})) ||
  fail "KIMI_Q2_DEVICE_LIST has ${#configured_devices[@]} devices but KIMI_Q2_TENSOR_SPLIT has ${#tensor_split_weights[@]} weights"

local_devices=()
rpc_device_occurrences=0
rpc_draft_device_occurrences=0
for ((device_index = 0; device_index < ${#configured_devices[@]}; device_index++)); do
  configured_device=${configured_devices[device_index]}
  for ((earlier_index = 0; earlier_index < device_index; earlier_index++)); do
    if [[ ${configured_device} == "${configured_devices[earlier_index]}" ]]; then
      fail "KIMI_Q2_DEVICE_LIST contains duplicate device ${configured_device}"
    fi
  done
  if [[ ${configured_device} == "${rpc_device_name}" ]]; then
    rpc_device_occurrences=$((rpc_device_occurrences + 1))
  elif [[ -n ${rpc_draft_device_name} &&
    ${configured_device} == "${rpc_draft_device_name}" ]]; then
    rpc_draft_device_occurrences=$((rpc_draft_device_occurrences + 1))
  else
    local_devices+=("${configured_device}")
  fi
done

if [[ ${rpc_enabled} == 1 ]]; then
  ((rpc_device_occurrences == 1)) ||
    fail "RPC topology must contain ${rpc_device_name} exactly once"
  [[ ${rpc_tensor_source_root} == /* ]] ||
    fail "KIMI_Q2_RPC_TENSOR_SOURCE_ROOT must be an absolute path on the RPC host"
  trusted_rpc_client_model_directory=${default_model_path%/*}
  configured_model_directory=${model_path%/*}
  [[ ${configured_model_directory} == "${trusted_rpc_client_model_directory}" ]] ||
    fail "file-aware RPC requires the canonical Sanic EDR model directory ${trusted_rpc_client_model_directory}; staged or alternate client copies require KIMI_Q2_RPC_ENABLED=0"
  [[ ${rpc_tensor_source_root} == "${default_rpc_tensor_source_root}" ]] ||
    fail "file-aware RPC requires fwuff's canonical tensor source root ${default_rpc_tensor_source_root}"
  ((rpc_draft_device_occurrences == 0)) ||
    fail "draft-only RPC device ${rpc_draft_device_name} cannot be part of the target KIMI_Q2_DEVICE_LIST"
else
  ((rpc_device_occurrences == 0)) ||
    fail "local-only topology cannot contain RPC device ${rpc_device_name}"
  [[ -z ${rpc_draft_device_name} ]] ||
    fail "KIMI_Q2_RPC_DRAFT_DEVICE_NAME requires KIMI_Q2_RPC_ENABLED=1"
fi

# The checked-in preset weights intentionally sum to --gpu-layers, which makes
# their expected layer boundaries readable. Explicit overrides remain ordinary
# llama.cpp relative weights and therefore need only align with the device CSV.
if [[ ${device_list} == "${preset_device_list}" &&
  ${tensor_split} == "${preset_tensor_split}" &&
  ${gpu_layer_count} == "${preset_gpu_layer_count}" ]]; then
  ((tensor_split_sum == gpu_layer_count)) ||
    fail "internal preset ${topology_preset} has tensor split sum ${tensor_split_sum}, expected ${gpu_layer_count}"
fi

minimum_gpu_free_mibs=()
if [[ -n ${minimum_gpu_free_mibs_override} ]]; then
  [[ ${minimum_gpu_free_mibs_override} =~ ^[0-9]+(,[0-9]+)*$ ]] ||
    fail "KIMI_Q2_MINIMUM_GPU_FREE_MIBS must be an integer CSV"
  IFS=, read -r -a minimum_gpu_free_mibs <<<"${minimum_gpu_free_mibs_override}"
elif [[ -n ${minimum_gpu_free_mib_scalar} ]]; then
  require_unsigned_integer KIMI_Q2_MINIMUM_GPU_FREE_MIB "${minimum_gpu_free_mib_scalar}"
  for _configured_device in "${local_devices[@]}"; do
    minimum_gpu_free_mibs+=("${minimum_gpu_free_mib_scalar}")
  done
else
  IFS=, read -r -a minimum_gpu_free_mibs <<<"${preset_minimum_gpu_free_mibs}"
fi
((${#minimum_gpu_free_mibs[@]} == ${#local_devices[@]})) ||
  fail "GPU-free-memory minima have ${#minimum_gpu_free_mibs[@]} entries but the topology has ${#local_devices[@]} local devices"
for minimum_gpu_free_mib in "${minimum_gpu_free_mibs[@]}"; do
  require_unsigned_integer KIMI_Q2_MINIMUM_GPU_FREE_MIBS "${minimum_gpu_free_mib}"
done

[[ ${rpc_server_device} =~ ^[^,[:space:]]+(,[^,[:space:]]+)?$ ]] ||
  fail "KIMI_Q2_RPC_SERVER_DEVICE must be one device or a target,draft device pair without whitespace"
for ((rpc_server_device_index = 0;  \
rpc_server_device_index < ${#rpc_server_devices[@]};  \
rpc_server_device_index++)); do
  _rpc_server_device=${rpc_server_devices[rpc_server_device_index]}
  [[ ${_rpc_server_device} == CPU ||
    ${_rpc_server_device} =~ ^CUDA[0-9]+$ ]] ||
    fail "unsupported RPC server device ${_rpc_server_device}; expected CPU or CUDA<N>"
  for ((earlier_index = 0;  \
  earlier_index < rpc_server_device_index;  \
  earlier_index++)); do
    [[ ${_rpc_server_device} != "${rpc_server_devices[earlier_index]}" ]] ||
      fail "KIMI_Q2_RPC_SERVER_DEVICE contains duplicate device ${_rpc_server_device}"
  done
done
[[ ${rpc_device_name} =~ ^[^,[:space:]]+$ ]] ||
  fail "KIMI_Q2_RPC_DEVICE_NAME must be one device name without whitespace"
if ((${#rpc_server_devices[@]} == 1)); then
  [[ -z ${rpc_draft_device_name} ]] ||
    fail "KIMI_Q2_RPC_DRAFT_DEVICE_NAME requires a target,draft KIMI_Q2_RPC_SERVER_DEVICE pair"
  ((minimum_rpc_draft_device_free_mib == 0)) ||
    fail "KIMI_Q2_MINIMUM_RPC_DRAFT_DEVICE_FREE_MIB requires KIMI_Q2_RPC_DRAFT_DEVICE_NAME"
else
  [[ -n ${rpc_draft_device_name} ]] ||
    fail "a target,draft KIMI_Q2_RPC_SERVER_DEVICE pair requires KIMI_Q2_RPC_DRAFT_DEVICE_NAME"
  [[ ${rpc_draft_device_name} =~ ^[^,[:space:]]+$ ]] ||
    fail "KIMI_Q2_RPC_DRAFT_DEVICE_NAME must be one device name without whitespace"
  [[ ${rpc_draft_device_name} != "${rpc_device_name}" ]] ||
    fail "target and draft RPC device names must be distinct"
  [[ ${rpc_device_name} =~ ^RPC([0-9]+)$ ]] ||
    fail "mixed RPC target device must have the enumerated form RPC<N>"
  rpc_target_device_index=$((10#${BASH_REMATCH[1]}))
  [[ ${rpc_draft_device_name} == "RPC$((rpc_target_device_index + 1))" ]] ||
    fail "mixed RPC draft device must immediately follow ${rpc_device_name} in RPC enumeration"
  [[ ${rpc_server_devices[1]} =~ ^CUDA[0-9]+$ ]] ||
    fail "the second RPC server device must be a CUDA GPU for K3 DSpark"
  ((minimum_rpc_draft_device_free_mib > 0)) ||
    fail "KIMI_Q2_MINIMUM_RPC_DRAFT_DEVICE_FREE_MIB must be positive for a remote draft GPU"
fi
[[ ${required_cuda_architectures} =~ ^[0-9]+(\;[0-9]+)*$ ]] ||
  fail "KIMI_Q2_REQUIRED_CUDA_ARCHITECTURES must be a semicolon-delimited SM list"
if [[ -n ${built_cuda_architectures} ]]; then
  [[ ${built_cuda_architectures} =~ ^[0-9]+[a-z]?(-[a-z]+)?(\;[0-9]+[a-z]?(-[a-z]+)?)*$ ]] ||
    fail "KIMI_Q2_BUILT_CUDA_ARCHITECTURES is not a valid CMake CUDA architecture list"
fi

if ((print_rpc_command == 1)); then
  [[ ${rpc_enabled} == 1 ]] ||
    fail "preset ${topology_preset} is local-only and has no RPC command"
  validate_rpc_command_artifacts
  rpc_command=()
  if [[ ${rpc_disable_cuda_graphs} == 1 ]]; then
    rpc_command+=(/usr/bin/env GGML_CUDA_DISABLE_GRAPHS=1)
  fi
  rpc_command=(
    "${rpc_command[@]}"
    "${prlimit_binary}"
    --core=0:0
    --
    /usr/bin/numactl
    "--physcpubind=${rpc_processor_bind}"
    "--membind=${rpc_memory_node}"
    "${rpc_server_binary}"
    --host "${rpc_host}"
    --port "${rpc_port}"
    --device "${rpc_server_device}"
    --threads "${rpc_threads}"
    --tensor-source-root "${rpc_tensor_source_root}"
    --tensor-source-max-files "${rpc_tensor_source_max_files}"
  )
  print_shell_command "${rpc_command[@]}"
  exit 0
fi

[[ -x ${prlimit_binary} ]] ||
  fail "core-dump limiter is not executable: ${prlimit_binary}"
required_commands=(awk cat find grep hostname install numactl stat timeout)
if [[ ${rpc_enabled} == 1 ]]; then
  required_commands+=(ssh)
fi
for required_command in "${required_commands[@]}"; do
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
[[ ${version_output} =~ \(${expected_short_commit}[0-9a-f]*\) ]] ||
  fail "runtime is not pinned commit ${expected_short_commit}: ${version_output//$'\n'/; }"

if [[ -z ${built_cuda_architectures} ]]; then
  built_cuda_architectures=$(cuda_architectures_for_binary "${server_binary}")
fi
if [[ -n ${built_cuda_architectures} ]]; then
  validate_cuda_architecture_set \
    "preset ${topology_preset}" \
    "${built_cuda_architectures}" \
    "${required_cuda_architectures}"
elif [[ ${require_cuda_architecture_evidence} == 1 ]]; then
  fail "preset ${topology_preset} requires fail-closed CUDA architecture evidence for ${required_cuda_architectures}; no CMakeCache value or KIMI_Q2_BUILT_CUDA_ARCHITECTURES was provided"
else
  log "warning=unable to verify CUDA build architectures; expected ${required_cuda_architectures}. Set KIMI_Q2_BUILT_CUDA_ARCHITECTURES after verifying the binary."
fi

help_output=$("${server_binary}" --help 2>&1) ||
  fail "failed to query llama-server arguments"
required_options=(
  --device
  --tensor-split
  --gpu-layers
  --split-mode
  --fit
  --ctx-checkpoints
  --cache-ram
  --no-cache-prompt
  --no-warmup
  --slot-save-path
  --reasoning-format
)
if [[ ${rpc_enabled} == 1 ]]; then
  required_options+=(--rpc --rpc-tensor-source --rpc-tensor-source-mode)
fi
if [[ -n ${rpc_draft_device_name} ]]; then
  required_options+=(--spec-draft-device)
fi
if [[ ${no_host_model_weights} == 1 ]]; then
  required_options+=(--no-host)
fi
for required_option in "${required_options[@]}"; do
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
    shard_allocated_bytes=$(($(stat --format=%b "${shard_path}") * 512))
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

rpc_host_mem_available_mib=0
rpc_host_effective_available_mib=0
rpc_host_cgroup_headroom_mib=-1
if [[ ${rpc_enabled} == 1 ]]; then
  query_rpc_host_headroom
fi

if [[ ${skip_device_check} != 1 ]]; then
  device_discovery_command=(
    timeout
    "${device_check_timeout_seconds}"
    "${server_binary}"
  )
  if [[ ${rpc_enabled} == 1 ]]; then
    device_discovery_command+=(--rpc "${rpc_endpoint}")
  fi
  device_discovery_command+=(--list-devices)
  if ! device_listing=$(
    "${device_discovery_command[@]}" 2>&1
  ); then
    if [[ ${rpc_enabled} == 1 ]]; then
      fail "device discovery failed; start the RPC endpoint with: $("$0" --print-rpc-command)"
    fi
    fail "local device discovery failed: ${device_listing//$'\n'/; }"
  fi

  for required_device in "${configured_devices[@]}"; do
    if ! required_device_line=$(
      device_line_for_name "${required_device}" <<<"${device_listing}"
    ); then
      fail "required device ${required_device} was not enumerated: ${device_listing//$'\n'/; }"
    fi
  done

  if [[ ${rpc_enabled} == 1 ]]; then
    if ! remote_device_line=$(
      device_line_for_name "${rpc_device_name}" <<<"${device_listing}"
    ); then
      fail "target RPC device ${rpc_device_name} was not enumerated: ${device_listing//$'\n'/; }"
    fi
    [[ ${remote_device_line} == *"${rpc_device_name}: ${rpc_endpoint} ("* ]] ||
      fail "${rpc_device_name} does not describe expected RPC endpoint ${rpc_endpoint}: ${remote_device_line}"
    if ((minimum_rpc_device_free_mib > 0)); then
      if [[ ${remote_device_line} =~ \([0-9]+\ MiB,\ ([0-9]+)\ MiB\ free\) ]]; then
        rpc_device_free_mib=${BASH_REMATCH[1]}
        ((rpc_device_free_mib >= minimum_rpc_device_free_mib)) ||
          fail "${rpc_device_name} has only ${rpc_device_free_mib} MiB device memory free; require at least ${minimum_rpc_device_free_mib} MiB for RPC server device ${rpc_target_server_device}"
      else
        fail "could not parse free device memory for ${rpc_device_name}: ${remote_device_line}"
      fi
    fi
    if [[ -n ${rpc_draft_device_name} ]]; then
      if ! draft_remote_device_line=$(
        device_line_for_name "${rpc_draft_device_name}" <<<"${device_listing}"
      ); then
        fail "draft RPC device ${rpc_draft_device_name} was not enumerated: ${device_listing//$'\n'/; }"
      fi
      [[ ${draft_remote_device_line} == *"${rpc_draft_device_name}: ${rpc_endpoint} ("* ]] ||
        fail "${rpc_draft_device_name} does not describe expected draft RPC endpoint ${rpc_endpoint}: ${draft_remote_device_line}"
      if [[ ${draft_remote_device_line} =~ \([0-9]+\ MiB,\ ([0-9]+)\ MiB\ free\) ]]; then
        rpc_draft_device_free_mib=${BASH_REMATCH[1]}
        ((rpc_draft_device_free_mib >= minimum_rpc_draft_device_free_mib)) ||
          fail "${rpc_draft_device_name} has only ${rpc_draft_device_free_mib} MiB device memory free; require at least ${minimum_rpc_draft_device_free_mib} MiB for K3 DSpark"
      else
        fail "could not parse free device memory for ${rpc_draft_device_name}: ${draft_remote_device_line}"
      fi
    fi
  fi

  for ((local_device_index = 0; local_device_index < ${#local_devices[@]}; local_device_index++)); do
    gpu_name=${local_devices[local_device_index]}
    minimum_gpu_free_mib=${minimum_gpu_free_mibs[local_device_index]}
    if ((minimum_gpu_free_mib > 0)); then
      gpu_line=$(device_line_for_name "${gpu_name}" <<<"${device_listing}")
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
  log "live device check skipped by KIMI_Q2_SKIP_DEVICE_CHECK=1"
fi

server_arguments=(
  --model "${model_path}"
)
if [[ ${rpc_enabled} == 1 ]]; then
  server_arguments+=(
    --rpc-tensor-source "${model_path%/*}"
    --rpc-tensor-source-mode "${rpc_tensor_source_mode}"
  )
fi
server_arguments+=(
  --alias "${model_alias}"
)
if [[ ${rpc_enabled} == 1 ]]; then
  server_arguments+=(--rpc "${rpc_endpoint}")
fi
if [[ -n ${rpc_draft_device_name} ]]; then
  server_arguments+=(--spec-draft-device "${rpc_draft_device_name}")
fi
server_arguments+=(
  --device "${device_list}"
  --split-mode layer
  --gpu-layers "${gpu_layer_count}"
  --tensor-split "${tensor_split}"
  --fit off
  --load-mode none
)
if [[ ${no_host_model_weights} == 1 ]]; then
  # CPU-resident model weights do not benefit from pinning when operation
  # offload is disabled. Avoid a serial ~700-GiB cudaMallocHost pass while
  # retaining the separate pinned activation and upload staging buffers.
  server_arguments+=(--no-host)
fi
server_arguments+=(
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
  "${prlimit_binary}"
  --core=0:0
  --
  /usr/bin/numactl
  "--physcpubind=${processor_bind}"
  "--interleave=${memory_nodes}"
  "${server_binary}"
  "${server_arguments[@]}"
)

log "runtime=${expected_short_commit} binary=${server_binary}"
log "model=${model_path} expected_bytes=${expected_model_bytes} verified_bytes=${actual_model_bytes}"
log "topology=${topology_preset} placement=layer gpu_layers=${gpu_layer_count} devices=${device_list} tensor_split=${tensor_split}"
log "cuda_architectures=required:${required_cuda_architectures} built:${built_cuda_architectures:-unverified}"
if [[ ${rpc_enabled} == 1 ]]; then
  log "rpc_tensor_source=${rpc_tensor_source_root}; mode=${rpc_tensor_source_mode}; RPC-host tensors are read server-local without a dwagon round trip"
  log "loader_schedule=d29a524e prioritizes RPC-bearing contexts but does not overlap mixed-context local and RPC reads"
  log "transport=${rpc_endpoint} over EDR IPoIB; rpc_server_device=${rpc_server_device}; target_rpc_device=${rpc_device_name}:${rpc_target_server_device}; draft_rpc_device=${rpc_draft_device_name:-none}; model staging source is selected independently"
else
  log "transport=local-only; no RPC endpoint or RPC tensor-source argument is enabled"
fi
log "capacity=tensor-split values are relative proportions; checked-in preset weights sum to gpu-layers; --fit is forced off"
if [[ ${no_host_model_weights} == 1 ]]; then
  log "host_weights=pageable; --no-host avoids whole-model CUDA pinning while activation staging remains pinned"
else
  log "host_weights=pinned; expect a large serial cudaMallocHost initialization pass"
fi
kernel_numa_balancing=unknown
if [[ -r /proc/sys/kernel/numa_balancing ]]; then
  IFS= read -r kernel_numa_balancing </proc/sys/kernel/numa_balancing
fi
log "numa=physical_cpus:${processor_bind} memory_interleave:${memory_nodes} threads:${generation_threads}/${batch_threads}"
log "kernel_numa_balancing=${kernel_numa_balancing}; launcher records but never mutates this host-global setting"
log "context=${context_size} batch=${batch_size} ubatch=${physical_batch_size} slots=${parallel_slots}"
log "slot_save_path=${slot_save_path}; required for explicit cache-cold slot erasure"
log "api=http://${listen_host}:${listen_port} alias=${model_alias} thinking_effort=${thinking_effort}"
log "warmup=built-in-disabled; send one sacrificial real prompt before collecting the representative measured run"
if ((available_memory_mib > 0)); then
  log "dwagon_available_memory_mib=${available_memory_mib}"
fi
if [[ ${rpc_enabled} == 1 ]]; then
  log "${rpc_expected_hostname}_memory_mib=effective:${rpc_host_effective_available_mib} mem_available:${rpc_host_mem_available_mib} cgroup_headroom:${rpc_host_cgroup_headroom_mib} required:${minimum_rpc_host_available_mib}"
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
