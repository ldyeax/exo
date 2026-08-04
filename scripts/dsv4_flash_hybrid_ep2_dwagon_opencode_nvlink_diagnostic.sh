#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"
default_python=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python

# This wrapper is also a narrow Python shim.  The OpenCode launcher inserts its
# own shim ahead of us; when SGLang is finally invoked, append only diagnostic
# communication flags and otherwise forward every Python command untouched.
if [[ ${DSV4_TP2_INTERCONNECT_PYTHON_SHIM:-0} == 1 ]]; then
  real_python="${DSV4_TP2_INTERCONNECT_REAL_PYTHON:?missing diagnostic real Python}"
  if [[ $# -ge 3 && $1 == -u && $2 == -m && $3 == sglang.launch_server ]]; then
    "$real_python" "${DSV4_TP2_INTERCONNECT_QUALIFIER_PATH:?missing qualifier path}" \
      --validate-receipt "${DSV4_TP2_INTERCONNECT_RECEIPT:?missing receipt}" \
      --maximum-receipt-age-seconds "${DSV4_TP2_INTERCONNECT_MAX_RECEIPT_AGE_SECONDS:?missing receipt age}" \
      --devices 0,1
    exec "$real_python" "$@" \
      --enable-p2p-check \
      --pre-warm-nccl \
      --disable-custom-all-reduce
  fi
  exec "$real_python" "$@"
fi

mode=prepare
if [[ ${1:-} == "--launch" ]]; then
  mode=launch
fi
if [[ $# -gt 1 || ($# -eq 1 && $mode != launch) ]]; then
  echo "usage: $0 [--launch]" >&2
  exit 2
fi

real_python="${DSV4_TP2_INTERCONNECT_REAL_PYTHON:-${DSV4_OPENCODE_REAL_PYTHON:-${DSV4_PYTHON:-$default_python}}}"
if [[ ! -x $real_python ]]; then
  echo "DeepSeek V4 diagnostic Python interpreter is not executable: $real_python" >&2
  exit 1
fi

# A topology diagnostic must not silently turn into an easier model launch.
# These are the already accepted OpenCode capacity, graph, and coherency
# contracts.  Explicit conflicting values fail before qualification or model
# preparation, while unset values are pinned below.
require_setting() {
  local name="$1"
  local expected="$2"
  local actual="${!name-}"
  if [[ -n $actual && $actual != "$expected" ]]; then
    echo "$name must remain $expected for the qualified OpenCode diagnostic (got $actual)" >&2
    exit 2
  fi
  printf -v "$name" '%s' "$expected"
  export "$name"
}

require_setting DSV4_CONTEXT_LENGTH 524288
require_setting DSV4_MAX_TOTAL_TOKENS 524288
require_setting DSV4_KV_CACHE_DTYPE fp8_e4m3
require_setting DSV4_SWA_FULL_TOKENS_RATIO 0.0048828125
require_setting DSV4_CHUNKED_PREFILL_SIZE 1024
require_setting DSV4_PREFILL_GRAPH_TIERS "256 512 1024"
require_setting DSV4_PREFILL_GRAPH_MAX 1024
require_setting DSV4_PREFILL_GRAPH_BACKEND breakable
require_setting DSV4_DECODE_GRAPH_BACKEND full
require_setting DSV4_CAPTURE_ATTN_IN_BCG 0
require_setting DSV4_EAGER_ATTN_MODULE_IN_BCG 1
require_setting DSV4_DSPARK_BLOCK_SIZE 5
require_setting DSV4_DISABLE_SPECULATIVE 0
require_setting DSV4_TENSOR_PARALLEL_SIZE 2
require_setting DSV4_EXPERT_PARALLEL_SIZE 2
require_setting DSV4_PIPELINE_PARALLEL_SIZE 1
require_setting DSV4_CUDA_VISIBLE_DEVICES 0,1
require_setting DSV4_MAX_RUNNING_REQUESTS 1
require_setting NCCL_P2P_LEVEL NVL

maximum_receipt_age_seconds="${DSV4_TP2_INTERCONNECT_MAX_RECEIPT_AGE_SECONDS:-300}"
if [[ ! $maximum_receipt_age_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "DSV4_TP2_INTERCONNECT_MAX_RECEIPT_AGE_SECONDS must be a positive integer" >&2
  exit 2
fi
export DSV4_TP2_INTERCONNECT_MAX_RECEIPT_AGE_SECONDS="$maximum_receipt_age_seconds"
export DSV4_TP2_INTERCONNECT_QUALIFIER_PATH="${script_dir}/qualify_dsv4_tp2_interconnect.py"

if [[ $mode == launch ]]; then
  qualifier="$DSV4_TP2_INTERCONNECT_QUALIFIER_PATH"
  if [[ ! -f $qualifier ]]; then
    echo "DSV4 TP2 interconnect qualifier does not exist: $qualifier" >&2
    exit 1
  fi
  receipt="${DSV4_TP2_INTERCONNECT_RECEIPT:-/tmp/dsv4-tp2-interconnect-${USER:-unknown}-$$.json}"
  if [[ -e $receipt || -e ${receipt}.artifacts ]]; then
    echo "refusing to overwrite DSV4 TP2 interconnect evidence: $receipt" >&2
    exit 1
  fi
  "$real_python" "$qualifier" \
    --output "$receipt" \
    --python "$real_python" \
    --devices 0,1
  if [[ ! -s $receipt ]]; then
    echo "DSV4 TP2 qualifier returned without a receipt: $receipt" >&2
    exit 1
  fi
  export DSV4_TP2_INTERCONNECT_RECEIPT="$receipt"
  echo "DSV4 TP2 diagnostic requires and accepted interconnect evidence: $receipt"
fi

# Keep the diagnostic local even if the parent shell contains cluster NCCL
# settings.  SHM remains enabled only so an unexpected fallback is visible;
# the qualification receipt has already failed closed unless both rank edges
# used P2P and moved all four NVLinks.
unset NCCL_NET NCCL_IB_HCA NCCL_NET_GDR_LEVEL
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET,ENV
runtime_log_tag="${DSV4_TP2_INTERCONNECT_RUNTIME_LOG_TAG:-dsv4-opencode-nccl-$$}"
export NCCL_DEBUG_FILE="${DSV4_TP2_INTERCONNECT_RUNTIME_NCCL_LOG:-/tmp/${runtime_log_tag}-%h-%p.log}"

# Install this wrapper underneath the existing OpenCode Python shim without
# changing the accepted launcher's model-serving flags or preparation steps.
unset DSV4_OPENCODE_PYTHON_SHIM DSV4_OPENCODE_REAL_PYTHON
export DSV4_TP2_INTERCONNECT_PYTHON_SHIM=1
export DSV4_TP2_INTERCONNECT_REAL_PYTHON="$real_python"
export DSV4_PYTHON="$script_path"

exec "${script_dir}/dsv4_flash_hybrid_ep2_dwagon_opencode.sh" "$@"
