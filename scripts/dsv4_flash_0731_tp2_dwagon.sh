#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launch_entrypoint="${DSV4_LAUNCH_ENTRYPOINT:-$0}"
mode=prepare
if [[ ${1:-} == "--launch" ]]; then
  mode=launch
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--launch]" >&2
  exit 2
fi

source_path="${DSV4_SGLANG_SOURCE:-${repo_root}/vendor/sglang}"
kt_source_path="${DSV4_KTRANSFORMERS_SOURCE:-${repo_root}/vendor/ktransformers}"
python_path="${DSV4_PYTHON:-/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python}"
ordering_path="${DSV4_EXPERT_ORDERING:-${repo_root}/scripts/data/dsv4_flash_0731_agentic_expert_order.json}"
sps_table_path="${DSV4_SPS_TABLE:-${repo_root}/scripts/data/dsv4_flash_tp2_finetiers_sps.json}"
patch_path="${repo_root}/scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch"
cache_root="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-0731-release}"
context_length="${DSV4_CONTEXT_LENGTH:-65536}"
tensor_parallel_size="${DSV4_TENSOR_PARALLEL_SIZE:-2}"
cuda_visible_devices="${DSV4_CUDA_VISIBLE_DEVICES:-}"
gpu_experts_per_layer="${DSV4_GPU_EXPERTS_PER_LAYER:-48}"
plan_path="${DSV4_GPU_EXPERT_PLAN:-${cache_root}/agentic-hot${gpu_experts_per_layer}-mask.pt}"
max_deferred_experts_per_token="${DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN:-2}"
dspark_block_size="${DSV4_DSPARK_BLOCK_SIZE:-5}"
chunked_prefill_size="${DSV4_CHUNKED_PREFILL_SIZE:-2048}"
max_total_tokens="${DSV4_MAX_TOTAL_TOKENS:-$((context_length + chunked_prefill_size))}"
cpuinfer_parallel="${DSV4_CPUINFER_PARALLEL:-16}"
cpuinfer_threads="${DSV4_CPUINFER_THREADS:-104}"
threadpool_count="${DSV4_KT_THREADPOOL_COUNT:-${tensor_parallel_size}}"
kt_numa_nodes_text="${DSV4_KT_NUMA_NODES:-}"
numactl_nodes="${DSV4_NUMACTL_NODES:-}"
mem_fraction_static="${DSV4_MEM_FRACTION_STATIC:-0.92}"
kv_cache_dtype="${DSV4_KV_CACHE_DTYPE:-fp8_e4m3}"
if [[ $kv_cache_dtype == bf16 ]]; then
  kv_cache_dtype=bfloat16
fi
decode_graph_backend="${DSV4_DECODE_GRAPH_BACKEND:-breakable}"
prefill_graph_backend="${DSV4_PREFILL_GRAPH_BACKEND:-breakable}"
ragged_verify_mode="${DSV4_RAGGED_VERIFY_MODE:-compact}"
fine_ragged_verify_tiers="${DSV4_FINE_RAGGED_VERIFY_TIERS:-1}"
expert_location_mode="${DSV4_EXPERT_LOCATION_MODE:-mask}"
disable_radix_cache="${DSV4_DISABLE_RADIX_CACHE:-1}"
swa_full_tokens_ratio="${DSV4_SWA_FULL_TOKENS_RATIO:-0.15}"
fwuff_parity_mode="${DSV4_FWUFF_PARITY_MODE:-0}"
disable_speculative="${DSV4_DISABLE_SPECULATIVE:-0}"
capture_attn_in_bcg="${DSV4_CAPTURE_ATTN_IN_BCG:-}"
eager_attn_module_in_bcg="${DSV4_EAGER_ATTN_MODULE_IN_BCG:-}"
reuse_main_q_shared_mlp="${DSV4_REUSE_MAIN_Q_FOR_SHARED_MLP:-}"
flashmla_sparse_prefill="${DSV4_FLASHMLA_SPARSE_PREFILL:-}"
fp8_paged_mqa_logits_torch="${DSV4_FP8_PAGED_MQA_LOGITS_TORCH:-}"
# CuTe DSL 4.6 aborts while compiling the SM86 RMSNorm path under the pinned
# local Python 3.12 runtime. FlashInfer's CUDA JIT implementation is the
# supported Ampere backend and is also safe for TP graph capture.
flashinfer_use_cuda_norm="${DSV4_FLASHINFER_USE_CUDA_NORM:-1}"
read -r -a prefill_graph_tiers <<<"${DSV4_PREFILL_GRAPH_TIERS:-256 512 1024 2048}"
prefill_graph_max="${DSV4_PREFILL_GRAPH_MAX:-${prefill_graph_tiers[-1]}}"

case "$tensor_parallel_size" in
1)
  cuda_visible_devices="${cuda_visible_devices:-0}"
  kt_numa_nodes_text="${kt_numa_nodes_text:-0}"
  numactl_nodes="${numactl_nodes:-0}"
  ;;
2)
  cuda_visible_devices="${cuda_visible_devices:-0,1}"
  kt_numa_nodes_text="${kt_numa_nodes_text:-0 1}"
  numactl_nodes="${numactl_nodes:-0,1}"
  ;;
*)
  echo "DSV4_TENSOR_PARALLEL_SIZE must be 1 or 2 on dwagon" >&2
  exit 2
  ;;
esac
read -r -a kt_numa_nodes <<<"$kt_numa_nodes_text"
case "$expert_location_mode" in
mask | init) ;;
*)
  echo "DSV4_EXPERT_LOCATION_MODE must be mask or init" >&2
  exit 2
  ;;
esac
for boolean_value in \
  "$fine_ragged_verify_tiers" \
  "$disable_radix_cache" \
  "$fwuff_parity_mode" \
  "$disable_speculative"; do
  if [[ $boolean_value != 0 && $boolean_value != 1 ]]; then
    echo "DSV4 boolean settings must be 0 or 1" >&2
    exit 2
  fi
done

if [[ -n ${DSV4_MODEL_PATH:-} ]]; then
  model_path="$DSV4_MODEL_PATH"
elif [[ -f /mnt/sanic-edr/llm_models/DeepSeek-V4-Flash-0731/config.json ]]; then
  model_path=/mnt/sanic-edr/llm_models/DeepSeek-V4-Flash-0731
else
  model_path=/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731
fi

for required_path in \
  "$source_path" \
  "$kt_source_path" \
  "$python_path" \
  "$ordering_path" \
  "$sps_table_path" \
  "$patch_path"; do
  if [[ ! -e $required_path ]]; then
    echo "missing required DSV4 launch input: $required_path" >&2
    exit 1
  fi
done

if git -C "$source_path" apply --reverse --check "$patch_path" 2>/dev/null; then
  : # Legacy DSpark base with the accumulated audit patch applied.
elif [[ -f "$source_path/python/sglang/srt/layers/quantization/mxfp4_deepseek.py" &&
  -f "$source_path/python/sglang/kernels/ops/attention/dsv4/bf16_decode.py" ]] &&
  grep -q "class DeepSeekMxfp4MoEMethod" \
    "$source_path/python/sglang/srt/layers/quantization/mxfp4_deepseek.py"; then
  : # Newer fwuff release base carries the audited implementation in-tree.
else
  echo "the audited DSV4 source implementation is not installed in $source_path" >&2
  exit 1
fi

mkdir -p \
  "$cache_root" \
  "${cache_root}/xdg" \
  "${cache_root}/triton" \
  "${cache_root}/flashinfer" \
  "${cache_root}/torchinductor"
"$python_path" "$repo_root/scripts/prepare_dsv4_flash_0731.py" \
  --model "$model_path" \
  --ordering "$ordering_path" \
  --gpu-experts-per-layer "$gpu_experts_per_layer" \
  --write-plan "$plan_path"

if [[ $mode == prepare ]]; then
  echo "DSV4 Flash 0731 launch is prepared; no model process was started."
  echo "Run $launch_entrypoint --launch only after the requested GPUs are free."
  exit 0
fi

if command -v ss >/dev/null 2>&1 && ss -H -ltn 'sport = :30010' | grep -q .; then
  echo "TCP port 30010 is already in use" >&2
  exit 1
fi

mapfile -t gpu_free_mib < <(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
)
minimum_gpu_free_mib="${DSV4_MINIMUM_GPU_FREE_MIB:-22000}"
IFS=',' read -r -a visible_gpu_ids <<<"$cuda_visible_devices"
if [[ ${#visible_gpu_ids[@]} -lt $tensor_parallel_size ]]; then
  echo "CUDA device list has fewer entries than the requested TP size" >&2
  exit 1
fi
for ((local_rank = 0; local_rank < tensor_parallel_size; local_rank++)); do
  gpu_index="${visible_gpu_ids[$local_rank]//[[:space:]]/}"
  if [[ ! $gpu_index =~ ^[0-9]+$ ]] || ((gpu_index >= ${#gpu_free_mib[@]})); then
    echo "invalid physical GPU index in DSV4_CUDA_VISIBLE_DEVICES: $gpu_index" >&2
    exit 1
  fi
  free_mib="${gpu_free_mib[$gpu_index]//[[:space:]]/}"
  if [[ ! $free_mib =~ ^[0-9]+$ ]] || ((free_mib < minimum_gpu_free_mib)); then
    echo "GPU ${gpu_index} has ${free_mib:-unknown} MiB free; require ${minimum_gpu_free_mib} MiB" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
export PYTHONPATH="${source_path}/python:${kt_source_path}"
export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_BACKEND=amx
export CPUINFER_PARALLEL="$cpuinfer_parallel"
export OMP_NUM_THREADS="$cpuinfer_threads"
export KMP_AFFINITY=granularity=fine,static,1,0
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_RAGGED_VERIFY_MODE="$ragged_verify_mode"
export SGLANG_ENABLE_CUDA_GRAPH_DEDUP=0

if [[ $fwuff_parity_mode == 1 ]]; then
  unset KT_MXFP4_AMX_MIN_EXPERT_TOKENS
  unset SGLANG_OPT_USE_TILELANG_MHC_PRE SGLANG_OPT_USE_TILELANG_MHC_POST
  unset SGLANG_DSV4_CAPTURE_ATTN_IN_BCG SGLANG_FP8_PAGED_MQA_LOGITS_TORCH
  unset SGLANG_OPT_USE_TOPK_V2 SGLANG_OPT_FLASHMLA_SPARSE_PREFILL
  unset SGLANG_DSV4_TARGET_VERIFY_EAGER SGLANG_KT_DRAFT_GPU_EXPERTS
  unset SGLANG_V4_USE_TRITON_KERNELS
else
  export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=8
  export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
  export SGLANG_OPT_USE_TILELANG_MHC_POST=1
  export SGLANG_DSV4_CAPTURE_ATTN_IN_BCG=1
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  export SGLANG_OPT_USE_TOPK_V2=0
  export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
  export SGLANG_DSV4_TARGET_VERIFY_EAGER=0
  export SGLANG_KT_DRAFT_GPU_EXPERTS=0
  export SGLANG_V4_USE_TRITON_KERNELS=1
fi

# The evolved DSV4 model can capture the full attention module in BCG.  An
# explicit launch override is applied after the parity/default environment so
# wrappers can reproduce the older fwuff branch's effective graph boundary
# (captured projections with only its attention kernel outside the graph).
if [[ -n $capture_attn_in_bcg ]]; then
  if [[ $capture_attn_in_bcg != 0 && $capture_attn_in_bcg != 1 ]]; then
    echo "DSV4_CAPTURE_ATTN_IN_BCG must be 0 or 1" >&2
    exit 2
  fi
  if [[ $capture_attn_in_bcg == 1 ]]; then
    export SGLANG_DSV4_CAPTURE_ATTN_IN_BCG=1
  else
    unset SGLANG_DSV4_CAPTURE_ATTN_IN_BCG
  fi
fi

if [[ -n $eager_attn_module_in_bcg ]]; then
  if [[ $eager_attn_module_in_bcg != 0 && $eager_attn_module_in_bcg != 1 ]]; then
    echo "DSV4_EAGER_ATTN_MODULE_IN_BCG must be 0 or 1" >&2
    exit 2
  fi
  if [[ $eager_attn_module_in_bcg == 1 ]]; then
    export SGLANG_DSV4_EAGER_ATTN_MODULE_IN_BCG=1
  else
    unset SGLANG_DSV4_EAGER_ATTN_MODULE_IN_BCG
  fi
fi

if [[ -n $reuse_main_q_shared_mlp ]]; then
  if [[ $reuse_main_q_shared_mlp != 0 && $reuse_main_q_shared_mlp != 1 ]]; then
    echo "DSV4_REUSE_MAIN_Q_FOR_SHARED_MLP must be 0 or 1" >&2
    exit 2
  fi
  export SGLANG_DSV4_REUSE_MAIN_Q_FOR_SHARED_MLP="$reuse_main_q_shared_mlp"
fi

if [[ -n $flashmla_sparse_prefill ]]; then
  if [[ $flashmla_sparse_prefill != 0 && $flashmla_sparse_prefill != 1 ]]; then
    echo "DSV4_FLASHMLA_SPARSE_PREFILL must be 0 or 1" >&2
    exit 2
  fi
  export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL="$flashmla_sparse_prefill"
fi

if [[ -n $fp8_paged_mqa_logits_torch ]]; then
  if [[ $fp8_paged_mqa_logits_torch != 0 && $fp8_paged_mqa_logits_torch != 1 ]]; then
    echo "DSV4_FP8_PAGED_MQA_LOGITS_TORCH must be 0 or 1" >&2
    exit 2
  fi
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="$fp8_paged_mqa_logits_torch"
fi

if [[ -n $flashinfer_use_cuda_norm ]]; then
  if [[ $flashinfer_use_cuda_norm != 0 && $flashinfer_use_cuda_norm != 1 ]]; then
    echo "DSV4_FLASHINFER_USE_CUDA_NORM must be 0 or 1" >&2
    exit 2
  fi
  export FLASHINFER_USE_CUDA_NORM="$flashinfer_use_cuda_norm"
fi

if [[ $fine_ragged_verify_tiers == 1 ]]; then
  export SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS=1
else
  unset SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS
fi

expert_location_args=()
unset SGLANG_KT_EXPERT_PROFILE
if [[ $expert_location_mode == mask ]]; then
  export SGLANG_KT_GPU_EXPERT_MASK_PLAN="$plan_path"
else
  unset SGLANG_KT_GPU_EXPERT_MASK_PLAN
  expert_location_args=(--init-expert-location "$ordering_path")
fi

radix_cache_args=()
if [[ $disable_radix_cache == 1 ]]; then
  radix_cache_args=(--disable-radix-cache)
fi
speculative_args=()
if [[ $disable_speculative == 0 ]]; then
  speculative_args=(
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$dspark_block_size"
    --speculative-dspark-align-verify-tokens-to-graph-tier
    --speculative-dspark-sps-table-path "$sps_table_path"
  )
fi
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export TILELANG_LIBCUDART_PATH="${DSV4_TILELANG_LIBCUDART_PATH:-/opt/cuda/lib64/libcudart.so.13}"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TRITON_CACHE_DIR="${cache_root}/triton"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL

exec numactl --cpunodebind="$numactl_nodes" --membind="$numactl_nodes" "$python_path" -u -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30010 \
  --model-path "$model_path" \
  --trust-remote-code \
  --kt-weight-path "$model_path" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts "$gpu_experts_per_layer" \
  --kt-cpuinfer "$cpuinfer_threads" \
  --kt-threadpool-count "$threadpool_count" \
  --kt-numa-nodes "${kt_numa_nodes[@]}" \
  --kt-max-deferred-experts-per-token "$max_deferred_experts_per_token" \
  --tensor-parallel-size "$tensor_parallel_size" \
  "${expert_location_args[@]}" \
  "${speculative_args[@]}" \
  --context-length "$context_length" \
  --max-total-tokens "$max_total_tokens" \
  --swa-full-tokens-ratio "$swa_full_tokens_ratio" \
  --kv-cache-dtype "$kv_cache_dtype" \
  --attention-backend flashinfer \
  --moe-runner-backend triton \
  --mem-fraction-static "$mem_fraction_static" \
  --chunked-prefill-size "$chunked_prefill_size" \
  --max-prefill-tokens "$chunked_prefill_size" \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --cuda-graph-backend-decode "$decode_graph_backend" \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill "$prefill_graph_backend" \
  --cuda-graph-max-bs-prefill "$prefill_graph_max" \
  --cuda-graph-bs-prefill "${prefill_graph_tiers[@]}" \
  "${radix_cache_args[@]}" \
  --skip-server-warmup
