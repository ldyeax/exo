#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=prepare
if [[ ${1:-} == "--launch" ]]; then
  mode=launch
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--launch]" >&2
  exit 2
fi

source_path="${DSV4_SGLANG_SOURCE:-/var/lib/exo/sources/sglang-dspark-30261}"
kt_source_path="${DSV4_KTRANSFORMERS_SOURCE:-${repo_root}/vendor/ktransformers}"
python_path="${DSV4_PYTHON:-/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python}"
ordering_path="${DSV4_EXPERT_ORDERING:-${repo_root}/scripts/data/dsv4_flash_0731_agentic_expert_order.json}"
sps_table_path="${DSV4_SPS_TABLE:-${repo_root}/scripts/data/dsv4_flash_tp2_finetiers_sps.json}"
patch_path="${repo_root}/scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch"
cache_root="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-0731-release}"
plan_path="${DSV4_GPU_EXPERT_PLAN:-${cache_root}/agentic-hot48-mask.pt}"

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

if ! git -C "$source_path" apply --reverse --check "$patch_path"; then
  echo "the audited DSV4 source patch is not installed in $source_path" >&2
  exit 1
fi

mkdir -p "$cache_root" "${cache_root}/triton" "${cache_root}/flashinfer" "${cache_root}/torchinductor"
"$python_path" "$repo_root/scripts/prepare_dsv4_flash_0731.py" \
  --model "$model_path" \
  --ordering "$ordering_path" \
  --gpu-experts-per-layer 48 \
  --write-plan "$plan_path"

if [[ $mode == prepare ]]; then
  echo "DSV4 Flash 0731 launch is prepared; no model process was started."
  echo "Run $0 --launch only after both GPUs are free."
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
if [[ ${#gpu_free_mib[@]} -lt 2 ]]; then
  echo "the TP2 launch requires two visible NVIDIA GPUs" >&2
  exit 1
fi
for gpu_index in 0 1; do
  free_mib="${gpu_free_mib[$gpu_index]//[[:space:]]/}"
  if [[ ! $free_mib =~ ^[0-9]+$ ]] || ((free_mib < minimum_gpu_free_mib)); then
    echo "GPU ${gpu_index} has ${free_mib:-unknown} MiB free; require ${minimum_gpu_free_mib} MiB" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=0,1
export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
export PYTHONPATH="${source_path}/python:${kt_source_path}"
export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_BACKEND=amx
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=8
export CPUINFER_PARALLEL=16
export OMP_NUM_THREADS=104
export KMP_AFFINITY=granularity=fine,static,1,0
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_DSV4_CAPTURE_ATTN_IN_BCG=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_RAGGED_VERIFY_MODE=compact
export SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS=1
export SGLANG_DSV4_TARGET_VERIFY_EAGER=0
export SGLANG_KT_DRAFT_GPU_EXPERTS=0
export SGLANG_KT_GPU_EXPERT_MASK_PLAN="$plan_path"
unset SGLANG_KT_EXPERT_PROFILE
export SGLANG_V4_USE_TRITON_KERNELS=1
export SGLANG_ENABLE_CUDA_GRAPH_DEDUP=0
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export TRITON_CACHE_DIR="${cache_root}/triton"
export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL

context_length="${DSV4_CONTEXT_LENGTH:-65536}"

exec numactl --cpunodebind=0,1 --membind=0,1 "$python_path" -u -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30010 \
  --model-path "$model_path" \
  --trust-remote-code \
  --kt-weight-path "$model_path" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 48 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-max-deferred-experts-per-token 2 \
  --tensor-parallel-size 2 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --speculative-dspark-align-verify-tokens-to-graph-tier \
  --speculative-dspark-sps-table-path "$sps_table_path" \
  --context-length "$context_length" \
  --max-total-tokens "$context_length" \
  --swa-full-tokens-ratio 0.15 \
  --attention-backend flashinfer \
  --moe-runner-backend triton \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-max-bs-prefill 2048 \
  --cuda-graph-bs-prefill 256 512 1024 2048 \
  --disable-radix-cache \
  --skip-server-warmup
