#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! $1 =~ ^[01]$ ]]; then
  echo "usage: $0 <pipeline-rank: 0|1>" >&2
  exit 2
fi

pipeline_rank=$1
model_path=/mnt/sanic/llm_models/DeepSeek-V4-Flash-DSpark

export CUDA_VISIBLE_DEVICES="${pipeline_rank}"
export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
export PYTHONPATH=/var/lib/exo/sources/sglang-dspark-30261/python:/root/exo/vendor/ktransformers
export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_BACKEND=amx
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=8
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_V4_USE_TRITON_KERNELS=1
export SGLANG_RAGGED_VERIFY_MODE=static
# Draft capture is not coherent yet: its replay freezes proposal-side dynamic
# state and collapses real acceptance.  Keep the target verifier graphed while
# the three-layer DSpark draft runs eagerly.
export SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1
export SGLANG_PP_LAYER_PARTITION=22,21
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export TRITON_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/triton-flash-dspark-pp2-smoke-rank${pipeline_rank}
export FLASHINFER_WORKSPACE_BASE=/var/lib/exo/cache/dsv4-native-mxfp4/flashinfer-flash-dspark-pp2-smoke-rank${pipeline_rank}
export TORCHINDUCTOR_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/torch-flash-dspark-pp2-smoke-rank${pipeline_rank}
export NCCL_SOCKET_IFNAME=ibs5
export GLOO_SOCKET_IFNAME=ibs5
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_DISABLE=0

python_path=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
numa_node=${pipeline_rank}
listen_port=$((30020 + pipeline_rank))

exec /usr/bin/numactl --cpunodebind="${numa_node}" --localalloc \
  "${python_path}" -m sglang.launch_server \
  --host 127.0.0.1 \
  --port "${listen_port}" \
  --model "${model_path}" \
  --kt-weight-path "${model_path}" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-cpuinfer 52 \
  --kt-threadpool-count 1 \
  --kt-numa-nodes "${numa_node}" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank "${pipeline_rank}" \
  --dist-init-addr 10.44.0.1:29541 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --context-length 8192 \
  --max-total-tokens 8192 \
  --swa-full-tokens-ratio 0.75 \
  --attention-backend flashinfer \
  --moe-runner-backend triton \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 1024 \
  --max-prefill-tokens 1024 \
  --max-running-requests 1 \
  --watchdog-timeout 1200 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  --skip-server-warmup
