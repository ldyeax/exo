#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/exo/vendor/ktransformers/third_party/sglang/python:/root/exo/vendor/ktransformers
export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=8
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export SGLANG_PP_LAYER_PARTITION=21,23,17
export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export TRITON_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/triton-pro-pp3-rank0
export FLASHINFER_WORKSPACE_BASE=/var/lib/exo/cache/dsv4-native-mxfp4/flashinfer-pro-pp3-rank0
export TORCHINDUCTOR_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/torch-pro-pp3-rank0
export NCCL_SOCKET_IFNAME=ibs5
export GLOO_SOCKET_IFNAME=ibs5
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

exec /usr/bin/numactl --cpunodebind=0 --membind=0 \
  /var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30000 \
  --model /mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark \
  --kt-weight-path /mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 0 \
  --kt-expert-placement-strategy frequency \
  --kt-cpuinfer 52 \
  --kt-threadpool-count 1 \
  --kt-numa-nodes 0 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 3 \
  --nnodes 3 \
  --node-rank 0 \
  --dist-init-addr 10.44.0.1:29520 \
  --context-length 1048576 \
  --max-total-tokens 1048576 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.99 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --max-running-requests 1 \
  --watchdog-timeout 2400 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --skip-server-warmup
