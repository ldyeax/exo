#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
export PYTHONPATH=/var/lib/exo/sources/sglang-dspark-30261/python:/root/exo/vendor/ktransformers

# Native checkpoint MXFP4 experts: force the AMX/AVX-512 implementation and
# retain AMX for sufficiently populated prefill tiles.
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
export SGLANG_DSV4_CAPTURE_ATTN_IN_BCG=1
# The DSpark draft model and greedy/Markov proposal sampler are captured. Each
# draft stage retains one complete native-MXFP4 MoE eager bridge.
# DeepGEMM's FP8 MMA is Hopper-only. On SM86, retain the checkpoint's packed
# FP8 cache and select the fused TileLang FP8-to-BF16 paged-MQA kernel. It
# executes the page products with Ampere BF16 tensor cores and reduces the 64
# head scores inside each CTA instead of materializing the reference BMM.
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
# The v2 top-k planner uses Hopper thread-block clusters.  The v1 transform
# remains a fused CUDA top-k/page-table kernel and supports SM86.
export SGLANG_OPT_USE_TOPK_V2=0
# The installed sgl-kernel sparse-prefill ABI predates the DSpark branch and
# its CUDA implementation is Hopper-oriented.  Route extend through the
# validated SM86 packed-cache Triton attention dispatcher.
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_RAGGED_VERIFY_MODE=compact
# Capture one through six target-verify token tiers for bs=1. Without these
# tiers, compact verification rounds every budget to six and runs padded
# candidates through all 43 native-MXFP4 MoE layers.
export SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS=1
# Keep target verification in the breakable CUDA graph. An eager-target
# experiment avoided its 43 host MoE boundaries, but live DSV4 attention
# metadata construction made it substantially slower and it failed the strict
# repeated-output coherency gate.
export SGLANG_DSV4_TARGET_VERIFY_EAGER=0
# Target and three-stage DSpark draft both use the capture-stable DSV4 metadata
# contract. Keep their SM86 attention/indexer work captured; native-MXFP4 MoE
# remains an eager graph boundary for CPU/GPU overlap.
export SGLANG_KT_EXPERT_PROFILE=/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v16-32k-decode-experts.pt
# Draft experts remain on AMX/AVX-512. Small-token GPU execution was measured
# slower on SM86 even with CPU/GPU overlap; the opt-in remains available for
# future profile-guided placement.
export SGLANG_KT_DRAFT_GPU_EXPERTS=0
# SM86 executes the retained packed E2M1/UE8M0 experts through the portable
# Triton gather/scatter kernel.  FlashInfer has no Ampere FP4 binary and the
# July Marlin MXFP4 kernel faults on its first real Ampere invocation.
export SGLANG_V4_USE_TRITON_KERNELS=1
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export TRITON_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/triton-dspark
export FLASHINFER_WORKSPACE_BASE=/var/lib/exo/cache/dsv4-native-mxfp4/flashinfer-dspark
export TORCHINDUCTOR_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/torch-dspark
export NCCL_P2P_LEVEL=NVL

model_path=/mnt/sanic/llm_models/DeepSeek-V4-Flash-DSpark
python_path=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
sps_table_path=/var/lib/exo/profiles/dsv4-native-mxfp4/flash-dspark-v3j-finetiers-sps.json

exec numactl --interleave=all "${python_path}" -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30010 \
  --model "${model_path}" \
  --kt-weight-path "${model_path}" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 48 \
  --kt-cpuinfer 104 \
  --kt-threadpool-count 2 \
  --tensor-parallel-size 2 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --speculative-dspark-sps-table-path "${sps_table_path}" \
  --context-length 65536 \
  --max-total-tokens 65536 \
  --swa-full-tokens-ratio 0.15 \
  --attention-backend flashinfer \
  --moe-runner-backend triton \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
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
