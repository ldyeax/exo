#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DSV4_LAUNCH_ENTRYPOINT="${DSV4_LAUNCH_ENTRYPOINT:-${repo_root}/scripts/dsv4_flash_fwuff_parity.sh}"

export DSV4_TENSOR_PARALLEL_SIZE="${DSV4_TENSOR_PARALLEL_SIZE:-1}"
export DSV4_CUDA_VISIBLE_DEVICES="${DSV4_CUDA_VISIBLE_DEVICES:-0}"
# The recorded fwuff benchmark is 2,694 prompt tokens plus 512 generated
# tokens.  Keep enough headroom for that authoritative workload without
# charging every launch for an unrelated 128K capacity target.  Capacity can
# still be raised explicitly with these two environment variables.
export DSV4_CONTEXT_LENGTH="${DSV4_CONTEXT_LENGTH:-8192}"
export DSV4_MAX_TOTAL_TOKENS="${DSV4_MAX_TOTAL_TOKENS:-8192}"
export DSV4_GPU_EXPERTS_PER_LAYER="${DSV4_GPU_EXPERTS_PER_LAYER:-12}"
export DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN="${DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN:-2}"
export DSV4_DSPARK_BLOCK_SIZE="${DSV4_DSPARK_BLOCK_SIZE:-5}"
export DSV4_CHUNKED_PREFILL_SIZE="${DSV4_CHUNKED_PREFILL_SIZE:-2048}"
export DSV4_CPUINFER_PARALLEL="${DSV4_CPUINFER_PARALLEL:-16}"
export DSV4_CPUINFER_THREADS="${DSV4_CPUINFER_THREADS:-60}"
export DSV4_KT_THREADPOOL_COUNT="${DSV4_KT_THREADPOOL_COUNT:-1}"
export DSV4_KT_NUMA_NODES="${DSV4_KT_NUMA_NODES:-0}"
export DSV4_NUMACTL_NODES="${DSV4_NUMACTL_NODES:-0}"
export DSV4_MEM_FRACTION_STATIC="${DSV4_MEM_FRACTION_STATIC:-0.90}"
export DSV4_KV_CACHE_DTYPE="${DSV4_KV_CACHE_DTYPE:-bfloat16}"
export DSV4_DECODE_GRAPH_BACKEND="${DSV4_DECODE_GRAPH_BACKEND:-full}"
export DSV4_PREFILL_GRAPH_BACKEND="${DSV4_PREFILL_GRAPH_BACKEND:-breakable}"
# Preserve the exact deployed graph shape. The authoritative request uses the
# 2,048 tier followed by a sub-1K tail, while the smaller tiers remain useful
# for warm-up and shorter prompts.
export DSV4_PREFILL_GRAPH_TIERS="256 512 1024 2048"
export DSV4_PREFILL_GRAPH_MAX=2048
export DSV4_RAGGED_VERIFY_MODE="${DSV4_RAGGED_VERIFY_MODE:-static}"
export DSV4_FINE_RAGGED_VERIFY_TIERS="${DSV4_FINE_RAGGED_VERIFY_TIERS:-0}"
export DSV4_EXPERT_LOCATION_MODE="${DSV4_EXPERT_LOCATION_MODE:-init}"
export DSV4_DISABLE_RADIX_CACHE="${DSV4_DISABLE_RADIX_CACHE:-0}"
# 0.10 rounds down to three 256-token SWA pages at the default 8K capacity.
# The authoritative 2,694/512 request can require a fourth page while a
# chunk is retired, so retain the TP2 launcher's 0.15 headroom.
export DSV4_SWA_FULL_TOKENS_RATIO="${DSV4_SWA_FULL_TOKENS_RATIO:-0.15}"
export DSV4_FWUFF_PARITY_MODE=1
# fwuff captures attention projections and breaks around only the backend
# attention call.  Select the equivalent boundary in the evolved local source.
export DSV4_CAPTURE_ATTN_IN_BCG="${DSV4_CAPTURE_ATTN_IN_BCG:-0}"
export DSV4_EAGER_ATTN_MODULE_IN_BCG="${DSV4_EAGER_ATTN_MODULE_IN_BCG:-0}"
export DSV4_REUSE_MAIN_Q_FOR_SHARED_MLP=0
export DSV4_FLASHMLA_SPARSE_PREFILL=0
export DSV4_FP8_PAGED_MQA_LOGITS_TORCH=1
# CuTe DSL 4.6 aborts while compiling RMSNorm for sm86 under the local Python
# 3.12 runtime.  FlashInfer's CUDA JIT norm is its supported Ampere fallback;
# it preserves the operation while avoiding a non-serving startup.
export DSV4_FLASHINFER_USE_CUDA_NORM=1
export DSV4_SPS_TABLE="${repo_root}/scripts/data/dsv4_flash_fwuff_sm86_sps.json"
export DSV4_CACHE_ROOT="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-fwuff-parity}"

exec "${repo_root}/scripts/dsv4_flash_0731_tp2_dwagon.sh" "$@"
