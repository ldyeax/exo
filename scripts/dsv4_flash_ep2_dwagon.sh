#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${DSV4_PYTHON:-/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python}"
profile_path="${DSV4_CPU_EXPERT_PROFILE:-/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v16-32k-decode-experts.pt}"
cache_root="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-ep2}"
shard_plan="${DSV4_CPU_EXPERT_SHARD_PLAN:-${cache_root}/balanced-cpu-ep2.pt}"

mkdir -p "$(dirname "$shard_plan")"
"$python_path" "$repo_root/scripts/build_dsv4_kt_expert_shard_plan.py" \
  --profile "$profile_path" \
  --output "$shard_plan" \
  --rank-counts 128,128 \
  --balance-all-ranks

export DSV4_LAUNCH_ENTRYPOINT="${repo_root}/scripts/dsv4_flash_ep2_dwagon.sh"
export DSV4_TENSOR_PARALLEL_SIZE=2
export DSV4_EXPERT_PARALLEL_SIZE=2
export DSV4_CUDA_VISIBLE_DEVICES=0,1
export DSV4_GPU_EXPERTS_PER_LAYER=0
export DSV4_EXPERT_LOCATION_MODE=mask
export DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN="${DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN:-0}"
export DSV4_CPUINFER_THREADS="${DSV4_CPUINFER_THREADS:-56}"
export DSV4_KT_THREADPOOL_COUNT=1
export DSV4_KT_NUMA_NODES="0 1"
export DSV4_NUMACTL_NODES=0,1
export DSV4_CACHE_ROOT="$cache_root"
unset SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN
export SGLANG_KT_CPU_EXPERT_SHARD_PLAN="$shard_plan"
export KT_WORKER_SPIN_US="${KT_WORKER_SPIN_US:-1000}"
export KT_AMX_FINE_GRAINED_DECODE="${KT_AMX_FINE_GRAINED_DECODE:-1}"

exec "${repo_root}/scripts/dsv4_flash_fwuff_parity.sh" "$@"
