#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${DSV4_PYTHON:-/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python}"
profile_path="${DSV4_HYBRID_EXPERT_PROFILE:-/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v4-agentic-distinct-decode-calls.pt}"
ordering_path="${DSV4_EXPERT_ORDERING:-${repo_root}/scripts/data/dsv4_flash_0731_agentic_expert_order.json}"
cache_root="${DSV4_CACHE_ROOT:-/var/lib/exo/cache/dsv4-flash-hybrid-ep2}"
gpu_experts_per_rank="${DSV4_GPU_EXPERTS_PER_LAYER:-12}"
gpu_selection_strategy="${DSV4_HYBRID_GPU_SELECTION_STRATEGY:-profile-hot-prefix-profile-fill}"
profile_hot_prefix_experts_per_layer="${DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER:-}"
fill_profile_path="${DSV4_HYBRID_EXPERT_FILL_PROFILE:-}"
draft_recorder_profiles_text="${DSV4_DRAFT_EXPERT_RECORDER_PROFILES:-}"
draft_profile_path="${DSV4_DRAFT_HYBRID_EXPERT_PROFILE:-}"
draft_shard_plan="${DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN:-${SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN:-}}"
draft_ordering_layer_indices="${DSV4_DRAFT_ORDERING_LAYER_INDICES:-40,41,42}"
multi_stream_overlap="${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP:-0}"

if [[ $gpu_selection_strategy == "profile-hot-prefix-profile-fill" ]]; then
  profile_hot_prefix_experts_per_layer="${profile_hot_prefix_experts_per_layer:-12}"
  fill_profile_path="${fill_profile_path:-/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v16-32k-decode-experts.pt}"
fi

# Validate values supplied through the environment before Bash evaluates them
# as arithmetic or interpolates them into a cache path. Bash recursively
# expands arithmetic operands, so treating an unchecked string as a number is
# both unsafe and liable to produce surprising octal/overflow behavior.
if [[ ! $gpu_experts_per_rank =~ ^(0|[1-9][0-9]{0,2})$ ]]; then
  echo "DSV4_GPU_EXPERTS_PER_LAYER must be a decimal integer between 0 and 128 for EP2" >&2
  exit 2
fi
gpu_experts_per_rank=$((10#$gpu_experts_per_rank))
if ((gpu_experts_per_rank > 128)); then
  echo "DSV4_GPU_EXPERTS_PER_LAYER must be a decimal integer between 0 and 128 for EP2" >&2
  exit 2
fi
if [[ -n $profile_hot_prefix_experts_per_layer ]]; then
  if [[ ! $profile_hot_prefix_experts_per_layer =~ ^(0|[1-9][0-9]{0,2})$ ]]; then
    echo "DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER must be a decimal integer" >&2
    exit 2
  fi
  profile_hot_prefix_experts_per_layer=$((10#$profile_hot_prefix_experts_per_layer))
fi
case "${multi_stream_overlap,,}" in
true | 1 | yes | y)
  multi_stream_overlap=1
  ;;
false | 0 | no | n)
  multi_stream_overlap=0
  ;;
*)
  echo "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP must be a boolean value" >&2
  exit 2
  ;;
esac

cpu_experts_per_rank=$(((256 - (2 * gpu_experts_per_rank)) / 2))
selection_cache_suffix=""
if [[ $gpu_selection_strategy == "profile-hot-prefix-profile-fill" && -n $profile_hot_prefix_experts_per_layer ]]; then
  selection_cache_suffix="-profile-hot${profile_hot_prefix_experts_per_layer}-profile-fill"
fi
explicit_shard_plan="${DSV4_HYBRID_EXPERT_SHARD_PLAN:-}"
shard_plan="${explicit_shard_plan:-${cache_root}/hybrid-gpu${gpu_experts_per_rank}${selection_cache_suffix}-ep2.pt}"

if (((256 - (2 * gpu_experts_per_rank)) % 2 != 0)); then
  echo "hybrid EP2 placement must leave an even number of CPU experts" >&2
  exit 2
fi

if [[ -n $explicit_shard_plan ]]; then
  if [[ ! -f $shard_plan || ! -r $shard_plan ]]; then
    echo "hybrid expert shard plan does not exist or is not readable: $shard_plan" >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "$shard_plan")"
  plan_builder_arguments=(
    --profile "$profile_path"
    --ordering "$ordering_path"
    --output "$shard_plan"
    --gpu-rank-counts "${gpu_experts_per_rank},${gpu_experts_per_rank}"
    --cpu-rank-counts "${cpu_experts_per_rank},${cpu_experts_per_rank}"
    --gpu-selection "$gpu_selection_strategy"
  )
  if [[ -n $profile_hot_prefix_experts_per_layer ]]; then
    plan_builder_arguments+=(
      --profile-hot-prefix-experts-per-layer "$profile_hot_prefix_experts_per_layer"
    )
  fi
  if [[ -n $fill_profile_path ]]; then
    plan_builder_arguments+=(--gpu-fill-profile "$fill_profile_path")
  fi
  "$python_path" "$repo_root/scripts/build_dsv4_kt_hybrid_shard_plan.py" \
    "${plan_builder_arguments[@]}"
fi

# Draft placement is deliberately opt-in. A replicated per-pass recorder can
# be reduced to the three DSpark stage rows here, or callers can provide an
# already reduced profile/plan. With none of these variables, the runtime
# retains its all-CPU draft behavior.
if [[ -n $draft_recorder_profiles_text ]]; then
  read -r -a draft_recorder_profiles <<<"$draft_recorder_profiles_text"
  if ((${#draft_recorder_profiles[@]} == 0)); then
    echo "DSV4_DRAFT_EXPERT_RECORDER_PROFILES did not contain a profile" >&2
    exit 2
  fi
  draft_profile_path="${draft_profile_path:-${cache_root}/draft-distinct-decode-calls.pt}"
  "$python_path" "$repo_root/scripts/build_dsv4_decode_call_profile.py" \
    --rank-profiles "${draft_recorder_profiles[@]}" \
    --profile-topology replicated-per-pass \
    --active-prefix-layer-count 3 \
    --decode-routes-per-layer 30 \
    --output "$draft_profile_path"
fi
if [[ -n $draft_profile_path ]]; then
  draft_shard_plan="${draft_shard_plan:-${cache_root}/draft-hybrid-gpu${gpu_experts_per_rank}-ep2.pt}"
  "$python_path" "$repo_root/scripts/build_dsv4_kt_hybrid_shard_plan.py" \
    --profile "$draft_profile_path" \
    --ordering "$ordering_path" \
    --ordering-layer-indices "$draft_ordering_layer_indices" \
    --output "$draft_shard_plan" \
    --gpu-rank-counts "${gpu_experts_per_rank},${gpu_experts_per_rank}" \
    --cpu-rank-counts "${cpu_experts_per_rank},${cpu_experts_per_rank}" \
    --gpu-selection profile-hot
elif [[ -n $draft_shard_plan && ! -f $draft_shard_plan ]]; then
  echo "draft hybrid expert shard plan does not exist: $draft_shard_plan" >&2
  exit 2
fi

export DSV4_LAUNCH_ENTRYPOINT="${repo_root}/scripts/dsv4_flash_hybrid_ep2_dwagon.sh"
export DSV4_TENSOR_PARALLEL_SIZE=2
export DSV4_EXPERT_PARALLEL_SIZE=2
export DSV4_CUDA_VISIBLE_DEVICES=0,1
export DSV4_GPU_EXPERTS_PER_LAYER="$gpu_experts_per_rank"
export DSV4_HYBRID_GPU_SELECTION_STRATEGY="$gpu_selection_strategy"
if [[ -n $profile_hot_prefix_experts_per_layer ]]; then
  export DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER="$profile_hot_prefix_experts_per_layer"
else
  unset DSV4_HYBRID_PROFILE_HOT_PREFIX_EXPERTS_PER_LAYER
fi
if [[ -n $fill_profile_path ]]; then
  export DSV4_HYBRID_EXPERT_FILL_PROFILE="$fill_profile_path"
else
  unset DSV4_HYBRID_EXPERT_FILL_PROFILE
fi
export DSV4_EXPERT_LOCATION_MODE=hybrid
export DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN="${DSV4_MAX_DEFERRED_EXPERTS_PER_TOKEN:-0}"
export DSV4_CPUINFER_THREADS="${DSV4_CPUINFER_THREADS:-56}"
export DSV4_KT_THREADPOOL_COUNT=1
export DSV4_KT_NUMA_NODES="0 1"
export DSV4_NUMACTL_NODES=0,1
export DSV4_CACHE_ROOT="$cache_root"
unset SGLANG_KT_CPU_EXPERT_SHARD_PLAN SGLANG_KT_GPU_EXPERT_MASK_PLAN
export SGLANG_KT_HYBRID_EXPERT_SHARD_PLAN="$shard_plan"
if [[ -n $draft_shard_plan ]]; then
  export DSV4_DRAFT_HYBRID_EXPERT_PROFILE="$draft_profile_path"
  export DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN="$draft_shard_plan"
  export SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN="$draft_shard_plan"
else
  unset DSV4_DRAFT_HYBRID_EXPERT_PROFILE
  unset DSV4_DRAFT_HYBRID_EXPERT_SHARD_PLAN
  unset SGLANG_KT_DRAFT_HYBRID_EXPERT_SHARD_PLAN
fi
export KT_WORKER_SPIN_US="${KT_WORKER_SPIN_US:-1000}"
export KT_AMX_FINE_GRAINED_DECODE="${KT_AMX_FINE_GRAINED_DECODE:-1}"
# On the validated TP2/EP2 placement, gamma 6 amortizes target verification
# while retaining the target-model verification step for every draft block.
export DSV4_DSPARK_BLOCK_SIZE="${DSV4_DSPARK_BLOCK_SIZE:-6}"
# The DSV4 MoE side-stream path is not yet coherent on this hybrid Ampere
# configuration.  Keep the safe single-stream path as the launcher default,
# while preserving an explicit caller override for future revalidation.
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="$multi_stream_overlap"

exec "${repo_root}/scripts/dsv4_flash_fwuff_parity.sh" "$@"
