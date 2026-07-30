#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[012]$ ]]; then
  echo "usage: $0 <pipeline-rank: 0|1|2>" >&2
  exit 2
fi

pipeline_rank=$1
model_path=/mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark
rendezvous_address=${DSV4_RENDEZVOUS_ADDRESS:-10.44.0.1:29531}
gpu_experts_per_layer=${DSV4_KT_NUM_GPU_EXPERTS:-0}
max_running_requests=${DSV4_MAX_RUNNING_REQUESTS:-1}
chunked_prefill_size=${DSV4_CHUNKED_PREFILL_SIZE:-1024}
max_prefill_tokens=${DSV4_MAX_PREFILL_TOKENS:-1024}
mem_fraction_static=${DSV4_MEM_FRACTION_STATIC:-0.99}
cuda_graph_max_batch_size=${DSV4_CUDA_GRAPH_MAX_BS_DECODE:-1}
read -r -a cuda_graph_batch_sizes <<<"${DSV4_CUDA_GRAPH_BS_DECODE:-1}"
cuda_graph_backend_decode=${DSV4_CUDA_GRAPH_BACKEND_DECODE:-breakable}

export CUDA_VISIBLE_DEVICES=0
export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_BACKEND=amx
# AMX wins once a routed expert has at least five verify tokens; four-token
# experts remain on AVX512 because padding them to the 32-row tile is slower.
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=5
# The native AVX512 kernel reuses each decoded weight row across two- and
# three-token expert tails. Isolated same-checkpoint A/Bs make this profitable
# starting at two tokens while preserving the one-token mat-vec path.
export KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS=2
export KT_AMX_FINE_GRAINED_DECODE=${KT_AMX_FINE_GRAINED_DECODE:-0}
# Decode advances one pipeline stage at a time. Sleeping idle native workers
# after a short spin avoids burning cycles in clock_gettime on inactive stages.
export KT_WORKER_SPIN_US=${KT_WORKER_SPIN_US:-1000}
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
# Preserve genuine DSpark proposal state.  The target verifier remains on the
# breakable CUDA graph; only the currently incoherent draft graph is disabled.
export SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1
export SGLANG_KT_DRAFT_GPU_EXPERTS=0
# Leave enough host memory on fwuff for the complete three-layer DSpark draft.
export SGLANG_PP_LAYER_PARTITION=${SGLANG_PP_LAYER_PARTITION:-23,25,13}
export FLASHINFER_CUDA_ARCH_LIST=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

if [[ ${pipeline_rank} -eq 2 ]]; then
  export PYTHONPATH=/var/lib/exo/sources/sglang-dspark-30261/python:/var/lib/exo/sources/ktransformers-dsv4-native-mxfp4:/var/lib/exo/runtimes/dsv4-native-mxfp4/fwuff/amx-prefill-v1/site-packages
  export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-fwuff/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
  export NCCL_SOCKET_IFNAME=ibs2
  export GLOO_SOCKET_IFNAME=ibs2
  export NCCL_IB_HCA=mlx5_0
  python_path=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-fwuff/venv/bin/python
  processor_count=60
  numa_node=0
  numa_arguments=(--cpunodebind="${numa_node}" --membind="${numa_node}")
  listen_host=10.44.0.2
  listen_port=30000
else
  export LD_PRELOAD=/opt/cuda/lib64/libcudart.so.13
  export PYTHONPATH=/var/lib/exo/sources/sglang-dspark-30261/python:/root/exo/vendor/ktransformers
  export TILELANG_LIBCUDART_PATH=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
  export NCCL_SOCKET_IFNAME=ibs5
  export GLOO_SOCKET_IFNAME=ibs5
  export NCCL_IB_HCA=mlx5_0
  python_path=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
  # Use every physical core in the selected 56-core socket. SMT siblings stay
  # available to Python/network work but are not assigned native MXFP4 workers.
  processor_count=56
  # The stage feeding fwuff sends the large per-token hidden-state tensor.
  # Put that stage on GPU/NUMA 0, local to both IB HCAs. PP0 then uses the
  # NVLink-connected GPU/NUMA 1; its returned control/logit traffic is small.
  numa_node=$((1 - pipeline_rank))
  # A full local-expert rank does not leave enough per-socket headroom for a
  # hard memory wall. More aggressive remote-expert plans can opt back into
  # strict binding once their projected and measured residency fits.
  if [[ ${DSV4_STRICT_MAIN_NUMA:-0} == 1 ]]; then
    numa_arguments=(--cpunodebind="${numa_node}" --membind="${numa_node}")
  else
    numa_arguments=(--cpunodebind="${numa_node}" --localalloc)
  fi
  listen_host=127.0.0.1
  # SGLang starts a per-node health listener on nonzero PP ranks. Two PP
  # ranks share dwagon, so rank 1 must not contend with rank 0's API port.
  listen_port=$((30000 + pipeline_rank))
  export CUDA_VISIBLE_DEVICES=$((1 - pipeline_rank))
fi

export TRITON_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/triton-pro-clean-pp3-rank${pipeline_rank}
export FLASHINFER_WORKSPACE_BASE=/var/lib/exo/cache/dsv4-native-mxfp4/flashinfer-pro-clean-pp3-rank${pipeline_rank}
export TORCHINDUCTOR_CACHE_DIR=/var/lib/exo/cache/dsv4-native-mxfp4/torch-pro-clean-pp3-rank${pipeline_rank}

# Compact GPU/remote ownership relies on NativeMoEWrapper filtering the
# checkpoint pointer table by global expert ID. Fail before allocating the
# model if an older installed kt_kernel Python package silently ignores it.
"${python_path}" -c '
import inspect
from kt_kernel.utils.amx import NativeMoEWrapper
if "weight_expert_ids" not in inspect.getsource(NativeMoEWrapper.load_weights):
    raise RuntimeError("installed kt_kernel lacks compact expert-shard loading")
'

recorder_arguments=()
if [[ ${SGLANG_DSV4_RECORD_EXPERT_DISTRIBUTION:-0} == 1 ]]; then
  recorder_arguments=(
    --expert-distribution-recorder-mode stat
    # One stat row is [61, 384] int32 (~94 KiB). 2,048 rows cover the
    # 32K/4K single-stream profile without consuming the entire 24 GiB GPU.
    --expert-distribution-recorder-buffer-size 2048
  )
fi

exec /usr/bin/numactl "${numa_arguments[@]}" \
  "${python_path}" -m sglang.launch_server \
  --host "${listen_host}" \
  --port "${listen_port}" \
  --model "${model_path}" \
  --kt-weight-path "${model_path}" \
  --kt-method MXFP4 \
  --kt-num-gpu-experts "${gpu_experts_per_layer}" \
  --kt-cpuinfer "${processor_count}" \
  --kt-threadpool-count 1 \
  --kt-numa-nodes "${numa_node}" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 3 \
  --nnodes 3 \
  --node-rank "${pipeline_rank}" \
  --dist-init-addr "${rendezvous_address}" \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --context-length 1048576 \
  --max-total-tokens 1048576 \
  --attention-backend flashinfer \
  --moe-runner-backend triton \
  --mem-fraction-static "${mem_fraction_static}" \
  --chunked-prefill-size "${chunked_prefill_size}" \
  --max-prefill-tokens "${max_prefill_tokens}" \
  --max-running-requests "${max_running_requests}" \
  --watchdog-timeout 2400 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --cuda-graph-backend-decode "${cuda_graph_backend_decode}" \
  --cuda-graph-max-bs-decode "${cuda_graph_max_batch_size}" \
  --cuda-graph-bs-decode "${cuda_graph_batch_sizes[@]}" \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  "${recorder_arguments[@]}" \
  --skip-server-warmup
