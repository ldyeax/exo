#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 PLAN BIND_HOST PORT NUMA_NODE LAYER_START LAYER_END_EXCLUSIVE [THREADS]" >&2
  exit 2
fi

plan=$1
bind_host=$2
port=$3
numa_node=$4
layer_start=$5
layer_end_exclusive=$6
processor_count=${7:-}
model_path=/mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark

case "$(hostname -s)" in
fwuff)
  python_path=/var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-fwuff/venv/bin/python
  export PYTHONPATH=/var/lib/exo/sources/ktransformers-dsv4-native-mxfp4:/var/lib/exo/runtimes/dsv4-native-mxfp4/fwuff/amx-prefill-v1/site-packages
  processor_count=${processor_count:-60}
  sidecar_script=/var/lib/exo/scripts/dsv4_mxfp4_expert_sidecar.py
  ;;
*)
  python_path=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
  export PYTHONPATH=/root/exo/vendor/ktransformers
  # Match the 56 physical cores in one dwagon socket; do not use SMT workers.
  processor_count=${processor_count:-56}
  sidecar_script=/root/exo/scripts/dsv4_mxfp4_expert_sidecar.py
  ;;
esac

export KT_KERNEL_VARIANT=amx
export KT_CPU_BACKEND=amx
export KT_MXFP4_BACKEND=amx
# Preserve the measured AMX/AVX512 crossover used by the owning ranks.
export KT_MXFP4_AMX_MIN_EXPERT_TOKENS=5
export KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS=2
export KT_AMX_FINE_GRAINED_DECODE=${KT_AMX_FINE_GRAINED_DECODE:-1}
# Sidecars coexist with serving pools. A short spin preserves back-to-back
# layer latency while allowing inactive pipeline stages to relinquish cores.
export KT_WORKER_SPIN_US=${KT_WORKER_SPIN_US:-1000}
# This process owns CPU AMX/AVX512 experts only.  Hiding CUDA avoids an
# otherwise idle 256 MiB context on the serving GPU; the expert mask does not
# need pinned storage in this mode.
export KT_CPU_ONLY_SIDECAR=1
export CUDA_VISIBLE_DEVICES=-1

"${python_path}" -c '
import inspect
from kt_kernel.utils.amx import NativeMoEWrapper
if "weight_expert_ids" not in inspect.getsource(NativeMoEWrapper.load_weights):
    raise RuntimeError("installed kt_kernel lacks compact expert-shard loading")
'

exec /usr/bin/numactl \
  --cpunodebind="${numa_node}" \
  --membind="${numa_node}" \
  "${python_path}" "${sidecar_script}" \
  --model "${model_path}" \
  --plan "${plan}" \
  --bind-host "${bind_host}" \
  --port "${port}" \
  --processor-count "${processor_count}" \
  --numa-node "${numa_node}" \
  --max-tokens 4096 \
  --layer-start "${layer_start}" \
  --layer-end "${layer_end_exclusive}"
