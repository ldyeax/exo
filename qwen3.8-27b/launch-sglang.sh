#!/usr/bin/env bash
set -euo pipefail

model_path=${QWEN_MODEL_PATH:-/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
token_map=${QWEN_TOKEN_MAP:-/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/extension-final-20260815/frspec/qwen38-frspec-english-python-opencode-32768.pt}
flashinfer_overlay=${QWEN_FLASHINFER_OVERLAY:-/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/flashinfer-sm86-overlay}
runtime_python=${QWEN_RUNTIME_PYTHON:-/var/lib/exo/runtimes/dsv4-fwuff-cu130/venv/bin/python}
sglang_source=${QWEN_SGLANG_SOURCE:-/root/exo/vendor/sglang/python}
model_name=${QWEN_MODEL_NAME:-Qwen3.8-27B-FP8}
server_host=${QWEN_SERVER_HOST:-127.0.0.1}
server_port=${QWEN_SERVER_PORT:-30022}

if [[ ! -d $model_path ]]; then
  echo "Model snapshot is missing: $model_path" >&2
  exit 1
fi
if [[ ! -r $token_map ]]; then
  echo "FR-Spec token map is missing: $token_map" >&2
  exit 1
fi
if [[ ! -d $flashinfer_overlay ]]; then
  echo "FlashInfer SM86 overlay is missing: $flashinfer_overlay" >&2
  exit 1
fi
if [[ ! -x $runtime_python ]]; then
  echo "SGLang runtime Python is missing: $runtime_python" >&2
  exit 1
fi
if [[ ! -d $sglang_source ]]; then
  echo "SGLang source tree is missing: $sglang_source" >&2
  exit 1
fi

exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONPATH="$flashinfer_overlay:$sglang_source" \
  XDG_CACHE_HOME="${QWEN_XDG_CACHE_HOME:-/tmp/qwen38-sgl-xdg}" \
  TORCH_EXTENSIONS_DIR="${QWEN_TORCH_EXTENSIONS_DIR:-/tmp/qwen38-sgl-torch-fixed}" \
  TRITON_CACHE_DIR="${QWEN_TRITON_CACHE_DIR:-/tmp/qwen38-sgl-triton}" \
  FLASHINFER_WORKSPACE_BASE="${QWEN_FLASHINFER_WORKSPACE_BASE:-/tmp/qwen38-sgl-flashinfer-fixed}" \
  SGLANG_FORCE_FP8_MARLIN=1 \
  SGLANG_FR_SPEC_FP8_MARLIN_HEAD=1 \
  SGLANG_DISABLE_FLASHINFER_NORM=1 \
  SGLANG_ENABLE_TORCH_INFERENCE_MODE=1 \
  SGLANG_OLLAMA_ROOT_ROUTE=/ \
  NCCL_P2P_LEVEL=NVL \
  PYTHONUNBUFFERED=1 \
  "$runtime_python" -m sglang.launch_server \
  --model-path "$model_path" \
  --served-model-name "$model_name" \
  --host "$server_host" \
  --port "$server_port" \
  --tp-size 2 \
  --dtype bfloat16 \
  --quantization fp8 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --max-mamba-cache-size 1 \
  --disable-radix-cache \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --mem-fraction-static 0.96 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --linear-attn-backend triton \
  --language-only \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --page-size 1 \
  --scheduler-recv-interval 16 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$model_path" \
  --speculative-num-steps 6 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --speculative-token-map "$token_map" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --decode-log-interval 100000 \
  --log-level info
