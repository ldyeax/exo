#!/usr/bin/env bash
set -euo pipefail

session_name=${QWEN_SCREEN_SESSION:-qwen38-vs2026}
server_port=${QWEN_SERVER_PORT:-30022}
bridge_port=${QWEN_OLLAMA_PORT:-11434}

if screen -S "$session_name" -Q windows >/dev/null 2>&1; then
  echo "screen: running ($session_name)"
else
  echo "screen: stopped ($session_name)"
fi

if curl --fail --silent --show-error --max-time 5 \
  "http://127.0.0.1:$server_port/health" >/dev/null; then
  echo "SGLang: ready (http://127.0.0.1:$server_port/v1)"
else
  echo "SGLang: not ready"
fi

if curl --fail --silent --show-error --max-time 5 \
  "http://127.0.0.1:$bridge_port/api/tags" >/dev/null; then
  echo "Ollama bridge: ready (http://127.0.0.1:$bridge_port)"
else
  echo "Ollama bridge: not ready"
fi
