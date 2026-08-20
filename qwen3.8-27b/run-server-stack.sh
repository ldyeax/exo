#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bridge_python=${QWEN_BRIDGE_PYTHON:-python3}
bridge_host=${QWEN_OLLAMA_HOST:-127.0.0.1}
bridge_port=${QWEN_OLLAMA_PORT:-11434}
backend_host=${QWEN_SERVER_HOST:-127.0.0.1}
backend_port=${QWEN_SERVER_PORT:-30022}
model_name=${QWEN_MODEL_NAME:-Qwen3.8-27B-FP8}

server_pid=
bridge_pid=

terminate_children() {
  trap - EXIT HUP INT TERM
  if [[ -n $bridge_pid ]] && kill -0 "$bridge_pid" 2>/dev/null; then
    kill -TERM "$bridge_pid" 2>/dev/null || true
  fi
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
  fi
  [[ -z $bridge_pid ]] || wait "$bridge_pid" 2>/dev/null || true
  [[ -z $server_pid ]] || wait "$server_pid" 2>/dev/null || true
}
trap terminate_children EXIT HUP INT TERM

"$bridge_python" "$script_dir/ollama_openai_bridge.py" \
  --listen-host "$bridge_host" \
  --listen-port "$bridge_port" \
  --backend-url "http://$backend_host:$backend_port/v1/chat/completions" \
  --model "$model_name" \
  --context-length 262144 \
  --max-output-tokens 16384 &
bridge_pid=$!

"$script_dir/launch-sglang.sh" &
server_pid=$!

set +e
wait -n "$bridge_pid" "$server_pid"
status=$?
set -e
exit "$status"
