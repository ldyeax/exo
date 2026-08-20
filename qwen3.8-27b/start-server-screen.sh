#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
session_name=${QWEN_SCREEN_SESSION:-qwen38-vs2026}
screen_log=${QWEN_SCREEN_LOG:-/tmp/qwen38-vs2026-screen.log}

if screen -S "$session_name" -Q windows >/dev/null 2>&1; then
  echo "Screen session '$session_name' is already running."
  exit 0
fi

screen -DmS "$session_name" -L -Logfile "$screen_log" \
  bash "$script_dir/run-server-stack.sh"

sleep 1
if ! screen -S "$session_name" -Q windows >/dev/null 2>&1; then
  echo "Screen session '$session_name' exited during startup; inspect $screen_log" >&2
  exit 1
fi

echo "Started screen session: $session_name"
echo "Attach: screen -r $session_name"
echo "Log: $screen_log"
echo "OpenAI endpoint: http://127.0.0.1:${QWEN_SERVER_PORT:-30022}/v1"
echo "Ollama endpoint: http://127.0.0.1:${QWEN_OLLAMA_PORT:-11434}"
