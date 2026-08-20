#!/usr/bin/env bash
set -euo pipefail

session_name=${QWEN_SCREEN_SESSION:-qwen38-vs2026}

if ! screen -S "$session_name" -Q windows >/dev/null 2>&1; then
  echo "Screen session '$session_name' is not running."
  exit 0
fi

screen -S "$session_name" -X quit
echo "Stopped screen session: $session_name"
