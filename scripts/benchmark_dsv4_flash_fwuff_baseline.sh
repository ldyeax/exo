#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${DSV4_PYTHON:-/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python}"

exec "$python_path" "$repo_root/scripts/benchmark_dsv4_flash_128k.py" "$@"
