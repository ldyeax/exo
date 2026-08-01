#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 REVISION QUANT OUTPUT_DIRECTORY" >&2
  exit 2
fi

revision=$1
quant=$2
output_directory=$3
repository=unsloth/Kimi-K3-GGUF
repository_url=https://huggingface.co
tree_url="${repository_url}/api/models/${repository}/tree/${revision}/${quant}?recursive=1&expand=true"

for required_command in aria2c curl jq stat; do
  command -v "${required_command}" >/dev/null
done

install -d -m 0755 "${output_directory}"
temporary_directory=$(mktemp -d)
trap 'rm -rf -- "${temporary_directory}"' EXIT
tree_metadata="${temporary_directory}/tree.json"
download_manifest="${temporary_directory}/aria2-input.txt"

curl \
  --location \
  --fail \
  --silent \
  --show-error \
  "${tree_url}" \
  --output "${tree_metadata}"

jq --exit-status '
  type == "array"
  and length > 0
  and all(.[]; .type == "file" and (.path | endswith(".gguf")))
' "${tree_metadata}" >/dev/null

jq \
  --raw-output \
  --arg repository_url "${repository_url}" \
  --arg repository "${repository}" \
  --arg revision "${revision}" \
  '
    sort_by(.path)[]
    | "\($repository_url)/\($repository)/resolve/\($revision)/\(.path)?download=true\n  out=\(.path | split("/") | last)"
  ' \
  "${tree_metadata}" >"${download_manifest}"

aria2c \
  --input-file="${download_manifest}" \
  --continue=true \
  --dir="${output_directory}" \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --file-allocation=none \
  --max-concurrent-downloads=4 \
  --max-connection-per-server=1 \
  --split=1 \
  --min-split-size=64M \
  --max-tries=0 \
  --retry-wait=10 \
  --timeout=60 \
  --connect-timeout=30 \
  --disk-cache=256M \
  --summary-interval=30 \
  --console-log-level=notice \
  --download-result=full

while IFS=$'\t' read -r file_name expected_size; do
  downloaded_file="${output_directory}/${file_name}"
  actual_size=$(stat --format=%s "${downloaded_file}")
  if [[ ${actual_size} -ne ${expected_size} ]]; then
    echo \
      "Size mismatch for ${downloaded_file}: expected ${expected_size}, got ${actual_size}" \
      >&2
    exit 1
  fi
  allocated_size=$(($(stat --format=%b "${downloaded_file}") * 512))
  if [[ ${allocated_size} -lt ${actual_size} ]]; then
    echo \
      "Sparse/incomplete file ${downloaded_file}: ${actual_size} logical bytes, ${allocated_size} allocated bytes" \
      >&2
    exit 1
  fi
done < <(
  jq \
    --raw-output \
    'sort_by(.path)[] | [(.path | split("/") | last), (.size | tostring)] | @tsv' \
    "${tree_metadata}"
)

remaining_control=$(
  find "${output_directory}" -maxdepth 1 -type f -name '*.aria2' -print -quit
)
if [[ -n ${remaining_control} ]]; then
  echo "Incomplete aria2 control files remain in ${output_directory}" >&2
  exit 1
fi

expected_total=$(jq '[.[].size] | add' "${tree_metadata}")
actual_total=$(du --bytes --summarize "${output_directory}" | cut --fields=1)
echo "Verified ${quant}: ${expected_total} repository bytes (${actual_total} allocated directory bytes)"
