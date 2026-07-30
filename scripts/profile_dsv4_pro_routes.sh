#!/usr/bin/env bash
set -euo pipefail

base_url=${DSV4_BASE_URL:-http://127.0.0.1:30000}
request_id=${DSV4_PROFILE_REQUEST_ID:-dsv4-pro-routes-random32k-out4096}
output_file=${DSV4_PROFILE_OUTPUT_FILE:-/var/lib/exo/benchmarks/dsv4-pro-routes-random32k-out4096-seed42.jsonl}
sglang_source=/var/lib/exo/sources/sglang-dspark-30261
python_path=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python

if [[ -e ${output_file} ]]; then
  echo "refusing to overwrite benchmark artifact: ${output_file}" >&2
  exit 2
fi

curl --fail --silent --show-error --request POST \
  "${base_url}/start_expert_distribution_record"

# Do not add a trap that issues recorder controls here. If the client is
# interrupted, pipeline ranks may still be finishing the streaming request.
# Stop/dump is intentionally reached only after bench_serve exits successfully
# at a request boundary.
PYTHONPATH="${sglang_source}/python" "${python_path}" \
  -m sglang.benchmark.serving \
  --backend sglang \
  --base-url "${base_url}" \
  --dataset-name random \
  --random-input-len 32768 \
  --random-output-len 4096 \
  --random-range-ratio 1.0 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --seed 42 \
  --temperature 0 \
  --tokenize-prompt \
  --warmup-requests 0 \
  --extra-request-body "{\"rid\":\"${request_id}\"}" \
  --output-file "${output_file}"

curl --fail --silent --show-error --request POST \
  "${base_url}/stop_expert_distribution_record"
curl --fail --silent --show-error --request POST \
  "${base_url}/dump_expert_distribution_record"
