#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${KIMI_K3_STAGE_SNAPSHOT_ACTIVE:-0} != 1 ||
  -z ${KIMI_K3_STAGE_ORIGINAL_SCRIPT_DIRECTORY:-} ]]; then
  # Bash can lazily parse a long-running script. Re-exec from an unlinked copy
  # so a repository edit cannot change commands that this transaction has not
  # parsed yet; the open descriptor keeps the anonymous copy alive.
  snapshot_file=$(mktemp /tmp/exo-kimi-k3-stage.XXXXXXXX)
  cp -- "${BASH_SOURCE[0]}" "${snapshot_file}"
  exec {snapshot_file_descriptor}<"${snapshot_file}"
  rm -- "${snapshot_file}"
  export KIMI_K3_STAGE_SNAPSHOT_ACTIVE=1
  export KIMI_K3_STAGE_ORIGINAL_SCRIPT_DIRECTORY=${script_directory}
  exec bash "/proc/self/fd/${snapshot_file_descriptor}" "$@"
fi
script_directory=${KIMI_K3_STAGE_ORIGINAL_SCRIPT_DIRECTORY}
unset KIMI_K3_STAGE_SNAPSHOT_ACTIVE
unset KIMI_K3_STAGE_ORIGINAL_SCRIPT_DIRECTORY

quant=UD-Q2_K_XL
destination_root=/mnt/llm-models/Kimi-K3-GGUF
parallel_copies=4
hash_workers=1
plan_only=0
verify_complete_partial=0
source_directories=()
manifest_file=${KIMI_K3_GGUF_MANIFEST_FILE:-"${script_directory}/data/kimi_k3_gguf_manifests.json"}

print_usage() {
  cat <<'USAGE'
Usage: scripts/stage_kimi_k3_gguf.sh [OPTIONS]

Select the fastest complete local or mounted source for one Kimi K3 GGUF
quantization, then resume-copy its shards into a temporary directory and
atomically publish the verified snapshot on dwagon's model RAID.

Options:
  --quant NAME              Quant directory name (default: UD-Q2_K_XL)
  --source DIRECTORY        Candidate quant directory; may be repeated
  --destination-root PATH   Parent for the published quant directory
  --parallel-copies N       Concurrent shard copies (default: 4)
  --hash-workers N          Concurrent SHA-256 readers (default: 1)
  --verify-complete-partial Skip rsync only when an existing partial already
                            has every pinned name, size, and allocated byte;
                            SHA-256 verification still runs before publication
  --plan                    Print the selected source and capacity decision only
  --help                    Show this help

When no --source is supplied, the EDR and Ethernet NFS views are considered.
On dwagon, the measured NVMe-backed NFS path over 10.44.0.2 (EDR) outranks
local block storage; other network filesystems rank lower. Existing partial
files are retained for resumption. Candidate names, sizes, and SHA-256 values
are pinned to scripts/data/kimi_k3_gguf_manifests.json. The expensive content
hash runs after the resumable copy and before atomic publication.
USAGE
}

fail() {
  printf '[kimi-stage] error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[kimi-stage] %s\n' "$*" >&2
}

while (($# > 0)); do
  case "$1" in
  --quant)
    (($# >= 2)) || fail "--quant requires a value"
    quant=$2
    shift
    ;;
  --source)
    (($# >= 2)) || fail "--source requires a value"
    source_directories+=("$2")
    shift
    ;;
  --destination-root)
    (($# >= 2)) || fail "--destination-root requires a value"
    destination_root=$2
    shift
    ;;
  --parallel-copies)
    (($# >= 2)) || fail "--parallel-copies requires a value"
    parallel_copies=$2
    shift
    ;;
  --hash-workers)
    (($# >= 2)) || fail "--hash-workers requires a value"
    hash_workers=$2
    shift
    ;;
  --verify-complete-partial)
    verify_complete_partial=1
    ;;
  --plan)
    plan_only=1
    ;;
  --help)
    print_usage
    exit 0
    ;;
  *)
    fail "unknown argument: $1"
    ;;
  esac
  shift
done

[[ ${quant} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
  fail "quant must be a safe directory name"
[[ ${destination_root} == /* ]] ||
  fail "destination root must be absolute"
[[ ${parallel_copies} =~ ^[1-9][0-9]*$ ]] ||
  fail "parallel copy count must be positive"
((parallel_copies <= 16)) ||
  fail "parallel copy count must not exceed 16"
[[ ${hash_workers} =~ ^[1-9][0-9]*$ ]] ||
  fail "hash worker count must be positive"
((hash_workers <= 16)) ||
  fail "hash worker count must not exceed 16"
[[ ${manifest_file} == /* && -f ${manifest_file} && ! -L ${manifest_file} ]] ||
  fail "pinned manifest must be an absolute regular non-symlink file: ${manifest_file}"

for required_command in awk bash chmod cmp cut date find findmnt flock install jq mktemp mv rsync sha256sum sort stat sync xargs; do
  command -v "${required_command}" >/dev/null ||
    fail "required command is unavailable: ${required_command}"
done

if ((${#source_directories[@]} == 0)); then
  source_directories=(
    "/mnt/sanic-edr/llm_models/Kimi-K3-GGUF/${quant}"
    "/mnt/sanic/llm_models/Kimi-K3-GGUF/${quant}"
  )
fi

working_directory=$(mktemp -d)
trap 'rm -rf -- "${working_directory}"' EXIT

manifest_repository=$(jq --raw-output '.repository // empty' "${manifest_file}")
manifest_revision=$(jq --raw-output '.revision // empty' "${manifest_file}")
manifest_file_sha256=$(sha256sum "${manifest_file}" | awk '{ print $1 }')
pinned_manifest="${working_directory}/pinned.tsv"

jq --exit-status \
  --raw-output \
  --arg quant "${quant}" \
  '
    select(
      .schema_version == 1 and
      (.repository | type == "string" and length > 0) and
      (.revision | type == "string" and test("^[0-9a-f]{40}$")) and
      (.quantizations[$quant] | type == "array" and length > 0)
    ) |
    .quantizations[$quant][] |
    @tsv
  ' \
  "${manifest_file}" >"${pinned_manifest}" ||
  fail "quant is absent or malformed in the pinned manifest: ${quant}"

awk -F '\t' '
  NF != 3 ||
  $1 !~ /^[A-Za-z0-9][A-Za-z0-9_.-]*[.]gguf$/ ||
  $2 !~ /^[1-9][0-9]*$/ ||
  $3 !~ /^[0-9a-f]{64}$/ {
    exit 1
  }
  seen[$1]++ {
    exit 1
  }
' "${pinned_manifest}" ||
  fail "pinned manifest contains an unsafe, malformed, or duplicate entry"

sort --output="${pinned_manifest}" "${pinned_manifest}"
manifest_sha256=$(sha256sum "${pinned_manifest}" | awk '{ print $1 }')
pinned_name_size_manifest="${working_directory}/pinned-name-size.tsv"
cut --fields=1,2 "${pinned_manifest}" | sort >"${pinned_name_size_manifest}"
pinned_count=$(awk 'END { print NR }' "${pinned_manifest}")
pinned_bytes=$(awk -F '\t' '{ total += $2 } END { printf "%.0f", total }' "${pinned_manifest}")

write_manifest() {
  local directory=$1
  local output=$2
  find "${directory}" -maxdepth 1 -type f -name '*.gguf' \
    -printf '%f\t%s\n' |
    sort >"${output}"
}

validate_manifest() {
  local manifest=$1
  local file_count
  file_count=$(awk 'END { print NR }' "${manifest}")
  ((file_count > 0)) || return 1
  awk -F '\t' '
    NF != 2 ||
    $1 !~ /^[A-Za-z0-9][A-Za-z0-9_.-]*[.]gguf$/ ||
    $2 !~ /^[1-9][0-9]*$/ {
      exit 1
    }
  ' "${manifest}"
}

source_score() {
  local directory=$1
  local filesystem_type
  local mount_source
  read -r filesystem_type mount_source < <(
    findmnt --noheadings --output FSTYPE,SOURCE --target "${directory}" |
      awk 'END { print $1, $2 }'
  )
  case "${filesystem_type}:${mount_source}" in
  nfs*:10.44.0.2:*)
    # fwuff's three-NVMe RAID over EDR is materially faster than dwagon's
    # ten-HDD RAID for the large serial reads used by the model loader.
    printf '600\n'
    ;;
  nfs*:*)
    printf '200\n'
    ;;
  fuse.*:* | cifs:* | sshfs:*)
    printf '100\n'
    ;;
  *)
    printf '500\n'
    ;;
  esac
}

selected_source=
selected_manifest=
selected_score=-1
selected_bytes=0
selected_count=0
candidate_index=0

for candidate in "${source_directories[@]}"; do
  [[ ${candidate} == /* && -d ${candidate} ]] || continue
  candidate_manifest="${working_directory}/candidate-${candidate_index}.tsv"
  candidate_index=$((candidate_index + 1))
  write_manifest "${candidate}" "${candidate_manifest}"
  validate_manifest "${candidate_manifest}" || continue
  if ! cmp --silent "${pinned_name_size_manifest}" "${candidate_manifest}"; then
    log "rejecting candidate whose GGUF names or sizes differ from pinned ${quant}: ${candidate}"
    continue
  fi
  candidate_score=$(source_score "${candidate}")
  candidate_bytes=$(awk -F '\t' '{ total += $2 } END { printf "%.0f", total }' "${candidate_manifest}")
  candidate_count=$(awk 'END { print NR }' "${candidate_manifest}")
  log "candidate score=${candidate_score} files=${candidate_count} bytes=${candidate_bytes} path=${candidate}"
  if ((candidate_score > selected_score)); then
    selected_source=$candidate
    selected_manifest=$candidate_manifest
    selected_score=$candidate_score
    selected_bytes=$candidate_bytes
    selected_count=$candidate_count
  fi
done

[[ -n ${selected_source} ]] ||
  fail "no source exactly matches the pinned ${quant} manifest"
((selected_count == pinned_count && selected_bytes == pinned_bytes)) ||
  fail "selected source totals diverge from the pinned manifest"

destination_directory="${destination_root}/${quant}"
partial_directory="${destination_root}/.${quant}.exo-partial"
reserve_bytes=$((selected_bytes / 10 + 16 * 1024 * 1024 * 1024))
required_bytes=$((selected_bytes + reserve_bytes))

if [[ -d ${destination_root} ]]; then
  available_bytes=$(
    stat --file-system --format='%a %S' "${destination_root}" |
      awk '{ printf "%.0f", $1 * $2 }'
  )
else
  destination_parent=${destination_root%/*}
  [[ -d ${destination_parent} ]] ||
    fail "destination parent does not exist: ${destination_parent}"
  available_bytes=$(
    stat --file-system --format='%a %S' "${destination_parent}" |
      awk '{ printf "%.0f", $1 * $2 }'
  )
fi

jq --null-input \
  --arg quant "${quant}" \
  --arg source "${selected_source}" \
  --arg destination "${destination_directory}" \
  --arg manifest_repository "${manifest_repository}" \
  --arg manifest_revision "${manifest_revision}" \
  --arg manifest_sha256 "${manifest_sha256}" \
  --arg manifest_file_sha256 "${manifest_file_sha256}" \
  --argjson source_score "${selected_score}" \
  --argjson file_count "${selected_count}" \
  --argjson expected_bytes "${selected_bytes}" \
  --argjson reserve_bytes "${reserve_bytes}" \
  --argjson available_bytes "${available_bytes}" \
  --argjson hash_workers "${hash_workers}" \
  '{
    quant: $quant,
    source: $source,
    destination: $destination,
    pinned_manifest: {
      repository: $manifest_repository,
      revision: $manifest_revision,
      quant_manifest_sha256: $manifest_sha256,
      source_file_sha256: $manifest_file_sha256
    },
    source_score: $source_score,
    file_count: $file_count,
    expected_bytes: $expected_bytes,
    reserve_bytes: $reserve_bytes,
    available_bytes: $available_bytes,
    content_verification: {
      algorithm: "sha256",
      workers: $hash_workers,
      timing: "after copy and before atomic publication"
    },
    capacity_ok: ($available_bytes >= ($expected_bytes + $reserve_bytes))
  }'

((available_bytes >= required_bytes)) ||
  fail "destination lacks the model bytes plus a 10% and 16-GiB reserve"
((plan_only == 0)) || exit 0

install --directory --mode=0750 -- "${destination_root}"
exec 9>"${destination_root}/.exo-kimi-k3-stage.lock"
flock --nonblock 9 ||
  fail "another Kimi K3 staging transaction owns the destination lock"

validate_non_sparse_snapshot() {
  local directory=$1
  local file_name
  local expected_size
  local expected_hash
  local actual_size
  local allocated_size
  local snapshot_file

  while IFS=$'\t' read -r file_name expected_size expected_hash; do
    snapshot_file="${directory}/${file_name}"
    [[ -f ${snapshot_file} && ! -L ${snapshot_file} ]] ||
      fail "snapshot entry is not a regular non-symlink file: ${snapshot_file}"
    actual_size=$(stat --format=%s "${snapshot_file}")
    ((actual_size == expected_size)) ||
      fail "snapshot entry has the wrong logical size: ${snapshot_file}"
    allocated_size=$(($(stat --format=%b "${snapshot_file}") * 512))
    ((allocated_size >= expected_size)) ||
      fail "snapshot entry is sparse or incomplete: ${snapshot_file}"
  done <"${pinned_manifest}"
}

verify_snapshot_content() {
  local directory=$1
  local file_name
  local expected_size
  local expected_hash

  export KIMI_STAGE_HASH_DIRECTORY=${directory}
  log "verifying ${pinned_count} pinned SHA-256 values with ${hash_workers} worker(s)"
  while IFS=$'\t' read -r file_name expected_size expected_hash; do
    printf '%s\0%s\0' "${file_name}" "${expected_hash}"
  done <"${pinned_manifest}" |
    xargs \
      --null \
      --max-args=2 \
      --max-procs="${hash_workers}" \
      bash -c '
        set -euo pipefail
        file_name=$1
        expected_hash=$2
        hash_output=$(sha256sum -- "${KIMI_STAGE_HASH_DIRECTORY}/${file_name}")
        actual_hash=${hash_output%% *}
        if [[ ${actual_hash} != "${expected_hash}" ]]; then
          printf "[kimi-stage] error: SHA-256 mismatch for %s: expected=%s actual=%s\n" \
            "${file_name}" "${expected_hash}" "${actual_hash}" >&2
          exit 1
        fi
        printf "[kimi-stage] verified sha256=%s file=%s\n" "${actual_hash}" "${file_name}" >&2
      ' _
}

write_receipt() {
  local directory=$1
  local receipt_temporary
  local receipt

  receipt="${directory}/.exo-kimi-k3-stage-receipt.json"
  receipt_temporary=$(mktemp --tmpdir="${directory}" .exo-kimi-k3-stage-receipt.json.tmp.XXXXXX)
  jq --null-input \
    --argjson schema_version 2 \
    --arg quant "${quant}" \
    --arg source "${selected_source}" \
    --arg destination "${destination_directory}" \
    --arg completed_at "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    --arg manifest_repository "${manifest_repository}" \
    --arg manifest_revision "${manifest_revision}" \
    --arg manifest_sha256 "${manifest_sha256}" \
    --argjson file_count "${selected_count}" \
    --argjson expected_bytes "${selected_bytes}" \
    '{
      schema_version: $schema_version,
      quant: $quant,
      source: $source,
      destination: $destination,
      completed_at: $completed_at,
      pinned_manifest: {
        repository: $manifest_repository,
        revision: $manifest_revision,
        quant_manifest_sha256: $manifest_sha256
      },
      file_count: $file_count,
      expected_bytes: $expected_bytes,
      verification: "pinned names, exact logical sizes, non-sparse allocation, and per-shard SHA-256"
    }' >"${receipt_temporary}"
  chmod 0640 "${receipt_temporary}"
  mv -- "${receipt_temporary}" "${receipt}"
}

receipt_is_current() {
  local receipt=$1
  [[ -f ${receipt} && ! -L ${receipt} ]] || return 1
  jq --exit-status \
    --arg quant "${quant}" \
    --arg destination "${destination_directory}" \
    --arg manifest_repository "${manifest_repository}" \
    --arg manifest_revision "${manifest_revision}" \
    --arg manifest_sha256 "${manifest_sha256}" \
    --argjson file_count "${selected_count}" \
    --argjson expected_bytes "${selected_bytes}" \
    '
      .schema_version == 2 and
      .quant == $quant and
      .destination == $destination and
      .pinned_manifest.repository == $manifest_repository and
      .pinned_manifest.revision == $manifest_revision and
      .pinned_manifest.quant_manifest_sha256 == $manifest_sha256 and
      .file_count == $file_count and
      .expected_bytes == $expected_bytes and
      .verification == "pinned names, exact logical sizes, non-sparse allocation, and per-shard SHA-256"
    ' \
    "${receipt}" >/dev/null
}

if [[ -e ${destination_directory} ]]; then
  [[ -d ${destination_directory} && ! -L ${destination_directory} ]] ||
    fail "published destination is not a regular directory"
  published_manifest="${working_directory}/published.tsv"
  write_manifest "${destination_directory}" "${published_manifest}"
  if ! cmp --silent "${pinned_name_size_manifest}" "${published_manifest}"; then
    fail "published destination does not match the pinned manifest"
  fi
  validate_non_sparse_snapshot "${destination_directory}"
  receipt="${destination_directory}/.exo-kimi-k3-stage-receipt.json"
  if receipt_is_current "${receipt}"; then
    log "SHA-256-verified snapshot is already published at ${destination_directory}"
    exit 0
  fi
  log "published snapshot lacks a current content receipt; verifying it in place"
  verify_snapshot_content "${destination_directory}"
  write_receipt "${destination_directory}"
  sync --file-system "${destination_directory}"
  log "upgraded the published ${quant} snapshot to a pinned SHA-256 receipt"
  exit 0
fi

if [[ -e ${partial_directory} ]]; then
  [[ -d ${partial_directory} && ! -L ${partial_directory} ]] ||
    fail "partial destination is not a regular directory"
else
  install --directory --mode=0750 -- "${partial_directory}"
fi

skip_resumable_copy=0
if ((verify_complete_partial == 1)); then
  existing_partial_manifest="${working_directory}/existing-partial.tsv"
  write_manifest "${partial_directory}" "${existing_partial_manifest}"
  if cmp --silent "${pinned_name_size_manifest}" "${existing_partial_manifest}"; then
    validate_non_sparse_snapshot "${partial_directory}"
    skip_resumable_copy=1
    log "existing partial has every pinned name, size, and allocated byte; skipping redundant rsync before SHA-256 verification"
  else
    log "existing partial is not structurally complete; resuming the copy before SHA-256 verification"
  fi
fi

if ((skip_resumable_copy == 0)); then
  export KIMI_STAGE_SELECTED_SOURCE=${selected_source}
  export KIMI_STAGE_PARTIAL_DIRECTORY=${partial_directory}
  export KIMI_STAGE_RSYNC_PROTECT_ARGS=1

  log "copying ${selected_count} shards with ${parallel_copies} EDR-aware workers"
  cut --fields=1 "${selected_manifest}" |
    while IFS= read -r file_name; do
      printf '%s\0' "${file_name}"
    done |
    xargs --null --max-args=1 --max-procs="${parallel_copies}" \
      bash -c '
        set -euo pipefail
        file_name=$1
        rsync \
          --archive \
          --partial \
          --append-verify \
          --no-whole-file \
          --protect-args \
          -- "${KIMI_STAGE_SELECTED_SOURCE}/${file_name}" \
          "${KIMI_STAGE_PARTIAL_DIRECTORY}/${file_name}"
      ' _
fi

staged_manifest="${working_directory}/staged.tsv"
write_manifest "${partial_directory}" "${staged_manifest}"
cmp --silent "${pinned_name_size_manifest}" "${staged_manifest}" ||
  fail "staged shard names or sizes do not match the pinned manifest"

validate_non_sparse_snapshot "${partial_directory}"
verify_snapshot_content "${partial_directory}"
write_receipt "${partial_directory}"
sync --file-system "${partial_directory}"
mv --no-clobber -- "${partial_directory}" "${destination_directory}"
[[ -d ${destination_directory} && ! -e ${partial_directory} ]] ||
  fail "atomic publication lost a race; verified partial snapshot was retained"
sync --file-system "${destination_directory}"
sync --file-system "${destination_root}"
log "published ${quant} atomically at ${destination_directory}"
