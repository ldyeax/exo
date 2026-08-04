#!/usr/bin/env python3
"""Controlled DSV4 prompt-hotspot and speculative-verify benchmark.

The five-phase sequence distinguishes exact/partial radix reuse from warm CPU
weight-page locality without dropping the OS page cache or restarting the
server.  It consumes the existing DSpark debug dump and emits aggregates only;
prompt text, generated text, and per-token IDs are never written to the receipt.

For an undistorted performance run, launch with::

  SGLANG_DSPARK_DEBUG_DUMP=core,reqs,step_cpu_time,step_gpu_time,\
draft_gpu_time,target_verify_gpu_time

Add ``verify_logits`` only for a diagnostic run that needs per-position target
logit margins.  Exact verify tiers require a non-static ragged verify mode and
are selected through the existing ``/set_internal_state`` control plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, cast

try:
    from scripts import benchmark_dsv4_flash_128k as baseline
    from scripts.qualify_dsv4_tp2_interconnect import (
        QualificationError as InterconnectQualificationError,
    )
    from scripts.qualify_dsv4_tp2_interconnect import (
        counter_deltas as interconnect_counter_deltas,
    )
    from scripts.qualify_dsv4_tp2_interconnect import (
        parse_nvlink_counters,
        require_command,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    import benchmark_dsv4_flash_128k as baseline
    from qualify_dsv4_tp2_interconnect import (
        QualificationError as InterconnectQualificationError,
    )
    from qualify_dsv4_tp2_interconnect import (
        counter_deltas as interconnect_counter_deltas,
    )
    from qualify_dsv4_tp2_interconnect import (
        parse_nvlink_counters,
        require_command,
    )


RECEIPT_VERSION = 2
FLUSH_RETRY_SECONDS = 0.1
NVLINK_COUNTER_COMMAND = ("nvidia-smi", "nvlink", "-gt", "d")
NVLINK_DEVICES = (0, 1)
NVLINK_LINKS_PER_DEVICE = 4
MINIMUM_DETERMINISTIC_REPEAT_COUNT = 2
MAXIMUM_DETERMINISTIC_REPEAT_COUNT = 5
MAXIMUM_DETERMINISTIC_REPEAT_OUTPUT_TOKENS = 256
REQUIRED_TRACE_COMPONENTS = frozenset(
    {
        "core",
        "reqs",
        "step_cpu_time",
        "step_gpu_time",
        "draft_gpu_time",
        "target_verify_gpu_time",
    }
)
OSCAR_SPLIT_HISTORY_EXECUTION = "sm86-oscar-int2-split-history-fp32-online-v1"
OSCAR_SPLIT_HISTORY_SPLIT_MAP = {
    "1": 16,
    "2": 16,
    "3": 8,
    "4": 4,
    "5": 4,
    "6": 4,
    "7": 4,
    "8": 2,
}
OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES = 4_210_688
OSCAR_SPLIT_HISTORY_WORKER_FIELDS = {
    "dsv4_oscar_int2_split_history": True,
    "dsv4_oscar_int2_split_history_execution": OSCAR_SPLIT_HISTORY_EXECUTION,
    "dsv4_oscar_int2_split_history_split_map": OSCAR_SPLIT_HISTORY_SPLIT_MAP,
    "dsv4_oscar_int2_split_history_workspace_bytes": (
        OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
    ),
    "dsv4_oscar_int2_split_history_max_partial_rows": 32,
    "dsv4_oscar_int2_split_history_sink_owner": "stage2-exactly-once",
    "dsv4_oscar_int2_split_history_prefill_enabled": False,
    "dsv4_oscar_int2_split_history_fixed_address": True,
}


class HotspotBenchmarkError(RuntimeError):
    pass


class ReadableResponse(Protocol):
    status: int

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def read(self) -> bytes: ...


@dataclass(frozen=True)
class HotspotArguments:
    generate_url: str
    server_info_url: str
    flush_url: str
    control_url: str
    hotspot_url: str
    model_path: Path
    input_tokens: int
    output_tokens: int
    timeout_seconds: float
    flush_timeout_seconds: float
    progress_every: int
    ignore_eos: bool
    near_prefix_ratio: float
    verify_policy: str
    hotspot_plan: Path | None
    hotspot_generation: int | None
    hotspot_commit: bool
    expert_recorder_directory: Path | None
    expected_expert_plan: Path | None
    require_nvlink_traffic: bool
    require_trace: bool
    output_file: Path | None
    require_oscar_split_history: bool = False
    deterministic_repeat_count: int = 0
    deterministic_repeat_output_tokens: int = 128


@dataclass(frozen=True)
class PromptPair:
    exact_ids: list[int]
    near_ids: list[int]
    mutation_record_index: int
    common_prefix_tokens: int
    common_suffix_tokens: int


@dataclass(frozen=True)
class Phase:
    name: str
    prompt_kind: str
    flush_before: bool


PHASES = (
    Phase("cold_first_exact", "exact", True),
    Phase("radix_hot_exact", "exact", False),
    Phase("radix_hot_near", "near", False),
    Phase("warm_no_radix_near", "near", True),
    Phase("warm_no_radix_exact", "exact", True),
)


def _derive_endpoint(generate_url: str, path: str, query: str = "") -> str:
    parsed = urllib.parse.urlsplit(generate_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid generate URL: {generate_url!r}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))


def expected_nvlink_counter_keys() -> frozenset[str]:
    return frozenset(
        f"gpu{device}.link{link}.{direction}_kib"
        for device in NVLINK_DEVICES
        for link in range(NVLINK_LINKS_PER_DEVICE)
        for direction in ("tx", "rx")
    )


def snapshot_nvlink_counters(
    command_reader: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, int]:
    """Read and project the local payload counters used by this TP2 run.

    The raw command output is deliberately discarded.  A caller can inject a
    command reader in unit tests without patching subprocess state.
    """

    read_command = require_command if command_reader is None else command_reader
    try:
        raw_counters = read_command(NVLINK_COUNTER_COMMAND)
    except InterconnectQualificationError as error:
        raise HotspotBenchmarkError(
            f"could not snapshot NVLink payload counters: {error}"
        ) from error
    parsed = parse_nvlink_counters(raw_counters)
    expected = expected_nvlink_counter_keys()
    missing = sorted(expected - parsed.keys())
    if missing:
        raise HotspotBenchmarkError(
            "NVLink payload snapshot is missing expected counters: " + ",".join(missing)
        )
    # Ignore unrelated devices while preventing raw nvidia-smi text or device
    # identity from entering the model benchmark receipt.
    return {key: parsed[key] for key in sorted(expected)}


def build_nvlink_traffic_receipt(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, object]:
    expected = expected_nvlink_counter_keys()
    before_keys = frozenset(before)
    after_keys = frozenset(after)
    if before_keys != expected or after_keys != expected:
        missing_before = sorted(expected - before_keys)
        missing_after = sorted(expected - after_keys)
        unexpected_before = sorted(before_keys - expected)
        unexpected_after = sorted(after_keys - expected)
        raise HotspotBenchmarkError(
            "NVLink traffic attribution has the wrong counter set: "
            f"missing_before={missing_before}, missing_after={missing_after}, "
            f"unexpected_before={unexpected_before}, "
            f"unexpected_after={unexpected_after}"
        )
    counters_before = {key: int(before[key]) for key in sorted(expected)}
    counters_after = {key: int(after[key]) for key in sorted(expected)}
    try:
        deltas = interconnect_counter_deltas(counters_before, counters_after)
    except InterconnectQualificationError as error:
        raise HotspotBenchmarkError(
            f"NVLink traffic attribution is invalid: {error}"
        ) from error
    idle_counters = sorted(key for key, value in deltas.items() if value <= 0)
    if idle_counters:
        raise HotspotBenchmarkError(
            "real prompt requests did not move every expected NVLink payload "
            "counter: " + ",".join(idle_counters)
        )
    return {
        "counters_before": counters_before,
        "counters_after": counters_after,
        "counter_deltas": deltas,
    }


def parse_args() -> HotspotArguments:
    parser = argparse.ArgumentParser(prog="benchmark_dsv4_flash_hotspot")
    parser.add_argument("--url", default="http://127.0.0.1:30010/generate")
    parser.add_argument("--server-info-url")
    parser.add_argument("--flush-url")
    parser.add_argument("--control-url")
    parser.add_argument("--hotspot-url")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/tmp/dsv4-local-checkpoint-0731"),
    )
    parser.add_argument("--input-tokens", type=int, default=2_694)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--flush-timeout", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--near-prefix-ratio", type=float, default=0.70)
    parser.add_argument(
        "--verify-policy",
        choices=("production", "adaptive", "2", "3", "4", "5", "6"),
        default="production",
        help="current production scheduler, adaptive/SPS, or an exact verify length",
    )
    parser.add_argument(
        "--hotspot-plan",
        type=Path,
        help="server-local expert hotspot plan; always dry-run validated first",
    )
    parser.add_argument(
        "--hotspot-generation",
        type=int,
        help="monotonic generation required with --hotspot-plan",
    )
    parser.add_argument(
        "--hotspot-commit",
        action="store_true",
        help="commit the validated plan before phase one (default is dry-run only)",
    )
    parser.add_argument(
        "--expert-recorder-dir",
        type=Path,
        help=(
            "directory from SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR; capture "
            "two rank files around each phase"
        ),
    )
    parser.add_argument(
        "--expected-expert-plan",
        type=Path,
        help=(
            "absolute loaded hybrid plan to hash and bind against the "
            "loader-validated /server_info digest"
        ),
    )
    parser.add_argument(
        "--require-nvlink-traffic",
        action="store_true",
        help=(
            "require every Tx/Rx payload counter on GPUs 0-1 and NVLinks 0-3 "
            "to increase during the five real prompt requests"
        ),
    )
    parser.add_argument(
        "--http-only",
        action="store_true",
        help=(
            "measure validated HTTP decode/TTFT and NVLink traffic without "
            "requiring launch-time DSpark trace instrumentation"
        ),
    )
    parser.add_argument(
        "--require-oscar-split-history",
        action="store_true",
        help=(
            "require the exact SM86 Oscar split-history kernel, fixed workspace, "
            "and monolithic-prefill telemetry"
        ),
    )
    parser.add_argument(
        "--deterministic-repeat-count",
        type=int,
        default=0,
        help=(
            "after the five performance phases, run 2-5 cache-flushed, "
            "fixed-length exact-prompt repeats as diagnostic-only evidence"
        ),
    )
    parser.add_argument(
        "--deterministic-repeat-output-tokens",
        type=int,
        default=128,
        help="fixed diagnostic output length (2-256 tokens; default: 128)",
    )
    parser.add_argument("--output-file", type=Path)
    raw = parser.parse_args()
    generate_url = cast(str, raw.url)
    near_prefix_ratio = cast(float, raw.near_prefix_ratio)
    if not 0.20 <= near_prefix_ratio <= 0.90:
        parser.error("--near-prefix-ratio must be in [0.20, 0.90]")
    hotspot_plan = cast(Path | None, raw.hotspot_plan)
    hotspot_generation = cast(int | None, raw.hotspot_generation)
    if hotspot_plan is not None and hotspot_generation is None:
        parser.error("--hotspot-generation is required with --hotspot-plan")
    if hotspot_plan is None and (
        hotspot_generation is not None or cast(bool, raw.hotspot_commit)
    ):
        parser.error("--hotspot-generation/--hotspot-commit require --hotspot-plan")
    if hotspot_generation is not None and hotspot_generation < 0:
        parser.error("--hotspot-generation must be non-negative")
    deterministic_repeat_count = cast(int, raw.deterministic_repeat_count)
    if deterministic_repeat_count != 0 and not (
        MINIMUM_DETERMINISTIC_REPEAT_COUNT
        <= deterministic_repeat_count
        <= MAXIMUM_DETERMINISTIC_REPEAT_COUNT
    ):
        parser.error("--deterministic-repeat-count must be 0 or in [2, 5]")
    deterministic_repeat_output_tokens = cast(
        int, raw.deterministic_repeat_output_tokens
    )
    if not (
        2
        <= deterministic_repeat_output_tokens
        <= MAXIMUM_DETERMINISTIC_REPEAT_OUTPUT_TOKENS
    ):
        parser.error("--deterministic-repeat-output-tokens must be in [2, 256]")
    return HotspotArguments(
        generate_url=generate_url,
        server_info_url=cast(
            str,
            raw.server_info_url or _derive_endpoint(generate_url, "/get_server_info"),
        ),
        flush_url=cast(
            str,
            raw.flush_url
            or _derive_endpoint(generate_url, "/flush_cache", "timeout=30"),
        ),
        control_url=cast(
            str,
            raw.control_url or _derive_endpoint(generate_url, "/set_internal_state"),
        ),
        hotspot_url=cast(
            str,
            raw.hotspot_url or _derive_endpoint(generate_url, "/kt_expert_hotspot"),
        ),
        model_path=cast(Path, raw.model_path),
        input_tokens=cast(int, raw.input_tokens),
        output_tokens=cast(int, raw.output_tokens),
        timeout_seconds=cast(float, raw.timeout),
        flush_timeout_seconds=cast(float, raw.flush_timeout),
        progress_every=cast(int, raw.progress_every),
        ignore_eos=cast(bool, raw.ignore_eos),
        near_prefix_ratio=near_prefix_ratio,
        verify_policy=cast(str, raw.verify_policy),
        hotspot_plan=hotspot_plan,
        hotspot_generation=hotspot_generation,
        hotspot_commit=cast(bool, raw.hotspot_commit),
        expert_recorder_directory=cast(Path | None, raw.expert_recorder_dir),
        expected_expert_plan=cast(Path | None, raw.expected_expert_plan),
        require_nvlink_traffic=cast(bool, raw.require_nvlink_traffic),
        require_trace=not cast(bool, raw.http_only),
        output_file=cast(Path | None, raw.output_file),
        require_oscar_split_history=cast(bool, raw.require_oscar_split_history),
        deterministic_repeat_count=deterministic_repeat_count,
        deterministic_repeat_output_tokens=deterministic_repeat_output_tokens,
    )


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _common_suffix_length(left: Sequence[int], right: Sequence[int]) -> int:
    matched = 0
    for left_token, right_token in zip(reversed(left), reversed(right), strict=False):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _near_record_builder(mutation_record_index: int) -> Callable[[int], str]:
    def build(index: int) -> str:
        record = baseline.measurement_record(index)
        if index != mutation_record_index:
            return record
        return (
            record + " The controlled near-repeat variant labels this observation BETA."
        )

    return build


def build_prompt_pair(
    tokenizer: baseline.PromptTokenizer,
    encode_messages: baseline.MessageEncoder,
    target_tokens: int,
    *,
    desired_prefix_ratio: float = 0.70,
) -> PromptPair:
    """Create equal-length coherent prompts with one controlled record mutation."""
    exact_ids = baseline.build_exact_prompt_ids(
        tokenizer, encode_messages, target_tokens
    )
    # Coarse-to-fine probing avoids hundreds of full tokenizer passes while
    # still positioning the mutation well inside the retained prefix.
    coarse_candidates = tuple(range(0, 128, 4))
    observations: dict[int, tuple[list[int], int]] = {}

    def observe(index: int) -> None:
        if index in observations:
            return
        near_ids = baseline.build_exact_prompt_ids(
            tokenizer,
            encode_messages,
            target_tokens,
            record_builder=_near_record_builder(index),
        )
        common_prefix = _common_prefix_length(exact_ids, near_ids)
        if common_prefix < target_tokens:
            observations[index] = (near_ids, common_prefix)

    for index in coarse_candidates:
        observe(index)
    if not observations:
        raise HotspotBenchmarkError(
            "controlled record mutations were outside the retained prompt prefix"
        )
    coarse_best = min(
        observations,
        key=lambda index: abs(
            observations[index][1] / target_tokens - desired_prefix_ratio
        ),
    )
    for index in range(max(0, coarse_best - 3), coarse_best + 4):
        observe(index)
    best = min(
        observations,
        key=lambda index: abs(
            observations[index][1] / target_tokens - desired_prefix_ratio
        ),
    )
    near_ids, common_prefix = observations[best]
    common_suffix = _common_suffix_length(exact_ids, near_ids)
    if len(near_ids) != target_tokens or exact_ids == near_ids:
        raise HotspotBenchmarkError(
            "near-repeat prompt did not preserve shape/difference"
        )
    if common_prefix < int(target_tokens * 0.20):
        raise HotspotBenchmarkError(
            "near-repeat prompt shares too little prefix for a locality experiment"
        )
    return PromptPair(
        exact_ids=exact_ids,
        near_ids=near_ids,
        mutation_record_index=best,
        common_prefix_tokens=common_prefix,
        common_suffix_tokens=common_suffix,
    )


def _request_json(
    url: str,
    *,
    timeout_seconds: float,
    payload: Mapping[str, object] | None = None,
) -> tuple[int, bytes, object | None]:
    body = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with cast(
            ReadableResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
        ) as response:
            response_body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        response_body = error.read()
        status = error.code
    try:
        parsed = cast(object, json.loads(response_body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    return status, response_body, parsed


def get_server_info(url: str, timeout_seconds: float) -> dict[str, object]:
    status, _body, parsed = _request_json(url, timeout_seconds=timeout_seconds)
    if status != 200 or not isinstance(parsed, dict):
        raise HotspotBenchmarkError("get_server_info did not return a JSON object")
    return cast(dict[str, object], parsed)


def validate_oscar_split_history_workers(
    server_info: Mapping[str, object],
) -> dict[str, object]:
    """Prove that both EP2 model workers own a live, fixed Oscar workspace."""
    gathered_workers: list[dict[str, object]] = []
    for state in _internal_states(server_info):
        raw_workers = state.get("dsv4_oscar_worker_telemetry_workers")
        if isinstance(raw_workers, list):
            gathered_workers.extend(
                cast(dict[str, object], worker)
                for worker in cast(list[object], raw_workers)
                if isinstance(worker, dict)
            )

    workers_by_rank: dict[tuple[object, object, object], dict[str, object]] = {}
    for worker in gathered_workers:
        rank = (worker.get("dp_rank"), worker.get("pp_rank"), worker.get("tp_rank"))
        previous = workers_by_rank.get(rank)
        if previous is not None and previous != worker:
            raise HotspotBenchmarkError(
                "Oscar split-history worker telemetry has a conflicting rank"
            )
        workers_by_rank[rank] = worker

    workers = list(workers_by_rank.values())
    if len(workers) != 2:
        raise HotspotBenchmarkError(
            "Oscar split-history worker telemetry must prove exactly two workers"
        )

    identities: set[tuple[int, int, int]] = set()
    pids: set[int] = set()
    workspace_addresses: list[int] = []
    for worker in workers:
        pid = worker.get("pid")
        gpu_id = worker.get("gpu_id")
        tp_rank = worker.get("tp_rank")
        pp_rank = worker.get("pp_rank")
        if (
            type(pid) is not int
            or type(gpu_id) is not int
            or type(tp_rank) is not int
            or type(pp_rank) is not int
            or cast(int, pid) <= 0
            or cast(int, pp_rank) != 0
            or cast(int, gpu_id) != cast(int, tp_rank)
            or cast(int, tp_rank) not in (0, 1)
            or worker.get("dp_rank") not in (None, 0)
        ):
            raise HotspotBenchmarkError(
                "Oscar split-history worker telemetry violates the EP2 identity"
            )
        mismatches = {
            key: {"expected": expected, "observed": worker.get(key)}
            for key, expected in OSCAR_SPLIT_HISTORY_WORKER_FIELDS.items()
            if worker.get(key) != expected
        }
        if mismatches:
            raise HotspotBenchmarkError(
                "Oscar split-history worker telemetry contract mismatch: "
                + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
            )
        workspace_address = worker.get(
            "dsv4_oscar_int2_split_history_workspace_address"
        )
        if type(workspace_address) is not int or cast(int, workspace_address) <= 0:
            raise HotspotBenchmarkError(
                "Oscar split-history worker workspace address is invalid"
            )
        identities.add((cast(int, tp_rank), cast(int, pp_rank), cast(int, gpu_id)))
        pids.add(cast(int, pid))
        workspace_addresses.append(cast(int, workspace_address))

    if identities != {(0, 0, 0), (1, 0, 1)} or len(pids) != 2:
        raise HotspotBenchmarkError(
            "Oscar split-history telemetry did not prove two distinct EP2 workers"
        )
    return {
        "worker_count": 2,
        "worker_pids": sorted(pids),
        "tp_pp_gpu_ranks": [list(identity) for identity in sorted(identities)],
        "workspace_bytes_per_worker": OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES,
        "workspace_addresses": sorted(workspace_addresses),
        "fixed_address": True,
    }


def validate_server_contract(
    server_info: Mapping[str, object],
    *,
    require_oscar_split_history: bool = False,
) -> None:
    """Fail closed on the 524K, two-GPU, CPU-offload production constraints."""
    issues: list[str] = []

    def is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    context_length = server_info.get("context_length")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < 524_288
    ):
        issues.append("context_length_below_524288")
    max_total_tokens = server_info.get("max_total_tokens")
    if (
        not isinstance(max_total_tokens, int)
        or isinstance(max_total_tokens, bool)
        or max_total_tokens < 524_288
    ):
        issues.append("max_total_tokens_below_524288")
    kv_dtype = server_info.get("kv_cache_dtype")
    if not isinstance(kv_dtype, str) or "fp8" not in kv_dtype.casefold():
        issues.append("kv_cache_not_fp8")
    if server_info.get("dsv4_oscar_int2_kv_storage") is not True:
        issues.append("oscar_int2_kv_not_admitted")
    if server_info.get("dsv4_oscar_algorithm") != "oscar-int2-asym-g64-v1":
        issues.append("oscar_algorithm_mismatch")
    if not is_sha256(server_info.get("dsv4_oscar_artifact_sha256")):
        issues.append("oscar_artifact_sha256_missing")
    if not is_sha256(server_info.get("dsv4_oscar_model_config_sha256")):
        issues.append("oscar_model_config_sha256_missing")
    for proof_field in (
        "dsv4_oscar_artifact_provenance_sha256",
        "dsv4_oscar_checkpoint_sha256",
        "dsv4_oscar_checkpoint_fingerprint_sha256",
        "dsv4_oscar_admission_sha256",
        "dsv4_oscar_admission_receipt_sha256",
    ):
        if not is_sha256(server_info.get(proof_field)):
            issues.append(f"{proof_field}_missing")
    if server_info.get("dsv4_oscar_model_id") != "deepseek-ai/DeepSeek-V4-Flash":
        issues.append("oscar_model_id_mismatch")
    if server_info.get("dsv4_kv_storage_mode") != (
        "oscar_int2_asymmetric+protected_swa_bfloat16"
    ):
        issues.append("oscar_physical_layout_mismatch")
    if server_info.get("dsv4_swa_kv_bytes_per_token") != 1_024:
        issues.append("oscar_swa_row_not_1024_bytes")
    if server_info.get("dsv4_c4_kv_bytes_per_token") != 272:
        issues.append("oscar_c4_row_not_272_bytes")
    if server_info.get("dsv4_c128_kv_bytes_per_token") != 272:
        issues.append("oscar_c128_row_not_272_bytes")
    if server_info.get("dsv4_oscar_c4_scorer") is not True:
        issues.append("oscar_c4_scorer_not_admitted")
    if (
        server_info.get("dsv4_oscar_c4_scorer_algorithm")
        != "oscar-int2-c4-asym-c128-fp32-adjacent4-v1"
    ):
        issues.append("oscar_c4_scorer_algorithm_mismatch")
    if server_info.get("dsv4_c4_indexer_bytes_per_token") != 40:
        issues.append("oscar_c4_scorer_row_not_40_bytes")
    if server_info.get("dsv4_int4_kv_storage") is not False:
        issues.append("generic_int4_kv_enabled")
    if server_info.get("dsv4_int4_c4_indexer_storage") is not False:
        issues.append("generic_int4_c4_enabled")
    if server_info.get("dsv4_sm86_c128_bf16_storage") is not False:
        issues.append("selective_c128_bf16_enabled")
    if require_oscar_split_history:
        if server_info.get("dsv4_oscar_int2_split_history") is not True:
            issues.append("oscar_split_history_not_enabled")
        if server_info.get("dsv4_oscar_int2_split_history_execution") != (
            OSCAR_SPLIT_HISTORY_EXECUTION
        ):
            issues.append("oscar_split_history_execution_mismatch")
        if server_info.get("dsv4_oscar_int2_split_history_split_map") != (
            OSCAR_SPLIT_HISTORY_SPLIT_MAP
        ):
            issues.append("oscar_split_history_map_mismatch")
        if server_info.get("dsv4_oscar_int2_split_history_workspace_bytes") != (
            OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
        ):
            issues.append("oscar_split_history_workspace_size_mismatch")
        if server_info.get("dsv4_oscar_int2_split_history_max_partial_rows") != 32:
            issues.append("oscar_split_history_partial_row_limit_mismatch")
        if server_info.get("dsv4_oscar_int2_split_history_sink_owner") != (
            "stage2-exactly-once"
        ):
            issues.append("oscar_split_history_sink_owner_mismatch")
        if (
            server_info.get("dsv4_oscar_int2_split_history_prefill_enabled")
            is not False
        ):
            issues.append("oscar_split_history_prefill_not_monolithic")
        if server_info.get("dsv4_oscar_int2_split_history_fixed_address") is not True:
            issues.append("oscar_split_history_address_not_fixed")
        workspace_address = server_info.get(
            "dsv4_oscar_int2_split_history_workspace_address"
        )
        if (
            not isinstance(workspace_address, int)
            or isinstance(workspace_address, bool)
            or workspace_address <= 0
        ):
            issues.append("oscar_split_history_workspace_address_invalid")
        try:
            validate_oscar_split_history_workers(server_info)
        except HotspotBenchmarkError as error:
            issues.append(f"oscar_split_history_workers_invalid:{error}")
    if (
        server_info.get("disable_cuda_graph") is True
        or server_info.get("disable_decode_cuda_graph") is True
    ):
        issues.append("decode_cuda_graph_disabled")
    if server_info.get("disable_prefill_cuda_graph") is True:
        issues.append("prefill_cuda_graph_disabled")
    if server_info.get("cuda_graph_backend_decode") != "full":
        issues.append("decode_cuda_graph_backend_not_full")
    if server_info.get("cuda_graph_backend_prefill") != "breakable":
        issues.append("prefill_cuda_graph_backend_not_breakable")
    if server_info.get("enable_p2p_check") is not True:
        issues.append("cuda_p2p_check_not_enabled")
    if server_info.get("pre_warm_nccl") is not True:
        issues.append("nccl_not_prewarmed")
    tp_size = server_info.get("tp_size")
    ep_size = server_info.get("ep_size")
    if tp_size != 2:
        issues.append("tp_size_not_2")
    if server_info.get("pp_size") != 1:
        issues.append("pp_size_not_1")
    if ep_size != 2:
        issues.append("ep_size_not_2")
    cpu_offload = server_info.get("kt_cpuinfer")
    if (
        not isinstance(cpu_offload, (int, float))
        or isinstance(cpu_offload, bool)
        or cpu_offload <= 0
    ):
        issues.append("kt_cpu_offload_not_active")
    if server_info.get("speculative_dspark_block_size") != 5:
        issues.append("dspark_block_size_not_5")
    if server_info.get("speculative_num_draft_tokens") != 6:
        issues.append("dspark_verify_width_not_6")
    if issues:
        raise HotspotBenchmarkError("server contract mismatch: " + ",".join(issues))


def validate_verify_policy(
    server_info: Mapping[str, object],
    *,
    verify_policy: str,
    ragged_mode: str | None,
) -> None:
    if verify_policy == "adaptive" and ragged_mode != "compact":
        raise HotspotBenchmarkError(
            "adaptive policy requires SGLANG_RAGGED_VERIFY_MODE=compact"
        )
    if verify_policy == "adaptive" and not server_info.get(
        "speculative_dspark_sps_table_path"
    ):
        raise HotspotBenchmarkError(
            "adaptive policy requires --speculative-dspark-sps-table-path"
        )
    if (
        verify_policy == "adaptive"
        and server_info.get("speculative_dspark_fixed_verify_len") is not None
    ):
        raise HotspotBenchmarkError(
            "adaptive policy requires launching with an explicitly empty "
            "DSV4_DSPARK_FIXED_VERIFY_LEN"
        )
    if verify_policy in {"2", "3", "4", "5", "6"} and ragged_mode != "compact":
        raise HotspotBenchmarkError(
            "fixed verify tiers require SGLANG_RAGGED_VERIFY_MODE=compact"
        )


def validate_loaded_expert_plan(
    server_info: Mapping[str, object], expected_plan: Path
) -> dict[str, str]:
    if not expected_plan.is_absolute():
        raise HotspotBenchmarkError("expected expert plan path must be absolute")
    if expected_plan.is_symlink() or not expected_plan.is_file():
        raise HotspotBenchmarkError(
            "expected expert plan must be a regular, non-symlink file"
        )
    resolved_plan = expected_plan.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved_plan.open("rb") as plan_file:
        for chunk in iter(lambda: plan_file.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_sha256 = digest.hexdigest()
    if server_info.get("kt_hybrid_expert_plan_sha256") != expected_sha256:
        raise HotspotBenchmarkError(
            "loaded expert plan SHA-256 does not match the expected plan"
        )
    return {
        "expected_plan_path": str(resolved_plan),
        "expected_plan_sha256": expected_sha256,
        "binding": "launcher-hash-and-kt-loader-validated",
    }


def set_dspark_control(
    url: str,
    timeout_seconds: float,
    **controls: object,
) -> dict[str, object]:
    status, body, parsed = _request_json(
        url,
        timeout_seconds=timeout_seconds,
        payload={"server_args": controls},
    )
    if status != 200:
        raise HotspotBenchmarkError("DSpark control endpoint returned non-200")
    if isinstance(parsed, list):
        rank_results = cast(list[object], parsed)
        updated = bool(rank_results) and all(value is True for value in rank_results)
        result: dict[str, object] = {
            "updated": updated,
            "rank_results": rank_results,
        }
    elif isinstance(parsed, dict):
        result = cast(dict[str, object], parsed)
        updated = result.get("updated") is True
    else:
        raise HotspotBenchmarkError("DSpark control endpoint returned malformed data")
    if not updated:
        raise HotspotBenchmarkError(
            "DSpark control was rejected: " + hashlib.sha256(body).hexdigest()
        )
    return result


def request_hotspot_plan(
    url: str,
    timeout_seconds: float,
    *,
    plan_path: Path,
    generation: int,
    dry_run: bool,
) -> dict[str, object]:
    status, body, parsed = _request_json(
        url,
        timeout_seconds=timeout_seconds,
        payload={
            "plan_path": str(plan_path),
            "generation": generation,
            "dry_run": dry_run,
        },
    )
    if status != 200 or not isinstance(parsed, dict):
        raise HotspotBenchmarkError(
            "expert hotspot endpoint returned malformed data: "
            + hashlib.sha256(body).hexdigest()
        )
    result = cast(dict[str, object], parsed)
    raw_receipts = result.get("receipts")
    if (
        result.get("success") is not True
        or not isinstance(raw_receipts, list)
        or not raw_receipts
    ):
        raise HotspotBenchmarkError("expert hotspot plan was rejected")
    expected_path = str(plan_path.resolve())
    typed_receipts = cast(list[object], raw_receipts)
    for root_index, raw_receipt in enumerate(typed_receipts):
        if not isinstance(raw_receipt, dict):
            raise HotspotBenchmarkError(
                f"expert hotspot receipt {root_index} is not an object"
            )
        receipt = cast(dict[str, object], raw_receipt)
        if (
            receipt.get("generation") != generation
            or receipt.get("dry_run") is not dry_run
            or receipt.get("plan_path") != expected_path
        ):
            raise HotspotBenchmarkError(
                f"expert hotspot receipt {root_index} does not identify the request"
            )
        ep_size = receipt.get("ep_size")
        rank_receipts = receipt.get("rank_receipts")
        typed_rank_receipts = (
            cast(list[object], rank_receipts)
            if isinstance(rank_receipts, list)
            else None
        )
        if (
            not isinstance(ep_size, int)
            or isinstance(ep_size, bool)
            or ep_size < 2
            or typed_rank_receipts is None
            or len(typed_rank_receipts) != ep_size
        ):
            raise HotspotBenchmarkError(
                f"expert hotspot receipt {root_index} has incomplete EP coverage"
            )
        observed_ep_ranks: set[int] = set()
        for raw_rank_receipt in typed_rank_receipts:
            if not isinstance(raw_rank_receipt, dict):
                raise HotspotBenchmarkError(
                    f"expert hotspot receipt {root_index} has a malformed rank receipt"
                )
            rank_receipt = cast(dict[str, object], raw_rank_receipt)
            ep_rank = rank_receipt.get("ep_rank")
            if (
                not isinstance(ep_rank, int)
                or isinstance(ep_rank, bool)
                or rank_receipt.get("ep_size") != ep_size
                or rank_receipt.get("generation") != generation
                or rank_receipt.get("dry_run") is not dry_run
                or rank_receipt.get("plan_path") != expected_path
            ):
                raise HotspotBenchmarkError(
                    f"expert hotspot receipt {root_index} has rank disagreement"
                )
            observed_ep_ranks.add(ep_rank)
            if (
                not dry_run
                and rank_receipt.get("last_committed_generation") != generation
            ):
                raise HotspotBenchmarkError(
                    f"expert hotspot receipt {root_index} did not commit every rank"
                )
        if observed_ep_ranks != set(range(ep_size)):
            raise HotspotBenchmarkError(
                f"expert hotspot receipt {root_index} has duplicate or missing EP ranks"
            )
        if not dry_run and receipt.get("last_committed_generation") != generation:
            raise HotspotBenchmarkError(
                f"expert hotspot receipt {root_index} did not commit its root rank"
            )
    return result


def flush_cache(url: str, timeout_seconds: float) -> dict[str, object]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    retries = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise HotspotBenchmarkError("cache flush remained busy until timeout")
        try:
            status, body, parsed = _request_json(
                url, timeout_seconds=remaining, payload={}
            )
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            error.read()
            retries += 1
            time.sleep(min(FLUSH_RETRY_SECONDS, max(deadline - time.perf_counter(), 0)))
            continue
        structured = (
            isinstance(parsed, dict)
            and cast(dict[str, object], parsed).get("success") is True
        )
        if status == 200 and (body.startswith(b"Cache flushed.\n") or structured):
            return {
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "busy_retries": retries,
                "response_sha256": hashlib.sha256(body).hexdigest(),
            }
        if status != 400:
            raise HotspotBenchmarkError("cache flush did not confirm success")
        retries += 1
        time.sleep(min(FLUSH_RETRY_SECONDS, max(deadline - time.perf_counter(), 0)))


def _recorder_control(
    generate_url: str, timeout_seconds: float, operation: str
) -> dict[str, object]:
    endpoint = {
        "start": "/start_expert_distribution_record",
        "stop": "/stop_expert_distribution_record",
        "dump": "/dump_expert_distribution_record",
    }.get(operation)
    if endpoint is None:
        raise ValueError(f"unknown recorder operation {operation!r}")
    status, body, _parsed = _request_json(
        _derive_endpoint(generate_url, endpoint),
        timeout_seconds=timeout_seconds,
        payload={},
    )
    if status != 200:
        raise HotspotBenchmarkError(f"expert recorder {operation} failed")
    return {
        "operation": operation,
        "http_status": status,
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _recorder_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("expert_distribution_recorder_*.pt")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_recorder_receipts(
    directory: Path,
    before_names: set[str],
    timeout_seconds: float,
) -> list[dict[str, object]]:
    """Wait for the single TP/EP-reduced ``stat`` recorder artifact.

    SGLang's stat accumulator all-reduces logical counts across the
    non-pipeline process group and deliberately writes only from recorder rank
    zero. Detail modes write one file per rank, but this harness rejects those
    modes because their per-pass payload perturbs the timing experiment.
    """
    deadline = time.perf_counter() + timeout_seconds
    while True:
        new_paths = sorted(
            (
                path
                for path in directory.glob("expert_distribution_recorder_*.pt")
                if path.name not in before_names and path.is_file()
            ),
            key=lambda path: path.name,
        )
        if any(path.stat().st_size > 512 * 1024 * 1024 for path in new_paths):
            raise HotspotBenchmarkError("expert recorder file exceeds 512 MiB bound")
        if len(new_paths) == 1 and new_paths[0].stat().st_size > 0:
            return [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in new_paths
            ]
        if len(new_paths) > 1:
            raise HotspotBenchmarkError(
                "stat expert recorder produced more than one reduced file for one phase"
            )
        if time.perf_counter() >= deadline:
            raise HotspotBenchmarkError(
                f"expert recorder produced {len(new_paths)} files, expected one"
            )
        time.sleep(0.05)


def _internal_states(server_info: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw_states = server_info.get("internal_states")
    if not isinstance(raw_states, list):
        return [server_info]
    states: list[Mapping[str, object]] = []
    for raw_state in cast(list[object], raw_states):
        if isinstance(raw_state, dict):
            states.append(cast(dict[str, object], raw_state))
    return states


def _ragged_verify_mode(server_info: Mapping[str, object]) -> str | None:
    observed_mode = next(
        (
            str(cast(dict[str, object], state.get("dspark_info_record")).get("mode"))
            for state in _internal_states(server_info)
            if isinstance(state.get("dspark_info_record"), dict)
        ),
        None,
    )
    if observed_mode is not None:
        return observed_mode
    # Production-clean launches intentionally omit DSpark trace collection.
    # Startup fixed tiers are accepted by ServerArgs only in compact ragged
    # mode, so the published fixed-tier field is a fail-closed contract witness
    # when no diagnostic dump exists.
    fixed_verify_len = server_info.get("speculative_dspark_fixed_verify_len")
    if isinstance(fixed_verify_len, int) and not isinstance(fixed_verify_len, bool):
        return "compact"
    return None


def _trace_sources(
    server_info: Mapping[str, object], request_id: str
) -> tuple[list[dict[str, object]], int, list[int], set[str]]:
    candidates: list[tuple[int, list[dict[str, object]], set[str]]] = []
    source_counts: list[int] = []
    for source_index, state in enumerate(_internal_states(server_info)):
        raw_dump = state.get("dspark_info_record")
        if not isinstance(raw_dump, dict):
            source_counts.append(0)
            continue
        dump = cast(dict[str, object], raw_dump)
        raw_components = dump.get("components")
        components: set[str] = (
            {str(item) for item in cast(list[object], raw_components)}
            if isinstance(raw_components, list)
            else set()
        )
        raw_records = dump.get("records")
        matching: list[dict[str, object]] = []
        if isinstance(raw_records, list):
            for raw_record in cast(list[object], raw_records):
                if not isinstance(raw_record, dict):
                    continue
                record = cast(dict[str, object], raw_record)
                raw_reqs = record.get("reqs")
                if not isinstance(raw_reqs, list):
                    continue
                if any(
                    isinstance(raw_req, dict)
                    and cast(dict[str, object], raw_req).get("rid") == request_id
                    for raw_req in cast(list[object], raw_reqs)
                ):
                    matching.append(record)
        source_counts.append(len(matching))
        candidates.append((source_index, matching, components))
    if not candidates:
        return [], -1, source_counts, set()
    source_index, records, components = max(candidates, key=lambda item: len(item[1]))
    return records, source_index, source_counts, components


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[rank]


def _numeric_summary(values: Iterable[object]) -> dict[str, float | int | None]:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "count": len(numbers),
        "mean": round(statistics.fmean(numbers), 6) if numbers else None,
        "p50": round(float(statistics.median(numbers)), 6) if numbers else None,
        "p95": (round(cast(float, _percentile(numbers, 0.95)), 6) if numbers else None),
    }


def summarize_trace(
    records: Sequence[Mapping[str, object]], request_id: str
) -> dict[str, object]:
    req_details: list[dict[str, object]] = []
    for record in records:
        raw_reqs = record.get("reqs")
        if not isinstance(raw_reqs, list):
            continue
        for raw_req in cast(list[object], raw_reqs):
            if isinstance(raw_req, dict):
                req = cast(dict[str, object], raw_req)
                if req.get("rid") == request_id:
                    req_details.append(req)
    acceptance = [
        int(value)
        for req in req_details
        if isinstance((value := req.get("acc_len")), int)
        and not isinstance(value, bool)
    ]
    verify_lens = [
        int(value)
        for req in req_details
        if isinstance((value := req.get("verify_len")), int)
        and not isinstance(value, bool)
    ]
    acceptance_distribution = Counter(acceptance)
    verify_distribution = Counter(verify_lens)
    graph_key_distribution = Counter(
        int(value)
        for record in records
        if isinstance((value := record.get("verify_tokens_graph_key")), int)
        and not isinstance(value, bool)
    )

    step_match_count: Counter[int] = Counter()
    step_total_count: Counter[int] = Counter()
    margin_by_position: dict[int, list[float]] = {}
    for req in req_details:
        matches = req.get("greedy_step_matches")
        margins = req.get("greedy_target_logit_margins")
        if isinstance(matches, list):
            for position, matched in enumerate(cast(list[object], matches), start=1):
                if isinstance(matched, bool):
                    step_total_count[position] += 1
                    step_match_count[position] += int(matched)
        if isinstance(margins, list):
            for position, margin in enumerate(cast(list[object], margins), start=1):
                if isinstance(margin, (int, float)) and not isinstance(margin, bool):
                    margin_by_position.setdefault(position, []).append(float(margin))

    stepwise: list[dict[str, object]] = []
    positions = sorted(set(step_total_count) | set(margin_by_position))
    for position in positions:
        total = step_total_count[position]
        margins = margin_by_position.get(position, [])
        stepwise.append(
            {
                "draft_position": position,
                "match_count": step_match_count[position],
                "comparison_count": total,
                "match_rate": round(step_match_count[position] / total, 6)
                if total
                else None,
                "target_top1_minus_draft_logit": _numeric_summary(margins),
            }
        )

    counterfactual: dict[str, dict[str, float | int]] = {}
    full_block_observations = [
        acc
        for acc, verify_len in zip(acceptance, verify_lens, strict=False)
        if verify_len == 6
    ]
    if full_block_observations:
        for tier in range(2, 7):
            commits = [min(acc, tier) for acc in full_block_observations]
            counterfactual[str(tier)] = {
                "mean_committed_tokens_per_cycle": round(statistics.fmean(commits), 6),
                "committed_tokens": sum(commits),
            }

    return {
        "cycle_count": len(records),
        "request_observation_count": len(req_details),
        "committed_tokens": sum(acceptance),
        "mean_committed_tokens_per_cycle": round(statistics.fmean(acceptance), 6)
        if acceptance
        else None,
        "acceptance_distribution": {
            str(key): acceptance_distribution[key]
            for key in sorted(acceptance_distribution)
        },
        "verify_len_distribution": {
            str(key): verify_distribution[key] for key in sorted(verify_distribution)
        },
        "verify_graph_key_distribution": {
            str(key): graph_key_distribution[key]
            for key in sorted(graph_key_distribution)
        },
        "step_cpu_ms": _numeric_summary(
            record.get("step_cpu_ms") for record in records
        ),
        "step_gpu_ms": _numeric_summary(
            record.get("step_gpu_ms") for record in records
        ),
        "draft_gpu_ms": _numeric_summary(
            record.get("draft_gpu_ms") for record in records
        ),
        "target_verify_gpu_ms": _numeric_summary(
            record.get("target_verify_gpu_ms") for record in records
        ),
        "stepwise_greedy_comparison": stepwise,
        "full_block_counterfactual_tiers": counterfactual,
    }


def _mean_from_phase(
    phases: Mapping[str, Mapping[str, object]], phase: str, metric: str
) -> float | None:
    trace = phases[phase].get("trace")
    if not isinstance(trace, dict):
        return None
    field = cast(dict[str, object], trace).get(metric)
    if not isinstance(field, dict):
        return None
    mean = cast(dict[str, object], field).get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def _metric_from_phase(
    phases: Mapping[str, Mapping[str, object]], phase: str, metric: str
) -> float | None:
    benchmark = phases[phase].get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    value = cast(dict[str, object], benchmark).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def validate_cache_contract(
    phases: Mapping[str, Mapping[str, object]], pair: PromptPair
) -> None:
    cached: dict[str, int] = {}
    for name in (phase.name for phase in PHASES):
        value = _metric_from_phase(phases, name, "server_cached_tokens")
        if value is None or not value.is_integer():
            raise HotspotBenchmarkError(
                f"phase {name} did not report an integer cached-token count"
            )
        cached[name] = int(value)
    for name in ("cold_first_exact", "warm_no_radix_near", "warm_no_radix_exact"):
        if cached[name] != 0:
            raise HotspotBenchmarkError(
                f"cache-flushed phase {name} unexpectedly reused prompt KV"
            )
    if not 0 < cached["radix_hot_exact"] <= len(pair.exact_ids):
        raise HotspotBenchmarkError(
            "exact-repeat phase did not demonstrate radix reuse"
        )
    # The DSV4 SWA/chunk cache may retain exact paths while declining a
    # partially matching branch.  Zero is therefore a meaningful measured
    # result for the near-repeat experiment, not a malformed server receipt.
    # Any reported reuse must still be bounded by the independently computed
    # common prefix.
    if not 0 <= cached["radix_hot_near"] <= pair.common_prefix_tokens:
        raise HotspotBenchmarkError(
            "near-repeat cached-token count exceeds its controlled common prefix"
        )


def summarize_repeat_trajectories(
    phases: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Report whether greedy repeats followed the same token trajectory.

    Natural-stop DeepSeek responses can remain semantically valid while taking
    different reasoning paths across requests. Such drift changes expert
    routes and therefore makes a strict paired timing attribution ineligible,
    but it is not itself a coherency failure. Missing hashes/counts remain a
    hard receipt error; the caller records drift explicitly.
    """

    repeat_groups = {
        "exact": (
            "cold_first_exact",
            "radix_hot_exact",
            "warm_no_radix_exact",
        ),
        "near": ("radix_hot_near", "warm_no_radix_near"),
    }
    summaries: dict[str, dict[str, object]] = {}
    for prompt_kind, names in repeat_groups.items():
        hashes: list[str] = []
        token_counts: list[int] = []
        for name in names:
            benchmark = phases[name].get("benchmark")
            if not isinstance(benchmark, dict):
                raise HotspotBenchmarkError(f"phase {name} has no benchmark receipt")
            output_hash = cast(dict[str, object], benchmark).get("output_sha256")
            completion_tokens = cast(dict[str, object], benchmark).get(
                "completion_tokens"
            )
            if not isinstance(output_hash, str) or not output_hash:
                raise HotspotBenchmarkError(
                    f"phase {name} has no generated-output hash"
                )
            if not isinstance(completion_tokens, int) or isinstance(
                completion_tokens, bool
            ):
                raise HotspotBenchmarkError(
                    f"phase {name} has no completion-token count"
                )
            hashes.append(output_hash)
            token_counts.append(completion_tokens)
        hashes_equal = len(set(hashes)) == 1
        token_counts_equal = len(set(token_counts)) == 1
        summaries[prompt_kind] = {
            "phase_names": list(names),
            "output_sha256": hashes,
            "completion_tokens": token_counts,
            "hashes_equal": hashes_equal,
            "token_counts_equal": token_counts_equal,
            "paired_trajectory_comparable": hashes_equal and token_counts_equal,
        }
    return {
        "all_paired_trajectories_comparable": all(
            summary.get("paired_trajectory_comparable") is True
            for summary in summaries.values()
        ),
        "groups": summaries,
    }


def _delta(left: float | None, right: float | None) -> float | None:
    return round(left - right, 6) if left is not None and right is not None else None


def build_attribution(
    phases: Mapping[str, Mapping[str, object]], server_info: Mapping[str, object]
) -> dict[str, object]:
    disable_all_graphs = server_info.get("disable_cuda_graph") is True
    disable_decode_graphs = server_info.get("disable_decode_cuda_graph") is True
    disable_prefill_graphs = server_info.get("disable_prefill_cuda_graph") is True
    decode_graphs_enabled = not disable_all_graphs and not disable_decode_graphs
    prefill_graphs_enabled = not disable_all_graphs and not disable_prefill_graphs

    cold_ttft = _metric_from_phase(
        phases, "cold_first_exact", "time_to_first_token_seconds"
    )
    warm_exact_ttft = _metric_from_phase(
        phases, "warm_no_radix_exact", "time_to_first_token_seconds"
    )
    hot_exact_ttft = _metric_from_phase(
        phases, "radix_hot_exact", "time_to_first_token_seconds"
    )
    hot_near_ttft = _metric_from_phase(
        phases, "radix_hot_near", "time_to_first_token_seconds"
    )
    warm_near_ttft = _metric_from_phase(
        phases, "warm_no_radix_near", "time_to_first_token_seconds"
    )
    cold_target = _mean_from_phase(phases, "cold_first_exact", "target_verify_gpu_ms")
    warm_target = _mean_from_phase(
        phases, "warm_no_radix_exact", "target_verify_gpu_ms"
    )
    cold_exact_benchmark = cast(
        Mapping[str, object], phases["cold_first_exact"]["benchmark"]
    )
    warm_exact_benchmark = cast(
        Mapping[str, object], phases["warm_no_radix_exact"]["benchmark"]
    )
    exact_trajectory_comparable = bool(
        isinstance(cold_exact_benchmark.get("output_sha256"), str)
        and cold_exact_benchmark.get("output_sha256")
        == warm_exact_benchmark.get("output_sha256")
        and isinstance(cold_exact_benchmark.get("completion_tokens"), int)
        and not isinstance(cold_exact_benchmark.get("completion_tokens"), bool)
        and cold_exact_benchmark.get("completion_tokens")
        == warm_exact_benchmark.get("completion_tokens")
    )
    locality_ineligibility_reasons: list[str] = []
    if not decode_graphs_enabled:
        locality_ineligibility_reasons.append("decode_graphs_not_enabled")
    if not exact_trajectory_comparable:
        locality_ineligibility_reasons.append("exact_output_trajectory_drifted")
    if cold_target is None or warm_target is None:
        locality_ineligibility_reasons.append("target_verify_trace_missing")
    cached_tokens = {
        name: _metric_from_phase(phases, name, "server_cached_tokens")
        for name in phases
    }
    return {
        "radix_kv_reuse": {
            "exact_ttft_seconds_saved_vs_warm_flushed_control": _delta(
                warm_exact_ttft, hot_exact_ttft
            ),
            "near_ttft_seconds_saved_vs_warm_flushed_control": _delta(
                warm_near_ttft, hot_near_ttft
            ),
            "basis": "paired prompt after no flush versus same prompt after flush",
            "server_cached_tokens_by_phase": cached_tokens,
            "near_prefix_reuse_observed": bool(
                (cached_tokens.get("radix_hot_near") or 0) > 0
            ),
        },
        "cpu_weight_page_cache_locality": {
            "cold_minus_warm_target_verify_mean_ms": _delta(cold_target, warm_target),
            "eligible": not locality_ineligibility_reasons,
            "ineligibility_reasons": locality_ineligibility_reasons,
            "exact_output_trajectory_comparable": exact_trajectory_comparable,
            "basis": (
                "same exact prompt, radix flushed in both phases; decode/spec graph "
                "state is captured at launch"
            ),
        },
        "compilation": {
            "decode_and_speculative_graphs_enabled": decode_graphs_enabled,
            "prefill_graphs_enabled": prefill_graphs_enabled,
            "cold_minus_warm_flushed_exact_ttft_seconds": _delta(
                cold_ttft, warm_exact_ttft
            ),
            "ttft_interpretation": (
                "prefill compilation is controlled by captured graphs"
                if prefill_graphs_enabled
                else (
                    "eager-prefill first-use compilation and CPU page faults "
                    "remain confounded"
                )
            ),
        },
    }


def _prompt_receipt(pair: PromptPair) -> dict[str, object]:
    return {
        "input_tokens": len(pair.exact_ids),
        "exact_sha256": baseline.hash_token_ids(pair.exact_ids),
        "near_sha256": baseline.hash_token_ids(pair.near_ids),
        "mutation_record_index": pair.mutation_record_index,
        "common_prefix_tokens": pair.common_prefix_tokens,
        "common_prefix_ratio": round(
            pair.common_prefix_tokens / len(pair.exact_ids), 6
        ),
        "common_suffix_tokens": pair.common_suffix_tokens,
    }


def run_deterministic_repeat_diagnostic(
    args: HotspotArguments,
    *,
    input_ids: list[int],
    decode_output_ids: Callable[[list[int]], str],
    terminal_token_ids: Sequence[int],
) -> dict[str, object]:
    """Run bounded fixed-work repeats without making a performance claim."""

    count = args.deterministic_repeat_count
    output_tokens = args.deterministic_repeat_output_tokens
    if count == 0:
        return {
            "enabled": False,
            "diagnostic_only": True,
            "performance_claim_eligible": False,
            "coherency_claim_eligible": False,
        }
    if (
        not MINIMUM_DETERMINISTIC_REPEAT_COUNT
        <= count
        <= MAXIMUM_DETERMINISTIC_REPEAT_COUNT
    ):
        raise ValueError("deterministic repeat count must be in [2, 5]")
    if not 2 <= output_tokens <= MAXIMUM_DETERMINISTIC_REPEAT_OUTPUT_TOKENS:
        raise ValueError("deterministic repeat output tokens must be in [2, 256]")

    run_id = uuid.uuid4().hex[:12]
    runs: list[dict[str, object]] = []
    output_hashes: list[str] = []
    output_token_hashes: list[str | None] = []
    completion_counts: list[int] = []
    for repeat_index in range(count):
        flush_receipt = flush_cache(args.flush_url, args.flush_timeout_seconds)
        benchmark_receipt = baseline.run_benchmark(
            args.generate_url,
            input_ids,
            output_tokens,
            args.timeout_seconds,
            args.progress_every,
            ignore_eos=True,
            decode_output_ids=decode_output_ids,
            terminal_token_ids=terminal_token_ids,
            request_id=f"dsv4-fixed-repeat-{run_id}-{repeat_index}",
            diagnostic_only=True,
        )
        output_hash = benchmark_receipt.get("output_sha256")
        token_hash = benchmark_receipt.get("output_token_ids_sha256")
        completion_count = benchmark_receipt.get("completion_tokens")
        if (
            benchmark_receipt.get("diagnostic_only") is not True
            or benchmark_receipt.get("performance_claim_eligible") is not False
            or benchmark_receipt.get("ignore_eos") is not True
            or benchmark_receipt.get("exact_requested_token_shape") is not True
            or not isinstance(output_hash, str)
            or len(output_hash) != 64
            or any(character not in "0123456789abcdef" for character in output_hash)
            or (
                token_hash is not None
                and (
                    not isinstance(token_hash, str)
                    or len(token_hash) != 64
                    or any(
                        character not in "0123456789abcdef" for character in token_hash
                    )
                )
            )
            or not isinstance(completion_count, int)
            or isinstance(completion_count, bool)
            or completion_count != output_tokens
        ):
            raise HotspotBenchmarkError(
                "fixed-length repeat did not return a diagnostic-only exact-shape receipt"
            )
        output_hashes.append(output_hash)
        output_token_hashes.append(token_hash)
        completion_counts.append(completion_count)
        runs.append(
            {
                "repeat_index": repeat_index,
                "flush": flush_receipt,
                "benchmark": benchmark_receipt,
            }
        )

    all_token_hashes_available = all(
        token_hash is not None for token_hash in output_token_hashes
    )
    comparison_basis = (
        "output_token_ids_sha256"
        if all_token_hashes_available
        else "decoded_output_sha256_and_completion_tokens"
    )
    comparison_values = (
        cast(list[str], output_token_hashes)
        if all_token_hashes_available
        else [
            f"{output_hash}:{completion_count}"
            for output_hash, completion_count in zip(
                output_hashes, completion_counts, strict=True
            )
        ]
    )
    return {
        "enabled": True,
        "diagnostic_only": True,
        "performance_claim_eligible": False,
        "coherency_claim_eligible": False,
        "cache_flush_before_every_request": True,
        "sampling": {"temperature": 0.0, "sampling_seed": 0},
        "requested_repetitions": count,
        "requested_output_tokens": output_tokens,
        "comparison_basis": comparison_basis,
        "all_output_token_hashes_available": all_token_hashes_available,
        "output_token_ids_sha256": output_token_hashes,
        "decoded_output_sha256": output_hashes,
        "completion_tokens": completion_counts,
        "fixed_work_shape_observed": all(
            completion_count == output_tokens for completion_count in completion_counts
        ),
        "repeat_trajectory_equal": len(set(comparison_values)) == 1,
        "runs": runs,
    }


def run_hotspot(
    args: HotspotArguments,
    *,
    nvlink_snapshotter: Callable[[], Mapping[str, int]] | None = None,
) -> dict[str, object]:
    tokenizer = baseline.load_tokenizer(args.model_path)
    encode_messages = baseline.load_message_encoder(args.model_path)
    pair = build_prompt_pair(
        tokenizer,
        encode_messages,
        args.input_tokens,
        desired_prefix_ratio=args.near_prefix_ratio,
    )

    def decode_output_ids(token_ids: list[int]) -> str:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    initial_info = get_server_info(args.server_info_url, args.timeout_seconds)
    validate_server_contract(
        initial_info,
        require_oscar_split_history=args.require_oscar_split_history,
    )
    expert_plan_provenance = (
        validate_loaded_expert_plan(initial_info, args.expected_expert_plan)
        if args.expected_expert_plan is not None
        else None
    )
    if (
        args.expert_recorder_directory is not None
        and initial_info.get("expert_distribution_recorder_mode") != "stat"
    ):
        raise HotspotBenchmarkError(
            "--expert-recorder-dir requires expert_distribution_recorder_mode=stat"
        )
    ragged_mode = _ragged_verify_mode(initial_info)
    validate_verify_policy(
        initial_info,
        verify_policy=args.verify_policy,
        ragged_mode=ragged_mode,
    )
    hotspot_receipt: dict[str, object] = {
        "requested": args.hotspot_plan is not None,
        "committed": False,
    }
    if args.hotspot_plan is not None:
        if args.hotspot_generation is None:
            raise AssertionError("hotspot generation validation drifted")
        hotspot_receipt["dry_run"] = request_hotspot_plan(
            args.hotspot_url,
            args.timeout_seconds,
            plan_path=args.hotspot_plan,
            generation=args.hotspot_generation,
            dry_run=True,
        )
        if args.hotspot_commit:
            hotspot_receipt["commit"] = request_hotspot_plan(
                args.hotspot_url,
                args.timeout_seconds,
                plan_path=args.hotspot_plan,
                generation=args.hotspot_generation,
                dry_run=False,
            )
            hotspot_receipt["committed"] = True
    requested_verify_len = (
        None
        if args.verify_policy in {"production", "adaptive"}
        else int(args.verify_policy)
    )
    phases: dict[str, dict[str, object]] = {}
    run_id = uuid.uuid4().hex[:12]
    nvlink_traffic: dict[str, object] | None = None
    deterministic_repeat_diagnostic: dict[str, object] = {
        "enabled": False,
        "diagnostic_only": True,
        "performance_claim_eligible": False,
        "coherency_claim_eligible": False,
    }
    counter_snapshot = (
        snapshot_nvlink_counters if nvlink_snapshotter is None else nvlink_snapshotter
    )
    nvlink_before: dict[str, int] | None = None
    control_attempted = False
    failure_pending = False
    try:
        # Mark the attempt before fan-out: a partial all-rank response can mean
        # one scheduler accepted the override before another rejected it.
        control_attempted = True
        set_dspark_control(
            args.control_url,
            args.timeout_seconds,
            dspark_force_verify_len=requested_verify_len,
        )
        if args.require_nvlink_traffic:
            nvlink_before = dict(counter_snapshot())
        for phase in PHASES:
            flush_receipt = (
                flush_cache(args.flush_url, args.flush_timeout_seconds)
                if phase.flush_before
                else None
            )
            set_dspark_control(
                args.control_url,
                args.timeout_seconds,
                dspark_clear_info_records=True,
            )
            request_id = f"dsv4-hotspot-{run_id}-{phase.name}"
            input_ids = (
                pair.exact_ids if phase.prompt_kind == "exact" else pair.near_ids
            )
            recorder_operations: list[dict[str, object]] = []
            recorder_files: list[dict[str, object]] = []
            recorder_before: set[str] = (
                _recorder_names(args.expert_recorder_directory)
                if args.expert_recorder_directory is not None
                else set()
            )
            recorder_started = False
            try:
                if args.expert_recorder_directory is not None:
                    recorder_operations.append(
                        _recorder_control(
                            args.generate_url, args.timeout_seconds, "start"
                        )
                    )
                    recorder_started = True
                benchmark_receipt = baseline.run_benchmark(
                    args.generate_url,
                    input_ids,
                    args.output_tokens,
                    args.timeout_seconds,
                    args.progress_every,
                    ignore_eos=args.ignore_eos,
                    decode_output_ids=decode_output_ids,
                    terminal_token_ids=baseline.terminal_token_ids(tokenizer),
                    request_id=request_id,
                )
            finally:
                if recorder_started:
                    recorder_operations.append(
                        _recorder_control(
                            args.generate_url, args.timeout_seconds, "stop"
                        )
                    )
                    recorder_operations.append(
                        _recorder_control(
                            args.generate_url, args.timeout_seconds, "dump"
                        )
                    )
            if args.expert_recorder_directory is not None:
                recorder_files = wait_for_recorder_receipts(
                    args.expert_recorder_directory,
                    recorder_before,
                    args.timeout_seconds,
                )
            source_index: int
            source_counts: list[int]
            components: set[str]
            trace_summary: dict[str, object]
            if args.require_trace:
                phase_info = get_server_info(args.server_info_url, args.timeout_seconds)
                records, source_index, source_counts, components = _trace_sources(
                    phase_info, request_id
                )
                missing_components = sorted(REQUIRED_TRACE_COMPONENTS - components)
                if missing_components:
                    raise HotspotBenchmarkError(
                        "DSpark trace is missing required components: "
                        + ",".join(missing_components)
                    )
                if not records:
                    raise HotspotBenchmarkError(
                        f"DSpark trace has no decode records for phase {phase.name}"
                    )
                trace_summary = summarize_trace(records, request_id)
                if requested_verify_len is not None:
                    distribution = trace_summary["verify_len_distribution"]
                    typed_distribution = (
                        cast(dict[object, object], distribution)
                        if isinstance(distribution, dict)
                        else None
                    )
                    if typed_distribution is None or set(typed_distribution) != {
                        str(requested_verify_len)
                    }:
                        raise HotspotBenchmarkError(
                            "fixed verify tier was not honored in every observed cycle"
                        )
                    graph_distribution = trace_summary["verify_graph_key_distribution"]
                    typed_graph_distribution = (
                        cast(dict[object, object], graph_distribution)
                        if isinstance(graph_distribution, dict)
                        else None
                    )
                    if typed_graph_distribution is None or set(
                        typed_graph_distribution
                    ) != {str(requested_verify_len)}:
                        raise HotspotBenchmarkError(
                            "fixed verify tier was padded to another graph key; "
                            "launch with SGLANG_DSV4_FINE_RAGGED_VERIFY_TIERS=1"
                        )
            else:
                source_index = -1
                source_counts = []
                components = set()
                trace_summary = {}
            phases[phase.name] = {
                "prompt_kind": phase.prompt_kind,
                "radix_flush_before": phase.flush_before,
                "flush": flush_receipt,
                "benchmark": benchmark_receipt,
                "trace": trace_summary,
                "trace_source_index": source_index,
                "trace_records_by_source": source_counts,
                "trace_components": sorted(components),
                "expert_recorder": {
                    "enabled": args.expert_recorder_directory is not None,
                    "control_operations": recorder_operations,
                    "files": recorder_files,
                },
            }
        if args.require_nvlink_traffic:
            if nvlink_before is None:
                raise AssertionError("NVLink before-snapshot validation drifted")
            nvlink_traffic = build_nvlink_traffic_receipt(
                nvlink_before,
                dict(counter_snapshot()),
            )
        # Keep the optional repeat probe outside the five timed phases and after
        # the NVLink performance snapshot.  It deliberately flushes before every
        # request and cannot make either a throughput or coherency claim.
        deterministic_repeat_diagnostic = run_deterministic_repeat_diagnostic(
            args,
            input_ids=pair.exact_ids,
            decode_output_ids=decode_output_ids,
            terminal_token_ids=baseline.terminal_token_ids(tokenizer),
        )
    except BaseException:
        failure_pending = True
        raise
    finally:
        # Never leave a diagnostic tier pinned after either success or failure.
        if control_attempted:
            try:
                set_dspark_control(
                    args.control_url,
                    args.timeout_seconds,
                    dspark_force_verify_len=None,
                )
            except Exception:
                if not failure_pending:
                    raise

    validate_cache_contract(phases, pair)
    repeat_trajectories = summarize_repeat_trajectories(phases)

    verify_logits_diagnostic = any(
        "verify_logits" in cast(list[object], phase["trace_components"])
        for phase in phases.values()
    )
    receipt: dict[str, object] = {
        "receipt_version": RECEIPT_VERSION,
        "accepted": True,
        "performance_claim_eligible": (
            args.require_trace
            and args.require_nvlink_traffic
            and expert_plan_provenance is not None
            and not verify_logits_diagnostic
            and args.deterministic_repeat_count == 0
            and args.expert_recorder_directory is None
            and repeat_trajectories["all_paired_trajectories_comparable"] is True
        ),
        "measurement_mode": "trace" if args.require_trace else "http_only",
        "verify_policy": args.verify_policy,
        "verify_logits_diagnostic": verify_logits_diagnostic,
        "deterministic_repeat_diagnostic": deterministic_repeat_diagnostic,
        "expert_hotspot": hotspot_receipt,
        "expert_plan_provenance": expert_plan_provenance,
        "prompt_pair": _prompt_receipt(pair),
        "server_contract": {
            "context_length": initial_info.get("context_length"),
            "max_total_tokens": initial_info.get("max_total_tokens"),
            "kv_cache_dtype": initial_info.get("kv_cache_dtype"),
            "tp_size": initial_info.get("tp_size"),
            "pp_size": initial_info.get("pp_size"),
            "ep_size": initial_info.get("ep_size"),
            "kt_cpuinfer": initial_info.get("kt_cpuinfer"),
            "speculative_algorithm": initial_info.get("speculative_algorithm"),
            "speculative_dspark_block_size": initial_info.get(
                "speculative_dspark_block_size"
            ),
            "speculative_num_draft_tokens": initial_info.get(
                "speculative_num_draft_tokens"
            ),
            "speculative_dspark_fixed_verify_len": initial_info.get(
                "speculative_dspark_fixed_verify_len"
            ),
            "disable_cuda_graph": initial_info.get("disable_cuda_graph"),
            "disable_decode_cuda_graph": initial_info.get("disable_decode_cuda_graph"),
            "disable_prefill_cuda_graph": initial_info.get(
                "disable_prefill_cuda_graph"
            ),
            "cuda_graph_backend_decode": initial_info.get("cuda_graph_backend_decode"),
            "cuda_graph_backend_prefill": initial_info.get(
                "cuda_graph_backend_prefill"
            ),
            "enable_p2p_check": initial_info.get("enable_p2p_check"),
            "pre_warm_nccl": initial_info.get("pre_warm_nccl"),
            "ragged_verify_mode": ragged_mode,
            "dsv4_oscar_int2_split_history": initial_info.get(
                "dsv4_oscar_int2_split_history"
            ),
            "dsv4_oscar_int2_split_history_execution": initial_info.get(
                "dsv4_oscar_int2_split_history_execution"
            ),
            "dsv4_oscar_int2_split_history_split_map": initial_info.get(
                "dsv4_oscar_int2_split_history_split_map"
            ),
            "dsv4_oscar_int2_split_history_workspace_bytes": initial_info.get(
                "dsv4_oscar_int2_split_history_workspace_bytes"
            ),
            "dsv4_oscar_int2_split_history_max_partial_rows": initial_info.get(
                "dsv4_oscar_int2_split_history_max_partial_rows"
            ),
            "dsv4_oscar_int2_split_history_sink_owner": initial_info.get(
                "dsv4_oscar_int2_split_history_sink_owner"
            ),
            "dsv4_oscar_int2_split_history_prefill_enabled": initial_info.get(
                "dsv4_oscar_int2_split_history_prefill_enabled"
            ),
            "dsv4_oscar_int2_split_history_fixed_address": initial_info.get(
                "dsv4_oscar_int2_split_history_fixed_address"
            ),
            "dsv4_oscar_int2_split_history_workers": (
                validate_oscar_split_history_workers(initial_info)
                if args.require_oscar_split_history
                else None
            ),
        },
        "phases": phases,
        "repeat_trajectories": repeat_trajectories,
        "attribution": build_attribution(phases, initial_info),
        "receipt_limits": [
            "OS page cache is observed, never destructively dropped",
            (
                "with prefill graphs disabled, first-use prefill compilation and "
                "TTFT page faults cannot be numerically separated in-process"
            ),
            (
                "verify_logits is diagnostic-only and adds an argmax/gather "
                "outside graph replay"
            ),
            (
                "fixed-length deterministic repeats are cache-flushed diagnostics; "
                "their timings and semantic assessments are not performance or "
                "coherency claims"
            ),
            *(
                []
                if args.require_trace
                else [
                    (
                        "HTTP-only mode does not prove the per-cycle verify graph "
                        "key; the startup server contract and control-plane "
                        "acceptance are recorded instead"
                    )
                ]
            ),
        ],
    }
    if nvlink_traffic is not None:
        receipt["nvlink_traffic"] = nvlink_traffic
    return receipt


def _emit(payload: Mapping[str, object], output_file: Path | None) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if output_file is not None:
        output_file.write_text(serialized, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        receipt = run_hotspot(args)
    except (
        HotspotBenchmarkError,
        baseline.BenchmarkValidationError,
        urllib.error.URLError,
        TimeoutError,
    ) as error:
        issue_codes = (
            list(error.issue_codes)
            if isinstance(error, baseline.BenchmarkValidationError)
            else [str(error)]
        )
        _emit(
            {
                "receipt_version": RECEIPT_VERSION,
                "accepted": False,
                "performance_claim_eligible": False,
                "issue_codes": issue_codes,
            },
            args.output_file,
        )
        return 2
    _emit(receipt, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
