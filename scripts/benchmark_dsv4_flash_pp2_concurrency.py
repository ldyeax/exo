#!/usr/bin/env python3
"""Measure semantically valid two-lane DSV4 PP concurrency.

The benchmark performs two cache-cold synchronized groups:

* two native ``/generate`` streams with distinct, varied 2,694-token code-agent
  prompts, natural EOS, and lane-specific semantic response contracts; and
* two OpenAI-compatible forced tool-call streams with distinct repository tasks
  and exact side-effect-free argument schemas.

Timing is performance-claim eligible only when every response is coherent,
non-degenerate, naturally terminated, structurally complete, and genuinely
overlapped with its peer. The live service must also prove calibrated Oscar INT2
physical history on both PP workers; ``fp8_e4m3`` is accepted only as the public
raw-byte carrier, never as generic FP8 storage. Only counts, timing, hashes, and
validation metrics are emitted. Model output text and tool arguments are never
included in the report or an error message.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast
from urllib.parse import urlsplit

try:
    from scripts.benchmark_dsv4_flash_hotspot import (
        HotspotBenchmarkError as NvlinkTrafficError,
    )
    from scripts.benchmark_dsv4_flash_hotspot import (
        build_nvlink_traffic_receipt,
        snapshot_nvlink_counters,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from benchmark_dsv4_flash_hotspot import (
        HotspotBenchmarkError as NvlinkTrafficError,
    )
    from benchmark_dsv4_flash_hotspot import (
        build_nvlink_traffic_receipt,
        snapshot_nvlink_counters,
    )

LANE_COUNT = 2
INPUT_TOKEN_COUNT = 2_694
OUTPUT_TOKEN_COUNT = 512
DEFAULT_CHAT_MAX_TOKENS = 128
DEFAULT_MODEL_PATH = Path("/tmp/dsv4-local-checkpoint-0731")
DEFAULT_GENERATE_URL = "http://127.0.0.1:30010/generate"
DEFAULT_CHAT_URL = "http://127.0.0.1:30010/v1/chat/completions"
DEFAULT_FLUSH_URL = "http://127.0.0.1:30010/flush_cache"
DEFAULT_SERVER_INFO_URL = "http://127.0.0.1:30010/server_info"
DEFAULT_MODEL = "deepseek-v4-flash"
TOOL_NAME = "record_benchmark_marker"
TOOL_MARKER = "pp2-concurrency-v2"
RECEIPT_VERSION = 4
TARGET_LAYER_COUNT = 43
TARGET_EP_SIZE = 1
TARGET_COMPRESSED_LAYER_IDS = frozenset(range(2, TARGET_LAYER_COUNT))
ADMITTED_CPUINFER_THREAD_COUNTS = frozenset({56})
PP2_CPU_WORKER_IDENTITIES = frozenset({(0, 0, 0, 0, 0), (0, 1, 0, 0, 1)})
OSCAR_STATIC_SERVER_INFO: dict[str, object] = {
    "dsv4_oscar_int2_kv_storage": True,
    "dsv4_oscar_algorithm": "oscar-int2-asym-g64-v1",
    "dsv4_oscar_model_id": "deepseek-ai/DeepSeek-V4-Flash",
    "dsv4_kv_storage_mode": "oscar_int2_asymmetric+protected_swa_bfloat16",
    "dsv4_latent_kv_bytes_per_token": 1_024,
    "dsv4_swa_kv_bytes_per_token": 1_024,
    "dsv4_c4_kv_bytes_per_token": 272,
    "dsv4_c128_kv_bytes_per_token": 272,
    "dsv4_oscar_c4_scorer": True,
    "dsv4_oscar_c4_scorer_algorithm": ("oscar-int2-c4-asym-c128-fp32-adjacent4-v1"),
    "dsv4_oscar_masked_writer_execution": "device-uniform-live-mask-row-v1",
    "dsv4_oscar_c4_masked_writer_execution": ("device-uniform-live-mask-row-v1"),
    "dsv4_oscar_c4_query_rotation_execution": ("once-per-query-stable-workspace-v1"),
    "dsv4_c4_indexer_bytes_per_token": 40,
    "dsv4_int4_kv_storage": False,
    "dsv4_int4_c4_indexer_storage": False,
    "dsv4_sm86_c128_bf16_storage": False,
}
OSCAR_SPLIT_HISTORY_SERVER_INFO: dict[str, object] = {
    "dsv4_oscar_int2_split_history": True,
    "dsv4_oscar_int2_split_history_execution": (
        "sm86-oscar-int2-split-history-fp32-online-v1"
    ),
    "dsv4_oscar_int2_split_history_split_map": {
        "1": 16,
        "2": 16,
        "3": 8,
        "4": 4,
        "5": 4,
        "6": 4,
        "7": 4,
        "8": 2,
    },
    "dsv4_oscar_int2_split_history_workspace_bytes": 4_210_688,
    "dsv4_oscar_int2_split_history_max_partial_rows": 32,
    "dsv4_oscar_int2_split_history_sink_owner": "stage2-exactly-once",
    "dsv4_oscar_int2_split_history_prefill_enabled": False,
    "dsv4_oscar_int2_split_history_fixed_address": True,
}
OSCAR_ADMISSION_HASH_KEYS = (
    "dsv4_oscar_artifact_sha256",
    "dsv4_oscar_model_config_sha256",
    "dsv4_oscar_artifact_provenance_sha256",
    "dsv4_oscar_checkpoint_sha256",
    "dsv4_oscar_checkpoint_fingerprint_sha256",
    "dsv4_oscar_admission_sha256",
    "dsv4_oscar_admission_receipt_sha256",
)
OSCAR_WO_A_EXACT_STATE: dict[str, object] = {
    "enabled": True,
    "consumer_role": "target_compressed",
    "target_only": True,
    "applied": True,
    "apply_count": 1,
    "all_local_target_compressed_layers_absorbed": True,
    "all_local_target_compressed_layers_skip_runtime_restore": True,
    "weight_dtype": "bfloat16",
    "head_layout": "per-head-nope448-rope64",
    "fold_orientation": "wo_a_nope@rotation",
    "rope_columns_unchanged": True,
}
SUPPORTED_EXPERT_PLAN_FORMATS = frozenset(
    {
        "sglang_kt_hybrid_expert_shard_v1",
        "sglang_kt_hybrid_expert_shard_v2_variable",
    }
)
MINIMUM_NATIVE_WORDS = 100
TAIL_WORD_COUNT = 40
REPEATED_NGRAM_SIZE = 4
COPY_NGRAM_SIZE = 8
MAXIMUM_START_SKEW_SECONDS = 0.25
FLUSH_RETRY_SECONDS = 0.1
WORD_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*")
SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\s\"')\]]|$)")
RUN_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", re.ASCII)


@dataclass(frozen=True, slots=True)
class NativeSemanticContract:
    profile: str
    baseline: str
    optimized: str
    improvement: str
    safeguards: tuple[str, str, str]
    required_ending: str

    @property
    def required_markers(self) -> tuple[str, ...]:
        return (
            self.profile,
            self.baseline,
            self.optimized,
            self.improvement,
            *self.safeguards,
        )


NATIVE_CONTRACTS = (
    NativeSemanticContract(
        profile="ORCHID-17",
        baseline="41",
        optimized="68",
        improvement="27",
        safeguards=("race condition", "cancellation safety", "resource cleanup"),
        required_ending="Lane ALPHA review complete.",
    ),
    NativeSemanticContract(
        profile="COBALT-29",
        baseline="38",
        optimized="71",
        improvement="33",
        safeguards=("schema validation", "cache isolation", "tool-call parsing"),
        required_ending="Lane BRAVO review complete.",
    ),
)

TOOL_COMPONENTS = ("src/exo/routing/router.py", "src/exo/worker/worker.py")
TOOL_DECISIONS = ("inspect-routing", "inspect-cancellation")


class BenchmarkError(RuntimeError):
    """A benchmark invariant failed without retaining model output."""


class PromptTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class ServingTokenizer(PromptTokenizer, Protocol):
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class TokenizerFactory(Protocol):
    def from_pretrained(self, model_path: str) -> ServingTokenizer: ...


class MessageEncoder(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_mode: str,
    ) -> str: ...


class HttpResponse(Protocol):
    status: int

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def read(self) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    generate_url: str
    chat_url: str
    flush_url: str
    model: str
    request_timeout_seconds: float
    flush_timeout_seconds: float
    chat_max_tokens: int
    require_nvlink_traffic: bool = False
    server_info_url: str | None = None
    expected_chunked_prefill_size: int | None = None
    expected_gpu_experts_per_layer: int | None = None
    expected_expert_plan_format: str | None = None
    expected_pp_async_batch_depth: int = 0
    expected_cpuinfer_threads: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkArguments:
    generate_url: str
    chat_url: str
    flush_url: str
    model: str
    model_path: Path
    request_timeout_seconds: float
    flush_timeout_seconds: float
    chat_max_tokens: int
    output_file: Path | None
    run_label: str | None
    expert_plan: Path | None
    require_nvlink_traffic: bool
    server_info_url: str | None
    expected_chunked_prefill_size: int | None
    expected_gpu_experts_per_layer: int | None
    expected_expert_plan_format: str | None
    expected_pp_async_batch_depth: int
    expected_cpuinfer_threads: int | None
    oscar_admission_receipt: Path
    launch_authorization_receipt: Path


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    run_label: str
    expert_plan_path: str
    expert_plan_sha256: str

    def safe_receipt(self) -> dict[str, str]:
        return {
            "run_label": self.run_label,
            "expert_plan_path": self.expert_plan_path,
            "expert_plan_sha256": self.expert_plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class OscarProvenance:
    admission_receipt_path: str
    admission_receipt_sha256: str
    artifact_path: str
    artifact_sha256: str
    admission_sha256: str

    def safe_receipt(self) -> dict[str, str]:
        return {
            "admission_receipt_path": self.admission_receipt_path,
            "admission_receipt_sha256": self.admission_receipt_sha256,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "admission_sha256": self.admission_sha256,
        }


@dataclass(frozen=True, slots=True)
class LaunchAuthorizationProvenance:
    authorization_receipt_path: str
    authorization_receipt_sha256: str
    ordinal: int
    run_role: str
    ep_confirmation_receipt_sha256: str
    ep_coherency_receipt_sha256: str

    def safe_receipt(self) -> dict[str, object]:
        return {
            "authorization_receipt_path": self.authorization_receipt_path,
            "authorization_receipt_sha256": self.authorization_receipt_sha256,
            "ordinal": self.ordinal,
            "run_role": self.run_role,
            "ep_confirmation_receipt_sha256": (self.ep_confirmation_receipt_sha256),
            "ep_coherency_receipt_sha256": self.ep_coherency_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class FlushObservation:
    status: int
    elapsed_seconds: float
    busy_retries: int
    response_sha256: str


@dataclass(frozen=True, slots=True)
class NativeSemanticAssessment:
    issue_codes: tuple[str, ...]
    word_count: int
    unique_word_ratio: float
    dominant_word_ratio: float
    repeated_four_gram_ratio: float
    tail_unique_word_ratio: float
    printable_character_ratio: float
    alphabetic_character_ratio: float
    sentence_count: int
    prompt_copy_eight_gram_ratio: float | None
    dominant_token_ratio: float | None
    repeated_token_four_gram_ratio: float | None
    maximum_repeated_token_run: int | None
    semantic_markers_present: int
    semantic_markers_required: int

    @property
    def passed(self) -> bool:
        return not self.issue_codes

    def safe_receipt(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issue_codes": list(self.issue_codes),
            "word_count": self.word_count,
            "unique_word_ratio": _rounded(self.unique_word_ratio),
            "dominant_word_ratio": _rounded(self.dominant_word_ratio),
            "repeated_four_gram_ratio": _rounded(self.repeated_four_gram_ratio),
            "tail_unique_word_ratio": _rounded(self.tail_unique_word_ratio),
            "printable_character_ratio": _rounded(self.printable_character_ratio),
            "alphabetic_character_ratio": _rounded(self.alphabetic_character_ratio),
            "sentence_count": self.sentence_count,
            "prompt_copy_eight_gram_ratio": (
                _rounded(self.prompt_copy_eight_gram_ratio)
                if self.prompt_copy_eight_gram_ratio is not None
                else None
            ),
            "dominant_token_ratio": (
                _rounded(self.dominant_token_ratio)
                if self.dominant_token_ratio is not None
                else None
            ),
            "repeated_token_four_gram_ratio": (
                _rounded(self.repeated_token_four_gram_ratio)
                if self.repeated_token_four_gram_ratio is not None
                else None
            ),
            "maximum_repeated_token_run": self.maximum_repeated_token_run,
            "semantic_markers_present": self.semantic_markers_present,
            "semantic_markers_required": self.semantic_markers_required,
            "required_ending_present": "missing_required_ending"
            not in self.issue_codes,
        }


@dataclass(frozen=True, slots=True)
class NativeLaneObservation:
    lane: int
    request_started_at: float
    first_output_at: float
    last_output_at: float
    request_finished_at: float
    input_sha256: str
    completion_tokens: int
    first_event_completion_tokens: int
    server_prompt_tokens: int | None
    finish_reason: str | None
    output_sha256: str
    output_hash_source: str
    semantic_text_source: str
    output_text_bytes: int
    semantic_assessment: NativeSemanticAssessment
    event_count: int
    saw_done: bool

    @property
    def time_to_first_token_seconds(self) -> float:
        return self.first_output_at - self.request_started_at

    @property
    def total_seconds(self) -> float:
        return self.request_finished_at - self.request_started_at

    @property
    def decode_seconds(self) -> float:
        return self.last_output_at - self.first_output_at

    @property
    def decode_tokens_per_second(self) -> float:
        tokens_after_first_event = (
            self.completion_tokens - self.first_event_completion_tokens
        )
        if tokens_after_first_event <= 0 or self.decode_seconds <= 0:
            return 0.0
        return tokens_after_first_event / self.decode_seconds


@dataclass(slots=True)
class ToolCallParts:
    identifier: str | None = None
    kind: str | None = None
    name_fragments: list[str] = field(default_factory=list)
    argument_fragments: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChatLaneObservation:
    lane: int
    request_started_at: float
    first_output_at: float | None
    last_output_at: float | None
    request_finished_at: float
    output_sha256: str
    output_bytes: int
    tool_arguments_sha256: str
    event_count: int
    completion_tokens: int | None
    finish_reason: str | None
    structural_parser_success: bool
    parser_issue_codes: tuple[str, ...]
    saw_done: bool

    @property
    def time_to_first_token_seconds(self) -> float | None:
        if self.first_output_at is None:
            return None
        return self.first_output_at - self.request_started_at

    @property
    def total_seconds(self) -> float:
        return self.request_finished_at - self.request_started_at


def _rounded(value: float) -> float:
    return round(value, 6)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _derive_server_info_url(generate_url: str) -> str:
    parsed = urlsplit(generate_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid generate URL: {generate_url!r}")
    return parsed._replace(path="/server_info", query="", fragment="").geturl()


def get_server_info(url: str, timeout_seconds: float) -> dict[str, object]:
    request = urllib.request.Request(url, method="GET")
    with cast(
        HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
    ) as response:
        body = response.read()
        status = response.status
    if status != 200:
        raise BenchmarkError("server-info endpoint returned a non-200 status")
    try:
        payload = cast(object, json.loads(body))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError("server-info endpoint did not return JSON") from error
    if not isinstance(payload, dict):
        raise BenchmarkError("server-info endpoint did not return an object")
    return cast(dict[str, object], payload)


def _casefolded(value: object) -> str | None:
    return value.casefold() if isinstance(value, str) else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_oscar_pp_worker_contract(
    server_info: dict[str, object],
    *,
    expected_artifact_sha256: str | None,
    expected_admission_sha256: str | None,
    expected_admission_receipt_sha256: str | None,
    issues: list[str],
) -> dict[str, object]:
    for key, expected_value in OSCAR_STATIC_SERVER_INFO.items():
        if server_info.get(key) != expected_value:
            issues.append(f"oscar_static_{key}_mismatch")
    for key, expected_value in OSCAR_SPLIT_HISTORY_SERVER_INFO.items():
        if server_info.get(key) != expected_value:
            issues.append(f"oscar_split_history_{key}_mismatch")
    top_level_workspace_address = server_info.get(
        "dsv4_oscar_int2_split_history_workspace_address"
    )
    if (
        type(top_level_workspace_address) is not int
        or cast(int, top_level_workspace_address) <= 0
    ):
        issues.append("oscar_split_history_workspace_address_invalid")

    top_level_hashes: dict[str, str] = {}
    for key in OSCAR_ADMISSION_HASH_KEYS:
        observed_value = server_info.get(key)
        if not _is_sha256(observed_value):
            issues.append(f"{key}_invalid")
        else:
            top_level_hashes[key] = cast(str, observed_value)
    expected_hashes = {
        "dsv4_oscar_artifact_sha256": expected_artifact_sha256,
        "dsv4_oscar_admission_sha256": expected_admission_sha256,
        "dsv4_oscar_admission_receipt_sha256": (expected_admission_receipt_sha256),
    }
    for key, expected_value in expected_hashes.items():
        if expected_value is not None and server_info.get(key) != expected_value:
            issues.append(f"{key}_not_expected")

    raw_internal_states = server_info.get("internal_states")
    if not isinstance(raw_internal_states, list) or not raw_internal_states:
        issues.append("oscar_pp_internal_states_missing")
        return {
            "static_server_info": dict(OSCAR_STATIC_SERVER_INFO),
            "admission_hashes": top_level_hashes,
            "workers": [],
            "compressed_layer_union": [],
        }

    gathered_workers: list[dict[str, object]] = []
    for state_index, raw_state in enumerate(raw_internal_states):
        if not isinstance(raw_state, dict):
            issues.append(f"oscar_pp_internal_state_{state_index}_not_object")
            continue
        raw_gathered = raw_state.get("dsv4_oscar_worker_telemetry_workers")
        if isinstance(raw_gathered, list):
            if not all(isinstance(worker, dict) for worker in raw_gathered):
                issues.append(
                    f"oscar_pp_internal_state_{state_index}_worker_gather_malformed"
                )
            else:
                gathered_workers.extend(cast(list[dict[str, object]], raw_gathered))
            continue
        raw_local_worker = raw_state.get("dsv4_oscar_worker_telemetry")
        if isinstance(raw_local_worker, dict):
            gathered_workers.append(cast(dict[str, object], raw_local_worker))
        else:
            issues.append(f"oscar_pp_internal_state_{state_index}_worker_missing")

    workers_by_rank: dict[tuple[object, object, object], dict[str, object]] = {}
    for worker in gathered_workers:
        rank_key = (
            repr(worker.get("dp_rank")),
            repr(worker.get("pp_rank")),
            repr(worker.get("tp_rank")),
        )
        prior_worker = workers_by_rank.get(rank_key)
        if prior_worker is not None and prior_worker != worker:
            issues.append("oscar_pp_worker_duplicate_rank_conflict")
        else:
            workers_by_rank[rank_key] = worker
    raw_workers = list(workers_by_rank.values())
    if len(raw_workers) != 2:
        issues.append("oscar_pp_worker_records_not_exactly_2")

    worker_receipts: list[dict[str, object]] = []
    pp_ranks: set[int] = set()
    gpu_ids: set[int] = set()
    pids: set[int] = set()
    worker_identities: set[tuple[int, int, int]] = set()
    split_workspace_addresses_by_pp_rank: dict[int, int] = {}
    layer_sets: list[set[int]] = []
    for index, worker in enumerate(raw_workers):
        tp_rank = worker.get("tp_rank")
        pp_rank = worker.get("pp_rank")
        dp_rank = worker.get("dp_rank")
        gpu_id = worker.get("gpu_id")
        pid = worker.get("pid")
        if tp_rank != 0:
            issues.append(f"oscar_pp_worker_{index}_tp_rank_not_0")
        if (
            not isinstance(pp_rank, int)
            or isinstance(pp_rank, bool)
            or pp_rank not in (0, 1)
        ):
            issues.append(f"oscar_pp_worker_{index}_pp_rank_invalid")
        else:
            pp_ranks.add(pp_rank)
        if dp_rank not in (None, 0):
            issues.append(f"oscar_pp_worker_{index}_dp_rank_invalid")
        if (
            not isinstance(gpu_id, int)
            or isinstance(gpu_id, bool)
            or gpu_id not in (0, 1)
        ):
            issues.append(f"oscar_pp_worker_{index}_gpu_id_invalid")
        else:
            gpu_ids.add(gpu_id)
        if (
            dp_rank in (None, 0)
            and isinstance(pp_rank, int)
            and not isinstance(pp_rank, bool)
            and isinstance(tp_rank, int)
            and not isinstance(tp_rank, bool)
            and isinstance(gpu_id, int)
            and not isinstance(gpu_id, bool)
        ):
            worker_identities.add((pp_rank, tp_rank, gpu_id))
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            issues.append(f"oscar_pp_worker_{index}_pid_invalid")
        else:
            pids.add(pid)

        for key, expected_value in OSCAR_STATIC_SERVER_INFO.items():
            if worker.get(key) != expected_value:
                issues.append(f"oscar_pp_worker_{index}_{key}_mismatch")
        for key, expected_value in OSCAR_SPLIT_HISTORY_SERVER_INFO.items():
            if worker.get(key) != expected_value:
                issues.append(f"oscar_pp_worker_{index}_split_history_{key}_mismatch")
        split_workspace_address = worker.get(
            "dsv4_oscar_int2_split_history_workspace_address"
        )
        if (
            type(split_workspace_address) is not int
            or cast(int, split_workspace_address) <= 0
        ):
            issues.append(
                f"oscar_pp_worker_{index}_split_history_workspace_address_invalid"
            )
        elif isinstance(pp_rank, int) and not isinstance(pp_rank, bool):
            split_workspace_addresses_by_pp_rank[pp_rank] = cast(
                int, split_workspace_address
            )
        worker_hashes: dict[str, str] = {}
        for key in OSCAR_ADMISSION_HASH_KEYS:
            observed_value = worker.get(key)
            if not _is_sha256(observed_value):
                issues.append(f"oscar_pp_worker_{index}_{key}_invalid")
            else:
                worker_hashes[key] = cast(str, observed_value)
            if key in top_level_hashes and observed_value != top_level_hashes[key]:
                issues.append(f"oscar_pp_worker_{index}_{key}_not_top_level")

        raw_absorption = worker.get("dsv4_oscar_wo_a_absorption_state")
        local_layers: list[int] = []
        if not isinstance(raw_absorption, dict):
            issues.append(f"oscar_pp_worker_{index}_wo_a_state_missing")
            absorption_receipt: dict[str, object] = {}
        else:
            absorption = cast(dict[str, object], raw_absorption)
            for key, expected_value in OSCAR_WO_A_EXACT_STATE.items():
                if absorption.get(key) != expected_value:
                    issues.append(f"oscar_pp_worker_{index}_wo_a_{key}_mismatch")
            if absorption.get("artifact_sha256") != worker.get(
                "dsv4_oscar_artifact_sha256"
            ):
                issues.append(f"oscar_pp_worker_{index}_wo_a_artifact_not_admitted")
            if absorption.get("admission_sha256") != worker.get(
                "dsv4_oscar_admission_sha256"
            ):
                issues.append(f"oscar_pp_worker_{index}_wo_a_admission_not_admitted")
            layer_fields = (
                "expected_local_compressed_layer_ids",
                "absorbed_local_layer_ids",
                "runtime_restore_skipped_layer_ids",
            )
            raw_layer_lists = [absorption.get(field) for field in layer_fields]
            if any(
                not isinstance(layer_ids, list)
                or not layer_ids
                or any(
                    not isinstance(layer_id, int) or isinstance(layer_id, bool)
                    for layer_id in layer_ids
                )
                or layer_ids != sorted(set(layer_ids))
                for layer_ids in raw_layer_lists
            ):
                issues.append(f"oscar_pp_worker_{index}_wo_a_layers_invalid")
            elif not all(
                layer_ids == raw_layer_lists[0] for layer_ids in raw_layer_lists[1:]
            ):
                issues.append(f"oscar_pp_worker_{index}_wo_a_layers_inexact")
            else:
                local_layers = cast(list[int], raw_layer_lists[0])
                if not set(local_layers) <= TARGET_COMPRESSED_LAYER_IDS:
                    issues.append(f"oscar_pp_worker_{index}_wo_a_layers_out_of_range")
                layer_sets.append(set(local_layers))
            absorption_receipt = dict(absorption)

        worker_receipts.append(
            {
                "pid": pid,
                "gpu_id": gpu_id,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "dp_rank": dp_rank,
                "static_server_info": dict(OSCAR_STATIC_SERVER_INFO),
                "admission_hashes": worker_hashes,
                "wo_a_absorption_state": absorption_receipt,
            }
        )

    if pp_ranks != {0, 1}:
        issues.append("oscar_pp_worker_pp_rank_coverage_not_0_1")
    if gpu_ids != {0, 1}:
        issues.append("oscar_pp_worker_gpu_coverage_not_0_1")
    if worker_identities != {(0, 0, 0), (1, 0, 1)}:
        issues.append("oscar_pp_worker_identity_coverage_not_exact")
    if len(pids) != 2:
        issues.append("oscar_pp_worker_pids_not_distinct")
    if set(split_workspace_addresses_by_pp_rank) != {0, 1}:
        issues.append("oscar_pp_split_history_workspace_stage_coverage_incomplete")
    if len(layer_sets) != 2:
        issues.append("oscar_pp_wo_a_stage_coverage_incomplete")
        layer_union: set[int] = set()
        for layer_set in layer_sets:
            layer_union.update(layer_set)
    else:
        if layer_sets[0] & layer_sets[1]:
            issues.append("oscar_pp_wo_a_stage_layers_overlap")
        layer_union = layer_sets[0] | layer_sets[1]
        if layer_union != TARGET_COMPRESSED_LAYER_IDS:
            issues.append("oscar_pp_wo_a_layer_union_not_2_42")

    def worker_rank_sort_key(worker: dict[str, object]) -> int:
        rank = worker.get("pp_rank")
        return rank if isinstance(rank, int) and not isinstance(rank, bool) else 2

    worker_receipts.sort(key=worker_rank_sort_key)
    return {
        "static_server_info": dict(OSCAR_STATIC_SERVER_INFO),
        "admission_hashes": top_level_hashes,
        "workers": worker_receipts,
        "compressed_layer_union": sorted(layer_union),
        "split_history": {
            "worker_identities": [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
            ],
            "workspace_bytes_per_worker": 4_210_688,
            "workspace_addresses_by_pp_rank": {
                str(rank): address
                for rank, address in sorted(
                    split_workspace_addresses_by_pp_rank.items()
                )
            },
            "fixed_address": True,
        },
    }


def _validate_confirmed_cpu_tuple_telemetry(
    server_info: dict[str, object],
    *,
    issues: list[str],
) -> dict[str, object]:
    """Require the confirmed inline/N128-LUT tuple on both PP workers."""

    proof_contracts = (
        (
            "kt_single_numa_inline_dispatch",
            "inline",
        ),
        (
            "kt_mxfp4_avx_scale_fold",
            "scale_fold",
        ),
    )
    receipts: dict[str, object] = {}
    for prefix, receipt_name in proof_contracts:
        exact_top_level = {
            f"{prefix}_configured": True,
            f"{prefix}_expected_worker_count": 2,
            f"{prefix}_reporting_worker_count": 2,
            f"{prefix}_active_worker_count": 2,
            f"{prefix}_invalid_worker_count": 0,
            f"{prefix}_duplicate_worker_count": 0,
            f"{prefix}_rank_coverage_valid": True,
            f"{prefix}_topology": "pp2-ep1",
            f"{prefix}_supported_topology_valid": True,
            f"{prefix}_ep2_topology_valid": False,
            f"{prefix}_all_workers_active": True,
        }
        for key, expected in exact_top_level.items():
            if server_info.get(key) != expected:
                issues.append(f"pp2_cpu_{receipt_name}_{key}_mismatch")

        raw_workers = server_info.get(f"{prefix}_worker_telemetry")
        workers = (
            cast(list[dict[str, object]], raw_workers)
            if isinstance(raw_workers, list)
            and all(isinstance(worker, dict) for worker in raw_workers)
            else []
        )
        identities: set[tuple[int, int, int, int, int]] = set()
        for worker_index, worker in enumerate(workers):
            raw_identity = (
                0 if worker.get("dp_rank") is None else worker.get("dp_rank"),
                worker.get("pp_rank"),
                worker.get("tp_rank"),
                worker.get("moe_ep_rank"),
                worker.get("gpu_id"),
            )
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_identity
            ):
                issues.append(
                    f"pp2_cpu_{receipt_name}_worker_{worker_index}_identity_invalid"
                )
                continue
            identity = cast(tuple[int, int, int, int, int], raw_identity)
            identities.add(identity)
            telemetry = worker.get("telemetry")
            if not isinstance(telemetry, dict):
                issues.append(
                    f"pp2_cpu_{receipt_name}_worker_{worker_index}_telemetry_missing"
                )
                continue
            if receipt_name == "inline":
                if (
                    telemetry.get("required_worker_count") != 56
                    or telemetry.get("environment_enabled") is not True
                    or telemetry.get("all_live_worker_affinities_exact") is not True
                    or telemetry.get("all_worker_cpus_in_expected_numa") is not True
                ):
                    issues.append(
                        f"pp2_cpu_inline_worker_{worker_index}_native_proof_mismatch"
                    )
            elif (
                telemetry.get("requested_mode") != "lut-v1"
                or telemetry.get("n_block") != 128
                or telemetry.get("lut_hash") != "06d1a83dbf20f545"
                or telemetry.get("zero_invalid_or_fallback_counts") is not True
            ):
                issues.append(
                    f"pp2_cpu_scale_fold_worker_{worker_index}_native_proof_mismatch"
                )
        if identities != PP2_CPU_WORKER_IDENTITIES:
            issues.append(f"pp2_cpu_{receipt_name}_worker_identity_coverage_mismatch")
        receipts[receipt_name] = {
            "worker_identities": [list(identity) for identity in sorted(identities)],
            "worker_count": len(workers),
        }
    return receipts


def validate_pp2_server_contract(
    server_info: dict[str, object],
    *,
    expected_chunked_prefill_size: int | None,
    expected_gpu_experts_per_layer: int | None,
    expected_expert_plan_format: str | None = None,
    expected_pp_async_batch_depth: int = 0,
    expected_expert_plan_sha256: str | None = None,
    expected_cpuinfer_threads: int | None = None,
    expected_oscar_artifact_sha256: str | None = None,
    expected_oscar_admission_sha256: str | None = None,
    expected_oscar_admission_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Fail closed unless the live service is the exact local PP2 target."""

    issues: list[str] = []
    if server_info.get("tp_size") != 1:
        issues.append("tp_size_not_1")
    if server_info.get("pp_size") != 2:
        issues.append("pp_size_not_2")
    if server_info.get("ep_size") != 1:
        issues.append("ep_size_not_1")
    if server_info.get("context_length") != 524_288:
        issues.append("context_length_not_524288")
    if server_info.get("max_total_tokens") != 524_288:
        issues.append("max_total_tokens_not_524288")
    if _casefolded(server_info.get("kv_cache_dtype")) != "fp8_e4m3":
        issues.append("kv_cache_not_fp8_e4m3")
    if server_info.get("disable_cuda_graph") is not False:
        issues.append("cuda_graph_not_enabled")
    if server_info.get("disable_decode_cuda_graph") is not False:
        issues.append("decode_cuda_graph_not_enabled")
    if _casefolded(server_info.get("cuda_graph_backend_decode")) != "full":
        issues.append("decode_cuda_graph_backend_not_full")
    decode_graph_max_batch_size = server_info.get("cuda_graph_max_bs_decode")
    if (
        not isinstance(decode_graph_max_batch_size, int)
        or isinstance(decode_graph_max_batch_size, bool)
        or decode_graph_max_batch_size != 2
    ):
        issues.append("decode_cuda_graph_max_batch_size_not_2")
    raw_decode_graph_batch_sizes = server_info.get("cuda_graph_bs_decode")
    decode_graph_batch_sizes = (
        cast(list[object], raw_decode_graph_batch_sizes)
        if isinstance(raw_decode_graph_batch_sizes, list)
        else None
    )
    if (
        decode_graph_batch_sizes is None
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in decode_graph_batch_sizes
        )
        or decode_graph_batch_sizes != [1, 2]
    ):
        issues.append("decode_cuda_graph_batch_sizes_not_1_2")
    if _casefolded(server_info.get("cuda_graph_backend_prefill")) != "disabled":
        issues.append("prefill_cuda_graph_backend_not_disabled")
    if server_info.get("disable_overlap_schedule") is not True:
        issues.append("overlap_schedule_not_disabled")
    speculative_algorithm = server_info.get("speculative_algorithm")
    if speculative_algorithm not in (None, ""):
        issues.append("speculative_decoding_not_disabled")
    if server_info.get("max_running_requests") != 2:
        issues.append("max_running_requests_not_2")
    if server_info.get("pp_max_micro_batch_size") != 1:
        issues.append("pp_max_micro_batch_size_not_1")
    if server_info.get("pp_async_batch_depth") != expected_pp_async_batch_depth:
        issues.append("pp_async_batch_depth_not_expected")
    chunked_prefill_size = server_info.get("chunked_prefill_size")
    if chunked_prefill_size not in (512, 1024):
        issues.append("chunked_prefill_size_not_512_or_1024")
    if (
        expected_chunked_prefill_size is not None
        and chunked_prefill_size != expected_chunked_prefill_size
    ):
        issues.append("chunked_prefill_size_not_expected")
    if server_info.get("max_prefill_tokens") != chunked_prefill_size:
        issues.append("max_prefill_tokens_not_chunked_prefill_size")

    gpu_experts_per_layer = server_info.get("kt_num_gpu_experts")
    if (
        not isinstance(gpu_experts_per_layer, int)
        or isinstance(gpu_experts_per_layer, bool)
        or gpu_experts_per_layer <= 0
    ):
        issues.append("gpu_expert_count_invalid")
    if (
        expected_gpu_experts_per_layer is not None
        and gpu_experts_per_layer != expected_gpu_experts_per_layer
    ):
        issues.append("gpu_expert_count_not_expected")
    if server_info.get("kt_gpu_expert_admission_ceiling") != gpu_experts_per_layer:
        issues.append("gpu_expert_admission_ceiling_mismatch")

    plan_format = server_info.get("kt_hybrid_expert_plan_format")
    if plan_format not in SUPPORTED_EXPERT_PLAN_FORMATS:
        issues.append("expert_plan_format_not_supported")
    if (
        expected_expert_plan_format is not None
        and plan_format != expected_expert_plan_format
    ):
        issues.append("expert_plan_format_not_expected")

    raw_gpu_counts = server_info.get("kt_hybrid_gpu_rank_counts_by_layer")
    gpu_counts: list[int] | None = None
    if (
        isinstance(raw_gpu_counts, list)
        and len(raw_gpu_counts) == TARGET_EP_SIZE
        and isinstance(raw_gpu_counts[0], list)
        and len(raw_gpu_counts[0]) == TARGET_LAYER_COUNT
        and all(
            isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 256
            for value in raw_gpu_counts[0]
        )
    ):
        gpu_counts = cast(list[int], raw_gpu_counts[0])
    else:
        issues.append("expert_plan_gpu_counts_not_ep1_x_43")
    if gpu_counts is not None:
        minimum_gpu_count = min(gpu_counts)
        maximum_gpu_count = max(gpu_counts)
        total_gpu_count = sum(gpu_counts)
        if (
            server_info.get("kt_hybrid_min_gpu_experts_per_rank_per_layer")
            != minimum_gpu_count
        ):
            issues.append("expert_plan_minimum_gpu_count_mismatch")
        if (
            server_info.get("kt_hybrid_max_gpu_experts_per_rank_per_layer")
            != maximum_gpu_count
        ):
            issues.append("expert_plan_maximum_gpu_count_mismatch")
        if server_info.get("kt_hybrid_total_gpu_expert_layers_by_rank") != [
            total_gpu_count
        ]:
            issues.append("expert_plan_total_gpu_count_mismatch")
        if maximum_gpu_count != gpu_experts_per_layer:
            issues.append("gpu_expert_admission_ceiling_not_plan_maximum")

    placement_semantics_sha256 = server_info.get("kt_hybrid_placement_semantics_sha256")
    if (
        not isinstance(placement_semantics_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", placement_semantics_sha256) is None
    ):
        issues.append("expert_plan_semantics_sha256_invalid")

    cpuinfer_threads = server_info.get("kt_cpuinfer")
    if cpuinfer_threads not in ADMITTED_CPUINFER_THREAD_COUNTS:
        issues.append("cpu_offload_threads_not_admitted")
    if (
        expected_cpuinfer_threads is not None
        and cpuinfer_threads != expected_cpuinfer_threads
    ):
        issues.append("cpu_offload_threads_not_expected")
    if server_info.get("kt_threadpool_count") != 2:
        issues.append("cpu_offload_threadpool_count_not_2")
    if server_info.get("swa_full_tokens_ratio") != 0.0048828125:
        issues.append("swa_full_tokens_ratio_not_0_0048828125")
    if server_info.get("mem_fraction_static") != 0.90:
        issues.append("mem_fraction_static_not_0_90")
    if server_info.get("disable_radix_cache") is not False:
        issues.append("radix_cache_not_enabled")
    if server_info.get("dsv4_small_row_routing_configured") is not True:
        issues.append("small_row_routing_not_enabled")
    if server_info.get("dsv4_sm86_small_batch_gemm_configured") is not True:
        issues.append("sm86_small_batch_gemm_not_enabled")
    if server_info.get("kt_amx_fine_grained_decode_configured") is not True:
        issues.append("amx_fine_grained_decode_not_enabled")
    if server_info.get("kt_mxfp4_amx_min_expert_tokens") != 5:
        issues.append("amx_min_expert_tokens_not_5")
    if server_info.get("kt_mxfp4_avx_tiled_min_expert_tokens") != 2:
        issues.append("avx_tiled_min_expert_tokens_not_2")
    for server_key in (
        "dsv4_int4_c4_indexer_storage",
        "dsv4_int4_kv_storage",
        "dsv4_sm86_c128_bf16_storage",
    ):
        if server_info.get(server_key) is not False:
            issues.append(f"{server_key}_not_disabled_for_oscar")
    if server_info.get("enable_p2p_check") is not True:
        issues.append("cuda_p2p_check_not_enabled")
    if server_info.get("pre_warm_nccl") is not True:
        issues.append("nccl_not_prewarmed")
    if (
        expected_expert_plan_sha256 is not None
        and server_info.get("kt_hybrid_expert_plan_sha256")
        != expected_expert_plan_sha256
    ):
        issues.append("loaded_expert_plan_sha256_not_expected")
    loaded_plan_sha256 = server_info.get("kt_hybrid_expert_plan_sha256")
    if (
        not isinstance(loaded_plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", loaded_plan_sha256) is None
    ):
        issues.append("loaded_expert_plan_sha256_invalid")
    oscar_pp_contract = _validate_oscar_pp_worker_contract(
        server_info,
        expected_artifact_sha256=expected_oscar_artifact_sha256,
        expected_admission_sha256=expected_oscar_admission_sha256,
        expected_admission_receipt_sha256=(expected_oscar_admission_receipt_sha256),
        issues=issues,
    )
    confirmed_cpu_tuple = _validate_confirmed_cpu_tuple_telemetry(
        server_info,
        issues=issues,
    )
    if issues:
        raise BenchmarkError("PP2 server contract mismatch: " + ",".join(issues))

    receipt_keys = (
        "tp_size",
        "pp_size",
        "ep_size",
        "context_length",
        "max_total_tokens",
        "kv_cache_dtype",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "cuda_graph_backend_decode",
        "cuda_graph_max_bs_decode",
        "cuda_graph_bs_decode",
        "cuda_graph_backend_prefill",
        "disable_overlap_schedule",
        "speculative_algorithm",
        "max_running_requests",
        "pp_max_micro_batch_size",
        "pp_async_batch_depth",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "kt_num_gpu_experts",
        "kt_gpu_expert_admission_ceiling",
        "kt_hybrid_expert_plan_format",
        "kt_hybrid_placement_semantics_sha256",
        "kt_hybrid_gpu_rank_counts_by_layer",
        "kt_hybrid_min_gpu_experts_per_rank_per_layer",
        "kt_hybrid_max_gpu_experts_per_rank_per_layer",
        "kt_hybrid_total_gpu_expert_layers_by_rank",
        "kt_cpuinfer",
        "kt_threadpool_count",
        "swa_full_tokens_ratio",
        "mem_fraction_static",
        "disable_radix_cache",
        "dsv4_small_row_routing_configured",
        "dsv4_sm86_small_batch_gemm_configured",
        "kt_amx_fine_grained_decode_configured",
        "kt_mxfp4_amx_min_expert_tokens",
        "kt_mxfp4_avx_tiled_min_expert_tokens",
        "dsv4_int4_c4_indexer_storage",
        "dsv4_int4_kv_storage",
        "dsv4_sm86_c128_bf16_storage",
        "enable_p2p_check",
        "pre_warm_nccl",
        "kt_hybrid_expert_plan_sha256",
        *OSCAR_STATIC_SERVER_INFO,
        *OSCAR_SPLIT_HISTORY_SERVER_INFO,
        "dsv4_oscar_int2_split_history_workspace_address",
        *OSCAR_ADMISSION_HASH_KEYS,
    )
    return {
        **{key: server_info.get(key) for key in receipt_keys},
        "oscar_pp_worker_contract": oscar_pp_contract,
        "confirmed_cpu_tuple": confirmed_cpu_tuple,
    }


def hash_token_ids(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError("token IDs must be non-negative")
        digest.update(token_id.to_bytes(8, byteorder="big", signed=True))
    return digest.hexdigest()


def load_benchmark_provenance(
    *,
    run_label: str | None,
    expert_plan: Path | None,
) -> BenchmarkProvenance | None:
    if (run_label is None) != (expert_plan is None):
        raise ValueError("run label and expert plan must be supplied together")
    if run_label is None or expert_plan is None:
        return None
    if RUN_LABEL_PATTERN.fullmatch(run_label) is None:
        raise ValueError(
            "run label must contain 1-64 ASCII letters, digits, dots, dashes, or underscores"
        )
    if not expert_plan.is_absolute():
        raise ValueError(f"expert plan path must be absolute: {expert_plan}")
    if expert_plan.is_symlink():
        raise ValueError(f"expert plan must not be a symlink: {expert_plan}")
    try:
        resolved_plan = expert_plan.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"cannot resolve expert plan {expert_plan}: {error}"
        ) from error
    if not resolved_plan.is_file():
        raise ValueError(f"expert plan is not a regular file: {resolved_plan}")
    digest = hashlib.sha256()
    try:
        with resolved_plan.open("rb") as plan_file:
            while chunk := plan_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash expert plan {resolved_plan}: {error}") from error
    return BenchmarkProvenance(
        run_label=run_label,
        expert_plan_path=str(resolved_plan),
        expert_plan_sha256=digest.hexdigest(),
    )


def _hash_absolute_regular_file(path: Path, *, label: str) -> tuple[Path, str]:
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve {label} {path}: {error}") from error
    if not resolved_path.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved_path}")
    digest = hashlib.sha256()
    try:
        with resolved_path.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash {label} {resolved_path}: {error}") from error
    return resolved_path, digest.hexdigest()


def load_launch_authorization_provenance(
    authorization_receipt: Path,
) -> LaunchAuthorizationProvenance:
    resolved, receipt_sha256 = _hash_absolute_regular_file(
        authorization_receipt,
        label="PP2 launch authorization receipt",
    )
    try:
        raw = cast(object, json.loads(resolved.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "PP2 launch authorization receipt is not readable JSON"
        ) from error
    if not isinstance(raw, dict):
        raise TypeError("PP2 launch authorization receipt must contain an object")
    receipt = cast(dict[str, object], raw)
    ep_winner = receipt.get("ep_winner")
    configuration = receipt.get("configuration")
    if not isinstance(ep_winner, dict) or not isinstance(configuration, dict):
        raise TypeError("PP2 launch authorization receipt is malformed")
    ep_confirmation_sha256 = ep_winner.get("receipt_sha256")
    ep_coherency_sha256 = ep_winner.get("coherency_receipt_sha256")
    if (
        receipt.get("format") != "dsv4_pp2_model_launch_authorization_v1"
        or receipt.get("ordinal") not in (1, 2)
        or receipt.get("run_role") not in ("transfer", "optimized")
        or receipt.get("run_role")
        != ("transfer" if receipt.get("ordinal") == 1 else "optimized")
        or not _is_sha256(ep_confirmation_sha256)
        or not _is_sha256(ep_coherency_sha256)
        or configuration.get("tensor_parallel_size") != 1
        or configuration.get("pipeline_parallel_size") != 2
        or configuration.get("expert_parallel_size") != 1
        or configuration.get("context_length") != 524_288
        or configuration.get("max_total_tokens") != 524_288
        or configuration.get("decode_cuda_graph_backend") != "full"
        or configuration.get("kv_cache_public_carrier") != "fp8_e4m3"
        or configuration.get("physical_kv_cache_storage") != "oscar-int2-asymmetric"
        or configuration.get("oscar_split_history") is not True
        or configuration.get("oscar_split_history_execution")
        != "sm86-oscar-int2-split-history-fp32-online-v1"
        or configuration.get("oscar_split_history_split_map")
        != OSCAR_SPLIT_HISTORY_SERVER_INFO["dsv4_oscar_int2_split_history_split_map"]
        or configuration.get("oscar_split_history_workspace_bytes_per_worker")
        != 4_210_688
        or configuration.get("oscar_split_history_worker_identities")
        != [[0, 0, 0, 0, 0], [0, 1, 0, 0, 1]]
        or not _is_sha256(configuration.get("transferred_plan_sha256"))
        or configuration.get("native_artifact_sha256")
        != "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043"
        or configuration.get("cpuinfer_threads") != 56
        or configuration.get("worker_spin_us") != 1000
        or configuration.get("task_queue_pin_first_core") is not True
        or configuration.get("single_numa_inline_dispatch") is not True
        or configuration.get("scale_fold_mode") != "lut-v1"
        or configuration.get("scale_fold_n_block") != 128
        or configuration.get("scale_fold_lut_hash") != "06d1a83dbf20f545"
    ):
        raise ValueError("PP2 launch authorization violates the serving contract")
    return LaunchAuthorizationProvenance(
        authorization_receipt_path=str(resolved),
        authorization_receipt_sha256=receipt_sha256,
        ordinal=cast(int, receipt["ordinal"]),
        run_role=cast(str, receipt["run_role"]),
        ep_confirmation_receipt_sha256=cast(str, ep_confirmation_sha256),
        ep_coherency_receipt_sha256=cast(str, ep_coherency_sha256),
    )


def load_oscar_provenance(admission_receipt: Path) -> OscarProvenance:
    resolved_receipt, receipt_sha256 = _hash_absolute_regular_file(
        admission_receipt,
        label="OSCAR admission receipt",
    )
    try:
        raw_receipt = cast(object, json.loads(resolved_receipt.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OSCAR admission receipt is not readable JSON") from error
    if not isinstance(raw_receipt, dict):
        raise TypeError("OSCAR admission receipt must be a JSON object")
    receipt = cast(dict[str, object], raw_receipt)
    required_fields = {
        "format",
        "format_version",
        "admitted",
        "model_id",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_provenance_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_sha256",
        "checkpoint_fingerprint_path",
        "checkpoint_fingerprint_sha256",
        "validation_policy",
        "admission_sha256",
    }
    if set(receipt) != required_fields:
        raise ValueError("OSCAR admission receipt has the wrong exact schema")
    exact_fields: dict[str, object] = {
        "format": "dsv4-oscar-int2-admission",
        "format_version": 1,
        "admitted": True,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
    }
    if any(receipt.get(key) != value for key, value in exact_fields.items()):
        raise ValueError("OSCAR admission receipt violates the serving contract")
    artifact_path_value = receipt.get("artifact_path")
    if not isinstance(artifact_path_value, str):
        raise TypeError("OSCAR admission artifact_path must be a string")
    resolved_artifact, artifact_sha256 = _hash_absolute_regular_file(
        Path(artifact_path_value),
        label="OSCAR calibration artifact",
    )
    if receipt.get("artifact_file_sha256") != artifact_sha256:
        raise ValueError("OSCAR calibration artifact digest does not match admission")
    admission_sha256 = receipt.get("admission_sha256")
    if (
        not isinstance(admission_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", admission_sha256) is None
    ):
        raise ValueError("OSCAR admission digest is not SHA-256")
    canonical_receipt = dict(receipt)
    canonical_receipt.pop("admission_sha256")
    computed_admission_sha256 = hashlib.sha256(
        json.dumps(
            canonical_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if admission_sha256 != computed_admission_sha256:
        raise ValueError("OSCAR admission receipt canonical digest mismatch")
    return OscarProvenance(
        admission_receipt_path=str(resolved_receipt),
        admission_receipt_sha256=receipt_sha256,
        artifact_path=str(resolved_artifact),
        artifact_sha256=artifact_sha256,
        admission_sha256=admission_sha256,
    )


def jains_fairness(values: Sequence[float]) -> float:
    if not values or any(value < 0 for value in values):
        raise ValueError("fairness values must be a non-empty non-negative sequence")
    squared_sum = sum(value * value for value in values)
    if squared_sum == 0:
        return 0.0
    fairness = sum(values) ** 2 / (len(values) * squared_sum)
    return min(1.0, fairness)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(text)]


def _repeated_ngram_ratio(values: Sequence[Hashable], size: int) -> float:
    if len(values) < size:
        return 0.0
    ngrams = [
        tuple(values[index : index + size]) for index in range(len(values) - size + 1)
    ]
    counts = Counter(ngrams)
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    return _ratio(repeated_occurrences, len(ngrams))


def _prompt_copy_ratio(
    prompt_ids: Sequence[int], output_ids: Sequence[int], size: int
) -> float | None:
    if len(prompt_ids) < size or len(output_ids) < size:
        return None
    output_ngrams = [
        tuple(output_ids[index : index + size])
        for index in range(len(output_ids) - size + 1)
    ]
    prompt_ngrams = {
        tuple(prompt_ids[index : index + size])
        for index in range(len(prompt_ids) - size + 1)
    }
    return _ratio(
        sum(ngram in prompt_ngrams for ngram in output_ngrams), len(output_ngrams)
    )


def _maximum_repeated_run(values: Sequence[int]) -> int:
    maximum = 0
    current = 0
    previous: int | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        maximum = max(maximum, current)
    return maximum


def assess_native_output(
    lane: int,
    output_text: str,
    *,
    prompt_ids: Sequence[int] = (),
    output_ids: Sequence[int] = (),
) -> NativeSemanticAssessment:
    """Reject semantically wrong, copied, malformed, or repetitive prose."""

    if lane not in range(LANE_COUNT):
        raise ValueError("lane must be zero or one")
    contract = NATIVE_CONTRACTS[lane]
    words = _words(output_text)
    word_counts = Counter(words)
    word_count = len(words)
    tail_words = words[-TAIL_WORD_COUNT:]
    unique_word_ratio = _ratio(len(word_counts), word_count)
    dominant_word_ratio = _ratio(max(word_counts.values(), default=0), word_count)
    repeated_word_ratio = _repeated_ngram_ratio(words, REPEATED_NGRAM_SIZE)
    tail_unique_word_ratio = _ratio(len(set(tail_words)), len(tail_words))
    printable_character_count = sum(
        character.isprintable() or character in "\n\r\t" for character in output_text
    )
    visible_character_count = sum(not character.isspace() for character in output_text)
    alphabetic_character_count = sum(character.isalpha() for character in output_text)
    printable_character_ratio = _ratio(printable_character_count, len(output_text))
    alphabetic_character_ratio = _ratio(
        alphabetic_character_count, visible_character_count
    )
    sentence_count = len(SENTENCE_END_PATTERN.findall(output_text))
    folded_output = output_text.casefold()
    semantic_markers_present = sum(
        marker.casefold() in folded_output for marker in contract.required_markers
    )

    prompt_copy_ratio = (
        _prompt_copy_ratio(prompt_ids, output_ids, COPY_NGRAM_SIZE)
        if prompt_ids and output_ids
        else None
    )
    dominant_token_ratio = (
        _ratio(max(Counter(output_ids).values(), default=0), len(output_ids))
        if output_ids
        else None
    )
    repeated_token_ratio = (
        _repeated_ngram_ratio(output_ids, REPEATED_NGRAM_SIZE) if output_ids else None
    )
    maximum_repeated_token_run = (
        _maximum_repeated_run(output_ids) if output_ids else None
    )

    issue_codes: set[str] = set()
    if not output_text.strip():
        issue_codes.add("output_empty")
    if word_count < MINIMUM_NATIVE_WORDS:
        issue_codes.add("output_too_short")
    for marker_index, marker in enumerate(contract.required_markers):
        if marker.casefold() not in folded_output:
            issue_codes.add(f"missing_semantic_marker_{marker_index}")
    if (
        not output_text.rstrip()
        .casefold()
        .endswith(contract.required_ending.casefold())
    ):
        issue_codes.add("missing_required_ending")
    if unique_word_ratio < 0.35:
        issue_codes.add("low_lexical_diversity")
    if dominant_word_ratio > 0.10:
        issue_codes.add("dominant_repeated_word")
    if repeated_word_ratio > 0.08:
        issue_codes.add("repeated_four_gram")
    if len(tail_words) < TAIL_WORD_COUNT or tail_unique_word_ratio < 0.35:
        issue_codes.add("degenerate_output_tail")
    if printable_character_ratio < 0.995 or any(
        unicodedata.category(character) in {"Cc", "Cs"} and character not in "\n\r\t"
        for character in output_text
    ):
        issue_codes.add("invalid_character_mix")
    if alphabetic_character_ratio < 0.55:
        issue_codes.add("low_alphabetic_content")
    if sentence_count < 5:
        issue_codes.add("too_few_sentences")
    if prompt_copy_ratio is not None and prompt_copy_ratio > 0.35:
        issue_codes.add("prompt_copying")
    if dominant_token_ratio is not None and dominant_token_ratio > 0.15:
        issue_codes.add("dominant_repeated_token")
    if repeated_token_ratio is not None and repeated_token_ratio > 0.10:
        issue_codes.add("repeated_token_four_gram")
    if maximum_repeated_token_run is not None and maximum_repeated_token_run > 8:
        issue_codes.add("repeated_token_run")

    return NativeSemanticAssessment(
        issue_codes=tuple(sorted(issue_codes)),
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        dominant_word_ratio=dominant_word_ratio,
        repeated_four_gram_ratio=repeated_word_ratio,
        tail_unique_word_ratio=tail_unique_word_ratio,
        printable_character_ratio=printable_character_ratio,
        alphabetic_character_ratio=alphabetic_character_ratio,
        sentence_count=sentence_count,
        prompt_copy_eight_gram_ratio=prompt_copy_ratio,
        dominant_token_ratio=dominant_token_ratio,
        repeated_token_four_gram_ratio=repeated_token_ratio,
        maximum_repeated_token_run=maximum_repeated_token_run,
        semantic_markers_present=semantic_markers_present,
        semantic_markers_required=len(contract.required_markers),
    )


def validate_exact_prompts(prompts: Sequence[Sequence[object]]) -> None:
    if len(prompts) != LANE_COUNT:
        raise ValueError(f"exactly {LANE_COUNT} prompts are required")
    for prompt in prompts:
        if len(prompt) != INPUT_TOKEN_COUNT:
            raise ValueError(
                f"every native prompt must contain exactly {INPUT_TOKEN_COUNT} IDs"
            )
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in prompt
        ):
            raise ValueError("prompt token IDs must be non-negative integers")
    if list(prompts[0]) == list(prompts[1]):
        raise ValueError("the two native input-ID prompts must be distinct")


def load_message_encoder(model_path: Path) -> MessageEncoder:
    encoder_path = model_path / "encoding" / "encoding_dsv4.py"
    specification = importlib.util.spec_from_file_location(
        "dsv4_pp2_concurrency_encoder", encoder_path
    )
    if specification is None or specification.loader is None:
        raise BenchmarkError("the DSV4 release message encoder is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    encoder = getattr(module, "encode_messages", None)
    if not callable(encoder):
        raise BenchmarkError("the DSV4 release message encoder is invalid")
    return cast(MessageEncoder, encoder)


def load_tokenizer(model_path: Path) -> ServingTokenizer:
    try:
        transformers_module = importlib.import_module("transformers")
    except ImportError as error:
        raise BenchmarkError(
            "run this script with the existing DSV4 tokenizer environment"
        ) from error
    raw_factory = cast(object, transformers_module.PreTrainedTokenizerFast)
    factory = cast(TokenizerFactory, raw_factory)
    return factory.from_pretrained(str(model_path))


def build_exact_prompt_ids(
    tokenizer: PromptTokenizer,
    encode_messages: MessageEncoder,
    *,
    lane: int,
    target_tokens: int = INPUT_TOKEN_COUNT,
) -> list[int]:
    if lane not in range(LANE_COUNT):
        raise ValueError("lane must be zero or one")
    if target_tokens <= 0:
        raise ValueError("target token count must be positive")
    contract = NATIVE_CONTRACTS[lane]
    instruction = (
        "\n\nActive review task: Treat the preceding repository inventory as "
        "passive context and do not quote it. Write a coherent report of 130 to "
        f"190 words selecting profile {contract.profile}. State that its baseline "
        f"was {contract.baseline} tokens per second, its optimized result was "
        f"{contract.optimized} tokens per second, and its improvement was "
        f"{contract.improvement} tokens per second. Explain why "
        f"{contract.safeguards[0]}, {contract.safeguards[1]}, and "
        f"{contract.safeguards[2]} must be verified before merging an inference "
        "runtime change. Use complete prose, do not enumerate the inventory, do "
        "not repeat a sentence, and stop naturally after the required final "
        f"sentence. Your final sentence must be exactly: {contract.required_ending}"
    )

    def encode_content(content: str) -> list[int]:
        encoded_prompt = encode_messages(
            [{"role": "user", "content": content}],
            thinking_mode="chat",
        )
        return tokenizer.encode(encoded_prompt, add_special_tokens=False)

    instruction_ids = encode_content(instruction)
    if len(instruction_ids) >= target_tokens:
        raise BenchmarkError("the requested prompt is shorter than its framing")

    areas = ("routing", "worker", "master", "runtime", "telemetry", "api")
    checks = (
        "stream cancellation",
        "immutable event replay",
        "request schema validation",
        "resource ownership",
        "cache lifetime isolation",
        "tool parser integrity",
    )

    def inventory_record(index: int) -> str:
        shifted_index = index + lane * 3
        area = areas[shifted_index % len(areas)]
        check = checks[(shifted_index * 5 + 1) % len(checks)]
        ticket = 1000 + lane * 500 + index * 7
        return (
            f"Inventory item R{lane}-{index:04d}: src/exo/{area}/component_"
            f"{index:04d}.py is associated with change ticket EXO-{ticket}; its "
            f"focused review checks {check}, deterministic cleanup, and typed "
            f"error propagation in test_component_{index:04d}.py."
        )

    record_count = 16
    while record_count <= 1_048_576:
        context = "\n".join(inventory_record(index) for index in range(record_count))
        prompt_ids = encode_content(context + instruction)
        common_suffix = 0
        maximum_suffix = min(len(prompt_ids), len(instruction_ids))
        while (
            common_suffix < maximum_suffix
            and prompt_ids[-common_suffix - 1] == instruction_ids[-common_suffix - 1]
        ):
            common_suffix += 1
        minimum_intact_suffix = min(64, max(16, len(instruction_ids) // 3))
        if common_suffix < minimum_intact_suffix:
            raise BenchmarkError("tokenizer did not preserve the semantic task suffix")
        if len(prompt_ids) > target_tokens:
            prefix_length = target_tokens - common_suffix
            if prefix_length <= 0:
                raise BenchmarkError("exact prompt cannot retain semantic framing")
            exact_prompt = prompt_ids[:prefix_length] + prompt_ids[-common_suffix:]
            if len(exact_prompt) != target_tokens:
                raise AssertionError("exact prompt construction changed token shape")
            return exact_prompt
        record_count *= 2
    raise BenchmarkError("could not construct an exact-length varied DSV4 prompt")


def prepare_native_prompts(model_path: Path) -> tuple[list[int], list[int]]:
    tokenizer = load_tokenizer(model_path)
    message_encoder = load_message_encoder(model_path)
    prompts = (
        build_exact_prompt_ids(tokenizer, message_encoder, lane=0),
        build_exact_prompt_ids(tokenizer, message_encoder, lane=1),
    )
    validate_exact_prompts(prompts)
    return prompts


def native_payload(input_ids: Sequence[int]) -> dict[str, object]:
    return {
        "input_ids": list(input_ids),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": OUTPUT_TOKEN_COUNT,
            "ignore_eos": False,
            "sampling_seed": 0,
        },
        "stream": True,
        "return_logprob": False,
        "log_metrics": True,
    }


def tool_call_payload(
    *,
    lane: int,
    model: str,
    maximum_tokens: int,
) -> dict[str, object]:
    if lane not in range(LANE_COUNT):
        raise ValueError("lane must be zero or one")
    expected_component = TOOL_COMPONENTS[lane]
    expected_decision = TOOL_DECISIONS[lane]
    tool = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Record an inert marker in the benchmark response. This tool has "
                "no side effects and is never executed."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "lane": {"type": "integer", "enum": [lane]},
                    "marker": {"type": "string", "enum": [TOOL_MARKER]},
                    "component": {
                        "type": "string",
                        "enum": [expected_component],
                    },
                    "decision": {
                        "type": "string",
                        "enum": [expected_decision],
                    },
                },
                "required": ["lane", "marker", "component", "decision"],
                "additionalProperties": False,
            },
        },
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an OpenCode-compatible coding agent performing a "
                    "read-only concurrency health check. Never claim to execute "
                    "the requested tool; only return its structured call."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Call {TOOL_NAME} exactly once with lane {lane} and marker "
                    f"{TOOL_MARKER}. Copy component {expected_component} and "
                    f"decision {expected_decision} exactly. Do not answer with "
                    "prose or call any other tool."
                ),
            },
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": maximum_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }


def _iter_sse_data(response: Iterable[bytes]) -> Iterator[bytes]:
    for raw_line in response:
        line = raw_line.rstrip(b"\r\n")
        if not line or line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            raise BenchmarkError("a streaming endpoint returned a non-SSE line")
        yield line.removeprefix(b"data:").lstrip(b" ")


def _decode_json_object(encoded: bytes) -> dict[str, object]:
    try:
        value = cast(object, json.loads(encoded))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError("a stream event was not valid JSON") from error
    if not isinstance(value, dict):
        raise BenchmarkError("a stream event was not a JSON object")
    return cast(dict[str, object], value)


def _update_output_ids(
    accumulated: list[int],
    event_output_ids: object,
    *,
    completion_tokens: int,
) -> None:
    if not isinstance(event_output_ids, list):
        raise BenchmarkError("native output IDs were malformed")
    raw_output_ids = cast(list[object], event_output_ids)
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in raw_output_ids
    ):
        raise BenchmarkError("native output IDs were malformed")
    typed_output_ids = cast(list[int], raw_output_ids)
    if len(typed_output_ids) == completion_tokens:
        if (
            len(typed_output_ids) < len(accumulated)
            or typed_output_ids[: len(accumulated)] != accumulated
        ):
            raise BenchmarkError("cumulative native output IDs changed prefix")
        accumulated[:] = typed_output_ids
        return
    if len(accumulated) + len(typed_output_ids) == completion_tokens:
        accumulated.extend(typed_output_ids)
        return
    raise BenchmarkError("native output IDs disagreed with completion metadata")


def _merge_stream_text(accumulated: str, event_text: str) -> str:
    if not accumulated or event_text.startswith(accumulated):
        return event_text
    if accumulated.endswith(event_text):
        return accumulated
    return accumulated + event_text


def _finish_reason_type(raw_finish_reason: object) -> str | None:
    if isinstance(raw_finish_reason, str):
        return raw_finish_reason
    if isinstance(raw_finish_reason, dict):
        typed_finish_reason = cast(dict[object, object], raw_finish_reason)
        raw_type = typed_finish_reason.get("type")
        if isinstance(raw_type, str):
            return raw_type
    return None


def run_native_lane(
    *,
    lane: int,
    url: str,
    input_ids: Sequence[int],
    timeout_seconds: float,
    request_started_at: float,
    request_body: bytes | None = None,
    decode_output_ids: Callable[[list[int]], str] | None = None,
) -> NativeLaneObservation:
    request = urllib.request.Request(
        url,
        data=(
            request_body
            if request_body is not None
            else _canonical_json_bytes(native_payload(input_ids))
        ),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    first_output_at: float | None = None
    last_output_at: float | None = None
    first_event_completion_tokens: int | None = None
    completion_tokens = 0
    server_prompt_tokens: int | None = None
    output_ids: list[int] = []
    saw_output_ids = False
    output_text = ""
    finish_reason: str | None = None
    event_count = 0
    saw_done = False

    with cast(
        HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
    ) as response:
        if response.status != 200:
            raise BenchmarkError(f"native lane {lane} returned a non-200 status")
        for encoded_event in _iter_sse_data(response):
            received_at = time.perf_counter()
            if encoded_event == b"[DONE]":
                if saw_done:
                    raise BenchmarkError("native stream emitted duplicate done events")
                saw_done = True
                continue
            if saw_done:
                raise BenchmarkError("native stream emitted data after its done event")
            event = _decode_json_object(encoded_event)
            event_count += 1

            event_text = event.get("text")
            if isinstance(event_text, str):
                output_text = _merge_stream_text(output_text, event_text)
            elif event_text is not None:
                raise BenchmarkError("native text field was malformed")

            metadata = event.get("meta_info")
            if not isinstance(metadata, dict):
                continue
            typed_metadata = cast(dict[str, object], metadata)
            parsed_finish_reason = _finish_reason_type(
                typed_metadata.get("finish_reason")
            )
            if parsed_finish_reason is not None:
                if finish_reason not in (None, parsed_finish_reason):
                    raise BenchmarkError("native finish reason changed mid-stream")
                finish_reason = parsed_finish_reason
            raw_completion_tokens = typed_metadata.get("completion_tokens")
            if not isinstance(raw_completion_tokens, int) or isinstance(
                raw_completion_tokens, bool
            ):
                continue
            if raw_completion_tokens < completion_tokens:
                raise BenchmarkError("native completion metadata moved backward")

            raw_prompt_tokens = typed_metadata.get("prompt_tokens")
            if isinstance(raw_prompt_tokens, int) and not isinstance(
                raw_prompt_tokens, bool
            ):
                if server_prompt_tokens is None:
                    server_prompt_tokens = raw_prompt_tokens
                elif server_prompt_tokens != raw_prompt_tokens:
                    raise BenchmarkError("native prompt metadata changed mid-stream")

            event_output_ids = event.get("output_ids")
            if event_output_ids is not None:
                _update_output_ids(
                    output_ids,
                    event_output_ids,
                    completion_tokens=raw_completion_tokens,
                )
                saw_output_ids = True

            if raw_completion_tokens > completion_tokens:
                if first_output_at is None:
                    first_output_at = received_at
                    first_event_completion_tokens = raw_completion_tokens
                last_output_at = received_at
                completion_tokens = raw_completion_tokens

    request_finished_at = time.perf_counter()
    if (
        first_output_at is None
        or last_output_at is None
        or first_event_completion_tokens is None
    ):
        raise BenchmarkError("native stream produced no token-bearing events")
    if completion_tokens <= 0 or completion_tokens > OUTPUT_TOKEN_COUNT:
        raise BenchmarkError(
            f"native lane {lane} returned an invalid natural completion count"
        )
    if server_prompt_tokens != INPUT_TOKEN_COUNT:
        raise BenchmarkError(
            f"native lane {lane} server prompt count did not equal {INPUT_TOKEN_COUNT}"
        )
    if saw_output_ids:
        if len(output_ids) != completion_tokens:
            raise BenchmarkError("native output-ID stream was incomplete")
        output_sha256 = hash_token_ids(output_ids)
        output_hash_source = "output_ids"
    elif output_text:
        output_sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        output_hash_source = "stream_text_events"
    else:
        raise BenchmarkError("native stream provided no hashable output")

    semantic_text_source = "stream_text_events"
    if decode_output_ids is not None and len(output_ids) == completion_tokens:
        output_text = decode_output_ids(output_ids)
        semantic_text_source = "decoded_output_ids"

    semantic_assessment = assess_native_output(
        lane,
        output_text,
        prompt_ids=input_ids,
        output_ids=output_ids if len(output_ids) == completion_tokens else (),
    )
    if not saw_done:
        raise BenchmarkError("native stream did not emit its done event")
    if finish_reason != "stop":
        raise BenchmarkError("native stream did not terminate at natural EOS")

    return NativeLaneObservation(
        lane=lane,
        request_started_at=request_started_at,
        first_output_at=first_output_at,
        last_output_at=last_output_at,
        request_finished_at=request_finished_at,
        input_sha256=hash_token_ids(input_ids),
        completion_tokens=completion_tokens,
        first_event_completion_tokens=first_event_completion_tokens,
        server_prompt_tokens=server_prompt_tokens,
        finish_reason=finish_reason,
        output_sha256=output_sha256,
        output_hash_source=output_hash_source,
        semantic_text_source=semantic_text_source,
        output_text_bytes=len(output_text.encode("utf-8")),
        semantic_assessment=semantic_assessment,
        event_count=event_count,
        saw_done=saw_done,
    )


def _choice_delta_has_output(delta: dict[str, object]) -> bool:
    for key in ("content", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return True
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    return len(cast(list[object], tool_calls)) > 0


def _consume_tool_fragments(
    delta: dict[str, object],
    tool_calls: dict[int, ToolCallParts],
    issue_codes: set[str],
) -> None:
    raw_tool_calls = delta.get("tool_calls")
    if raw_tool_calls is None:
        return
    if not isinstance(raw_tool_calls, list):
        issue_codes.add("tool_calls_not_list")
        return
    for raw_tool_call in cast(list[object], raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            issue_codes.add("tool_call_not_object")
            continue
        typed_tool_call = cast(dict[str, object], raw_tool_call)
        raw_index = typed_tool_call.get("index", 0)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            issue_codes.add("tool_call_index_invalid")
            continue
        parts = tool_calls.setdefault(raw_index, ToolCallParts())
        raw_identifier = typed_tool_call.get("id")
        if isinstance(raw_identifier, str) and raw_identifier:
            if parts.identifier not in (None, raw_identifier):
                issue_codes.add("tool_call_id_changed")
            parts.identifier = raw_identifier
        raw_kind = typed_tool_call.get("type")
        if isinstance(raw_kind, str) and raw_kind:
            if parts.kind not in (None, raw_kind):
                issue_codes.add("tool_call_type_changed")
            parts.kind = raw_kind
        raw_function = typed_tool_call.get("function")
        if raw_function is None:
            continue
        if not isinstance(raw_function, dict):
            issue_codes.add("tool_function_not_object")
            continue
        typed_function = cast(dict[str, object], raw_function)
        raw_name = typed_function.get("name")
        if isinstance(raw_name, str):
            parts.name_fragments.append(raw_name)
        elif raw_name is not None:
            issue_codes.add("tool_name_not_string")
        raw_arguments = typed_function.get("arguments")
        if isinstance(raw_arguments, str):
            parts.argument_fragments.append(raw_arguments)
        elif raw_arguments is not None:
            issue_codes.add("tool_arguments_not_string")


def _validate_tool_structure(
    *,
    lane: int,
    tool_calls: dict[int, ToolCallParts],
    finish_reason: str | None,
    saw_done: bool,
    issue_codes: set[str],
) -> tuple[bool, tuple[str, ...], str]:
    arguments_text = ""
    if set(tool_calls) != {0}:
        issue_codes.add("expected_one_tool_call_at_index_zero")
    else:
        parts = tool_calls[0]
        if not parts.identifier:
            issue_codes.add("tool_call_id_missing")
        if parts.kind != "function":
            issue_codes.add("tool_call_type_invalid")
        if "".join(parts.name_fragments) != TOOL_NAME:
            issue_codes.add("tool_name_invalid")
        arguments_text = "".join(parts.argument_fragments)
        try:
            arguments = cast(object, json.loads(arguments_text))
        except (UnicodeDecodeError, json.JSONDecodeError):
            issue_codes.add("tool_arguments_invalid_json")
        else:
            expected_arguments = {
                "lane": lane,
                "marker": TOOL_MARKER,
                "component": TOOL_COMPONENTS[lane],
                "decision": TOOL_DECISIONS[lane],
            }
            if arguments != expected_arguments:
                issue_codes.add("tool_arguments_schema_mismatch")
    if finish_reason != "tool_calls":
        issue_codes.add("finish_reason_not_tool_calls")
    if not saw_done:
        issue_codes.add("done_event_missing")
    ordered_issues = tuple(sorted(issue_codes))
    return not ordered_issues, ordered_issues, arguments_text


def run_chat_lane(
    *,
    lane: int,
    url: str,
    model: str,
    maximum_tokens: int,
    timeout_seconds: float,
    request_started_at: float,
    request_body: bytes | None = None,
) -> ChatLaneObservation:
    request = urllib.request.Request(
        url,
        data=(
            request_body
            if request_body is not None
            else _canonical_json_bytes(
                tool_call_payload(
                    lane=lane,
                    model=model,
                    maximum_tokens=maximum_tokens,
                )
            )
        ),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    first_output_at: float | None = None
    last_output_at: float | None = None
    output_digest = hashlib.sha256()
    output_bytes = 0
    tool_calls: dict[int, ToolCallParts] = {}
    issue_codes: set[str] = set()
    event_count = 0
    completion_tokens: int | None = None
    finish_reason: str | None = None
    saw_done = False
    saw_unexpected_content = False
    saw_unexpected_reasoning = False

    with cast(
        HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
    ) as response:
        if response.status != 200:
            raise BenchmarkError(f"chat lane {lane} returned a non-200 status")
        for encoded_event in _iter_sse_data(response):
            received_at = time.perf_counter()
            if encoded_event == b"[DONE]":
                if saw_done:
                    issue_codes.add("duplicate_done_event")
                saw_done = True
                continue
            if saw_done:
                issue_codes.add("event_after_done")
            event = _decode_json_object(encoded_event)
            event_count += 1
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                typed_usage = cast(dict[str, object], raw_usage)
                raw_completion_tokens = typed_usage.get("completion_tokens")
                if isinstance(raw_completion_tokens, int) and not isinstance(
                    raw_completion_tokens, bool
                ):
                    completion_tokens = raw_completion_tokens
            raw_choices = event.get("choices")
            if not isinstance(raw_choices, list):
                issue_codes.add("choices_not_list")
                continue
            for raw_choice in cast(list[object], raw_choices):
                if not isinstance(raw_choice, dict):
                    issue_codes.add("choice_not_object")
                    continue
                typed_choice = cast(dict[str, object], raw_choice)
                raw_choice_index = typed_choice.get("index")
                if (
                    not isinstance(raw_choice_index, int)
                    or isinstance(raw_choice_index, bool)
                    or raw_choice_index != 0
                ):
                    issue_codes.add("choice_index_invalid")
                raw_finish_reason = typed_choice.get("finish_reason")
                if isinstance(raw_finish_reason, str):
                    if finish_reason not in (None, raw_finish_reason):
                        issue_codes.add("finish_reason_changed")
                    finish_reason = raw_finish_reason
                raw_delta = typed_choice.get("delta")
                if not isinstance(raw_delta, dict):
                    issue_codes.add("delta_not_object")
                    continue
                delta = cast(dict[str, object], raw_delta)
                encoded_delta = _canonical_json_bytes(delta)
                output_digest.update(len(encoded_delta).to_bytes(8, "big"))
                output_digest.update(encoded_delta)
                output_bytes += len(encoded_delta)
                if _choice_delta_has_output(delta):
                    if first_output_at is None:
                        first_output_at = received_at
                    last_output_at = received_at
                raw_content = delta.get("content")
                if isinstance(raw_content, str) and raw_content.strip():
                    saw_unexpected_content = True
                elif raw_content is not None and not isinstance(raw_content, str):
                    issue_codes.add("content_not_string")
                raw_reasoning = delta.get("reasoning_content")
                if isinstance(raw_reasoning, str) and raw_reasoning.strip():
                    saw_unexpected_reasoning = True
                elif raw_reasoning is not None and not isinstance(raw_reasoning, str):
                    issue_codes.add("reasoning_content_not_string")
                _consume_tool_fragments(delta, tool_calls, issue_codes)

    request_finished_at = time.perf_counter()
    if first_output_at is None:
        issue_codes.add("output_event_missing")
    if event_count == 0 or output_bytes == 0:
        issue_codes.add("stream_output_empty")
    if completion_tokens is None:
        issue_codes.add("completion_usage_missing")
    elif completion_tokens <= 0:
        issue_codes.add("completion_usage_invalid")
    if saw_unexpected_content:
        issue_codes.add("unexpected_prose_content")
    if saw_unexpected_reasoning:
        issue_codes.add("unexpected_reasoning_content")
    success, ordered_issues, arguments_text = _validate_tool_structure(
        lane=lane,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        saw_done=saw_done,
        issue_codes=issue_codes,
    )
    return ChatLaneObservation(
        lane=lane,
        request_started_at=request_started_at,
        first_output_at=first_output_at,
        last_output_at=last_output_at,
        request_finished_at=request_finished_at,
        output_sha256=output_digest.hexdigest(),
        output_bytes=output_bytes,
        tool_arguments_sha256=hashlib.sha256(
            arguments_text.encode("utf-8")
        ).hexdigest(),
        event_count=event_count,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        structural_parser_success=success,
        parser_issue_codes=ordered_issues,
        saw_done=saw_done,
    )


def _run_synchronized_pair[LaneObservation](
    operation: Callable[[int, float], LaneObservation],
) -> tuple[float, tuple[LaneObservation, LaneObservation]]:
    release_times: list[float] = []
    barrier = threading.Barrier(
        LANE_COUNT + 1,
        action=lambda: release_times.append(time.perf_counter()),
        timeout=10.0,
    )
    results: list[LaneObservation | None] = [None] * LANE_COUNT
    errors: list[Exception | None] = [None] * LANE_COUNT

    def execute_lane(lane: int) -> None:
        try:
            barrier.wait()
            request_started_at = time.perf_counter()
            results[lane] = operation(lane, request_started_at)
        except Exception as error:  # noqa: BLE001
            # Error details can contain model output, so retain only the object.
            errors[lane] = error

    threads = tuple(
        threading.Thread(
            target=execute_lane,
            args=(lane,),
            name=f"dsv4-pp2-benchmark-lane-{lane}",
        )
        for lane in range(LANE_COUNT)
    )
    for thread in threads:
        thread.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError as error:
        raise BenchmarkError("the two-lane start barrier failed") from error
    for thread in threads:
        thread.join()

    for lane, error in enumerate(errors):
        if error is not None:
            raise BenchmarkError(
                f"lane {lane} failed with {type(error).__name__}"
            ) from error
    if len(release_times) != 1 or any(result is None for result in results):
        raise BenchmarkError("the synchronized pair did not produce two results")
    typed_results = cast(list[LaneObservation], results)
    return release_times[0], (typed_results[0], typed_results[1])


def flush_cache(url: str, timeout_seconds: float) -> FlushObservation:
    started_at = time.perf_counter()
    deadline = started_at + timeout_seconds
    busy_retries = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise BenchmarkError("cache flush remained busy until timeout")
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with cast(
                HttpResponse, urllib.request.urlopen(request, timeout=remaining)
            ) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            error.read()
            busy_retries += 1
            time.sleep(
                min(
                    FLUSH_RETRY_SECONDS,
                    max(deadline - time.perf_counter(), 0.0),
                )
            )
            continue
        if status != 200:
            raise BenchmarkError("cache flush returned a non-200 status")

        # Current SGLang returns a plain-text success receipt. Older/local test
        # surfaces may return the structured control-plane object instead, so
        # accept either contract but fail closed on every other 200 response.
        plaintext_success = body.startswith(b"Cache flushed.\n")
        structured_success = False
        try:
            flush_payload = cast(object, json.loads(body))
        except (UnicodeDecodeError, json.JSONDecodeError):
            flush_payload = None
        if isinstance(flush_payload, dict):
            typed_flush_payload = cast(dict[str, object], flush_payload)
            structured_success = typed_flush_payload.get("success") is True
        if not plaintext_success and not structured_success:
            raise BenchmarkError("cache flush response did not confirm success")
        return FlushObservation(
            status=status,
            elapsed_seconds=time.perf_counter() - started_at,
            busy_retries=busy_retries,
            response_sha256=hashlib.sha256(body).hexdigest(),
        )


def _flush_report(observation: FlushObservation) -> dict[str, object]:
    return {
        "http_status": observation.status,
        "elapsed_seconds": _rounded(observation.elapsed_seconds),
        "busy_retries": observation.busy_retries,
        "response_sha256": observation.response_sha256,
    }


def _native_group_report(
    *,
    released_at: float,
    observations: Sequence[NativeLaneObservation],
    flush_observation: FlushObservation,
) -> dict[str, object]:
    if len(observations) != LANE_COUNT:
        raise ValueError("native group requires exactly two observations")
    makespan = max(item.request_finished_at for item in observations) - released_at
    time_to_first_tokens = [item.time_to_first_token_seconds for item in observations]
    decode_rates = [item.decode_tokens_per_second for item in observations]
    start_times = [item.request_started_at for item in observations]
    finish_times = [item.request_finished_at for item in observations]
    start_skew = max(start_times) - min(start_times)
    overlap_seconds = min(finish_times) - max(start_times)
    output_overlap_seconds = min(item.last_output_at for item in observations) - max(
        item.first_output_at for item in observations
    )
    completion_tokens = sum(item.completion_tokens for item in observations)
    aggregate_decode_tokens = sum(
        item.completion_tokens - item.first_event_completion_tokens
        for item in observations
    )
    decode_window_seconds = max(item.last_output_at for item in observations) - min(
        item.first_output_at for item in observations
    )
    semantic_success_count = sum(
        item.semantic_assessment.passed for item in observations
    )
    all_streams_complete = all(item.saw_done for item in observations)
    all_naturally_terminated = all(
        item.finish_reason == "stop" for item in observations
    )
    cross_lane_outputs_distinct = (
        len({item.output_sha256 for item in observations}) == LANE_COUNT
    )
    all_decode_timing_valid = all(rate > 0 for rate in decode_rates)
    concurrent_start_confirmed = start_skew <= MAXIMUM_START_SKEW_SECONDS
    concurrent_overlap_observed = overlap_seconds > 0
    concurrent_output_overlap_observed = output_overlap_seconds > 0
    lanes = [
        {
            "lane": item.lane,
            "input_tokens": INPUT_TOKEN_COUNT,
            "input_ids_sha256": item.input_sha256,
            "completion_tokens": item.completion_tokens,
            "time_to_first_token_seconds": _rounded(item.time_to_first_token_seconds),
            "decode_seconds": _rounded(item.decode_seconds),
            "decode_tokens_per_second": _rounded(item.decode_tokens_per_second),
            "total_seconds": _rounded(item.total_seconds),
            "output_sha256": item.output_sha256,
            "output_hash_source": item.output_hash_source,
            "semantic_text_source": item.semantic_text_source,
            "output_text_bytes": item.output_text_bytes,
            "stream_events": item.event_count,
            "server_prompt_tokens": item.server_prompt_tokens,
            "finish_reason": item.finish_reason,
            "semantic_validation": item.semantic_assessment.safe_receipt(),
            "saw_done": item.saw_done,
        }
        for item in observations
    ]
    return {
        "flush_cache": _flush_report(flush_observation),
        "request_count": LANE_COUNT,
        "input_tokens_each": INPUT_TOKEN_COUNT,
        "requested_max_output_tokens_each": OUTPUT_TOKEN_COUNT,
        "ignore_eos": False,
        "lanes": lanes,
        "request_start_skew_seconds": _rounded(start_skew),
        "concurrent_start_maximum_skew_seconds": MAXIMUM_START_SKEW_SECONDS,
        "concurrent_start_confirmed": concurrent_start_confirmed,
        "concurrent_overlap_seconds": _rounded(max(0.0, overlap_seconds)),
        "concurrent_overlap_observed": concurrent_overlap_observed,
        "concurrent_output_overlap_seconds": _rounded(max(0.0, output_overlap_seconds)),
        "concurrent_output_overlap_observed": concurrent_output_overlap_observed,
        "makespan_seconds": _rounded(makespan),
        "aggregate_requests_per_second": _rounded(LANE_COUNT / makespan),
        "aggregate_output_tokens": completion_tokens,
        "aggregate_output_tokens_per_second": _rounded(completion_tokens / makespan),
        "aggregate_decode_tokens": aggregate_decode_tokens,
        "aggregate_decode_window_seconds": _rounded(decode_window_seconds),
        "aggregate_decode_tokens_per_second": _rounded(
            aggregate_decode_tokens / decode_window_seconds
            if decode_window_seconds > 0
            else 0.0
        ),
        "p50_time_to_first_token_seconds": _rounded(
            statistics.median(time_to_first_tokens)
        ),
        "max_time_to_first_token_seconds": _rounded(max(time_to_first_tokens)),
        "decode_rate_jain_fairness": _rounded(jains_fairness(decode_rates)),
        "decode_rate_min_max_ratio": _rounded(
            min(decode_rates) / max(decode_rates) if max(decode_rates) > 0 else 0.0
        ),
        "semantic_success_count": semantic_success_count,
        "all_semantically_valid": semantic_success_count == LANE_COUNT,
        "all_naturally_terminated": all_naturally_terminated,
        "cross_lane_outputs_distinct": cross_lane_outputs_distinct,
        "all_decode_timing_valid": all_decode_timing_valid,
        "all_streams_complete": all_streams_complete,
        "quality_gate_passed": bool(
            semantic_success_count == LANE_COUNT
            and all_naturally_terminated
            and cross_lane_outputs_distinct
            and all_decode_timing_valid
            and all_streams_complete
            and concurrent_start_confirmed
            and concurrent_output_overlap_observed
        ),
    }


def _chat_group_report(
    *,
    released_at: float,
    observations: Sequence[ChatLaneObservation],
    flush_observation: FlushObservation,
) -> dict[str, object]:
    if len(observations) != LANE_COUNT:
        raise ValueError("chat group requires exactly two observations")
    makespan = max(item.request_finished_at for item in observations) - released_at
    valid_time_to_first_tokens = [
        value
        for item in observations
        if (value := item.time_to_first_token_seconds) is not None
    ]
    start_times = [item.request_started_at for item in observations]
    finish_times = [item.request_finished_at for item in observations]
    start_skew = max(start_times) - min(start_times)
    overlap_seconds = min(finish_times) - max(start_times)
    first_output_times = [item.first_output_at for item in observations]
    last_output_times = [item.last_output_at for item in observations]
    output_overlap_seconds = (
        min(cast(list[float], last_output_times))
        - max(cast(list[float], first_output_times))
        if all(value is not None for value in first_output_times)
        and all(value is not None for value in last_output_times)
        else 0.0
    )
    concurrent_start_confirmed = start_skew <= MAXIMUM_START_SKEW_SECONDS
    concurrent_overlap_observed = overlap_seconds > 0
    concurrent_output_overlap_observed = output_overlap_seconds > 0
    structurally_valid_count = sum(
        item.structural_parser_success for item in observations
    )
    completion_token_values = [
        item.completion_tokens
        for item in observations
        if item.completion_tokens is not None
    ]
    all_completion_usage_valid = len(completion_token_values) == LANE_COUNT and all(
        value > 0 for value in completion_token_values
    )
    tool_arguments_distinct = (
        len({item.tool_arguments_sha256 for item in observations}) == LANE_COUNT
    )
    lanes = [
        {
            "lane": item.lane,
            "time_to_first_token_seconds": (
                _rounded(item.time_to_first_token_seconds)
                if item.time_to_first_token_seconds is not None
                else None
            ),
            "total_seconds": _rounded(item.total_seconds),
            "stream_sha256": item.output_sha256,
            "stream_output_bytes": item.output_bytes,
            "tool_arguments_sha256": item.tool_arguments_sha256,
            "stream_events": item.event_count,
            "completion_tokens": item.completion_tokens,
            "finish_reason": item.finish_reason,
            "structural_parser_success": item.structural_parser_success,
            "parser_issue_codes": list(item.parser_issue_codes),
            "saw_done": item.saw_done,
        }
        for item in observations
    ]
    return {
        "flush_cache": _flush_report(flush_observation),
        "request_count": LANE_COUNT,
        "lanes": lanes,
        "request_start_skew_seconds": _rounded(start_skew),
        "concurrent_start_maximum_skew_seconds": MAXIMUM_START_SKEW_SECONDS,
        "concurrent_start_confirmed": concurrent_start_confirmed,
        "concurrent_overlap_seconds": _rounded(max(0.0, overlap_seconds)),
        "concurrent_overlap_observed": concurrent_overlap_observed,
        "concurrent_output_overlap_seconds": _rounded(max(0.0, output_overlap_seconds)),
        "concurrent_output_overlap_observed": concurrent_output_overlap_observed,
        "makespan_seconds": _rounded(makespan),
        "aggregate_requests_per_second": _rounded(LANE_COUNT / makespan),
        "aggregate_completion_tokens": sum(completion_token_values),
        "aggregate_completion_tokens_per_second": _rounded(
            sum(completion_token_values) / makespan
        ),
        "p50_time_to_first_token_seconds": (
            _rounded(statistics.median(valid_time_to_first_tokens))
            if valid_time_to_first_tokens
            else None
        ),
        "max_time_to_first_token_seconds": (
            _rounded(max(valid_time_to_first_tokens))
            if valid_time_to_first_tokens
            else None
        ),
        "structural_parser_success_count": structurally_valid_count,
        "all_structurally_valid": structurally_valid_count == LANE_COUNT,
        "all_completion_usage_valid": all_completion_usage_valid,
        "tool_arguments_distinct": tool_arguments_distinct,
        "all_streams_complete": all(item.saw_done for item in observations),
        "quality_gate_passed": bool(
            structurally_valid_count == LANE_COUNT
            and all_completion_usage_valid
            and tool_arguments_distinct
            and len(valid_time_to_first_tokens) == LANE_COUNT
            and concurrent_start_confirmed
            and concurrent_output_overlap_observed
        ),
    }


def run_benchmark(
    config: BenchmarkConfig,
    prompts: Sequence[Sequence[int]],
    *,
    decode_output_ids: Callable[[list[int]], str] | None = None,
    provenance: BenchmarkProvenance | None = None,
    oscar_provenance: OscarProvenance | None = None,
    launch_authorization: LaunchAuthorizationProvenance | None = None,
    nvlink_snapshotter: Callable[[], dict[str, int]] | None = None,
) -> dict[str, object]:
    validate_exact_prompts(prompts)
    if config.request_timeout_seconds <= 0 or config.flush_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    if config.chat_max_tokens <= 0:
        raise ValueError("chat maximum tokens must be positive")
    if config.expected_chunked_prefill_size not in (None, 512, 1024):
        raise ValueError("expected chunked-prefill size must be 512 or 1024")
    if (
        config.expected_gpu_experts_per_layer is not None
        and not 1 <= config.expected_gpu_experts_per_layer <= 256
    ):
        raise ValueError("expected GPU expert count must be between 1 and 256")
    if (
        config.expected_expert_plan_format is not None
        and config.expected_expert_plan_format not in SUPPORTED_EXPERT_PLAN_FORMATS
    ):
        raise ValueError("expected expert plan format is not supported")
    if config.expected_pp_async_batch_depth not in (0, 1):
        raise ValueError("expected PP async batch depth must be zero or one")
    if (
        config.expected_cpuinfer_threads is not None
        and config.expected_cpuinfer_threads not in ADMITTED_CPUINFER_THREAD_COUNTS
    ):
        raise ValueError("expected CPUInfer threads must be 56")

    server_info_url = config.server_info_url or _derive_server_info_url(
        config.generate_url
    )
    server_contract = validate_pp2_server_contract(
        get_server_info(server_info_url, config.flush_timeout_seconds),
        expected_chunked_prefill_size=config.expected_chunked_prefill_size,
        expected_gpu_experts_per_layer=config.expected_gpu_experts_per_layer,
        expected_expert_plan_format=config.expected_expert_plan_format,
        expected_pp_async_batch_depth=config.expected_pp_async_batch_depth,
        expected_expert_plan_sha256=(
            provenance.expert_plan_sha256 if provenance is not None else None
        ),
        expected_cpuinfer_threads=config.expected_cpuinfer_threads,
        expected_oscar_artifact_sha256=(
            oscar_provenance.artifact_sha256 if oscar_provenance is not None else None
        ),
        expected_oscar_admission_sha256=(
            oscar_provenance.admission_sha256 if oscar_provenance is not None else None
        ),
        expected_oscar_admission_receipt_sha256=(
            oscar_provenance.admission_receipt_sha256
            if oscar_provenance is not None
            else None
        ),
    )

    counter_snapshot = (
        snapshot_nvlink_counters if nvlink_snapshotter is None else nvlink_snapshotter
    )
    try:
        nvlink_before = (
            dict(counter_snapshot()) if config.require_nvlink_traffic else None
        )
    except NvlinkTrafficError as error:
        raise BenchmarkError(
            "NVLink payload counters could not be sampled before the requests"
        ) from error

    native_flush = flush_cache(config.flush_url, config.flush_timeout_seconds)
    native_bodies = tuple(
        _canonical_json_bytes(native_payload(prompt)) for prompt in prompts
    )

    def native_operation(lane: int, started_at: float) -> NativeLaneObservation:
        return run_native_lane(
            lane=lane,
            url=config.generate_url,
            input_ids=prompts[lane],
            timeout_seconds=config.request_timeout_seconds,
            request_started_at=started_at,
            request_body=native_bodies[lane],
            decode_output_ids=decode_output_ids,
        )

    native_released_at, native_observations = _run_synchronized_pair(native_operation)

    chat_flush = flush_cache(config.flush_url, config.flush_timeout_seconds)
    chat_bodies = tuple(
        _canonical_json_bytes(
            tool_call_payload(
                lane=lane,
                model=config.model,
                maximum_tokens=config.chat_max_tokens,
            )
        )
        for lane in range(LANE_COUNT)
    )

    def chat_operation(lane: int, started_at: float) -> ChatLaneObservation:
        return run_chat_lane(
            lane=lane,
            url=config.chat_url,
            model=config.model,
            maximum_tokens=config.chat_max_tokens,
            timeout_seconds=config.request_timeout_seconds,
            request_started_at=started_at,
            request_body=chat_bodies[lane],
        )

    chat_released_at, chat_observations = _run_synchronized_pair(chat_operation)

    nvlink_traffic: dict[str, object] | None = None
    if config.require_nvlink_traffic:
        if nvlink_before is None:
            raise AssertionError("NVLink before-snapshot validation drifted")
        try:
            nvlink_traffic = build_nvlink_traffic_receipt(
                nvlink_before,
                dict(counter_snapshot()),
            )
        except NvlinkTrafficError as error:
            raise BenchmarkError(
                "the concurrent model requests did not prove complete NVLink traffic"
            ) from error

    native_report = _native_group_report(
        released_at=native_released_at,
        observations=native_observations,
        flush_observation=native_flush,
    )
    chat_report = _chat_group_report(
        released_at=chat_released_at,
        observations=chat_observations,
        flush_observation=chat_flush,
    )
    quality_gate_passed = bool(
        native_report["quality_gate_passed"] and chat_report["quality_gate_passed"]
    )
    performance_inputs_bound = bool(
        provenance is not None
        and oscar_provenance is not None
        and launch_authorization is not None
        and config.require_nvlink_traffic
        and config.expected_chunked_prefill_size is not None
        and config.expected_gpu_experts_per_layer is not None
        and config.expected_expert_plan_format is not None
        and config.expected_cpuinfer_threads in ADMITTED_CPUINFER_THREAD_COUNTS
    )
    report: dict[str, object] = {
        "schema_version": RECEIPT_VERSION,
        "ok": quality_gate_passed,
        "performance_claim_eligible": bool(
            quality_gate_passed and performance_inputs_bound
        ),
        "configuration": {
            "generate_url": config.generate_url,
            "chat_url": config.chat_url,
            "flush_url": config.flush_url,
            "model": config.model,
            "request_timeout_seconds": config.request_timeout_seconds,
            "flush_timeout_seconds": config.flush_timeout_seconds,
            "chat_max_tokens": config.chat_max_tokens,
            "server_info_url": server_info_url,
            "server_contract": server_contract,
            "expected_chunked_prefill_size": (config.expected_chunked_prefill_size),
            "expected_gpu_experts_per_layer": (config.expected_gpu_experts_per_layer),
            "expected_expert_plan_format": config.expected_expert_plan_format,
            "expected_pp_async_batch_depth": config.expected_pp_async_batch_depth,
            "expected_int4_c4_indexer_storage": False,
            "expected_int4_kv_storage": False,
            "expected_c128_bf16_storage": False,
            "expected_cpuinfer_threads": config.expected_cpuinfer_threads,
            "cache_policy": "flush_once_before_each_synchronized_group",
            "native_termination": "natural_eos_required",
            "native_prompt_contract": "varied_opencode_semantic_v2",
            "tool_call_contract": "exact_distinct_opencode_tools_v2",
            "require_nvlink_traffic": config.require_nvlink_traffic,
            "benchmark_provenance": (
                provenance.safe_receipt() if provenance is not None else None
            ),
            "oscar_provenance": (
                oscar_provenance.safe_receipt()
                if oscar_provenance is not None
                else None
            ),
            "launch_authorization": (
                launch_authorization.safe_receipt()
                if launch_authorization is not None
                else None
            ),
        },
        "native_generate": native_report,
        "openai_tool_calls": chat_report,
    }
    if nvlink_traffic is not None:
        report["nvlink_traffic"] = nvlink_traffic
    return report


def parse_args(arguments: Sequence[str] | None = None) -> BenchmarkArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Run two cache-cold semantic PP2 concurrency probes without printing "
            "generated content."
        )
    )
    parser.add_argument("--generate-url", default=DEFAULT_GENERATE_URL)
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL)
    parser.add_argument("--flush-url", default=DEFAULT_FLUSH_URL)
    parser.add_argument(
        "--server-info-url",
        help=f"live server contract endpoint (default: {DEFAULT_SERVER_INFO_URL})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    parser.add_argument("--flush-timeout", type=float, default=60.0)
    parser.add_argument("--chat-max-tokens", type=int, default=DEFAULT_CHAT_MAX_TOKENS)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument(
        "--run-label",
        help="stable receipt label; requires --expert-plan",
    )
    parser.add_argument(
        "--expert-plan",
        type=Path,
        help="absolute PP2 expert-plan path to hash into the receipt; requires --run-label",
    )
    parser.add_argument(
        "--require-nvlink-traffic",
        action="store_true",
        help=(
            "require every Tx/Rx payload counter on GPUs 0-1 and NVLinks 0-3 "
            "to move across the two synchronized request groups"
        ),
    )
    parser.add_argument(
        "--expected-chunked-prefill-size",
        type=int,
        choices=(512, 1024),
        help="fail unless the live PP2 server uses this prefill chunk",
    )
    parser.add_argument(
        "--expected-gpu-experts-per-layer",
        type=int,
        choices=range(1, 257),
        metavar="COUNT",
        help=(
            "fail unless the live PP2 admission ceiling equals the widest target "
            "GPU layer"
        ),
    )
    parser.add_argument(
        "--expected-expert-plan-format",
        choices=tuple(sorted(SUPPORTED_EXPERT_PLAN_FORMATS)),
        help="fail unless the live PP2 server loaded this exact plan format",
    )
    parser.add_argument(
        "--expected-pp-async-batch-depth",
        type=int,
        choices=(0, 1),
        default=0,
        help="fail unless the live PP2 scheduler uses this async pipeline depth",
    )
    parser.add_argument(
        "--expected-cpuinfer-threads",
        type=int,
        choices=tuple(sorted(ADMITTED_CPUINFER_THREAD_COUNTS)),
        help="fail unless every PP stage uses this exact CPUInfer thread count",
    )
    parser.add_argument(
        "--oscar-admission-receipt",
        type=Path,
        required=True,
        help=(
            "absolute admitted OSCAR receipt to rehash and bind to both live PP workers"
        ),
    )
    parser.add_argument(
        "--launch-authorization-receipt",
        type=Path,
        required=True,
        help=(
            "absolute receipt emitted by the two-launch PP2 controller for the "
            "currently running service"
        ),
    )
    parsed = parser.parse_args(arguments)
    return BenchmarkArguments(
        generate_url=cast(str, parsed.generate_url),
        chat_url=cast(str, parsed.chat_url),
        flush_url=cast(str, parsed.flush_url),
        model=cast(str, parsed.model),
        model_path=cast(Path, parsed.model_path),
        request_timeout_seconds=cast(float, parsed.request_timeout),
        flush_timeout_seconds=cast(float, parsed.flush_timeout),
        chat_max_tokens=cast(int, parsed.chat_max_tokens),
        output_file=cast(Path | None, parsed.output_file),
        run_label=cast(str | None, parsed.run_label),
        expert_plan=cast(Path | None, parsed.expert_plan),
        require_nvlink_traffic=cast(bool, parsed.require_nvlink_traffic),
        server_info_url=cast(str | None, parsed.server_info_url),
        expected_chunked_prefill_size=cast(
            int | None, parsed.expected_chunked_prefill_size
        ),
        expected_gpu_experts_per_layer=cast(
            int | None, parsed.expected_gpu_experts_per_layer
        ),
        expected_expert_plan_format=cast(
            str | None, parsed.expected_expert_plan_format
        ),
        expected_pp_async_batch_depth=cast(int, parsed.expected_pp_async_batch_depth),
        expected_cpuinfer_threads=cast(int | None, parsed.expected_cpuinfer_threads),
        oscar_admission_receipt=cast(Path, parsed.oscar_admission_receipt),
        launch_authorization_receipt=cast(Path, parsed.launch_authorization_receipt),
    )


def main() -> int:
    arguments = parse_args()
    provenance: BenchmarkProvenance | None = None
    oscar_provenance: OscarProvenance | None = None
    launch_authorization: LaunchAuthorizationProvenance | None = None
    try:
        provenance = load_benchmark_provenance(
            run_label=arguments.run_label,
            expert_plan=arguments.expert_plan,
        )
        oscar_provenance = load_oscar_provenance(arguments.oscar_admission_receipt)
        launch_authorization = load_launch_authorization_provenance(
            arguments.launch_authorization_receipt
        )
        prompts = prepare_native_prompts(arguments.model_path)
        tokenizer = load_tokenizer(arguments.model_path)

        def decode_output_ids(token_ids: list[int]) -> str:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        config = BenchmarkConfig(
            generate_url=arguments.generate_url,
            chat_url=arguments.chat_url,
            flush_url=arguments.flush_url,
            model=arguments.model,
            request_timeout_seconds=arguments.request_timeout_seconds,
            flush_timeout_seconds=arguments.flush_timeout_seconds,
            chat_max_tokens=arguments.chat_max_tokens,
            require_nvlink_traffic=arguments.require_nvlink_traffic,
            server_info_url=arguments.server_info_url,
            expected_chunked_prefill_size=(arguments.expected_chunked_prefill_size),
            expected_gpu_experts_per_layer=(arguments.expected_gpu_experts_per_layer),
            expected_expert_plan_format=arguments.expected_expert_plan_format,
            expected_pp_async_batch_depth=arguments.expected_pp_async_batch_depth,
            expected_cpuinfer_threads=arguments.expected_cpuinfer_threads,
        )
        report = run_benchmark(
            config,
            prompts,
            decode_output_ids=decode_output_ids,
            provenance=provenance,
            oscar_provenance=oscar_provenance,
            launch_authorization=launch_authorization,
        )
    except Exception as error:  # noqa: BLE001
        # Exception text can embed response content, so report only its class.
        failure: dict[str, object] = {
            "schema_version": RECEIPT_VERSION,
            "ok": False,
            "performance_claim_eligible": False,
            "error_type": type(error).__name__,
        }
        # BenchmarkError messages are constructed exclusively from fixed
        # protocol/invariant labels and lane numbers above; they never contain
        # response bodies, generated text, or tool arguments. Preserve that
        # safe diagnostic so a rejected live concurrency run is actionable.
        if isinstance(error, BenchmarkError):
            failure["benchmark_error"] = str(error)
            if error.__cause__ is not None:
                failure["cause_type"] = type(error.__cause__).__name__
        if provenance is not None:
            failure["benchmark_provenance"] = provenance.safe_receipt()
        if oscar_provenance is not None:
            failure["oscar_provenance"] = oscar_provenance.safe_receipt()
        if launch_authorization is not None:
            failure["launch_authorization"] = launch_authorization.safe_receipt()
        print(json.dumps(failure, sort_keys=True))
        return 1

    encoded_report = json.dumps(report, separators=(",", ":"), sort_keys=True)
    if arguments.output_file is not None:
        arguments.output_file.write_text(encoded_report + "\n", encoding="utf-8")
    print(encoded_report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
